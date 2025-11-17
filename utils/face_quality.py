import cv2
import numpy as np
from deepface import DeepFace

class FaceQualityValidator:
    """Validador de qualidade de imagem facial"""
    
    @staticmethod
    def validar_qualidade_completa(frame):
        """
        Validação completa de qualidade
        Retorna: (bool, dict, str)
        """
        resultados = {
            'rosto_detectado': False,
            'nitidez': 0.0,
            'iluminacao': 0.0,
            'tamanho_rosto': 0.0,
            'centralizado': False,
            'dentro_guia': False,
            'score_geral': 0.0,
            'warnings': []
        }
        
        try:
            # 1. DETECTAR ROSTO COM DEEPFACE
            faces = DeepFace.extract_faces(
                img_path=frame,
                enforce_detection=True,
                detector_backend='opencv',
                align=True
            )
            
            if not faces or len(faces) == 0:
                return False, resultados, "[X] Nenhum rosto detectado"
            
            if len(faces) > 1:
                return False, resultados, "[X] Multiplos rostos detectados. Apenas um por vez."
            
            face_data = faces[0]
            resultados['rosto_detectado'] = True
            
            # 2. VALIDAR NITIDEZ (APENAS AVISO)
            score_nitidez = FaceQualityValidator._calcular_nitidez(frame)
            resultados['nitidez'] = score_nitidez
            
            if score_nitidez < 100:
                resultados['warnings'].append(f"[!] Imagem pode estar desfocada (nitidez: {score_nitidez:.0f})")
            
            # 3. VALIDAR ILUMINAÇÃO (APENAS AVISO)
            score_iluminacao = FaceQualityValidator._calcular_iluminacao(frame)
            resultados['iluminacao'] = score_iluminacao
            
            if score_iluminacao < 0.3:
                resultados['warnings'].append(f"[!] Imagem escura (iluminacao: {score_iluminacao:.0%})")
            elif score_iluminacao > 0.8:
                resultados['warnings'].append(f"[!] Imagem muito clara (iluminacao: {score_iluminacao:.0%})")
            
            # 4. VALIDAR TAMANHO E POSIÇÃO COM O GUIA OVAL
            facial_area = face_data['facial_area']
            img_height, img_width = frame.shape[:2]
            
            face_x = facial_area['x']
            face_y = facial_area['y']
            face_width = facial_area['w']
            face_height = facial_area['h']
            
            # Centro do rosto
            face_center_x = face_x + face_width / 2
            face_center_y = face_y + face_height / 2
            
            # VERIFICAR SE ESTÁ DENTRO DO GUIA OVAL
            guia_info = FaceQualityValidator._verificar_posicao_guia(
                face_center_x, face_center_y, face_width, face_height, img_width, img_height
            )
            
            resultados['centralizado'] = guia_info['centralizado']
            resultados['dentro_guia'] = guia_info['dentro_guia']
            
            if not guia_info['centralizado']:
                if guia_info['direcao']:
                    resultados['warnings'].append(f"[!] Mova-se para: {guia_info['direcao']}")
            
            if not guia_info['tamanho_ok']:
                resultados['warnings'].append(f"[!] {guia_info['instrucao_tamanho']}")
            
            # Percentual de tamanho
            percent_width = face_width / img_width
            percent_height = face_height / img_height
            resultados['tamanho_rosto'] = (percent_width + percent_height) / 2
            
            # 5. CALCULAR SCORE GERAL
            score_geral = (
                min(score_nitidez / 500, 1.0) * 0.30 +  # 30% nitidez
                score_iluminacao * 0.20 +                # 20% iluminação
                resultados['tamanho_rosto'] * 0.20 +     # 20% tamanho
                (1.0 if guia_info['centralizado'] else 0.5) * 0.15 +  # 15% centralização
                (1.0 if guia_info['dentro_guia'] else 0.5) * 0.15     # 15% dentro do guia
            )
            
            resultados['score_geral'] = score_geral
            
            # MENSAGEM BASEADA NO GUIA (SEM EMOJIS)
            if guia_info['dentro_guia'] and guia_info['centralizado']:
                mensagem = f"[OK] Perfeitamente enquadrado! (score: {score_geral:.0%})"
            elif guia_info['dentro_guia']:
                mensagem = f"[!] Quase la! Centralize mais (score: {score_geral:.0%})"
            else:
                mensagem = f"[!] Posicione-se dentro do guia oval"
            
            # SEMPRE RETORNA TRUE (permite captura)
            return True, resultados, mensagem
            
        except Exception as e:
            return False, resultados, f"[X] Erro na validacao: {str(e)}"
    
    @staticmethod
    def _verificar_posicao_guia(face_center_x, face_center_y, face_width, face_height, img_width, img_height):
        """
        Verifica se o rosto está dentro e centralizado no guia oval VERTICAL
        """
        # GUIA OVAL VERTICAL (RETRATO) - Altura maior que largura
        guia_width = img_width * 0.30   # 30% largura (MENOR)
        guia_height = img_height * 0.55  # 55% altura (MAIOR)
        guia_center_x = img_width / 2
        guia_center_y = img_height / 2
        
        # Calcular distância do centro do rosto ao centro do guia
        dist_x = abs(face_center_x - guia_center_x)
        dist_y = abs(face_center_y - guia_center_y)
        
        # Verificar se está dentro do oval (equação da elipse)
        dentro_oval = ((dist_x ** 2) / (guia_width ** 2)) + ((dist_y ** 2) / (guia_height ** 2)) <= 1
        
        # Verificar centralização (tolerância de 15%)
        centralizado = dist_x < (guia_width * 0.15) and dist_y < (guia_height * 0.15)
        
        # Verificar tamanho ideal
        tamanho_ideal_min = guia_width * 0.65
        tamanho_ideal_max = guia_width * 1.3
        tamanho_ok = tamanho_ideal_min <= face_width <= tamanho_ideal_max
        
        # Direção de movimento
        direcao = ""
        if not centralizado:
            if dist_x > guia_width * 0.15:
                direcao = "ESQUERDA" if face_center_x > guia_center_x else "DIREITA"
            if dist_y > guia_height * 0.15:
                if direcao:
                    direcao += " e "
                direcao += "CIMA" if face_center_y > guia_center_y else "BAIXO"
        
        # Instrução de tamanho
        instrucao_tamanho = ""
        if face_width < tamanho_ideal_min:
            instrucao_tamanho = "Aproxime-se da camera"
        elif face_width > tamanho_ideal_max:
            instrucao_tamanho = "Afaste-se da camera"
        
        return {
            'dentro_guia': dentro_oval,
            'centralizado': centralizado,
            'tamanho_ok': tamanho_ok,
            'direcao': direcao,
            'instrucao_tamanho': instrucao_tamanho,
            'distancia_centro': (dist_x, dist_y)
        }
    
    @staticmethod
    def _calcular_nitidez(frame):
        """Calcula nitidez usando Variância de Laplaciano"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return variance
    
    @staticmethod
    def _calcular_iluminacao(frame):
        """Calcula iluminação média (0.0 a 1.0)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray) / 255.0
        return mean_brightness
    
    @staticmethod
    def desenhar_feedback_visual(frame, resultados, mensagem):
        """Desenha feedback visual com GUIA OVAL VERTICAL (SEM EMOJIS)"""
        frame_copy = frame.copy()
        height, width = frame.shape[:2]
        
        # ========== DESENHAR GUIA OVAL VERTICAL ==========
        guia_width = int(width * 0.30)
        guia_height = int(height * 0.55)
        guia_center = (width // 2, height // 2)
        
        # Cor do guia baseada no enquadramento
        if resultados.get('dentro_guia') and resultados.get('centralizado'):
            guia_cor = (0, 255, 0)  # Verde - Perfeito!
            guia_espessura = 4
        elif resultados.get('dentro_guia'):
            guia_cor = (0, 165, 255)  # Laranja - Quase lá
            guia_espessura = 3
        else:
            guia_cor = (0, 0, 255)  # Vermelho - Fora do guia
            guia_espessura = 3
        
        # DESENHAR OVAL PRINCIPAL
        cv2.ellipse(frame_copy, guia_center, 
                   (guia_width, guia_height),
                   0, 0, 360, guia_cor, guia_espessura, cv2.LINE_AA)
        
        # DESENHAR OVAL INTERNO (ZONA IDEAL)
        cv2.ellipse(frame_copy, guia_center, 
                   (int(guia_width * 0.85), int(guia_height * 0.85)), 
                   0, 0, 360, guia_cor, 2, cv2.LINE_AA)
        
        # DESENHAR LINHAS DE CENTRALIZAÇÃO
        cv2.line(frame_copy, 
                (width // 2, guia_center[1] - guia_height - 20), 
                (width // 2, guia_center[1] + guia_height + 20), 
                guia_cor, 1, cv2.LINE_AA)
        
        cv2.line(frame_copy, 
                (guia_center[0] - guia_width - 20, height // 2), 
                (guia_center[0] + guia_width + 20, height // 2), 
                guia_cor, 1, cv2.LINE_AA)
        
        # TEXTO NO CENTRO DO GUIA
        if not resultados.get('rosto_detectado'):
            texto_guia = "Posicione seu rosto"
            texto_guia2 = "dentro do oval"
            
            (text_width, text_height), _ = cv2.getTextSize(texto_guia, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(frame_copy, texto_guia, 
                       (guia_center[0] - text_width // 2, guia_center[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            (text_width2, text_height2), _ = cv2.getTextSize(texto_guia2, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(frame_copy, texto_guia2, 
                       (guia_center[0] - text_width2 // 2, guia_center[1] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # ========== PAINEL DE STATUS SUPERIOR ==========
        if "[OK]" in mensagem:
            cor_status = (0, 255, 0)  # Verde
        elif "[!]" in mensagem:
            cor_status = (0, 165, 255)  # Laranja
        else:
            cor_status = (0, 0, 255)  # Vermelho
        
        panel_height = 140 if len(resultados.get('warnings', [])) > 0 else 120
        cv2.rectangle(frame_copy, (10, 10), (width - 10, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame_copy, (10, 10), (width - 10, panel_height), cor_status, 3)
        
        cv2.putText(frame_copy, mensagem, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_status, 2, cv2.LINE_AA)
        
        if resultados['rosto_detectado']:
            metricas = [
                f"Nitidez: {resultados['nitidez']:.0f}/500",
                f"Luz: {resultados['iluminacao']:.0%}",
                f"Score: {resultados['score_geral']:.0%}"
            ]
            
            if resultados.get('centralizado'):
                metricas.append("[OK] Centralizado")
            if resultados.get('dentro_guia'):
                metricas.append("[OK] No guia")
            
            texto = " | ".join(metricas)
            cv2.putText(frame_copy, texto, (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            if resultados.get('warnings'):
                y_warning = 110
                for warning in resultados['warnings'][:2]:
                    cv2.putText(frame_copy, warning, (20, y_warning), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1, cv2.LINE_AA)
                    y_warning += 25
        
        # ========== INSTRUÇÕES LATERAIS ==========
        instructions = [
            "ENQUADRAMENTO:",
            "",
            "1. Rosto dentro",
            "   do oval VERTICAL",
            "",
            "2. Centralize nas",
            "   linhas cruzadas",
            "",
            "3. Verde = OK",
            "   para capturar"
        ]
        
        panel_x = width - 230
        panel_y_start = height - 240
        
        cv2.rectangle(frame_copy, 
                     (panel_x - 10, panel_y_start - 10), 
                     (width - 10, height - 10), 
                     (0, 0, 0), -1)
        cv2.rectangle(frame_copy, 
                     (panel_x - 10, panel_y_start - 10), 
                     (width - 10, height - 10), 
                     (100, 100, 100), 2)
        
        y_pos = panel_y_start
        for instruction in instructions:
            if instruction == "ENQUADRAMENTO:":
                cv2.putText(frame_copy, instruction, (panel_x, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 215, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame_copy, instruction, (panel_x, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            y_pos += 24
        
        return frame_copy
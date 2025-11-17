import numpy as np
from deepface import DeepFace
from typing import Tuple, Optional, Dict

class FaceRecognitionSystem:
    """Sistema híbrido de reconhecimento facial OTIMIZADO"""
    
    # Modelos disponíveis do DeepFace
    MODELS = {
        'VGG-Face': {'size': 2622, 'accuracy': 0.9865, 'speed': 'slow'},
        'Facenet512': {'size': 512, 'accuracy': 0.9965, 'speed': 'fast'},  # ⭐ MELHOR
        'Facenet': {'size': 128, 'accuracy': 0.9915, 'speed': 'fast'},
        'OpenFace': {'size': 128, 'accuracy': 0.9301, 'speed': 'fast'},
        'DeepFace': {'size': 4096, 'accuracy': 0.9743, 'speed': 'slow'},
        'DeepID': {'size': 160, 'accuracy': 0.9747, 'speed': 'medium'},
        'ArcFace': {'size': 512, 'accuracy': 0.9981, 'speed': 'medium'},  # ⭐ MAIS PRECISO
    }
    
    DEFAULT_MODEL = 'Facenet512'
    
    @staticmethod
    def normalizar_embedding(embedding):
        """
        ✅ NORMALIZAR EMBEDDING (L2 normalization)
        Transforma o vetor para ter magnitude 1
        Torna a comparação mais robusta a variações (óculos, barba, etc)
        """
        emb = np.array(embedding)
        norm = np.linalg.norm(emb)
        
        if norm == 0:
            return emb
        
        return emb / norm
    
    @staticmethod
    def gerar_embedding(frame, model_name=None):
        """
        Gera embedding facial NORMALIZADO
        ⭐ FORÇA USO DO FACENET512 PARA CONSISTÊNCIA
        """
        model_name = 'Facenet512'
        
        try:
            # 1. EXTRAIR FACE
            faces = DeepFace.extract_faces(
                img_path=frame,
                enforce_detection=True,
                detector_backend='opencv',
                align=True
            )
            
            if not faces:
                return None, None, 0.0
            
            face_data = faces[0]
            facial_area = face_data['facial_area']
            confidence = face_data.get('confidence', 0.0)
            
            # 2. GERAR EMBEDDING
            embedding_result = DeepFace.represent(
                img_path=frame,
                model_name=model_name,
                enforce_detection=False,
                detector_backend='skip'
            )
            
            embedding = embedding_result[0]['embedding']
            
            # ✅ VALIDAR TAMANHO (DEVE SER 512)
            if len(embedding) != 512:
                raise Exception(f"Embedding inválido: {len(embedding)} dimensões (esperado: 512)")
            
            # ✅ NORMALIZAR EMBEDDING PARA MELHOR COMPARAÇÃO
            embedding_normalizado = FaceRecognitionSystem.normalizar_embedding(embedding)
            
            print(f"✅ Embedding gerado:")
            print(f"   Modelo: {model_name}")
            print(f"   Dimensões: {len(embedding_normalizado)}")
            print(f"   Magnitude: {np.linalg.norm(embedding_normalizado):.3f}")  # ~1.0
            print(f"   Confiança: {confidence:.2%}")
            print(f"   Área facial: {facial_area}")
            
            return embedding_normalizado, facial_area, confidence
            
        except Exception as e:
            print(f"❌ Erro ao gerar embedding: {e}")
            import traceback
            traceback.print_exc()
            return None, None, 0.0
    
    @staticmethod
    def calcular_similaridade(embedding1, embedding2, method='cosine'):
        """
        ✅ CALCULAR SIMILARIDADE (APENAS NUMPY, SEM SCIPY)
        
        Methods:
        - 'cosine': Similaridade de cosseno (0-1, maior = mais similar)
        - 'euclidean': Distância euclidiana (menor = mais similar)
        """
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        if method == 'cosine':
            # Similaridade de cosseno usando numpy
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        
        elif method == 'euclidean':
            # Distância euclidiana usando numpy
            distance = np.linalg.norm(emb1 - emb2)
            
            # Normalizar para 0-1 (quanto menor a distância, maior a similaridade)
            max_distance = 150  # Threshold típico para Facenet512
            similarity = max(0, 1 - (distance / max_distance))
            
            return float(similarity)
        
        else:
            raise ValueError(f"Método desconhecido: {method}")
    
    @staticmethod
    def comparar_faces_hibrido_otimizado(current_embedding, current_area, stored_embedding, stored_area):
        """
        ✅ COMPARAÇÃO HÍBRIDA OTIMIZADA (SEM SCIPY)
        
        Usa múltiplas métricas para maior robustez a variações:
        - Similaridade de cosseno (peso 70%) - mais robusto
        - Distância euclidiana (peso 20%) - complementar
        - Comparação geométrica facial (peso 10%)
        
        Retorna: score_final (0.0 a 1.0)
        """
        try:
            # 1️⃣ SIMILARIDADE DE COSSENO (mais robusto a óculos/barba)
            cosine_similarity = FaceRecognitionSystem.calcular_similaridade(
                current_embedding, 
                stored_embedding, 
                method='cosine'
            )
            
            # 2️⃣ DISTÂNCIA EUCLIDIANA (complementar)
            euclidean_similarity = FaceRecognitionSystem.calcular_similaridade(
                current_embedding,
                stored_embedding,
                method='euclidean'
            )
            
            # 3️⃣ COMPARAÇÃO GEOMÉTRICA (proporções faciais)
            geometric_score = 0.5  # Default
            
            if current_area and stored_area:
                # Proporção largura/altura
                current_ratio = current_area.get('w', 0) / max(current_area.get('h', 1), 1)
                stored_ratio = stored_area.get('w', 0) / max(stored_area.get('h', 1), 1)
                
                ratio_diff = abs(current_ratio - stored_ratio)
                
                # ✅ TOLERÂNCIA MAIOR PARA VARIAÇÕES
                if ratio_diff < 0.15:
                    geometric_score = 1.0
                elif ratio_diff < 0.25:
                    geometric_score = 0.8
                elif ratio_diff < 0.35:
                    geometric_score = 0.6
                else:
                    geometric_score = 0.4
            
            # 4️⃣ SCORE FINAL PONDERADO (COSSENO MAIS IMPORTANTE)
            score_final = (
                cosine_similarity * 0.70 +      # 70% cosseno (AUMENTADO)
                euclidean_similarity * 0.20 +   # 20% euclidiana (REDUZIDO)
                geometric_score * 0.10          # 10% geometria
            )
            
            print(f"   📊 Scores: Cosseno={cosine_similarity:.3f} | Euclidiana={euclidean_similarity:.3f} | Geométrico={geometric_score:.3f} | FINAL={score_final:.3f}")
            
            return score_final
            
        except Exception as e:
            print(f"❌ Erro na comparação: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    @staticmethod
    def verificar_match(score_final, threshold=0.75):
        """
        ✅ VERIFICAÇÃO DE MATCH COM THRESHOLD MAIS RIGOROSO
        
        Thresholds:
        - >= 0.90: Match excelente (mesma pessoa, mesma foto)
        - >= 0.80: Match muito bom (mesma pessoa, condições similares)
        - >= 0.75: Match bom (mesma pessoa, foto diferente)
        - >= 0.70: Match aceitável (pode ter variações)
        - < 0.70: Não é a mesma pessoa
        """
        if score_final >= 0.90:
            return True, "Excelente"
        elif score_final >= 0.80:
            return True, "Muito Bom"
        elif score_final >= 0.75:
            return True, "Bom"
        elif score_final >= 0.70:
            return True, "Aceitável (verificar manualmente)"
        else:
            return False, "Rejeitado"
    
    @staticmethod
    def comparar_faces_hibrido(current_embedding, current_area, stored_embedding, stored_area):
        """
        ✅ VERSÃO COMPATÍVEL (chama a otimizada)
        Mantém compatibilidade com código antigo
        """
        return FaceRecognitionSystem.comparar_faces_hibrido_otimizado(
            current_embedding, current_area, stored_embedding, stored_area
        )
    
    @staticmethod
    def verificar_match(score_final, threshold=0.70):
        """
        ✅ VERIFICAÇÃO DE MATCH COM THRESHOLD ADAPTATIVO
        
        Thresholds:
        - >= 0.85: Match excelente (mesma pessoa, mesma foto)
        - >= 0.75: Match bom (mesma pessoa, foto diferente)
        - >= 0.70: Match aceitável (pode ter óculos/barba)
        - < 0.70: Não é a mesma pessoa
        """
        if score_final >= 0.85:
            return True, "Excelente"
        elif score_final >= 0.75:
            return True, "Bom"
        elif score_final >= threshold:
            return True, "Aceitável"
        else:
            return False, "Rejeitado"
    
    @staticmethod
    def recomendar_modelo(prioridade='balanced'):
        """Recomendar melhor modelo"""
        if prioridade == 'accuracy':
            return 'ArcFace'
        elif prioridade == 'speed':
            return 'Facenet'
        else:
            return 'Facenet512'
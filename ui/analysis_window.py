import cv2
import sys
import os
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QMessageBox, QProgressBar, QFileDialog)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from deepface import DeepFace

# Adicionar caminho
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqlite_db import find_employee_by_face, save_emotion_log
from utils.face_recognition import FaceRecognitionSystem
from utils.face_quality import FaceQualityValidator


class AnalysisWindow(QWidget):
    """Janela de análise emocional"""
    
    closed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.cap = None
        self.timer = None
        self.current_frame = None
        self.captured_photo = None
        self.employee_data = None
        self.emotion_samples = []
        self.analysis_timer = None
        self.analysis_progress = 0
        self.min_samples = 8  
        self.confidence_threshold = 50.0 
        
        self.init_ui()
        self.init_camera()
    
    def init_ui(self):
        """Inicializar interface"""
        self.setWindowTitle('EmpathIA - Análise Emocional')
        self.setFixedSize(900, 800)
        
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #f0f0f0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #007acc;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 12px 25px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #005f99;
            }
            QPushButton#CaptureButton {
                background-color: #FF9800;
                font-size: 16px;
                padding: 15px 35px;
            }
            QPushButton#CaptureButton:hover {
                background-color: #F57C00;
            }
            QPushButton#UploadButton {
                background-color: #9C27B0;
                font-size: 14px;
                padding: 12px 25px;
            }
            QPushButton#UploadButton:hover {
                background-color: #7B1FA2;
            }
            QPushButton#AnalyzeButton {
                background-color: #4CAF50;
                font-size: 16px;
                padding: 15px 35px;
            }
            QPushButton#AnalyzeButton:hover {
                background-color: #45a049;
            }
            QPushButton#CancelButton {
                background-color: #d32f2f;
            }
            QPushButton#CancelButton:hover {
                background-color: #b71c1c;
            }
            QLabel#VideoLabel {
                border: 3px solid #007acc;
                border-radius: 10px;
                background-color: #0a0a0a;
            }
            QLabel#TitleLabel {
                font-size: 24px;
                font-weight: bold;
                color: #4CAF50;
            }
            QLabel#StatusLabel {
                font-size: 14px;
                padding: 8px;
            }
            QLabel#EmployeeLabel {
                font-size: 16px;
                font-weight: bold;
                color: #FFD700;
                padding: 5px;
            }
            QProgressBar {
                border: 2px solid #007acc;
                border-radius: 5px;
                text-align: center;
                background-color: #2d2d2d;
                color: white;
                font-weight: bold;
                font-size: 13px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        
        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        
        # Título
        title = QLabel("🧠 Análise Emocional")
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # Label do funcionário reconhecido
        self.label_employee = QLabel("👤 Aguardando reconhecimento...")
        self.label_employee.setObjectName("EmployeeLabel")
        self.label_employee.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.label_employee)
        
        # Status
        self.label_status = QLabel("📹 Câmera inicializada - Posicione-se para reconhecimento")
        self.label_status.setObjectName("StatusLabel")
        self.label_status.setAlignment(Qt.AlignCenter)
        self.label_status.setStyleSheet("color: #FFD700;")
        main_layout.addWidget(self.label_status)
        
        # Vídeo
        self.label_video = QLabel()
        self.label_video.setObjectName("VideoLabel")
        self.label_video.setAlignment(Qt.AlignCenter)
        self.label_video.setFixedSize(640, 480)
        main_layout.addWidget(self.label_video, alignment=Qt.AlignCenter)
        
        # Barra de progresso
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Coletando amostras: %p%")
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Botões - Linha 1
        buttons_layout1 = QHBoxLayout()
        
        self.btn_capture = QPushButton("📸 Capturar da Câmera")
        self.btn_capture.setObjectName("CaptureButton")
        self.btn_capture.clicked.connect(self.capture_and_recognize)
        buttons_layout1.addWidget(self.btn_capture)
        
        # ✅ NOVO: Botão de upload
        self.btn_upload = QPushButton("📁 Upload de Foto")
        self.btn_upload.setObjectName("UploadButton")
        self.btn_upload.clicked.connect(self.upload_and_recognize)
        buttons_layout1.addWidget(self.btn_upload)
        
        main_layout.addLayout(buttons_layout1)
        
        # Botões - Linha 2
        buttons_layout2 = QHBoxLayout()
        
        self.btn_analyze = QPushButton("🧠 Iniciar Análise Emocional")
        self.btn_analyze.setObjectName("AnalyzeButton")
        self.btn_analyze.clicked.connect(self.start_analysis)
        self.btn_analyze.setEnabled(False)
        buttons_layout2.addWidget(self.btn_analyze)
        
        buttons_layout2.addStretch()
        
        self.btn_cancel = QPushButton("❌ Cancelar")
        self.btn_cancel.setObjectName("CancelButton")
        self.btn_cancel.clicked.connect(self.close)
        buttons_layout2.addWidget(self.btn_cancel)
        
        main_layout.addLayout(buttons_layout2)
        
        self.setLayout(main_layout)
    
    def init_camera(self):
        """Inicializar câmera"""
        try:
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                raise Exception("Câmera não detectada")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)
            
            print("✅ Câmera inicializada")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Erro na Câmera",
                f"Não foi possível acessar a câmera.\n\n{str(e)}"
            )
    
    def update_frame(self):
        """Atualizar frame da câmera COM GUIA OVAL"""
        if not self.cap or not self.cap.isOpened():
            return
        
        ret, frame = self.cap.read()
        
        if not ret:
            return
        
        self.current_frame = frame.copy()
        
        # ✅ VALIDAR QUALIDADE COM GUIA OVAL
        valido, resultados, mensagem = FaceQualityValidator.validar_qualidade_completa(frame)
        
        # ✅ DESENHAR GUIA OVAL E FEEDBACK
        frame_feedback = FaceQualityValidator.desenhar_feedback_visual(
            frame, resultados, mensagem
        )
        
        # Atualizar status
        if valido:
            if len(resultados.get('warnings', [])) > 0:
                self.label_status.setText(f"{mensagem}")
                self.label_status.setStyleSheet("color: #FF9800;")
            else:
                self.label_status.setText(mensagem)
                self.label_status.setStyleSheet("color: #4CAF50;")
            
            self.btn_capture.setEnabled(True)
        else:
            self.label_status.setText(mensagem)
            self.label_status.setStyleSheet("color: #F44336;")
            self.btn_capture.setEnabled(False)
        
        # Exibir frame
        rgb_frame = cv2.cvtColor(frame_feedback, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        scaled_pixmap = pixmap.scaled(
            self.label_video.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.label_video.setPixmap(scaled_pixmap)
    
    # ✅ NOVO MÉTODO: Upload e reconhecimento
    def upload_and_recognize(self):
        """Upload de foto e reconhecimento"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar Foto",
            "",
            "Imagens (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if not file_path:
            return
        
        try:
            # Carregar imagem
            image = cv2.imread(file_path)
            
            if image is None:
                QMessageBox.critical(
                    self,
                    "Erro",
                    "Não foi possível carregar a imagem.\n\nVerifique o formato do arquivo."
                )
                return
            
            # Parar câmera
            if self.timer:
                self.timer.stop()
            
            # Processar como se fosse captura
            self.current_frame = image.copy()
            self.process_recognition(image)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Erro",
                f"Erro ao processar upload.\n\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def capture_and_recognize(self):
        """Capturar foto e reconhecer funcionário"""
        if self.current_frame is None:
            QMessageBox.warning(self, "Aviso", "Nenhum frame disponível")
            return
        
        self.process_recognition(self.current_frame.copy())
    
    def process_recognition(self, image):
        """Processar reconhecimento de uma imagem"""
        try:
            # Validar qualidade
            valido, resultados, mensagem = FaceQualityValidator.validar_qualidade_completa(image)
            
            if not valido:
                QMessageBox.warning(
                    self,
                    "Rosto Não Detectado",
                    f"{mensagem}\n\nSelecione uma foto com rosto visível."
                )
                self.restart_camera()
                return
            
            # Verificar avisos
            warnings = resultados.get('warnings', [])
            if len(warnings) > 0:
                warning_text = "\n".join([f"• {w}" for w in warnings])
                
                reply = QMessageBox.question(
                    self,
                    "Avisos de Qualidade",
                    f"📊 Score: {resultados['score_geral']:.0%}\n\n"
                    f"Avisos:\n{warning_text}\n\n"
                    "Continuar mesmo assim?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.No:
                    self.restart_camera()
                    return
            
            # ✅ CAPTURAR FOTO
            self.captured_photo = image.copy()
            
            # Parar vídeo
            if self.timer:
                self.timer.stop()
            
            self.label_status.setText("🔍 Reconhecendo pessoa...")
            self.label_status.setStyleSheet("color: #FFD700;")
            
            # ✅ GERAR EMBEDDING COM FACENET512
            embedding, facial_area, confidence = FaceRecognitionSystem.gerar_embedding(
                self.captured_photo,
                model_name='Facenet512'
            )
            
            if embedding is None:
                QMessageBox.warning(
                    self,
                    "Erro",
                    "Não foi possível gerar embedding facial.\n\nTente novamente."
                )
                self.restart_camera()
                return
            
            # Validar tamanho
            if len(embedding) != 512:
                QMessageBox.critical(
                    self,
                    "Erro",
                    f"Embedding com tamanho incorreto: {len(embedding)}D\n"
                    f"Esperado: 512D (Facenet512)"
                )
                self.restart_camera()
                return
            
            # ✅ BUSCAR NO BANCO
            print("🔍 Buscando funcionário no banco SQLite...")
            self.employee_data = find_employee_by_face(embedding, facial_area)
            
            if self.employee_data is None:
                # NÃO CADASTRADO
                self.label_status.setText("❌ Pessoa não cadastrada")
                self.label_status.setStyleSheet("color: #F44336;")
                
                reply = QMessageBox.question(
                    self,
                    "Cadastro Necessário",
                    "❌ Você não está cadastrado no sistema.\n\n"
                    "É necessário cadastrar antes de fazer análises.\n\n"
                    "Deseja ir para a tela de cadastro?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    self.close()
                    from ui.register_window import RegisterFormWindow
                    self.register_window = RegisterFormWindow()
                    self.register_window.show()
                else:
                    self.restart_camera()
                
                return
            
            # ✅ RECONHECIDO!
            employee_name = self.employee_data.get('name', 'Desconhecido')
            employee_role = self.employee_data.get('role', 'N/A')
            employee_id = self.employee_data.get('_id', 'N/A')
            
            self.label_employee.setText(f"👤 {employee_name} - {employee_role}")
            self.label_status.setText(f"✅ Reconhecido! Clique em 'Iniciar Análise'")
            self.label_status.setStyleSheet("color: #4CAF50;")
            
            # Mostrar foto capturada com feedback
            frame_feedback = FaceQualityValidator.desenhar_feedback_visual(
                self.captured_photo, resultados, f"✅ {employee_name} reconhecido!"
            )
            
            rgb_frame = cv2.cvtColor(frame_feedback, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            scaled_pixmap = pixmap.scaled(
                self.label_video.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.label_video.setPixmap(scaled_pixmap)
            
            print(f"✅ Funcionário reconhecido: {employee_name} (ID: {employee_id})")
            
            # Habilitar botão de análise
            self.btn_analyze.setEnabled(True)
            self.btn_capture.setText("🔄 Capturar Novamente")
            self.btn_upload.setText("🔄 Upload Novamente")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Erro",
                f"Erro ao processar reconhecimento.\n\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
            self.restart_camera()
    
    def start_analysis(self):
        """Iniciar análise emocional DA FOTO CAPTURADA (otimizada)"""
        if self.captured_photo is None:
            QMessageBox.warning(
                self,
                "Aviso",
                "Capture uma foto primeiro antes de analisar."
            )
            return
        
        if self.employee_data is None:
            QMessageBox.warning(
                self,
                "Aviso",
                "Funcionário não reconhecido. Capture novamente."
            )
            return
        
        try:
            employee_name = self.employee_data.get('name', 'Desconhecido')
            
            QMessageBox.information(
                self,
                "Análise Iniciada",
                f"✅ Olá, {employee_name}!\n\n"
                f"🧠 Analisando emoções com alta confiança...\n\n"
                f"⏱️ Coletando {self.min_samples} amostras da sua foto\n"
                f"com diferentes parâmetros para maior precisão."
            )
            
            self.label_status.setText("🧠 Analisando emoções da foto...")
            self.label_status.setStyleSheet("color: #2196F3;")
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Coletando amostras: 0%")
            
            self.btn_capture.setEnabled(False)
            self.btn_upload.setEnabled(False)
            self.btn_analyze.setEnabled(False)
            
            self.emotion_samples = []
            self.collect_emotion_samples_from_photo()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Erro",
                f"Erro ao iniciar análise.\n\n{str(e)}"
            )
            import traceback
            traceback.print_exc()

    def collect_emotion_samples_from_photo(self):
        """
        ✅ COLETAR MÚLTIPLAS AMOSTRAS DA MESMA FOTO
        Usa diferentes backends e configurações do DeepFace para maior confiança
        """
        try:
            sample_count = 0
            max_attempts = self.min_samples * 3
            
            print(f"\n🔍 Iniciando análise multi-sample da foto...")
            
            # ✅ APENAS BACKENDS QUE FUNCIONAM (REMOVIDO SSD)
            backends = ['opencv', 'mtcnn']  # SEM SSD
            enforce_detections = [False, True]
            
            for attempt in range(max_attempts):
                try:
                    # Variar configurações para obter amostras diferentes
                    backend = backends[attempt % len(backends)]
                    enforce = enforce_detections[attempt % len(enforce_detections)]
                    
                    # ✅ ANALISAR COM DEEPFACE (MODELO OTIMIZADO)
                    emotion_result = DeepFace.analyze(
                        img_path=self.captured_photo.copy(),
                        actions=['emotion'],
                        enforce_detection=enforce,
                        detector_backend=backend,
                        align=True  # ✅ ALINHAMENTO FACIAL MELHORA ACURÁCIA
                    )
                    
                    if emotion_result and len(emotion_result) > 0:
                        analysis = emotion_result[0]
                        
                        # ✅ VALIDAR SE FACE FOI DETECTADA
                        if 'region' not in analysis or 'emotion' not in analysis:
                            continue
                        
                        emotions = analysis['emotion']
                        dominant_emotion = analysis['dominant_emotion']
                        confidence = emotions[dominant_emotion]
                        
                        # ✅ THRESHOLD DINÂMICO (aceita 50%+)
                        if confidence >= 50.0:
                            self.emotion_samples.append({
                                'dominant_emotion': dominant_emotion,
                                'confidence': confidence,
                                'all_emotions': emotions,
                                'backend': backend,
                                'timestamp': datetime.now().strftime('%H:%M:%S.%f')[:-3]
                            })
                            
                            sample_count += 1
                            progress = int((sample_count / self.min_samples) * 100)
                            self.progress_bar.setValue(progress)
                            self.progress_bar.setFormat(f"Amostras: {sample_count}/{self.min_samples}")
                            
                            print(f"  ✅ Amostra {sample_count}: {dominant_emotion} ({confidence:.1f}%) [{backend}]")
                            
                            if sample_count >= self.min_samples:
                                break
                        else:
                            print(f"  ⚠️ Confiança baixa: {confidence:.1f}% [{backend}] - ignorando")
                
                except Exception as e:
                    # Silencioso - não mostrar erros
                    continue
            
            # ✅ SE NÃO CONSEGUIU AMOSTRAS SUFICIENTES, TENTAR COM THRESHOLD MENOR
            if len(self.emotion_samples) < self.min_samples:
                print(f"\n⚠️ Apenas {len(self.emotion_samples)} amostras. Tentando threshold 40%...")
                
                for attempt in range(self.min_samples * 2):
                    try:
                        backend = backends[attempt % len(backends)]
                        
                        emotion_result = DeepFace.analyze(
                            img_path=self.captured_photo.copy(),
                            actions=['emotion'],
                            enforce_detection=False,
                            detector_backend=backend,
                            align=True
                        )
                        
                        if emotion_result and len(emotion_result) > 0:
                            analysis = emotion_result[0]
                            emotions = analysis.get('emotion', {})
                            dominant_emotion = analysis.get('dominant_emotion')
                            
                            if dominant_emotion and emotions:
                                confidence = emotions[dominant_emotion]
                                
                                # ✅ THRESHOLD REDUZIDO: 40%
                                if confidence >= 40.0:
                                    self.emotion_samples.append({
                                        'dominant_emotion': dominant_emotion,
                                        'confidence': confidence,
                                        'all_emotions': emotions,
                                        'backend': backend,
                                        'timestamp': datetime.now().strftime('%H:%M:%S.%f')[:-3]
                                    })
                                    
                                    if len(self.emotion_samples) >= self.min_samples:
                                        break
                    
                    except:
                        continue
            
            # ✅ PROCESSAR RESULTADOS
            if len(self.emotion_samples) >= self.min_samples:
                self.process_emotion_results()
            else:
                raise Exception(
                    f"Não foi possível coletar amostras suficientes.\n"
                    f"Coletadas: {len(self.emotion_samples)}/{self.min_samples}\n\n"
                    f"Tente capturar uma foto com melhor iluminação\n"
                    f"e expressão facial mais clara."
                )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Erro na Análise",
                f"Não foi possível completar a análise.\n\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
            self.restart_camera()
        
    def process_emotion_results(self):
        """Processar resultados das múltiplas amostras COM ANÁLISE ESTATÍSTICA"""
        try:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("Processando resultados...")
            
            # ✅ 1. CONTAR FREQUÊNCIA DE CADA EMOÇÃO
            emotion_counts = {}
            emotion_confidences = {}
            
            for sample in self.emotion_samples:
                emotion = sample['dominant_emotion']
                confidence = sample['confidence']
                
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                if emotion not in emotion_confidences:
                    emotion_confidences[emotion] = []
                emotion_confidences[emotion].append(confidence)
            
            # ✅ 2. EMOÇÃO DOMINANTE = MAIS FREQUENTE
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            
            # ✅ 3. CONFIANÇA = MÉDIA DAS AMOSTRAS DA EMOÇÃO DOMINANTE
            avg_confidence = np.mean(emotion_confidences[dominant_emotion])
            
            # ✅ 4. CALCULAR CONSISTÊNCIA (% de amostras com a mesma emoção)
            frequency_ratio = emotion_counts[dominant_emotion] / len(self.emotion_samples)
            
            # ✅ 5. BÔNUS DE CONSISTÊNCIA (até +15%)
            consistency_bonus = frequency_ratio * 15
            
            # ✅ 6. PENALIDADE POR BAIXA CONFIANÇA INDIVIDUAL
            std_dev = np.std(emotion_confidences[dominant_emotion])
            stability_bonus = max(0, 5 - (std_dev / 10))  # Até +5% se estável
            
            # ✅ 7. CONFIANÇA FINAL AJUSTADA
            adjusted_confidence = min(100, avg_confidence + consistency_bonus + stability_bonus)
            
            # ✅ 8. CALCULAR MÉDIA DE TODAS AS EMOÇÕES (PARA GRÁFICOS)
            all_emotions_avg = {}
            for sample in self.emotion_samples:
                for emotion, score in sample['all_emotions'].items():
                    if emotion not in all_emotions_avg:
                        all_emotions_avg[emotion] = []
                    all_emotions_avg[emotion].append(score)
            
            for emotion in all_emotions_avg:
                all_emotions_avg[emotion] = np.mean(all_emotions_avg[emotion])
            
            # ✅ 9. LOG DETALHADO
            print(f"\n{'='*60}")
            print("ANÁLISE EMOCIONAL CONCLUÍDA")
            print(f"{'='*60}")
            print(f"📊 Amostras coletadas: {len(self.emotion_samples)}")
            print(f"🎯 Emoção dominante: {dominant_emotion.upper()}")
            print(f"📈 Frequência: {emotion_counts[dominant_emotion]}/{len(self.emotion_samples)} ({frequency_ratio:.0%})")
            print(f"💯 Confiança média: {avg_confidence:.1f}%")
            print(f"📊 Desvio padrão: {std_dev:.1f}%")
            print(f"🎁 Bônus consistência: +{consistency_bonus:.1f}%")
            print(f"🎁 Bônus estabilidade: +{stability_bonus:.1f}%")
            print(f"✅ Confiança final: {adjusted_confidence:.1f}%")
            print(f"\n📉 Distribuição completa:")
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"   • {emotion.capitalize()}: {count}x ({count/len(self.emotion_samples)*100:.0f}%)")
            print(f"{'='*60}\n")
            
            # ✅ 10. SALVAR NO BANCO
            employee_id = self.employee_data.get('_id')
            
            emotion_data = {
                'dominant_emotion': dominant_emotion,
                'confidence': adjusted_confidence,
                'all_emotions': all_emotions_avg,
                'analysis_duration': 0,
                'samples_collected': len(self.emotion_samples),
                'consistency': frequency_ratio * 100,
                'stability_score': 100 - std_dev,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            save_emotion_log(employee_id, emotion_data)
            
            # ✅ 11. MOSTRAR RESULTADO
            self.show_emotion_result(dominant_emotion, adjusted_confidence, all_emotions_avg, emotion_counts)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Erro",
                f"Erro ao processar resultados.\n\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
            self.restart_camera()

    def show_emotion_result(self, dominant_emotion, confidence, all_emotions, distribution):
        """Mostrar resultado OTIMIZADO da análise emocional"""
        emotion_emojis = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'surprise': '😲',
            'fear': '😨',
            'disgust': '🤢',
            'neutral': '😐'
        }
        
        emoji = emotion_emojis.get(dominant_emotion.lower(), '❓')
        
        # ✅ CALCULAR QUALIDADE DA ANÁLISE
        total_samples = sum(distribution.values())
        frequency = distribution[dominant_emotion]
        consistency_percent = (frequency / total_samples * 100) if total_samples > 0 else 0
        
        # ✅ CLASSIFICAÇÃO DE QUALIDADE APRIMORADA
        if consistency_percent >= 75 and confidence >= 85:
            quality = "EXCELENTE ⭐⭐⭐"
            quality_color = "#4CAF50"
            reliability = "Altamente confiável"
        elif consistency_percent >= 60 and confidence >= 70:
            quality = "BOA ⭐⭐"
            quality_color = "#FF9800"
            reliability = "Confiável"
        elif consistency_percent >= 50 and confidence >= 60:
            quality = "MODERADA ⭐"
            quality_color = "#FFC107"
            reliability = "Razoavelmente confiável"
        else:
            quality = "BAIXA ⚠️"
            quality_color = "#F44336"
            reliability = "Pouco confiável - considere refazer"
        
        # Formatar emoções ordenadas
        emotions_text = "\n".join([
            f"  • {emotion.capitalize()}: {score:.1f}%"
            for emotion, score in sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)
        ])
        
        # Formatar distribuição
        distribution_text = "\n".join([
            f"  • {emotion.capitalize()}: {count}x ({count/total_samples*100:.0f}%)"
            for emotion, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        ])
        
        result_message = (
            f"🧠 ANÁLISE EMOCIONAL CONCLUÍDA\n\n"
            f"👤 Funcionário: {self.employee_data.get('name')}\n"
            f"📊 Qualidade: {quality}\n"
            f"🎯 Confiabilidade: {reliability}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{emoji} EMOÇÃO DOMINANTE:\n"
            f"   {dominant_emotion.upper()}\n\n"
            f"📈 Confiança Final: {confidence:.1f}%\n"
            f"🎯 Consistência: {consistency_percent:.0f}% ({frequency}/{total_samples} amostras)\n"
            f"📸 Amostras Analisadas: {len(self.emotion_samples)}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📉 Distribuição nas {total_samples} amostras:\n\n"
            f"{distribution_text}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📈 Scores médios de todas as emoções:\n\n"
            f"{emotions_text}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"💡 Dica: Quanto maior a consistência e confiança,\n"
            f"mais precisa é a análise.\n\n"
            f"✅ Análise salva no banco de dados!"
        )
        
        QMessageBox.information(
            self,
            "Análise Concluída",
            result_message
        )
        
        self.label_status.setText(f"{emoji} {dominant_emotion.upper()} ({confidence:.0f}%) - {quality}")
        self.label_status.setStyleSheet(f"color: {quality_color};")
        
        self.reset_for_new_analysis()
            
    def reset_for_new_analysis(self):
        """Resetar interface para nova análise"""
        self.captured_photo = None
        self.employee_data = None
        self.emotion_samples = []
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.confidence_threshold = 70.0  # Resetar threshold
        
        self.label_employee.setText("👤 Aguardando reconhecimento...")
        self.btn_capture.setText("📸 Capturar da Câmera")
        self.btn_upload.setText("📁 Upload de Foto")
        self.btn_analyze.setEnabled(False)
        
        # Reiniciar câmera
        self.restart_camera()
    
    def restart_camera(self):
        """Reiniciar câmera"""
        if self.timer:
            self.timer.start(30)
        
        self.btn_capture.setEnabled(True)
        self.btn_upload.setEnabled(True)
        self.label_status.setText("📹 Câmera ativa - Posicione-se para nova captura")
        self.label_status.setStyleSheet("color: #FFD700;")
    
    def closeEvent(self, event):
        """Liberar recursos ao fechar"""
        if self.timer:
            self.timer.stop()
        if self.analysis_timer:
            self.analysis_timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        self.closed.emit()
        event.accept()
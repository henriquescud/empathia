import cv2
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QMessageBox, QComboBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from deepface import DeepFace


class RegistrationDialog(QDialog):
    """Diálogo para cadastro de funcionários"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cadastrar Funcionário")
        self.setModal(True)
        self.resize(800, 700)
        
        self.camera = None
        self.timer = None
        self.captured_frame = None
        self.face_embedding = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Configurar interface"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Cadastro de Funcionário")
        title_font = QFont("Arial", 16, QFont.Bold)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Campos de entrada
        form_layout = QVBoxLayout()
        
        # Nome
        form_layout.addWidget(QLabel("Nome:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Digite o nome completo")
        form_layout.addWidget(self.name_input)
        
        # Cargo
        form_layout.addWidget(QLabel("Cargo:"))
        self.role_input = QLineEdit()
        self.role_input.setPlaceholderText("Ex: Desenvolvedor, Gerente, etc.")
        form_layout.addWidget(self.role_input)
        
        # Departamento
        form_layout.addWidget(QLabel("Departamento:"))
        self.department_combo = QComboBox()
        self.department_combo.addItems([
            "Selecione...",
            "TI",
            "RH",
            "Financeiro",
            "Vendas",
            "Marketing",
            "Operações",
            "Outro"
        ])
        form_layout.addWidget(self.department_combo)
        
        layout.addLayout(form_layout)
        
        # Área de captura de foto
        photo_label = QLabel("Foto Facial:")
        photo_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(photo_label)
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #3498db; background-color: #ecf0f1;")
        layout.addWidget(self.video_label)
        
        # Status da captura
        self.capture_status = QLabel("Status: Aguardando captura...")
        self.capture_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.capture_status)
        
        # Botões de câmera
        camera_buttons = QHBoxLayout()
        
        self.start_camera_btn = QPushButton("Iniciar Câmera")
        self.start_camera_btn.clicked.connect(self.start_camera)
        camera_buttons.addWidget(self.start_camera_btn)
        
        self.capture_btn = QPushButton("Capturar Foto")
        self.capture_btn.clicked.connect(self.capture_photo)
        self.capture_btn.setEnabled(False)
        camera_buttons.addWidget(self.capture_btn)
        
        self.stop_camera_btn = QPushButton("Parar Câmera")
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.stop_camera_btn.setEnabled(False)
        camera_buttons.addWidget(self.stop_camera_btn)
        
        layout.addLayout(camera_buttons)
        
        # Botões de ação
        buttons_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Salvar Cadastro")
        self.save_btn.clicked.connect(self.accept)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-size: 14px;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        buttons_layout.addWidget(self.save_btn)
        
        cancel_btn = QPushButton("Cancelar")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-size: 14px;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        buttons_layout.addWidget(cancel_btn)
        
        layout.addLayout(buttons_layout)
        
    def start_camera(self):
        """Iniciar câmera"""
        self.camera = cv2.VideoCapture(0)
        
        if not self.camera.isOpened():
            QMessageBox.critical(self, "Erro", "Não foi possível abrir a câmera!")
            return
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(30)
        
        self.start_camera_btn.setEnabled(False)
        self.capture_btn.setEnabled(True)
        self.stop_camera_btn.setEnabled(True)
        self.capture_status.setText("Status: Câmera ativa - Posicione seu rosto")
        
    def stop_camera(self):
        """Parar câmera"""
        if self.timer:
            self.timer.stop()
        
        if self.camera:
            self.camera.release()
        
        self.start_camera_btn.setEnabled(True)
        self.capture_btn.setEnabled(False)
        self.stop_camera_btn.setEnabled(False)
        
    def update_camera(self):
        """Atualizar frame da câmera"""
        ret, frame = self.camera.read()
        
        if not ret:
            return
        
        # Converter para RGB e exibir
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        
    def capture_photo(self):
        """Capturar foto e extrair embedding"""
        if not self.camera:
            return
        
        ret, frame = self.camera.read()
        
        if not ret:
            QMessageBox.warning(self, "Erro", "Não foi possível capturar a foto!")
            return
        
        try:
            # Extrair embedding facial
            embedding_obj = DeepFace.represent(frame, model_name='Facenet', enforce_detection=True)
            
            if isinstance(embedding_obj, list):
                self.face_embedding = embedding_obj[0]['embedding']
            else:
                self.face_embedding = embedding_obj['embedding']
            
            self.captured_frame = frame
            
            # Parar câmera
            self.stop_camera()
            
            # Mostrar foto capturada
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)
            
            self.capture_status.setText("Status: Foto capturada com sucesso! ✓")
            self.capture_status.setStyleSheet("color: green; font-weight: bold;")
            self.save_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao processar face:\n{str(e)}\n\nCertifique-se de que seu rosto está visível!")
            
    def get_employee_data(self):
        """Retornar dados do funcionário cadastrado"""
        name = self.name_input.text().strip()
        role = self.role_input.text().strip()
        department = self.department_combo.currentText()
        
        if not name or not role or department == "Selecione..." or not self.face_embedding:
            QMessageBox.warning(self, "Dados Incompletos", 
                              "Por favor, preencha todos os campos e capture uma foto!")
            return None
        
        return {
            'name': name,
            'role': role,
            'department': department,
            'embedding': self.face_embedding,
            'photo': self.captured_frame  # ✅ Incluir a foto capturada
        }
    
    def closeEvent(self, event):
        """Limpar recursos ao fechar"""
        self.stop_camera()
        event.accept()
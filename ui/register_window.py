import cv2
import sys
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QLineEdit, QMessageBox, QTableWidget,
                             QTableWidgetItem, QHeaderView, QStackedWidget,
                             QFormLayout, QGroupBox, QDialog, QTextEdit, 
                             QScrollArea, QFileDialog)  # ✅ ADICIONAR QFileDialog
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from deepface import DeepFace
import json
from datetime import datetime
import traceback

# Adicionar caminho
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqlite_db import save_employee_data, get_all_employees, delete_employee, get_employee_emotion_history
from utils.face_quality import FaceQualityValidator
from utils.face_recognition import FaceRecognitionSystem


class EmployeeDetailDialog(QDialog):
    """Janela de detalhes completos do funcionário"""
    
    def __init__(self, employee_data, parent=None):
        super().__init__(parent)
        self.employee_data = employee_data
        self.init_ui()
        
    def init_ui(self):
        """Inicializar interface de detalhes"""
        
        # ✅ DEBUG: Ver o que está vindo do banco
        print("\n" + "="*60)
        print("🔍 DEBUG - Dados do funcionário:")
        print("="*60)
        print(f"Nome: {self.employee_data.get('name')}")
        print(f"Tem campo 'photo': {self.employee_data.get('photo') is not None}")
        print(f"Tipo de 'photo': {type(self.employee_data.get('photo'))}")
        print(f"Tem campo 'photo_base64': {self.employee_data.get('photo_base64') is not None}")
        if self.employee_data.get('photo_base64'):
            print(f"Tamanho Base64: {len(self.employee_data['photo_base64'])} chars")
        print(f"Tem campo 'photo_path': {self.employee_data.get('photo_path') is not None}")
        print("="*60 + "\n")
        
        self.setWindowTitle(f'Detalhes - {self.employee_data.get("name", "N/A")}')
        self.setFixedSize(900, 800)
        
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #f0f0f0;
            }
            QLabel {
                color: #f0f0f0;
                font-size: 13px;
            }
            QLabel#TitleLabel {
                font-size: 22px;
                font-weight: bold;
                color: #4CAF50;
                padding: 15px;
            }
            QLabel#PhotoLabel {
                border: 3px solid #007acc;
                border-radius: 10px;
                background-color: #0a0a0a;
            }
            QLabel#SectionLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2196F3;
                padding: 8px;
                border-bottom: 2px solid #2196F3;
                margin-top: 10px;
            }
            QTextEdit {
                background-color: #2d2d2d;
                color: #f0f0f0;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 12px 30px;
                border: none;
                border-radius: 6px;
                min-width: 140px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton#DeleteButton {
                background-color: #F44336;
            }
            QPushButton#DeleteButton:hover {
                background-color: #D32F2F;
            }
            QPushButton#ChangePhotoButton {
                background-color: #9C27B0;
            }
            QPushButton#ChangePhotoButton:hover {
                background-color: #7B1FA2;
            }
            QGroupBox {
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 12px;
                padding: 15px;
                background-color: #252525;
            }
            QGroupBox::title {
                color: #FFD700;
                font-weight: bold;
                font-size: 15px;
                padding: 0 10px;
            }
            QScrollArea {
                border: none;
                background-color: #1e1e1e;
            }
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666;
            }
        """)
        
        # Layout principal com scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        content_widget = QWidget()
        main_layout = QVBoxLayout(content_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # ========== TÍTULO ==========
        title = QLabel(f"👤 {self.employee_data.get('name', 'N/A')}")
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # ✅ ========== SEÇÃO: FOTO DO FUNCIONÁRIO (CORRIGIDO) ==========
        photo_group = QGroupBox("📸 Foto Cadastrada")
        photo_layout = QVBoxLayout()
        
        self.label_photo = QLabel()
        self.label_photo.setObjectName("PhotoLabel")
        self.label_photo.setAlignment(Qt.AlignCenter)
        self.label_photo.setFixedSize(320, 320)
        
        # ✅ CARREGAR FOTO DO BASE64 OU PHOTO
        photo_loaded = False
        
        # Tentar carregar do campo 'photo' (imagem OpenCV já convertida)
        if self.employee_data.get('photo') is not None:
            try:
                photo_cv = self.employee_data['photo']
                
                # Converter OpenCV para QPixmap
                rgb_frame = cv2.cvtColor(photo_cv, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                
                scaled_pixmap = pixmap.scaled(
                    self.label_photo.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.label_photo.setPixmap(scaled_pixmap)
                photo_loaded = True
                
                print("✅ Foto carregada do campo 'photo'")
                
            except Exception as e:
                print(f"⚠️ Erro ao carregar foto do campo 'photo': {e}")
        
        # ✅ FALLBACK: Tentar carregar do Base64
        if not photo_loaded and self.employee_data.get('photo_base64'):
            try:
                import base64
                import numpy as np
                
                photo_base64 = self.employee_data['photo_base64']
                
                # Decodificar Base64
                img_data = base64.b64decode(photo_base64)
                nparr = np.frombuffer(img_data, np.uint8)
                photo_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if photo_cv is not None:
                    # Converter OpenCV para QPixmap
                    rgb_frame = cv2.cvtColor(photo_cv, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    
                    scaled_pixmap = pixmap.scaled(
                        self.label_photo.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.label_photo.setPixmap(scaled_pixmap)
                    photo_loaded = True
                    
                    print("✅ Foto carregada do Base64")
                
            except Exception as e:
                print(f"⚠️ Erro ao carregar do Base64: {e}")
                import traceback
                traceback.print_exc()
        
        # ✅ FALLBACK 2: Tentar carregar de photo_path
        if not photo_loaded and self.employee_data.get('photo_path'):
            photo_path = self.employee_data.get('photo_path')
            if os.path.exists(photo_path):
                try:
                    pixmap = QPixmap(photo_path)
                    scaled_pixmap = pixmap.scaled(
                        self.label_photo.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.label_photo.setPixmap(scaled_pixmap)
                    photo_loaded = True
                    
                    print(f"✅ Foto carregada do caminho: {photo_path}")
                    
                except Exception as e:
                    print(f"⚠️ Erro ao carregar de photo_path: {e}")
        
        # Se não conseguiu carregar de jeito nenhum
        if not photo_loaded:
            self.label_photo.setText("📷 Foto não disponível")
            self.label_photo.setStyleSheet("color: #FF9800; font-size: 14px;")
            print("❌ Foto não disponível em nenhum formato")
        
        photo_layout.addWidget(self.label_photo, alignment=Qt.AlignCenter)
        
        # ✅ BOTÃO TROCAR FOTO
        self.btn_change_photo = QPushButton("🔄 Trocar Foto")
        self.btn_change_photo.setObjectName("ChangePhotoButton")
        self.btn_change_photo.clicked.connect(self.change_photo)
        photo_layout.addWidget(self.btn_change_photo)
        
        photo_group.setLayout(photo_layout)
        main_layout.addWidget(photo_group)
        
        # ========== SEÇÃO: INFORMAÇÕES PESSOAIS ==========
        info_group = QGroupBox("📋 Informações Pessoais")
        info_layout = QFormLayout()
        info_layout.setSpacing(12)
        info_layout.setLabelAlignment(Qt.AlignRight)
        
        # ID
        id_label = QLabel(str(self.employee_data.get('_id', 'N/A')))
        id_label.setStyleSheet("color: #FFD700; font-weight: bold;")
        info_layout.addRow("🆔 ID:", id_label)
        
        # Nome
        name_label = QLabel(self.employee_data.get('name', 'N/A'))
        name_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        info_layout.addRow("👤 Nome:", name_label)
        
        # Cargo
        info_layout.addRow("💼 Cargo:", QLabel(self.employee_data.get('role', 'N/A')))
        
        # Departamento
        info_layout.addRow("🏢 Departamento:", QLabel(self.employee_data.get('department', 'N/A')))
        
        # Email
        email_label = QLabel(self.employee_data.get('email', 'N/A'))
        email_label.setStyleSheet("color: #2196F3;")
        info_layout.addRow("📧 Email:", email_label)
        
        # Data de Cadastro
        created_at = self.employee_data.get('created_at', 'N/A')
        info_layout.addRow("📅 Cadastrado em:", QLabel(created_at))
        
        info_group.setLayout(info_layout)
        main_layout.addWidget(info_group)
        
        # ========== SEÇÃO: DADOS BIOMÉTRICOS ==========
        bio_group = QGroupBox("🎯 Dados Biométricos Faciais")
        bio_layout = QVBoxLayout()
        bio_layout.setSpacing(10)
        
        # Confiança facial
        confidence = self.employee_data.get('face_confidence', 0.0)
        confidence_color = '#4CAF50' if confidence > 0.85 else '#FF9800' if confidence > 0.70 else '#F44336'
        confidence_label = QLabel(f"🎯 Confiança da Detecção Facial: {confidence:.2%}")
        confidence_label.setStyleSheet(f"color: {confidence_color}; font-weight: bold; font-size: 14px;")
        bio_layout.addWidget(confidence_label)
        
        # Área facial
        facial_area = self.employee_data.get('facial_area', {})
        if facial_area:
            area_width = facial_area.get('w', 0)
            area_height = facial_area.get('h', 0)
            area_x = facial_area.get('x', 0)
            area_y = facial_area.get('y', 0)
            
            area_info = QLabel(
                f"📐 Dimensões: {area_width}x{area_height} pixels | "
                f"Posição: ({area_x}, {area_y})"
            )
            area_info.setStyleSheet("color: #2196F3;")
            bio_layout.addWidget(area_info)
            
            # Olhos (se disponível)
            if 'left_eye' in facial_area and 'right_eye' in facial_area:
                left_eye = facial_area.get('left_eye')
                right_eye = facial_area.get('right_eye')
                eyes_info = QLabel(
                    f"👁️ Pontos dos Olhos: "
                    f"Esquerdo {left_eye} | Direito {right_eye}"
                )
                eyes_info.setStyleSheet("color: #9C27B0;")
                bio_layout.addWidget(eyes_info)
        
        # Separador
        separator = QLabel("─" * 80)
        separator.setStyleSheet("color: #555;")
        bio_layout.addWidget(separator)
        
        # Embedding
        embedding = self.employee_data.get('embedding', [])
        embedding_info = QLabel(f"🧬 Embedding Facial: {len(embedding)} dimensões (Modelo: Facenet512)")
        embedding_info.setStyleSheet("color: #00BCD4; font-weight: bold;")
        bio_layout.addWidget(embedding_info)
        
        if embedding:
            # Preview do embedding (primeiros 20 valores)
            embedding_preview = QTextEdit()
            embedding_preview.setReadOnly(True)
            embedding_preview.setMaximumHeight(120)
            
            preview_values = embedding[:20]
            preview_text = "[\n  " + ",\n  ".join([f"{v:.8f}" for v in preview_values])
            preview_text += f",\n  ... (+ {len(embedding) - 20} valores restantes)\n]"
            
            embedding_preview.setText(preview_text)
            
            preview_label = QLabel("📊 Preview do Embedding (primeiros 20 valores):")
            preview_label.setStyleSheet("color: #FFD700; font-size: 12px; margin-top: 5px;")
            bio_layout.addWidget(preview_label)
            bio_layout.addWidget(embedding_preview)
        
        bio_group.setLayout(bio_layout)
        main_layout.addWidget(bio_group)
        
        # ========== SEÇÃO: HISTÓRICO DE ANÁLISES EMOCIONAIS ==========
        history_group = QGroupBox("📈 Histórico de Análises Emocionais")
        history_layout = QVBoxLayout()
        history_layout.setSpacing(10)
        
        # Buscar histórico
        employee_id = self.employee_data.get('_id')
        emotion_logs = get_employee_emotion_history(employee_id)
        
        if emotion_logs:
            # Resumo
            history_summary = QLabel(f"📊 Total de Análises Realizadas: {len(emotion_logs)}")
            history_summary.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 14px;")
            history_layout.addWidget(history_summary)
            
            # Últimas 5 análises
            history_text = QTextEdit()
            history_text.setReadOnly(True)
            history_text.setMaximumHeight(180)
            
            history_content = "🕒 Últimas 5 Análises:\n" + "─" * 60 + "\n\n"
            
            emotion_emojis = {
                'happy': '😊',
                'sad': '😢',
                'angry': '😠',
                'surprise': '😲',
                'fear': '😨',
                'disgust': '🤢',
                'neutral': '😐'
            }
            
            for i, log in enumerate(emotion_logs[:5], 1):
                emotion = log.get('dominant_emotion', 'N/A')
                confidence = log.get('confidence', 0)
                timestamp = log.get('timestamp', 'N/A')
                duration = log.get('analysis_duration', 0)
                samples = log.get('samples_collected', 0)
                
                emoji = emotion_emojis.get(emotion.lower(), '❓')
                
                history_content += (
                    f"{i}. {emoji} {emotion.upper()}\n"
                    f"   Confiança: {confidence:.1f}%\n"
                    f"   Data/Hora: {timestamp}\n"
                    f"   Duração: {duration}s | Amostras: {samples}\n"
                    f"   {'-' * 55}\n\n"
                )
            
            history_text.setText(history_content)
            history_layout.addWidget(history_text)
            
            # Estatísticas
            emotions_count = {}
            total_confidence = 0
            
            for log in emotion_logs:
                emotion = log.get('dominant_emotion', 'N/A')
                emotions_count[emotion] = emotions_count.get(emotion, 0) + 1
                total_confidence += log.get('confidence', 0)
            
            most_common = max(emotions_count, key=emotions_count.get) if emotions_count else 'N/A'
            avg_confidence = total_confidence / len(emotion_logs) if emotion_logs else 0
            
            stats_layout = QHBoxLayout()
            
            most_common_label = QLabel(
                f"🏆 Emoção Predominante: {most_common.upper()} "
                f"({emotions_count.get(most_common, 0)}x)"
            )
            most_common_label.setStyleSheet("color: #FFD700; font-weight: bold;")
            stats_layout.addWidget(most_common_label)
            
            avg_conf_label = QLabel(f"📊 Confiança Média: {avg_confidence:.1f}%")
            avg_conf_label.setStyleSheet("color: #00BCD4; font-weight: bold;")
            stats_layout.addWidget(avg_conf_label)
            
            history_layout.addLayout(stats_layout)
            
        else:
            no_history = QLabel("⚠️ Nenhuma análise emocional registrada ainda")
            no_history.setStyleSheet("color: #FF9800; font-style: italic; font-size: 14px;")
            no_history.setAlignment(Qt.AlignCenter)
            history_layout.addWidget(no_history)
        
        history_group.setLayout(history_layout)
        main_layout.addWidget(history_group)
        
        # ========== SEÇÃO: DADOS TÉCNICOS (JSON) ==========
        json_group = QGroupBox("🔧 Dados Técnicos (Formato JSON)")
        json_layout = QVBoxLayout()
        
        json_text = QTextEdit()
        json_text.setReadOnly(True)
        json_text.setMaximumHeight(160)
        
        # Preparar JSON (sem embedding completo)
        display_data = self.employee_data.copy()
        if 'embedding' in display_data and len(display_data['embedding']) > 10:
            display_data['embedding'] = f"<Array com {len(display_data['embedding'])} valores>"
        
        try:
            json_formatted = json.dumps(display_data, indent=2, ensure_ascii=False)
        except:
            json_formatted = str(display_data)
        
        json_text.setText(json_formatted)
        json_layout.addWidget(json_text)
        
        json_group.setLayout(json_layout)
        main_layout.addWidget(json_group)
        
        # ========== BOTÕES DE AÇÃO ==========
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(15)
        
        btn_close = QPushButton("✅ Fechar")
        btn_close.clicked.connect(self.accept)
        buttons_layout.addWidget(btn_close)
        
        buttons_layout.addStretch()
        
        btn_delete = QPushButton("🗑️ Deletar Funcionário")
        btn_delete.setObjectName("DeleteButton")
        btn_delete.clicked.connect(self.delete_employee)
        buttons_layout.addWidget(btn_delete)
        
        main_layout.addLayout(buttons_layout)
        
        # Finalizar scroll
        scroll.setWidget(content_widget)
        
        dialog_layout = QVBoxLayout()
        dialog_layout.setContentsMargins(0, 0, 0, 0)
        dialog_layout.addWidget(scroll)
        self.setLayout(dialog_layout)
    
    # ✅ NOVO MÉTODO: TROCAR FOTO
    def change_photo(self):
        """Trocar foto do funcionário"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar Nova Foto",
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
            
            # Validar qualidade
            valido, resultados, mensagem = FaceQualityValidator.validar_qualidade_completa(image)
            
            if not valido:
                QMessageBox.warning(
                    self,
                    "Rosto Não Detectado",
                    f"{mensagem}\n\nSelecione uma foto com rosto visível."
                )
                return
            
            # Avisos de qualidade
            warnings = resultados.get('warnings', [])
            if len(warnings) > 0:
                warning_text = "\n".join([f"• {w}" for w in warnings])
                
                reply = QMessageBox.question(
                    self,
                    "Avisos de Qualidade",
                    f"📊 Score: {resultados['score_geral']:.0%}\n\n"
                    f"Avisos:\n{warning_text}\n\n"
                    "Usar esta foto mesmo assim?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.No:
                    return
            
            # Gerar novo embedding
            embedding, facial_area, confidence = FaceRecognitionSystem.gerar_embedding(
                image,
                model_name='Facenet512'
            )
            
            if embedding is None:
                QMessageBox.critical(
                    self,
                    "Erro",
                    "Não foi possível gerar embedding facial.\n\nTente outra foto."
                )
                return
            
            # ✅ CORRIGIR: Converter para lista se necessário
            if hasattr(embedding, 'tolist'):
                embedding_list = embedding.tolist()
            else:
                embedding_list = embedding  # Já é lista
            
            # Atualizar dados no banco
            from sqlite_db import update_employee_photo
            
            employee_id = self.employee_data.get('_id')
            success = update_employee_photo(
                employee_id,
                image,
                embedding_list,  # ✅ USAR A LISTA
                facial_area,
                confidence
            )
            
            if success:
                # Atualizar exibição
                rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                
                scaled_pixmap = pixmap.scaled(
                    self.label_photo.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.label_photo.setPixmap(scaled_pixmap)
                
                QMessageBox.information(
                    self,
                    "✅ Sucesso",
                    f"Foto atualizada com sucesso!\n\n"
                    f"📊 Nova Confiança: {confidence:.2%}\n"
                    f"🧬 Embedding: {len(embedding_list)}D"
                )
                
                # Atualizar dados locais
                self.employee_data['photo'] = image
                self.employee_data['embedding'] = embedding_list
                self.employee_data['facial_area'] = facial_area
                self.employee_data['face_confidence'] = confidence
                
            else:
                QMessageBox.critical(
                    self,
                    "Erro",
                    "Não foi possível atualizar a foto no banco de dados."
                )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Erro",
                f"Erro ao trocar foto.\n\n{str(e)}"
            )
        traceback.print_exc()
    
    def delete_employee(self):
        """Deletar funcionário com confirmação"""
        reply = QMessageBox.question(
            self,
            "⚠️ Confirmar Exclusão",
            f"Tem certeza que deseja deletar o funcionário:\n\n"
            f"📛 Nome: {self.employee_data.get('name')}\n"
            f"🆔 ID: {self.employee_data.get('_id')}\n\n"
            f"❌ ATENÇÃO: Esta ação NÃO pode ser desfeita!\n\n"
            f"Todos os dados biométricos e histórico de análises\n"
            f"emocionais serão permanentemente removidos.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            employee_id = self.employee_data.get('_id')
            employee_name = self.employee_data.get('name')
            
            if delete_employee(employee_id):
                QMessageBox.information(
                    self,
                    "✅ Sucesso",
                    f"Funcionário '{employee_name}' foi deletado com sucesso!\n\n"
                    f"O registro foi removido permanentemente do sistema."
                )
                self.accept()  # Fecha o diálogo
            else:
                QMessageBox.critical(
                    self,
                    "❌ Erro",
                    "Não foi possível deletar o funcionário.\n\n"
                    "Tente novamente ou verifique os logs do sistema."
                )


class RegisterWindow(QWidget):
    """Janela de gerenciamento de funcionários"""
    
    closed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.employees = []
        self.init_ui()
        self.load_employees()
        
    def init_ui(self):
        """Inicializar interface"""
        self.setWindowTitle('EmpathIA - Gerenciamento de Funcionários')
        self.setFixedSize(1100, 700)
        
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
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #005f99;
            }
            QPushButton#AddButton {
                background-color: #4CAF50;
                font-size: 16px;
                padding: 12px 30px;
            }
            QPushButton#AddButton:hover {
                background-color: #45a049;
            }
            QPushButton#DeleteButton {
                background-color: #d32f2f;
            }
            QPushButton#DeleteButton:hover {
                background-color: #b71c1c;
            }
            QPushButton#BackButton {
                background-color: #666;
            }
            QPushButton#BackButton:hover {
                background-color: #555;
            }
            QTableWidget {
                background-color: #2d2d2d;
                border: 2px solid #007acc;
                border-radius: 8px;
                gridline-color: #3d3d3d;
                alternate-background-color: #252525;
            }
            QTableWidget::item {
                padding: 8px;
                color: #f0f0f0;
            }
            QTableWidget::item:alternate {
                background-color: #252525;
            }
            QTableWidget::item:selected {
                background-color: #007acc;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #1a1a1a;
                color: #4CAF50;
                font-weight: bold;
                padding: 10px;
                border: none;
                border-bottom: 2px solid #007acc;
            }
            QLineEdit {
                background-color: #2d2d2d;
                border: 2px solid #007acc;
                border-radius: 6px;
                padding: 10px;
                color: #f0f0f0;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #4CAF50;
            }
            QGroupBox {
                border: 2px solid #007acc;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                font-size: 16px;
                font-weight: bold;
                color: #4CAF50;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel#TitleLabel {
                font-size: 24px;
                font-weight: bold;
                color: #4CAF50;
            }
            QLabel#TipLabel {
                color: #FFD700;
                font-style: italic;
                font-size: 13px;
                padding: 8px;
            }
        """)
        
        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        
        # Título
        title = QLabel("👥 Gerenciamento de Funcionários")
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # Área de lista de funcionários
        list_group = QGroupBox("📋 Funcionários Cadastrados")
        list_layout = QVBoxLayout()
        
        # Tabela de funcionários
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            'ID', 'Nome', 'Cargo', 'Departamento', 'Email', 'Cadastro'
        ])
        
        # Configurar largura das colunas
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        
        # ✅ DUPLO CLIQUE PARA ABRIR DETALHES (COM FOTO)
        self.table.cellDoubleClicked.connect(self.show_employee_details)
        
        list_layout.addWidget(self.table)
        
        # Dica de uso
        tip_label = QLabel("💡 Dica: Clique duplo para ver detalhes e foto do funcionário")
        tip_label.setObjectName("TipLabel")
        tip_label.setAlignment(Qt.AlignCenter)
        list_layout.addWidget(tip_label)
        
        # Botões de ação da tabela
        table_buttons = QHBoxLayout()
        
        self.btn_refresh = QPushButton("🔄 Atualizar Lista")
        self.btn_refresh.clicked.connect(self.load_employees)
        table_buttons.addWidget(self.btn_refresh)
        
        self.btn_details = QPushButton("👁️ Ver Detalhes")
        self.btn_details.clicked.connect(self.show_employee_details)
        table_buttons.addWidget(self.btn_details)
        
        self.btn_delete = QPushButton("🗑️ Excluir Selecionado")
        self.btn_delete.setObjectName("DeleteButton")
        self.btn_delete.clicked.connect(self.delete_selected)
        table_buttons.addWidget(self.btn_delete)
        
        table_buttons.addStretch()
        
        self.btn_add = QPushButton("➕ Adicionar Novo Funcionário")
        self.btn_add.setObjectName("AddButton")
        self.btn_add.clicked.connect(self.show_register_form)
        table_buttons.addWidget(self.btn_add)
        
        list_layout.addLayout(table_buttons)
        list_group.setLayout(list_layout)
        main_layout.addWidget(list_group)
        
        # Botão voltar
        back_layout = QHBoxLayout()
        back_layout.addStretch()
        
        self.btn_back = QPushButton("⬅️ Voltar ao Menu")
        self.btn_back.setObjectName("BackButton")
        self.btn_back.clicked.connect(self.close)
        back_layout.addWidget(self.btn_back)
        
        main_layout.addLayout(back_layout)
        
        self.setLayout(main_layout)
    
    def load_employees(self):
        """Carregar funcionários do banco"""
        try:
            self.employees = get_all_employees()
            
            # Limpar tabela
            self.table.setRowCount(0)
            
            # Preencher tabela
            for employee in self.employees:
                row = self.table.rowCount()
                self.table.insertRow(row)
                
                self.table.setItem(row, 0, QTableWidgetItem(str(employee.get('_id', ''))))
                self.table.setItem(row, 1, QTableWidgetItem(employee.get('name', '')))
                self.table.setItem(row, 2, QTableWidgetItem(employee.get('role', '')))
                self.table.setItem(row, 3, QTableWidgetItem(employee.get('department', '')))
                self.table.setItem(row, 4, QTableWidgetItem(employee.get('email', '')))
                
                created_at = employee.get('created_at', '')
                if created_at:
                    try:
                        # Formatar data
                        created_at = created_at.split('.')[0]  # Remover microsegundos
                    except:
                        pass
                
                self.table.setItem(row, 5, QTableWidgetItem(created_at))
            
            print(f"✅ {len(self.employees)} funcionários carregados")
            
        except Exception as e:
            QMessageBox.warning(
                self,
                "Erro ao Carregar",
                f"Não foi possível carregar a lista de funcionários.\n\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def show_employee_details(self):
        """Mostrar detalhes completos do funcionário selecionado (COM FOTO)"""
        selected_row = self.table.currentRow()
        
        if selected_row < 0:
            QMessageBox.warning(
                self,
                "⚠️ Aviso",
                "Selecione um funcionário na tabela para ver os detalhes."
            )
            return
        
        # Pegar dados completos do funcionário
        employee_data = self.employees[selected_row]
        
        # Abrir janela de detalhes
        detail_dialog = EmployeeDetailDialog(employee_data, self)
        if detail_dialog.exec_() == QDialog.Accepted:
            # Recarregar tabela após fechar (caso tenha deletado)
            self.load_employees()
    
    def show_register_form(self):
        """Mostrar formulário de cadastro"""
        self.register_form = RegisterFormWindow()
        self.register_form.show()
        
        # Atualizar lista quando fechar
        self.register_form.closed.connect(self.load_employees)
    
    def delete_selected(self):
        """Excluir funcionário selecionado"""
        selected_row = self.table.currentRow()
        
        if selected_row < 0:
            QMessageBox.warning(
                self,
                "⚠️ Aviso",
                "Selecione um funcionário na tabela para excluir."
            )
            return
        
        employee_id = self.table.item(selected_row, 0).text()
        employee_name = self.table.item(selected_row, 1).text()
        
        reply = QMessageBox.question(
            self,
            "⚠️ Confirmar Exclusão",
            f"Deseja realmente excluir o funcionário:\n\n"
            f"📛 {employee_name}\n\n"
            f"❌ Esta ação não pode ser desfeita.\n"
            f"Todos os dados e histórico serão perdidos.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if delete_employee(employee_id):
                QMessageBox.information(
                    self,
                    "✅ Sucesso",
                    f"Funcionário '{employee_name}' excluído com sucesso!"
                )
                self.load_employees()
            else:
                QMessageBox.critical(
                    self,
                    "❌ Erro",
                    "Não foi possível excluir o funcionário."
                )
    
    def closeEvent(self, event):
        """Emitir sinal ao fechar"""
        self.closed.emit()
        event.accept()


class RegisterFormWindow(QWidget):
    """Formulário de cadastro de funcionário"""
    
    closed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.cap = None
        self.timer = None
        self.current_frame = None
        self.captured_photo = None
        self.uploaded_photo = None  # ✅ NOVO
        
        self.init_ui()
        self.init_camera()
    
    def init_ui(self):
        """Inicializar interface do formulário"""
        self.setWindowTitle('EmpathIA - Cadastrar Funcionário')
        self.setFixedSize(900, 800)  # ✅ Aumentar altura
        
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
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #005f99;
            }
            QPushButton#CaptureButton {
                background-color: #FF9800;
                font-size: 16px;
                padding: 12px 30px;
            }
            QPushButton#CaptureButton:hover {
                background-color: #F57C00;
            }
            QPushButton#UploadButton {
                background-color: #9C27B0;
                font-size: 16px;
                padding: 12px 30px;
            }
            QPushButton#UploadButton:hover {
                background-color: #7B1FA2;
            }
            QPushButton#SaveButton {
                background-color: #4CAF50;
                font-size: 16px;
                padding: 12px 30px;
            }
            QPushButton#SaveButton:hover {
                background-color: #45a049;
            }
            QPushButton#CancelButton {
                background-color: #d32f2f;
            }
            QPushButton#CancelButton:hover {
                background-color: #b71c1c;
            }
            QLineEdit {
                background-color: #2d2d2d;
                border: 2px solid #007acc;
                border-radius: 6px;
                padding: 10px;
                color: #f0f0f0;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #4CAF50;
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
                color: #FFD700;
                padding: 5px;
            }
        """)
        
        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        
        # Título
        title = QLabel("📸 Cadastrar Novo Funcionário")
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # Status
        self.label_status = QLabel("📹 Câmera inicializada - Posicione-se ou faça upload")
        self.label_status.setObjectName("StatusLabel")
        self.label_status.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.label_status)
        
        # Layout horizontal: Vídeo + Formulário
        content_layout = QHBoxLayout()
        
        # Área de vídeo/foto
        video_layout = QVBoxLayout()
        
        self.label_video = QLabel()
        self.label_video.setObjectName("VideoLabel")
        self.label_video.setAlignment(Qt.AlignCenter)
        self.label_video.setFixedSize(480, 360)
        video_layout.addWidget(self.label_video)
        
        # ✅ BOTÕES DE CAPTURA/UPLOAD
        photo_buttons = QHBoxLayout()
        
        self.btn_capture = QPushButton("📸 Capturar")
        self.btn_capture.setObjectName("CaptureButton")
        self.btn_capture.clicked.connect(self.capture_photo)
        photo_buttons.addWidget(self.btn_capture)
        
        self.btn_upload = QPushButton("📁 Upload")
        self.btn_upload.setObjectName("UploadButton")
        self.btn_upload.clicked.connect(self.upload_photo)
        photo_buttons.addWidget(self.btn_upload)
        
        video_layout.addLayout(photo_buttons)
        
        content_layout.addLayout(video_layout)
        
        # Formulário
        form_layout = QFormLayout()
        form_layout.setSpacing(15)
        
        self.input_name = QLineEdit()
        self.input_name.setPlaceholderText("Digite o nome completo")
        form_layout.addRow("Nome:", self.input_name)
        
        self.input_role = QLineEdit()
        self.input_role.setPlaceholderText("Ex: Desenvolvedor, Gerente...")
        form_layout.addRow("Cargo:", self.input_role)
        
        self.input_department = QLineEdit()
        self.input_department.setPlaceholderText("Ex: TI, RH, Vendas...")
        form_layout.addRow("Departamento:", self.input_department)
        
        self.input_email = QLineEdit()
        self.input_email.setPlaceholderText("exemplo@empresa.com")
        form_layout.addRow("Email:", self.input_email)
        
        content_layout.addLayout(form_layout)
        
        main_layout.addLayout(content_layout)
        
        # Botões de ação
        buttons_layout = QHBoxLayout()
        
        self.btn_cancel = QPushButton("❌ Cancelar")
        self.btn_cancel.setObjectName("CancelButton")
        self.btn_cancel.clicked.connect(self.close)
        buttons_layout.addWidget(self.btn_cancel)
        
        buttons_layout.addStretch()
        
        self.btn_save = QPushButton("💾 Salvar Cadastro")
        self.btn_save.setObjectName("SaveButton")
        self.btn_save.clicked.connect(self.save_employee)
        self.btn_save.setEnabled(False)
        buttons_layout.addWidget(self.btn_save)
        
        main_layout.addLayout(buttons_layout)
        
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
            
        except Exception as e:
            QMessageBox.warning(
                self,
                "Aviso",
                f"Câmera não disponível.\n\n{str(e)}\n\nVocê pode fazer upload de uma foto."
            )
    
    def update_frame(self):
        """Atualizar frame da câmera COM VALIDAÇÃO DE QUALIDADE"""
        if not self.cap or not self.cap.isOpened():
            return
        
        ret, frame = self.cap.read()
        
        if not ret:
            return
        
        self.current_frame = frame.copy()
        
        # Validar qualidade
        valido, resultados, mensagem = FaceQualityValidator.validar_qualidade_completa(frame)
        
        # Desenhar feedback visual
        frame_feedback = FaceQualityValidator.desenhar_feedback_visual(
            frame, resultados, mensagem
        )
        
        # Atualizar status
        if valido:
            if len(resultados.get('warnings', [])) > 0:
                self.label_status.setText(f"{mensagem} - Pode capturar com avisos")
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
    
    def capture_photo(self):
        """Capturar foto da câmera"""
        if self.current_frame is None:
            QMessageBox.warning(self, "Aviso", "Nenhum frame disponível")
            return
        
        try:
            valido, resultados, mensagem = FaceQualityValidator.validar_qualidade_completa(
                self.current_frame
            )
            
            if not valido:
                QMessageBox.warning(
                    self,
                    "Rosto Não Detectado",
                    f"{mensagem}\n\nPosicione-se dentro do guia oval."
                )
                return
            
            warnings = resultados.get('warnings', [])
            if len(warnings) > 0:
                warning_text = "\n".join([f"• {w}" for w in warnings])
                
                reply = QMessageBox.question(
                    self,
                    "Avisos de Qualidade",
                    f"📊 Score: {resultados['score_geral']:.0%}\n\n"
                    f"Avisos detectados:\n\n{warning_text}\n\n"
                    f"Métricas:\n"
                    f"• Nitidez: {resultados['nitidez']:.0f}/500\n"
                    f"• Iluminação: {resultados['iluminacao']:.0%}\n"
                    f"• Tamanho: {resultados['tamanho_rosto']:.0%}\n\n"
                    "Deseja capturar mesmo assim?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.No:
                    return
            
            self.captured_photo = self.current_frame.copy()
            self.uploaded_photo = None  # Limpar upload
            
            if self.timer:
                self.timer.stop()
            
            frame_feedback = FaceQualityValidator.desenhar_feedback_visual(
                self.captured_photo, resultados, "✅ Foto capturada!"
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
            
            status_msg = f"✅ Foto capturada! Score: {resultados['score_geral']:.0%}"
            if len(warnings) > 0:
                status_msg += f" ({len(warnings)} aviso(s))"
            
            self.label_status.setText(status_msg)
            self.btn_capture.setText("🔄 Recapturar")
            self.btn_save.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Erro na Captura",
                f"Erro ao capturar foto.\n\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    # ✅ NOVO MÉTODO: UPLOAD DE FOTO
    def upload_photo(self):
        """Fazer upload de foto do computador"""
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
            
            # Validar qualidade
            valido, resultados, mensagem = FaceQualityValidator.validar_qualidade_completa(image)
            
            if not valido:
                QMessageBox.warning(
                    self,
                    "Rosto Não Detectado",
                    f"{mensagem}\n\nSelecione uma foto com rosto visível."
                )
                return
            
            # Avisos de qualidade
            warnings = resultados.get('warnings', [])
            if len(warnings) > 0:
                warning_text = "\n".join([f"• {w}" for w in warnings])
                
                reply = QMessageBox.question(
                    self,
                    "Avisos de Qualidade",
                    f"📊 Score: {resultados['score_geral']:.0%}\n\n"
                    f"Avisos:\n{warning_text}\n\n"
                    "Usar esta foto mesmo assim?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.No:
                    return
            
            # Salvar foto
            self.uploaded_photo = image.copy()
            self.captured_photo = None  # Limpar captura
            
            # Parar câmera
            if self.timer:
                self.timer.stop()
            
            # Mostrar foto com feedback
            frame_feedback = FaceQualityValidator.desenhar_feedback_visual(
                self.uploaded_photo, resultados, "✅ Foto carregada!"
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
            
            self.label_status.setText(f"✅ Foto carregada! Score: {resultados['score_geral']:.0%}")
            self.label_status.setStyleSheet("color: #4CAF50;")
            
            self.btn_save.setEnabled(True)
            self.btn_upload.setText("🔄 Outro Upload")
            
            print(f"✅ Foto carregada: {os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Erro",
                f"Erro ao carregar foto.\n\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
            
    def save_employee(self):
        """Salvar funcionário"""
        # ✅ CORRIGIR NOMES DOS CAMPOS
        name = self.input_name.text().strip()
        role = self.input_role.text().strip()
        department = self.input_department.text().strip()
        email = self.input_email.text().strip()
        
        if not name:
            QMessageBox.warning(self, "Campo Obrigatório", "O campo Nome é obrigatório.")
            return
        
        # ✅ USAR FOTO CAPTURADA OU UPLOADED
        photo_to_use = self.captured_photo if self.captured_photo is not None else self.uploaded_photo
        
        if photo_to_use is None:
            QMessageBox.warning(
                self,
                "Foto Obrigatória",
                "Capture uma foto ou faça upload antes de cadastrar."
            )
            return
        
        # ✅ DEBUG: Verificar se a foto existe
        print("\n" + "="*60)
        print("🔍 DEBUG - ANTES DE SALVAR NO BANCO:")
        print("="*60)
        print(f"Nome: {name}")
        print(f"Cargo: {role}")
        print(f"Departamento: {department}")
        print(f"Email: {email}")
        print(f"captured_photo is None: {self.captured_photo is None}")
        print(f"uploaded_photo is None: {self.uploaded_photo is None}")
        print(f"photo_to_use is None: {photo_to_use is None}")
        if photo_to_use is not None:
            print(f"Shape da imagem: {photo_to_use.shape}")
        print("="*60 + "\n")
        
        try:
            self.label_status.setText("🔄 Gerando embedding facial...")
            
            # ✅ Gerar embedding
            embedding, facial_area, confidence = FaceRecognitionSystem.gerar_embedding(
                photo_to_use,
                model_name='Facenet512'
            )
            
            if embedding is None:
                raise Exception("Falha ao gerar embedding")
            
            print(f"✅ Embedding gerado: {len(embedding)} dimensões")
            print(f"   Confiança: {confidence:.2%}")
            print(f"   Área facial: {facial_area}")
            
            # ✅ Preparar dados COM A FOTO
            employee_data = {
                'name': name,
                'role': role,
                'department': department,
                'email': email,
                'photo': photo_to_use,  # ✅ ADICIONAR FOTO AQUI
                'embedding': embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                'facial_area': facial_area,
                'face_confidence': confidence
            }
            
            # ✅ DEBUG: Verificar dicionário
            print("\n📦 Dicionário employee_data:")
            print(f"   name: {employee_data['name']}")
            print(f"   photo is None: {employee_data['photo'] is None}")
            print(f"   photo type: {type(employee_data['photo'])}")
            if employee_data['photo'] is not None:
                print(f"   photo shape: {employee_data['photo'].shape}")
            print(f"   embedding size: {len(employee_data['embedding'])}")
            print()
            
            # Salvar no banco
            employee_id = save_employee_data(employee_data)
            
            if employee_id:
                QMessageBox.information(
                    self,
                    "✅ Sucesso",
                    f"Funcionário '{name}' cadastrado!\n\n"
                    f"📊 Detalhes:\n"
                    f"• ID: {employee_id}\n"
                    f"• Embedding: {len(embedding)} dimensões\n"
                    f"• Confiança: {confidence:.1%}\n"
                    f"• Modelo: Facenet512"
                )
                self.close()
            else:
                QMessageBox.critical(
                    self,
                    "❌ Erro",
                    "Não foi possível salvar o cadastro."
                )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Erro ao Salvar",
                f"Ocorreu um erro ao salvar o cadastro.\n\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def closeEvent(self, event):
        """Liberar recursos"""
        if self.timer:
            self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.closed.emit()
        event.accept()
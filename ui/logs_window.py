import sys
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QComboBox, QMessageBox, QTextEdit,
                             QDialog, QFormLayout, QGroupBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from datetime import datetime
import json

# Adicionar caminho
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqlite_db import get_all_employees, get_employee_emotion_history, get_recent_emotion_logs


class EmotionLogDetailDialog(QDialog):
    """Janela de detalhes de um log específico"""
    
    def __init__(self, log_data, employee_name, parent=None):
        super().__init__(parent)
        self.log_data = log_data
        self.employee_name = employee_name
        self.init_ui()
        
    def init_ui(self):
        """Inicializar interface de detalhes"""
        self.setWindowTitle(f'Detalhes da Análise - {self.employee_name}')
        self.setFixedSize(700, 600)
        
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
                font-size: 20px;
                font-weight: bold;
                color: #4CAF50;
                padding: 10px;
            }
            QTextEdit {
                background-color: #2d2d2d;
                color: #f0f0f0;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 8px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 25px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QGroupBox {
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
                background-color: #252525;
            }
            QGroupBox::title {
                color: #FFD700;
                font-weight: bold;
                font-size: 14px;
                padding: 0 10px;
            }
        """)
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        
        # Título
        emotion = self.log_data.get('dominant_emotion', 'N/A')
        emotion_emojis = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'surprise': '😲',
            'fear': '😨',
            'disgust': '🤢',
            'neutral': '😐'
        }
        emoji = emotion_emojis.get(emotion.lower(), '❓')
        
        title = QLabel(f"{emoji} Análise Emocional - {self.employee_name}")
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # === INFORMAÇÕES PRINCIPAIS ===
        info_group = QGroupBox("📊 Informações da Análise")
        info_layout = QFormLayout()
        info_layout.setSpacing(10)
        
        # Emoção Dominante
        emotion_label = QLabel(f"{emoji} {emotion.upper()}")
        emotion_label.setStyleSheet("color: #4CAF50; font-size: 16px; font-weight: bold;")
        info_layout.addRow("🎯 Emoção Dominante:", emotion_label)
        
        # Confiança
        confidence = self.log_data.get('confidence', 0)
        confidence_label = QLabel(f"{confidence:.2f}%")
        confidence_label.setStyleSheet(f"color: {'#4CAF50' if confidence > 70 else '#FF9800'}; font-weight: bold;")
        info_layout.addRow("📈 Confiança:", confidence_label)
        
        # Timestamp
        timestamp = self.log_data.get('timestamp', 'N/A')
        info_layout.addRow("🕒 Data/Hora:", QLabel(timestamp))
        
        # Duração
        duration = self.log_data.get('analysis_duration', 0)
        info_layout.addRow("⏱️ Duração:", QLabel(f"{duration}s"))
        
        # Amostras
        samples = self.log_data.get('samples_collected', 0)
        info_layout.addRow("📸 Amostras:", QLabel(str(samples)))
        
        info_group.setLayout(info_layout)
        main_layout.addWidget(info_group)
        
        # === TODAS AS EMOÇÕES ===
        emotions_group = QGroupBox("📈 Distribuição de Todas as Emoções")
        emotions_layout = QVBoxLayout()
        
        all_emotions = self.log_data.get('all_emotions', {})
        
        if all_emotions:
            # Criar texto formatado
            emotions_text = QTextEdit()
            emotions_text.setReadOnly(True)
            emotions_text.setMaximumHeight(200)
            
            # Ordenar por valor (maior para menor)
            sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)
            
            content = ""
            for i, (emo, score) in enumerate(sorted_emotions, 1):
                emo_emoji = emotion_emojis.get(emo.lower(), '❓')
                bar_length = int(score / 2)  # Barra visual (0-50 chars)
                bar = "█" * bar_length
                
                content += f"{i}. {emo_emoji} {emo.upper():<12} {score:>6.2f}%  {bar}\n"
            
            emotions_text.setText(content)
            emotions_layout.addWidget(emotions_text)
        else:
            no_data = QLabel("⚠️ Dados de emoções não disponíveis")
            no_data.setStyleSheet("color: #FF9800; font-style: italic;")
            emotions_layout.addWidget(no_data)
        
        emotions_group.setLayout(emotions_layout)
        main_layout.addWidget(emotions_group)
        
        # === DADOS TÉCNICOS (JSON) ===
        json_group = QGroupBox("🔧 Dados Técnicos (JSON)")
        json_layout = QVBoxLayout()
        
        json_text = QTextEdit()
        json_text.setReadOnly(True)
        json_text.setMaximumHeight(150)
        
        try:
            json_formatted = json.dumps(self.log_data, indent=2, ensure_ascii=False)
        except:
            json_formatted = str(self.log_data)
        
        json_text.setText(json_formatted)
        json_layout.addWidget(json_text)
        json_group.setLayout(json_layout)
        main_layout.addWidget(json_group)
        
        # Botão Fechar
        btn_close = QPushButton("✅ Fechar")
        btn_close.clicked.connect(self.accept)
        main_layout.addWidget(btn_close)
        
        self.setLayout(main_layout)


class LogsWindow(QWidget):
    """Janela de visualização de logs de análises emocionais"""
    
    closed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.employees = []
        self.current_employee_id = None
        self.current_logs = []
        
        self.init_ui()
        self.load_employees()
    
    def init_ui(self):
        """Inicializar interface"""
        self.setWindowTitle('EmpathIA - Histórico de Análises Emocionais')
        self.setFixedSize(1200, 700)
        
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
            QPushButton#BackButton {
                background-color: #666;
            }
            QPushButton#BackButton:hover {
                background-color: #555;
            }
            QPushButton#RefreshButton {
                background-color: #4CAF50;
            }
            QPushButton#RefreshButton:hover {
                background-color: #45a049;
            }
            QComboBox {
                background-color: #2d2d2d;
                color: #f0f0f0;
                border: 2px solid #007acc;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
                min-width: 300px;
            }
            QComboBox:hover {
                border-color: #4CAF50;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #f0f0f0;
                margin-right: 10px;
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
            QLabel#TitleLabel {
                font-size: 24px;
                font-weight: bold;
                color: #4CAF50;
            }
            QLabel#StatsLabel {
                font-size: 14px;
                color: #FFD700;
                padding: 5px;
            }
        """)
        
        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        
        # Título
        title = QLabel("📊 Histórico de Análises Emocionais")
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # Seletor de funcionário
        selector_layout = QHBoxLayout()
        
        selector_label = QLabel("👤 Selecione o Funcionário:")
        selector_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        selector_layout.addWidget(selector_label)
        
        self.combo_employees = QComboBox()
        self.combo_employees.currentIndexChanged.connect(self.on_employee_changed)
        selector_layout.addWidget(self.combo_employees)
        
        selector_layout.addStretch()
        
        main_layout.addLayout(selector_layout)
        
        # Estatísticas
        self.label_stats = QLabel("📈 Selecione um funcionário para ver estatísticas")
        self.label_stats.setObjectName("StatsLabel")
        self.label_stats.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.label_stats)
        
        # Tabela de logs
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            'Data/Hora', 'Emoção', 'Confiança (%)', 'Duração (s)', 'Amostras', 'ID'
        ])
        
        # Configurar largura das colunas
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        
        # Duplo clique para ver detalhes
        self.table.cellDoubleClicked.connect(self.show_log_details)
        
        main_layout.addWidget(self.table)
        
        # Dica
        tip_label = QLabel("💡 Dica: Clique duplo em qualquer linha para ver detalhes completos da análise")
        tip_label.setStyleSheet("color: #FFD700; font-style: italic; font-size: 12px;")
        tip_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(tip_label)
        
        # Botões
        buttons_layout = QHBoxLayout()
        
        self.btn_refresh = QPushButton("🔄 Atualizar")
        self.btn_refresh.setObjectName("RefreshButton")
        self.btn_refresh.clicked.connect(self.refresh_logs)
        buttons_layout.addWidget(self.btn_refresh)
        
        self.btn_details = QPushButton("👁️ Ver Detalhes")
        self.btn_details.clicked.connect(self.show_log_details)
        buttons_layout.addWidget(self.btn_details)
        
        buttons_layout.addStretch()
        
        self.btn_back = QPushButton("⬅️ Voltar")
        self.btn_back.setObjectName("BackButton")
        self.btn_back.clicked.connect(self.close)
        buttons_layout.addWidget(self.btn_back)
        
        main_layout.addLayout(buttons_layout)
        
        self.setLayout(main_layout)
    
    def load_employees(self):
        """Carregar lista de funcionários"""
        try:
            self.employees = get_all_employees()
            
            self.combo_employees.clear()
            self.combo_employees.addItem("-- Selecione um funcionário --", None)
            
            for emp in self.employees:
                name = emp.get('name', 'N/A')
                role = emp.get('role', 'N/A')
                emp_id = emp.get('_id')
                
                display_text = f"{name} - {role}"
                self.combo_employees.addItem(display_text, emp_id)
            
            print(f"✅ {len(self.employees)} funcionários carregados")
            
        except Exception as e:
            QMessageBox.warning(
                self,
                "Erro",
                f"Não foi possível carregar funcionários.\n\n{str(e)}"
            )
    
    def on_employee_changed(self, index):
        """Quando funcionário é selecionado"""
        if index <= 0:
            self.current_employee_id = None
            self.table.setRowCount(0)
            self.label_stats.setText("📈 Selecione um funcionário para ver estatísticas")
            return
        
        employee_id = self.combo_employees.currentData()
        self.current_employee_id = employee_id
        
        self.load_employee_logs(employee_id)
    
    def load_employee_logs(self, employee_id):
        """Carregar logs de um funcionário específico"""
        try:
            self.current_logs = get_employee_emotion_history(employee_id)
            
            # Limpar tabela
            self.table.setRowCount(0)
            
            if not self.current_logs:
                self.label_stats.setText("⚠️ Nenhuma análise encontrada para este funcionário")
                return
            
            # Preencher tabela
            emotion_emojis = {
                'happy': '😊',
                'sad': '😢',
                'angry': '😠',
                'surprise': '😲',
                'fear': '😨',
                'disgust': '🤢',
                'neutral': '😐'
            }
            
            # Estatísticas
            emotions_count = {}
            total_confidence = 0
            
            for log in self.current_logs:
                row = self.table.rowCount()
                self.table.insertRow(row)
                
                # Dados
                timestamp = log.get('timestamp', 'N/A')
                emotion = log.get('dominant_emotion', 'N/A')
                confidence = log.get('confidence', 0)
                duration = log.get('analysis_duration', 0)
                samples = log.get('samples_collected', 0)
                log_id = log.get('_id', 'N/A')
                
                # Estatísticas
                emotions_count[emotion] = emotions_count.get(emotion, 0) + 1
                total_confidence += confidence
                
                # Preencher linha
                emoji = emotion_emojis.get(emotion.lower(), '❓')
                
                self.table.setItem(row, 0, QTableWidgetItem(timestamp))
                self.table.setItem(row, 1, QTableWidgetItem(f"{emoji} {emotion.upper()}"))
                self.table.setItem(row, 2, QTableWidgetItem(f"{confidence:.1f}"))
                self.table.setItem(row, 3, QTableWidgetItem(str(duration)))
                self.table.setItem(row, 4, QTableWidgetItem(str(samples)))
                self.table.setItem(row, 5, QTableWidgetItem(str(log_id)))
            
            # Calcular estatísticas
            total_logs = len(self.current_logs)
            avg_confidence = total_confidence / total_logs if total_logs > 0 else 0
            most_common = max(emotions_count, key=emotions_count.get) if emotions_count else 'N/A'
            most_common_emoji = emotion_emojis.get(most_common.lower(), '❓')
            
            stats_text = (
                f"📊 Total de análises: {total_logs} | "
                f"📈 Confiança média: {avg_confidence:.1f}% | "
                f"🏆 Emoção predominante: {most_common_emoji} {most_common.upper()} "
                f"({emotions_count.get(most_common, 0)}x)"
            )
            
            self.label_stats.setText(stats_text)
            
            print(f"✅ {total_logs} logs carregados para funcionário ID {employee_id}")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Erro",
                f"Erro ao carregar logs.\n\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def refresh_logs(self):
        """Atualizar logs do funcionário atual"""
        if self.current_employee_id:
            self.load_employee_logs(self.current_employee_id)
        else:
            QMessageBox.information(
                self,
                "Aviso",
                "Selecione um funcionário primeiro."
            )
    
    def show_log_details(self):
        """Mostrar detalhes de um log específico"""
        selected_row = self.table.currentRow()
        
        if selected_row < 0:
            QMessageBox.warning(
                self,
                "Aviso",
                "Selecione uma análise na tabela para ver os detalhes."
            )
            return
        
        if not self.current_logs or selected_row >= len(self.current_logs):
            QMessageBox.warning(
                self,
                "Erro",
                "Não foi possível carregar os dados da análise."
            )
            return
        
        # Pegar dados do log
        log_data = self.current_logs[selected_row]
        
        # Pegar nome do funcionário
        employee_name = self.combo_employees.currentText().split(' - ')[0]
        
        # Abrir janela de detalhes
        detail_dialog = EmotionLogDetailDialog(log_data, employee_name, self)
        detail_dialog.exec_()
    
    def closeEvent(self, event):
        """Emitir sinal ao fechar"""
        self.closed.emit()
        event.accept()
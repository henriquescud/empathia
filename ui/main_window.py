import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from ui.logs_window import LogsWindow

class MainWindow(QWidget):
    """Janela principal do EmpathIA"""
    
    def __init__(self):
        super().__init__()
        self.register_window = None
        self.analysis_window = None
        self.logs_window = None  # ✅ ADICIONAR
        self.init_ui()
    
    def init_ui(self):
        """Inicializar interface"""
        self.setWindowTitle('EmpathIA - Sistema de Análise Emocional')
        self.setFixedSize(600, 550)  # ✅ Aumentar altura para novo botão
        
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #f0f0f0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #007acc;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
                border: none;
                border-radius: 8px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #005f99;
            }
            QPushButton#RegisterButton {
                background-color: #4CAF50;
            }
            QPushButton#RegisterButton:hover {
                background-color: #45a049;
            }
            QPushButton#AnalysisButton {
                background-color: #2196F3;
            }
            QPushButton#AnalysisButton:hover {
                background-color: #1976D2;
            }
            QPushButton#LogsButton {
                background-color: #FF9800;
            }
            QPushButton#LogsButton:hover {
                background-color: #F57C00;
            }
            QPushButton#ExitButton {
                background-color: #d32f2f;
            }
            QPushButton#ExitButton:hover {
                background-color: #b71c1c;
            }
            QLabel#TitleLabel {
                font-size: 32px;
                font-weight: bold;
                color: #4CAF50;
            }
            QLabel#SubtitleLabel {
                font-size: 14px;
                color: #b0b0b0;
                font-style: italic;
            }
        """)
        
        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(40, 40, 40, 40)
        
        # Logo/Título
        title_layout = QVBoxLayout()
        title_layout.setSpacing(5)
        
        title = QLabel('🧠 EmpathIA')
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title)
        
        subtitle = QLabel('Sistema Inteligente de Análise Emocional')
        subtitle.setObjectName("SubtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(subtitle)
        
        main_layout.addLayout(title_layout)
        main_layout.addSpacing(20)
        
        # Botões de funcionalidades
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(10)
        
        # Botão Gerenciar Funcionários
        self.btn_register = QPushButton('👥 Gerenciar Funcionários')
        self.btn_register.setObjectName("RegisterButton")
        self.btn_register.clicked.connect(self.show_register)
        buttons_layout.addWidget(self.btn_register)
        
        # Botão Análise Emocional
        self.btn_analysis = QPushButton('🧠 Iniciar Análise Emocional')
        self.btn_analysis.setObjectName("AnalysisButton")
        self.btn_analysis.clicked.connect(self.show_analysis)
        buttons_layout.addWidget(self.btn_analysis)
        
        # ✅ BOTÃO HISTÓRICO DE ANÁLISES
        self.btn_logs = QPushButton('📊 Histórico de Análises')
        self.btn_logs.setObjectName("LogsButton")
        self.btn_logs.clicked.connect(self.show_logs)
        buttons_layout.addWidget(self.btn_logs)
        
        # Botão Sair
        self.btn_exit = QPushButton('❌ Sair')
        self.btn_exit.setObjectName("ExitButton")
        self.btn_exit.clicked.connect(self.close)
        buttons_layout.addWidget(self.btn_exit)
        
        main_layout.addLayout(buttons_layout)
        
        # Rodapé
        footer = QLabel('Versão 1.0 | Desenvolvido com PyQt5 e DeepFace')
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #666; font-size: 11px;")
        main_layout.addWidget(footer)
        
        self.setLayout(main_layout)
    
    def show_register(self):
        """Abrir janela de gerenciamento de funcionários"""
        from ui.register_window import RegisterWindow
        
        self.register_window = RegisterWindow()
        self.register_window.show()
        self.register_window.closed.connect(self.on_register_closed)
        self.hide()
    
    def on_register_closed(self):
        """Quando fechar janela de registro"""
        self.show()
        self.register_window = None
    
    def show_analysis(self):
        """Abrir janela de análise emocional"""
        from ui.analysis_window import AnalysisWindow
        
        self.analysis_window = AnalysisWindow()
        self.analysis_window.show()
        self.analysis_window.closed.connect(self.on_analysis_closed)
        self.hide()
    
    def on_analysis_closed(self):
        """Quando fechar janela de análise"""
        self.show()
        self.analysis_window = None
    
    # ✅ ADICIONAR MÉTODOS PARA LOGS
    def show_logs(self):
        self.logs_window = LogsWindow()
        self.logs_window.show()
        self.logs_window.closed.connect(self.on_logs_closed)
        self.hide()
    
    def on_logs_closed(self):
        """Quando fechar janela de logs"""
        self.show()
        self.logs_window = None
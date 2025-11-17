import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    """Ponto de entrada da aplicação - Interface gráfica pura"""
    app = QApplication(sys.argv)
    
    # Configurar estilo global da aplicação
    app.setStyle('Fusion')
    
    # Criar e exibir janela principal
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
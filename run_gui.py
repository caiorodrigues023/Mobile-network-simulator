"""
run_gui.py — Ponto de Entrada da Interface Gráfica Executável do Simulador de Redes Móveis.
"""

import os
import sys

# Garante que a raiz do repositório está no sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from src.simulator.gui import SimulatorMainWindow


def main():
    # Configurações de High-DPI para monitores modernos
    if hasattr(Qt.ApplicationAttribute, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Mobile Network Simulator 3D")
    app.setOrganizationName("MobileNetworkLab")

    window = SimulatorMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

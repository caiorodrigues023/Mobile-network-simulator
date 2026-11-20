# ---------------------------------------------------------------------------
# src/simulator/gui/styles.py — Folha de Estilos QSS e Temas da Interface
# ---------------------------------------------------------------------------

MODERN_DARK_THEME = """
/* Reset e Janela Principal */
QMainWindow, QDialog {
    background-color: #0f131a;
    color: #e2e8f0;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
}

QWidget {
    color: #e2e8f0;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
}

/* Barra de Menus */
QMenuBar {
    background-color: #161b26;
    color: #cbd5e1;
    border-bottom: 1px solid #232c3d;
    padding: 2px 4px;
}

QMenuBar::item {
    background: transparent;
    padding: 6px 10px;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #232c3d;
    color: #38bdf8;
}

QMenu {
    background-color: #161b26;
    color: #e2e8f0;
    border: 1px solid #2d3748;
    border-radius: 6px;
    padding: 4px;
}

QMenu::item {
    padding: 6px 24px 6px 12px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #0284c7;
    color: #ffffff;
}

/* Painel de Rolagem Lateral (Sidebar) */
QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    border: none;
    background: #111622;
    width: 8px;
    margin: 0px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #2d3748;
    min-height: 24px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #38bdf8;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* Caixas de Agrupamento (GroupBox) */
QGroupBox {
    font-weight: bold;
    border: 1px solid #232c3d;
    border-radius: 8px;
    margin-top: 18px;
    padding: 14px 10px 10px 10px;
    background-color: #141923;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #38bdf8;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Rótulos e Texto */
QLabel {
    color: #94a3b8;
    font-size: 12px;
}

QLabel[heading="true"] {
    color: #f8fafc;
    font-size: 14px;
    font-weight: bold;
}

/* Inputs de Texto, SpinBoxes e Combos */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #1a2232;
    color: #f1f5f9;
    border: 1px solid #2e3a4e;
    border-radius: 6px;
    padding: 5px 8px;
    font-size: 12px;
    selection-background-color: #0284c7;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #38bdf8;
    background-color: #1e283b;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid #2e3a4e;
}

QComboBox QAbstractItemView {
    background-color: #161b26;
    color: #e2e8f0;
    selection-background-color: #0284c7;
    border: 1px solid #2d3748;
    padding: 4px;
}

/* Sliders */
QSlider::groove:horizontal {
    border: none;
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #0ea5e9;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #f8fafc;
    border: 2px solid #0ea5e9;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #38bdf8;
    border-color: #ffffff;
}

/* Checkboxes */
QCheckBox {
    color: #cbd5e1;
    spacing: 8px;
    font-size: 12px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #334155;
    background-color: #1a2232;
}

QCheckBox::indicator:checked {
    background-color: #0ea5e9;
    border-color: #38bdf8;
}

/* Botões */
QPushButton {
    background-color: #1e293b;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 8px 14px;
    font-weight: 600;
    font-size: 12px;
}

QPushButton:hover {
    background-color: #2e3a4e;
    border-color: #475569;
}

QPushButton:pressed {
    background-color: #0f172a;
}

QPushButton[primary="true"] {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #0ea5e9);
    border: 1px solid #38bdf8;
    color: #ffffff;
    font-size: 13px;
    padding: 10px 16px;
}

QPushButton[primary="true"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0369a1, stop:1 #0284c7);
    border-color: #7dd3fc;
}

QPushButton[accent="true"] {
    background-color: #10b981;
    border: 1px solid #34d399;
    color: #ffffff;
}

QPushButton[accent="true"]:hover {
    background-color: #059669;
}

/* Abas (TabWidget) */
QTabWidget::pane {
    border: 1px solid #232c3d;
    background-color: #111622;
    border-radius: 8px;
    top: -1px;
}

QTabBar::tab {
    background-color: #161b26;
    color: #94a3b8;
    border: 1px solid #232c3d;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 10px 18px;
    margin-right: 4px;
    font-weight: 600;
    font-size: 12px;
}

QTabBar::tab:selected {
    background-color: #111622;
    color: #38bdf8;
    border-top: 2px solid #38bdf8;
}

QTabBar::tab:hover:!selected {
    background-color: #1c2331;
    color: #e2e8f0;
}

/* Barra de Status */
QStatusBar {
    background-color: #0b0e14;
    color: #94a3b8;
    border-top: 1px solid #1e2638;
    font-size: 11px;
    padding: 2px 6px;
}

/* Barra de Progresso */
QProgressBar {
    background-color: #1a2232;
    border: 1px solid #2e3a4e;
    border-radius: 4px;
    text-align: center;
    color: #f8fafc;
    font-weight: bold;
    height: 14px;
    font-size: 10px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #10b981);
    border-radius: 3px;
}

/* Tabelas */
QTableWidget {
    background-color: #131822;
    border: 1px solid #232c3d;
    gridline-color: #1e2638;
    color: #e2e8f0;
    border-radius: 6px;
    font-size: 12px;
}

QTableWidget::item {
    padding: 6px;
}

QTableWidget::item:selected {
    background-color: #0369a1;
    color: #ffffff;
}

QHeaderView::section {
    background-color: #18202e;
    color: #94a3b8;
    padding: 8px;
    border: none;
    border-right: 1px solid #232c3d;
    border-bottom: 1px solid #232c3d;
    font-weight: bold;
    font-size: 11px;
    text-transform: uppercase;
}

/* Divisores */
QSplitter::handle {
    background-color: #1a2232;
}

QSplitter::handle:horizontal {
    width: 4px;
}

QSplitter::handle:vertical {
    height: 4px;
}
"""

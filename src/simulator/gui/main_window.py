# ---------------------------------------------------------------------------
# src/simulator/gui/main_window.py — Janela Principal da Interface Gráfica
# ---------------------------------------------------------------------------

import time
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSlider,
    QSpinBox, QSplitter, QStatusBar, QTabWidget, QVBoxLayout, QWidget,
)
from PyQt6.QtWebEngineWidgets import QWebEngineView

from .kpi_view import KPIViewGenerator
from .mobility_view import MobilityView
from .styles import MODERN_DARK_THEME
from .users_table import UsersTableView
from .view3d import View3DGenerator
from .worker import MonteCarloWorker, SnapshotCalculator


class SimulatorMainWindow(QMainWindow):
    """Janela Principal do Simulador de Redes Móveis com Visão 3D e KPIs."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mobile Network Simulator 3D — Simulador de Redes Móveis")
        self.resize(1420, 900)
        self.setMinimumSize(1100, 700)

        # Estado da aplicação
        self._current_snapshot = None
        self._latest_simulation = None
        self._worker = None

        # Configura estilo
        self.setStyleSheet(MODERN_DARK_THEME)

        # Inicializa UI
        self._setup_ui()
        self._setup_statusbar()

        # Dispara o cálculo do primeiro cenário após inicializar
        QTimer.singleShot(150, self.update_3d_snapshot)

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # -------------------------------------------------------------------
        # Painel Lateral Esquerdo (Controles Completos de Simulação)
        # -------------------------------------------------------------------
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFixedWidth(380)

        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(10)

        # 1. Topologia da Rede e Alocação
        grp_topo = QGroupBox("📐 Topologia & Alocação")
        lay_topo = QGridLayout(grp_topo)

        lay_topo.addWidget(QLabel("APs (M):"), 0, 0)
        self.combo_aps = QComboBox()
        for m_val, g_lbl in ((4, "4 (2×2)"), (9, "9 (3×3)"), (16, "16 (4×4)"), (25, "25 (5×5)"), (36, "36 (6×6)")):
            self.combo_aps.addItem(g_lbl, m_val)
        self.combo_aps.currentIndexChanged.connect(self._on_param_changed)
        lay_topo.addWidget(self.combo_aps, 0, 1)

        lay_topo.addWidget(QLabel("Usuários (K):"), 1, 0)
        self.spin_ues = QSpinBox()
        self.spin_ues.setRange(1, 128)
        self.spin_ues.setValue(8)
        self.spin_ues.valueChanged.connect(self._on_param_changed)
        lay_topo.addWidget(self.spin_ues, 1, 1)

        lay_topo.addWidget(QLabel("Canais (N):"), 2, 0)
        self.spin_channels = QSpinBox()
        self.spin_channels.setRange(1, 16)
        self.spin_channels.setValue(1)
        self.spin_channels.valueChanged.connect(self._on_param_changed)
        lay_topo.addWidget(self.spin_channels, 2, 1)

        lay_topo.addWidget(QLabel("Alocação:"), 3, 0)
        self.combo_alloc = QComboBox()
        self.combo_alloc.addItem("Ortogonal intra-AP", "ortogonal")
        self.combo_alloc.addItem("Aleatória (Random)", "random")
        self.combo_alloc.currentIndexChanged.connect(self._on_param_changed)
        lay_topo.addWidget(self.combo_alloc, 3, 1)

        lay_topo.addWidget(QLabel("Altura AP / UE (m):"), 4, 0)
        h_box = QHBoxLayout()
        self.spin_h_ap = QDoubleSpinBox()
        self.spin_h_ap.setRange(5.0, 100.0)
        self.spin_h_ap.setValue(25.0)
        self.spin_h_ap.setSingleStep(5.0)
        self.spin_h_ap.valueChanged.connect(self._on_param_changed)
        self.spin_h_ue = QDoubleSpinBox()
        self.spin_h_ue.setRange(0.5, 20.0)
        self.spin_h_ue.setValue(1.5)
        self.spin_h_ue.setSingleStep(0.5)
        self.spin_h_ue.valueChanged.connect(self._on_param_changed)
        h_box.addWidget(self.spin_h_ap)
        h_box.addWidget(self.spin_h_ue)
        lay_topo.addLayout(h_box, 4, 1)

        lay_topo.addWidget(QLabel("Seed (-1 = aleatório):"), 5, 0)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(-1, 999999)
        self.spin_seed.setValue(-1)
        self.spin_seed.valueChanged.connect(self._on_param_changed)
        lay_topo.addWidget(self.spin_seed, 5, 1)

        sidebar_layout.addWidget(grp_topo)

        # 2. Antenas e Feixes
        grp_ant = QGroupBox("📡 Antenas & Beamforming")
        lay_ant = QGridLayout(grp_ant)

        lay_ant.addWidget(QLabel("Modelo:"), 0, 0)
        self.combo_ant_model = QComboBox()
        self.combo_ant_model.addItem("SMB (Switched Multi-Beam)", "smb")
        self.combo_ant_model.addItem("Setorizado", "sector")
        self.combo_ant_model.addItem("Omnidirecional", "omni")
        self.combo_ant_model.addItem("Adaptativo MRC (Ideal 0°)", "adaptive")
        self.combo_ant_model.addItem("Adaptativo MRC (Erro 1°)", "adaptive_1")
        self.combo_ant_model.addItem("Adaptativo MRC (Erro 5°)", "adaptive_5")
        self.combo_ant_model.addItem("Adaptativo MRC (Erro 10°)", "adaptive_10")
        self.combo_ant_model.currentIndexChanged.connect(self._on_param_changed)
        lay_ant.addWidget(self.combo_ant_model, 0, 1)

        lay_ant.addWidget(QLabel("Elementos (L) / Feixes (J):"), 1, 0)
        lj_box = QHBoxLayout()
        self.combo_l = QComboBox()
        for l_val in (2, 4, 8, 16, 32, 64):
            self.combo_l.addItem(f"L={l_val}", l_val)
        self.combo_l.setCurrentIndex(2)  # L=8
        self.combo_l.currentIndexChanged.connect(self._on_param_changed)
        self.combo_j = QComboBox()
        for j_val in (4, 8, 12, 16, 24, 32):
            self.combo_j.addItem(f"J={j_val}", j_val)
        self.combo_j.setCurrentIndex(1)  # J=8
        self.combo_j.currentIndexChanged.connect(self._on_param_changed)
        lj_box.addWidget(self.combo_l)
        lj_box.addWidget(self.combo_j)
        lay_ant.addLayout(lj_box, 1, 1)

        lay_ant.addWidget(QLabel("Erro AoA (°) / Abertura (°):"), 2, 0)
        ang_box = QHBoxLayout()
        self.spin_aoa_err = QDoubleSpinBox()
        self.spin_aoa_err.setRange(0.0, 30.0)
        self.spin_aoa_err.setValue(0.0)
        self.spin_aoa_err.setSingleStep(1.0)
        self.spin_aoa_err.valueChanged.connect(self._on_param_changed)
        self.spin_beamwidth = QDoubleSpinBox()
        self.spin_beamwidth.setRange(30.0, 360.0)
        self.spin_beamwidth.setValue(120.0)
        self.spin_beamwidth.setSingleStep(15.0)
        self.spin_beamwidth.valueChanged.connect(self._on_param_changed)
        ang_box.addWidget(self.spin_aoa_err)
        ang_box.addWidget(self.spin_beamwidth)
        lay_ant.addLayout(ang_box, 2, 1)

        lay_ant.addWidget(QLabel("Lóbulo Traseiro (g_min):"), 3, 0)
        self.spin_g_min = QDoubleSpinBox()
        self.spin_g_min.setRange(0.001, 0.5)
        self.spin_g_min.setDecimals(3)
        self.spin_g_min.setSingleStep(0.005)
        self.spin_g_min.setValue(0.01)
        self.spin_g_min.valueChanged.connect(self._on_param_changed)
        lay_ant.addWidget(self.spin_g_min, 3, 1)

        sidebar_layout.addWidget(grp_ant)

        # 3. Canal de Propagação & Ruído
        grp_chan = QGroupBox("🌊 Canal & Propagação")
        lay_chan = QGridLayout(grp_chan)

        lay_chan.addWidget(QLabel("Banda Total (MHz):"), 0, 0)
        self.spin_bw = QDoubleSpinBox()
        self.spin_bw.setRange(1.0, 1000.0)
        self.spin_bw.setValue(100.0)
        self.spin_bw.setSingleStep(10.0)
        self.spin_bw.valueChanged.connect(self._on_param_changed)
        lay_chan.addWidget(self.spin_bw, 0, 1)

        lay_chan.addWidget(QLabel("Pathloss (n) / d₀ (m):"), 1, 0)
        pl_box = QHBoxLayout()
        self.spin_pathloss = QDoubleSpinBox()
        self.spin_pathloss.setRange(2.0, 6.0)
        self.spin_pathloss.setDecimals(2)
        self.spin_pathloss.setSingleStep(0.2)
        self.spin_pathloss.setValue(4.0)
        self.spin_pathloss.valueChanged.connect(self._on_param_changed)
        self.spin_d0 = QDoubleSpinBox()
        self.spin_d0.setRange(0.5, 10.0)
        self.spin_d0.setSingleStep(0.5)
        self.spin_d0.setValue(1.0)
        self.spin_d0.valueChanged.connect(self._on_param_changed)
        pl_box.addWidget(self.spin_pathloss)
        pl_box.addWidget(self.spin_d0)
        lay_chan.addLayout(pl_box, 1, 1)

        lay_chan.addWidget(QLabel("Constante K / Ruído K₀:"), 2, 0)
        kk_box = QHBoxLayout()
        self.edit_k_const = QLineEdit("1e-4")
        self.edit_k_const.editingFinished.connect(self._on_param_changed)
        self.edit_k_0 = QLineEdit("1e-20")
        self.edit_k_0.editingFinished.connect(self._on_param_changed)
        kk_box.addWidget(self.edit_k_const)
        kk_box.addWidget(self.edit_k_0)
        lay_chan.addLayout(kk_box, 2, 1)

        lay_chan.addWidget(QLabel("Shadowing σX / σR:"), 3, 0)
        sig_box = QHBoxLayout()
        self.spin_sigma_x = QDoubleSpinBox()
        self.spin_sigma_x.setRange(0.0, 10.0)
        self.spin_sigma_x.setDecimals(2)
        self.spin_sigma_x.setSingleStep(0.25)
        self.spin_sigma_x.setValue(2.0)
        self.spin_sigma_x.valueChanged.connect(self._on_param_changed)
        self.spin_sigma_r = QDoubleSpinBox()
        self.spin_sigma_r.setRange(0.1, 2.0)
        self.spin_sigma_r.setDecimals(4)
        self.spin_sigma_r.setSingleStep(0.05)
        self.spin_sigma_r.setValue(0.7071)
        self.spin_sigma_r.valueChanged.connect(self._on_param_changed)
        sig_box.addWidget(self.spin_sigma_x)
        sig_box.addWidget(self.spin_sigma_r)
        lay_chan.addLayout(sig_box, 3, 1)

        self.chk_rayleigh = QCheckBox("Habilitar Fast Fading (Rayleigh)")
        self.chk_rayleigh.setChecked(False)
        self.chk_rayleigh.toggled.connect(self._on_param_changed)
        lay_chan.addWidget(self.chk_rayleigh, 4, 0, 1, 2)

        sidebar_layout.addWidget(grp_chan)

        # 4. Controle de Potência (DPC & Dual-Loop)
        grp_pwr = QGroupBox("⚡ Controle de Potência")
        lay_pwr = QGridLayout(grp_pwr)

        lay_pwr.addWidget(QLabel("Algoritmo:"), 0, 0)
        self.combo_power = QComboBox()
        self.combo_power.addItem("Full Power (P_max)", "full_power")
        self.combo_power.addItem("DPC Fixo", "dpc_fixed")
        self.combo_power.addItem("Dual-Loop DPC (Desliga Outage)", "dual_loop")
        self.combo_power.currentIndexChanged.connect(self._on_param_changed)
        lay_pwr.addWidget(self.combo_power, 0, 1)

        lay_pwr.addWidget(QLabel("P_max / P_min (dBm):"), 1, 0)
        pmm_box = QHBoxLayout()
        self.spin_p_max = QDoubleSpinBox()
        self.spin_p_max.setRange(0.0, 50.0)
        self.spin_p_max.setValue(30.0)
        self.spin_p_max.valueChanged.connect(self._on_param_changed)
        self.spin_p_min = QDoubleSpinBox()
        self.spin_p_min.setRange(-40.0, 20.0)
        self.spin_p_min.setValue(0.0)
        self.spin_p_min.valueChanged.connect(self._on_param_changed)
        pmm_box.addWidget(self.spin_p_max)
        pmm_box.addWidget(self.spin_p_min)
        lay_pwr.addLayout(pmm_box, 1, 1)

        lay_pwr.addWidget(QLabel("Passo μ (dB) / Tol (dB):"), 2, 0)
        mt_box = QHBoxLayout()
        self.spin_mu = QDoubleSpinBox()
        self.spin_mu.setRange(0.1, 5.0)
        self.spin_mu.setSingleStep(0.5)
        self.spin_mu.setValue(1.0)
        self.spin_mu.valueChanged.connect(self._on_param_changed)
        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setRange(0.1, 3.0)
        self.spin_tol.setSingleStep(0.1)
        self.spin_tol.setValue(0.5)
        self.spin_tol.valueChanged.connect(self._on_param_changed)
        mt_box.addWidget(self.spin_mu)
        mt_box.addWidget(self.spin_tol)
        lay_pwr.addLayout(mt_box, 2, 1)

        lay_pwr.addWidget(QLabel("γ_tar / γ_min / γ_max (dB):"), 3, 0)
        gam_box = QHBoxLayout()
        self.spin_g_tar = QDoubleSpinBox()
        self.spin_g_tar.setRange(-5.0, 30.0)
        self.spin_g_tar.setValue(10.0)
        self.spin_g_tar.valueChanged.connect(self._on_param_changed)
        self.spin_g_min = QDoubleSpinBox()
        self.spin_g_min.setRange(-30.0, 5.0)
        self.spin_g_min.setValue(-10.0)
        self.spin_g_min.valueChanged.connect(self._on_param_changed)
        self.spin_g_max = QDoubleSpinBox()
        self.spin_g_max.setRange(5.0, 40.0)
        self.spin_g_max.setValue(20.0)
        self.spin_g_max.valueChanged.connect(self._on_param_changed)
        gam_box.addWidget(self.spin_g_tar)
        gam_box.addWidget(self.spin_g_min)
        gam_box.addWidget(self.spin_g_max)
        lay_pwr.addLayout(gam_box, 3, 1)

        lay_pwr.addWidget(QLabel("Loop Ext / Int (iters):"), 4, 0)
        lp_box = QHBoxLayout()
        self.spin_t_outer = QSpinBox()
        self.spin_t_outer.setRange(5, 200)
        self.spin_t_outer.setValue(50)
        self.spin_t_outer.valueChanged.connect(self._on_param_changed)
        self.spin_t_inner = QSpinBox()
        self.spin_t_inner.setRange(10, 1000)
        self.spin_t_inner.setValue(150)
        self.spin_t_inner.valueChanged.connect(self._on_param_changed)
        lp_box.addWidget(self.spin_t_outer)
        lp_box.addWidget(self.spin_t_inner)
        lay_pwr.addLayout(lp_box, 4, 1)

        sidebar_layout.addWidget(grp_pwr)

        # 5. Camadas da Visão 3D e Mapa de Calor
        grp_view = QGroupBox("🗺️ Camadas da Visão 3D")
        lay_view = QVBoxLayout(grp_view)

        lay_view.addWidget(QLabel("Métrica do Mapa de Calor:"))
        self.combo_heatmap_metric = QComboBox()
        self.combo_heatmap_metric.addItem("SINR (dB)", "sinr")
        self.combo_heatmap_metric.addItem("RSRP / Potência Recebida (dBm)", "rsrp")
        self.combo_heatmap_metric.addItem("Capacidade de Shannon (Mbps)", "cap")
        self.combo_heatmap_metric.addItem("Célula Dominante (AP Servidor)", "server")
        self.combo_heatmap_metric.currentIndexChanged.connect(self._on_layer_toggled)
        lay_view.addWidget(self.combo_heatmap_metric)

        lay_view.addWidget(QLabel("Resolução da Malha:"))
        self.combo_heatmap_res = QComboBox()
        self.combo_heatmap_res.addItem("Rápida (25×25)", 25)
        self.combo_heatmap_res.addItem("Balanceada (45×45)", 45)
        self.combo_heatmap_res.addItem("Normal (55×55)", 55)
        self.combo_heatmap_res.addItem("Alta (75×75)", 75)
        self.combo_heatmap_res.setCurrentIndex(1)
        self.combo_heatmap_res.currentIndexChanged.connect(self._on_layer_toggled)
        lay_view.addWidget(self.combo_heatmap_res)

        lay_view.addWidget(QLabel("Opacidade do Mapa de Calor:"))
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(10, 100)
        self.slider_opacity.setValue(75)
        self.slider_opacity.valueChanged.connect(self._on_layer_toggled)
        lay_view.addWidget(self.slider_opacity)

        self.chk_show_heatmap = QCheckBox("Exibir Mapa de Calor de Cobertura")
        self.chk_show_heatmap.setChecked(True)
        self.chk_show_heatmap.toggled.connect(self._on_layer_toggled)
        lay_view.addWidget(self.chk_show_heatmap)

        self.chk_show_beams = QCheckBox("Exibir Feixes das Antenas 3D")
        self.chk_show_beams.setChecked(True)
        self.chk_show_beams.toggled.connect(self._on_layer_toggled)
        lay_view.addWidget(self.chk_show_beams)

        self.chk_show_links = QCheckBox("Exibir Enlaces (Linhas UE ↔ AP)")
        self.chk_show_links.setChecked(True)
        self.chk_show_links.toggled.connect(self._on_layer_toggled)
        lay_view.addWidget(self.chk_show_links)

        self.chk_show_towers = QCheckBox("Exibir Torres dos APs")
        self.chk_show_towers.setChecked(True)
        self.chk_show_towers.toggled.connect(self._on_layer_toggled)
        lay_view.addWidget(self.chk_show_towers)

        self.chk_show_ues = QCheckBox("Exibir Usuários (UEs)")
        self.chk_show_ues.setChecked(True)
        self.chk_show_ues.toggled.connect(self._on_layer_toggled)
        lay_view.addWidget(self.chk_show_ues)

        sidebar_layout.addWidget(grp_view)

        # 6. Experimento de Monte Carlo
        grp_mc = QGroupBox("🎲 Experimento Monte Carlo")
        lay_mc = QGridLayout(grp_mc)

        lay_mc.addWidget(QLabel("Escopo:"), 0, 0)
        self.combo_mc_scope = QComboBox()
        self.combo_mc_scope.addItem("Completo (Antenas + SciChal4A)", "both")
        self.combo_mc_scope.addItem("Apenas Antenas", "antennas")
        self.combo_mc_scope.addItem("Apenas Controle de Potência", "power")
        lay_mc.addWidget(self.combo_mc_scope, 0, 1)

        lay_mc.addWidget(QLabel("Iterações:"), 1, 0)
        self.spin_sim_iters = QSpinBox()
        self.spin_sim_iters.setRange(10, 5000)
        self.spin_sim_iters.setValue(100)
        self.spin_sim_iters.setSingleStep(50)
        lay_mc.addWidget(self.spin_sim_iters, 1, 1)

        sidebar_layout.addWidget(grp_mc)

        # Botões Principais de Ação
        self.btn_run_sim = QPushButton("🚀 Executar Simulação Monte Carlo")
        self.btn_run_sim.setProperty("primary", "true")
        self.btn_run_sim.clicked.connect(self.run_monte_carlo)
        sidebar_layout.addWidget(self.btn_run_sim)

        self.btn_new_snapshot = QPushButton("🎲 Gerar Novo Cenário 3D")
        self.btn_new_snapshot.setProperty("accent", "true")
        self.btn_new_snapshot.clicked.connect(self.update_3d_snapshot)
        sidebar_layout.addWidget(self.btn_new_snapshot)

        btn_export = QPushButton("💾 Exportar Snapshot HTML")
        btn_export.clicked.connect(self.export_snapshot_html)
        sidebar_layout.addWidget(btn_export)

        sidebar_layout.addStretch()

        sidebar_scroll.setWidget(sidebar_content)
        splitter.addWidget(sidebar_scroll)

        # -------------------------------------------------------------------
        # Painel Central de Abas
        # -------------------------------------------------------------------
        self.tabs = QTabWidget()

        self.web_view_3d = QWebEngineView()
        self.tabs.addTab(self.web_view_3d, "🌐 Visão 3D Interativa do Mapa")

        self.web_view_kpi = QWebEngineView()
        self.tabs.addTab(self.web_view_kpi, "📊 KPIs & Gráficos Estatísticos")

        self.tab_mobility = MobilityView()
        self.tabs.addTab(self.tab_mobility, "🚗 Mobilidade & Beam Tracking")

        self.tab_users = UsersTableView()
        self.tabs.addTab(self.tab_users, "📋 Tabela de Usuários & Links")

        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet(
            "background-color: #0d1117; color: #58a6ff; font-family: 'Consolas', monospace; font-size: 11px;"
        )
        self.tabs.addTab(self.log_console, "📝 Console de Logs")

        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        self.lbl_status = QLabel("Pronto")
        self.lbl_status.setStyleSheet("color: #38bdf8; font-weight: bold; padding-left: 6px;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(220)
        self.progress_bar.setVisible(False)

        self.lbl_info = QLabel("Área: 1000×1000m | Simulador Pronto")
        self.lbl_info.setStyleSheet("color: #94a3b8; padding-right: 10px;")

        self.statusbar.addWidget(self.lbl_status, stretch=1)
        self.statusbar.addPermanentWidget(self.progress_bar)
        self.statusbar.addPermanentWidget(self.lbl_info)

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_console.appendPlainText(f"[{timestamp}] {message}")

    def _parse_float(self, txt: str, fallback: float) -> float:
        try:
            return float(txt.strip())
        except Exception:
            return fallback

    def _get_current_params(self) -> dict:
        return {
            "M": int(self.combo_aps.currentData()),
            "K": int(self.spin_ues.value()),
            "N": int(self.spin_channels.value()),
            "allocation": str(self.combo_alloc.currentData()),
            "h_ap": float(self.spin_h_ap.value()),
            "h_ue": float(self.spin_h_ue.value()),
            "seed": int(self.spin_seed.value()),
            "model": str(self.combo_ant_model.currentData()),
            "L": int(self.combo_l.currentData()),
            "J": int(self.combo_j.currentData()),
            "aoa_err_deg": float(self.spin_aoa_err.value()),
            "beamwidth_deg": float(self.spin_beamwidth.value()),
            "g_min": float(self.spin_g_min.value()),
            "b_total_mhz": float(self.spin_bw.value()),
            "pathloss_exp": float(self.spin_pathloss.value()),
            "d_0": float(self.spin_d0.value()),
            "k_const": self._parse_float(self.edit_k_const.text(), 1e-4),
            "k_0": self._parse_float(self.edit_k_0.text(), 1e-20),
            "sigma_x": float(self.spin_sigma_x.value()),
            "sigma_r": float(self.spin_sigma_r.value()),
            "use_rayleigh": bool(self.chk_rayleigh.isChecked()),
            "power_mode": str(self.combo_power.currentData()),
            "p_max_dbm": float(self.spin_p_max.value()),
            "p_min_dbm": float(self.spin_p_min.value()),
            "mu_step_db": float(self.spin_mu.value()),
            "tol_db": float(self.spin_tol.value()),
            "gamma_tar_db": float(self.spin_g_tar.value()),
            "gamma_min_db": float(self.spin_g_min.value()),
            "gamma_max_db": float(self.spin_g_max.value()),
            "t_outer": int(self.spin_t_outer.value()),
            "t_inner": int(self.spin_t_inner.value()),
            "heatmap_res": int(self.combo_heatmap_res.currentData()),
            "heatmap_metric": str(self.combo_heatmap_metric.currentData()),
            "sim": int(self.spin_sim_iters.value()),
            "mc_scope": str(self.combo_mc_scope.currentData()),
        }

    def _on_param_changed(self):
        self.update_3d_snapshot()

    def _on_layer_toggled(self):
        if self._current_snapshot is None:
            self.update_3d_snapshot()
            return

        metric = str(self.combo_heatmap_metric.currentData())
        res = int(self.combo_heatmap_res.currentData())
        if (
            self._current_snapshot.get("heatmap", {}).get("metric") != metric
            or len(self._current_snapshot.get("heatmap", {}).get("x", [])) != res
        ):
            self.update_3d_snapshot()
            return

        self._render_3d_view()

    def update_3d_snapshot(self):
        params = self._get_current_params()
        self.lbl_status.setText("Calculando novo cenário 3D e mapa de calor...")
        self.log(f"Gerando cenário 3D (M={params['M']}, K={params['K']}, modelo={params['model']}, pot={params['power_mode']})...")

        t0 = time.time()
        try:
            snapshot = SnapshotCalculator.compute_snapshot(params)
            self._current_snapshot = snapshot

            self._render_3d_view()
            self.tab_users.update_data(snapshot["ues"], snapshot["aps"])

            # Atualiza aba de KPIs instantâneos enquanto não houver Monte Carlo rodado
            if self._latest_simulation is None:
                self.web_view_kpi.setHtml(KPIViewGenerator.generate_snapshot_kpi_html(snapshot))

            dt = (time.time() - t0) * 1000
            self.lbl_status.setText(f"Cenário 3D atualizado em {dt:.0f} ms")
            self.lbl_info.setText(
                f"M={params['M']} APs | K={params['K']} UEs | SINR Médio: {snapshot['avg_sinr']:+.1f} dB | Vazão: {snapshot['sum_capacity']:.1f} Mbps"
            )
            self.log(f"Cenário 3D atualizado ({dt:.1f} ms). SINR Médio: {snapshot['avg_sinr']:+.1f} dB.")

        except Exception as e:
            import traceback

            err = traceback.format_exc()
            self.lbl_status.setText("Erro ao gerar cenário")
            self.log(f"ERRO: {str(e)}\n{err}")
            QMessageBox.critical(self, "Erro", f"Falha ao gerar cenário 3D:\n{str(e)}")

    def _render_3d_view(self):
        if self._current_snapshot is None:
            return

        opacity = float(self.slider_opacity.value()) / 100.0
        html = View3DGenerator.generate_html(
            snapshot_data=self._current_snapshot,
            show_heatmap=self.chk_show_heatmap.isChecked(),
            show_beams=self.chk_show_beams.isChecked(),
            show_links=self.chk_show_links.isChecked(),
            show_towers=self.chk_show_towers.isChecked(),
            show_ues=self.chk_show_ues.isChecked(),
            heatmap_opacity=opacity,
        )
        self.web_view_3d.setHtml(html)

    def run_monte_carlo(self):
        params = self._get_current_params()
        params["power_control"] = params["power_mode"]
        params["models"] = ("omni", "sector", "smb", "adaptive", "adaptive_1", "adaptive_5", "adaptive_10")

        self.btn_run_sim.setEnabled(False)
        self.btn_new_snapshot.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Executando Simulação Monte Carlo...")

        self.log(f"Iniciando Monte Carlo ({params['mc_scope']}, {params['sim']} iterações)...")

        self._worker = MonteCarloWorker(params)
        self._worker.progress_changed.connect(self._on_worker_progress)
        self._worker.simulation_finished.connect(self._on_simulation_finished)
        self._worker.simulation_error.connect(self._on_simulation_error)
        self._worker.start()

    def _on_worker_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.lbl_status.setText(msg)

    def _on_simulation_finished(self, results: dict):
        self._latest_simulation = results
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)
        self.btn_run_sim.setEnabled(True)
        self.btn_new_snapshot.setEnabled(True)
        self.lbl_status.setText("Simulação Monte Carlo finalizada com sucesso!")

        self.log("Simulação Monte Carlo concluída. Atualizando painel de KPIs...")
        kpi_html = KPIViewGenerator.generate_kpi_html(results)
        self.web_view_kpi.setHtml(kpi_html)
        if results.get("mc_rounds"):
            self.tab_users.set_mc_rounds(results["mc_rounds"])
        self.tabs.setCurrentIndex(1)

    def _on_simulation_error(self, err_msg: str):
        self.progress_bar.setVisible(False)
        self.btn_run_sim.setEnabled(True)
        self.btn_new_snapshot.setEnabled(True)
        self.lbl_status.setText("Erro na simulação")
        self.log(f"ERRO DE SIMULAÇÃO:\n{err_msg}")
        QMessageBox.critical(self, "Erro na Simulação", err_msg)

    def export_snapshot_html(self):
        if self._current_snapshot is None:
            QMessageBox.warning(self, "Exportar", "Nenhum cenário 3D disponível para exportar.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Salvar Cenário 3D em HTML", "mapa_3d_cobertura.html", "HTML (*.html)")
        if not path:
            return

        try:
            opacity = float(self.slider_opacity.value()) / 100.0
            html = View3DGenerator.generate_html(
                snapshot_data=self._current_snapshot,
                show_heatmap=self.chk_show_heatmap.isChecked(),
                show_beams=self.chk_show_beams.isChecked(),
                show_links=self.chk_show_links.isChecked(),
                show_towers=self.chk_show_towers.isChecked(),
                show_ues=self.chk_show_ues.isChecked(),
                heatmap_opacity=opacity,
            )
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            QMessageBox.information(self, "Sucesso", f"Cenário 3D exportado com sucesso em:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha ao exportar HTML:\n{str(e)}")

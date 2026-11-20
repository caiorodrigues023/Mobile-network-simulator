# ---------------------------------------------------------------------------
# src/simulator/gui/main_window.py — Janela Principal da Interface Gráfica
# ---------------------------------------------------------------------------

import os
import sys
import time
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTabWidget, QGroupBox, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QPushButton, QSlider, QProgressBar, QStatusBar,
    QScrollArea, QFileDialog, QMessageBox, QTextEdit, QPlainTextEdit
)
from PyQt6.QtWebEngineWidgets import QWebEngineView

from .styles import MODERN_DARK_THEME
from .worker import MonteCarloWorker, SnapshotCalculator
from .view3d import View3DGenerator
from .kpi_view import KPIViewGenerator
from .users_table import UsersTableView
from .mobility_view import MobilityView


class SimulatorMainWindow(QMainWindow):
    """Janela Principal do Simulador de Redes Móveis com Visão 3D e KPIs."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mobile Network Simulator 3D — Simulador de Redes Móveis")
        self.resize(1380, 880)
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
        # Painel Lateral Esquerdo (Controles de Simulação)
        # -------------------------------------------------------------------
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFixedWidth(360)

        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(12)

        # 1. Topologia da Rede
        grp_topo = QGroupBox("📐 Topologia da Rede")
        lay_topo = QVBoxLayout(grp_topo)

        lay_topo.addWidget(QLabel("Número de APs (M):"))
        self.combo_aps = QComboBox()
        self.combo_aps.addItem("4 APs (Grade 2×2)", 4)
        self.combo_aps.addItem("9 APs (Grade 3×3)", 9)
        self.combo_aps.addItem("16 APs (Grade 4×4)", 16)
        self.combo_aps.currentIndexChanged.connect(self._on_param_changed)
        lay_topo.addWidget(self.combo_aps)

        lay_topo.addWidget(QLabel("Número de Usuários (K):"))
        self.spin_ues = QSpinBox()
        self.spin_ues.setRange(1, 128)
        self.spin_ues.setValue(8)
        self.spin_ues.valueChanged.connect(self._on_param_changed)
        lay_topo.addWidget(self.spin_ues)

        lay_topo.addWidget(QLabel("Canais Ortogonais (N):"))
        self.spin_channels = QSpinBox()
        self.spin_channels.setRange(1, 16)
        self.spin_channels.setValue(1)
        self.spin_channels.valueChanged.connect(self._on_param_changed)
        lay_topo.addWidget(self.spin_channels)

        sidebar_layout.addWidget(grp_topo)

        # 2. Antenas e Feixes (Beamforming)
        grp_ant = QGroupBox("📡 Tecnologia de Antena & Feixes")
        lay_ant = QVBoxLayout(grp_ant)

        lay_ant.addWidget(QLabel("Modelo de Antena:"))
        self.combo_ant_model = QComboBox()
        self.combo_ant_model.addItem("SMB (Switched Multi-Beam UCA)", "smb")
        self.combo_ant_model.addItem("Setorizado (3 setores de 120°)", "sector")
        self.combo_ant_model.addItem("Omnidirecional", "omni")
        self.combo_ant_model.addItem("Adaptativo MRC (Ideal 0°)", "adaptive")
        self.combo_ant_model.addItem("Adaptativo MRC (Erro 1°)", "adaptive_1")
        self.combo_ant_model.addItem("Adaptativo MRC (Erro 5°)", "adaptive_5")
        self.combo_ant_model.addItem("Adaptativo MRC (Erro 10°)", "adaptive_10")
        self.combo_ant_model.currentIndexChanged.connect(self._on_param_changed)
        lay_ant.addWidget(self.combo_ant_model)

        lay_ant.addWidget(QLabel("Elementos do Arranjo (L):"))
        self.combo_l = QComboBox()
        for l_val in (4, 8, 16, 32):
            self.combo_l.addItem(f"L = {l_val} elementos", l_val)
        self.combo_l.setCurrentIndex(1)  # L=8
        self.combo_l.currentIndexChanged.connect(self._on_param_changed)
        lay_ant.addWidget(self.combo_l)

        lay_ant.addWidget(QLabel("Feixes SMB (J):"))
        self.combo_j = QComboBox()
        for j_val in (4, 8, 12, 16):
            self.combo_j.addItem(f"J = {j_val} feixes", j_val)
        self.combo_j.setCurrentIndex(1)  # J=8
        self.combo_j.currentIndexChanged.connect(self._on_param_changed)
        lay_ant.addWidget(self.combo_j)

        sidebar_layout.addWidget(grp_ant)

        # 3. Propagação e Canal
        grp_chan = QGroupBox("🌊 Canal & Propagação")
        lay_chan = QVBoxLayout(grp_chan)

        self.chk_rayleigh = QCheckBox("Habilitar Fast Fading (Rayleigh)")
        self.chk_rayleigh.setChecked(False)
        self.chk_rayleigh.toggled.connect(self._on_param_changed)
        lay_chan.addWidget(self.chk_rayleigh)

        lay_chan.addWidget(QLabel("Controle de Potência:"))
        self.combo_power = QComboBox()
        self.combo_power.addItem("Full Power (1W / 30 dBm)", "full_power")
        self.combo_power.addItem("DPC Fixo (Meta 10 dB)", "dpc_fixed")
        self.combo_power.addItem("Dual-Loop DPC (Mitigação Outage)", "dual_loop")
        self.combo_power.currentIndexChanged.connect(self._on_param_changed)
        lay_chan.addWidget(self.combo_power)

        sidebar_layout.addWidget(grp_chan)

        # 4. Configurações da Visão 3D e Mapa de Calor
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
        self.combo_heatmap_res.addItem("Rápida (35×35)", 35)
        self.combo_heatmap_res.addItem("Normal (50×50)", 50)
        self.combo_heatmap_res.addItem("Alta (70×70)", 70)
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

        # 5. Experimento de Monte Carlo
        grp_mc = QGroupBox("🎲 Experimento Monte Carlo")
        lay_mc = QVBoxLayout(grp_mc)

        lay_mc.addWidget(QLabel("Iterações de Simulação:"))
        self.spin_sim_iters = QSpinBox()
        self.spin_sim_iters.setRange(10, 10000)
        self.spin_sim_iters.setValue(100)
        self.spin_sim_iters.setSingleStep(50)
        lay_mc.addWidget(self.spin_sim_iters)

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

        # Aba 1: Visão 3D do Mapa
        self.web_view_3d = QWebEngineView()
        self.tabs.addTab(self.web_view_3d, "🌐 Visão 3D Interativa do Mapa")

        # Aba 2: KPIs & Gráficos
        self.web_view_kpi = QWebEngineView()
        self.tabs.addTab(self.web_view_kpi, "📊 KPIs & Gráficos Estatísticos")

        # Aba 3: Mobilidade & Beam Tracking (Atividade 2)
        self.tab_mobility = MobilityView()
        self.tabs.addTab(self.tab_mobility, "🚗 Mobilidade & Beam Tracking")

        # Aba 4: Tabela de Usuários
        self.tab_users = UsersTableView()
        self.tabs.addTab(self.tab_users, "📋 Tabela de Usuários & Links")

        # Aba 5: Logs do Sistema
        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background-color: #0d1117; color: #58a6ff; font-family: 'Consolas', monospace; font-size: 11px;")
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
        """Adiciona mensagem formatada com timestamp no console de logs."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_console.appendPlainText(f"[{timestamp}] {message}")

    def _get_current_params(self) -> dict:
        return {
            "M": int(self.combo_aps.currentData()),
            "K": int(self.spin_ues.value()),
            "N": int(self.spin_channels.value()),
            "model": str(self.combo_ant_model.currentData()),
            "L": int(self.combo_l.currentData()),
            "J": int(self.combo_j.currentData()),
            "use_rayleigh": bool(self.chk_rayleigh.isChecked()),
            "power_mode": str(self.combo_power.currentData()),
            "heatmap_res": int(self.combo_heatmap_res.currentData()),
            "heatmap_metric": str(self.combo_heatmap_metric.currentData()),
            "h_ap": 25.0,
            "h_ue": 1.5,
        }

    def _on_param_changed(self):
        """Dispara atualização instantânea do cenário 3D quando parâmetros mudam."""
        self.update_3d_snapshot()

    def _on_layer_toggled(self):
        """Re-renderiza o HTML 3D sem recalcular a geometria inteira caso apenas layers tenham mudado."""
        if self._current_snapshot is None:
            self.update_3d_snapshot()
            return

        # Atualiza a métrica de heatmap se mudou
        metric = str(self.combo_heatmap_metric.currentData())
        res = int(self.combo_heatmap_res.currentData())
        if (
            self._current_snapshot.get("heatmap", {}).get("metric") != metric or
            len(self._current_snapshot.get("heatmap", {}).get("x", [])) != res
        ):
            self.update_3d_snapshot()
            return

        self._render_3d_view()

    def update_3d_snapshot(self):
        """Calcula um novo snapshot representativo e atualiza a cena 3D e a tabela."""
        params = self._get_current_params()
        self.lbl_status.setText("Calculando novo cenário 3D e mapa de calor...")
        self.log(f"Gerando cenário 3D com M={params['M']}, K={params['K']}, modelo={params['model']}...")

        t0 = time.time()
        try:
            snapshot = SnapshotCalculator.compute_snapshot(params)
            self._current_snapshot = snapshot

            self._render_3d_view()

            # Atualiza tabela de usuários
            self.tab_users.update_data(snapshot["ues"], snapshot["aps"])

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
        """Compila a cena 3D com Plotly e carrega no WebEngine."""
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
        """Executa a simulação Monte Carlo em uma thread separada."""
        params = self._get_current_params()
        params["sim"] = int(self.spin_sim_iters.value())
        params["power_control"] = str(self.combo_power.currentData())
        params["models"] = ("omni", "sector", "smb", "adaptive")

        self.btn_run_sim.setEnabled(False)
        self.btn_new_snapshot.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Executando Simulação Monte Carlo...")

        self.log(f"Iniciando simulação Monte Carlo com {params['sim']} iterações...")

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

        # Atualiza o painel de KPIs e gráficos
        kpi_html = KPIViewGenerator.generate_kpi_html(results)
        self.web_view_kpi.setHtml(kpi_html)

        # Muda automaticamente para a aba de KPIs para exibir os resultados ao usuário
        self.tabs.setCurrentIndex(1)

    def _on_simulation_error(self, err_msg: str):
        self.progress_bar.setVisible(False)
        self.btn_run_sim.setEnabled(True)
        self.btn_new_snapshot.setEnabled(True)
        self.lbl_status.setText("Erro na simulação")
        self.log(f"ERRO DE SIMULAÇÃO:\n{err_msg}")
        QMessageBox.critical(self, "Erro na Simulação", err_msg)

    def export_snapshot_html(self):
        """Exporta a cena 3D atual como arquivo HTML autocontido."""
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

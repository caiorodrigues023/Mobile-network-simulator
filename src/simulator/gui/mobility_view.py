# ---------------------------------------------------------------------------
# src/simulator/gui/mobility_view.py — Painel de Mobilidade e Beam Tracking
# ---------------------------------------------------------------------------
"""
Contém:
  - generate_mobility_html()  → Função standalone que gera HTML (usada pelo modo web)
  - MobilityView              → Widget Qt (só instanciado no modo desktop)
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

from ..antennas import calc_ula_gain
from ..config import (
    MOBILITY_DISTANCES_M,
    MOBILITY_PROFILES_KMH,
    MOBILITY_T_SIM,
    MOBILITY_ULA_SIZES,
    MOBILITY_UPDATE_PERIODS_S,
)


# ---------------------------------------------------------------------------
# Função standalone — funciona SEM Qt, usada pelo app_web.py
# ---------------------------------------------------------------------------
def generate_mobility_html(
    v_kmh: float = 120.0,
    T_update: float = 0.100,
    D: float = 50.0,
    L_selected: int = 16,
    plotly_url: str | None = None,
) -> str:
    """Gera o HTML completo do painel de mobilidade/beam-tracking."""

    if plotly_url is None:
        assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
        plotly_src = "file:///" + os.path.join(assets_dir, "plotly.min.js").replace("\\", "/")
    else:
        plotly_src = plotly_url

    v_ms = v_kmh / 3.6
    delta_x = v_ms * T_update
    t_sim = MOBILITY_T_SIM

    # Deslocamento angular e perda por desalinhamento no broadside
    theta_0 = 0.0  # broadside
    theta_1 = float(np.arctan2(delta_x, D))
    theta_1_deg = np.degrees(theta_1)

    # Ganho do ULA na direção real vs apontada
    gain_perfect = float(calc_ula_gain(np.array([theta_0]), np.array([theta_0]), L_selected)[0])
    gain_lag = float(calc_ula_gain(np.array([theta_1]), np.array([theta_0]), L_selected)[0])
    loss_db = 10.0 * np.log10(max(gain_lag / gain_perfect, 1e-6))

    # Geração dos subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Trajetória Tangencial e Lóbulos ULA (D = {D:.0f} m, v = {v_kmh:.0f} km/h)",
            f"Perda de Desalinhamento Angular no Broadside (Δx = {delta_x:.2f} m, Δθ = {theta_1_deg:.2f}°)",
        ),
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.10,
    )

    # -------------------------------------------------------------------
    # Painel A: Trajetória no espaço 2D
    # -------------------------------------------------------------------
    t = np.linspace(0, t_sim, 500)
    x_traj = v_ms * (t - t_sim / 2.0)
    y_traj = np.full_like(x_traj, D)

    fig.add_trace(
        go.Scatter(
            x=x_traj, y=y_traj,
            mode="lines",
            line=dict(color="#38bdf8", width=3),
            name=f"Trajetória do UE ({v_kmh:g} km/h)",
            hovertemplate="X: %{x:.1f} m<br>Y: %{y:.1f} m<extra></extra>",
        ),
        row=1, col=1,
    )

    # Marcadores de atualização de feixe ao longo da trajetória
    t_samples = np.arange(0, t_sim, T_update)
    x_samples = v_ms * (t_samples - t_sim / 2.0)
    y_samples = np.full_like(x_samples, D)
    fig.add_trace(
        go.Scatter(
            x=x_samples, y=y_samples,
            mode="markers",
            marker=dict(symbol="circle-open", size=6, color="#f59e0b", line=dict(width=1.5)),
            name=f"Atualizações do Feixe (ΔT = {T_update*1000:.0f} ms)",
            hoverinfo="skip",
        ),
        row=1, col=1,
    )

    # Posição do AP na origem
    fig.add_trace(
        go.Scatter(
            x=[0], y=[0],
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=14, color="#10b981", line=dict(color="#ffffff", width=2)),
            text=["📡 AP (ULA)"],
            textposition="bottom center",
            name="AP (Estação Base)",
            hoverinfo="text",
            hovertext="AP na Origem (0, 0) com Arranjo ULA",
        ),
        row=1, col=1,
    )

    # Desenho do lóbulo de radiação apontado para broadside (90° em relação ao eixo da ULA)
    psi = np.linspace(-np.pi / 2, np.pi / 2, 800)
    lobe_r = D * 0.9 * (calc_ula_gain(psi, np.zeros_like(psi), L_selected) / L_selected)
    lobe_x = lobe_r * np.sin(psi)
    lobe_y = lobe_r * np.cos(psi)

    fig.add_trace(
        go.Scatter(
            x=lobe_x, y=lobe_y,
            mode="lines",
            fill="toself",
            fillcolor="rgba(16, 185, 129, 0.2)",
            line=dict(color="#10b981", width=2),
            name=f"Lóbulo ULA (L = {L_selected})",
            hoverinfo="skip",
        ),
        row=1, col=1,
    )

    # -------------------------------------------------------------------
    # Painel B: Zoom no Broadside e Perda por L
    # -------------------------------------------------------------------
    L_candidates = [8, 16, 32]
    colors_l = ["#38bdf8", "#10b981", "#f43f5e"]
    angles_test = np.linspace(0, np.radians(15), 400)

    for l_c, col in zip(L_candidates, colors_l):
        g_cand = calc_ula_gain(angles_test, np.zeros_like(angles_test), l_c)
        g_peak = calc_ula_gain(np.zeros(1), np.zeros(1), l_c)[0]
        loss_curve = 10.0 * np.log10(np.maximum(g_cand / g_peak, 1e-4))

        fig.add_trace(
            go.Scatter(
                x=np.degrees(angles_test),
                y=loss_curve,
                mode="lines",
                name=f"ULA L = {l_c}",
                line=dict(color=col, width=2.5 if l_c == L_selected else 1.5,
                          dash="solid" if l_c == L_selected else "dot"),
                hovertemplate=f"<b>L = {l_c}</b><br>Desalinhamento: %{{x:.2f}}°<br>Ganho: %{{y:.2f}} dB<extra></extra>",
            ),
            row=1, col=2,
        )

    # Linha vertical indicando o erro angular atual
    fig.add_vline(
        x=theta_1_deg,
        line_width=2,
        line_dash="dash",
        line_color="#f97316",
        annotation_text=f"Δθ = {theta_1_deg:.2f}° (Perda: {loss_db:+.1f} dB)",
        annotation_position="bottom right",
        annotation_font_color="#f97316",
        row=1, col=2,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0e14",
        plot_bgcolor="#111622",
        font=dict(family="Segoe UI, sans-serif", color="#e2e8f0"),
        margin=dict(l=40, r=30, t=60, b=40),
        height=580,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(17, 22, 34, 0.8)",
            bordercolor="#232c3d",
        ),
    )

    fig.update_xaxes(title="Distância X (m)", gridcolor="#1e293b", row=1, col=1)
    fig.update_yaxes(title="Distância Y (m)", gridcolor="#1e293b", row=1, col=1)

    fig.update_xaxes(title="Erro Angular Δθ (graus)", gridcolor="#1e293b", row=1, col=2)
    fig.update_yaxes(title="Atenuação por Desalinhamento (dB)", range=[-25, 1], gridcolor="#1e293b", row=1, col=2)

    chart_div = pyo.plot(
        fig,
        include_plotlyjs=False,
        output_type="div",
        config={"responsive": True, "displayModeBar": True},
    )

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="{plotly_src}"></script>
    <style>
        body {{
            background-color: #0b0e14;
            color: #e2e8f0;
            font-family: 'Segoe UI', sans-serif;
            margin: 0;
            padding: 12px;
            overflow-y: auto;
        }}
        .metrics-banner {{
            display: flex;
            gap: 12px;
            margin-bottom: 12px;
        }}
        .banner-item {{
            background: #141923;
            border: 1px solid #232c3d;
            border-radius: 6px;
            padding: 10px 14px;
            flex: 1;
        }}
        .banner-label {{ font-size: 10px; color: #94a3b8; text-transform: uppercase; }}
        .banner-val {{ font-size: 16px; font-weight: bold; color: #38bdf8; }}
        .chart-box {{
            background: #141923;
            border: 1px solid #232c3d;
            border-radius: 8px;
            padding: 8px;
        }}
    </style>
</head>
<body>
    <div class="metrics-banner">
        <div class="banner-item">
            <div class="banner-label">Deslocamento por Período (Δx)</div>
            <div class="banner-val">{delta_x:.2f} <span style="font-size: 11px; color:#94a3b8;">metros</span></div>
        </div>
        <div class="banner-item">
            <div class="banner-label">Atraso Angular de Rastreamento (Δθ)</div>
            <div class="banner-val">{theta_1_deg:.2f} <span style="font-size: 11px; color:#94a3b8;">graus</span></div>
        </div>
        <div class="banner-item">
            <div class="banner-label">Perda por Desalinhamento (L={L_selected})</div>
            <div class="banner-val" style="color: {'#10b981' if loss_db > -3.0 else '#f43f5e'};">{loss_db:+.2f} <span style="font-size: 11px; color:#94a3b8;">dB</span></div>
        </div>
        <div class="banner-item">
            <div class="banner-label">Velocidade do Usuário</div>
            <div class="banner-val">{v_kmh:.0f} <span style="font-size: 11px; color:#94a3b8;">km/h ({v_ms:.1f} m/s)</span></div>
        </div>
    </div>
    <div class="chart-box">
        {chart_div}
    </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Widget Qt — só é importado quando o modo desktop (PyQt6) está ativo
# ---------------------------------------------------------------------------
try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
        QDoubleSpinBox, QSpinBox, QPushButton, QGroupBox
    )
    from PyQt6.QtWebEngineWidgets import QWebEngineView

    class MobilityView(QWidget):
        """Aba interativa para análise do cenário de Mobilidade e Beam Tracking (Atividade 2)."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self._setup_ui()
            self._update_plot()

        def _setup_ui(self):
            main_layout = QVBoxLayout(self)
            main_layout.setContentsMargins(12, 12, 12, 12)
            main_layout.setSpacing(10)

            # Barra de parâmetros de mobilidade
            params_box = QGroupBox("⚙️ Parâmetros do Cenário de Mobilidade (Atividade 2)")
            box_layout = QHBoxLayout(params_box)
            box_layout.setSpacing(14)

            # Perfil de Velocidade
            lbl_speed = QLabel("Perfil de Velocidade:")
            self.combo_speed = QComboBox()
            for name, val in MOBILITY_PROFILES_KMH.items():
                self.combo_speed.addItem(f"{name} ({val:g} km/h)", val)
            self.combo_speed.setCurrentIndex(2)  # Highway 120 km/h
            self.combo_speed.currentIndexChanged.connect(self._update_plot)

            # Período de Atualização Delta T
            lbl_dt = QLabel("Período de Atualização (ΔT):")
            self.combo_dt = QComboBox()
            self.combo_dt.addItem("10 ms (Rápido / High)", 0.010)
            self.combo_dt.addItem("50 ms (Médio / Fair)", 0.050)
            self.combo_dt.addItem("100 ms (Lento / Slow)", 0.100)
            self.combo_dt.setCurrentIndex(2)
            self.combo_dt.currentIndexChanged.connect(self._update_plot)

            # Distância D ao AP
            lbl_dist = QLabel("Distância Tangencial (D):")
            self.combo_dist = QComboBox()
            for d in MOBILITY_DISTANCES_M:
                self.combo_dist.addItem(f"{d:g} metros", d)
            self.combo_dist.setCurrentIndex(1)  # 50m
            self.combo_dist.currentIndexChanged.connect(self._update_plot)

            # Tamanho do ULA (L)
            lbl_l = QLabel("Elementos do ULA (L):")
            self.combo_l = QComboBox()
            for l_val in MOBILITY_ULA_SIZES:
                self.combo_l.addItem(f"L = {l_val} elementos", l_val)
            self.combo_l.setCurrentIndex(1)  # 16
            self.combo_l.currentIndexChanged.connect(self._update_plot)

            btn_refresh = QPushButton("🔄 Atualizar Análise")
            btn_refresh.clicked.connect(self._update_plot)

            box_layout.addWidget(lbl_speed)
            box_layout.addWidget(self.combo_speed)
            box_layout.addWidget(lbl_dt)
            box_layout.addWidget(self.combo_dt)
            box_layout.addWidget(lbl_dist)
            box_layout.addWidget(self.combo_dist)
            box_layout.addWidget(lbl_l)
            box_layout.addWidget(self.combo_l)
            box_layout.addWidget(btn_refresh)
            box_layout.addStretch()

            main_layout.addWidget(params_box)

            # WebEngineView para exibir os gráficos interativos de mobilidade
            self.webview = QWebEngineView()
            main_layout.addWidget(self.webview)

        def _update_plot(self):
            v_kmh = float(self.combo_speed.currentData())
            T_update = float(self.combo_dt.currentData())
            D = float(self.combo_dist.currentData())
            L_sel = int(self.combo_l.currentData())

            html = generate_mobility_html(v_kmh, T_update, D, L_sel)
            self.webview.setHtml(html)

except ImportError:
    # PyQt6 não disponível — a classe MobilityView não será definida,
    # mas generate_mobility_html() continua funcionando.
    pass

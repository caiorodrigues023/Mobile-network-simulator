# ---------------------------------------------------------------------------
# src/simulator/gui/kpi_view.py — Painel Analítico Completo de KPIs e Gráficos
# ---------------------------------------------------------------------------

import json
import os
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

from ..config import (
    B_TOTAL,
    GAMMA_MAX_DB,
    GAMMA_MIN_DB,
    GAMMA_TAR_INIT,
    MODEL_LABELS,
    MU_STEP,
    P_MAX_DBM,
    P_MIN_DBM,
)


def _jain_fairness(values: np.ndarray) -> float:
    """Calcula o Índice de Justiça de Jain (0 a 1) para um vetor de capacidades."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0
    s1 = float(np.sum(arr))
    s2 = float(np.sum(arr**2))
    if s2 <= 0.0:
        return 0.0
    return (s1**2) / (len(arr) * s2)


class KPIViewGenerator:
    """Gera o painel completo de métricas (KPIs), tabelas benchmark e gráficos Plotly."""

    ANTENNA_COLOR_MAP = {
        "omni": "#38bdf8",
        "sector": "#f59e0b",
        "smb": "#10b981",
        "adaptive": "#f43f5e",
        "adaptive_1": "#8b5cf6",
        "adaptive_5": "#ec4899",
        "adaptive_10": "#a855f7",
    }

    ALGO_ORDER = ("full_power", "dpc_fixed", "dual_loop")
    ALGO_LABELS = {
        "full_power": "Full Power (P_max)",
        "dpc_fixed": "DPC Fixo (Meta SINR)",
        "dual_loop": "Dual-Loop DPC (Outage)",
    }
    ALGO_COLORS = {
        "full_power": "#94a3b8",
        "dpc_fixed": "#38bdf8",
        "dual_loop": "#f97316",
    }

    @classmethod
    def _resolve_plotly_src(cls, plotly_url: str | None) -> str:
        if plotly_url is None:
            assets_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "assets")
            )
            return "file:///" + os.path.join(assets_dir, "plotly.min.js").replace(
                "\\", "/"
            )
        return plotly_url

    @classmethod
    def generate_kpi_html(
        cls, simulation_results: dict, plotly_url: str = None
    ) -> str:
        """
        Gera o HTML completo para resultados de Monte Carlo:
          - type == 'combined': sub-abas Antenas + Controle de Potência SciChal4A
          - type == 'antenna_comparison': apenas Antenas
          - type == 'scichal4a': apenas Controle de Potência SciChal4A
        """
        plotly_src = cls._resolve_plotly_src(plotly_url)
        if not simulation_results:
            return cls._empty_state_html(plotly_src)

        res_type = simulation_results.get("type", "antenna_comparison")
        data = simulation_results.get("data", {})
        params = simulation_results.get("params", {})

        if res_type == "combined":
            ant_data = data.get("antennas", {})
            sci_data = data.get("scichal4a", {})
            if not ant_data and not sci_data:
                return cls._empty_state_html(plotly_src)
            if ant_data and sci_data:
                ant_body = cls._build_antenna_section(ant_data, params)
                dl_hist = sci_data.get("dual_loop_hist", [])
                conv_body = (
                    cls._build_dual_loop_convergence_section(dl_hist, params, section_uid="dl")
                    if dl_hist
                    else ""
                )
                sci_body = cls._build_scichal4a_section(
                    sci_data, params, has_dedicated_conv_tab=bool(conv_body)
                )
                return cls._wrap_document(
                    plotly_src=plotly_src,
                    params=params,
                    title="Dashboard Estatístico Monte Carlo",
                    subtitle="Comparativo de Antenas, Controle de Potência (SciChal4A) & Convergência Temporal Dual-Loop",
                    body_html=cls._build_combined_tabs(ant_body, sci_body, conv_body),
                )
            elif sci_data:
                res_type, data = "scichal4a", sci_data
            else:
                res_type, data = "antenna_comparison", ant_data

        if res_type == "scichal4a":
            if not data:
                return cls._empty_state_html(plotly_src)
            dl_hist = data.get("dual_loop_hist", [])
            conv_body = (
                cls._build_dual_loop_convergence_section(dl_hist, params, section_uid="dl")
                if dl_hist
                else ""
            )
            sci_body = cls._build_scichal4a_section(
                data, params, has_dedicated_conv_tab=False, embedded_conv_html=conv_body
            )
            return cls._wrap_document(
                plotly_src=plotly_src,
                params=params,
                title="Controle de Potência — Joint MRC & Dual-Loop PC",
                subtitle="Avaliação física com corte de Outage e convergência da potência Tx",
                body_html=sci_body,
            )

        if not data:
            return cls._empty_state_html(plotly_src)
        ant_body = cls._build_antenna_section(data, params)
        ant_dl_hist = simulation_results.get("dual_loop_hist", [])
        if ant_dl_hist:
            conv_body = cls._build_dual_loop_convergence_section(
                ant_dl_hist, params, section_uid="dl"
            )
            body_html = f"""
            <div class="subtab-bar">
                <button class="subtab-btn active" onclick="switchSubTab(0)">📡 Comparativo de Antenas</button>
                <button id="subtab-dl-btn" class="subtab-btn" onclick="switchSubTab(1)">🔬 Convergência Temporal Dual-Loop (Potência Tx &amp; SINR)</button>
            </div>
            <div id="subpanel-0" class="subtab-pane active">{ant_body}</div>
            <div id="subpanel-1" class="subtab-pane">{conv_body}</div>
            """
        else:
            body_html = ant_body
        return cls._wrap_document(
            plotly_src=plotly_src,
            params=params,
            title="Comparativo de Tecnologias de Antena",
            subtitle="Avaliação estatística de CDFs, ganho de borda e capacidade",
            body_html=body_html,
        )

    @classmethod
    def generate_snapshot_kpi_html(
        cls, snapshot_data: dict, plotly_url: str = None
    ) -> str:
        """Gera painel de KPIs instantâneos para o cenário 3D atual."""
        plotly_src = cls._resolve_plotly_src(plotly_url)
        if not snapshot_data or not snapshot_data.get("ues"):
            return cls._empty_state_html(plotly_src)

        params = snapshot_data.get("params", {})
        ues = snapshot_data.get("ues", [])
        aps = snapshot_data.get("aps", [])
        gamma_min_db = float(params.get("gamma_min_db", GAMMA_MIN_DB))
        b_total_mhz = float(params.get("b_total_mhz", B_TOTAL / 1e6))
        N = max(1, int(params.get("N", 1)))
        b_c_mhz = b_total_mhz / N

        active_ues = [u for u in ues if u.get("active", True) and u.get("power_w", 0) > 0]
        outage_cnt = sum(
            1
            for u in ues
            if (not u.get("active", True))
            or (u.get("power_w", 0) <= 0)
            or (u.get("sinr_db", -999.0) < gamma_min_db)
        )
        outage_pct = (outage_cnt / len(ues) * 100.0) if ues else 0.0
        sum_cap = float(sum(u["cap_mbps"] for u in ues))
        sum_se = sum_cap / b_c_mhz if b_c_mhz > 0 else 0.0
        total_tx_w = float(sum(u["power_w"] for u in ues))
        ee_mbit_j = (sum_cap / total_tx_w) if total_tx_w > 0 else 0.0
        active_sinrs = [u["sinr_db"] for u in active_ues]
        avg_sinr = float(np.mean(active_sinrs)) if active_sinrs else 0.0
        p10_sinr = float(np.percentile(active_sinrs, 10)) if active_sinrs else 0.0
        jain = _jain_fairness(np.array([u["cap_mbps"] for u in ues], dtype=float))

        banner = f"""
        <div class="info-banner">
            <div>
                <b>⚡ KPIs Instantâneos do Cenário 3D Atual</b>
                <span>Exibindo a realização atual (M={len(aps)}, K={len(ues)}, modelo={snapshot_data.get('model', 'smb').upper()}, potência={params.get('power_mode', 'full_power')}).
                Para curvas CDF e estatísticas médias, clique em <b>🚀 Executar Simulação Monte Carlo</b> na barra lateral.</span>
            </div>
        </div>
        """

        cards = f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="card-header">
                    <span class="model-badge" style="border-left: 3px solid #38bdf8;">Desempenho Agregado</span>
                    <span class="outage-tag {'safe' if outage_pct < 5.0 else 'warn'}">Outage: {outage_pct:.1f}% ({outage_cnt}/{len(ues)})</span>
                </div>
                <div class="kpi-metric-grid">
                    <div><div class="kpi-label">Vazão Total</div><div class="kpi-value" style="color:#38bdf8;">{sum_cap:.1f} <span class="unit">Mbps</span></div></div>
                    <div><div class="kpi-label">Sum-SE</div><div class="kpi-value">{sum_se:.2f} <span class="unit">bps/Hz</span></div></div>
                    <div><div class="kpi-label">UEs Ativos</div><div class="kpi-value">{len(active_ues)} / {len(ues)}</div></div>
                    <div><div class="kpi-label">Justiça de Jain</div><div class="kpi-value">{jain:.3f}</div></div>
                </div>
            </div>
            <div class="kpi-card">
                <div class="card-header">
                    <span class="model-badge" style="border-left: 3px solid #10b981;">Qualidade de Sinal & Energia</span>
                    <span class="outage-tag safe">N={N} canais</span>
                </div>
                <div class="kpi-metric-grid">
                    <div><div class="kpi-label">SINR Médio (Ativos)</div><div class="kpi-value" style="color:#10b981;">{avg_sinr:+.1f} <span class="unit">dB</span></div></div>
                    <div><div class="kpi-label">SINR 10º Pct (Borda)</div><div class="kpi-value">{p10_sinr:+.1f} <span class="unit">dB</span></div></div>
                    <div><div class="kpi-label">Potência Tx Total</div><div class="kpi-value">{total_tx_w:.3f} <span class="unit">W</span></div></div>
                    <div><div class="kpi-label">Eficiência Energética</div><div class="kpi-value">{ee_mbit_j:.1f} <span class="unit">Mb/J</span></div></div>
                </div>
            </div>
        </div>
        """

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "SINR por UE (dB)",
                "Capacidade de Shannon por UE (Mbps)",
                "Potência de Transmissão por UE (dBm)",
                "Carga e Vazão Agregada por AP",
            ),
            specs=[[{}, {}], [{}, {"secondary_y": True}]],
            vertical_spacing=0.16,
            horizontal_spacing=0.09,
        )

        ue_labels = [f"UE {u['id']}" for u in ues]
        ue_sinr_plot = [
            u["sinr_db"] if u.get("active", True) and u.get("power_w", 0) > 0 else gamma_min_db - 2.0
            for u in ues
        ]
        ue_colors = []
        for u in ues:
            if not u.get("active", True) or u.get("power_w", 0) <= 0:
                ue_colors.append("#ef4444")
            elif u["sinr_db"] < gamma_min_db:
                ue_colors.append("#f97316")
            elif u["sinr_db"] < 0:
                ue_colors.append("#fbbf24")
            else:
                ue_colors.append("#10b981")

        fig.add_trace(
            go.Bar(
                x=ue_labels,
                y=ue_sinr_plot,
                marker_color=ue_colors,
                text=[
                    "OFF" if not u.get("active", True) else f"{u['sinr_db']:+.1f}"
                    for u in ues
                ],
                textposition="auto",
                name="SINR (dB)",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_hline(
            y=gamma_min_db,
            line_width=1.5,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Outage ({gamma_min_db:g} dB)",
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=ue_labels,
                y=[u["cap_mbps"] for u in ues],
                marker_color="#38bdf8",
                text=[f"{u['cap_mbps']:.1f}" for u in ues],
                textposition="auto",
                name="Capacidade (Mbps)",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        p_min_dbm = float(params.get("p_min_dbm", P_MIN_DBM))
        fig.add_trace(
            go.Bar(
                x=ue_labels,
                y=[
                    u["power_dbm"] if u.get("active", True) and u.get("power_w", 0) > 0 else p_min_dbm - 2.0
                    for u in ues
                ],
                marker_color=[
                    "#ef4444" if not u.get("active", True) or u.get("power_w", 0) <= 0 else "#8b5cf6"
                    for u in ues
                ],
                text=[
                    "OFF" if not u.get("active", True) or u.get("power_w", 0) <= 0 else f"{u['power_dbm']:.1f}"
                    for u in ues
                ],
                textposition="auto",
                name="Potência Tx (dBm)",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        ap_labels = [f"AP {a['id']}" for a in aps]
        fig.add_trace(
            go.Bar(
                x=ap_labels,
                y=[a["sum_cap_mbps"] for a in aps],
                name="Vazão (Mbps)",
                marker_color="#10b981",
            ),
            row=2,
            col=2,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=ap_labels,
                y=[a["connected_ues"] for a in aps],
                name="UEs Associados",
                mode="lines+markers",
                line=dict(color="#f59e0b", width=2.5),
                marker=dict(size=8),
            ),
            row=2,
            col=2,
            secondary_y=True,
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0e14",
            plot_bgcolor="#111622",
            font=dict(family="Segoe UI, sans-serif", color="#e2e8f0"),
            margin=dict(l=45, r=45, t=55, b=45),
            height=640,
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5),
        )
        fig.update_xaxes(gridcolor="#1e293b", zerolinecolor="#334155")
        fig.update_yaxes(gridcolor="#1e293b", zerolinecolor="#334155")
        fig.update_yaxes(title_text="SINR (dB)", row=1, col=1)
        fig.update_yaxes(title_text="Mbps", row=1, col=2)
        fig.update_yaxes(title_text="Potência (dBm)", row=2, col=1)
        fig.update_yaxes(title_text="Mbps", row=2, col=2, secondary_y=False)
        fig.update_yaxes(title_text="Nº UEs", row=2, col=2, secondary_y=True)

        chart_div = pyo.plot(
            fig,
            include_plotlyjs=False,
            output_type="div",
            config={"responsive": True, "displayModeBar": True},
        )

        snap_conv_html = ""
        if snapshot_data.get("dual_loop_hist"):
            snap_conv_html = cls._build_dual_loop_convergence_section(
                snapshot_data["dual_loop_hist"], params, section_uid="dl_snap"
            )

        return cls._wrap_document(
            plotly_src=plotly_src,
            params=params,
            title="KPIs do Cenário Atual (Snapshot 3D)",
            subtitle="Métricas por UE e carga dos Pontos de Acesso",
            body_html=f"{banner}{cards}<div class='chart-box'>{chart_div}</div>{snap_conv_html}",
            is_snapshot=True,
        )

    # -----------------------------------------------------------------------
    # Seção 1: Comparativo de Tecnologias de Antena
    # -----------------------------------------------------------------------
    @classmethod
    def _build_antenna_section(cls, data: dict, params: dict) -> str:
        models = list(data.keys())
        b_total_mhz = float(params.get("b_total_mhz", B_TOTAL / 1e6))
        N = max(1, int(params.get("N", 1)))
        K = max(1, int(params.get("K", 8)))
        b_c_mhz = b_total_mhz / N
        p_max_dbm = float(params.get("p_max_dbm", P_MAX_DBM))
        gamma_min_db = float(params.get("gamma_min_db", GAMMA_MIN_DB))
        gamma_tar_db = float(params.get("gamma_tar_db", GAMMA_TAR_INIT))
        gamma_max_db = float(params.get("gamma_max_db", GAMMA_MAX_DB))
        power_mode = str(params.get("power_mode", params.get("power_control", "full_power")))
        gamma_min_lin = 10.0 ** (gamma_min_db / 10.0)
        p_tx_tot_w = (10.0 ** (p_max_dbm / 10.0) / 1000.0) * K

        metrics_rows = []
        cards_html = []

        for m in models:
            r = data[m]
            sinr_vals = np.asarray(r.get("sinr", []), dtype=float)
            cap_vals = np.asarray(r.get("cap", []), dtype=float)
            rx_vals = np.asarray(r.get("power", []), dtype=float)
            sum_cap = float(r.get("sum_cap", 0.0))
            sum_se = sum_cap / b_c_mhz if b_c_mhz > 0 else 0.0
            p_model_tx_w = float(r.get("tx_power", p_tx_tot_w))

            if len(sinr_vals) > 0:
                outage_mask = (sinr_vals <= 0.0) | (sinr_vals < gamma_min_lin)
                outage_rate = float(np.mean(outage_mask) * 100.0)
                sinr_pos = sinr_vals[sinr_vals > 0.0]
                if len(sinr_pos) > 0:
                    sinr_db_arr = 10.0 * np.log10(sinr_pos)
                    s_p10 = float(np.percentile(sinr_db_arr, 10))
                    s_med = float(np.median(sinr_db_arr))
                    s_p90 = float(np.percentile(sinr_db_arr, 90))
                else:
                    s_p10 = s_med = s_p90 = 0.0
            else:
                outage_rate = s_p10 = s_med = s_p90 = 0.0

            c_p10 = float(np.percentile(cap_vals, 10)) if len(cap_vals) > 0 else 0.0
            border_se = c_p10 / b_c_mhz if b_c_mhz > 0 else 0.0
            rx_pos = rx_vals[rx_vals > 0.0]
            rx_mean_dbm = (
                float(10.0 * np.log10(max(float(np.mean(rx_pos)), 1e-18) * 1000.0))
                if len(rx_pos) > 0
                else -150.0
            )
            ee = (sum_cap / p_model_tx_w) if p_model_tx_w > 0 else 0.0
            fairness = _jain_fairness(cap_vals)

            lbl = MODEL_LABELS.get(m, m.capitalize())
            c_accent = cls.ANTENNA_COLOR_MAP.get(m, "#38bdf8")

            metrics_rows.append({
                "key": m,
                "label": lbl,
                "color": c_accent,
                "sum_cap": sum_cap,
                "sum_se": sum_se,
                "s_p10": s_p10,
                "s_med": s_med,
                "s_p90": s_p90,
                "c_p10": c_p10,
                "border_se": border_se,
                "p_tx_w": p_model_tx_w,
                "rx_mean_dbm": rx_mean_dbm,
                "ee": ee,
                "fairness": fairness,
                "outage": outage_rate,
            })

            cards_html.append(f"""
            <div class="kpi-card">
                <div class="card-header">
                    <span class="model-badge" style="border-left: 3px solid {c_accent};">{lbl}</span>
                    <span class="outage-tag {'safe' if outage_rate < 5.0 else 'warn'}">Outage: {outage_rate:.1f}%</span>
                </div>
                <div class="kpi-metric-grid">
                    <div>
                        <div class="kpi-label">Sum-Cap / Sum-SE</div>
                        <div class="kpi-value" style="color:{c_accent};">{sum_cap:.1f} <span class="unit">Mbps</span> <span class="subval">({sum_se:.2f} b/Hz)</span></div>
                    </div>
                    <div>
                        <div class="kpi-label">SINR 10º / 50º / 90º</div>
                        <div class="kpi-value" style="font-size:13px;">{s_p10:+.1f} / {s_med:+.1f} / {s_p90:+.1f} <span class="unit">dB</span></div>
                    </div>
                    <div>
                        <div class="kpi-label">Borda 10º (Cap / SE)</div>
                        <div class="kpi-value">{c_p10:.2f} <span class="unit">Mbps</span> <span class="subval">({border_se:.2f})</span></div>
                    </div>
                    <div>
                        <div class="kpi-label">Efic. Energ. / Jain</div>
                        <div class="kpi-value">{ee:.1f} <span class="unit">Mb/J</span> <span class="subval">(J={fairness:.2f})</span></div>
                    </div>
                </div>
            </div>
            """)

        table_html = cls._render_benchmark_table(metrics_rows, table_id="tbl-antennas", csv_name="kpis_antenas.csv", power_label="Potência Tx (W)")

        # -------------------------------------------------------------------
        # Grid 3x2 de Gráficos Plotly (6 gráficos)
        # -------------------------------------------------------------------
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "1. CDF de SINR por UE (dB)",
                "2. CDF de Capacidade de Shannon por UE (Mbps)",
                "3. CDF de Potência Recebida por UE (dBm)",
                "4. Ganho de SINR 10º Pct (Borda) vs Baseline (dB)",
                "5. Capacidade Normalizada vs Baseline (10º Pct & Soma)",
                "6. Vazão Agregada (Mbps) & Taxa de Outage (%)",
            ),
            specs=[[{}, {}], [{}, {}], [{}, {"secondary_y": True}]],
            vertical_spacing=0.10,
            horizontal_spacing=0.08,
        )

        p = np.linspace(0, 100, 300)

        # 1. CDF de SINR
        for row_m in metrics_rows:
            m = row_m["key"]
            sinr = np.asarray(data[m].get("sinr", []), dtype=float)
            sinr_pos = sinr[sinr > 0.0]
            if len(sinr_pos) > 0:
                q = np.percentile(10.0 * np.log10(sinr_pos), p)
                fig.add_trace(
                    go.Scatter(
                        x=q,
                        y=p / 100.0,
                        mode="lines",
                        name=row_m["label"],
                        legendgroup=m,
                        line=dict(color=row_m["color"], width=2.4),
                        hovertemplate=f"<b>{row_m['label']}</b><br>SINR: %{{x:.1f}} dB<br>Prob: %{{y:.2f}}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

        fig.add_vline(
            x=gamma_min_db,
            line_width=1.2,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Outage ({gamma_min_db:g} dB)",
            annotation_position="top left",
            annotation_font_color="#ef4444",
            annotation_font_size=10,
            row=1,
            col=1,
        )
        if power_mode in ("dpc_fixed", "dual_loop"):
            fig.add_vline(
                x=gamma_tar_db,
                line_width=1.1,
                line_dash="dash",
                line_color="#38bdf8",
                annotation_text=f"γ_tar ({gamma_tar_db:g} dB)",
                annotation_position="top right",
                annotation_font_color="#38bdf8",
                annotation_font_size=9,
                row=1,
                col=1,
            )
        if power_mode == "dual_loop":
            fig.add_vline(
                x=gamma_max_db,
                line_width=1.1,
                line_dash="dash",
                line_color="#10b981",
                annotation_text=f"γ_max ({gamma_max_db:g} dB)",
                annotation_position="top right",
                annotation_font_color="#10b981",
                annotation_font_size=9,
                row=1,
                col=1,
            )
        for y_ref, lbl_ref in ((0.10, "10º Pct"), (0.50, "50º Pct")):
            fig.add_hline(y=y_ref, line_width=1, line_dash="dot", line_color="#64748b", row=1, col=1)
            fig.add_hline(y=y_ref, line_width=1, line_dash="dot", line_color="#64748b", row=1, col=2)
            fig.add_hline(y=y_ref, line_width=1, line_dash="dot", line_color="#64748b", row=2, col=1)

        # 2. CDF de Capacidade
        for row_m in metrics_rows:
            m = row_m["key"]
            cap = np.asarray(data[m].get("cap", []), dtype=float)
            if len(cap) > 0:
                q = np.percentile(cap, p)
                fig.add_trace(
                    go.Scatter(
                        x=q,
                        y=p / 100.0,
                        mode="lines",
                        name=row_m["label"],
                        legendgroup=m,
                        showlegend=False,
                        line=dict(color=row_m["color"], width=2.4),
                    ),
                    row=1,
                    col=2,
                )

        # 3. CDF de Potência Recebida (dBm) — apenas UEs ativos para evitar cauda artificial de 0 W
        for row_m in metrics_rows:
            m = row_m["key"]
            rx = np.asarray(data[m].get("power", []), dtype=float)
            rx_pos = rx[rx > 0.0]
            if len(rx_pos) > 0:
                rx_dbm = 10.0 * np.log10(rx_pos * 1000.0)
                q = np.percentile(rx_dbm, p)
                fig.add_trace(
                    go.Scatter(
                        x=q,
                        y=p / 100.0,
                        mode="lines",
                        name=row_m["label"],
                        legendgroup=m,
                        showlegend=False,
                        line=dict(color=row_m["color"], width=2.4),
                    ),
                    row=2,
                    col=1,
                )

        # Baseline (Omni se presente, senão primeiro modelo)
        base_row = next((r for r in metrics_rows if r["key"] == "omni"), metrics_rows[0])
        labels = [r["label"] for r in metrics_rows]
        colors = [r["color"] for r in metrics_rows]

        # 4. Ganho de SINR 10th Pct vs Baseline (dB)
        sinr_gains = [r["s_p10"] - base_row["s_p10"] for r in metrics_rows]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=sinr_gains,
                marker=dict(color=colors, line=dict(color="#ffffff", width=0.8)),
                text=[f"{v:+.1f} dB" for v in sinr_gains],
                textposition="auto",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        fig.add_hline(y=0.0, line_width=1, line_dash="dash", line_color="#94a3b8", row=2, col=2)

        # 5. Capacidade Normalizada (10º Percentil e Sum-Cap)
        base_c10 = max(base_row["c_p10"], 1e-6)
        base_sc = max(base_row["sum_cap"], 1e-6)
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[r["c_p10"] / base_c10 for r in metrics_rows],
                name="Cap. 10º Pct (Norm.)",
                marker_color="#38bdf8",
                text=[f"{r['c_p10']/base_c10:.2f}×" for r in metrics_rows],
                textposition="auto",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[r["sum_cap"] / base_sc for r in metrics_rows],
                name="Soma-Capacidade (Norm.)",
                marker_color="#10b981",
                text=[f"{r['sum_cap']/base_sc:.2f}×" for r in metrics_rows],
                textposition="auto",
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=1.0, line_width=1, line_dash="dash", line_color="#f59e0b", row=3, col=1)

        # 6. Vazão Agregada (Mbps) & Taxa de Outage (%)
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[r["sum_cap"] for r in metrics_rows],
                name="Sum-Cap (Mbps)",
                marker_color=colors,
                text=[f"{r['sum_cap']:.1f}" for r in metrics_rows],
                textposition="auto",
                showlegend=False,
            ),
            row=3,
            col=2,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=[r["outage"] for r in metrics_rows],
                name="Taxa Outage (%)",
                mode="lines+markers+text",
                text=[f"{r['outage']:.1f}%" for r in metrics_rows],
                textposition="top center",
                line=dict(color="#ef4444", width=2.5),
                marker=dict(size=8, color="#ef4444"),
            ),
            row=3,
            col=2,
            secondary_y=True,
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0e14",
            plot_bgcolor="#111622",
            font=dict(family="Segoe UI, sans-serif", color="#e2e8f0"),
            margin=dict(l=50, r=50, t=65, b=50),
            height=980,
            barmode="group",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.03,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(17, 22, 34, 0.85)",
                bordercolor="#232c3d",
                borderwidth=1,
                font=dict(size=11),
            ),
        )
        fig.update_xaxes(gridcolor="#1e293b", zerolinecolor="#334155")
        fig.update_yaxes(gridcolor="#1e293b", zerolinecolor="#334155")
        fig.update_xaxes(title_text="SINR (dB)", row=1, col=1)
        fig.update_yaxes(title_text="Probabilidade", row=1, col=1)
        fig.update_xaxes(title_text="Capacidade (Mbps)", row=1, col=2)
        fig.update_xaxes(title_text="Potência Recebida (dBm)", row=2, col=1)
        fig.update_yaxes(title_text="Probabilidade", row=2, col=1)
        fig.update_yaxes(title_text="Ganho vs Baseline (dB)", row=2, col=2)
        fig.update_yaxes(title_text="Razão vs Baseline (×)", row=3, col=1)
        fig.update_yaxes(title_text="Sum-Capacity (Mbps)", row=3, col=2, secondary_y=False)
        fig.update_yaxes(title_text="Outage (%)", range=[0, 105], row=3, col=2, secondary_y=True)

        chart_html = pyo.plot(
            fig,
            include_plotlyjs=False,
            output_type="div",
            config={"responsive": True, "displayModeBar": True},
        )

        return f"""
        <div class="kpi-container">{''.join(cards_html)}</div>
        {table_html}
        <div class="chart-box">{chart_html}</div>
        """

    # -----------------------------------------------------------------------
    # Seção 2: Controle de Potência SciChal4A
    # -----------------------------------------------------------------------
    @classmethod
    def _build_scichal4a_section(
        cls,
        stats: dict,
        params: dict,
        has_dedicated_conv_tab: bool = False,
        embedded_conv_html: str = "",
    ) -> str:
        algos = [a for a in cls.ALGO_ORDER if a in stats] or [
            k for k in stats.keys() if k != "dual_loop_hist"
        ]
        b_total_mhz = float(params.get("b_total_mhz", B_TOTAL / 1e6))
        N = max(1, int(params.get("N", 1)))
        b_c_mhz = b_total_mhz / N
        p_min_dbm = float(params.get("p_min_dbm", P_MIN_DBM))
        gamma_min_db = float(params.get("gamma_min_db", GAMMA_MIN_DB))
        gamma_tar_db = float(params.get("gamma_tar_db", GAMMA_TAR_INIT))
        gamma_max_db = float(params.get("gamma_max_db", GAMMA_MAX_DB))
        gamma_min_lin = 10.0 ** (gamma_min_db / 10.0)

        metrics_rows = []
        cards_html = []

        for a in algos:
            st = stats.get(a, {})
            se_vals = np.asarray(st.get("se", []), dtype=float)
            cap_vals = se_vals * b_c_mhz
            sinr_vals = np.asarray(st.get("sinr_lin", []), dtype=float)
            pw_vals = np.asarray(st.get("power_dist", []), dtype=float)

            # sum_cap em simular_scichal4a está em bps -> converte para Mbps
            raw_sum_cap = np.asarray(st.get("sum_cap", [0.0]), dtype=float)
            sum_cap_mbps = float(np.mean(raw_sum_cap)) / 1e6
            sum_se = sum_cap_mbps / b_c_mhz if b_c_mhz > 0 else 0.0

            p_mean_w = float(np.mean(st.get("power", [0.0])))
            ee = (sum_cap_mbps / p_mean_w) if p_mean_w > 0 else 0.0

            out_vals = st.get("outage", [])
            if len(out_vals) > 0:
                out_rate = float(np.mean(out_vals)) * 100.0
            elif len(sinr_vals) > 0:
                out_rate = float(np.mean((sinr_vals <= 0.0) | (sinr_vals < gamma_min_lin)) * 100.0)
            else:
                out_rate = 0.0

            sinr_pos = sinr_vals[sinr_vals > 0.0] if a == "dual_loop" else np.maximum(sinr_vals, 1e-12)
            if len(sinr_pos) > 0:
                s_db = 10.0 * np.log10(sinr_pos)
                s_p10 = float(np.percentile(s_db, 10))
                s_med = float(np.median(s_db))
                s_p90 = float(np.percentile(s_db, 90))
            else:
                s_p10 = s_med = s_p90 = 0.0

            c_p10 = float(np.percentile(cap_vals, 10)) if len(cap_vals) > 0 else 0.0
            border_se = float(np.percentile(se_vals, 10)) if len(se_vals) > 0 else 0.0
            fairness = _jain_fairness(cap_vals)

            lbl = cls.ALGO_LABELS.get(a, a)
            col = cls.ALGO_COLORS.get(a, "#38bdf8")

            metrics_rows.append({
                "key": a,
                "label": lbl,
                "color": col,
                "sum_cap": sum_cap_mbps,
                "sum_se": sum_se,
                "s_p10": s_p10,
                "s_med": s_med,
                "s_p90": s_p90,
                "c_p10": c_p10,
                "border_se": border_se,
                "p_tx_w": p_mean_w,
                "ee": ee,
                "fairness": fairness,
                "outage": out_rate,
            })

            cards_html.append(f"""
            <div class="kpi-card">
                <div class="card-header">
                    <span class="model-badge" style="border-left: 3px solid {col};">{lbl}</span>
                    <span class="outage-tag {'safe' if out_rate < 5.0 else 'warn'}">Outage: {out_rate:.1f}%</span>
                </div>
                <div class="kpi-metric-grid">
                    <div>
                        <div class="kpi-label">Sum-Cap / Sum-SE</div>
                        <div class="kpi-value" style="color:{col};">{sum_cap_mbps:.1f} <span class="unit">Mbps</span> <span class="subval">({sum_se:.2f} b/Hz)</span></div>
                    </div>
                    <div>
                        <div class="kpi-label">Border SE (10º Pct)</div>
                        <div class="kpi-value">{border_se:.2f} <span class="unit">bps/Hz</span> <span class="subval">({c_p10:.1f} Mbps)</span></div>
                    </div>
                    <div>
                        <div class="kpi-label">SINR 10º / 50º / 90º</div>
                        <div class="kpi-value" style="font-size:13px;">{s_p10:+.1f} / {s_med:+.1f} / {s_p90:+.1f} <span class="unit">dB</span></div>
                    </div>
                    <div>
                        <div class="kpi-label">Efic. Energ. / Potência</div>
                        <div class="kpi-value">{ee:.1f} <span class="unit">Mb/J</span> <span class="subval">({p_mean_w:.2f} W)</span></div>
                    </div>
                </div>
            </div>
            """)

        table_html = cls._render_benchmark_table(metrics_rows, table_id="tbl-scichal", csv_name="kpis_controle_potencia.csv", power_label="Potência Média (W)")

        # -------------------------------------------------------------------
        # Grid 3x2 de Gráficos Plotly
        # -------------------------------------------------------------------
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "1. CDF de SINR por UE (dB)",
                "2. CDF de Capacidade Física por UE (Mbps)",
                "3. CDF de Potência de Transmissão por UE (dBm)",
                "4. Border SE (10º Pct), Sum-SE & Outage (%)",
                "5. Eficiência Energética (Log) & Potência Média (W)",
                "6. Convergência Dual-Loop — Potência Tx (Loop Interno)",
            ),
            specs=[
                [{}, {}],
                [{}, {"secondary_y": True}],
                [{"secondary_y": True}, {}],
            ],
            vertical_spacing=0.10,
            horizontal_spacing=0.08,
        )

        p = np.linspace(0, 100, 300)

        # 1. CDF de SINR (Full Power & DPC Fixo completos; Dual-Loop nos ativos >= gamma_min)
        for row_m in metrics_rows:
            a = row_m["key"]
            sinr = np.asarray(stats.get(a, {}).get("sinr_lin", []), dtype=float)
            if a == "dual_loop":
                sinr_plot = sinr[sinr > 0.0]
            else:
                sinr_plot = np.maximum(sinr, 1e-12)
            if len(sinr_plot) > 0:
                q = np.percentile(10.0 * np.log10(sinr_plot), p)
                fig.add_trace(
                    go.Scatter(
                        x=q,
                        y=p / 100.0,
                        mode="lines",
                        name=row_m["label"],
                        legendgroup=a,
                        line=dict(color=row_m["color"], width=2.5),
                    ),
                    row=1,
                    col=1,
                )

        for x_ref, c_ref, txt_ref in (
            (gamma_min_db, "#ef4444", f"γ_min ({gamma_min_db:g})"),
            (gamma_tar_db, "#38bdf8", f"γ_tar ({gamma_tar_db:g})"),
            (gamma_max_db, "#10b981", f"γ_max ({gamma_max_db:g})"),
        ):
            fig.add_vline(
                x=x_ref,
                line_width=1.2,
                line_dash="dash",
                line_color=c_ref,
                annotation_text=txt_ref,
                annotation_font_color=c_ref,
                annotation_font_size=9,
                row=1,
                col=1,
            )

        # 2. CDF de Capacidade Física (Mbps)
        for row_m in metrics_rows:
            a = row_m["key"]
            cap = np.asarray(stats.get(a, {}).get("se", []), dtype=float) * b_c_mhz
            if len(cap) > 0:
                q = np.percentile(cap, p)
                fig.add_trace(
                    go.Scatter(
                        x=q,
                        y=p / 100.0,
                        mode="lines",
                        name=row_m["label"],
                        legendgroup=a,
                        showlegend=False,
                        line=dict(color=row_m["color"], width=2.5),
                    ),
                    row=1,
                    col=2,
                )

        # 3. CDF de Potência Tx (dBm; UEs desligados em 0 W plotados em p_min_dbm - 4 para evidenciar bloqueio)
        off_sentinel_dbm = p_min_dbm - 4.0
        for row_m in metrics_rows:
            a = row_m["key"]
            pw = np.asarray(stats.get(a, {}).get("power_dist", []), dtype=float)
            if len(pw) > 0:
                pw_dbm = np.where(
                    pw > 0.0,
                    10.0 * np.log10(np.maximum(pw, 1e-12) * 1000.0),
                    off_sentinel_dbm,
                )
                q = np.percentile(pw_dbm, p)
                fig.add_trace(
                    go.Scatter(
                        x=q,
                        y=p / 100.0,
                        mode="lines",
                        name=row_m["label"],
                        legendgroup=a,
                        showlegend=False,
                        line=dict(color=row_m["color"], width=2.5),
                    ),
                    row=2,
                    col=1,
                )
        fig.add_vline(
            x=p_min_dbm,
            line_width=1,
            line_dash="dot",
            line_color="#94a3b8",
            annotation_text=f"P_min ({p_min_dbm:g} dBm)",
            annotation_font_size=9,
            row=2,
            col=1,
        )

        labels = [r["label"] for r in metrics_rows]
        colors = [r["color"] for r in metrics_rows]

        # 4. Border SE, Sum-SE & Outage
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[r["border_se"] for r in metrics_rows],
                name="Border SE 10º [bps/Hz]",
                marker_color="#38bdf8",
                text=[f"{r['border_se']:.2f}" for r in metrics_rows],
                textposition="auto",
            ),
            row=2,
            col=2,
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[r["sum_se"] for r in metrics_rows],
                name="Sum-SE [bps/Hz]",
                marker_color="#10b981",
                text=[f"{r['sum_se']:.2f}" for r in metrics_rows],
                textposition="auto",
            ),
            row=2,
            col=2,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=[r["outage"] for r in metrics_rows],
                name="Outage [%]",
                mode="lines+markers+text",
                text=[f"{r['outage']:.1f}%" for r in metrics_rows],
                textposition="top center",
                line=dict(color="#ef4444", width=2.5),
                marker=dict(size=8, color="#ef4444"),
            ),
            row=2,
            col=2,
            secondary_y=True,
        )

        # 5. Eficiência Energética (Log) & Potência Média
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[max(r["ee"], 1e-3) for r in metrics_rows],
                name="Efic. Energética [Mb/J]",
                marker_color=colors,
                text=[f"{r['ee']:.1f}" for r in metrics_rows],
                textposition="auto",
                showlegend=False,
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=[r["p_tx_w"] for r in metrics_rows],
                name="Potência Média [W]",
                mode="lines+markers+text",
                text=[f"{r['p_tx_w']:.2f}W" for r in metrics_rows],
                textposition="top center",
                line=dict(color="#f59e0b", width=2.5),
                marker=dict(size=8, color="#f59e0b"),
            ),
            row=3,
            col=1,
            secondary_y=True,
        )

        # 6. Convergência do Dual-Loop (Potência Tx) + link para o Painel Completo
        hist_list = stats.get("dual_loop_hist", [])
        extra_conv_html = embedded_conv_html
        if hist_list:
            h = hist_list[0]
            pw_hist = np.asarray(h.get("power_w", []))
            if pw_hist.ndim == 2 and pw_hist.shape[0] > 0:
                T_steps, K_ues = pw_hist.shape
                it_inner = np.arange(T_steps)
                p_max_ref = float(h.get("p_max_dbm", params.get("p_max_dbm", P_MAX_DBM)))
                for k_idx in range(min(K_ues, 8)):
                    pw_col = pw_hist[:, k_idx]
                    p_dbm_vals = [
                        float(10.0 * np.log10(v * 1000.0)) if v > 0.0 else None
                        for v in pw_col
                    ]
                    fig.add_trace(
                        go.Scatter(
                            x=it_inner,
                            y=p_dbm_vals,
                            mode="lines",
                            name=f"UE {k_idx + 1}",
                            line=dict(width=1.8),
                            showlegend=False,
                        ),
                        row=3,
                        col=2,
                    )
                    off_idx = np.flatnonzero(pw_col <= 0.0)
                    if len(off_idx) > 0 and off_idx[0] > 0:
                        t_last = int(off_idx[0] - 1)
                        p_last = float(10.0 * np.log10(max(pw_col[t_last], 1e-12) * 1000.0))
                        fig.add_trace(
                            go.Scatter(
                                x=[t_last],
                                y=[p_last],
                                mode="markers",
                                marker=dict(symbol="x", size=8, color="#ef4444"),
                                name=f"UE {k_idx + 1} OFF",
                                showlegend=False,
                                hovertemplate=f"<b>UE {k_idx + 1} desligado (Outage)</b><br>Passo {t_last}<extra></extra>",
                            ),
                            row=3,
                            col=2,
                        )
                for y_lim, c_lim, lbl_lim in (
                    (p_max_ref, "#38bdf8", f"P_max ({p_max_ref:g})"),
                    (p_min_dbm, "#94a3b8", f"P_min ({p_min_dbm:g})"),
                ):
                    fig.add_hline(
                        y=y_lim,
                        line_width=1,
                        line_dash="dot",
                        line_color=c_lim,
                        annotation_text=lbl_lim,
                        annotation_font_color=c_lim,
                        annotation_font_size=9,
                        row=3,
                        col=2,
                    )
                for mk in h.get("outer_marks", [])[:25]:
                    fig.add_vline(
                        x=mk - 0.5,
                        line_width=0.7,
                        line_dash="dot",
                        line_color="#475569",
                        row=3,
                        col=2,
                    )

            if not embedded_conv_html and not has_dedicated_conv_tab:
                # Fallback se chamado diretamente sem painel completo
                tgt_hist = np.asarray(h.get("target_db", []))
                if tgt_hist.ndim == 2 and tgt_hist.shape[0] > 0:
                    fig_outer = go.Figure()
                    it_outer = np.arange(tgt_hist.shape[0])
                    for k_idx in range(min(tgt_hist.shape[1], 8)):
                        fig_outer.add_trace(
                            go.Scatter(
                                x=it_outer,
                                y=tgt_hist[:, k_idx],
                                mode="lines+markers",
                                line_shape="hv",
                                name=f"UE {k_idx + 1}",
                                line=dict(width=2),
                            )
                        )
                    fig_outer.add_hline(
                        y=gamma_max_db,
                        line_dash="dash",
                        line_color="#10b981",
                        annotation_text=f"Teto QAM ({gamma_max_db:g} dB)",
                    )
                    fig_outer.add_hline(
                        y=gamma_min_db,
                        line_dash="dash",
                        line_color="#ef4444",
                        annotation_text=f"Outage ({gamma_min_db:g} dB)",
                    )
                    fig_outer.update_layout(
                        title="Negociação do Alvo de SINR γ_tar (Loop Externo Dual-Loop, Snapshot 0)",
                        template="plotly_dark",
                        paper_bgcolor="#0b0e14",
                        plot_bgcolor="#111622",
                        font=dict(family="Segoe UI, sans-serif", color="#e2e8f0"),
                        height=320,
                        margin=dict(l=50, r=30, t=45, b=40),
                        xaxis_title="Iteração do Loop Externo",
                        yaxis_title="γ_tar (dB)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    )
                    fig_outer.update_xaxes(gridcolor="#1e293b")
                    fig_outer.update_yaxes(gridcolor="#1e293b")
                    outer_div = pyo.plot(
                        fig_outer,
                        include_plotlyjs=False,
                        output_type="div",
                        config={"responsive": True, "displayModeBar": True},
                    )
                    extra_conv_html = f"<div class='chart-box' style='margin-top:14px;'>{outer_div}</div>"

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0e14",
            plot_bgcolor="#111622",
            font=dict(family="Segoe UI, sans-serif", color="#e2e8f0"),
            margin=dict(l=50, r=50, t=65, b=50),
            height=980,
            barmode="group",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.03,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(17, 22, 34, 0.85)",
                bordercolor="#232c3d",
                borderwidth=1,
                font=dict(size=11),
            ),
        )
        fig.update_xaxes(gridcolor="#1e293b", zerolinecolor="#334155")
        fig.update_yaxes(gridcolor="#1e293b", zerolinecolor="#334155")
        fig.update_xaxes(title_text="SINR (dB)", row=1, col=1)
        fig.update_yaxes(title_text="Probabilidade", row=1, col=1)
        fig.update_xaxes(title_text="Capacidade (Mbps)", row=1, col=2)
        fig.update_xaxes(title_text=f"Potência Tx (dBm) [<{p_min_dbm:g} = OFF]", row=2, col=1)
        fig.update_yaxes(title_text="Probabilidade", row=2, col=1)
        fig.update_yaxes(title_text="Eficiência Espectral [bps/Hz]", row=2, col=2, secondary_y=False)
        fig.update_yaxes(title_text="Outage [%]", range=[0, 105], row=2, col=2, secondary_y=True)
        fig.update_yaxes(title_text="Mbits / Joule (Log)", type="log", row=3, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Potência Média [W]", row=3, col=1, secondary_y=True)
        fig.update_xaxes(title_text="Passo do Loop Interno", row=3, col=2)
        fig.update_yaxes(title_text="Potência Tx (dBm)", row=3, col=2)

        chart_html = pyo.plot(
            fig,
            include_plotlyjs=False,
            output_type="div",
            config={"responsive": True, "displayModeBar": True},
        )

        cta_html = ""
        if hist_list and (has_dedicated_conv_tab or embedded_conv_html):
            cta_html = """
            <div style="display:flex; justify-content:flex-end; margin: 10px 0 4px 0;">
                <button class="csv-btn" style="background:#0284c7; border-color:#38bdf8; padding:7px 14px;" onclick="openDualLoopPanel()">🔬 Abrir Convergência Temporal (Potência Tx &amp; SINR)</button>
            </div>
            """

        return f"""
        <div class="kpi-container">{''.join(cards_html)}</div>
        {table_html}
        {cta_html}
        <div class="chart-box">{chart_html}</div>
        {extra_conv_html}
        """

    # -----------------------------------------------------------------------
    # Tabela Benchmark Comparativa + Exportação CSV
    # -----------------------------------------------------------------------
    @classmethod
    def _render_benchmark_table(
        cls, rows: list, table_id: str, csv_name: str, power_label: str
    ) -> str:
        if not rows:
            return ""

        best_cap = max(r["sum_cap"] for r in rows)
        best_p10 = max(r["s_p10"] for r in rows)
        best_ee = max(r["ee"] for r in rows)
        best_fair = max(r["fairness"] for r in rows)
        best_out = min(r["outage"] for r in rows)

        body_rows = []
        for r in rows:
            def _hl(val, target, fmt):
                txt = fmt.format(val)
                if abs(val - target) < 1e-6:
                    return f"<span class='best-cell'>★ {txt}</span>"
                return txt

            body_rows.append(f"""
            <tr>
                <td style="text-align:left; font-weight:bold;"><span class="dot" style="background:{r['color']};"></span>{r['label']}</td>
                <td>{_hl(r['sum_cap'], best_cap, '{:.2f}')}</td>
                <td>{r['sum_se']:.2f}</td>
                <td>{_hl(r['s_p10'], best_p10, '{:+.1f}')}</td>
                <td>{r['s_med']:+.1f}</td>
                <td>{r['s_p90']:+.1f}</td>
                <td>{r['c_p10']:.2f}</td>
                <td>{r['p_tx_w']:.3f}</td>
                <td>{_hl(r['ee'], best_ee, '{:.1f}')}</td>
                <td>{_hl(r['fairness'], best_fair, '{:.3f}')}</td>
                <td>{_hl(r['outage'], best_out, '{:.1f}%')}</td>
            </tr>
            """)

        return f"""
        <div class="table-card">
            <div class="table-header">
                <h4>📋 Tabela Benchmark Comparativa</h4>
                <button class="csv-btn" onclick="exportKpiCsv('{table_id}', '{csv_name}')">📥 Baixar CSV</button>
            </div>
            <div style="overflow-x:auto;">
                <table id="{table_id}" class="bench-table">
                    <thead>
                        <tr>
                            <th style="text-align:left;">Modelo / Algoritmo</th>
                            <th>Sum-Cap (Mbps)</th>
                            <th>Sum-SE (bps/Hz)</th>
                            <th>SINR 10º (dB)</th>
                            <th>SINR 50º (dB)</th>
                            <th>SINR 90º (dB)</th>
                            <th>Cap. 10º (Mbps)</th>
                            <th>{power_label}</th>
                            <th>Efic. Energ. (Mb/J)</th>
                            <th>Justiça Jain</th>
                            <th>Outage</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(body_rows)}
                    </tbody>
                </table>
            </div>
        </div>
        """

    @classmethod
    def _build_combined_tabs(
        cls, ant_html: str, sci_html: str, conv_html: str = ""
    ) -> str:
        conv_btn = (
            '<button id="subtab-dl-btn" class="subtab-btn" onclick="switchSubTab(2)">🔬 Convergência Temporal Dual-Loop (Potência Tx &amp; SINR)</button>'
            if conv_html
            else ""
        )
        conv_pane = (
            f'<div id="subpanel-2" class="subtab-pane">{conv_html}</div>'
            if conv_html
            else ""
        )
        return f"""
        <div class="subtab-bar">
            <button class="subtab-btn active" onclick="switchSubTab(0)">📡 Comparativo de Antenas</button>
            <button class="subtab-btn" onclick="switchSubTab(1)">⚡ Controle de Potência (SciChal4A)</button>
            {conv_btn}
        </div>
        <div id="subpanel-0" class="subtab-pane active">{ant_html}</div>
        <div id="subpanel-1" class="subtab-pane">{sci_html}</div>
        {conv_pane}
        """

    @classmethod
    def _prepare_dl_snapshots(cls, hist_list: list, params: dict) -> dict:
        """Normaliza realizações temporais do Dual-Loop (Potência Tx e SINR) para JSON enxuto."""
        snaps = []

        p_max_def = float(params.get("p_max_dbm", P_MAX_DBM))
        p_min_def = float(params.get("p_min_dbm", P_MIN_DBM))
        mu_def = float(params.get("mu_step_db", MU_STEP))
        gmin_def = float(params.get("gamma_min_db", GAMMA_MIN_DB))
        gmax_def = float(params.get("gamma_max_db", GAMMA_MAX_DB))

        max_inner_pts = 400 if len(hist_list) > 2 else 550

        for idx, h in enumerate(hist_list):
            if not isinstance(h, dict):
                continue
            pw = np.asarray(h.get("power_w", []), dtype=float)
            if pw.ndim != 2 or pw.shape[0] == 0 or pw.shape[1] == 0:
                continue
            T, K = pw.shape
            sinr_lin = np.asarray(h.get("sinr_lin", []), dtype=float)
            if sinr_lin.shape != (T, K):
                sinr_lin = np.zeros((T, K), dtype=float)

            tgt = np.asarray(h.get("target_db", []), dtype=float)
            outer_marks = sorted({int(m) for m in h.get("outer_marks", []) if 0 < int(m) <= T})

            p_max_dbm = float(h.get("p_max_dbm", p_max_def))
            p_min_dbm = float(h.get("p_min_dbm", p_min_def))
            mu_step = float(h.get("mu_step", mu_def))
            gamma_min_db = float(h.get("gamma_min_db", gmin_def))
            gamma_max_db = float(h.get("gamma_max_db", gmax_def))

            if tgt.ndim != 2 or tgt.shape[1] != K:
                tgt = np.full((1, K), float(params.get("gamma_tar_db", GAMMA_TAR_INIT)))

            # Passo exato do desligamento por UE (se ocorreu)
            t_off_list = []
            for k in range(K):
                z_idx = np.flatnonzero(pw[:, k] <= 0.0)
                t_off_list.append(int(z_idx[0]) if len(z_idx) > 0 else -1)

            # Preserva pontos críticos e reamostra se T > max_inner_pts
            if T > max_inner_pts:
                crit = {0, T - 1}
                for m in outer_marks:
                    crit.add(max(0, min(T - 1, m - 1)))
                    crit.add(max(0, min(T - 1, m)))
                for t_off in t_off_list:
                    if t_off >= 0:
                        crit.add(max(0, t_off - 1))
                        crit.add(t_off)
                uniform_pts = set(np.linspace(0, T - 1, max_inner_pts, dtype=int).tolist())
                kept_idx = np.array(sorted(crit | uniform_pts), dtype=int)
            else:
                kept_idx = np.arange(T, dtype=int)

            # Mapeamento do alvo γ_tar vigente em cada passo interno mantido
            if outer_marks:
                outer_idx_map = np.minimum(
                    np.searchsorted(outer_marks, kept_idx, side="right"),
                    tgt.shape[0] - 1,
                )
            else:
                outer_idx_map = np.zeros(len(kept_idx), dtype=int)

            p_dbm_rows = []
            sinr_db_rows = []
            tgt_step_rows = []
            for pos, t_val in enumerate(kept_idx):
                p_row = []
                s_row = []
                tg_row = []
                m_idx = int(outer_idx_map[pos])
                for k in range(K):
                    pv = pw[t_val, k]
                    if pv > 0.0:
                        p_row.append(round(float(10.0 * np.log10(pv * 1000.0)), 3))
                        sv = sinr_lin[t_val, k]
                        if sv > 0.0:
                            s_row.append(round(float(10.0 * np.log10(sv)), 2))
                        else:
                            s_row.append(None)
                    else:
                        p_row.append(None)
                        s_row.append(None)
                    tg_row.append(round(float(tgt[m_idx, k]), 2))
                p_dbm_rows.append(p_row)
                sinr_db_rows.append(s_row)
                tgt_step_rows.append(tg_row)

            # Resumo final dos UEs desta realização
            final_pw = pw[-1]
            final_act = final_pw > 0.0
            n_act = int(np.sum(final_act))
            n_off = int(K - n_act)
            n_folga = 0
            final_p_dbm = []
            final_s_db = []
            err_list = []
            act_tgts = []
            for k in range(K):
                if final_act[k]:
                    pdbm = float(10.0 * np.log10(final_pw[k] * 1000.0))
                    sdb = float(10.0 * np.log10(max(sinr_lin[-1, k], 1e-12)))
                    tval = float(tgt[-1, k])
                    final_p_dbm.append(round(pdbm, 3))
                    final_s_db.append(round(sdb, 2))
                    err_list.append(abs(sdb - tval))
                    act_tgts.append(tval)
                    if pdbm < p_max_dbm - 1e-3:
                        n_folga += 1
                else:
                    final_p_dbm.append(None)
                    final_s_db.append(None)

            p_init_w = float(np.sum(pw[0]))
            p_final_w = float(np.sum(final_pw))
            savings_pct = ((1.0 - p_final_w / p_init_w) * 100.0) if p_init_w > 0.0 else 0.0
            n_outer = max(1, len(outer_marks))
            avg_sinr_act = float(np.mean([v for v in final_s_db if v is not None])) if n_act > 0 else 0.0
            mean_err = float(np.mean(err_list)) if err_list else 0.0

            snap_num = int(h.get("snapshot", idx)) + 1
            model_tag = str(h.get("model", params.get("model", "adaptive"))).upper()
            label = (
                f"Realização #{snap_num} ({model_tag}) — "
                f"{n_act}/{K} ativos, P_final={p_final_w:.2f}W (-{savings_pct:.0f}%), {T} passos"
            )

            snaps.append({
                "index": idx,
                "label": label,
                "K": int(K),
                "T": int(T),
                "n_outer": len(outer_marks),
                "avg_inner": round(T / n_outer, 1),
                "step_t": kept_idx.tolist(),
                "outer_marks": outer_marks,
                "t_off": t_off_list,
                "p_max_dbm": round(p_max_dbm, 3),
                "p_min_dbm": round(p_min_dbm, 3),
                "mu_step": round(mu_step, 3),
                "gamma_min_db": round(gamma_min_db, 2),
                "gamma_max_db": round(gamma_max_db, 2),
                "p_dbm": p_dbm_rows,
                "sinr_db": sinr_db_rows,
                "tgt_step_db": tgt_step_rows,
                "final_p_dbm": final_p_dbm,
                "summary": {
                    "n_active": n_act,
                    "n_off": n_off,
                    "n_folga": n_folga,
                    "n_at_max": n_act - n_folga,
                    "p_init_w": round(p_init_w, 3),
                    "p_final_w": round(p_final_w, 3),
                    "savings_pct": round(savings_pct, 1),
                    "avg_sinr_db": round(avg_sinr_act, 2),
                    "mean_err_db": round(mean_err, 2),
                    "tgt_min": round(min(act_tgts), 1) if act_tgts else round(gamma_min_db, 1),
                    "tgt_max": round(max(act_tgts), 1) if act_tgts else round(gamma_min_db, 1),
                },
            })

        return {"snapshots": snaps}

    @classmethod
    def _build_dual_loop_convergence_section(
        cls, hist_list: list, params: dict, section_uid: str = "dl"
    ) -> str:
        """Constrói painel interativo focado na convergência temporal de Potência Tx e SINR."""
        payload = cls._prepare_dl_snapshots(hist_list, params)
        if not payload["snapshots"]:
            return ""

        options_html = "".join(
            f'<option value="{s["index"]}">{s["label"]}</option>'
            for s in payload["snapshots"]
        )
        json_data = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
        uid = section_uid

        html = f"""
        <div id="dl-conv-anchor" class="dl-conv-section">
            <div class="info-banner" style="border-color:#f97316; background:rgba(249,115,22,0.10);">
                <div>
                    <b style="color:#fb923c;">🔬 Convergência Temporal Dual-Loop — Potência de TX &amp; SINR</b>
                    <span>Evolução passo a passo da potência de transmissão (±μ dB) e da SINR perseguindo o alvo dinâmico γ_tar ao longo das iterações internas e externas, com zoom horizontal sincronizado e marcação de cortes por Outage (0 W).</span>
                </div>
            </div>

            <div class="dl-toolbar">
                <div class="dl-ctrl-group" style="flex: 2; min-width: 240px;">
                    <label for="{uid}-snap">Realização / Cenário:</label>
                    <select id="{uid}-snap">{options_html}</select>
                </div>
                <div class="dl-ctrl-group">
                    <label for="{uid}-scale">Escala de Potência:</label>
                    <select id="{uid}-scale">
                        <option value="dbm" selected>dBm (Escala Log — Passos ±μ dB)</option>
                        <option value="mw">mW (Escala Linear — Economia Física)</option>
                    </select>
                </div>
                <div class="dl-ctrl-group">
                    <label for="{uid}-filter">Filtro de Usuários:</label>
                    <select id="{uid}-filter">
                        <option value="all" selected>Todos os UEs</option>
                        <option value="active">Apenas UEs Ativos (Convergidos)</option>
                        <option value="off">Apenas UEs Desligados (Outage)</option>
                    </select>
                </div>
            </div>

            <div id="{uid}-cards" class="dl-kpi-grid"></div>

            <div class="chart-box" style="margin-bottom:14px;">
                <div id="{uid}-chart-power" class="dl-plot" style="height:410px;"></div>
            </div>

            <div class="chart-box">
                <div id="{uid}-chart-sinr" class="dl-plot" style="height:410px;"></div>
            </div>
        </div>
        """

        script = """
        <script>
        (function() {
            const UID = "__UID__";
            const DATA = __JSON__;
            const PALETTE = [
                "#38bdf8", "#f97316", "#10b981", "#f43f5e",
                "#a855f7", "#eab308", "#06b6d4", "#ec4899",
                "#84cc16", "#6366f1", "#fb7185", "#14b8a6"
            ];
            const elSnap = document.getElementById(UID + "-snap");
            const elScale = document.getElementById(UID + "-scale");
            const elFilter = document.getElementById(UID + "-filter");
            const elCards = document.getElementById(UID + "-cards");
            const powerPlotId = UID + "-chart-power";
            const sinrPlotId = UID + "-chart-sinr";
            const plotIds = [powerPlotId, sinrPlotId];
            let zoomBound = false;
            let syncingZoom = false;

            function baseLayout(title, xTitle, yTitle, extra) {
                const layout = {
                    title: { text: title, font: { size: 13.5, color: "#f8fafc" }, x: 0.02, xanchor: "left" },
                    template: "plotly_dark",
                    paper_bgcolor: "#0b0e14",
                    plot_bgcolor: "#111622",
                    font: { family: "Segoe UI, system-ui, sans-serif", color: "#e2e8f0", size: 11 },
                    margin: { l: 54, r: 28, t: 54, b: 44 },
                    xaxis: { title: xTitle, gridcolor: "#1e293b", zerolinecolor: "#334155" },
                    yaxis: { title: yTitle, gridcolor: "#1e293b", zerolinecolor: "#334155" },
                    legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1.0, font: { size: 10 }, bgcolor: "rgba(17,22,34,0.75)" },
                    hovermode: "closest"
                };
                if (extra) Object.assign(layout, extra);
                return layout;
            }

            function selectUeIndices(snap, filterMode) {
                const K = snap.K;
                const offSet = [];
                const actSet = [];
                for (let k = 0; k < K; k++) {
                    if (snap.final_p_dbm[k] === null) offSet.push(k);
                    else actSet.push(k);
                }
                let list = filterMode === "active" ? actSet : (filterMode === "off" ? offSet : [...Array(K).keys()]);
                if (list.length > 20) {
                    const picked = new Set();
                    for (const k of list) {
                        if (snap.final_p_dbm[k] === null) picked.add(k);
                    }
                    const stride = list.length / 20;
                    for (let i = 0; i < list.length && picked.size < 20; i += stride) {
                        picked.add(list[Math.floor(i)]);
                    }
                    list = Array.from(picked).sort((a, b) => a - b);
                }
                return list;
            }

            function renderCards(snap) {
                const sm = snap.summary;
                const sinrText = sm.n_active > 0 ? ((sm.avg_sinr_db >= 0 ? "+" : "") + sm.avg_sinr_db.toFixed(1) + " dB") : "N/A";
                const tgtRange = sm.n_active > 0 ? `[${sm.tgt_min.toFixed(0)}, ${sm.tgt_max.toFixed(0)}] dB` : "Todos OFF";
                const outPct = snap.K > 0 ? (sm.n_off / snap.K * 100).toFixed(1) : "0.0";
                elCards.innerHTML = `
                    <div class="kpi-card">
                        <div class="card-header">
                            <span class="model-badge" style="border-left:3px solid #38bdf8;">Tempo de Convergência</span>
                            <span class="outage-tag safe">K=${snap.K} UEs</span>
                        </div>
                        <div class="kpi-metric-grid">
                            <div><div class="kpi-label">Passos Loop Interno</div><div class="kpi-value" style="color:#38bdf8;">${snap.T}</div></div>
                            <div><div class="kpi-label">Rodadas Loop Externo</div><div class="kpi-value">${snap.n_outer}</div></div>
                            <div><div class="kpi-label">Média Passos / Ext</div><div class="kpi-value">${snap.avg_inner}</div></div>
                            <div><div class="kpi-label">Passo μ</div><div class="kpi-value">±${snap.mu_step} <span class="unit">dB</span></div></div>
                        </div>
                    </div>
                    <div class="kpi-card">
                        <div class="card-header">
                            <span class="model-badge" style="border-left:3px solid #f97316;">Economia de Potência Tx</span>
                            <span class="outage-tag safe">-${sm.savings_pct.toFixed(1)}%</span>
                        </div>
                        <div class="kpi-metric-grid">
                            <div><div class="kpi-label">Potência Inicial (P_max)</div><div class="kpi-value">${sm.p_init_w.toFixed(3)} <span class="unit">W</span></div></div>
                            <div><div class="kpi-label">Potência Convergida</div><div class="kpi-value" style="color:#f97316;">${sm.p_final_w.toFixed(3)} <span class="unit">W</span></div></div>
                            <div><div class="kpi-label">Potência Economizada</div><div class="kpi-value" style="color:#10b981;">${Math.max(0, sm.p_init_w - sm.p_final_w).toFixed(3)} <span class="unit">W</span></div></div>
                            <div><div class="kpi-label">Faixa P_tx Permitida</div><div class="kpi-value">[${snap.p_min_dbm}, ${snap.p_max_dbm}] <span class="unit">dBm</span></div></div>
                        </div>
                    </div>
                    <div class="kpi-card">
                        <div class="card-header">
                            <span class="model-badge" style="border-left:3px solid #10b981;">Estado Final dos UEs</span>
                            <span class="outage-tag ${sm.n_off > 0 ? 'warn' : 'safe'}">Outage: ${outPct}%</span>
                        </div>
                        <div class="kpi-metric-grid">
                            <div><div class="kpi-label">Ativos (Transmitindo)</div><div class="kpi-value" style="color:#10b981;">${sm.n_active} <span class="unit">/ ${snap.K}</span></div></div>
                            <div><div class="kpi-label">Desligados (0 W)</div><div class="kpi-value" style="color:${sm.n_off > 0 ? '#ef4444' : '#94a3b8'};">${sm.n_off}</div></div>
                            <div><div class="kpi-label">Em Folga (P &lt; P_max)</div><div class="kpi-value" style="color:#38bdf8;">${sm.n_folga}</div></div>
                            <div><div class="kpi-label">No Teto (P_max)</div><div class="kpi-value">${sm.n_at_max}</div></div>
                        </div>
                    </div>
                    <div class="kpi-card">
                        <div class="card-header">
                            <span class="model-badge" style="border-left:3px solid #a855f7;">Rastreamento SINR &amp; Alvo</span>
                            <span class="outage-tag safe">γ_min=${snap.gamma_min_db} dB</span>
                        </div>
                        <div class="kpi-metric-grid">
                            <div><div class="kpi-label">SINR Médio (Ativos)</div><div class="kpi-value" style="color:#10b981;">${sinrText}</div></div>
                            <div><div class="kpi-label">Erro Médio |SINR - γ_tar|</div><div class="kpi-value">${sm.mean_err_db.toFixed(2)} <span class="unit">dB</span></div></div>
                            <div><div class="kpi-label">Faixa γ_tar Final</div><div class="kpi-value">${tgtRange}</div></div>
                            <div><div class="kpi-label">Teto QAM (γ_max)</div><div class="kpi-value">${snap.gamma_max_db} <span class="unit">dB</span></div></div>
                        </div>
                    </div>
                `;
            }

            function appendOuterGuides(shapes, annots, outerMarks) {
                outerMarks.slice(0, 30).forEach((mk, i) => {
                    shapes.push({
                        type: "line", yref: "paper", y0: 0, y1: 1,
                        x0: mk - 0.5, x1: mk - 0.5,
                        line: { color: "rgba(100,116,139,0.45)", width: 1, dash: "dot" }
                    });
                    if (i < 12) {
                        annots.push({
                            yref: "paper", y: 0.98, yanchor: "top",
                            x: mk - 0.5, xanchor: "left",
                            text: `E${i + 1}`, showarrow: false,
                            font: { color: "#64748b", size: 9 }
                        });
                    }
                });
            }

            function bindSyncZoom() {
                if (zoomBound || typeof Plotly === "undefined") return;
                const pEl = document.getElementById(powerPlotId);
                const sEl = document.getElementById(sinrPlotId);
                if (!pEl || !sEl || typeof pEl.on !== "function" || typeof sEl.on !== "function") return;

                const wire = (src, dst) => {
                    src.on("plotly_relayout", (ev) => {
                        if (syncingZoom || !ev) return;
                        const upd = {};
                        if (ev["xaxis.autorange"]) {
                            upd["xaxis.autorange"] = true;
                        } else if (Array.isArray(ev["xaxis.range"]) && ev["xaxis.range"].length === 2) {
                            upd["xaxis.range"] = ev["xaxis.range"];
                        } else if (ev["xaxis.range[0]"] !== undefined && ev["xaxis.range[1]"] !== undefined) {
                            upd["xaxis.range"] = [ev["xaxis.range[0]"], ev["xaxis.range[1]"]];
                        } else {
                            return;
                        }
                        syncingZoom = true;
                        Promise.resolve(Plotly.relayout(dst, upd)).finally(() => {
                            syncingZoom = false;
                        });
                    });
                };
                wire(pEl, sEl);
                wire(sEl, pEl);
                zoomBound = true;
            }

            function renderAll() {
                if (typeof Plotly === "undefined") return;
                const snapIdx = parseInt(elSnap.value || "0", 10);
                const snap = DATA.snapshots[snapIdx] || DATA.snapshots[0];
                const scale = elScale.value || "dbm";
                const filterMode = elFilter.value || "all";
                const uesShown = selectUeIndices(snap, filterMode);
                const isDbm = (scale === "dbm");

                renderCards(snap);

                const convVal = (dbm) => {
                    if (dbm === null || dbm === undefined) return null;
                    return isDbm ? dbm : Math.pow(10, dbm / 10.0);
                };
                const offFloor = isDbm ? (snap.p_min_dbm - 4.0) : 0.0;
                const pMaxVal = convVal(snap.p_max_dbm);
                const pMinVal = convVal(snap.p_min_dbm);
                const pRule2Val = convVal(snap.p_max_dbm - snap.mu_step);

                // --- Gráfico 1: Convergência Temporal da Potência de TX por UE ---
                const trPower = [];
                const pOffX = [], pOffY = [], pOffTxt = [];
                for (const k of uesShown) {
                    const col = PALETTE[k % PALETTE.length];
                    const tOff = snap.t_off[k];
                    const xActive = [], yActive = [];
                    let lastStep = -1, lastVal = null;
                    for (let r = 0; r < snap.step_t.length; r++) {
                        const t = snap.step_t[r];
                        const val = convVal(snap.p_dbm[r][k]);
                        if (val !== null) {
                            xActive.push(t);
                            yActive.push(val);
                            lastStep = t;
                            lastVal = val;
                        }
                    }
                    const isOff = (tOff >= 0);
                    trPower.push({
                        x: xActive,
                        y: yActive,
                        mode: snap.T <= 70 ? "lines+markers" : "lines",
                        marker: { size: 4 },
                        name: `UE ${k}` + (isOff ? " (OFF)" : ""),
                        line: { color: col, width: 2 },
                        hovertemplate: `<b>UE ${k}</b><br>Passo t: %{x}<br>Potência Tx: %{y:.2f} ${isDbm ? "dBm" : "mW"}<extra></extra>`
                    });
                    if (isOff) {
                        trPower.push({
                            x: lastStep >= 0 ? [lastStep, tOff] : [tOff],
                            y: lastVal !== null ? [lastVal, offFloor] : [offFloor],
                            mode: "lines",
                            line: { color: "#ef4444", width: 1.6, dash: "dot" },
                            showlegend: false,
                            hoverinfo: "skip"
                        });
                        pOffX.push(lastStep >= 0 ? lastStep : tOff);
                        pOffY.push(lastVal !== null ? lastVal : offFloor);
                        pOffTxt.push(`<b>UE ${k} Desligado (Outage)</b><br>Corte no passo t=${tOff}<br>Potência anterior: ${lastVal !== null ? lastVal.toFixed(2) : "0"} ${isDbm ? "dBm" : "mW"} → 0 W`);
                    }
                }
                if (pOffX.length > 0) {
                    trPower.push({
                        x: pOffX,
                        y: pOffY,
                        mode: "markers",
                        name: "Corte Outage (0 W)",
                        marker: { symbol: "x", size: 10, color: "#ef4444", line: { color: "#ffffff", width: 1 } },
                        text: pOffTxt,
                        hovertemplate: "%{text}<extra></extra>"
                    });
                }

                const shapesPower = [
                    { type: "line", xref: "paper", x0: 0, x1: 1, y0: pMaxVal, y1: pMaxVal, line: { color: "#38bdf8", width: 1.2, dash: "dash" } },
                    { type: "line", xref: "paper", x0: 0, x1: 1, y0: pMinVal, y1: pMinVal, line: { color: "#94a3b8", width: 1, dash: "dot" } },
                    { type: "line", xref: "paper", x0: 0, x1: 1, y0: pRule2Val, y1: pRule2Val, line: { color: "#f59e0b", width: 1, dash: "dot" } }
                ];
                if (isDbm) {
                    shapesPower.push({ type: "line", xref: "paper", x0: 0, x1: 1, y0: offFloor, y1: offFloor, line: { color: "#ef4444", width: 1, dash: "dashdot" } });
                }
                const annPower = [
                    { xref: "paper", x: 0.995, xanchor: "right", y: pMaxVal, yanchor: "bottom", text: `P_max (${pMaxVal.toFixed(1)} ${isDbm ? "dBm" : "mW"})`, showarrow: false, font: { color: "#38bdf8", size: 10 } },
                    { xref: "paper", x: 0.995, xanchor: "right", y: pMinVal, yanchor: "bottom", text: `P_min (${pMinVal.toFixed(1)})`, showarrow: false, font: { color: "#94a3b8", size: 10 } }
                ];
                if (isDbm) {
                    annPower.push({ xref: "paper", x: 0.995, xanchor: "right", y: offFloor, yanchor: "bottom", text: "OFF (0 W / Outage)", showarrow: false, font: { color: "#ef4444", size: 10 } });
                }
                appendOuterGuides(shapesPower, annPower, snap.outer_marks);

                Plotly.react(
                    powerPlotId,
                    trPower,
                    baseLayout(
                        `1. Convergência Temporal da Potência de TX por UE (${isDbm ? "dBm" : "mW"})`,
                        "Passo do Loop Interno (t)",
                        isDbm ? "Potência Tx (dBm)" : "Potência Tx (mW)",
                        {
                            margin: { l: 54, r: 28, t: 54, b: 36 },
                            shapes: shapesPower,
                            annotations: annPower,
                            xaxis: {
                                title: "Passo do Loop Interno (t)",
                                gridcolor: "#1e293b",
                                zerolinecolor: "#334155",
                                rangeslider: { visible: true, thickness: 0.08, bgcolor: "#0b0e14", bordercolor: "#232c3d" }
                            }
                        }
                    ),
                    { responsive: true, displayModeBar: true }
                );

                // --- Gráfico 2: Convergência Temporal da SINR & Alvo Dinâmico γ_tar ---
                const trSinr = [];
                const sOffX = [], sOffY = [], sOffTxt = [];
                for (const k of uesShown) {
                    const col = PALETTE[k % PALETTE.length];
                    const tOff = snap.t_off[k];
                    const ySinr = snap.sinr_db.map(r => r[k]);
                    const yTgt = snap.tgt_step_db.map((r, ri) => (snap.sinr_db[ri][k] !== null ? r[k] : null));
                    const isOff = (tOff >= 0);

                    trSinr.push({
                        x: snap.step_t,
                        y: ySinr,
                        mode: snap.T <= 70 ? "lines+markers" : "lines",
                        marker: { size: 4 },
                        name: `SINR UE ${k}` + (isOff ? " (OFF)" : ""),
                        legendgroup: `g_${k}`,
                        line: { color: col, width: 2 },
                        hovertemplate: `<b>UE ${k}</b><br>Passo t: %{x}<br>SINR Real: %{y:.1f} dB<extra></extra>`
                    });
                    trSinr.push({
                        x: snap.step_t,
                        y: yTgt,
                        mode: "lines",
                        name: `Alvo γ_tar UE ${k}`,
                        legendgroup: `g_${k}`,
                        showlegend: false,
                        line: { color: col, width: 1.25, dash: "dot", shape: "hv" },
                        hovertemplate: `<b>UE ${k}</b><br>Passo t: %{x}<br>Alvo γ_tar: %{y:.1f} dB<extra></extra>`
                    });

                    if (isOff) {
                        let lastStep = -1, lastSinr = null;
                        for (let r = 0; r < snap.step_t.length; r++) {
                            if (snap.sinr_db[r][k] !== null) {
                                lastStep = snap.step_t[r];
                                lastSinr = snap.sinr_db[r][k];
                            }
                        }
                        if (lastSinr !== null) {
                            sOffX.push(lastStep);
                            sOffY.push(lastSinr);
                            sOffTxt.push(`<b>UE ${k} Desligado (Outage)</b><br>Corte no passo t=${tOff}<br>Última SINR ativa: ${lastSinr.toFixed(1)} dB`);
                        }
                    }
                }
                if (sOffX.length > 0) {
                    trSinr.push({
                        x: sOffX,
                        y: sOffY,
                        mode: "markers",
                        name: "Corte Outage (0 W)",
                        marker: { symbol: "x", size: 10, color: "#ef4444", line: { color: "#ffffff", width: 1 } },
                        text: sOffTxt,
                        hovertemplate: "%{text}<extra></extra>"
                    });
                }

                const shapesSinr = [
                    { type: "line", xref: "paper", x0: 0, x1: 1, y0: snap.gamma_max_db, y1: snap.gamma_max_db, line: { color: "#10b981", width: 1.2, dash: "dash" } },
                    { type: "line", xref: "paper", x0: 0, x1: 1, y0: snap.gamma_min_db, y1: snap.gamma_min_db, line: { color: "#ef4444", width: 1.2, dash: "dash" } }
                ];
                const annSinr = [
                    { xref: "paper", x: 0.995, xanchor: "right", y: snap.gamma_max_db, yanchor: "bottom", text: `γ_max (${snap.gamma_max_db} dB)`, showarrow: false, font: { color: "#10b981", size: 10 } },
                    { xref: "paper", x: 0.995, xanchor: "right", y: snap.gamma_min_db, yanchor: "top", text: `γ_min / Outage (${snap.gamma_min_db} dB)`, showarrow: false, font: { color: "#ef4444", size: 10 } }
                ];
                appendOuterGuides(shapesSinr, annSinr, snap.outer_marks);

                Plotly.react(
                    sinrPlotId,
                    trSinr,
                    baseLayout(
                        "2. Convergência Temporal da SINR Real (Sólido) & Alvo Dinâmico γ_tar (Pontilhado)",
                        "Passo do Loop Interno (t)",
                        "SINR / Alvo γ_tar (dB)",
                        {
                            shapes: shapesSinr,
                            annotations: annSinr
                        }
                    ),
                    { responsive: true, displayModeBar: true }
                );

                bindSyncZoom();
            }

            elSnap.addEventListener("change", renderAll);
            elScale.addEventListener("change", renderAll);
            elFilter.addEventListener("change", renderAll);
            window.addEventListener("resize", () => {
                if (typeof Plotly === "undefined") return;
                plotIds.forEach(id => {
                    const gd = document.getElementById(id);
                    if (gd && gd.data && gd.offsetParent !== null) {
                        try { Plotly.Plots.resize(gd); } catch (e) {}
                    }
                });
            });

            if (document.readyState === "loading") {
                document.addEventListener("DOMContentLoaded", renderAll);
            } else {
                renderAll();
            }
        })();
        </script>
        """.replace("__UID__", uid).replace("__JSON__", json_data)

        return html + script

    @classmethod
    def _render_params_pills(cls, params: dict, is_snapshot: bool = False) -> str:
        if not params:
            return ""
        b_mhz = float(params.get("b_total_mhz", B_TOTAL / 1e6))
        ray = "Sim" if params.get("use_rayleigh", False) else "Não"
        if params.get("use_rayleigh", False):
            ray = f"Sim (σR={float(params.get('sigma_r', 0.707)):.2f})"

        pm_names = {
            "full_power": "Full Power",
            "dpc_fixed": "DPC Fixo",
            "dual_loop": "Dual-Loop DPC",
        }
        pm_val = str(params.get("power_mode", params.get("power_control", "full_power")))
        pm_lbl = pm_names.get(pm_val, pm_val)

        pills = [
            f"M={params.get('M', 4)} APs",
            f"K={params.get('K', 8)} UEs",
            f"N={params.get('N', 1)} ch ({params.get('allocation', 'ortogonal')})",
            f"L={params.get('L', 8)}, J={params.get('J', 8)}",
            f"Potência={pm_lbl}",
            f"B={b_mhz:g} MHz",
            f"n={float(params.get('pathloss_exp', 4.0)):g}",
            f"σX={float(params.get('sigma_x', 2.0)):g}",
            f"Rayleigh={ray}",
            f"P=[{float(params.get('p_min_dbm', 0)):g}, {float(params.get('p_max_dbm', 30)):g}] dBm",
            f"γ_tar={float(params.get('gamma_tar_db', 10)):g} dB | γ_min={float(params.get('gamma_min_db', -10)):g} dB | γ_max={float(params.get('gamma_max_db', 20)):g} dB",
        ]
        if not is_snapshot:
            pills.append(f"Iterações={params.get('sim', 100)}")

        return "<div class='pills-row'>" + "".join(
            f"<span class='param-pill'>{p}</span>" for p in pills
        ) + "</div>"

    @classmethod
    def _wrap_document(
        cls,
        plotly_src: str,
        params: dict,
        title: str,
        subtitle: str,
        body_html: str,
        is_snapshot: bool = False,
    ) -> str:
        pills_html = cls._render_params_pills(params, is_snapshot=is_snapshot)
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="{plotly_src}"></script>
    <script>
        if (typeof Plotly === 'undefined') {{
            document.write('<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"><\\/script>');
        }}
    </script>
    <style>
        body {{
            background-color: #0b0e14;
            color: #e2e8f0;
            font-family: 'Segoe UI', system-ui, sans-serif;
            margin: 0;
            padding: 16px;
            overflow-y: auto;
        }}
        .dash-header {{
            margin-bottom: 14px;
            border-bottom: 1px solid #1e2638;
            padding-bottom: 12px;
        }}
        .dash-header h2 {{
            margin: 0 0 4px 0;
            font-size: 18px;
            color: #38bdf8;
        }}
        .dash-header .subtitle {{
            font-size: 12px;
            color: #94a3b8;
            margin-bottom: 8px;
        }}
        .pills-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}
        .param-pill {{
            background: #141923;
            border: 1px solid #232c3d;
            border-radius: 999px;
            padding: 3px 10px;
            font-size: 11px;
            color: #cbd5e1;
        }}
        .info-banner {{
            background: rgba(14, 165, 233, 0.12);
            border: 1px solid #0284c7;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 14px;
            font-size: 12px;
            line-height: 1.5;
        }}
        .info-banner b {{ color: #38bdf8; display: block; margin-bottom: 2px; }}
        .subtab-bar {{
            display: flex;
            gap: 8px;
            margin-bottom: 14px;
        }}
        .subtab-btn {{
            background: #141923;
            color: #94a3b8;
            border: 1px solid #232c3d;
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 12px;
            font-weight: bold;
            cursor: pointer;
        }}
        .subtab-btn.active {{
            background: #0284c7;
            color: #ffffff;
            border-color: #38bdf8;
        }}
        .subtab-pane {{ display: none; }}
        .subtab-pane.active {{ display: block; }}
        .kpi-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }}
        .kpi-card {{
            background: #141923;
            border: 1px solid #232c3d;
            border-radius: 8px;
            padding: 13px 15px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 7px;
            border-bottom: 1px solid #1e2638;
        }}
        .model-badge {{
            font-weight: bold;
            font-size: 13px;
            color: #f8fafc;
            padding-left: 8px;
        }}
        .outage-tag {{
            font-size: 10px;
            padding: 2px 7px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .outage-tag.safe {{
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
            border: 1px solid #059669;
        }}
        .outage-tag.warn {{
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
            border: 1px solid #dc2626;
        }}
        .kpi-metric-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            row-gap: 9px;
            column-gap: 8px;
        }}
        .kpi-label {{
            font-size: 10px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }}
        .kpi-value {{
            font-size: 15px;
            font-weight: bold;
            color: #f1f5f9;
        }}
        .unit {{ font-size: 10px; font-weight: normal; color: #94a3b8; }}
        .subval {{ font-size: 11px; font-weight: normal; color: #cbd5e1; }}
        .table-card {{
            background: #141923;
            border: 1px solid #232c3d;
            border-radius: 8px;
            padding: 12px 14px;
            margin-bottom: 16px;
        }}
        .table-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        .table-header h4 {{ margin: 0; font-size: 13px; color: #38bdf8; }}
        .csv-btn {{
            background: #1e293b;
            color: #e2e8f0;
            border: 1px solid #334155;
            border-radius: 5px;
            padding: 5px 10px;
            font-size: 11px;
            font-weight: bold;
            cursor: pointer;
        }}
        .csv-btn:hover {{ background: #334155; }}
        .bench-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 11.5px;
        }}
        .bench-table th, .bench-table td {{
            padding: 7px 8px;
            text-align: center;
            border-bottom: 1px solid #1e2638;
            white-space: nowrap;
        }}
        .bench-table th {{ color: #94a3b8; background: #111622; font-size: 10.5px; text-transform: uppercase; }}
        .bench-table tr:hover {{ background: #1a2232; }}
        .best-cell {{ color: #34d399; font-weight: bold; }}
        .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}
        .chart-box {{
            background: #141923;
            border: 1px solid #232c3d;
            border-radius: 8px;
            padding: 10px;
        }}
        .dl-conv-section {{
            margin-top: 14px;
        }}
        .dl-toolbar {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            background: #141923;
            border: 1px solid #232c3d;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 14px;
            align-items: flex-end;
        }}
        .dl-ctrl-group {{
            display: flex;
            flex-direction: column;
            gap: 4px;
            min-width: 190px;
            flex: 1;
        }}
        .dl-ctrl-group label {{
            font-size: 11px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }}
        .dl-ctrl-group select {{
            background: #0f172a;
            color: #f1f5f9;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 12px;
        }}
        .dl-kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 12px;
            margin-bottom: 14px;
        }}
        .dl-plot {{
            width: 100%;
        }}
    </style>
</head>
<body>
    <div class="dash-header">
        <h2>📊 {title}</h2>
        <div class="subtitle">{subtitle}</div>
        {pills_html}
    </div>
    {body_html}
    <script>
        function switchSubTab(idx) {{
            document.querySelectorAll('.subtab-btn').forEach((b, i) => b.classList.toggle('active', i === idx));
            document.querySelectorAll('.subtab-pane').forEach((p, i) => p.classList.toggle('active', i === idx));
            const pane = document.getElementById('subpanel-' + idx);
            if (pane && typeof Plotly !== 'undefined') {{
                const resizePlots = () => {{
                    pane.querySelectorAll('.plotly-graph-div, .dl-plot').forEach(gd => {{
                        try {{ if (gd && gd.data) Plotly.Plots.resize(gd); }} catch (e) {{}}
                    }});
                }};
                resizePlots();
                if (typeof requestAnimationFrame === 'function') {{
                    requestAnimationFrame(resizePlots);
                }}
                setTimeout(resizePlots, 60);
            }}
            window.dispatchEvent(new Event('resize'));
        }}

        function openDualLoopPanel() {{
            const btn = document.getElementById('subtab-dl-btn');
            if (btn) {{
                btn.click();
                window.scrollTo({{ top: 0, behavior: 'smooth' }});
                return;
            }}
            const anchor = document.querySelector('.dl-conv-section');
            if (anchor) anchor.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
        }}

        function exportKpiCsv(tableId, filename) {{
            const tbl = document.getElementById(tableId);
            if (!tbl) return;
            const lines = [];
            tbl.querySelectorAll('tr').forEach(tr => {{
                const cols = [];
                tr.querySelectorAll('th, td').forEach(cell => {{
                    const txt = cell.innerText.replace(/★/g, '').trim().replace(/"/g, '""');
                    cols.push('"' + txt + '"');
                }});
                lines.push(cols.join(','));
            }});
            const blob = new Blob([lines.join('\\n')], {{ type: 'text/csv;charset=utf-8;' }});
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            window.URL.revokeObjectURL(url);
        }}
    </script>
</body>
</html>"""

    @classmethod
    def _empty_state_html(cls, plotly_path: str) -> str:
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            background-color: #0b0e14;
            color: #94a3b8;
            font-family: 'Segoe UI', sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            text-align: center;
        }}
        .box {{
            background: #131822;
            border: 1px dashed #232c3d;
            border-radius: 12px;
            padding: 40px;
            max-width: 500px;
        }}
        h2 {{ color: #38bdf8; margin-top: 0; }}
        p {{ line-height: 1.6; font-size: 13px; }}
    </style>
</head>
<body>
    <div class="box">
        <h2>📊 Painel de KPIs & Gráficos</h2>
        <p>Nenhum dado carregado ainda.</p>
        <p>Gere um cenário 3D para KPIs instantâneos ou execute a <b>Simulação Monte Carlo</b> na barra lateral para gerar curvas estatísticas completas.</p>
    </div>
</body>
</html>"""

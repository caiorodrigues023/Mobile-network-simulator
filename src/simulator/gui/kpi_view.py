# ---------------------------------------------------------------------------
# src/simulator/gui/kpi_view.py — Painel de KPIs e Gráficos Estatísticos
# ---------------------------------------------------------------------------

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

from ..config import MODEL_COLORS, MODEL_LABELS, GAMMA_MIN_DB, GAMMA_TAR_INIT


class KPIViewGenerator:
    """Gera o painel completo de métricas chave de desempenho (KPIs) e gráficos."""

    @classmethod
    def generate_kpi_html(cls, simulation_results: dict, plotly_url: str = None) -> str:
        """
        Recebe o dicionário com os resultados da simulação (Monte Carlo)
        ou snapshot atual e gera uma página HTML rica com cards de KPI e gráficos.
        """
        if plotly_url is None:
            assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
            plotly_src = "file:///" + os.path.join(assets_dir, "plotly.min.js").replace("\\", "/")
        else:
            plotly_src = plotly_url

        if not simulation_results:
            return cls._empty_state_html(plotly_src)

        res_type = simulation_results.get("type", "antenna_comparison")
        data = simulation_results.get("data", {})
        params = simulation_results.get("params", {})

        if res_type == "scichal4a":
            return cls._render_scichal4a_html(data, params, plotly_src)
        else:
            return cls._render_antenna_comparison_html(data, params, plotly_src)

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
        .btn-hint {{
            display: inline-block;
            margin-top: 15px;
            background: #0284c7;
            color: #fff;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="box">
        <h2>📊 Painel de KPIs & Gráficos</h2>
        <p>Nenhuma simulação de Monte Carlo foi executada ainda nesta sessão.</p>
        <p>Clique no botão <b>"Executar Simulação Monte Carlo"</b> na barra lateral para gerar curvas estatísticas completas de SINR, Capacidade, Potência e comparação de modelos.</p>
        <div class="btn-hint">🚀 Iniciar Simulação na Barra Lateral</div>
    </div>
</body>
</html>"""

    @classmethod
    def _render_antenna_comparison_html(cls, data: dict, params: dict, plotly_src: str) -> str:
        models = list(data.keys())
        if not models:
            return cls._empty_state_html(plotly_src)

        # -------------------------------------------------------------------
        # Extração de Métricas Principais
        # -------------------------------------------------------------------
        cards_html = ""
        cards = []
        for m in models:
            r = data[m]
            sinr_vals = np.array(r.get("sinr", []))
            cap_vals = np.array(r.get("cap", []))
            sum_cap = r.get("sum_cap", 0.0)

            if len(sinr_vals) > 0:
                s_p10 = float(10.0 * np.log10(np.maximum(np.percentile(sinr_vals, 10), 1e-12)))
                s_med = float(10.0 * np.log10(np.maximum(np.median(sinr_vals), 1e-12)))
                c_p10 = float(np.percentile(cap_vals, 10))
                outage_rate = float(np.mean(sinr_vals < 10.0 ** (GAMMA_MIN_DB / 10.0)) * 100.0)
            else:
                s_p10, s_med, c_p10, outage_rate = 0.0, 0.0, 0.0, 0.0

            lbl = MODEL_LABELS.get(m, m.capitalize())
            c_accent = "#38bdf8" if "smb" in m else ("#f43f5e" if "adaptive" in m else "#3b82f6")

            cards.append(f"""
            <div class="kpi-card">
                <div class="card-header">
                    <span class="model-badge" style="border-left: 3px solid {c_accent};">{lbl}</span>
                    <span class="outage-tag {'safe' if outage_rate < 1.0 else 'warn'}">Outage: {outage_rate:.1f}%</span>
                </div>
                <div class="kpi-metric-grid">
                    <div>
                        <div class="kpi-label">Vazão Total (Sum-Cap)</div>
                        <div class="kpi-value" style="color: #38bdf8;">{sum_cap:.1f} <span class="unit">Mbps</span></div>
                    </div>
                    <div>
                        <div class="kpi-label">SINR 10º Perc. (Borda)</div>
                        <div class="kpi-value" style="color: {'#10b981' if s_p10 >= 0 else '#f43f5e'};">{s_p10:+.1f} <span class="unit">dB</span></div>
                    </div>
                    <div>
                        <div class="kpi-label">SINR Mediano (50º)</div>
                        <div class="kpi-value">{s_med:+.1f} <span class="unit">dB</span></div>
                    </div>
                    <div>
                        <div class="kpi-label">Capacidade 10º Perc.</div>
                        <div class="kpi-value">{c_p10:.1f} <span class="unit">Mbps</span></div>
                    </div>
                </div>
            </div>
            """)

        cards_html = "".join(cards)

        # -------------------------------------------------------------------
        # Gráficos Plotly: CDFs de SINR e Capacidade + Barras de Comparação
        # -------------------------------------------------------------------
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "CDF de SINR (dB)",
                "CDF de Capacidade de Shannon (Mbps)",
                "Comparativo de Vazão Agregada (Sum Capacity)",
                "Ganho Relativo ao Modelo Omni (dB)",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        p = np.linspace(0, 100, 300)

        # Paleta de cores para os traços
        model_color_map = {
            "omni": "#38bdf8",
            "sector": "#f59e0b",
            "smb": "#10b981",
            "adaptive": "#f43f5e",
            "adaptive_1": "#8b5cf6",
            "adaptive_5": "#ec4899",
            "adaptive_10": "#a855f7",
        }

        # 1. CDF de SINR
        for m in models:
            sinr = np.array(data[m].get("sinr", []))
            if len(sinr) > 0:
                s_db = 10.0 * np.log10(np.maximum(sinr, 1e-12))
                q = np.percentile(s_db, p)
                c = model_color_map.get(m, "#cbd5e1")
                fig.add_trace(
                    go.Scatter(
                        x=q, y=p / 100.0,
                        mode="lines",
                        name=MODEL_LABELS.get(m, m),
                        line=dict(color=c, width=2.5),
                        hovertemplate=f"<b>{MODEL_LABELS.get(m, m)}</b><br>SINR: %{{x:.1f}} dB<br>Probabilidade: %{{y:.2f}}<extra></extra>",
                    ),
                    row=1, col=1,
                )

        # Linha vertical de limiar de outage (-10 dB)
        fig.add_vline(x=GAMMA_MIN_DB, line_width=1, line_dash="dash", line_color="#ef4444",
                      annotation_text="Limiar Outage (-10 dB)", annotation_position="top left",
                      annotation_font_color="#ef4444", annotation_font_size=10, row=1, col=1)

        # 2. CDF de Capacidade
        for m in models:
            cap = np.array(data[m].get("cap", []))
            if len(cap) > 0:
                q = np.percentile(cap, p)
                c = model_color_map.get(m, "#cbd5e1")
                fig.add_trace(
                    go.Scatter(
                        x=q, y=p / 100.0,
                        mode="lines",
                        name=MODEL_LABELS.get(m, m),
                        line=dict(color=c, width=2.5),
                        showlegend=False,
                        hovertemplate=f"<b>{MODEL_LABELS.get(m, m)}</b><br>Capacidade: %{{x:.2f}} Mbps<br>Probabilidade: %{{y:.2f}}<extra></extra>",
                    ),
                    row=1, col=2,
                )

        # 3. Gráfico de Barras: Sum Capacity
        sum_caps = [data[m].get("sum_cap", 0.0) for m in models]
        labels = [MODEL_LABELS.get(m, m) for m in models]
        bar_colors = [model_color_map.get(m, "#38bdf8") for m in models]

        fig.add_trace(
            go.Bar(
                x=labels,
                y=sum_caps,
                marker=dict(color=bar_colors, line=dict(color="#ffffff", width=1)),
                text=[f"{v:.1f} Mbps" for v in sum_caps],
                textposition="auto",
                showlegend=False,
                hovertemplate="<b>%{x}</b><br>Sum Capacity: %{y:.1f} Mbps<extra></extra>",
            ),
            row=2, col=1,
        )

        # 4. Gráfico de Barras: Ganho em relação ao Omni
        omni_cap = data.get("omni", {}).get("sum_cap", 1e-6)
        if omni_cap <= 0:
            omni_cap = 1e-6

        gains_db = [float(10.0 * np.log10(max(sc / omni_cap, 1e-3))) for sc in sum_caps]

        fig.add_trace(
            go.Bar(
                x=labels,
                y=gains_db,
                marker=dict(color=bar_colors, line=dict(color="#ffffff", width=1)),
                text=[f"{v:+.1f} dB" for v in gains_db],
                textposition="auto",
                showlegend=False,
                hovertemplate="<b>%{x}</b><br>Ganho vs Omni: %{y:+.2f} dB<extra></extra>",
            ),
            row=2, col=2,
        )

        # Layout styling
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0e14",
            plot_bgcolor="#111622",
            font=dict(family="Segoe UI, sans-serif", color="#e2e8f0"),
            margin=dict(l=50, r=30, t=50, b=50),
            height=680,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.04,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(17, 22, 34, 0.8)",
                bordercolor="#232c3d",
                borderwidth=1,
                font=dict(size=11),
            ),
        )

        fig.update_xaxes(gridcolor="#1e293b", zerolinecolor="#334155")
        fig.update_yaxes(gridcolor="#1e293b", zerolinecolor="#334155")

        chart_html = pyo.plot(
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
    <script>
        if (typeof Plotly === 'undefined') {{
            document.write('<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"><\\/script>');
        }}
    </script>
    <style>
        body {{
            background-color: #0b0e14;
            color: #e2e8f0;
            font-family: 'Segoe UI', sans-serif;
            margin: 0;
            padding: 16px;
            overflow-y: auto;
        }}
        .kpi-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }}
        .kpi-card {{
            background: #141923;
            border: 1px solid #232c3d;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 8px;
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
            padding: 2px 6px;
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
            row-gap: 10px;
            column-gap: 8px;
        }}
        .kpi-label {{
            font-size: 10px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .kpi-value {{
            font-size: 16px;
            font-weight: bold;
            color: #f1f5f9;
        }}
        .unit {{
            font-size: 10px;
            font-weight: normal;
            color: #94a3b8;
        }}
        .chart-box {{
            background: #141923;
            border: 1px solid #232c3d;
            border-radius: 8px;
            padding: 10px;
        }}
    </style>
</head>
<body>
    <div class="kpi-container">
        {cards_html}
    </div>
    <div class="chart-box">
        {chart_html}
    </div>
</body>
</html>"""

    @classmethod
    def _render_scichal4a_html(cls, stats: dict, params: dict, plotly_src: str) -> str:
        """Renderiza painel específico de controle de potência (SciChal4A)."""
        algos = ["full_power", "dpc_fixed", "dual_loop"]
        labels = {
            "full_power": "Full Power (1W)",
            "dpc_fixed": "DPC Fixo (10 dB)",
            "dual_loop": "Dual-Loop DPC",
        }
        colors = {
            "full_power": "#94a3b8",
            "dpc_fixed": "#38bdf8",
            "dual_loop": "#f97316",
        }

        cards = []
        for a in algos:
            st = stats.get(a, {})
            se_mean = float(np.mean(st.get("se", [0])))
            sum_cap_mean = float(np.mean(st.get("sum_cap", [0])))
            p_mean = float(np.mean(st.get("power", [0])))
            out_mean = float(np.mean(st.get("outage", [0]))) * 100.0

            cards.append(f"""
            <div class="kpi-card">
                <div class="card-header">
                    <span class="model-badge" style="border-left: 3px solid {colors[a]};">{labels[a]}</span>
                    <span class="outage-tag {'safe' if out_mean < 1.0 else 'warn'}">Outage: {out_mean:.1f}%</span>
                </div>
                <div class="kpi-metric-grid">
                    <div>
                        <div class="kpi-label">Vazão Total Média</div>
                        <div class="kpi-value" style="color: {colors[a]};">{sum_cap_mean:.1f} <span class="unit">Mbps</span></div>
                    </div>
                    <div>
                        <div class="kpi-label">Eficiência Espectral</div>
                        <div class="kpi-value">{se_mean:.2f} <span class="unit">bps/Hz</span></div>
                    </div>
                    <div>
                        <div class="kpi-label">Potência Total Consumida</div>
                        <div class="kpi-value">{p_mean:.2f} <span class="unit">Watts</span></div>
                    </div>
                    <div>
                        <div class="kpi-label">Taxa de Outage</div>
                        <div class="kpi-value" style="color: {'#10b981' if out_mean < 1.0 else '#f43f5e'};">{out_mean:.1f} <span class="unit">%</span></div>
                    </div>
                </div>
            </div>
            """)

        cards_html = "".join(cards)

        # Gráficos
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "CDF de SINR (dB)",
                "CDF de Potência de Transmissão (Watts)",
                "Consumo de Potência Médio (W)",
                "Convergência do Algoritmo Dual-Loop",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        p = np.linspace(0, 100, 300)

        # 1. CDF de SINR
        for a in algos:
            sinr = np.array(stats.get(a, {}).get("sinr_lin", []))
            if len(sinr) > 0:
                s_db = 10.0 * np.log10(np.maximum(sinr, 1e-12))
                q = np.percentile(s_db, p)
                fig.add_trace(
                    go.Scatter(x=q, y=p / 100.0, mode="lines", name=labels[a],
                               line=dict(color=colors[a], width=2.5)),
                    row=1, col=1,
                )

        # 2. CDF de Potência
        for a in algos:
            pw = np.array(stats.get(a, {}).get("power_dist", []))
            if len(pw) > 0:
                q = np.percentile(pw, p)
                fig.add_trace(
                    go.Scatter(x=q, y=p / 100.0, mode="lines", name=labels[a],
                               line=dict(color=colors[a], width=2.5), showlegend=False),
                    row=1, col=2,
                )

        # 3. Bar chart de Potência média
        p_means = [float(np.mean(stats.get(a, {}).get("power", [0]))) for a in algos]
        fig.add_trace(
            go.Bar(
                x=[labels[a] for a in algos],
                y=p_means,
                marker=dict(color=[colors[a] for a in algos]),
                text=[f"{v:.2f} W" for v in p_means],
                textposition="auto",
                showlegend=False,
            ),
            row=2, col=1,
        )

        # 4. Convergência Dual-Loop (se histórico disponível)
        hist_list = stats.get("dual_loop_hist", [])
        if hist_list:
            h = hist_list[0]
            pw_hist = h.get("power_w", np.array([]))
            if len(pw_hist) > 0:
                T, K = pw_hist.shape
                it_inner = np.arange(T)
                for k in range(min(K, 6)):
                    # Converte para dBm
                    p_dbm = 10.0 * np.log10(np.maximum(pw_hist[:, k], 1e-15) * 1000.0)
                    fig.add_trace(
                        go.Scatter(
                            x=it_inner, y=p_dbm, mode="lines",
                            name=f"UE {k+1}",
                            line=dict(width=1.5),
                        ),
                        row=2, col=2,
                    )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0e14",
            plot_bgcolor="#111622",
            font=dict(family="Segoe UI, sans-serif", color="#e2e8f0"),
            margin=dict(l=50, r=30, t=50, b=50),
            height=680,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.04,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(17, 22, 34, 0.8)",
                bordercolor="#232c3d",
                borderwidth=1,
            ),
        )

        chart_html = pyo.plot(
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
            padding: 16px;
            overflow-y: auto;
        }}
        .kpi-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }}
        .kpi-card {{
            background: #141923;
            border: 1px solid #232c3d;
            border-radius: 8px;
            padding: 14px 16px;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 8px;
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
            padding: 2px 6px;
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
            row-gap: 10px;
            column-gap: 8px;
        }}
        .kpi-label {{
            font-size: 10px;
            color: #94a3b8;
            text-transform: uppercase;
        }}
        .kpi-value {{
            font-size: 16px;
            font-weight: bold;
            color: #f1f5f9;
        }}
        .unit {{ font-size: 10px; color: #94a3b8; font-weight: normal; }}
        .chart-box {{
            background: #141923;
            border: 1px solid #232c3d;
            border-radius: 8px;
            padding: 10px;
        }}
    </style>
</head>
<body>
    <div class="kpi-container">
        {cards_html}
    </div>
    <div class="chart-box">
        {chart_html}
    </div>
</body>
</html>"""

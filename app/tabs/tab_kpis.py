# ---------------------------------------------------------------------------
# app/tabs/tab_kpis.py — Aba 3: Painel 2x2 de KPIs, Tabela Comparativa e Export
# ---------------------------------------------------------------------------

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.charts import build_kpi_2x2_figure
from simulator.config import MODEL_LABELS
from simulator.schemas import MonteCarloResult, ScenarioMetrics


def _metrics_to_dataframe(
    metrics_dict: dict[str, dict[int, ScenarioMetrics]],
    k_scenarios: tuple[int, ...],
) -> pd.DataFrame:
    rows = []
    for k in k_scenarios:
        for key, by_k in metrics_dict.items():
            sm = by_k[k]
            rows.append(
                {
                    "Carga (K)": k,
                    "Modelo / Algoritmo": sm.label,
                    "Border SE 10º [bps/Hz]": round(sm.border_se_p10, 3),
                    "Sum-SE [bps/Hz]": round(sm.sum_se_mean, 3),
                    "Sum-Cap [Mbps]": round(sm.sum_cap_mean_mbps, 2),
                    "Cap 10º [Mbps]": round(sm.cap_p10_mbps, 2),
                    "SINR 10º [dB]": round(sm.sinr_p10_db, 2),
                    "SINR 50º [dB]": round(sm.sinr_p50_db, 2),
                    "SINR 90º [dB]": round(sm.sinr_p90_db, 2),
                    "Potência Média [W]": round(sm.mean_tx_power_w, 4),
                    "Efic. Energ. [Mb/J]": round(sm.energy_efficiency_mbit_j, 2),
                    "Outage [%]": round(sm.outage_rate_pct, 2),
                    "Justiça Jain": round(sm.jain_fairness, 3),
                    "ASE [bps/Hz/km²]": round(sm.ase_bps_hz_km2, 3),
                }
            )
    return pd.DataFrame(rows)


def render_tab_kpis(mc_result: MonteCarloResult | None) -> None:
    """Painel 2x2 de KPIs + Tabela interativa e exportação CSV/JSON."""
    st.subheader("🏆 Painel Executivo de KPIs & Benchmark Comparativo")
    if mc_result is None:
        st.info("Execute a simulação de **Monte Carlo** na barra lateral para gerar o painel de KPIs.")
        return

    groups: list[tuple[str, str, dict]] = []
    if mc_result.power_metrics:
        groups.append(
            (
                "power",
                f"Controle de Potência (Antena: {MODEL_LABELS.get(mc_result.config.snapshot_antenna_model, mc_result.config.snapshot_antenna_model)})",
                mc_result.power_metrics,
            )
        )
    if mc_result.antenna_metrics:
        groups.append(
            (
                "antennas",
                f"Modelos de Antena (Potência: {MODEL_LABELS.get(mc_result.config.snapshot_power_mode, mc_result.config.snapshot_power_mode)})",
                mc_result.antenna_metrics,
            )
        )

    for grp_key, grp_title, grp_metrics in groups:
        st.markdown(f"### {grp_title}")
        fig = build_kpi_2x2_figure(grp_metrics, mc_result.k_scenarios, title=f"KPIs — {grp_title}")
        st.plotly_chart(fig, width="stretch")

        df = _metrics_to_dataframe(grp_metrics, mc_result.k_scenarios)
        st.dataframe(df, width="stretch", hide_index=True)

        c1, c2 = st.columns(2)
        c1.download_button(
            f"📥 Baixar Tabela CSV ({grp_key})",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"kpis_{grp_key}.csv",
            mime="text/csv",
            width="stretch",
        )
        c2.download_button(
            f"📥 Baixar Tabela JSON ({grp_key})",
            data=df.to_json(orient="records", indent=2).encode("utf-8"),
            file_name=f"kpis_{grp_key}.json",
            mime="application/json",
            width="stretch",
        )
        st.divider()

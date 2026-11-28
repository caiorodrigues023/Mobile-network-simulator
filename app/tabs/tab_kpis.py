# ---------------------------------------------------------------------------
# app/tabs/tab_kpis.py — Aba 2: KPIs Instantâneos do Snapshot + Benchmark MC
# ---------------------------------------------------------------------------

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.charts import (
    build_antenna_baseline_gain_figure,
    build_convergence_figures,
    build_kpi_2x2_figure,
    build_snapshot_kpis_2x2_figure,
)
from app.tabs.tab_snapshot import render_snapshot_summary_cards
from simulator.config import MODEL_LABELS
from simulator.schemas import MonteCarloResult, ScenarioMetrics, SimulationConfig, SnapshotResult


def _metrics_to_dataframe(
    metrics_dict: dict[str, dict[int, ScenarioMetrics]],
    k_scenarios: tuple[int, ...],
) -> pd.DataFrame:
    rows = []
    for k in k_scenarios:
        for _, by_k in metrics_dict.items():
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


def render_tab_kpis(
    config: SimulationConfig,
    snapshot: SnapshotResult,
    mc_result: MonteCarloResult | None,
) -> None:
    """
    Aba 2:
      - Sempre exibe imediatamente os KPIs do Snapshot atual (cards + grid 2x2:
        SINR por UE, Capacidade por UE, Potência por UE, Carga & Vazão por AP + convergência).
      - Quando houver Monte Carlo, exibe blocos completos de Benchmark para
        Antenas (incluindo Ganho vs. Baseline) e/ou Controle de Potência (incluindo Convergência).
    """
    st.subheader("📊 KPIs & Gráficos Comparativos")

    # ------------------------------------------------------------------
    # 1. KPIs Instantâneos do Snapshot Atual (sempre disponível!)
    # ------------------------------------------------------------------
    ant_lbl = MODEL_LABELS.get(snapshot.antenna_model, snapshot.antenna_model)
    pwr_lbl = MODEL_LABELS.get(snapshot.power_mode, snapshot.power_mode)
    st.markdown(
        f"### ⚡ KPIs do Snapshot Atual — `{ant_lbl}` | `{pwr_lbl}` | "
        f"M={len(snapshot.aps)} APs, K={len(snapshot.ues)} UEs (Seed={snapshot.seed})"
    )
    render_snapshot_summary_cards(snapshot)

    fig_snap_kpi = build_snapshot_kpis_2x2_figure(
        snapshot,
        gamma_min_db=config.power_control.gamma_min_db,
        gamma_tar_db=config.power_control.gamma_tar_init_db,
    )
    st.plotly_chart(fig_snap_kpi, width="stretch", key="tab2_snap_kpi_2x2")

    if snapshot.convergence_history is not None:
        with st.expander("🔬 Preview da Convergência do Algoritmo neste Snapshot", expanded=False):
            fig_p, fig_s, _ = build_convergence_figures(
                snapshot.convergence_history, power_unit="dbm", ue_filter="all"
            )
            c_left, c_right = st.columns(2)
            c_left.plotly_chart(fig_p, width="stretch", key="tab2_snap_conv_p")
            c_right.plotly_chart(fig_s, width="stretch", key="tab2_snap_conv_s")

    # ------------------------------------------------------------------
    # 2. Benchmark Estatístico do Monte Carlo (Antenas e/ou Potência)
    # ------------------------------------------------------------------
    st.divider()
    if mc_result is None:
        st.info(
            "Execute **▶️ Executar Monte Carlo** na barra lateral para gerar o Benchmark estatístico "
            "comparativo entre todos os Modelos de Antena e/ou Algoritmos de Controle de Potência selecionados."
        )
        return

    st.markdown(
        f"### 🏆 Benchmark Monte Carlo ({mc_result.config.monte_carlo_runs} rodadas × "
        f"K ∈ {list(mc_result.k_scenarios)})"
    )

    if mc_result.antenna_metrics:
        pwr_ref = MODEL_LABELS.get(
            mc_result.config.snapshot_power_mode, mc_result.config.snapshot_power_mode
        )
        st.markdown(f"#### 📡 Modelos de Antena (Potência fixa: {pwr_ref})")
        fig_ant = build_kpi_2x2_figure(
            mc_result.antenna_metrics,
            mc_result.k_scenarios,
            title=f"KPIs 2×2 — Modelos de Antena ({pwr_ref})",
        )
        st.plotly_chart(fig_ant, width="stretch", key="tab2_mc_kpi_ant")

        if len(mc_result.antenna_metrics) > 1:
            fig_gain = build_antenna_baseline_gain_figure(
                mc_result.antenna_metrics, mc_result.k_scenarios
            )
            st.plotly_chart(fig_gain, width="stretch", key="tab2_mc_gain_ant")

        df_ant = _metrics_to_dataframe(mc_result.antenna_metrics, mc_result.k_scenarios)
        st.dataframe(df_ant, width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        c1.download_button(
            "📥 Baixar Tabela CSV (Antenas)",
            data=df_ant.to_csv(index=False).encode("utf-8"),
            file_name="kpis_antenas.csv",
            mime="text/csv",
            width="stretch",
        )
        c2.download_button(
            "📥 Baixar Tabela JSON (Antenas)",
            data=df_ant.to_json(orient="records", indent=2).encode("utf-8"),
            file_name="kpis_antenas.json",
            mime="application/json",
            width="stretch",
        )
        st.divider()

    if mc_result.power_metrics:
        ant_ref = MODEL_LABELS.get(
            mc_result.config.snapshot_antenna_model, mc_result.config.snapshot_antenna_model
        )
        st.markdown(f"#### ⚡ Algoritmos de Controle de Potência (Antena fixa: {ant_ref})")
        fig_pwr = build_kpi_2x2_figure(
            mc_result.power_metrics,
            mc_result.k_scenarios,
            title=f"KPIs 2×2 — Controle de Potência ({ant_ref})",
        )
        st.plotly_chart(fig_pwr, width="stretch", key="tab2_mc_kpi_pwr")

        df_pwr = _metrics_to_dataframe(mc_result.power_metrics, mc_result.k_scenarios)
        st.dataframe(df_pwr, width="stretch", hide_index=True)
        c3, c4 = st.columns(2)
        c3.download_button(
            "📥 Baixar Tabela CSV (Potência)",
            data=df_pwr.to_csv(index=False).encode("utf-8"),
            file_name="kpis_potencia.csv",
            mime="text/csv",
            width="stretch",
        )
        c4.download_button(
            "📥 Baixar Tabela JSON (Potência)",
            data=df_pwr.to_json(orient="records", indent=2).encode("utf-8"),
            file_name="kpis_potencia.json",
            mime="application/json",
            width="stretch",
        )

        # Preview da convergência de uma realização salva no Monte Carlo
        sample_hist = None
        for k_val in reversed(mc_result.k_scenarios):
            h_list = mc_result.convergence_histories.get(k_val)
            if h_list:
                sample_hist = h_list[0]
                break
        if sample_hist is not None:
            st.markdown("##### 🔬 Convergência Representativa no Monte Carlo")
            fig_p, fig_s, _ = build_convergence_figures(sample_hist, power_unit="dbm", ue_filter="all")
            cc1, cc2 = st.columns(2)
            cc1.plotly_chart(fig_p, width="stretch", key="tab2_mc_conv_p")
            cc2.plotly_chart(fig_s, width="stretch", key="tab2_mc_conv_s")

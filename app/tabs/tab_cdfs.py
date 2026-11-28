# ---------------------------------------------------------------------------
# app/tabs/tab_cdfs.py — Aba 3: CDFs Comparativas (Antenas e/ou Potência)
# ---------------------------------------------------------------------------

from __future__ import annotations

import streamlit as st

from app.components.charts import build_cdfs_figure
from simulator.config import MODEL_LABELS
from simulator.schemas import MonteCarloResult


def render_tab_cdfs(mc_result: MonteCarloResult | None) -> None:
    """
    Renderiza CDFs comparativas de Potência, SINR (dB) e Capacidade/SE por carga K,
    exibindo automaticamente todos os grupos disponíveis (Antenas e/ou Controle de Potência).
    """
    st.subheader("📈 CDFs Empíricas Comparativas (Monte Carlo)")
    if mc_result is None:
        st.info("Execute a simulação de **Monte Carlo** na barra lateral para visualizar as CDFs comparativas.")
        return

    c1, c2, c3 = st.columns([2.2, 1.4, 1.0])
    k_choices = c1.multiselect(
        "Cargas K a exibir",
        options=list(mc_result.k_scenarios),
        default=list(mc_result.k_scenarios),
        format_func=lambda k: f"K = {k} UEs",
        key="tab2_k_filter",
    )
    cap_metric = c2.selectbox(
        "Métrica de Capacidade",
        options=["auto", "se", "mbps"],
        format_func=lambda k: {
            "auto": "Automático (Mbps Antenas / SE Potência)",
            "se": "Eficiência Espectral (bps/Hz)",
            "mbps": "Capacidade (Mbps)",
        }[k],
        key="tab2_cap_metric",
    )
    show_refs = c3.checkbox("Linhas de Referência", value=True, key="tab2_refs")

    if not k_choices:
        st.warning("Selecione pelo menos uma carga K.")
        return

    if mc_result.antenna_metrics:
        pwr_ref = MODEL_LABELS.get(
            mc_result.config.snapshot_power_mode, mc_result.config.snapshot_power_mode
        )
        st.markdown(f"### 📡 Comparativo entre Modelos de Antena (Potência fixa: {pwr_ref})")
        for K in k_choices:
            fig = build_cdfs_figure(
                mc_result.antenna_metrics,
                K=int(K),
                gamma_min_db=mc_result.config.power_control.gamma_min_db,
                gamma_tar_db=mc_result.config.power_control.gamma_tar_init_db,
                gamma_max_db=mc_result.config.power_control.gamma_max_db,
                show_ref_lines=show_refs,
                power_metric="rx_dbm",
                cap_metric="mbps" if cap_metric == "auto" else cap_metric,
                title=f"CDFs de Antenas — Carga K = {K} UEs ({mc_result.config.monte_carlo_runs} rodadas)",
            )
            st.plotly_chart(fig, width="stretch", key=f"tab3_cdf_ant_k{K}")

    if mc_result.power_metrics:
        if mc_result.antenna_metrics:
            st.divider()
        ant_ref = MODEL_LABELS.get(
            mc_result.config.snapshot_antenna_model, mc_result.config.snapshot_antenna_model
        )
        st.markdown(f"### ⚡ Comparativo entre Algoritmos de Controle de Potência (Antena fixa: {ant_ref})")
        for K in k_choices:
            fig = build_cdfs_figure(
                mc_result.power_metrics,
                K=int(K),
                gamma_min_db=mc_result.config.power_control.gamma_min_db,
                gamma_tar_db=mc_result.config.power_control.gamma_tar_init_db,
                gamma_max_db=mc_result.config.power_control.gamma_max_db,
                show_ref_lines=show_refs,
                power_metric="tx_dbm",
                cap_metric="se" if cap_metric == "auto" else cap_metric,
                title=f"CDFs de Controle de Potência — Carga K = {K} UEs ({mc_result.config.monte_carlo_runs} rodadas)",
            )
            st.plotly_chart(fig, width="stretch", key=f"tab3_cdf_pwr_k{K}")

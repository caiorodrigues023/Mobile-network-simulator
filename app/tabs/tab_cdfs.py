# ---------------------------------------------------------------------------
# app/tabs/tab_cdfs.py — Aba 2: CDFs Comparativas Detalhadas
# ---------------------------------------------------------------------------

from __future__ import annotations

import streamlit as st

from app.components.charts import build_cdfs_figure
from simulator.config import MODEL_LABELS
from simulator.schemas import MonteCarloResult


def render_tab_cdfs(mc_result: MonteCarloResult | None) -> None:
    """Renderiza CDFs comparativas de Potência, SINR (dB) e Capacidade/SE por carga K."""
    st.subheader("📈 CDFs Empíricas Comparativas (Monte Carlo)")
    if mc_result is None:
        st.info("Execute a simulação de **Monte Carlo** na barra lateral para visualizar as CDFs comparativas.")
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

    c1, c2, c3, c4 = st.columns([2.2, 1.4, 1.4, 1.2])
    grp_idx = (
        c1.selectbox(
            "Grupo de Comparação",
            options=list(range(len(groups))),
            format_func=lambda i: groups[i][1],
            key="tab2_group",
        )
        if len(groups) > 1
        else 0
    )
    grp_key, _, grp_metrics = groups[grp_idx]

    power_metric = c2.selectbox(
        "Métrica de Potência",
        options=["tx_dbm", "rx_dbm"],
        index=0 if grp_key == "power" else 1,
        format_func=lambda k: "Potência Tx (dBm)" if k == "tx_dbm" else "Potência Rx (dBm)",
        key="tab2_pwr_metric",
    )
    cap_metric = c3.selectbox(
        "Métrica de Capacidade",
        options=["se", "mbps"],
        index=0 if grp_key == "power" else 1,
        format_func=lambda k: "Eficiência Espectral (bps/Hz)" if k == "se" else "Capacidade (Mbps)",
        key="tab2_cap_metric",
    )
    show_refs = c4.checkbox("Linhas de Referência", value=True, key="tab2_refs")

    k_choices = st.multiselect(
        "Cargas K a exibir",
        options=list(mc_result.k_scenarios),
        default=list(mc_result.k_scenarios),
        format_func=lambda k: f"K = {k} UEs",
        key="tab2_k_filter",
    )
    if not k_choices:
        st.warning("Selecione pelo menos uma carga K.")
        return

    for K in k_choices:
        fig = build_cdfs_figure(
            grp_metrics,
            K=int(K),
            gamma_min_db=mc_result.config.power_control.gamma_min_db,
            gamma_tar_db=mc_result.config.power_control.gamma_tar_init_db,
            gamma_max_db=mc_result.config.power_control.gamma_max_db,
            show_ref_lines=show_refs,
            power_metric=power_metric,
            cap_metric=cap_metric,
            title=f"CDFs Empíricas — Carga K = {K} UEs ({mc_result.config.monte_carlo_runs} rodadas)",
        )
        st.plotly_chart(fig, width="stretch")

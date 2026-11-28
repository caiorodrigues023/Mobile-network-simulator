# ---------------------------------------------------------------------------
# app/tabs/tab_snapshot.py — TELA 1: Snapshot Interativo + Resultados Monte Carlo
# ---------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import streamlit as st

from app.components.charts import (
    build_antenna_baseline_gain_figure,
    build_cdfs_figure,
    build_kpi_2x2_figure,
    build_snapshot_comparison_figure,
    build_snapshot_map_2d,
    build_snapshot_map_3d,
)
from app.components.sidebar import ANTENNA_OPTIONS, POWER_OPTIONS
from simulator.config import MODEL_LABELS
from simulator.runner import run_single_snapshot, run_snapshot_comparison
from simulator.schemas import MonteCarloResult, SimulationConfig, SnapshotResult


def render_snapshot_summary_cards(snap: SnapshotResult) -> None:
    """
    Renderiza os 2 blocos executivos trazidos da branch `interface`:
      - Desempenho Agregado (Vazão Total, Sum-SE, UEs Ativos, Justiça de Jain, Outage)
      - Qualidade de Sinal & Energia (SINR Médio, SINR 10º Pct Borda, Potência Tx, EE, Canais)
    """
    active_cnt = sum(1 for u in snap.ues if not u.is_blocked)
    outage_cnt = sum(1 for u in snap.ues if u.is_outage)

    b1, b2 = st.columns(2)
    with b1.container(border=True):
        st.caption(f"📊 **Desempenho Agregado** — Outage: {snap.outage_rate_pct:.1f}% ({outage_cnt}/{len(snap.ues)} UEs)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sum-Capacity", f"{snap.sum_capacity_mbps:.1f} Mbps")
        c2.metric("Sum-SE", f"{snap.sum_se_bps_hz:.2f} bps/Hz")
        c3.metric("UEs Ativos", f"{active_cnt} / {len(snap.ues)}")
        c4.metric("Justiça Jain", f"{snap.jain_fairness:.3f}")

    with b2.container(border=True):
        st.caption(f"📶 **Qualidade de Sinal & Energia** — N = {snap.N} canal(is) | ASE: {snap.ase_bps_hz_km2:.2f} bps/Hz/km²")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("SINR Médio", f"{snap.avg_sinr_db:+.1f} dB")
        c6.metric("SINR 10º Pct", f"{snap.p10_sinr_db:+.1f} dB")
        c7.metric("Potência Tx", f"{snap.total_tx_power_w:.3f} W")
        c8.metric("Efic. Energ.", f"{snap.energy_efficiency_mbit_j:.1f} Mb/J")


def render_tab_snapshot(
    config: SimulationConfig,
    mc_result: MonteCarloResult | None,
) -> SnapshotResult:
    """
    Tela 1:
      - Controles rápidos de inspeção (antena, potência, sortear seed, camadas, 2D/3D)
      - Cards agrupados (Desempenho Agregado + Qualidade de Sinal & Energia)
      - Mapa Interativo Plotly (hover rico em UEs e APs; linhas/feixes com hoverinfo='skip')
      - Comparativo instantâneo no Snapshot (quando >1 antena ou >1 potência selecionada)
      - Logo abaixo, todos os gráficos comparativos do Monte Carlo (Antenas e/ou Potência)
    """
    if "snap_seed_active" not in st.session_state:
        st.session_state["snap_seed_active"] = int(config.seed)
    if "snap_last_sidebar_seed" not in st.session_state:
        st.session_state["snap_last_sidebar_seed"] = int(config.seed)
    elif st.session_state["snap_last_sidebar_seed"] != int(config.seed):
        st.session_state["snap_seed_active"] = int(config.seed)
        st.session_state["snap_last_sidebar_seed"] = int(config.seed)

    # ------------------------------------------------------------------
    # Controles rápidos acima do mapa
    # ------------------------------------------------------------------
    c1, c2, c3, c4 = st.columns([2.2, 2.2, 1.4, 1.6])

    if "tab1_ant_select" not in st.session_state or st.session_state["tab1_ant_select"] not in ANTENNA_OPTIONS:
        st.session_state["tab1_ant_select"] = (
            config.snapshot_antenna_model
            if config.snapshot_antenna_model in ANTENNA_OPTIONS
            else ANTENNA_OPTIONS[0]
        )
    sel_ant = c1.selectbox(
        "Modelo de Antena (Snapshot)",
        options=ANTENNA_OPTIONS,
        format_func=lambda k: MODEL_LABELS.get(k, k),
        key="tab1_ant_select",
    )

    if "tab1_pwr_select" not in st.session_state or st.session_state["tab1_pwr_select"] not in POWER_OPTIONS:
        st.session_state["tab1_pwr_select"] = (
            config.snapshot_power_mode
            if config.snapshot_power_mode in POWER_OPTIONS
            else POWER_OPTIONS[0]
        )
    sel_pwr = c2.selectbox(
        "Controle de Potência (Snapshot)",
        options=POWER_OPTIONS,
        format_func=lambda k: MODEL_LABELS.get(k, k),
        key="tab1_pwr_select",
    )

    c3.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if c3.button("🎲 Novo Snapshot (Seed)", width="stretch"):
        st.session_state["snap_seed_active"] = int(np.random.default_rng().integers(0, 1_000_000))
        st.session_state.pop("snap_k_override", None)

    view_mode = c4.radio(
        "Visualização",
        options=["2D (Padrão)", "3D (Bônus)"],
        horizontal=True,
        key="tab1_view_mode",
    )
    is_3d = "3D" in view_mode

    # Camadas locais do mapa
    l1, l2, l3, l4, l5 = st.columns([1.3, 1.3, 1.4, 1.4, 1.5])
    show_beams = l1.checkbox("Exibir Lóbulos", value=True, key="tab1_show_beams")
    show_links = l2.checkbox("Exibir Enlaces UE↔AP", value=True, key="tab1_show_links")
    show_heatmap = l3.checkbox("Heatmap Cobertura", value=is_3d, key="tab1_show_heatmap")
    show_interf = l4.checkbox(
        "Interferência (Pior UE)", value=False, disabled=is_3d, key="tab1_show_interf"
    )
    hm_metric = l5.selectbox(
        "Métrica Heatmap",
        options=["sinr", "rsrp", "cap", "server"],
        format_func=lambda m: {
            "sinr": "SINR (dB)",
            "rsrp": "RSRP (dBm)",
            "cap": "Capacidade (Mbps)",
            "server": "AP Dominante",
        }[m],
        disabled=not show_heatmap,
        label_visibility="collapsed",
        key="tab1_hm_metric",
    )

    # ------------------------------------------------------------------
    # Execução do Snapshot atual (rápida — milissegundos)
    # ------------------------------------------------------------------
    active_seed = int(st.session_state["snap_seed_active"])
    k_override = st.session_state.get("snap_k_override")
    snap = run_single_snapshot(
        config=config,
        antenna_model=sel_ant,
        power_mode=sel_pwr,
        seed=active_seed,
        K_override=k_override,
        include_heatmap=show_heatmap,
        heatmap_metric=hm_metric,
        heatmap_res=40,
    )
    st.session_state["current_snapshot"] = snap

    # Cards estruturados (Desempenho Agregado + Qualidade de Sinal & Energia)
    render_snapshot_summary_cards(snap)

    # Mapa Interativo (2D padrão ou 3D bônus)
    if is_3d:
        fig_map = build_snapshot_map_3d(
            snap, show_beams=show_beams, show_links=show_links, show_heatmap=show_heatmap
        )
    else:
        fig_map = build_snapshot_map_2d(
            snap,
            show_beams=show_beams,
            show_links=show_links,
            show_heatmap=show_heatmap,
            show_interference=show_interf,
        )
    st.plotly_chart(fig_map, width="stretch", key="tab1_snapshot_map")

    # Comparativo instantâneo no mesmo Snapshot (quando >1 antena ou >1 algoritmo selecionado)
    has_multi_ant = len(config.antenna.models) > 1
    has_multi_pwr = len(config.power_control.algorithms) > 1
    if has_multi_ant or has_multi_pwr:
        with st.expander(
            f"⚡ Comparar Selecionados neste Mesmo Snapshot (Seed={snap.seed})",
            expanded=False,
        ):
            comp = run_snapshot_comparison(
                config,
                seed=snap.seed,
                K_override=len(snap.ues),
                base_antenna=sel_ant,
                base_power=sel_pwr,
            )
            if comp["antennas"]:
                fig_ca = build_snapshot_comparison_figure(
                    comp["antennas"],
                    title=f"📡 Antenas Selecionadas no Mesmo Canal (Potência: {MODEL_LABELS.get(sel_pwr, sel_pwr)})",
                )
                st.plotly_chart(fig_ca, width="stretch", key="tab1_snap_comp_ant")
            if comp["power"]:
                fig_cp = build_snapshot_comparison_figure(
                    comp["power"],
                    title=f"⚡ Algoritmos de Potência no Mesmo Canal (Antena: {MODEL_LABELS.get(sel_ant, sel_ant)})",
                )
                st.plotly_chart(fig_cp, width="stretch", key="tab1_snap_comp_pwr")

    # ------------------------------------------------------------------
    # RESULTADOS DO MONTE CARLO NA MESMA TELA (LOGO ABAIXO DO SNAPSHOT)
    # ------------------------------------------------------------------
    st.divider()
    if mc_result is None:
        st.info(
            "Configure **Monte Carlo Runs**, selecione as **Antenas** e/ou **Algoritmos de Controle de Potência** "
            "na barra lateral e clique em **▶️ Executar Monte Carlo** para gerar os gráficos comparativos diretamente aqui abaixo. "
            "Você também já pode ver os gráficos por UE e por AP deste Snapshot na aba **📊 KPIs & Gráficos**."
        )
        return snap

    st.subheader(
        f"📊 Resultados do Monte Carlo ({mc_result.config.monte_carlo_runs} rodadas × "
        f"K ∈ {list(mc_result.k_scenarios)} — concluído em {mc_result.elapsed_s:.2f} s)"
    )

    k_cdf = st.selectbox(
        "Carga K para inspeção rápida das CDFs nesta tela",
        options=list(mc_result.k_scenarios),
        index=len(mc_result.k_scenarios) - 1,
        format_func=lambda k: f"K = {k} UEs",
        key="tab1_mc_k_cdf",
    )

    # Renderiza AMBOS os grupos comparativos quando existirem (Antenas e/ou Controle de Potência)
    if mc_result.antenna_metrics:
        pwr_ref = MODEL_LABELS.get(
            mc_result.config.snapshot_power_mode, mc_result.config.snapshot_power_mode
        )
        st.markdown(f"#### 📡 Comparativo entre Modelos de Antena (Potência fixa: {pwr_ref})")
        fig_kpi_ant = build_kpi_2x2_figure(
            mc_result.antenna_metrics,
            mc_result.k_scenarios,
            title=f"Painel de KPIs — Antenas (Potência: {pwr_ref})",
        )
        st.plotly_chart(fig_kpi_ant, width="stretch", key="tab1_mc_kpi_ant")

        if len(mc_result.antenna_metrics) > 1:
            fig_base = build_antenna_baseline_gain_figure(
                mc_result.antenna_metrics, mc_result.k_scenarios, k_focus=int(k_cdf)
            )
            st.plotly_chart(fig_base, width="stretch", key="tab1_mc_base_ant")

        fig_cdf_ant = build_cdfs_figure(
            mc_result.antenna_metrics,
            K=int(k_cdf),
            gamma_min_db=mc_result.config.power_control.gamma_min_db,
            gamma_tar_db=mc_result.config.power_control.gamma_tar_init_db,
            gamma_max_db=mc_result.config.power_control.gamma_max_db,
            power_metric="rx_dbm",
            cap_metric="mbps",
            title=f"CDFs Comparativas de Antenas — K = {k_cdf} UEs",
        )
        st.plotly_chart(fig_cdf_ant, width="stretch", key="tab1_mc_cdf_ant")

    if mc_result.power_metrics:
        if mc_result.antenna_metrics:
            st.divider()
        ant_ref = MODEL_LABELS.get(
            mc_result.config.snapshot_antenna_model, mc_result.config.snapshot_antenna_model
        )
        st.markdown(f"#### ⚡ Comparativo entre Algoritmos de Controle de Potência (Antena fixa: {ant_ref})")
        fig_kpi_pwr = build_kpi_2x2_figure(
            mc_result.power_metrics,
            mc_result.k_scenarios,
            title=f"Painel de KPIs — Controle de Potência (Antena: {ant_ref})",
        )
        st.plotly_chart(fig_kpi_pwr, width="stretch", key="tab1_mc_kpi_pwr")

        fig_cdf_pwr = build_cdfs_figure(
            mc_result.power_metrics,
            K=int(k_cdf),
            gamma_min_db=mc_result.config.power_control.gamma_min_db,
            gamma_tar_db=mc_result.config.power_control.gamma_tar_init_db,
            gamma_max_db=mc_result.config.power_control.gamma_max_db,
            power_metric="tx_dbm",
            cap_metric="se",
            title=f"CDFs Comparativas de Controle de Potência — K = {k_cdf} UEs",
        )
        st.plotly_chart(fig_cdf_pwr, width="stretch", key="tab1_mc_cdf_pwr")

    return snap

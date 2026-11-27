# ---------------------------------------------------------------------------
# app/tabs/tab_snapshot.py — TELA 1: Snapshot Interativo + Resultados Monte Carlo
# ---------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.components.charts import (
    build_cdfs_figure,
    build_kpi_2x2_figure,
    build_snapshot_map_2d,
    build_snapshot_map_3d,
)
from app.components.sidebar import ANTENNA_OPTIONS, POWER_OPTIONS
from simulator.config import MODEL_LABELS
from simulator.runner import run_single_snapshot
from simulator.schemas import MonteCarloResult, SimulationConfig, SnapshotResult


def render_tab_snapshot(
    config: SimulationConfig,
    mc_result: MonteCarloResult | None,
) -> SnapshotResult:
    """
    Tela 1:
      - Controles rápidos de inspeção (antena, potência, sortear seed, camadas, 2D/3D)
      - Cards de métricas do snapshot atual
      - Mapa Interativo Plotly (hover rico em UEs e APs; linhas/feixes com hoverinfo='skip')
      - Logo abaixo (sempre mantendo o snapshot no topo), os gráficos comparativos do Monte Carlo!
    """
    # Estado de inspeção local do snapshot
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

    # Inspetor de Rodadas do Monte Carlo (quando houver MC concluído)
    if mc_result is not None and mc_result.round_inspector:
        ri = mc_result.round_inspector
        presets = ri["presets"]
        with st.expander(
            f"🔎 Inspetor de Rodadas do Monte Carlo (K={ri['K']} UEs, {ri['n_runs']} rodadas)",
            expanded=False,
        ):
            r1, r2, r3 = st.columns([2.5, 1.5, 1.5])
            preset_opts = {
                "current": "📍 Seed Atual do Snapshot",
                "first": f"1ª Rodada (#{presets['first']['round']}) — {presets['first']['sum_cap_mbps']:.1f} Mbps",
                "last": f"Última Rodada (#{presets['last']['round']}) — {presets['last']['sum_cap_mbps']:.1f} Mbps",
                "best": f"🏆 Melhor Rodada (#{presets['best']['round']}) — {presets['best']['sum_cap_mbps']:.1f} Mbps",
                "median": f"⚖️ Rodada Mediana (#{presets['median']['round']}) — {presets['median']['sum_cap_mbps']:.1f} Mbps",
                "worst": f"⚠️ Pior Rodada (#{presets['worst']['round']}) — {presets['worst']['sum_cap_mbps']:.1f} Mbps",
            }
            chosen_preset = r1.selectbox(
                "Rodada-chave",
                options=list(preset_opts.keys()),
                format_func=lambda k: preset_opts[k],
                key="tab1_mc_preset",
            )
            custom_round = r2.number_input(
                "Ou rodada #k",
                min_value=1,
                max_value=int(ri["n_runs"]),
                value=1,
                step=1,
                key="tab1_mc_round_k",
            )
            r3.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if r3.button("📍 Carregar Rodada no Mapa", width="stretch"):
                if chosen_preset != "current":
                    st.session_state["snap_seed_active"] = int(presets[chosen_preset]["seed"])
                else:
                    st.session_state["snap_seed_active"] = int(
                        ri["all_seeds"][int(custom_round) - 1]
                    )
                st.session_state["snap_k_override"] = int(ri["K"])

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

    # ------------------------------------------------------------------
    # Cards resumo do Snapshot atual
    # ------------------------------------------------------------------
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("SINR Média (Ativos)", f"{snap.avg_sinr_db:+.1f} dB", f"10º Pct: {snap.p10_sinr_db:+.1f} dB")
    m2.metric("Sum-Capacity", f"{snap.sum_capacity_mbps:.1f} Mbps", f"Sum-SE: {snap.sum_se_bps_hz:.2f} bps/Hz")
    m3.metric("Potência Tx Total", f"{snap.total_tx_power_w:.3f} W", f"Jain: {snap.jain_fairness:.3f}")
    m4.metric("Eficiência Energética", f"{snap.energy_efficiency_mbit_j:.1f} Mb/J", f"ASE: {snap.ase_bps_hz_km2:.2f}")
    m5.metric(
        "Taxa de Outage",
        f"{snap.outage_rate_pct:.1f}%",
        f"{sum(u.is_outage for u in snap.ues)}/{len(snap.ues)} UEs",
        delta_color="inverse",
    )

    # ------------------------------------------------------------------
    # Mapa Interativo (2D padrão ou 3D bônus)
    # ------------------------------------------------------------------
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
    st.plotly_chart(fig_map, width="stretch")

    # Tabelas detalhadas do Snapshot em expander
    with st.expander("📋 Inspecionar Tabelas de UEs e APs deste Snapshot", expanded=False):
        t_ue, t_ap = st.tabs(["📱 UEs", "📡 APs"])
        df_ues = pd.DataFrame(
            [
                {
                    "UE": f"UE {u.ue_id}",
                    "X (m)": round(u.x, 1),
                    "Y (m)": round(u.y, 1),
                    "AP": f"AP {u.serving_ap}",
                    "Dist (m)": round(u.distance_m, 1),
                    "AoA (°)": round(u.aoa_deg, 1),
                    "Canal": "CA" if u.is_aggregated else u.channel,
                    "Feixe": u.serving_beam,
                    "P_tx (W)": round(u.tx_power_w, 4),
                    "P_tx (dBm)": None if u.is_blocked else round(u.tx_power_dbm, 1),
                    "Ganho (dB)": round(u.rx_gain_db, 1),
                    "SINR (dB)": None if u.is_blocked else round(u.sinr_db, 2),
                    "SE (bps/Hz)": round(u.se_bps_hz, 2),
                    "Cap (Mbps)": round(u.capacity_mbps, 2),
                    "γ_tar (dB)": None if u.gamma_tar_db is None else round(u.gamma_tar_db, 1),
                    "Status": u.status,
                }
                for u in snap.ues
            ]
        )
        t_ue.dataframe(df_ues, width="stretch", hide_index=True)
        t_ue.download_button(
            "📥 Exportar UEs (CSV)",
            data=df_ues.to_csv(index=False).encode("utf-8"),
            file_name=f"snapshot_ues_seed{snap.seed}.csv",
            mime="text/csv",
        )

        df_aps = pd.DataFrame(
            [
                {
                    "AP": f"AP {ap.ap_id}",
                    "X (m)": round(ap.x, 1),
                    "Y (m)": round(ap.y, 1),
                    "Qtd. UEs": ap.num_ues,
                    "UEs Conectados": (
                        ", ".join(f"UE {uid}" for uid in ap.connected_ue_ids)
                        if ap.connected_ue_ids
                        else "Nenhum"
                    ),
                    "Sum-Capacity (Mbps)": round(ap.sum_capacity_mbps, 2),
                    "UEs em Outage": ap.num_outage,
                }
                for ap in snap.aps
            ]
        )
        t_ap.dataframe(df_aps, width="stretch", hide_index=True)

    # ------------------------------------------------------------------
    # RESULTADOS DO MONTE CARLO NA MESMA TELA (LOGO ABAIXO DO SNAPSHOT)
    # ------------------------------------------------------------------
    st.divider()
    if mc_result is None:
        st.info(
            "Configure **Monte Carlo Runs** na barra lateral e clique em **▶️ Executar Monte Carlo** "
            "para gerar todos os gráficos estatísticos comparativos diretamente aqui abaixo do Snapshot."
        )
        return snap

    st.subheader(
        f"📊 Resultados do Monte Carlo ({mc_result.config.monte_carlo_runs} rodadas × "
        f"K ∈ {list(mc_result.k_scenarios)} — concluído em {mc_result.elapsed_s:.2f} s)"
    )

    available_groups: list[tuple[str, str, dict]] = []
    if mc_result.power_metrics:
        available_groups.append(
            (
                "power",
                f"⚡ Controle de Potência (Antena: {MODEL_LABELS.get(mc_result.config.snapshot_antenna_model, mc_result.config.snapshot_antenna_model)})",
                mc_result.power_metrics,
            )
        )
    if mc_result.antenna_metrics:
        available_groups.append(
            (
                "antennas",
                f"📡 Modelos de Antena (Potência: {MODEL_LABELS.get(mc_result.config.snapshot_power_mode, mc_result.config.snapshot_power_mode)})",
                mc_result.antenna_metrics,
            )
        )

    if not available_groups:
        return snap

    if len(available_groups) == 2:
        grp_sel = st.radio(
            "Grupo Comparativo (Monte Carlo na Tela 1)",
            options=[0, 1],
            format_func=lambda idx: available_groups[idx][1],
            horizontal=True,
            key="tab1_mc_group",
        )
    else:
        grp_sel = 0

    grp_key, grp_title, grp_metrics = available_groups[grp_sel]

    # Painel 2x2 de KPIs comparativos
    fig_kpi = build_kpi_2x2_figure(grp_metrics, mc_result.k_scenarios, title=f"Painel de KPIs — {grp_title}")
    st.plotly_chart(fig_kpi, width="stretch")

    # CDFs principais do cenário selecionado
    k_cdf = st.selectbox(
        "Carga K para inspeção rápida das CDFs nesta tela",
        options=list(mc_result.k_scenarios),
        index=len(mc_result.k_scenarios) - 1,
        format_func=lambda k: f"K = {k} UEs",
        key="tab1_mc_k_cdf",
    )
    fig_cdf = build_cdfs_figure(
        grp_metrics,
        K=int(k_cdf),
        gamma_min_db=mc_result.config.power_control.gamma_min_db,
        gamma_tar_db=mc_result.config.power_control.gamma_tar_init_db,
        gamma_max_db=mc_result.config.power_control.gamma_max_db,
        power_metric="tx_dbm" if grp_key == "power" else "rx_dbm",
        cap_metric="se" if grp_key == "power" else "mbps",
    )
    st.plotly_chart(fig_cdf, width="stretch")

    return snap

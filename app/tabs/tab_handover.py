# ---------------------------------------------------------------------------
# app/tabs/tab_handover.py — Aba 7: Handover (Hard vs. Soft) na REDE COMPLETA
# ---------------------------------------------------------------------------

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.charts import (
    build_network_handover_aggregate_figure,
    build_network_handover_map_figure,
    build_network_handover_ue_detail_figure,
)
from simulator.handover import run_network_handover
from simulator.schemas import SimulationConfig, SnapshotResult


@st.cache_data(show_spinner=False)
def _cached_network_handover(config: SimulationConfig, seed: int, K: int):
    return run_network_handover(config, seed=seed, K_override=K)


def render_tab_handover(config: SimulationConfig, snapshot: SnapshotResult) -> None:
    """
    Aba 7 (visível apenas quando Modo do UE = 'Em Movimento'):
    Executa Hard vs. Soft Handover para TODOS os K UEs cruzando a grade completa de M APs,
    mostrando primeiro o agregado da rede inteira e abaixo a inspeção de qualquer UE individual.
    """
    ho = config.handover
    K_active = len(snapshot.ues)
    seed_active = int(snapshot.seed)

    st.subheader("🔄 Handover (Hard vs. Soft) — Visão da Rede Completa")
    st.caption(
        f"Escopo: **todos os {config.M} APs e {K_active} UEs** em movimento por T_ho = {ho.t_ho_s:g} s "
        f"(T_s = {ho.ho_step_s:g} s, Seed={seed_active}) | v = {config.mobility.selected_velocity_kmh:g} km/h | "
        f"P_min = {ho.p_min_dbm_network:g} dBm | Histerese = {ho.hysteresis_db:g} dB | "
        f"TTT = {ho.ttt_ms:g} ms | Soft = {ho.soft_combining}"
    )

    if config.M < 2:
        st.warning(
            "Com **M = 1 AP** não existem estações base vizinhas para realizar Handover. "
            "Selecione **M ≥ 4 APs** na seção *Topologia & Recursos* da barra lateral."
        )

    with st.spinner("Simulando Handover (Hard vs. Soft) para todos os UEs da rede..."):
        res = _cached_network_handover(config, seed_active, K_active)

    # ------------------------------------------------------------------
    # 1. Cards Agregados de Todos os K UEs
    # ------------------------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Outage Rede (Hard)",
        f"{res.hard_outage_rate_pct:.2f}%",
        f"Total ev.: {int(res.hard_outage_mask.sum())}",
    )
    c2.metric(
        "Outage Rede (Soft)",
        f"{res.soft_outage_rate_pct:.2f}%",
        f"{res.soft_outage_rate_pct - res.hard_outage_rate_pct:+.2f} pp vs. Hard",
        delta_color="inverse",
    )
    c3.metric(
        "Handovers Hard (Total)",
        f"{res.hard_handovers_total}",
        f"Média: {res.hard_handovers_mean:.1f} / UE",
    )
    c4.metric(
        "Handovers Soft (Total)",
        f"{res.soft_handovers_total}",
        f"Média: {res.soft_handovers_mean:.1f} / UE",
    )
    c5.metric(
        "Ganho Médio SNR (Soft)",
        f"{res.mean_snr_gain_soft_vs_hard_db:+.2f} dB",
        f"Sobre todos os {res.K} UEs",
    )

    # ------------------------------------------------------------------
    # 2. Mapa 2D da Rede Completa com Eventos de Handover e Outage
    # ------------------------------------------------------------------
    algo_map = st.radio(
        "Eventos exibidos no Mapa 2D da Rede",
        options=["hard", "soft"],
        format_func=lambda m: "Hard Handover" if m == "hard" else f"Soft Handover ({ho.soft_combining})",
        horizontal=True,
        key="tab_ho_map_algo",
    )
    fig_map = build_network_handover_map_figure(res, algo_view=algo_map)
    st.plotly_chart(fig_map, width="stretch", key="tab7_ho_map")

    # ------------------------------------------------------------------
    # 3. Painel 2x2 Agregado de Todos os K UEs
    # ------------------------------------------------------------------
    fig_agg = build_network_handover_aggregate_figure(res)
    st.plotly_chart(fig_agg, width="stretch", key="tab7_ho_agg")

    # ------------------------------------------------------------------
    # 4. Inspeção Individual de Qualquer UE
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### 🔎 Inspeção Detalhada de um UE Individual")
    sel_ue = st.selectbox(
        "Escolha o UE para inspecionar as trocas de AP, RSRP e SNR",
        options=list(range(res.K)),
        format_func=lambda k: (
            f"UE {k} ({res.ue_velocities_kmh[k]:.0f} km/h) — "
            f"HO Hard: {int(res.hard_handovers_per_ue[k])} | Outage Hard: {res.hard_outage_pct_per_ue[k]:.1f}%"
        ),
        key="tab_ho_ue_select",
    )
    fig_dec, fig_sig = build_network_handover_ue_detail_figure(res, ue_id=int(sel_ue))
    st.plotly_chart(fig_dec, width="stretch", key="tab7_ho_ue_dec")
    st.plotly_chart(fig_sig, width="stretch", key="tab7_ho_ue_sig")

    # ------------------------------------------------------------------
    # 5. Tabela Resumo por UE + Exportação CSV
    # ------------------------------------------------------------------
    df_ho = pd.DataFrame(
        [
            {
                "UE": f"UE {k}",
                "Velocidade (km/h)": round(float(res.ue_velocities_kmh[k]), 1),
                "Handovers (Hard)": int(res.hard_handovers_per_ue[k]),
                "Handovers (Soft)": int(res.soft_handovers_per_ue[k]),
                "Outage Hard [%]": round(float(res.hard_outage_pct_per_ue[k]), 2),
                "Outage Soft [%]": round(float(res.soft_outage_pct_per_ue[k]), 2),
                "SNR Média Hard [dB]": round(float(res.hard_snr_db[:, k].mean()), 2),
                "SNR Média Soft [dB]": round(float(res.soft_snr_db[:, k].mean()), 2),
            }
            for k in range(res.K)
        ]
    )
    with st.expander("📋 Tabela Detalhada de Handover por UE", expanded=False):
        st.dataframe(df_ho, width="stretch", hide_index=True)
        st.download_button(
            "📥 Exportar Resumo de Handover (CSV)",
            data=df_ho.to_csv(index=False).encode("utf-8"),
            file_name=f"handover_network_seed{seed_active}.csv",
            mime="text/csv",
            width="stretch",
        )

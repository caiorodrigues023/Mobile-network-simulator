# ---------------------------------------------------------------------------
# app/tabs/tab_mobility.py — Aba 6: Mobilidade & Beam Tracking na REDE COMPLETA
# ---------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.components.charts import (
    build_mobility_tradeoff_figure,
    build_network_mobility_overview_figure,
    build_network_mobility_timeseries_figure,
    build_network_sdma_figure,
)
from simulator.mobility import run_network_mobility, run_network_mobility_tradeoff
from simulator.schemas import SimulationConfig, SnapshotResult


@st.cache_data(show_spinner=False)
def _cached_network_mobility(config: SimulationConfig, seed: int, K: int):
    return run_network_mobility(config, seed=seed, K_override=K, compute_sdma=config.mobility.enable_sdma)


@st.cache_data(show_spinner=False)
def _cached_network_tradeoff(config: SimulationConfig, seed: int, K: int):
    return run_network_mobility_tradeoff(config, seed=seed, K_override=K)


def render_tab_mobility(config: SimulationConfig, snapshot: SnapshotResult) -> None:
    """
    Aba 6 (visível apenas quando Modo do UE = 'Em Movimento'):
    Simula TODOS os K UEs da rede se movendo sobre a grade completa de M APs,
    com painel de Perda por Desalinhamento Angular Δθ (L=8/16/32), séries temporais
    Hold vs. Tracking Preditivo (agregado ou por UE), Trade-off ΔT e SDMA (MRC vs. ZF).
    """
    mob = config.mobility
    K_active = len(snapshot.ues)
    seed_active = int(snapshot.seed)

    st.subheader("🚗 Mobilidade, Beam Tracking & SDMA — Visão da Rede Completa")
    st.caption(
        f"Escopo: **todos os {config.M} APs e {K_active} UEs** na área {config.area_m:g}×{config.area_m:g} m "
        f"(Seed={seed_active}) | v = {mob.selected_velocity_kmh:g} km/h"
        + (" (misto)" if mob.mixed_speeds else "")
        + f" | ΔT = {mob.selected_delta_t_s*1e3:g} ms | L = {mob.selected_L} ({mob.array_type.upper()}) "
        f"| σ_θ = {mob.sigma_theta_deg:g}° | dt = {mob.dt_s*1e3:g} ms | T_ovh = {mob.t_overhead_s*1e3:g} ms"
    )

    with st.spinner("Simulando mobilidade e beam tracking de todos os UEs da rede..."):
        res = _cached_network_mobility(config, seed_active, K_active)

    # ------------------------------------------------------------------
    # 1. Cards Executivos (Agregado de Todos os K UEs)
    # ------------------------------------------------------------------
    delta_x_m = float(np.mean(res.ue_velocities_kmh) / 3.6 * res.delta_t_s)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Deslocamento por ΔT (Δx)",
        f"{delta_x_m:.2f} m",
        f"v médio: {np.mean(res.ue_velocities_kmh):.0f} km/h",
    )
    c2.metric(
        "Erro Angular Médio (Hold)",
        f"{res.mean_delta_theta_hold_deg:.2f}°",
        f"Máx (pior UE): {res.max_delta_theta_hold_deg:.2f}°",
    )
    if mob.compare_predictive:
        loss_delta = res.mean_loss_pred_db - res.mean_loss_hold_db
        se_delta = res.mean_sum_se_pred - res.mean_sum_se_hold
        c3.metric(
            f"Perda Média (L={res.L})",
            f"{res.mean_loss_hold_db:.2f} dB",
            f"Preditivo: {res.mean_loss_pred_db:.2f} dB ({loss_delta:+.2f} dB)",
            delta_color="inverse",
        )
        c4.metric(
            f"Sum-SE Líquida ({res.K} UEs)",
            f"{res.mean_sum_se_pred:.2f} bps/Hz",
            f"{se_delta:+.2f} bps/Hz vs. Hold ({res.mean_sum_se_hold:.2f})",
        )
    else:
        c3.metric(f"Perda Média (L={res.L})", f"{res.mean_loss_hold_db:.2f} dB", f"Máx: {res.max_loss_hold_db:.1f} dB")
        c4.metric(f"Sum-SE Líquida ({res.K} UEs)", f"{res.mean_sum_se_hold:.2f} bps/Hz", f"Overhead: {res.overhead_fraction*100:.1f}%")

    # ------------------------------------------------------------------
    # 2. Mapa 2D da Rede Completa + Curva de Perda por Desalinhamento Δθ
    # ------------------------------------------------------------------
    fig_ov = build_network_mobility_overview_figure(res, compare_predictive=mob.compare_predictive)
    st.plotly_chart(fig_ov, width="stretch", key="tab6_mob_overview")

    # ------------------------------------------------------------------
    # 3. Evolução Temporal (Agregado dos K UEs vs. Individual por UE)
    # ------------------------------------------------------------------
    view_mode = st.radio(
        "Modo de Exibição Temporal",
        options=["aggregate", "per_ue"],
        format_func=lambda m: (
            f"📊 Agregado de Todos os {res.K} UEs (Média + Pior UE)"
            if m == "aggregate"
            else f"👤 Curvas Individuais por UE (UE 0 .. UE {res.K - 1})"
        ),
        horizontal=True,
        key="tab_mob_view_mode",
    )
    fig_ts = build_network_mobility_timeseries_figure(
        res, view_mode=view_mode, compare_predictive=mob.compare_predictive
    )
    st.plotly_chart(fig_ts, width="stretch", key="tab6_mob_timeseries")

    # ------------------------------------------------------------------
    # 4. Trade-off ΔT × L na Rede Completa
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### ⚖️ Trade-off ΔT vs. Sum-SE Líquida Agregada da Rede")
    with st.spinner("Calculando trade-off ΔT × L para a rede completa..."):
        tradeoff_rows = _cached_network_tradeoff(config, seed_active, K_active)

    tr_mode = "hold"
    if mob.compare_predictive:
        tr_mode = st.radio(
            "Modo de Rastreamento no Trade-off",
            options=["hold", "predictive"],
            format_func=lambda m: "Zero-Order Hold (ΔT)" if m == "hold" else "Rastreamento Preditivo (α-β)",
            horizontal=True,
            key="tab_mob_tradeoff_mode",
        )
    fig_trade = build_mobility_tradeoff_figure(tradeoff_rows, selected_mode=tr_mode)
    st.plotly_chart(fig_trade, width="stretch", key="tab6_mob_tradeoff")

    with st.expander("📋 Tabela de Trade-off (ΔT × L — Rede Completa)", expanded=False):
        df_tr = pd.DataFrame(tradeoff_rows)
        st.dataframe(df_tr, width="stretch", hide_index=True)
        st.download_button(
            "📥 Exportar Trade-off (CSV)",
            data=df_tr.to_csv(index=False).encode("utf-8"),
            file_name=f"mobility_network_tradeoff_seed{seed_active}.csv",
            mime="text/csv",
            width="stretch",
        )

    # ------------------------------------------------------------------
    # 5. SDMA Multi-usuário Móvel na Rede Completa (MRC vs. Zero-Forcing)
    # ------------------------------------------------------------------
    if mob.enable_sdma:
        st.divider()
        st.markdown(f"### 📶 SDMA Multi-usuário Móvel na Rede Completa (K={res.K} UEs, L={res.L})")
        s1, s2, s3 = st.columns(3)
        s1.metric("Sum-SE Média (MRC)", f"{res.mean_sum_se_mrc:.2f} bps/Hz")
        s2.metric("Sum-SE Média (Zero-Forcing)", f"{res.mean_sum_se_zf:.2f} bps/Hz")
        gain_pct = (
            ((res.mean_sum_se_zf / res.mean_sum_se_mrc) - 1.0) * 100.0
            if res.mean_sum_se_mrc > 0
            else 0.0
        )
        s3.metric("Ganho ZF vs. MRC", f"{gain_pct:+.1f}%", f"{res.mean_sum_se_zf - res.mean_sum_se_mrc:+.2f} bps/Hz")

        fig_sdma = build_network_sdma_figure(res, combiner=mob.sdma_combiner)
        st.plotly_chart(fig_sdma, width="stretch", key="tab6_mob_sdma")

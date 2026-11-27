# ---------------------------------------------------------------------------
# app/tabs/tab_mobility.py — Aba 4: Mobilidade, Tracking Preditivo & SDMA Móvel
# ---------------------------------------------------------------------------

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.charts import (
    build_mobility_sweeps_figure,
    build_mobility_tracking_figure,
    build_mobility_tradeoff_figure,
    build_sdma_figures,
)
from simulator.mobility import run_mobile_sdma, run_mobility_tradeoff_study
from simulator.schemas import ChannelConfig, MobilityConfig, SimulationConfig


@st.cache_data(show_spinner=False)
def _cached_tradeoff_study(
    mob: MobilityConfig, channel_cfg: ChannelConfig, p_tx_w: float
) -> dict:
    return run_mobility_tradeoff_study(mob, channel_cfg=channel_cfg, p_tx_w=p_tx_w)


@st.cache_data(show_spinner=False)
def _cached_mobile_sdma(
    mob: MobilityConfig,
    channel_cfg: ChannelConfig,
    p_tx_w: float,
    tracking_mode: str,
):
    return run_mobile_sdma(
        mob,
        channel_cfg=channel_cfg,
        p_tx_w=p_tx_w,
        tracking_mode=tracking_mode,
    )


def render_tab_mobility(config: SimulationConfig) -> None:
    """Aba 4: Motor Time-Driven (dt=1ms), Hold vs. Preditivo, Trade-off Net SE e SDMA Móvel."""
    mob = config.mobility
    st.subheader("🚗 Mobilidade, Rastreamento Preditivo de Feixes & SDMA Multi-usuário Móvel")
    st.caption(
        f"Cenário ativo: v = {mob.selected_velocity_kmh:g} km/h | ΔT = {mob.selected_delta_t_s*1e3:g} ms | "
        f"D = {mob.selected_distance_m:g} m | L = {mob.selected_L} | σ_θ = {mob.sigma_theta_deg:g}° | "
        f"dt = {mob.dt_s*1e3:g} ms | T_ovh = {mob.t_overhead_s*1e3:g} ms"
    )

    with st.spinner("Executando motor time-driven de mobilidade..."):
        study = _cached_tradeoff_study(mob, config.channel, config.power_control.p_max_w)

    res_hold = study["main_hold"]
    res_pred = study["main_predictive"]

    # Cards resumo comparando Hold vs. Tracking Preditivo
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Perda Média (Hold)", f"{res_hold.mean_loss_db:.2f} dB", f"Máx: {res_hold.max_loss_db:.1f} dB")
    c2.metric(
        "SE Líquida Média (Hold)",
        f"{res_hold.mean_se_net:.2f} bps/Hz",
        f"Overhead: {res_hold.overhead_fraction*100:.1f}%",
    )
    if res_pred is not None:
        loss_red = res_hold.mean_loss_db - res_pred.mean_loss_db
        se_gain = res_pred.mean_se_net - res_hold.mean_se_net
        c3.metric(
            "Perda Média (Preditivo)",
            f"{res_pred.mean_loss_db:.2f} dB",
            f"{-loss_red:+.2f} dB vs. Hold",
            delta_color="inverse",
        )
        c4.metric(
            "SE Líquida (Preditivo)",
            f"{res_pred.mean_se_net:.2f} bps/Hz",
            f"{se_gain:+.2f} bps/Hz vs. Hold",
        )
    else:
        c3.metric("Ganho Ideal ULA", f"{res_hold.gain_ideal_db[0]:.1f} dB", f"L = {mob.selected_L}")
        c4.metric("SE Bruta Média", f"{res_hold.mean_se_gross:.2f} bps/Hz")

    # 1. Array Gain vs. Time (Sawtooth vs. Preditivo vs. Ideal) e Beam Misalignment Loss
    fig_track = build_mobility_tracking_figure(res_hold, res_pred)
    st.plotly_chart(fig_track, width="stretch")

    # 2. Comparativos por Velocidade, L e Distância D
    fig_sweeps = build_mobility_sweeps_figure(study)
    st.plotly_chart(fig_sweeps, width="stretch")

    # 3. Trade-off ΔT vs. Eficiência Espectral Líquida Média
    tradeoff_mode = "hold"
    if mob.compare_predictive:
        tradeoff_mode = st.radio(
            "Modo de Rastreamento no Gráfico de Trade-off",
            options=["hold", "predictive"],
            format_func=lambda m: "Padrão (Zero-Order Hold)" if m == "hold" else "Rastreamento Preditivo (α-β)",
            horizontal=True,
            key="tab4_tradeoff_mode",
        )
    fig_trade = build_mobility_tradeoff_figure(study["tradeoff_rows"], selected_mode=tradeoff_mode)
    st.plotly_chart(fig_trade, width="stretch")

    with st.expander("📋 Tabela Completa de Trade-off (Velocidade × L × ΔT)", expanded=False):
        df_trade = pd.DataFrame(study["tradeoff_rows"])
        st.dataframe(df_trade, width="stretch", hide_index=True)
        st.download_button(
            "📥 Exportar Trade-off (CSV)",
            data=df_trade.to_csv(index=False).encode("utf-8"),
            file_name="mobility_tradeoff.csv",
            mime="text/csv",
        )

    # 4. SDMA Multi-usuário Móvel (MRC vs. Zero-Forcing)
    st.divider()
    st.markdown("### 📶 SDMA Multi-usuário Móvel (MRC vs. Zero-Forcing)")
    if not mob.enable_sdma:
        st.info("Habilite **SDMA Multi-usuário Móvel** na seção de Mobilidade da barra lateral.")
        return

    sdma_track = st.radio(
        "Rastreamento usado no SDMA Móvel",
        options=["hold", "predictive"] if mob.compare_predictive else ["hold"],
        format_func=lambda m: "Zero-Order Hold (ΔT)" if m == "hold" else "Rastreamento Preditivo",
        horizontal=True,
        key="tab4_sdma_track",
    )
    with st.spinner("Calculando matriz de acoplamento espacial G_eff(t) para SDMA..."):
        sdma = _cached_mobile_sdma(
            mob,
            config.channel,
            config.power_control.p_max_w,
            sdma_track,
        )

    s1, s2, s3 = st.columns(3)
    s1.metric("Sum-SE Média (MRC)", f"{sdma.mean_sum_se_mrc:.2f} bps/Hz", f"K = {sdma.K} UEs")
    s2.metric("Sum-SE Média (Zero-Forcing)", f"{sdma.mean_sum_se_zf:.2f} bps/Hz", f"L = {sdma.L} antenas")
    gain_pct = (
        ((sdma.mean_sum_se_zf / sdma.mean_sum_se_mrc) - 1.0) * 100.0
        if sdma.mean_sum_se_mrc > 0
        else 0.0
    )
    s3.metric("Ganho ZF vs. MRC", f"{gain_pct:+.1f}%", f"{sdma.mean_sum_se_zf - sdma.mean_sum_se_mrc:+.2f} bps/Hz")

    fig_traj, fig_sdma = build_sdma_figures(sdma, combiner=mob.sdma_combiner)
    st.plotly_chart(fig_traj, width="stretch")
    st.plotly_chart(fig_sdma, width="stretch")

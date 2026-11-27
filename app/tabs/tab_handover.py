# ---------------------------------------------------------------------------
# app/tabs/tab_handover.py — Aba 5: Handover (Hard vs. Soft) e Outage
# ---------------------------------------------------------------------------

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.charts import build_handover_figures
from simulator.handover import run_handover_simulation
from simulator.schemas import ChannelConfig, HandoverConfig, SimulationConfig


@st.cache_data(show_spinner=False)
def _cached_handover(ho: HandoverConfig, channel_cfg: ChannelConfig):
    return run_handover_simulation(ho, channel_cfg=channel_cfg)


def render_tab_handover(config: SimulationConfig) -> None:
    """Aba 5: Travessia entre APs comparando Hard Handover vs. Soft Handover."""
    ho = config.handover
    st.subheader("🔄 Handover: Hard vs. Soft e Detecção de Outage (Parte 2 — Ex. 12 & 13)")
    st.caption(
        f"Rodovia: {ho.dist_total_m:g} m | {ho.num_aps} APs | v = {ho.velocity_ms:g} m/s | "
        f"T_s = {ho.t_sample_s:g} s | P_min = {ho.p_min_w:.0e} W | Histerese = {ho.hysteresis_db:g} dB | "
        f"TTT = {ho.ttt_ms:g} ms | Soft = {ho.soft_combining} ({ho.mc_runs} rodadas MC)"
    )

    with st.spinner("Executando simulação de Handover (Hard vs. Soft)..."):
        res = _cached_handover(ho, config.channel)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Outage Hard Handover", f"{res.hard_outage_rate_pct:.2f}%", f"Amostra: {int(res.hard_outage_mask.sum())} ev.")
    c2.metric(
        "Outage Soft Handover",
        f"{res.soft_outage_rate_pct:.2f}%",
        f"{res.soft_outage_rate_pct - res.hard_outage_rate_pct:+.2f} pp",
        delta_color="inverse",
    )
    c3.metric("Trocas Hard (Média)", f"{res.hard_handovers_mean:.1f}", f"Amostra: {res.hard_handovers_sample}")
    c4.metric("Trocas Soft (Média)", f"{res.soft_handovers_mean:.1f}", f"Amostra: {res.soft_handovers_sample}")
    c5.metric("Ganho Médio SNR (Soft)", f"{res.mean_snr_gain_soft_vs_hard_db:+.2f} dB")

    fig_dec, fig_sig = build_handover_figures(res)
    st.plotly_chart(fig_dec, width="stretch")
    st.plotly_chart(fig_sig, width="stretch")

    df_summary = pd.DataFrame(
        [
            {
                "Algoritmo": "Hard Handover",
                "Taxa de Outage [%]": round(res.hard_outage_rate_pct, 3),
                "Handovers Médios (Ping-Pong)": round(res.hard_handovers_mean, 2),
                "Outages (Trajetória Exemplo)": int(res.hard_outage_mask.sum()),
                "SNR Média Exemplo [dB]": round(float(res.hard_snr_db.mean()), 2),
            },
            {
                "Algoritmo": f"Soft Handover ({ho.soft_combining})",
                "Taxa de Outage [%]": round(res.soft_outage_rate_pct, 3),
                "Handovers Médios (Ping-Pong)": round(res.soft_handovers_mean, 2),
                "Outages (Trajetória Exemplo)": int(res.soft_outage_mask.sum()),
                "SNR Média Exemplo [dB]": round(float(res.soft_snr_db.mean()), 2),
            },
        ]
    )
    st.dataframe(df_summary, width="stretch", hide_index=True)
    st.download_button(
        "📥 Exportar Resumo de Handover (CSV)",
        data=df_summary.to_csv(index=False).encode("utf-8"),
        file_name="handover_summary.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# app/tabs/tab_convergence.py — Aba 6: Convergência Passo a Passo do Snapshot
# ---------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from app.components.charts import build_convergence_figures
from simulator.runner import run_single_snapshot
from simulator.schemas import MonteCarloResult, SimulationConfig, SnapshotResult


def render_tab_convergence(
    config: SimulationConfig,
    snapshot: SnapshotResult,
    mc_result: MonteCarloResult | None,
) -> None:
    """
    Aba 6: Evolução por iteração de P_k (dBm/mW), SINR real (dB) e meta γ_tar_k (dB),
    destacando o instante de bloqueio (Outage) e o alívio de interferência nos sobreviventes.
    """
    st.subheader("🔬 Convergência Passo a Passo do Controle de Potência (Dual-Loop / DPC)")

    # Reúne históricos disponíveis: snapshot atual + snapshots salvos no Monte Carlo
    candidates: list[tuple[str, dict[str, Any]]] = []
    if snapshot.convergence_history is not None:
        candidates.append(
            (
                f"📍 Snapshot Atual ({snapshot.power_mode}, K={len(snapshot.ues)}, seed={snapshot.seed})",
                snapshot.convergence_history,
            )
        )
    if mc_result is not None:
        for k_val, hist_list in mc_result.convergence_histories.items():
            for h in hist_list:
                snap_n = int(h.get("snapshot", 0)) + 1
                candidates.append(
                    (
                        f"🎲 Monte Carlo — K={k_val}, Realização #{snap_n} (seed={h.get('seed', '?')})",
                        h,
                    )
                )

    if not candidates:
        st.warning(
            f"O algoritmo selecionado no Snapshot atual (`{snapshot.power_mode}`) é não-iterativo "
            "e não gera histórico de convergência. Selecione **Dual-Loop PC**, **DPC Fixo**, "
            "**MaxSum** ou **MaxProd**, ou clique abaixo para inspecionar o Dual-Loop neste mesmo cenário:"
        )
        if st.button("⚡ Simular Dual-Loop neste mesmo Snapshot", type="primary"):
            dl_snap = run_single_snapshot(
                config,
                antenna_model=snapshot.antenna_model,
                power_mode="dual_loop",
                seed=snapshot.seed,
                K_override=len(snapshot.ues),
            )
            st.session_state["tab6_fallback_hist"] = dl_snap.convergence_history

        fb = st.session_state.get("tab6_fallback_hist")
        if fb is None:
            return
        candidates.append(("⚡ Dual-Loop (Snapshot Atual)", fb))

    c1, c2, c3 = st.columns([2.6, 1.3, 1.3])
    sel_idx = c1.selectbox(
        "Realização a inspecionar",
        options=list(range(len(candidates))),
        format_func=lambda i: candidates[i][0],
        key="tab6_hist_sel",
    )
    power_unit = c2.selectbox(
        "Escala de Potência",
        options=["dbm", "mw"],
        format_func=lambda u: "dBm (Log — passos ±μ)" if u == "dbm" else "mW (Linear)",
        key="tab6_pwr_unit",
    )
    ue_filter = c3.selectbox(
        "Filtro de UEs",
        options=["all", "active", "off"],
        format_func=lambda f: {
            "all": "Todos os UEs",
            "active": "Apenas Ativos",
            "off": "Apenas Bloqueados (Outage)",
        }[f],
        key="tab6_ue_filter",
    )

    hist = candidates[sel_idx][1]
    pw = np.asarray(hist["power_w"], dtype=float)
    T, K = pw.shape
    p_init_w = float(np.sum(pw[0])) if T > 0 else 0.0
    p_final_w = float(np.sum(pw[-1])) if T > 0 else 0.0
    savings_pct = ((1.0 - p_final_w / p_init_w) * 100.0) if p_init_w > 0 else 0.0
    n_active = int(np.sum(pw[-1] > 0.0)) if T > 0 else 0
    n_off = K - n_active
    n_outer = len(hist.get("outer_marks", []))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Passos Loop Interno", f"{T}", f"Loop Externo: {n_outer} rodadas")
    m2.metric("UEs Ativos / Bloqueados", f"{n_active} / {K}", f"Bloqueados: {n_off}")
    m3.metric("Potência Inicial → Final", f"{p_final_w:.3f} W", f"Inicial: {p_init_w:.3f} W")
    m4.metric("Economia de Potência", f"{savings_pct:.1f}%")

    fig_p, fig_s, fig_t = build_convergence_figures(hist, power_unit=power_unit, ue_filter=ue_filter)
    st.plotly_chart(fig_p, width="stretch", key="tab5_conv_p")
    st.plotly_chart(fig_s, width="stretch", key="tab5_conv_s")
    if hist.get("algorithm") == "dual_loop":
        st.plotly_chart(fig_t, width="stretch", key="tab5_conv_t")

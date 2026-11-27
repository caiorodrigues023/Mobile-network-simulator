# ---------------------------------------------------------------------------
# app/main_app.py — Entrypoint Web Streamlit para o Mobile Network Simulator
# ---------------------------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

# Garante resolução dos pacotes `app` e `simulator` ao rodar `streamlit run app/main_app.py`
_ROOT_DIR = Path(__file__).resolve().parents[1]
_SRC_DIR = _ROOT_DIR / "src"
for _p in (str(_ROOT_DIR), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st  # noqa: E402

from app.components.sidebar import render_sidebar  # noqa: E402
from app.tabs.tab_cdfs import render_tab_cdfs  # noqa: E402
from app.tabs.tab_convergence import render_tab_convergence  # noqa: E402
from app.tabs.tab_handover import render_tab_handover  # noqa: E402
from app.tabs.tab_kpis import render_tab_kpis  # noqa: E402
from app.tabs.tab_mobility import render_tab_mobility  # noqa: E402
from app.tabs.tab_snapshot import render_tab_snapshot  # noqa: E402
from simulator.runner import run_monte_carlo  # noqa: E402


def main() -> None:
    st.set_page_config(
        page_title="Mobile Network Simulator",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    config, run_mc_clicked, refresh_snap_clicked = render_sidebar()

    if refresh_snap_clicked:
        st.session_state["snap_seed_active"] = int(config.seed)
        st.session_state.pop("snap_k_override", None)

    if run_mc_clicked:
        progress_bar = st.progress(0.0, text="Iniciando bateria Monte Carlo...")

        def _on_progress(frac: float, msg: str) -> None:
            progress_bar.progress(min(max(frac, 0.0), 1.0), text=msg)

        mc_res = run_monte_carlo(config, progress_callback=_on_progress)
        st.session_state["mc_result"] = mc_res
        st.session_state.pop("tab6_fallback_hist", None)
        progress_bar.empty()
        st.toast(
            f"Monte Carlo concluído em {mc_res.elapsed_s:.2f} s "
            f"({config.monte_carlo_runs} rodadas × K ∈ {list(config.k_scenarios)})!",
            icon="✅",
        )

    mc_result = st.session_state.get("mc_result")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "🗺️ Snapshot & Monte Carlo",
            "📈 CDFs Comparativas",
            "🏆 KPIs & Tabela",
            "🚗 Mobilidade, Tracking & SDMA",
            "🔄 Handover (Hard vs. Soft)",
            "🔬 Convergência do Snapshot",
        ]
    )

    with tab1:
        snapshot = render_tab_snapshot(config, mc_result)
    with tab2:
        render_tab_cdfs(mc_result)
    with tab3:
        render_tab_kpis(mc_result)
    with tab4:
        render_tab_mobility(config)
    with tab5:
        render_tab_handover(config)
    with tab6:
        render_tab_convergence(config, snapshot, mc_result)


if __name__ == "__main__":
    main()

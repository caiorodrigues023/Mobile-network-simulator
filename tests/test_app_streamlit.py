# ---------------------------------------------------------------------------
# tests/test_app_streamlit.py — Testes E2E da Interface Web Streamlit + Plotly
# ---------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.components.charts import build_snapshot_map_2d, build_snapshot_map_3d
from simulator.runner import run_single_snapshot
from simulator.schemas import SimulationConfig

APP_FILE = str(Path(__file__).resolve().parents[1] / "app" / "main_app.py")


def test_plotly_snapshot_hover_rules_2d_and_3d() -> None:
    """
    Linhas de conexão, grade, lóbulos e heatmap devem usar hoverinfo='skip',
    enquanto os marcadores de UEs e APs devem carregar hovertemplate e customdata ricos.
    """
    cfg = SimulationConfig(M=9, K=6, seed=42)
    snap = run_single_snapshot(
        cfg,
        antenna_model="mrc",
        power_mode="dual_loop",
        include_heatmap=True,
        heatmap_metric="sinr",
        heatmap_res=16,
    )

    for fig in (
        build_snapshot_map_2d(
            snap, show_beams=True, show_links=True, show_heatmap=True, show_interference=True
        ),
        build_snapshot_map_3d(snap, show_beams=True, show_links=True, show_heatmap=True),
    ):
        ue_traces = []
        ap_traces = []
        for tr in fig.data:
            name = str(getattr(tr, "name", "") or "")
            if name in ("UEs Ativos", "UEs em Outage"):
                ue_traces.append(tr)
            elif name == "Access Points (APs)":
                ap_traces.append(tr)
            else:
                assert tr.hoverinfo == "skip", (
                    f"Trace auxiliar '{name}' ({tr.type}) deve usar hoverinfo='skip', "
                    f"mas usou {tr.hoverinfo!r}"
                )

        assert ue_traces, "Mapa deve conter ao menos um trace de UEs."
        assert len(ap_traces) == 1, "Mapa deve conter exatamente um trace de APs."

        ap_tr = ap_traces[0]
        assert "UEs Conectados" in ap_tr.hovertemplate
        assert "Lista de UEs" in ap_tr.hovertemplate
        assert "Sum-Capacity" in ap_tr.hovertemplate
        assert "Outage" in ap_tr.hovertemplate
        assert len(ap_tr.customdata) == len(snap.aps)

        # Verifica correspondência exata entre customdata dos APs e snapshot.aps
        for ap, row in zip(snap.aps, ap_tr.customdata):
            assert row[0] == ap.ap_id
            assert str(ap.num_ues) in str(row[3])
            for uid in ap.connected_ue_ids:
                assert f"UE {uid}" in str(row[4])

        total_ue_points = sum(len(tr.customdata) for tr in ue_traces)
        assert total_ue_points == len(snap.ues)
        for tr in ue_traces:
            assert "AP Conectado" in tr.hovertemplate
            assert "SINR" in tr.hovertemplate
            assert "Capacidade" in tr.hovertemplate
            assert "Status" in tr.hovertemplate


def test_streamlit_app_initial_load_and_monte_carlo_flow() -> None:
    """
    Carrega app/main_app.py via AppTest:
      1. Verifica abertura imediata no Snapshot sem exceções.
      2. Executa Monte Carlo rápido (20 rodadas, K=1,4) e valida renderização na Tela 1.
      3. Testa alternância para modo 3D e aplicação de preset.
    """
    at = AppTest.from_file(APP_FILE, default_timeout=60.0)
    at.run()

    assert not at.exception, f"Exceções no carregamento inicial: {[e.value for e in at.exception]}"
    assert not at.error, f"Erros no carregamento inicial: {[e.value for e in at.error]}"
    assert "current_snapshot" in at.session_state
    snap = at.session_state["current_snapshot"]
    assert len(snap.aps) == 9
    assert len(snap.ues) == 4
    assert at.session_state.get("mc_result") is None

    # Configura Monte Carlo rápido (20 rodadas, K=1,4) e dispara execução
    at.sidebar.number_input(key="sb_mc_runs").set_value(20)
    at.sidebar.text_input(key="sb_k_scenarios").set_value("1, 4")
    at.sidebar.selectbox(key="sb_comp_mode").select("both")
    run_btn = next(b for b in at.sidebar.button if "Executar Monte Carlo" in b.label)
    run_btn.click().run()

    assert not at.exception, f"Exceções após Monte Carlo: {[e.value for e in at.exception]}"
    mc_res = at.session_state.get("mc_result")
    assert mc_res is not None
    assert mc_res.k_scenarios == (1, 4)
    assert mc_res.power_metrics
    assert mc_res.antenna_metrics
    assert mc_res.round_inspector is not None

    # Alterna visualização do Snapshot para 3D (Bônus) e verifica ausência de erros
    at.radio(key="tab1_view_mode").set_value("3D (Bônus)").run()
    assert not at.exception, f"Exceções no modo 3D: {[e.value for e in at.exception]}"

    # Aplica um preset dos roteiros pela sidebar e verifica sincronização dos dropdowns
    at.sidebar.selectbox(key="sb_preset_sel").select(
        "Parte 2 Ex. 6–7 — PAPOA + Carrier Aggregation (M=16, N=2)"
    ).run()
    apply_btn = next(b for b in at.sidebar.button if "Aplicar Preset" in b.label)
    apply_btn.click().run()
    assert not at.exception, f"Exceções ao aplicar preset: {[e.value for e in at.exception]}"
    assert at.session_state["sb_M"] == 16
    assert at.session_state["sb_alloc"] == "papoa_ca"
    assert at.session_state["tab1_ant_select"] == "omni"
    assert at.session_state["tab1_pwr_select"] == "full_power"

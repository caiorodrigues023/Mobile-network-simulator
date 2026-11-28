# ---------------------------------------------------------------------------
# tests/test_physical_sanity.py — Testes automatizados de sanidade física
# ---------------------------------------------------------------------------

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simulator import (  # noqa: E402
    AntennaConfig,
    ChannelConfig,
    DPC_Dual_Loop_Vectorized,
    HandoverConfig,
    MobilityConfig,
    PowerControlConfig,
    SimulationConfig,
    allocate_channels_vec,
    calc_ula_gain,
    calculate_sinr_vector,
    cenario_aleatorio,
    channel_inversion_power_control,
    evaluate_physical_kpis,
    gain_matrix,
    gerar_rayleigh,
    mrc_gain,
    optimal_sinr_power_control,
    plotar_mapa,
    resolve_comparison_flags,
    run_handover_simulation,
    run_mobile_sdma,
    run_mobility_scenario,
    run_monte_carlo,
    run_network_handover,
    run_network_mobility,
    run_power_control,
    run_single_snapshot,
    run_snapshot_comparison,
    smb_gains,
    ula_2d_gain,
)

# ---------------------------------------------------------------------------
# 1. Contratos, Validação e Serialização JSON
# ---------------------------------------------------------------------------


def test_schema_validation_and_json_roundtrip():
    with pytest.raises(ValueError):
        SimulationConfig(M=5)  # não é quadrado perfeito
    with pytest.raises(ValueError):
        SimulationConfig(N=0)
    with pytest.raises(ValueError):
        PowerControlConfig(p_min_dbm=35.0, p_max_dbm=30.0)
    with pytest.raises(ValueError):
        PowerControlConfig(gamma_min_db=20.0, gamma_max_db=10.0)

    cfg = SimulationConfig(
        M=9,
        K=6,
        k_scenarios=(1, 4, 8),
        N=2,
        allocation_policy="papoa_ca",
        comparison_mode="both",
        channel=ChannelConfig(use_shadowing=True, use_rayleigh=True),
        antenna=AntennaConfig(models=("omni", "smb", "mrc", "ula"), L=16, J=8),
        power_control=PowerControlConfig(
            algorithms=("full_power", "cipc", "optimal", "dpc_fixed", "dual_loop"),
            apply_physical_limits=True,
        ),
    )
    restored = SimulationConfig.from_json(cfg.to_json())
    assert restored == cfg


# ---------------------------------------------------------------------------
# 2. Determinismo de Seed (Methodology Remark)
# ---------------------------------------------------------------------------


def test_seed_determinism_snapshot_and_monte_carlo():
    cfg = SimulationConfig(
        M=9,
        K=6,
        k_scenarios=(4,),
        N=2,
        allocation_policy="papoa",
        monte_carlo_runs=8,
        seed=1234,
        comparison_mode="both",
        channel=ChannelConfig(use_shadowing=True, use_rayleigh=True),
        antenna=AntennaConfig(models=("omni", "smb", "mrc"), L=8, J=8),
        power_control=PowerControlConfig(algorithms=("full_power", "dpc_fixed", "dual_loop")),
    )
    s1 = run_single_snapshot(cfg, include_heatmap=False)
    s2 = run_single_snapshot(cfg, include_heatmap=False)
    np.testing.assert_allclose(s1.ue_xy, s2.ue_xy)
    np.testing.assert_array_equal(s1.serving_ap, s2.serving_ap)
    np.testing.assert_array_equal(s1.channels, s2.channels)
    np.testing.assert_allclose(s1.G_eff, s2.G_eff)
    assert s1.sum_capacity_mbps == pytest.approx(s2.sum_capacity_mbps)

    mc1 = run_monte_carlo(cfg)
    mc2 = run_monte_carlo(cfg)
    for m in ("omni", "smb", "mrc"):
        np.testing.assert_allclose(
            mc1.antenna_metrics[m][4].sinr_lin, mc2.antenna_metrics[m][4].sinr_lin
        )
    for a in ("full_power", "dpc_fixed", "dual_loop"):
        np.testing.assert_allclose(
            mc1.power_metrics[a][4].sinr_lin, mc2.power_metrics[a][4].sinr_lin
        )


# ---------------------------------------------------------------------------
# 3. Antenas e Arranjos (ULA, UCA MRC, SMB)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L", [4, 8, 16, 32])
def test_ula_and_uca_perfect_alignment_equals_L(L: int):
    angles = np.linspace(-np.pi / 3, np.pi / 3, 25)
    g_ula = calc_ula_gain(angles, angles, L)
    np.testing.assert_allclose(g_ula, float(L), rtol=1e-10)

    phi = np.linspace(0, 2 * np.pi, 25, endpoint=False)
    g_uca = mrc_gain(phi, phi, L)
    np.testing.assert_allclose(g_uca, float(L), rtol=1e-10)

    g_ula_2d = ula_2d_gain(phi, phi, L)
    np.testing.assert_allclose(g_ula_2d, float(L), rtol=1e-10)

    # Pico dos feixes SMB nas direções nominais = L
    theta_k = np.arange(8) * (2 * np.pi / 8)
    g_smb = smb_gains(theta_k, L=L, J=8)
    np.testing.assert_allclose(np.diag(g_smb), float(L), rtol=1e-10)


# ---------------------------------------------------------------------------
# 4. Canal (Shadowing / Rayleigh opcionais) e Alocação (PAPOA / CA)
# ---------------------------------------------------------------------------


def test_channel_shadowing_and_rayleigh_flags():
    ue_xy = np.array([[100.0, 0.0], [200.0, 0.0]])
    ap_xy = np.array([[0.0, 0.0]])
    G = np.ones((2, 1, 1))
    ch_det = ChannelConfig(use_shadowing=False, use_rayleigh=False, k_const=1e-4, pathloss_exp=4.0)
    g = gain_matrix(ue_xy, ap_xy, G, channel_cfg=ch_det)
    np.testing.assert_allclose(g[:, 0, 0], [1e-4 / 100.0**4, 1e-4 / 200.0**4], rtol=1e-12)

    rng = np.random.default_rng(7)
    r = gerar_rayleigh(500, 100, sigma_r=1.0 / np.sqrt(2.0), rng=rng)
    assert np.mean(r**2) == pytest.approx(1.0, abs=0.05)


def test_papoa_orthogonality_and_carrier_aggregation():
    rng = np.random.default_rng(99)
    # 3 UEs no AP 0, 1 UE solitário no AP 1, N = 3 canais
    serving_ap = np.array([0, 0, 0, 1])
    ch_papoa, agg_papoa = allocate_channels_vec(serving_ap, M=2, N=3, policy="papoa", rng=rng)
    assert len(set(ch_papoa[:3].tolist())) == 3  # sem reuso intra-AP
    assert not np.any(agg_papoa)

    ch_ca, agg_ca = allocate_channels_vec(serving_ap, M=2, N=3, policy="papoa_ca", rng=rng)
    assert len(set(ch_ca[:3].tolist())) == 3
    assert bool(agg_ca[3]) is True
    assert not np.any(agg_ca[:3])


# ---------------------------------------------------------------------------
# 5. Controle de Potência (Outage com interferência zero, CIPC, Ótimo, MaxSum/Prod)
# ---------------------------------------------------------------------------


def test_dual_loop_blocked_ues_emit_zero_interference():
    # Cenário propositalmente hostil para forçar bloqueio de pelo menos 1 UE:
    # UE 0 tem canal excelente; UE 1 e UE 2 estão fortemente acoplados com canal fraco
    G_eff = np.array(
        [
            [1e-8, 1e-12, 1e-12],
            [1e-12, 1e-14, 5e-11],
            [1e-12, 5e-11, 1e-14],
        ],
        dtype=float,
    )
    P_n = 1e-12
    p_w, sinr_lin, gamma_tar, hist = DPC_Dual_Loop_Vectorized(
        G_eff,
        P_n,
        gamma_tar_db=np.full(3, 10.0),
        ue_active=np.ones(3, dtype=bool),
        t_outer=35,
        t_inner=25,
        record_history=True,
    )
    assert hist is not None
    # Pelo menos um dos UEs fracos deve ter sido desligado (0 W)
    blocked = p_w <= 0.0
    assert np.any(blocked), "Esperado bloqueio de UE inviável no Dual-Loop"
    assert np.all(p_w[blocked] == 0.0)
    assert np.all(sinr_lin[blocked] == 0.0)

    # Verifica que UEs bloqueados contribuem com ZERO interferência nos sobreviventes
    survivors = ~blocked
    if np.any(survivors):
        G_sub = G_eff[np.ix_(survivors, survivors)]
        sinr_without_blocked = calculate_sinr_vector(G_sub, p_w[survivors], P_n)
        np.testing.assert_allclose(sinr_lin[survivors], sinr_without_blocked, rtol=1e-12)


def test_optimal_pc_exact_solution_and_infeasible_fallback():
    # Exercício 10 da Parte 2: g1=0.1, g2=0.05, pn=0.01
    G_eff = np.array([[0.1, 0.05], [0.1, 0.05]], dtype=float)
    pn = 0.01

    # Caso viável: gamma_t = 0.5 (-3.0103 dB) -> solução exata atinge SINR = 0.5 para ambos
    gamma_db_ok = 10.0 * np.log10(0.5)
    p_opt, feasible = optimal_sinr_power_control(G_eff, pn, gamma_tar_db=gamma_db_ok, p_max_w=10.0)
    assert feasible is True
    sinr = calculate_sinr_vector(G_eff, p_opt, pn)
    np.testing.assert_allclose(sinr, [0.5, 0.5], rtol=1e-9)

    # Caso inviável: gamma_t = 1.0 (0 dB, rho(A)=1) -> fallback para CIPC sem negativos/NaN
    p_fallback, feasible2 = optimal_sinr_power_control(G_eff, pn, gamma_tar_db=0.0, p_max_w=1.0)
    p_cipc = channel_inversion_power_control(G_eff, pn, gamma_tar_db=0.0, p_max_w=1.0)
    assert feasible2 is False
    np.testing.assert_allclose(p_fallback, p_cipc)


def test_maxsum_maxprod_and_physical_limits_toggle():
    G_eff = np.array([[1e-9, 1e-11], [2e-11, 8e-10]], dtype=float)
    pc_cfg = PowerControlConfig()
    for algo in ("full_power", "cipc", "optimal", "dpc_fixed", "dual_loop", "max_sum", "max_prod"):
        p_w, sinr, _, _ = run_power_control(algo, G_eff, 1e-12, pc_cfg)
        assert np.all(np.isfinite(p_w))
        assert np.all(p_w >= 0.0) and np.all(p_w <= pc_cfg.p_max_w + 1e-12)
        assert np.all(np.isfinite(sinr))

    # Limite físico QAM (+20 dB) e corte Outage (-10 dB)
    sinr_test = np.array([0.05, 500.0])  # -13 dB (outage) e +27 dB (acima de +20 dB)
    p_test = np.array([1.0, 1.0])
    se_lim, _, _, _, out = evaluate_physical_kpis(
        sinr_test, p_test, 100e6, gamma_min_db=-10.0, gamma_max_db=20.0, apply_physical_limits=True
    )
    assert bool(out[0]) is True and se_lim[0] == 0.0
    assert se_lim[1] == pytest.approx(np.log2(1.0 + 100.0), rel=1e-12)

    se_unlim, _, _, _, _ = evaluate_physical_kpis(
        sinr_test, p_test, 100e6, gamma_min_db=-10.0, gamma_max_db=20.0, apply_physical_limits=False
    )
    assert se_unlim[0] > 0.0
    assert se_unlim[1] > se_lim[1]


# ---------------------------------------------------------------------------
# 6. Mobilidade, Rastreamento Preditivo e SDMA Móvel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sigma_deg", [0.0, 1.0])
def test_predictive_beam_tracking_reduces_misalignment_loss(sigma_deg: float):
    res_hold = run_mobility_scenario(
        L=32, v_kmh=120.0, delta_t_s=0.100, D_m=50.0, sigma_theta_deg=sigma_deg,
        tracking_mode="hold", seed=42
    )
    res_pred = run_mobility_scenario(
        L=32, v_kmh=120.0, delta_t_s=0.100, D_m=50.0, sigma_theta_deg=sigma_deg,
        tracking_mode="predictive", seed=42
    )
    assert res_pred.mean_loss_db < res_hold.mean_loss_db
    assert res_pred.mean_se_net > res_hold.mean_se_net


def test_mobile_sdma_zero_forcing_beats_mrc():
    mob_cfg = MobilityConfig(
        selected_L=16,
        selected_velocity_kmh=50.0,
        selected_delta_t_s=0.050,
        selected_distance_m=50.0,
        sigma_theta_deg=0.5,
        sdma_k_ues=4,
        t_sim_s=1.0,
        seed=42,
    )
    sdma = run_mobile_sdma(mob_cfg, tracking_mode="predictive")
    assert sdma.sinr_mrc_db.shape == (1000, 4)
    assert sdma.sinr_zf_db.shape == (1000, 4)
    assert sdma.mean_sum_se_zf > sdma.mean_sum_se_mrc


# ---------------------------------------------------------------------------
# 7. Handover (Hard vs. Soft)
# ---------------------------------------------------------------------------


def test_handover_soft_outperforms_hard():
    ho_cfg = HandoverConfig(mc_runs=150, seed=42)
    res = run_handover_simulation(ho_cfg)
    assert res.soft_outage_rate_pct <= res.hard_outage_rate_pct
    assert res.mean_snr_gain_soft_vs_hard_db >= 0.0

    # Macro-diversidade >= seleção em potência recebida
    ho_macro = HandoverConfig(
        mc_runs=20, seed=42, active_set_margin_db=6.0, soft_combining="macro_diversity"
    )
    ho_sel = HandoverConfig(
        mc_runs=20, seed=42, active_set_margin_db=0.0, soft_combining="selection"
    )
    r_macro = run_handover_simulation(ho_macro)
    r_sel = run_handover_simulation(ho_sel)
    assert np.all(r_macro.soft_rx_power_w >= r_sel.soft_rx_power_w - 1e-25)


# ---------------------------------------------------------------------------
# 8. Consistência AP <-> UE no SnapshotResult, CLI e Desacoplamento
# ---------------------------------------------------------------------------


def test_snapshot_ap_ue_consistency_and_cli_map():
    cfg = SimulationConfig(
        M=9,
        K=12,
        N=2,
        allocation_policy="papoa_ca",
        snapshot_antenna_model="smb",
        snapshot_power_mode="dual_loop",
        seed=7,
    )
    snap = run_single_snapshot(cfg, include_heatmap=True, heatmap_res=20)
    assert len(snap.ues) == 12
    assert len(snap.aps) == 9
    assert snap.heatmap is not None

    # Verifica bijeção exata AP <-> UE
    seen_ues: list[int] = []
    for ap in snap.aps:
        assert ap.num_ues == len(ap.connected_ue_ids)
        seen_ues.extend(ap.connected_ue_ids)
        for uid in ap.connected_ue_ids:
            assert snap.ues[uid].serving_ap == ap.ap_id
        expected_out = sum(1 for uid in ap.connected_ue_ids if snap.ues[uid].is_outage)
        assert ap.num_outage == expected_out

    assert sorted(seen_ues) == list(range(12))
    assert sum(ap.sum_capacity_mbps for ap in snap.aps) == pytest.approx(
        snap.sum_capacity_mbps, rel=1e-9
    )

    # Verifica correção da plotar_mapa legada (Bug 1)
    aps, ues, shad = cenario_aleatorio(M=4, K=4, seed=42)
    fig = plotar_mapa(aps, ues, shadowing=shad, models=("omni", "sector", "smb"), show=False)
    assert fig is not None
    plt.close(fig)


def test_core_has_no_streamlit_imports():
    sim_dir = Path(__file__).resolve().parents[1] / "src" / "simulator"
    for py_file in sim_dir.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("streamlit"), f"Streamlit importado em {py_file}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not node.module.startswith("streamlit"), f"Streamlit importado em {py_file}"


# ---------------------------------------------------------------------------
# 10. Comparação Automática + Mobilidade e Handover na Rede Completa (M × K)
# ---------------------------------------------------------------------------


def test_auto_comparison_mode_and_snapshot_comparison():
    cfg_ant_only = SimulationConfig(
        M=4,
        K=4,
        k_scenarios=(4,),
        monte_carlo_runs=5,
        comparison_mode="auto",
        antenna=AntennaConfig(models=("omni", "smb")),
        power_control=PowerControlConfig(algorithms=("full_power",)),
    )
    assert resolve_comparison_flags(cfg_ant_only) == (True, False, "antennas")
    mc_a = run_monte_carlo(cfg_ant_only)
    assert set(mc_a.antenna_metrics.keys()) == {"omni", "smb"}
    assert not mc_a.power_metrics

    cfg_pwr_only = SimulationConfig(
        M=4,
        K=4,
        k_scenarios=(4,),
        monte_carlo_runs=5,
        comparison_mode="auto",
        antenna=AntennaConfig(models=("mrc",)),
        power_control=PowerControlConfig(algorithms=("full_power", "dual_loop")),
    )
    assert resolve_comparison_flags(cfg_pwr_only) == (False, True, "power")
    mc_p = run_monte_carlo(cfg_pwr_only)
    assert not mc_p.antenna_metrics
    assert set(mc_p.power_metrics.keys()) == {"full_power", "dual_loop"}

    cfg_both = SimulationConfig(
        M=4,
        K=4,
        k_scenarios=(4,),
        monte_carlo_runs=5,
        comparison_mode="auto",
        antenna=AntennaConfig(models=("omni", "mrc")),
        power_control=PowerControlConfig(algorithms=("full_power", "dual_loop")),
    )
    assert resolve_comparison_flags(cfg_both) == (True, True, "both")
    comp = run_snapshot_comparison(cfg_both)
    assert set(comp["antennas"].keys()) == {"omni", "mrc"}
    assert set(comp["power"].keys()) == {"full_power", "dual_loop"}


def test_network_mobility_full_grid_and_predictive_tracking():
    cfg = SimulationConfig(
        M=9,
        K=6,
        seed=42,
        ue_mode="mobile",
        mobility=MobilityConfig(
            selected_velocity_kmh=120.0,
            selected_delta_t_s=0.100,
            selected_L=32,
            dt_s=0.005,
            t_sim_s=3.0,
            sigma_theta_deg=0.0,
        ),
    )
    snap = run_single_snapshot(cfg)
    res = run_network_mobility(cfg, seed=snap.seed, K_override=len(snap.ues))

    assert res.M == 9 and res.K == 6
    assert res.ue_x_m.shape[1] == 6
    assert res.loss_hold_db.shape == res.loss_pred_db.shape
    # Posições iniciais t=0 devem coincidir exatamente com o Snapshot
    np.testing.assert_allclose(res.ue_x_m[0], snap.ue_xy[:, 0], atol=1e-9)
    np.testing.assert_allclose(res.ue_y_m[0], snap.ue_xy[:, 1], atol=1e-9)
    # Sem ruído de AoA, o rastreamento preditivo deve reduzir a perda por desalinhamento
    assert res.mean_loss_pred_db < res.mean_loss_hold_db
    # ZF deve manter Sum-SE >= MRC na rede co-canal
    assert res.mean_sum_se_zf >= res.mean_sum_se_mrc * 0.95


def test_network_handover_full_grid_soft_outperforms_hard():
    cfg = SimulationConfig(
        M=9,
        K=6,
        seed=42,
        ue_mode="mobile",
        mobility=MobilityConfig(selected_velocity_kmh=120.0),
        handover=HandoverConfig(
            t_ho_s=20.0,
            ho_step_s=0.2,
            p_min_dbm_network=-100.0,
            hysteresis_db=0.0,
            ttt_ms=0.0,
        ),
    )
    res = run_network_handover(cfg)
    assert res.M == 9 and res.K == 6
    assert res.hard_serving_ap.shape == (len(res.t_s), 6)
    assert res.soft_outage_rate_pct <= res.hard_outage_rate_pct + 1e-9
    assert res.mean_snr_gain_soft_vs_hard_db >= -1e-9


# ---------------------------------------------------------------------------
# runner.py — Motor Híbrido: Snapshot Único, Monte Carlo Vetorizado multi-carga,
#             Heatmap 2D/3D e Wrappers de Retrocompatibilidade CLI
# ---------------------------------------------------------------------------

from __future__ import annotations

import math
import sys
import time
from typing import Any, Callable, Optional

import numpy as np
from tqdm import tqdm

from .allocation import (
    DPC_Dual_Loop,
    allocate_channels_vec,
    alocar_canais_ortogonal,
    attach_AP_UE,
    pc_dpc_fixed,
    run_power_control,
)
from .antennas import (
    antenna_gain,
    compute_antenna_gain_matrix,
    is_adaptive_model,
    resolve_aoa_error_deg,
    smb_gains,
)
from .channel import (
    adaptive_gains,
    adaptive_gains_vec,
    angle_matrix,
    distance_matrix,
    gain_matrix,
    gerar_rayleigh,
    gerar_shadowing,
)
from .config import (
    AREA_SIZE,
    B_TOTAL,
    D_0,
    GAMMA_MAX_DB,
    GAMMA_MIN_DB,
    GAMMA_TAR_INIT,
    K_0,
    K_CONST,
    MODEL_LABELS,
    MU_STEP,
    P_MAX_DBM,
    P_MIN_DBM,
    PATHLOSS_EXP,
    SIGMA_R,
    SIGMA_X,
)
from .entities import AP, UE
from .metrics import (
    SINR,
    SINR_adaptive,
    area_spectrum_efficiency,
    calculate_sinr_vector,
    channel_capacity,
    evaluate_physical_kpis,
    jain_fairness,
)
from .schemas import (
    APSnapshot,
    MonteCarloResult,
    PowerControlConfig,
    ScenarioMetrics,
    SimulationConfig,
    SnapshotResult,
    UESnapshot,
)

MC_SEED_STRIDE: int = 104729


def mc_round_seed(base_seed: int, round_idx: int) -> int:
    """Seed determinística para a iteração `round_idx` (base 0)."""
    return int((base_seed + round_idx * MC_SEED_STRIDE) % 2147483647)


# ---------------------------------------------------------------------------
# Geometria Pura em NumPy (Data-Oriented Design)
# ---------------------------------------------------------------------------


def distribute_aps_xy(M: int, area_m: float = AREA_SIZE) -> np.ndarray:
    """Retorna matriz (M, 2) com coordenadas dos APs em grade regular na área [0, area_m]²."""
    grid_dim = math.isqrt(M)
    if grid_dim * grid_dim != M:
        raise ValueError(f"M deve ser quadrado perfeito para grade regular, recebido {M}")
    spacing = area_m / grid_dim
    coords = (np.arange(grid_dim, dtype=float) + 0.5) * spacing
    gx, gy = np.meshgrid(coords, coords)
    return np.column_stack((gx.ravel(), gy.ravel()))


def sample_ue_xy(
    K: int,
    ap_xy: np.ndarray,
    area_m: float = AREA_SIZE,
    d_0: float = D_0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sorteia K posições (x, y) uniformes garantindo distância >= d_0 de todos os APs."""
    if K == 0:
        return np.zeros((0, 2), dtype=float)
    gen = rng if rng is not None else np.random.default_rng()
    ue_xy = gen.uniform(0.0, area_m, size=(K, 2))
    # Reamostra apenas pontos que caíram a < d_0 de algum AP
    for _ in range(20):
        dists = np.linalg.norm(ue_xy[:, None, :] - ap_xy[None, :, :], axis=2)
        too_close = np.any(dists < d_0, axis=1)
        if not np.any(too_close):
            break
        ue_xy[too_close] = gen.uniform(0.0, area_m, size=(int(np.sum(too_close)), 2))
    return ue_xy


def _build_full_spatial_gain_matrix(
    gains: np.ndarray,
    serving_ap: np.ndarray,
    serving_beam: np.ndarray,
    Gm: Optional[np.ndarray],
    adaptive: bool,
) -> np.ndarray:
    """
    Constrói a matriz K×K de ganho espacial total (antes da máscara de canal):
        G_raw[k, i] = ganho do sinal do UE i recebido no AP/feixe do UE k.
    """
    K = len(serving_ap)
    if K == 0:
        return np.zeros((0, 0), dtype=float)
    if adaptive and Gm is not None:
        return gains[:, serving_ap, 0].T * Gm
    return gains[:, serving_ap, serving_beam].T


def _evaluate_carrier_aggregation(
    G_raw: np.ndarray,
    p_watts: np.ndarray,
    channels: np.ndarray,
    is_aggregated: np.ndarray,
    N: int,
    bw_channel: float,
    p_noise: float,
    pc_cfg: PowerControlConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Avalia SINR, SE e Capacidade para PAPOA + Carrier Aggregation (Parte 2, Ex. 7 / Remark 5):
      - UEs agregados transmitem em TODOS os N canais com potência p_k / N por canal.
      - A capacidade de cada UE agregado é a soma das capacidades de cada canal individual.
    Retorna: (sinr_eff_lin, se_per_ue_equiv, cap_mbps, outage_mask).
    """
    K = len(p_watts)
    diag_G = np.diagonal(G_raw)
    cap_bps_total = np.zeros(K, dtype=float)
    sinr_accum = np.zeros(K, dtype=float)
    ch_count = np.zeros(K, dtype=float)

    gamma_min_lin = 10.0 ** (pc_cfg.gamma_min_db / 10.0)
    gamma_max_lin = 10.0 ** (pc_cfg.gamma_max_db / 10.0)

    for ch in range(N):
        active_ch = (channels == ch) | is_aggregated
        if not np.any(active_ch):
            continue
        p_ch = np.where(
            is_aggregated,
            p_watts / float(N),
            np.where(channels == ch, p_watts, 0.0),
        )
        rx_tot = G_raw @ p_ch
        useful = diag_G * p_ch
        interf = np.maximum(rx_tot - useful, 0.0)
        sinr_ch = np.where(active_ch & (p_ch > 0.0), useful / (interf + p_noise), 0.0)

        if pc_cfg.apply_physical_limits:
            sinr_eff = np.where(
                (p_ch <= 0.0) | (sinr_ch < gamma_min_lin),
                0.0,
                np.minimum(sinr_ch, gamma_max_lin),
            )
        else:
            sinr_eff = np.where(p_ch <= 0.0, 0.0, sinr_ch)

        cap_bps_total += np.where(active_ch, bw_channel * np.log2(1.0 + sinr_eff), 0.0)
        sinr_accum += np.where(active_ch, sinr_ch, 0.0)
        ch_count += active_ch.astype(float)

    sinr_avg = np.where(ch_count > 0.0, sinr_accum / np.maximum(ch_count, 1.0), 0.0)
    outage_mask = (p_watts <= 0.0) | (sinr_avg < gamma_min_lin)
    se_equiv = cap_bps_total / bw_channel
    cap_mbps = cap_bps_total / 1e6
    return sinr_avg, se_equiv, cap_mbps, outage_mask


def _evaluate_single_realization(
    ue_xy: np.ndarray,
    ap_xy: np.ndarray,
    phi: np.ndarray,
    shadowing: np.ndarray,
    rayleigh: Optional[np.ndarray],
    aoa_unit_noise: np.ndarray,
    config: SimulationConfig,
    antenna_model: str,
    power_mode: str,
    channel_rng: np.random.Generator,
    record_history: bool = False,
) -> dict[str, Any]:
    """
    Avalia 1 realização pura em NumPy para um par (antenna_model, power_mode)
    reutilizando geometria, shadowing e fading pré-sorteados.
    """
    K, M = phi.shape
    N = config.N
    ant_cfg = config.antenna
    pc_cfg = config.power_control
    ch_cfg = config.channel

    G_ant = compute_antenna_gain_matrix(
        phi,
        model=antenna_model,
        beamwidth=np.deg2rad(ant_cfg.beamwidth_deg),
        g_min=ant_cfg.g_min,
        L=ant_cfg.L,
        J=ant_cfg.J,
    )
    gains = gain_matrix(
        ue_xy,
        ap_xy,
        G_ant,
        shadowing=shadowing,
        rayleigh=rayleigh,
        use_shadowing=ch_cfg.use_shadowing,
        use_rayleigh=ch_cfg.use_rayleigh,
        channel_cfg=ch_cfg,
    )

    serving_ap, serving_beam = attach_AP_UE(gains)
    channels, is_aggregated = allocate_channels_vec(
        serving_ap, M=M, N=N, policy=config.allocation_policy, rng=channel_rng
    )

    adaptive = is_adaptive_model(antenna_model)
    if adaptive:
        err_deg = resolve_aoa_error_deg(antenna_model, ant_cfg.aoa_err_deg)
        array_type = "ula" if antenna_model == "ula" else "uca"
        aoa_noise_rad = np.deg2rad(err_deg) * aoa_unit_noise if err_deg > 0.0 else None
        Gm, beam_angles_rad = adaptive_gains_vec(
            phi,
            serving_ap,
            L=ant_cfg.L,
            aoa_err_deg=err_deg,
            array_type=array_type,
            aoa_noise_rad=aoa_noise_rad,
        )
    else:
        Gm = None
        if antenna_model == "sector":
            boresights = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
            beam_angles_rad = boresights[serving_beam % 3]
        elif antenna_model == "smb":
            beam_angles_rad = serving_beam.astype(float) * (2.0 * np.pi / ant_cfg.J)
        else:
            beam_angles_rad = phi[np.arange(K), serving_ap]

    G_raw = _build_full_spatial_gain_matrix(gains, serving_ap, serving_beam, Gm, adaptive)
    same_ch = (channels[:, None] == channels[None, :]) | is_aggregated[:, None] | is_aggregated[None, :]
    G_eff = np.where(same_ch, G_raw, 0.0)

    bw_ch = config.bw_channel_hz
    p_noise = config.noise_power_w

    p_watts, sinr_lin, gamma_tar_db, conv_hist = run_power_control(
        power_mode, G_eff, p_noise, pc_cfg, record_history=record_history
    )

    if np.any(is_aggregated) and N > 1:
        sinr_lin, se_per_ue, cap_mbps, outage_mask = _evaluate_carrier_aggregation(
            G_raw, p_watts, channels, is_aggregated, N, bw_ch, p_noise, pc_cfg
        )
        sum_se = float(np.sum(se_per_ue))
        sum_cap_bps = float(np.sum(cap_mbps) * 1e6)
        tot_p_w = float(np.sum(p_watts))
    else:
        se_per_ue, sum_se, sum_cap_bps, tot_p_w, outage_mask = evaluate_physical_kpis(
            sinr_lin,
            p_watts,
            bw_ch,
            gamma_min_db=pc_cfg.gamma_min_db,
            gamma_max_db=pc_cfg.gamma_max_db,
            apply_physical_limits=pc_cfg.apply_physical_limits,
        )
        cap_mbps = (bw_ch * se_per_ue) / 1e6

    diag_g = np.diagonal(G_eff)
    rx_power_w = diag_g * p_watts
    snr_lin = np.where(p_watts > 0.0, rx_power_w / p_noise, 0.0)

    return {
        "serving_ap": serving_ap,
        "serving_beam": serving_beam,
        "channels": channels,
        "is_aggregated": is_aggregated,
        "beam_angles_rad": beam_angles_rad,
        "G_eff": G_eff,
        "diag_g": diag_g,
        "p_watts": p_watts,
        "rx_power_w": rx_power_w,
        "snr_lin": snr_lin,
        "sinr_lin": sinr_lin,
        "gamma_tar_db": gamma_tar_db,
        "se_per_ue": se_per_ue,
        "cap_mbps": cap_mbps,
        "sum_se": sum_se,
        "sum_cap_bps": sum_cap_bps,
        "total_power_w": tot_p_w,
        "outage_mask": outage_mask,
        "conv_hist": conv_hist,
    }


# ---------------------------------------------------------------------------
# Mapa de Calor de Cobertura (2D / 3D — com ganho linear fisicamente exato)
# ---------------------------------------------------------------------------


def compute_coverage_heatmap(
    ap_xy: np.ndarray,
    config: SimulationConfig,
    antenna_model: str,
    res: int = 45,
    metric: str = "sinr",
) -> dict[str, Any]:
    """
    Calcula grade 2D vetorial (res × res) de cobertura de sinal sem shadowing
    (mapa determinístico médio do padrão de antena e densidade de APs).
    """
    gx = np.linspace(0.0, config.area_m, res)
    gy = np.linspace(0.0, config.area_m, res)
    X, Y = np.meshgrid(gx, gy)
    pts = np.column_stack([X.ravel(), Y.ravel()])

    dist = distance_matrix(pts, ap_xy, d_0=config.channel.d_0)
    phi = angle_matrix(pts, ap_xy)

    L = config.antenna.L
    J = config.antenna.J

    if antenna_model == "omni":
        G = np.ones((len(pts), len(ap_xy)), dtype=float)
    elif antenna_model == "sector":
        boresights = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
        delta = np.abs(phi[:, :, None] - boresights[None, None, :]) % (2.0 * np.pi)
        delta = np.minimum(delta, 2.0 * np.pi - delta)
        half_bw = np.deg2rad(config.antenna.beamwidth_deg) / 2.0
        sec_g = np.where(delta <= half_bw, 1.0, config.antenna.g_min)
        G = np.max(sec_g, axis=-1)
    elif antenna_model == "smb":
        # Ganho real do melhor feixe SMB (pico = L, sem divisão artificial)
        smb_all = smb_gains(phi.ravel(), L=L, J=J)
        G = np.max(smb_all, axis=-1).reshape(len(pts), len(ap_xy))
    else:
        # Modelos adaptativos (MRC / ULA): ganho máximo de arranjo L no apontamento ideal
        G = np.full((len(pts), len(ap_xy)), float(L), dtype=float)

    pl = config.channel.k_const / (dist**config.channel.pathloss_exp)
    p_rx = config.power_control.p_max_w * pl * G

    best_ap = np.argmax(p_rx, axis=1)
    S = np.max(p_rx, axis=1)
    # Em modo adaptativo a interferência de outros APs é atenuada em média por ~1/L (ganho unitário)
    if is_adaptive_model(antenna_model):
        p_rx_interf = config.power_control.p_max_w * pl
        interf = np.maximum(
            np.sum(p_rx_interf, axis=1) - p_rx_interf[np.arange(len(pts)), best_ap], 0.0
        )
    else:
        interf = np.maximum(np.sum(p_rx, axis=1) - S, 0.0)

    pn = config.noise_power_w
    sinr = S / (interf + pn)
    sinr_db = 10.0 * np.log10(np.maximum(sinr, 1e-12))
    rsrp_dbm = 10.0 * np.log10(np.maximum(S, 1e-18) * 1000.0)
    cap_mbps = (config.bw_channel_hz * np.log2(1.0 + sinr)) / 1e6

    if metric == "rsrp":
        values, label = rsrp_dbm.reshape(res, res), "RSRP (dBm)"
    elif metric == "cap":
        values, label = cap_mbps.reshape(res, res), "Capacidade (Mbps)"
    elif metric == "server":
        values, label = best_ap.reshape(res, res).astype(float), "AP Servidor"
    else:
        values, label = sinr_db.reshape(res, res), "SINR (dB)"

    return {
        "x": gx,
        "y": gy,
        "z": values,
        "metric": metric,
        "label": label,
        "min_val": float(np.min(values)),
        "max_val": float(np.max(values)),
    }


# ---------------------------------------------------------------------------
# API Pública 1: Snapshot Único
# ---------------------------------------------------------------------------


def run_single_snapshot(
    config: SimulationConfig,
    antenna_model: Optional[str] = None,
    power_mode: Optional[str] = None,
    seed: Optional[int] = None,
    K_override: Optional[int] = None,
    include_heatmap: bool = False,
    heatmap_metric: str = "sinr",
    heatmap_res: int = 45,
) -> SnapshotResult:
    """
    Executa 1 realização rápida e determinística da rede e retorna o `SnapshotResult`
    com todos os detalhes por UE e por AP para o mapa interativo e aba de convergência.
    """
    ant_model = antenna_model or config.snapshot_antenna_model
    pwr_mode = power_mode or config.snapshot_power_mode
    snap_seed = config.seed if seed is None else int(seed)
    K = config.K if K_override is None else int(K_override)

    rng = np.random.default_rng(snap_seed)
    ap_xy = distribute_aps_xy(config.M, config.area_m)
    ue_xy = sample_ue_xy(K, ap_xy, config.area_m, config.channel.d_0, rng=rng)

    dists = distance_matrix(ue_xy, ap_xy, d_0=config.channel.d_0)
    phi = angle_matrix(ue_xy, ap_xy)
    shadowing = gerar_shadowing(
        K,
        config.M,
        sigma_x=config.channel.sigma_x,
        use_shadowing=config.channel.use_shadowing,
        rng=rng,
    )
    rayleigh = (
        gerar_rayleigh(K, config.M, sigma_r=config.channel.sigma_r, rng=rng)
        if config.channel.use_rayleigh
        else None
    )
    aoa_unit_noise = rng.normal(0.0, 1.0, size=K)
    ch_rng = np.random.default_rng(snap_seed + 17)

    want_hist = pwr_mode in ("dual_loop", "dpc_fixed", "max_sum", "max_prod")
    ev = _evaluate_single_realization(
        ue_xy=ue_xy,
        ap_xy=ap_xy,
        phi=phi,
        shadowing=shadowing,
        rayleigh=rayleigh,
        aoa_unit_noise=aoa_unit_noise,
        config=config,
        antenna_model=ant_model,
        power_mode=pwr_mode,
        channel_rng=ch_rng,
        record_history=want_hist,
    )

    serving_ap = ev["serving_ap"]
    serving_beam = ev["serving_beam"]
    channels = ev["channels"]
    is_agg = ev["is_aggregated"]
    p_watts = ev["p_watts"]
    sinr_lin = ev["sinr_lin"]
    snr_lin = ev["snr_lin"]
    diag_g = ev["diag_g"]
    se_ue = ev["se_per_ue"]
    cap_mbps = ev["cap_mbps"]
    outage_mask = ev["outage_mask"]
    gamma_tar = ev["gamma_tar_db"]

    ue_dists = dists[np.arange(K), serving_ap]
    ue_aoa = phi[np.arange(K), serving_ap]
    gamma_min_db = config.power_control.gamma_min_db

    ues_list: list[UESnapshot] = []
    for k in range(K):
        m_id = int(serving_ap[k])
        is_blocked = bool(p_watts[k] <= 0.0)
        is_out = bool(outage_mask[k])
        p_w = float(p_watts[k])
        p_dbm = float(10.0 * np.log10(p_w * 1000.0)) if not is_blocked else -np.inf
        s_lin = float(sinr_lin[k])
        s_db = float(10.0 * np.log10(max(s_lin, 1e-12))) if not is_blocked else -np.inf
        snr_db = float(10.0 * np.log10(max(float(snr_lin[k]), 1e-12))) if not is_blocked else -np.inf
        rx_g_db = float(10.0 * np.log10(max(float(diag_g[k]), 1e-30)))

        if is_blocked:
            status = "Outage (Bloqueado)"
        elif is_out:
            status = f"Outage (< {gamma_min_db:g} dB)"
        else:
            status = "Ativo"

        ues_list.append(
            UESnapshot(
                ue_id=k,
                x=float(ue_xy[k, 0]),
                y=float(ue_xy[k, 1]),
                z=float(config.h_ue_m),
                serving_ap=m_id,
                ap_x=float(ap_xy[m_id, 0]),
                ap_y=float(ap_xy[m_id, 1]),
                distance_m=float(ue_dists[k]),
                aoa_deg=float(np.rad2deg(ue_aoa[k])),
                channel=int(channels[k]) + 1,
                serving_beam=int(serving_beam[k]),
                tx_power_w=p_w,
                tx_power_dbm=p_dbm,
                rx_gain_db=rx_g_db,
                snr_db=snr_db,
                sinr_lin=s_lin,
                sinr_db=s_db,
                se_bps_hz=float(se_ue[k]),
                capacity_mbps=float(cap_mbps[k]),
                is_outage=is_out,
                is_blocked=is_blocked,
                status=status,
                gamma_tar_db=float(gamma_tar[k]) if pwr_mode == "dual_loop" else None,
                is_aggregated=bool(is_agg[k]),
            )
        )

    aps_list: list[APSnapshot] = []
    for m_id in range(config.M):
        conn = [u for u in ues_list if u.serving_ap == m_id]
        aps_list.append(
            APSnapshot(
                ap_id=m_id,
                x=float(ap_xy[m_id, 0]),
                y=float(ap_xy[m_id, 1]),
                z=float(config.h_ap_m),
                connected_ue_ids=[u.ue_id for u in conn],
                num_ues=len(conn),
                sum_capacity_mbps=float(sum(u.capacity_mbps for u in conn)),
                num_outage=sum(1 for u in conn if u.is_outage),
            )
        )

    active_sinr_db = [u.sinr_db for u in ues_list if not u.is_blocked]
    avg_sinr_db = float(np.mean(active_sinr_db)) if active_sinr_db else float(gamma_min_db)
    p10_sinr_db = (
        float(np.percentile(active_sinr_db, 10)) if active_sinr_db else float(gamma_min_db)
    )
    sum_cap_mbps = float(ev["sum_cap_bps"] / 1e6)
    tot_p_w = float(ev["total_power_w"])
    ee_mbit_j = (sum_cap_mbps / tot_p_w) if tot_p_w > 0.0 else 0.0
    outage_pct = float(np.mean(outage_mask) * 100.0) if K > 0 else 0.0
    fairness = jain_fairness(cap_mbps)
    ase = area_spectrum_efficiency(
        ev["sum_cap_bps"], config.channel.b_total_hz, config.area_m**2
    )

    heatmap_data = (
        compute_coverage_heatmap(
            ap_xy, config, antenna_model=ant_model, res=heatmap_res, metric=heatmap_metric
        )
        if include_heatmap
        else None
    )

    conv_hist = ev["conv_hist"]
    if conv_hist is not None:
        conv_hist["K"] = K
        conv_hist["seed"] = snap_seed
        conv_hist["antenna_model"] = ant_model
        conv_hist["power_mode"] = pwr_mode

    return SnapshotResult(
        seed=snap_seed,
        area_m=config.area_m,
        antenna_model=ant_model,
        power_mode=pwr_mode,
        allocation_policy=config.allocation_policy,
        L=config.antenna.L,
        J=config.antenna.J,
        N=config.N,
        beamwidth_deg=config.antenna.beamwidth_deg,
        g_min=config.antenna.g_min,
        ap_xy=ap_xy,
        ue_xy=ue_xy,
        serving_ap=serving_ap,
        serving_beam=serving_beam,
        channels=channels + 1,
        distances_m=ue_dists,
        aoa_rad=ue_aoa,
        beam_angles_rad=ev["beam_angles_rad"],
        G_eff=ev["G_eff"],
        ues=ues_list,
        aps=aps_list,
        avg_sinr_db=avg_sinr_db,
        p10_sinr_db=p10_sinr_db,
        sum_se_bps_hz=float(ev["sum_se"]),
        sum_capacity_mbps=sum_cap_mbps,
        total_tx_power_w=tot_p_w,
        energy_efficiency_mbit_j=ee_mbit_j,
        outage_rate_pct=outage_pct,
        jain_fairness=fairness,
        ase_bps_hz_km2=ase,
        convergence_history=conv_hist,
        heatmap=heatmap_data,
    )


# ---------------------------------------------------------------------------
# API Pública 2: Monte Carlo Multi-Carga Vetorizado (Methodology Remark)
# ---------------------------------------------------------------------------


def _consolidate_scenario_metrics(
    key: str,
    K: int,
    buf: dict[str, list[np.ndarray] | list[float]],
    config: SimulationConfig,
) -> ScenarioMetrics:
    """Consolida arrays brutos de todas as rodadas Monte Carlo em `ScenarioMetrics`."""
    sinr_lin = np.concatenate(buf["sinr_lin"])
    se_bps_hz = np.concatenate(buf["se"])
    cap_mbps = np.concatenate(buf["cap"])
    tx_w = np.concatenate(buf["tx_w"])
    rx_w = np.concatenate(buf["rx_w"])
    out_flags = np.concatenate(buf["outage"]).astype(bool)
    sum_cap_runs = np.asarray(buf["sum_cap_run"], dtype=float)
    tot_p_runs = np.asarray(buf["tot_p_run"], dtype=float)

    # Para CDF de SINR: UEs bloqueados (0 W) ficam em gamma_min - 5 dB (evita -inf)
    floor_db = config.power_control.gamma_min_db - 5.0
    sinr_db = np.where(
        (tx_w > 0.0) & (sinr_lin > 0.0),
        10.0 * np.log10(np.maximum(sinr_lin, 1e-12)),
        floor_db,
    )

    sinr_active_db = sinr_db[tx_w > 0.0]
    if sinr_active_db.size > 0:
        s_p10 = float(np.percentile(sinr_active_db, 10))
        s_p50 = float(np.percentile(sinr_active_db, 50))
        s_p90 = float(np.percentile(sinr_active_db, 90))
    else:
        s_p10 = s_p50 = s_p90 = floor_db

    sum_cap_mean_mbps = float(np.mean(sum_cap_runs))
    mean_p_w = float(np.mean(tot_p_runs))
    bw_ch_mhz = config.bw_channel_hz / 1e6

    return ScenarioMetrics(
        key=key,
        label=MODEL_LABELS.get(key, key),
        K=K,
        sinr_lin=sinr_lin,
        sinr_db=sinr_db,
        se_bps_hz=se_bps_hz,
        capacity_mbps=cap_mbps,
        tx_power_w=tx_w,
        rx_power_w=rx_w,
        outage_flags=out_flags,
        sum_cap_per_run_mbps=sum_cap_runs,
        total_power_per_run_w=tot_p_runs,
        border_se_p10=float(np.percentile(se_bps_hz, 10)),
        sum_se_mean=float(sum_cap_mean_mbps / bw_ch_mhz) if bw_ch_mhz > 0 else 0.0,
        sum_cap_mean_mbps=sum_cap_mean_mbps,
        sinr_p10_db=s_p10,
        sinr_p50_db=s_p50,
        sinr_p90_db=s_p90,
        cap_p10_mbps=float(np.percentile(cap_mbps, 10)),
        mean_tx_power_w=mean_p_w,
        energy_efficiency_mbit_j=(sum_cap_mean_mbps / mean_p_w) if mean_p_w > 0.0 else 0.0,
        outage_rate_pct=float(np.mean(out_flags) * 100.0),
        jain_fairness=jain_fairness(cap_mbps),
        ase_bps_hz_km2=area_spectrum_efficiency(
            sum_cap_mean_mbps * 1e6, config.channel.b_total_hz, config.area_m**2
        ),
    )


def run_snapshot_comparison(
    config: SimulationConfig,
    seed: Optional[int] = None,
    K_override: Optional[int] = None,
    base_antenna: Optional[str] = None,
    base_power: Optional[str] = None,
) -> dict[str, dict[str, SnapshotResult]]:
    """
    Avalia instantaneamente, na MESMA seed e MESMO canal do Snapshot:
      - `antennas`: todos os modelos selecionados em `config.antenna.models` (se > 1)
        usando `base_power` fixo;
      - `power`: todos os algoritmos selecionados em `config.power_control.algorithms` (se > 1)
        usando `base_antenna` fixo.
    """
    snap_seed = config.seed if seed is None else int(seed)
    ant_fix = base_antenna or config.snapshot_antenna_model
    pwr_fix = base_power or config.snapshot_power_mode

    ant_snaps: dict[str, SnapshotResult] = {}
    if len(config.antenna.models) > 1:
        for m in config.antenna.models:
            ant_snaps[m] = run_single_snapshot(
                config, antenna_model=m, power_mode=pwr_fix, seed=snap_seed, K_override=K_override
            )

    pwr_snaps: dict[str, SnapshotResult] = {}
    if len(config.power_control.algorithms) > 1:
        for a in config.power_control.algorithms:
            pwr_snaps[a] = run_single_snapshot(
                config, antenna_model=ant_fix, power_mode=a, seed=snap_seed, K_override=K_override
            )

    return {"antennas": ant_snaps, "power": pwr_snaps}


def resolve_comparison_flags(config: SimulationConfig) -> tuple[bool, bool, str]:
    """
    Resolve automaticamente quais comparativos executar no Monte Carlo:
      - 'auto': executa comparativo de Antenas sempre que >1 antena estiver selecionada,
                e comparativo de Potência sempre que >1 algoritmo estiver selecionado
                (se ambos tiverem tamanho 1, executa o grupo de potência para manter KPIs).
      - 'antennas' / 'power' / 'both': respeita override explícito.
    """
    mode = config.comparison_mode
    if mode == "auto":
        multi_ant = len(config.antenna.models) > 1
        multi_pwr = len(config.power_control.algorithms) > 1
        if multi_ant and multi_pwr:
            return True, True, "both"
        if multi_ant:
            return True, False, "antennas"
        if multi_pwr:
            return False, True, "power"
        return True, True, "both"
    return mode in ("antennas", "both"), mode in ("power", "both"), mode


def run_monte_carlo(
    config: SimulationConfig,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    n_saved_histories: int = 5,
) -> MonteCarloResult:
    """
    Executa bateria Monte Carlo 100% vetorizada para todos os cenários de carga
    em `config.k_scenarios`, respeitando estritamente o Methodology Remark
    (mesmo canal, posições, shadowing e Rayleigh para todos os modelos/algoritmos).

    Modos de comparação:
      - 'auto'     : detecta automaticamente múltiplas antenas e/ou múltiplos algoritmos.
      - 'antennas' : compara `config.antenna.models` com `config.snapshot_power_mode` fixo.
      - 'power'    : compara `config.power_control.algorithms` com `config.snapshot_antenna_model` fixo.
      - 'both'     : executa ambas as comparações limpas em paralelo sobre os mesmos canais.
    """
    t0 = time.perf_counter()
    ap_xy = distribute_aps_xy(config.M, config.area_m)
    k_list = config.k_scenarios
    n_runs = config.monte_carlo_runs

    run_ant, run_pwr, effective_mode = resolve_comparison_flags(config)
    ant_models = config.antenna.models if run_ant else ()
    pwr_algos = config.power_control.algorithms if run_pwr else ()

    insp_ant = (
        config.snapshot_antenna_model
        if config.snapshot_antenna_model in ant_models
        else (ant_models[0] if ant_models else None)
    )
    insp_pwr = (
        config.snapshot_power_mode
        if config.snapshot_power_mode in pwr_algos
        else (pwr_algos[0] if pwr_algos else None)
    )
    iter_algos = ("dual_loop", "dpc_fixed", "max_sum", "max_prod")
    hist_pwr_target = (
        "dual_loop"
        if "dual_loop" in pwr_algos
        else next((a for a in pwr_algos if a in iter_algos), None)
    )

    antenna_metrics: dict[str, dict[int, ScenarioMetrics]] = {m: {} for m in ant_models}
    power_metrics: dict[str, dict[int, ScenarioMetrics]] = {a: {} for a in pwr_algos}
    conv_histories: dict[int, list[dict[str, Any]]] = {k: [] for k in k_list}

    total_steps = len(k_list) * n_runs
    done_steps = 0
    cb_stride = max(1, total_steps // 40)

    # Metadados das rodadas para o Inspetor de Rodadas (focado no K do snapshot, ou primeiro K)
    inspector_k = config.K if config.K in k_list else k_list[0]
    inspector_records: list[dict[str, Any]] = []

    for k_idx, K in enumerate(k_list):
        base_seed_k = config.seed + k_idx * 1000
        ant_bufs: dict[str, dict[str, list]] = {
            m: {
                "sinr_lin": [],
                "se": [],
                "cap": [],
                "tx_w": [],
                "rx_w": [],
                "outage": [],
                "sum_cap_run": [],
                "tot_p_run": [],
            }
            for m in ant_models
        }
        pwr_bufs: dict[str, dict[str, list]] = {
            a: {
                "sinr_lin": [],
                "se": [],
                "cap": [],
                "tx_w": [],
                "rx_w": [],
                "outage": [],
                "sum_cap_run": [],
                "tot_p_run": [],
            }
            for a in pwr_algos
        }

        for r_idx in range(n_runs):
            r_seed = mc_round_seed(base_seed_k, r_idx)
            rng = np.random.default_rng(r_seed)

            ue_xy = sample_ue_xy(K, ap_xy, config.area_m, config.channel.d_0, rng=rng)
            phi = angle_matrix(ue_xy, ap_xy)
            shadowing = gerar_shadowing(
                K,
                config.M,
                sigma_x=config.channel.sigma_x,
                use_shadowing=config.channel.use_shadowing,
                rng=rng,
            )
            rayleigh = (
                gerar_rayleigh(K, config.M, sigma_r=config.channel.sigma_r, rng=rng)
                if config.channel.use_rayleigh
                else None
            )
            aoa_unit_noise = rng.normal(0.0, 1.0, size=K)

            # --- Comparativo 1: Modelos de Antena (potência fixa do snapshot) ---
            for m in ant_models:
                ch_rng = np.random.default_rng(r_seed + 17)
                want_h = (
                    (not run_pwr)
                    and m == insp_ant
                    and config.snapshot_power_mode in iter_algos
                    and r_idx < n_saved_histories
                )
                ev = _evaluate_single_realization(
                    ue_xy,
                    ap_xy,
                    phi,
                    shadowing,
                    rayleigh,
                    aoa_unit_noise,
                    config,
                    antenna_model=m,
                    power_mode=config.snapshot_power_mode,
                    channel_rng=ch_rng,
                    record_history=want_h,
                )
                b = ant_bufs[m]
                b["sinr_lin"].append(ev["sinr_lin"])
                b["se"].append(ev["se_per_ue"])
                b["cap"].append(ev["cap_mbps"])
                b["tx_w"].append(ev["p_watts"])
                b["rx_w"].append(ev["rx_power_w"])
                b["outage"].append(ev["outage_mask"])
                b["sum_cap_run"].append(ev["sum_cap_bps"] / 1e6)
                b["tot_p_run"].append(ev["total_power_w"])

                if want_h and ev["conv_hist"] is not None:
                    ev["conv_hist"]["K"] = K
                    ev["conv_hist"]["snapshot"] = r_idx
                    ev["conv_hist"]["seed"] = r_seed
                    ev["conv_hist"]["antenna_model"] = m
                    conv_histories[K].append(ev["conv_hist"])

                if K == inspector_k and (not run_pwr) and m == insp_ant:
                    inspector_records.append(
                        {
                            "round": r_idx + 1,
                            "seed": r_seed,
                            "sum_cap_mbps": ev["sum_cap_bps"] / 1e6,
                            "outage_cnt": int(np.sum(ev["outage_mask"])),
                        }
                    )

            # --- Comparativo 2: Algoritmos de Controle de Potência (antena fixa do snapshot) ---
            for algo in pwr_algos:
                ch_rng = np.random.default_rng(r_seed + 17)
                want_h = algo == hist_pwr_target and r_idx < n_saved_histories
                ev = _evaluate_single_realization(
                    ue_xy,
                    ap_xy,
                    phi,
                    shadowing,
                    rayleigh,
                    aoa_unit_noise,
                    config,
                    antenna_model=config.snapshot_antenna_model,
                    power_mode=algo,
                    channel_rng=ch_rng,
                    record_history=want_h,
                )
                b = pwr_bufs[algo]
                b["sinr_lin"].append(ev["sinr_lin"])
                b["se"].append(ev["se_per_ue"])
                b["cap"].append(ev["cap_mbps"])
                b["tx_w"].append(ev["p_watts"])
                b["rx_w"].append(ev["rx_power_w"])
                b["outage"].append(ev["outage_mask"])
                b["sum_cap_run"].append(ev["sum_cap_bps"] / 1e6)
                b["tot_p_run"].append(ev["total_power_w"])

                if want_h and ev["conv_hist"] is not None:
                    ev["conv_hist"]["K"] = K
                    ev["conv_hist"]["snapshot"] = r_idx
                    ev["conv_hist"]["seed"] = r_seed
                    ev["conv_hist"]["antenna_model"] = config.snapshot_antenna_model
                    conv_histories[K].append(ev["conv_hist"])

                if K == inspector_k and algo == insp_pwr:
                    inspector_records.append(
                        {
                            "round": r_idx + 1,
                            "seed": r_seed,
                            "sum_cap_mbps": ev["sum_cap_bps"] / 1e6,
                            "outage_cnt": int(np.sum(ev["outage_mask"])),
                        }
                    )

            done_steps += 1
            if progress_callback is not None and (
                done_steps % cb_stride == 0 or done_steps == total_steps
            ):
                progress_callback(
                    done_steps / total_steps,
                    f"K={K} UEs — rodada {r_idx + 1}/{n_runs} ({100.0 * done_steps / total_steps:.0f}%)",
                )

        for m in ant_models:
            antenna_metrics[m][K] = _consolidate_scenario_metrics(m, K, ant_bufs[m], config)
        for algo in pwr_algos:
            power_metrics[algo][K] = _consolidate_scenario_metrics(algo, K, pwr_bufs[algo], config)

    # Inspetor de rodadas (1ª, última, melhor, mediana, pior)
    round_inspector: Optional[dict[str, Any]] = None
    if inspector_records:
        ordered = sorted(inspector_records, key=lambda r: (r["sum_cap_mbps"], -r["outage_cnt"]))
        round_inspector = {
            "K": inspector_k,
            "n_runs": n_runs,
            "presets": {
                "first": inspector_records[0],
                "last": inspector_records[-1],
                "best": ordered[-1],
                "median": ordered[len(ordered) // 2],
                "worst": ordered[0],
            },
            "all_seeds": [r["seed"] for r in inspector_records],
        }

    rep_snapshot = run_single_snapshot(config, seed=config.seed)
    elapsed_s = time.perf_counter() - t0

    return MonteCarloResult(
        config=config,
        comparison_mode=effective_mode,
        k_scenarios=k_list,
        elapsed_s=elapsed_s,
        representative_snapshot=rep_snapshot,
        antenna_metrics=antenna_metrics,
        power_metrics=power_metrics,
        convergence_histories=conv_histories,
        round_inspector=round_inspector,
    )


# ---------------------------------------------------------------------------
# Wrappers de Retrocompatibilidade (main.py e research-activity-01.py)
# ---------------------------------------------------------------------------


def distribuir_AP(M: int) -> list[AP]:
    """Posiciona M APs em uma grade uniforme dentro da área 1000 × 1000 m (objetos AP)."""
    xy = distribute_aps_xy(M, AREA_SIZE)
    return [AP(float(xi), float(yi), idx) for idx, (xi, yi) in enumerate(xy)]


def cenario_aleatorio(M: int, K: int, seed: Optional[int] = None) -> tuple:
    """Gera um cenário (aps, ues, shadowing) reproduzível para visualização CLI."""
    if seed is not None:
        np.random.seed(seed)
    aps = distribuir_AP(M)
    UE.id_counter = 0
    ues = [UE(aps) for _ in range(K)]
    shadowing = gerar_shadowing(K, M)
    return aps, ues, shadowing


def _build_effective_gain_matrix(
    ues: list, gains: np.ndarray, Gm: Optional[np.ndarray] = None, is_adaptive: bool = False
) -> np.ndarray:
    """Constrói matriz K×K de ganho co-canal efetivo vista no AP servidor de cada UE."""
    K = len(ues)
    if K == 0:
        return np.zeros((0, 0), dtype=float)
    ap_ids = np.array([ue.ap.id for ue in ues], dtype=int)
    channels = np.array([ue.channel for ue in ues])
    if is_adaptive and Gm is not None:
        G_raw = gains[:, ap_ids, 0].T * Gm
    else:
        beams = np.array([ue.beam for ue in ues], dtype=int)
        G_raw = gains[:, ap_ids, beams].T
    same_ch = channels[:, None] == channels[None, :]
    return np.where(same_ch, G_raw, 0.0)


def aplicar_controle_potencia(
    ues: list,
    N: int,
    gains: np.ndarray,
    G: np.ndarray,
    Gm: Optional[np.ndarray] = None,
    is_adaptive: bool = False,
    power_mode: str = "full_power",
    b_total: float = B_TOTAL,
    k_0: float = K_0,
    p_max_dbm: float = P_MAX_DBM,
    p_min_dbm: float = P_MIN_DBM,
    mu_step: float = MU_STEP,
    gamma_tar_db: float = GAMMA_TAR_INIT,
    gamma_max_db: float = GAMMA_MAX_DB,
    gamma_min_db: float = GAMMA_MIN_DB,
    t_outer: int = 35,
    t_inner: int = 25,
    tol_db: float = 0.5,
    return_history: bool = False,
) -> np.ndarray | tuple:
    """Wrapper retrocompatível para aplicação de controle de potência sobre objetos UE."""
    K = len(ues)
    if K == 0:
        empty = np.zeros(0, dtype=float)
        return (empty, None) if return_history else empty

    pn = k_0 * (b_total / N)
    pc_cfg = PowerControlConfig(
        algorithms=(power_mode,),
        p_max_dbm=p_max_dbm,
        p_min_dbm=p_min_dbm,
        mu_step_db=mu_step,
        gamma_tar_init_db=gamma_tar_db,
        gamma_max_db=gamma_max_db,
        gamma_min_db=gamma_min_db,
        t_outer=t_outer,
        t_inner=t_inner,
        tol_db=tol_db,
    )
    G_eff = _build_effective_gain_matrix(ues, gains, Gm=Gm, is_adaptive=is_adaptive)
    p_w, sinr_arr, _, hist = run_power_control(
        power_mode, G_eff, pn, pc_cfg, record_history=return_history
    )
    for i, ue in enumerate(ues):
        ue.power = float(p_w[i])

    # Fallback para compatibilidade de saída quando power_mode == full_power com SINR legado
    _ = G
    if power_mode == "full_power":
        sinr_arr = np.asarray(
            SINR_adaptive(ues, N, gains, Gm, b_total=b_total, k0=k_0)
            if is_adaptive and Gm is not None
            else SINR(ues, N, gains, G, b_total=b_total, k0=k_0),
            dtype=float,
        ).flatten()
    return (sinr_arr, hist) if return_history else sinr_arr


def simular_experimento(
    M: int,
    N: int,
    K: int,
    sim: int,
    allocation: str = "ortogonal",
    antenna_models: tuple[str, ...] = (
        "omni",
        "adaptive",
        "adaptive_1",
        "adaptive_5",
        "adaptive_10",
    ),
    shadowing_params: Optional[dict] = None,
    use_rayleigh: bool = False,
    L: int = 8,
    J: int = 8,
    beamwidth_deg: float = 120.0,
    g_min: float = 0.01,
    aoa_err_deg: float = 0.0,
    b_total: float = B_TOTAL,
    k_0: float = K_0,
    pathloss_exp: float = PATHLOSS_EXP,
    k_const: float = K_CONST,
    d_0: float = D_0,
    sigma_x: float = SIGMA_X,
    sigma_r: float = SIGMA_R,
    p_max_dbm: float = P_MAX_DBM,
    power_mode: str = "full_power",
    p_min_dbm: float = P_MIN_DBM,
    mu_step: float = MU_STEP,
    gamma_tar_db: float = GAMMA_TAR_INIT,
    gamma_max_db: float = GAMMA_MAX_DB,
    gamma_min_db: float = GAMMA_MIN_DB,
    t_outer: int = 35,
    t_inner: int = 25,
    tol_db: float = 0.5,
    seed: Optional[int] = None,
) -> dict:
    """
    Executa 'sim' iterações de Monte Carlo para os modelos de antena indicados
    usando cálculos 100% vetorizados em NumPy, mantendo o contrato legado de `main.py`.
    """
    _ = shadowing_params
    rng = np.random.default_rng(seed)
    ap_xy = distribute_aps_xy(M, AREA_SIZE)
    policy = "random" if allocation == "random" else "papoa"
    pn = k_0 * (b_total / N)
    pc_cfg = PowerControlConfig(
        algorithms=(power_mode,),
        p_max_dbm=p_max_dbm,
        p_min_dbm=p_min_dbm,
        mu_step_db=mu_step,
        gamma_tar_init_db=gamma_tar_db,
        gamma_max_db=gamma_max_db,
        gamma_min_db=gamma_min_db,
        t_outer=t_outer,
        t_inner=t_inner,
        tol_db=tol_db,
    )

    resultados: dict[str, dict[str, Any]] = {
        model: {"power": [], "tx_power": [], "sinr": [], "cap": [], "sum_cap": []}
        for model in antenna_models
    }

    for _ in tqdm(range(sim), disable=(sys.stdout is None)):
        ue_xy = sample_ue_xy(K, ap_xy, AREA_SIZE, d_0, rng=rng)
        phi = angle_matrix(ue_xy, ap_xy)
        shadowing = gerar_shadowing(K, M, sigma_x=sigma_x, rng=rng)
        rayleigh = gerar_rayleigh(K, M, sigma_r=sigma_r, rng=rng) if use_rayleigh else None
        aoa_noise_u = rng.normal(0.0, 1.0, size=K)
        ch_seed = int(rng.integers(0, 2**31 - 1))

        for model in antenna_models:
            G = compute_antenna_gain_matrix(
                phi,
                model=model,
                beamwidth=np.deg2rad(beamwidth_deg),
                g_min=g_min,
                L=L,
                J=J,
            )
            gains = gain_matrix(
                ue_xy,
                ap_xy,
                G,
                shadowing=shadowing,
                rayleigh=rayleigh,
                use_rayleigh=use_rayleigh,
                pathloss_exp=pathloss_exp,
                k_const=k_const,
                d_0=d_0,
                sigma_x=sigma_x,
                sigma_r=sigma_r,
            )
            serving_ap, serving_beam = attach_AP_UE(gains)
            channels, _ = allocate_channels_vec(
                serving_ap, M=M, N=N, policy=policy, rng=np.random.default_rng(ch_seed)
            )
            adaptive = is_adaptive_model(model)
            if adaptive:
                err_deg = resolve_aoa_error_deg(model, aoa_err_deg)
                array_type = "ula" if model == "ula" else "uca"
                Gm, _ = adaptive_gains_vec(
                    phi,
                    serving_ap,
                    L=L,
                    aoa_err_deg=err_deg,
                    array_type=array_type,
                    aoa_noise_rad=np.deg2rad(err_deg) * aoa_noise_u if err_deg > 0.0 else None,
                )
            else:
                Gm = None

            G_raw = _build_full_spatial_gain_matrix(gains, serving_ap, serving_beam, Gm, adaptive)
            same_ch = channels[:, None] == channels[None, :]
            G_eff = np.where(same_ch, G_raw, 0.0)

            p_w, s, _, _ = run_power_control(power_mode, G_eff, pn, pc_cfg)
            rx_p = np.diagonal(G_eff) * p_w
            cap = np.asarray(channel_capacity(s, N, b_total=b_total)).flatten()
            if power_mode == "dual_loop":
                cap = np.where(p_w > 0.0, cap, 0.0)

            resultados[model]["power"].extend(rx_p.tolist())
            resultados[model]["tx_power"].append(float(np.sum(p_w)))
            resultados[model]["sinr"].extend(s.tolist())
            resultados[model]["cap"].extend(cap.tolist())
            resultados[model]["sum_cap"].append(float(np.sum(cap)))

    for model in antenna_models:
        resultados[model]["sum_cap"] = float(np.mean(resultados[model]["sum_cap"]))
        resultados[model]["tx_power"] = float(np.mean(resultados[model]["tx_power"]))

    return resultados


def simular_scichal4a(
    M: int,
    N: int,
    K: int,
    sim: int,
    n_hist: int = 5,
    L: int = 8,
    use_rayleigh: bool = True,
    b_total: float = B_TOTAL,
    k_0: float = K_0,
    pathloss_exp: float = PATHLOSS_EXP,
    k_const: float = K_CONST,
    d_0: float = D_0,
    sigma_x: float = SIGMA_X,
    sigma_r: float = SIGMA_R,
    p_max_dbm: float = P_MAX_DBM,
    p_min_dbm: float = P_MIN_DBM,
    mu_step: float = MU_STEP,
    gamma_tar_init: float = GAMMA_TAR_INIT,
    gamma_max_db: float = GAMMA_MAX_DB,
    gamma_min_db: float = GAMMA_MIN_DB,
    t_outer: int = 35,
    t_inner: int = 25,
    tol_db: float = 0.5,
    aoa_err_deg: float = 0.0,
    seed: Optional[int] = None,
    progress_cb=None,
) -> dict:
    """
    Executa a simulação Monte Carlo vetorizada para o SciChal4A (Full Power, DPC Fixo e Dual-Loop).
    Preservada para retrocompatibilidade com `research-activity-01.py`.
    """
    rng = np.random.default_rng(seed if (seed is not None and seed >= 0) else None)
    ap_xy = distribute_aps_xy(M, AREA_SIZE)
    B_c = b_total / N
    P_n = B_c * k_0

    pc_cfg = PowerControlConfig(
        algorithms=("full_power", "dpc_fixed", "dual_loop"),
        p_max_dbm=p_max_dbm,
        p_min_dbm=p_min_dbm,
        mu_step_db=mu_step,
        gamma_tar_init_db=gamma_tar_init,
        gamma_max_db=gamma_max_db,
        gamma_min_db=gamma_min_db,
        t_outer=t_outer,
        t_inner=t_inner,
        tol_db=tol_db,
    )

    algorithms = ["full_power", "dpc_fixed", "dual_loop"]
    stats: dict[str, Any] = {
        algo: {
            "se": [],
            "sum_cap": [],
            "power": [],
            "outage": [],
            "sinr_lin": [],
            "power_dist": [],
        }
        for algo in algorithms
    }
    stats["dual_loop_hist"] = []

    tqdm_stream = sys.stdout if sys.stdout is not None else sys.stderr
    step_cb = max(1, sim // 10)
    for sim_idx in tqdm(
        range(sim),
        desc=f"Monte Carlo K={K}",
        disable=(tqdm_stream is None),
        file=tqdm_stream,
    ):
        ue_xy = sample_ue_xy(K, ap_xy, AREA_SIZE, d_0, rng=rng)
        phi = angle_matrix(ue_xy, ap_xy)
        G_omni = np.ones((K, M, 1), dtype=float)
        gains_base = gain_matrix(
            ue_xy,
            ap_xy,
            G_omni,
            use_rayleigh=use_rayleigh,
            pathloss_exp=pathloss_exp,
            k_const=k_const,
            d_0=d_0,
            sigma_x=sigma_x,
            sigma_r=sigma_r,
            rng=rng,
        )
        serving_ap, _ = attach_AP_UE(gains_base)
        channels, _ = allocate_channels_vec(serving_ap, M=M, N=N, policy="papoa", rng=rng)
        Gm, _ = adaptive_gains_vec(phi, serving_ap, L=L, aoa_err_deg=aoa_err_deg, rng=rng)
        G_raw = gains_base[:, serving_ap, 0].T * Gm
        G_eff = np.where(channels[:, None] == channels[None, :], G_raw, 0.0)

        # 1. Full Power
        p_fp = np.full(K, pc_cfg.p_max_w, dtype=float)
        sinr_fp = calculate_sinr_vector(G_eff, p_fp, P_n)
        se, _, sum_cap, tot_p, out = evaluate_physical_kpis(
            sinr_fp, p_fp, B_c, gamma_min_db=gamma_min_db, gamma_max_db=gamma_max_db
        )
        stats["full_power"]["se"].extend(se)
        stats["full_power"]["sum_cap"].append(sum_cap)
        stats["full_power"]["power"].append(tot_p)
        stats["full_power"]["outage"].extend(out)
        stats["full_power"]["sinr_lin"].extend(sinr_fp)
        stats["full_power"]["power_dist"].extend(p_fp)

        # 2. DPC Fixo
        p_dpc, sinr_dpc, _, _ = pc_dpc_fixed(G_eff, P_n, pc_cfg, record_history=False)
        se, _, sum_cap, tot_p, out = evaluate_physical_kpis(
            sinr_dpc, p_dpc, B_c, gamma_min_db=gamma_min_db, gamma_max_db=gamma_max_db
        )
        stats["dpc_fixed"]["se"].extend(se)
        stats["dpc_fixed"]["sum_cap"].append(sum_cap)
        stats["dpc_fixed"]["power"].append(tot_p)
        stats["dpc_fixed"]["outage"].extend(out)
        stats["dpc_fixed"]["sinr_lin"].extend(sinr_dpc)
        stats["dpc_fixed"]["power_dist"].extend(p_dpc)

        # 3. Dual-Loop PC
        want_hist = sim_idx < n_hist
        out_dl = DPC_Dual_Loop(
            G_eff,
            P_n,
            np.full(K, float(gamma_tar_init)),
            np.ones(K, dtype=bool),
            t_outer=t_outer,
            t_inner=t_inner,
            tol_db=tol_db,
            return_history=want_hist,
            p_max_dbm=p_max_dbm,
            p_min_dbm=p_min_dbm,
            mu_step=mu_step,
            gamma_max_db=gamma_max_db,
            gamma_min_db=gamma_min_db,
        )
        if want_hist:
            p_dual, sinr_dual, hist = out_dl
            hist["K"] = K
            hist["snapshot"] = sim_idx
            stats["dual_loop_hist"].append(hist)
        else:
            p_dual, sinr_dual = out_dl

        se, _, sum_cap, tot_p, out = evaluate_physical_kpis(
            sinr_dual, p_dual, B_c, gamma_min_db=gamma_min_db, gamma_max_db=gamma_max_db
        )
        stats["dual_loop"]["se"].extend(se)
        stats["dual_loop"]["sum_cap"].append(sum_cap)
        stats["dual_loop"]["power"].append(tot_p)
        stats["dual_loop"]["outage"].extend(out)
        stats["dual_loop"]["sinr_lin"].extend(sinr_dual)
        stats["dual_loop"]["power_dist"].extend(p_dual)

        if progress_cb is not None and ((sim_idx + 1) % step_cb == 0 or sim_idx == sim - 1):
            progress_cb(sim_idx + 1, sim)

    return stats


# Alias para compatibilidade com scripts que referenciam o nome vetorizado
simular_scichal4a_vetorizado = simular_scichal4a

# Re-exportações auxiliares usadas em chamadas antigas
__all__ = [
    "MC_SEED_STRIDE",
    "mc_round_seed",
    "distribute_aps_xy",
    "sample_ue_xy",
    "compute_coverage_heatmap",
    "run_single_snapshot",
    "run_monte_carlo",
    "distribuir_AP",
    "cenario_aleatorio",
    "aplicar_controle_potencia",
    "simular_experimento",
    "simular_scichal4a",
    "simular_scichal4a_vetorizado",
    "antenna_gain",
    "adaptive_gains",
    "alocar_canais_ortogonal",
]

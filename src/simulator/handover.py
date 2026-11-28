# ---------------------------------------------------------------------------
# handover.py — Motor de Handover (Hard vs. Soft) e Detecção de Outage
#               Baseado na Parte 2 (Seção 8, Exercícios 12 e 13) com suporte
#               a múltiplos APs, margem de histerese, TTT e conjunto ativo
# ---------------------------------------------------------------------------

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .config import B_TOTAL, D_0, K_0, K_CONST, PATHLOSS_EXP, SIGMA_X
from .schemas import (
    ChannelConfig,
    HandoverConfig,
    HandoverResult,
    NetworkHandoverResult,
    SimulationConfig,
)


def _eval_hard_handover_run(
    rx_power_w: np.ndarray,
    p_min_w: float,
    hysteresis_db: float,
    ttt_steps: int,
    init_ap: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Executa Hard Handover para uma realização de potências recebidas `rx_power_w` (M, S).

    - Modo fiel ao Exercício 13 (hysteresis_db == 0 e ttt_steps == 0):
      Monitora apenas o AP atual; se P_curr < P_min, registra Outage no instante i
      e troca para o próximo/melhor AP alternativo a partir do instante i+1.
    - Modo com Histerese / Time-to-Trigger (hysteresis_db > 0 ou ttt_steps > 0):
      Troca proativamente para o melhor candidato j quando P_j(dB) > P_curr(dB) + hyst
      por `ttt_steps` amostras consecutivas, além de disparar troca reativa em Outage.

    Retorna: (serving_ap_0idx (S,), outage_mask (S,), served_power_w (S,), num_handovers).
    """
    M, S = rx_power_w.shape
    serving = np.zeros(S, dtype=int)
    outage = np.zeros(S, dtype=bool)
    served_p = np.zeros(S, dtype=float)

    hyst_lin = 10.0 ** (hysteresis_db / 10.0)
    use_a3 = (hysteresis_db > 0.0) or (ttt_steps > 0)
    required_ttt = max(1, ttt_steps)

    curr_ap = int(init_ap)
    cand_ap = -1
    ttt_counter = 0
    ho_count = 0

    for i in range(S):
        serving[i] = curr_ap
        p_curr = float(rx_power_w[curr_ap, i])
        served_p[i] = p_curr

        if p_curr < p_min_w:
            # Outage no passo atual; handover reativo entra em vigor no passo seguinte
            outage[i] = True
            if M == 2:
                next_ap = 1 - curr_ap
            else:
                # Escolhe o AP alternativo com maior potência no instante seguinte/atual
                others = [m for m in range(M) if m != curr_ap]
                next_ap = max(others, key=lambda m: float(rx_power_w[m, i]))
            if next_ap != curr_ap:
                curr_ap = next_ap
                ho_count += 1
            cand_ap = -1
            ttt_counter = 0
        elif use_a3:
            best_ap = int(np.argmax(rx_power_w[:, i]))
            if best_ap != curr_ap and float(rx_power_w[best_ap, i]) > p_curr * hyst_lin:
                if best_ap == cand_ap:
                    ttt_counter += 1
                else:
                    cand_ap = best_ap
                    ttt_counter = 1
                if ttt_counter >= required_ttt:
                    curr_ap = best_ap
                    ho_count += 1
                    cand_ap = -1
                    ttt_counter = 0
            else:
                cand_ap = -1
                ttt_counter = 0

    return serving, outage, served_p, ho_count


def _eval_soft_handover_run(
    rx_power_w: np.ndarray,
    p_min_w: float,
    active_set_margin_db: float,
    combining: str = "selection",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Executa Soft Handover para uma realização `rx_power_w` (M, S):
      - Modo fiel ao Exercício 13 (`combining == 'selection'` ou `active_set_margin_db == 0`):
        monitora todos os APs e seleciona sempre o maior sinal instantâneo max_m P_m.
      - Modo Macro-Diversidade (`combining == 'macro_diversity'` e `active_set_margin_db > 0`):
        forma conjunto ativo {m : P_m >= P_max / 10^(margin/10)} e combina sinais (Σ P_m).

    Retorna: (primary_ap_0idx (S,), outage_mask (S,), served_power_w (S,), num_handovers).
    """
    primary_ap = np.argmax(rx_power_w, axis=0)
    p_max = np.max(rx_power_w, axis=0)

    if combining == "macro_diversity" and active_set_margin_db > 0.0:
        margin_lin = 10.0 ** (active_set_margin_db / 10.0)
        active_set_mask = rx_power_w >= (p_max[None, :] / margin_lin)
        served_p = np.sum(np.where(active_set_mask, rx_power_w, 0.0), axis=0)
    else:
        served_p = p_max

    outage = served_p < p_min_w
    ho_count = int(np.sum(np.diff(primary_ap) != 0))
    return primary_ap, outage, served_p, ho_count


HANDOVER_REGISTRY: dict[str, Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, int]]] = {
    "hard": _eval_hard_handover_run,
    "soft": _eval_soft_handover_run,
}


def run_handover_simulation(
    ho_cfg: HandoverConfig,
    channel_cfg: Optional[ChannelConfig] = None,
) -> HandoverResult:
    """
    Simula a travessia do UE ao longo de uma rodovia entre M >= 2 APs comparando
    Hard Handover vs. Soft Handover sob realizações de canal idênticas (Remark 4 / Ex. 13).
    """
    rng = np.random.default_rng(ho_cfg.seed)

    M = ho_cfg.num_aps
    ap_positions = np.linspace(0.0, ho_cfg.dist_total_m, M, dtype=float)
    step_m = ho_cfg.velocity_ms * ho_cfg.t_sample_s
    ue_positions = np.arange(
        ho_cfg.start_margin_m,
        ho_cfg.dist_total_m - ho_cfg.start_margin_m + 1e-9,
        step_m,
        dtype=float,
    )
    S = len(ue_positions)
    t_s = np.arange(S, dtype=float) * ho_cfg.t_sample_s

    d_0 = channel_cfg.d_0 if channel_cfg else D_0
    k_const = channel_cfg.k_const if channel_cfg else K_CONST
    n_exp = channel_cfg.pathloss_exp if channel_cfg else PATHLOSS_EXP
    use_shadowing = channel_cfg.use_shadowing if channel_cfg else True
    sigma_x = channel_cfg.sigma_x if channel_cfg else SIGMA_X
    use_rayleigh = channel_cfg.use_rayleigh if channel_cfg else False
    sigma_r = channel_cfg.sigma_r if channel_cfg else 0.7071067811865476
    b_hz = channel_cfg.b_total_hz if channel_cfg else B_TOTAL
    k_0 = channel_cfg.k_0 if channel_cfg else K_0
    p_noise = b_hz * k_0

    # Distâncias (M, S) e ganho determinístico base
    dists = np.maximum(np.abs(ue_positions[None, :] - ap_positions[:, None]), d_0)
    pl_base = ho_cfg.p_tx_w * (k_const / (dists**n_exp))

    ttt_steps = int(round((ho_cfg.ttt_ms / 1000.0) / ho_cfg.t_sample_s)) if ho_cfg.ttt_ms > 0 else 0

    hard_outages_total = 0
    soft_outages_total = 0
    hard_ho_total = 0
    soft_ho_total = 0
    snr_gain_acc = 0.0

    # Trajetória representativa (realização 0)
    ex_rx_w = np.zeros((M, S), dtype=float)
    ex_hard_ap = np.zeros(S, dtype=int)
    ex_soft_ap = np.zeros(S, dtype=int)
    ex_hard_out = np.zeros(S, dtype=bool)
    ex_soft_out = np.zeros(S, dtype=bool)
    ex_hard_p = np.zeros(S, dtype=float)
    ex_soft_p = np.zeros(S, dtype=float)

    for run_idx in range(ho_cfg.mc_runs):
        if use_shadowing and sigma_x > 0.0:
            shad = rng.lognormal(0.0, sigma_x, size=(M, S))
        else:
            shad = np.ones((M, S), dtype=float)

        if use_rayleigh:
            x1 = rng.normal(0.0, sigma_r, size=(M, S))
            x2 = rng.normal(0.0, sigma_r, size=(M, S))
            ray_sq = x1 * x1 + x2 * x2
        else:
            ray_sq = 1.0

        rx_w = pl_base * shad * ray_sq

        h_ap, h_out, h_p, h_cnt = _eval_hard_handover_run(
            rx_w, ho_cfg.p_min_w, ho_cfg.hysteresis_db, ttt_steps
        )
        s_ap, s_out, s_p, s_cnt = _eval_soft_handover_run(
            rx_w, ho_cfg.p_min_w, ho_cfg.active_set_margin_db, ho_cfg.soft_combining
        )

        hard_outages_total += int(np.sum(h_out))
        soft_outages_total += int(np.sum(s_out))
        hard_ho_total += h_cnt
        soft_ho_total += s_cnt
        snr_gain_acc += float(
            np.mean(10.0 * np.log10(np.maximum(s_p, 1e-30) / np.maximum(h_p, 1e-30)))
        )

        if run_idx == 0:
            ex_rx_w = rx_w
            ex_hard_ap = h_ap
            ex_soft_ap = s_ap
            ex_hard_out = h_out
            ex_soft_out = s_out
            ex_hard_p = h_p
            ex_soft_p = s_p

    total_samples = ho_cfg.mc_runs * S
    return HandoverResult(
        config=ho_cfg,
        ap_positions_m=ap_positions,
        ue_positions_m=ue_positions,
        t_s=t_s,
        rx_power_w=ex_rx_w,
        rx_power_dbm=10.0 * np.log10(np.maximum(ex_rx_w, 1e-30) * 1000.0),
        hard_serving_ap=ex_hard_ap + 1,  # 1-indexed (AP 1, AP 2, ...) como no roteiro
        soft_serving_ap=ex_soft_ap + 1,
        hard_outage_mask=ex_hard_out,
        soft_outage_mask=ex_soft_out,
        hard_rx_power_w=ex_hard_p,
        soft_rx_power_w=ex_soft_p,
        hard_snr_db=10.0 * np.log10(np.maximum(ex_hard_p / p_noise, 1e-30)),
        soft_snr_db=10.0 * np.log10(np.maximum(ex_soft_p / p_noise, 1e-30)),
        hard_handovers_sample=int(np.sum(np.diff(ex_hard_ap) != 0)),
        soft_handovers_sample=int(np.sum(np.diff(ex_soft_ap) != 0)),
        hard_handovers_mean=float(hard_ho_total / ho_cfg.mc_runs),
        soft_handovers_mean=float(soft_ho_total / ho_cfg.mc_runs),
        hard_outage_rate_pct=float((hard_outages_total / total_samples) * 100.0),
        soft_outage_rate_pct=float((soft_outages_total / total_samples) * 100.0),
        mean_snr_gain_soft_vs_hard_db=float(snr_gain_acc / ho_cfg.mc_runs),
    )


# ---------------------------------------------------------------------------
# Motor de Handover (Hard vs. Soft) na REDE COMPLETA (M APs × K UEs)
# ---------------------------------------------------------------------------


def run_network_handover(
    config: SimulationConfig,
    seed: Optional[int] = None,
    K_override: Optional[int] = None,
) -> NetworkHandoverResult:
    """
    Simula Hard Handover vs. Soft Handover para TODOS os K UEs se movendo sobre
    a grade completa de M APs (`area_m × area_m`), com shadowing espacialmente
    correlacionado (modelo exponencial de Gudmundson).
    """
    from .mobility import generate_network_trajectories

    ho_cfg = config.handover
    snap_seed = config.seed if seed is None else int(seed)
    K = config.K if K_override is None else int(K_override)

    dt_ho = max(0.02, float(ho_cfg.ho_step_s))
    t_ho = max(dt_ho * 5.0, float(ho_cfg.t_ho_s))
    n_steps = max(5, int(round(t_ho / dt_ho)) + 1)
    t_s = np.arange(n_steps, dtype=float) * dt_ho

    ap_xy, ue_x, ue_y, ue_velocities_kmh, shad_init = generate_network_trajectories(
        config, t_s, seed=snap_seed, K_override=K
    )
    M = ap_xy.shape[0]

    d_0 = config.channel.d_0
    k_const = config.channel.k_const
    n_exp = config.channel.pathloss_exp
    p_tx_w = config.power_control.p_max_w
    p_noise = config.noise_power_w
    p_min_dbm = float(ho_cfg.p_min_dbm_network)
    p_min_w = 10.0 ** ((p_min_dbm - 30.0) / 10.0)

    # Distâncias (T, K, M)
    dx = ue_x[:, :, None] - ap_xy[None, None, :, 0]
    dy = ue_y[:, :, None] - ap_xy[None, None, :, 1]
    dists = np.maximum(np.hypot(dx, dy), d_0)
    pl_base = p_tx_w * (k_const / (dists**n_exp))

    # Shadowing espacialmente correlacionado (Gudmundson) por (UE, AP)
    rng_ch = np.random.default_rng(snap_seed + 1313)
    if config.channel.use_shadowing and config.channel.sigma_x > 0.0:
        sigma_x = float(config.channel.sigma_x)
        d_corr = max(1.0, float(ho_cfg.shadowing_corr_dist_m))
        step_d = (ue_velocities_kmh / 3.6) * dt_ho  # (K,)
        rho = np.exp(-step_d / d_corr)[:, None]  # (K, 1)
        innov_std = np.sqrt(np.maximum(1.0 - rho**2, 0.0)) * sigma_x

        z_log = np.empty((n_steps, K, M), dtype=float)
        z_log[0] = np.log(np.maximum(shad_init, 1e-12))
        for t_i in range(1, n_steps):
            z_log[t_i] = rho * z_log[t_i - 1] + innov_std * rng_ch.normal(0.0, 1.0, size=(K, M))
        shad_dyn = np.exp(z_log)
    else:
        shad_dyn = np.ones((n_steps, K, M), dtype=float)

    if config.channel.use_rayleigh:
        sr = config.channel.sigma_r
        x1 = rng_ch.normal(0.0, sr, size=(n_steps, K, M))
        x2 = rng_ch.normal(0.0, sr, size=(n_steps, K, M))
        ray_sq = x1 * x1 + x2 * x2
    else:
        ray_sq = 1.0

    rx_w = pl_base * shad_dyn * ray_sq  # (T, K, M)
    rx_dbm = 10.0 * np.log10(np.maximum(rx_w, 1e-30) * 1000.0)

    ttt_steps = int(round((ho_cfg.ttt_ms / 1000.0) / dt_ho)) if ho_cfg.ttt_ms > 0.0 else 0

    hard_ap = np.zeros((n_steps, K), dtype=int)
    soft_ap = np.zeros((n_steps, K), dtype=int)
    hard_out = np.zeros((n_steps, K), dtype=bool)
    soft_out = np.zeros((n_steps, K), dtype=bool)
    hard_p = np.zeros((n_steps, K), dtype=float)
    soft_p = np.zeros((n_steps, K), dtype=float)
    hard_ho_ue = np.zeros(K, dtype=int)
    soft_ho_ue = np.zeros(K, dtype=int)

    for k in range(K):
        rx_k = rx_w[:, k, :].T  # (M, T)
        init_ap = int(np.argmax(rx_k[:, 0]))
        if M == 1:
            hard_ap[:, k] = 0
            soft_ap[:, k] = 0
            hard_p[:, k] = rx_k[0]
            soft_p[:, k] = rx_k[0]
            hard_out[:, k] = rx_k[0] < p_min_w
            soft_out[:, k] = rx_k[0] < p_min_w
            hard_ho_ue[k] = 0
            soft_ho_ue[k] = 0
        else:
            h_ap, h_o, h_pw, h_cnt = _eval_hard_handover_run(
                rx_k, p_min_w, ho_cfg.hysteresis_db, ttt_steps, init_ap=init_ap
            )
            s_ap, s_o, s_pw, s_cnt = _eval_soft_handover_run(
                rx_k, p_min_w, ho_cfg.active_set_margin_db, ho_cfg.soft_combining
            )
            hard_ap[:, k] = h_ap
            soft_ap[:, k] = s_ap
            hard_out[:, k] = h_o
            soft_out[:, k] = s_o
            hard_p[:, k] = h_pw
            soft_p[:, k] = s_pw
            hard_ho_ue[k] = h_cnt
            soft_ho_ue[k] = s_cnt

    hard_snr_db = 10.0 * np.log10(np.maximum(hard_p / p_noise, 1e-30))
    soft_snr_db = 10.0 * np.log10(np.maximum(soft_p / p_noise, 1e-30))

    return NetworkHandoverResult(
        config=ho_cfg,
        M=M,
        K=K,
        area_m=config.area_m,
        seed=snap_seed,
        p_min_dbm=p_min_dbm,
        ap_xy=ap_xy,
        t_s=t_s,
        ue_x_m=ue_x,
        ue_y_m=ue_y,
        ue_velocities_kmh=ue_velocities_kmh,
        rx_power_dbm=rx_dbm,
        hard_serving_ap=hard_ap,
        soft_serving_ap=soft_ap,
        hard_outage_mask=hard_out,
        soft_outage_mask=soft_out,
        hard_snr_db=hard_snr_db,
        soft_snr_db=soft_snr_db,
        hard_handovers_per_ue=hard_ho_ue,
        soft_handovers_per_ue=soft_ho_ue,
        hard_outage_pct_per_ue=np.mean(hard_out, axis=0) * 100.0,
        soft_outage_pct_per_ue=np.mean(soft_out, axis=0) * 100.0,
        hard_outage_rate_pct=float(np.mean(hard_out) * 100.0),
        soft_outage_rate_pct=float(np.mean(soft_out) * 100.0),
        hard_handovers_total=int(np.sum(hard_ho_ue)),
        soft_handovers_total=int(np.sum(soft_ho_ue)),
        hard_handovers_mean=float(np.mean(hard_ho_ue)),
        soft_handovers_mean=float(np.mean(soft_ho_ue)),
        mean_snr_gain_soft_vs_hard_db=float(np.mean(soft_snr_db - hard_snr_db)),
    )

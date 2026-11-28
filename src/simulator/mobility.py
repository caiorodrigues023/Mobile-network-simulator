# ---------------------------------------------------------------------------
# mobility.py — Motor Time-Driven (dt = 1 ms), Rastreamento Preditivo de Feixes
#               (Hold vs. Preditivo) e SDMA Multi-usuário Móvel (MRC vs. ZF)
# ---------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from .allocation import allocate_channels_vec
from .antennas import calc_ula_gain, mrc_gain, uca_steering, ula_steering
from .channel import gerar_shadowing
from .config import B_TOTAL, D_0, K_0, K_CONST, PATHLOSS_EXP
from .schemas import (
    ChannelConfig,
    MobilityConfig,
    MobilityResult,
    NetworkMobilityResult,
    SDMAResult,
    SimulationConfig,
)

# ---------------------------------------------------------------------------
# Estratégias de Rastreamento de Feixe (Strategy Registry)
# ---------------------------------------------------------------------------


def track_zero_order_hold(
    t: np.ndarray,
    theta_real: np.ndarray,
    delta_t_s: float,
    dt_s: float,
    sigma_theta_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Rastreamento padrão Zero-Order Hold (SciChal4A, Seção 2.1):
      - Atualiza o apontamento a cada ΔT com ruído gaussiano N(0, σ_θ²).
      - Mantém o ângulo congelado entre atualizações (causando efeito dente-de-serra).
    """
    n_steps = len(t)
    step_samples = max(1, int(round(delta_t_s / dt_s)))
    update_indices = np.arange(0, n_steps, step_samples)
    n_upd = len(update_indices)

    err_rad = (
        np.deg2rad(rng.normal(0.0, sigma_theta_deg, size=n_upd))
        if sigma_theta_deg > 0.0
        else np.zeros(n_upd, dtype=float)
    )
    theta_meas = theta_real[update_indices] + err_rad
    block_ids = np.arange(n_steps) // step_samples
    return theta_meas[block_ids]


def track_predictive_alpha_beta(
    t: np.ndarray,
    theta_real: np.ndarray,
    delta_t_s: float,
    dt_s: float,
    sigma_theta_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Rastreamento Preditivo de Feixes (Filtro α-β + Extrapolação Cinemática):
      - A cada instante de medição t_k (a cada ΔT), estima ângulo filtrado θ̂(t_k)
        e velocidade angular θ̇̂(t_k) a partir das medições ruidosas anteriores.
      - Entre t_k e t_{k+1}, extrapola o apontamento a cada passo dt:
            θ_tgt(t) = θ̂(t_k) + θ̇̂(t_k) · (t - t_k)
      - Os ganhos (α, β) são adaptados ao nível de ruído σ_θ e ao período ΔT para
        suprimir ruído de estimação sem introduzir atraso excessivo.
    """
    n_steps = len(t)
    step_samples = max(1, int(round(delta_t_s / dt_s)))
    update_indices = np.arange(0, n_steps, step_samples)
    n_upd = len(update_indices)

    err_rad = (
        np.deg2rad(rng.normal(0.0, sigma_theta_deg, size=n_upd))
        if sigma_theta_deg > 0.0
        else np.zeros(n_upd, dtype=float)
    )
    theta_meas = theta_real[update_indices] + err_rad

    # Seleção adaptativa dos ganhos do filtro α-β:
    # sem ruído: estimação cinemática direta; com ruído: suavização otimizada
    if sigma_theta_deg <= 1e-6:
        alpha, beta = 1.0, 1.0
    elif delta_t_s <= 0.020:
        alpha, beta = 0.45, 0.10
    elif delta_t_s <= 0.060:
        alpha, beta = 0.60, 0.22
    else:
        alpha, beta = 0.72, 0.35

    theta_tgt = np.empty(n_steps, dtype=float)
    dt_eff = step_samples * dt_s

    theta_hat = float(theta_meas[0])
    theta_dot = 0.0

    for k, start_idx in enumerate(update_indices):
        end_idx = min(start_idx + step_samples, n_steps)
        if k == 0:
            theta_hat = float(theta_meas[0])
            theta_dot = 0.0
        elif k == 1:
            # Inicialização da velocidade angular pela primeira diferença
            raw_vel = (float(theta_meas[1]) - theta_hat) / dt_eff
            if sigma_theta_deg <= 1e-6:
                theta_hat = float(theta_meas[1])
                theta_dot = raw_vel
            else:
                theta_pred = theta_hat
                residual = float(theta_meas[1]) - theta_pred
                theta_hat = theta_pred + alpha * residual
                theta_dot = (beta / dt_eff) * residual
        else:
            theta_pred = theta_hat + theta_dot * dt_eff
            residual = float(theta_meas[k]) - theta_pred
            theta_hat = theta_pred + alpha * residual
            theta_dot = theta_dot + (beta / dt_eff) * residual

        # Extrapolação contínua passo a passo dt dentro do intervalo [t_k, t_{k+1})
        local_dt = (np.arange(end_idx - start_idx, dtype=float)) * dt_s
        theta_tgt[start_idx:end_idx] = np.clip(
            theta_hat + theta_dot * local_dt, -np.pi / 2.0, np.pi / 2.0
        )

    return theta_tgt


TRACKING_REGISTRY: dict[
    str,
    Callable[[np.ndarray, np.ndarray, float, float, float, np.random.Generator], np.ndarray],
] = {
    "hold": track_zero_order_hold,
    "predictive": track_predictive_alpha_beta,
}


# ---------------------------------------------------------------------------
# Simulação de 1 Cenário Time-Driven (UE único — Seção 2.2)
# ---------------------------------------------------------------------------


def run_mobility_scenario(
    L: int = 32,
    v_kmh: float = 120.0,
    delta_t_s: float = 0.100,
    D_m: float = 50.0,
    sigma_theta_deg: float = 1.0,
    tracking_mode: str = "hold",
    dt_s: float = 1e-3,
    t_sim_s: float = 4.0,
    t_overhead_s: float = 1.5e-3,
    array_type: str = "ula",
    channel_cfg: Optional[ChannelConfig] = None,
    p_tx_w: float = 1.0,
    gamma_max_db: Optional[float] = None,
    seed: int = 42,
) -> MobilityResult:
    """
    Executa a simulação temporal (dt = 1 ms) de um UE cruzando tangencialmente
    o broadside (+Y) de um AP na origem em distância perpendicular D_m:
        x(t) = v · (t - T_sim / 2),   y(t) = D_m,   θ_real(t) = arctan2(x, D_m)

    O gerador aleatório é reiniciado com `seed` no início de cada cenário para
    assegurar realizações idênticas de erro de estimação entre modos (Methodology Remark).
    """
    tracker = TRACKING_REGISTRY.get(tracking_mode)
    if tracker is None:
        raise ValueError(f"Modo de rastreamento desconhecido: {tracking_mode!r}")

    rng = np.random.default_rng(seed)
    n_steps = max(2, int(round(t_sim_s / dt_s)))
    t_s = np.arange(n_steps, dtype=float) * dt_s

    v_ms = v_kmh / 3.6
    x_m = v_ms * (t_s - t_sim_s / 2.0)
    y_m = np.full_like(x_m, float(D_m))
    dist_m = np.maximum(np.hypot(x_m, y_m), channel_cfg.d_0 if channel_cfg else D_0)
    theta_real = np.arctan2(x_m, y_m)

    theta_tgt = tracker(t_s, theta_real, delta_t_s, dt_s, sigma_theta_deg, rng)

    if array_type == "uca":
        gain_actual_lin = mrc_gain(theta_tgt, theta_real, L=L)
    else:
        gain_actual_lin = calc_ula_gain(theta_real, theta_tgt, L=L)

    gain_actual_lin = np.asarray(gain_actual_lin, dtype=float)
    gain_ideal_db = np.full(n_steps, 10.0 * np.log10(float(L)), dtype=float)
    gain_actual_db = 10.0 * np.log10(np.maximum(gain_actual_lin, 1e-12))
    misalignment_loss_db = np.maximum(gain_ideal_db - gain_actual_db, 0.0)

    k_const = channel_cfg.k_const if channel_cfg else K_CONST
    n_exp = channel_cfg.pathloss_exp if channel_cfg else PATHLOSS_EXP
    b_hz = channel_cfg.b_total_hz if channel_cfg else B_TOTAL
    k_0 = channel_cfg.k_0 if channel_cfg else K_0
    p_noise = b_hz * k_0

    path_gain = k_const / (dist_m**n_exp)
    snr_lin = (p_tx_w * path_gain * gain_actual_lin) / p_noise
    if gamma_max_db is not None:
        snr_cap_lin = 10.0 ** (gamma_max_db / 10.0)
        snr_eff = np.minimum(snr_lin, snr_cap_lin)
    else:
        snr_eff = snr_lin

    snr_db = 10.0 * np.log10(np.maximum(snr_lin, 1e-12))
    se_gross = np.log2(1.0 + snr_eff)
    overhead_fraction = float(min(1.0, t_overhead_s / delta_t_s))
    se_net = (1.0 - overhead_fraction) * se_gross

    return MobilityResult(
        tracking_mode=tracking_mode,
        L=L,
        v_kmh=v_kmh,
        delta_t_s=delta_t_s,
        D_m=D_m,
        sigma_theta_deg=sigma_theta_deg,
        t_s=t_s,
        x_m=x_m,
        y_m=y_m,
        dist_m=dist_m,
        theta_real_rad=theta_real,
        theta_tgt_rad=theta_tgt,
        gain_actual_lin=gain_actual_lin,
        gain_actual_db=gain_actual_db,
        gain_ideal_db=gain_ideal_db,
        misalignment_loss_db=misalignment_loss_db,
        snr_db=snr_db,
        se_gross=se_gross,
        se_net=se_net,
        mean_loss_db=float(np.mean(misalignment_loss_db)),
        max_loss_db=float(np.max(misalignment_loss_db)),
        mean_se_gross=float(np.mean(se_gross)),
        mean_se_net=float(np.mean(se_net)),
        overhead_fraction=overhead_fraction,
    )


def run_mobility_tradeoff_study(
    mob_cfg: MobilityConfig,
    channel_cfg: Optional[ChannelConfig] = None,
    p_tx_w: float = 1.0,
) -> dict[str, Any]:
    """
    Executa a bateria completa da Aba 4:
      - Comparativo Hold vs. Preditivo no cenário principal selecionado;
      - Varredura por ΔT (fixando v e L selecionados);
      - Varredura por L (fixando v e ΔT selecionados);
      - Varredura por Velocidade e por Distância D;
      - Matriz Trade-off ΔT × L × Velocidade para Eficiência Espectral Líquida Média.
    """
    common = dict(
        dt_s=mob_cfg.dt_s,
        t_sim_s=mob_cfg.t_sim_s,
        t_overhead_s=mob_cfg.t_overhead_s,
        sigma_theta_deg=mob_cfg.sigma_theta_deg,
        array_type=mob_cfg.array_type,
        channel_cfg=channel_cfg,
        p_tx_w=p_tx_w,
        seed=mob_cfg.seed,
    )

    main_hold = run_mobility_scenario(
        L=mob_cfg.selected_L,
        v_kmh=mob_cfg.selected_velocity_kmh,
        delta_t_s=mob_cfg.selected_delta_t_s,
        D_m=mob_cfg.selected_distance_m,
        tracking_mode="hold",
        **common,
    )
    main_pred = (
        run_mobility_scenario(
            L=mob_cfg.selected_L,
            v_kmh=mob_cfg.selected_velocity_kmh,
            delta_t_s=mob_cfg.selected_delta_t_s,
            D_m=mob_cfg.selected_distance_m,
            tracking_mode="predictive",
            **common,
        )
        if mob_cfg.compare_predictive
        else None
    )

    by_delta_t = {
        dt_val: run_mobility_scenario(
            L=mob_cfg.selected_L,
            v_kmh=mob_cfg.selected_velocity_kmh,
            delta_t_s=dt_val,
            D_m=mob_cfg.selected_distance_m,
            tracking_mode="hold",
            **common,
        )
        for dt_val in mob_cfg.delta_t_list_s
    }

    by_L = {
        L_val: run_mobility_scenario(
            L=L_val,
            v_kmh=mob_cfg.selected_velocity_kmh,
            delta_t_s=mob_cfg.selected_delta_t_s,
            D_m=mob_cfg.selected_distance_m,
            tracking_mode="hold",
            **common,
        )
        for L_val in mob_cfg.L_list
    }

    by_velocity = {
        v_val: run_mobility_scenario(
            L=mob_cfg.selected_L,
            v_kmh=v_val,
            delta_t_s=mob_cfg.selected_delta_t_s,
            D_m=mob_cfg.selected_distance_m,
            tracking_mode="hold",
            **common,
        )
        for v_val in mob_cfg.velocities_kmh
    }

    by_distance = {
        d_val: run_mobility_scenario(
            L=mob_cfg.selected_L,
            v_kmh=mob_cfg.selected_velocity_kmh,
            delta_t_s=mob_cfg.selected_delta_t_s,
            D_m=d_val,
            tracking_mode="hold",
            **common,
        )
        for d_val in mob_cfg.distances_m
    }

    # Matriz de trade-off: velocidade × L × ΔT (Hold e Preditivo)
    tradeoff_rows: list[dict[str, Any]] = []
    modes = ("hold", "predictive") if mob_cfg.compare_predictive else ("hold",)
    for mode in modes:
        for v_val in mob_cfg.velocities_kmh:
            for L_val in mob_cfg.L_list:
                for dt_val in mob_cfg.delta_t_list_s:
                    r = run_mobility_scenario(
                        L=L_val,
                        v_kmh=v_val,
                        delta_t_s=dt_val,
                        D_m=mob_cfg.selected_distance_m,
                        tracking_mode=mode,
                        **common,
                    )
                    tradeoff_rows.append(
                        {
                            "tracking_mode": mode,
                            "v_kmh": v_val,
                            "L": L_val,
                            "delta_t_ms": round(dt_val * 1000.0, 1),
                            "mean_loss_db": r.mean_loss_db,
                            "max_loss_db": r.max_loss_db,
                            "mean_se_gross": r.mean_se_gross,
                            "mean_se_net": r.mean_se_net,
                            "overhead_pct": r.overhead_fraction * 100.0,
                        }
                    )

    return {
        "main_hold": main_hold,
        "main_predictive": main_pred,
        "by_delta_t": by_delta_t,
        "by_L": by_L,
        "by_velocity": by_velocity,
        "by_distance": by_distance,
        "tradeoff_rows": tradeoff_rows,
    }


# ---------------------------------------------------------------------------
# SDMA Multi-usuário Móvel (MRC vs. Zero-Forcing)
# ---------------------------------------------------------------------------


def run_mobile_sdma(
    mob_cfg: MobilityConfig,
    channel_cfg: Optional[ChannelConfig] = None,
    p_tx_w: float = 1.0,
    tracking_mode: str = "hold",
) -> SDMAResult:
    """
    Estende o motor cinemático time-driven para K >= 1 UEs móveis simultâneos
    no mesmo canal (SDMA) servidos por um AP na origem com arranjo de L antenas.

    Em cada passo dt:
      - Atualiza os ângulos estimados dos K UEs a cada ΔT (com atraso e erro σ_θ);
      - Calcula os vetores de combinação MRC e Zero-Forcing (ZF):
            W_MRC[:, k] = a(θ̂_k) / √L
            W_ZF  = A_hat (A_hat^H A_hat + ε I)^{-1}  (colunas normalizadas ‖w_k‖=1)
      - Extrai a matriz de acoplamento espacial instantânea G_eff(t) = |W^H A_real|² · g_pl(t),
        SINR individual, potência de interferência inter-usuário e Sum-SE(t).
    """
    tracker = TRACKING_REGISTRY.get(tracking_mode)
    if tracker is None:
        raise ValueError(f"Modo de rastreamento desconhecido: {tracking_mode!r}")

    K = int(mob_cfg.sdma_k_ues)
    L = int(mob_cfg.selected_L)
    dt_s = mob_cfg.dt_s
    t_sim_s = mob_cfg.t_sim_s
    delta_t_s = mob_cfg.selected_delta_t_s

    rng = np.random.default_rng(mob_cfg.seed + 777)
    n_steps = max(2, int(round(t_sim_s / dt_s)))
    t_s = np.arange(n_steps, dtype=float) * dt_s

    # Distribui K UEs em faixas tangenciais distintas (direções alternadas e velocidades variadas)
    base_v = mob_cfg.selected_velocity_kmh
    v_factors = np.linspace(0.55, 1.15, K) if K > 1 else np.array([1.0])
    ue_velocities_kmh = base_v * v_factors
    directions = np.array([1.0 if k % 2 == 0 else -1.0 for k in range(K)], dtype=float)

    # Distâncias perpendiculares variadas em torno de selected_distance_m
    d_base = mob_cfg.selected_distance_m
    y_levels = (
        np.linspace(max(20.0, 0.6 * d_base), min(200.0, 1.7 * d_base), K)
        if K > 1
        else np.array([d_base])
    )
    # Deslocamento inicial angular para separar os UEs espacialmente e criar cruzamentos
    t_offsets = np.linspace(-0.28 * t_sim_s, 0.28 * t_sim_s, K) if K > 1 else np.array([0.0])

    ue_x = np.zeros((n_steps, K), dtype=float)
    ue_y = np.zeros((n_steps, K), dtype=float)
    theta_real = np.zeros((n_steps, K), dtype=float)
    theta_tgt = np.zeros((n_steps, K), dtype=float)

    for k in range(K):
        v_ms = (ue_velocities_kmh[k] / 3.6) * directions[k]
        ue_x[:, k] = v_ms * (t_s - t_sim_s / 2.0 + t_offsets[k])
        ue_y[:, k] = y_levels[k]
        theta_real[:, k] = np.arctan2(ue_x[:, k], ue_y[:, k])
        theta_tgt[:, k] = tracker(
            t_s, theta_real[:, k], delta_t_s, dt_s, mob_cfg.sigma_theta_deg, rng
        )

    d_0 = channel_cfg.d_0 if channel_cfg else D_0
    k_const = channel_cfg.k_const if channel_cfg else K_CONST
    n_exp = channel_cfg.pathloss_exp if channel_cfg else PATHLOSS_EXP
    b_hz = channel_cfg.b_total_hz if channel_cfg else B_TOTAL
    k_0 = channel_cfg.k_0 if channel_cfg else K_0
    p_noise = b_hz * k_0

    dist = np.maximum(np.hypot(ue_x, ue_y), d_0)  # (T, K)
    path_gain = k_const / (dist**n_exp)  # (T, K)

    # Vetores de direção instantâneos: A_real (T, K, L) e A_hat (T, K, L)
    if mob_cfg.array_type == "uca":
        A_real = uca_steering(theta_real, L)  # (T, K, L)
        A_hat = uca_steering(theta_tgt, L)  # (T, K, L)
    else:
        A_real = ula_steering(theta_real, L)
        A_hat = ula_steering(theta_tgt, L)

    # Transpõe para (T, L, K) para álgebra matricial em lote (batched)
    A_real_T = np.swapaxes(A_real, 1, 2)  # (T, L, K)
    A_hat_T = np.swapaxes(A_hat, 1, 2)  # (T, L, K)

    # --- Combiner 1: MRC ---
    W_mrc = A_hat_T / np.sqrt(float(L))  # (T, L, K)
    # H_mrc[t, k, i] = w_k^H a_real_i
    H_mrc = np.einsum("tlk,tli->tki", np.conj(W_mrc), A_real_T)
    Gm_mrc = np.abs(H_mrc) ** 2  # (T, K, K)

    # --- Combiner 2: Zero-Forcing (ZF) regularizado ---
    if K == 1:
        W_zf = W_mrc
    else:
        # Gram matrix G_hat = A_hat^H A_hat + eps I -> (T, K, K)
        gram = np.einsum("tlk,tli->tki", np.conj(A_hat_T), A_hat_T)
        reg = 1e-4 * float(L) * np.eye(K, dtype=complex)[None, :, :]
        gram_inv = np.linalg.inv(gram + reg)  # (T, K, K)
        W_zf_raw = np.einsum("tli,tik->tlk", A_hat_T, gram_inv)  # (T, L, K)
        norms = np.linalg.norm(W_zf_raw, axis=1, keepdims=True)
        W_zf = W_zf_raw / np.maximum(norms, 1e-12)

    H_zf = np.einsum("tlk,tli->tki", np.conj(W_zf), A_real_T)
    Gm_zf = np.abs(H_zf) ** 2  # (T, K, K)

    def _eval_combiner(Gm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # G_eff[t, k, i] = path_gain[t, i] * Gm[t, k, i]
        G_eff = path_gain[:, None, :] * Gm  # (T, K, K)
        rx_mat = G_eff * p_tx_w
        diag_idx = np.arange(K)
        useful = rx_mat[:, diag_idx, diag_idx]  # (T, K)
        interf = np.maximum(np.sum(rx_mat, axis=2) - useful, 0.0)
        sinr_lin = useful / (interf + p_noise)
        sinr_db = 10.0 * np.log10(np.maximum(sinr_lin, 1e-12))
        interf_dbm = 10.0 * np.log10(np.maximum(interf, 1e-18) * 1000.0)
        overhead = min(1.0, mob_cfg.t_overhead_s / delta_t_s)
        sum_se = (1.0 - overhead) * np.sum(np.log2(1.0 + sinr_lin), axis=1)
        return sinr_db, interf_dbm, sum_se

    sinr_mrc_db, interf_mrc_dbm, sum_se_mrc = _eval_combiner(Gm_mrc)
    sinr_zf_db, interf_zf_dbm, sum_se_zf = _eval_combiner(Gm_zf)

    return SDMAResult(
        K=K,
        L=L,
        delta_t_s=delta_t_s,
        tracking_mode=tracking_mode,
        t_s=t_s,
        ue_x_m=ue_x,
        ue_y_m=ue_y,
        ue_velocities_kmh=ue_velocities_kmh,
        sinr_mrc_db=sinr_mrc_db,
        sinr_zf_db=sinr_zf_db,
        interf_mrc_dbm=interf_mrc_dbm,
        interf_zf_dbm=interf_zf_dbm,
        sum_se_mrc=sum_se_mrc,
        sum_se_zf=sum_se_zf,
        mean_sum_se_mrc=float(np.mean(sum_se_mrc)),
        mean_sum_se_zf=float(np.mean(sum_se_zf)),
    )


# ---------------------------------------------------------------------------
# Simulação de Mobilidade & Beam Tracking na REDE COMPLETA (M APs × K UEs)
# ---------------------------------------------------------------------------


def _fold_positions_1d(pos: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Reflexão elástica vetorizada em [lo, hi] (onda triangular contínua)."""
    width = hi - lo
    if width <= 0.0:
        return np.full_like(pos, lo)
    u = np.mod(pos - lo, 2.0 * width)
    return lo + np.where(u <= width, u, 2.0 * width - u)


def _wrap_angle_diff(diff_rad: np.ndarray) -> np.ndarray:
    """Normaliza diferença angular para o intervalo [-π, π]."""
    return (diff_rad + np.pi) % (2.0 * np.pi) - np.pi


def _track_azimuth_segmented(
    theta_real: np.ndarray,
    serving_ap: np.ndarray,
    step_samples: int,
    dt_s: float,
    delta_t_s: float,
    sigma_theta_deg: float,
    err_rad: np.ndarray,
    predictive: bool,
) -> np.ndarray:
    """
    Rastreia azimute 2D em [-π, π] de um UE ao longo de `n_steps`, reiniciando
    suavemente quando o UE troca de AP servidor (evita salto artificial de 180°).
    """
    n_steps = len(theta_real)
    theta_tgt = np.empty(n_steps, dtype=float)
    dt_eff = step_samples * dt_s

    if sigma_theta_deg <= 1e-6:
        alpha, beta = 1.0, 1.0
    elif delta_t_s <= 0.020:
        alpha, beta = 0.45, 0.10
    elif delta_t_s <= 0.060:
        alpha, beta = 0.60, 0.22
    else:
        alpha, beta = 0.72, 0.35

    theta_hat = float(theta_real[0] + err_rad[0])
    theta_dot = 0.0
    seg_upd_cnt = 0
    last_upd_step = 0
    upd_idx = 0

    for t in range(n_steps):
        ap_changed = t > 0 and serving_ap[t] != serving_ap[t - 1]
        is_periodic_upd = (t % step_samples) == 0

        if t == 0 or ap_changed:
            err = float(err_rad[min(upd_idx, len(err_rad) - 1)])
            theta_hat = float(theta_real[t]) + err
            theta_dot = 0.0
            seg_upd_cnt = 1
            last_upd_step = t
            if is_periodic_upd:
                upd_idx += 1
        elif is_periodic_upd:
            err = float(err_rad[min(upd_idx, len(err_rad) - 1)])
            upd_idx += 1
            meas = float(theta_real[t]) + err
            if not predictive:
                theta_hat = meas
                theta_dot = 0.0
            else:
                dt_local = max((t - last_upd_step) * dt_s, dt_s)
                if seg_upd_cnt == 1:
                    raw_diff = float(_wrap_angle_diff(np.array(meas - theta_hat)))
                    if sigma_theta_deg <= 1e-6:
                        theta_hat = theta_hat + raw_diff
                        theta_dot = raw_diff / dt_local
                    else:
                        theta_hat = theta_hat + alpha * raw_diff
                        theta_dot = (beta / dt_local) * raw_diff
                else:
                    theta_pred = theta_hat + theta_dot * dt_local
                    residual = float(_wrap_angle_diff(np.array(meas - theta_pred)))
                    theta_hat = theta_pred + alpha * residual
                    theta_dot = theta_dot + (beta / max(dt_eff, dt_local)) * residual
                seg_upd_cnt += 1
            last_upd_step = t

        if predictive:
            theta_tgt[t] = theta_hat + theta_dot * ((t - last_upd_step) * dt_s)
        else:
            theta_tgt[t] = theta_hat

    return theta_tgt


def generate_network_trajectories(
    config: SimulationConfig,
    t_s: np.ndarray,
    seed: Optional[int] = None,
    K_override: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Gera posições iniciais dos M APs e K UEs idênticas ao Snapshot (`seed`)
    e propaga todos os K UEs em 2D dentro de `[margin, area_m - margin]²`
    com reflexão elástica nas bordas.

    Retorna: (ap_xy (M,2), ue_x (T,K), ue_y (T,K), ue_velocities_kmh (K,), shadowing (K,M)).
    """
    from .runner import distribute_aps_xy, sample_ue_xy

    snap_seed = config.seed if seed is None else int(seed)
    K = config.K if K_override is None else int(K_override)
    rng_topo = np.random.default_rng(snap_seed)

    ap_xy = distribute_aps_xy(config.M, config.area_m)
    ue_xy_0 = sample_ue_xy(K, ap_xy, config.area_m, config.channel.d_0, rng=rng_topo)
    shadowing = gerar_shadowing(
        K,
        config.M,
        sigma_x=config.channel.sigma_x,
        use_shadowing=config.channel.use_shadowing,
        rng=rng_topo,
    )

    rng_kin = np.random.default_rng(snap_seed + 404)
    base_v = float(config.mobility.selected_velocity_kmh)
    if config.mobility.mixed_speeds and K > 1:
        profiles = np.asarray(config.mobility.velocities_kmh, dtype=float)
        if profiles.size >= 2:
            ue_velocities_kmh = np.array([profiles[k % profiles.size] for k in range(K)], dtype=float)
        else:
            ue_velocities_kmh = base_v * np.linspace(0.55, 1.25, K)
    else:
        ue_velocities_kmh = np.full(K, base_v, dtype=float)

    headings = rng_kin.uniform(0.0, 2.0 * np.pi, size=K)
    v_ms = ue_velocities_kmh / 3.6
    vx = v_ms * np.cos(headings)  # (K,)
    vy = v_ms * np.sin(headings)  # (K,)

    margin = max(config.channel.d_0 * 2.0, min(15.0, config.area_m * 0.02))
    lo, hi = margin, config.area_m - margin

    raw_x = ue_xy_0[None, :, 0] + t_s[:, None] * vx[None, :]  # (T, K)
    raw_y = ue_xy_0[None, :, 1] + t_s[:, None] * vy[None, :]  # (T, K)

    ue_x = _fold_positions_1d(raw_x, lo, hi)
    ue_y = _fold_positions_1d(raw_y, lo, hi)
    return ap_xy, ue_x, ue_y, ue_velocities_kmh, shadowing


def run_network_mobility(
    config: SimulationConfig,
    seed: Optional[int] = None,
    K_override: Optional[int] = None,
    L_override: Optional[int] = None,
    delta_t_override: Optional[float] = None,
    velocity_override: Optional[float] = None,
    compute_sdma: bool = True,
) -> NetworkMobilityResult:
    """
    Executa a simulação temporal (dt) de mobilidade, Beam Tracking (Hold vs. Preditivo)
    e SDMA Multi-usuário (MRC vs. Zero-Forcing) sobre a REDE COMPLETA (M APs × K UEs).
    """
    mob = config.mobility
    snap_seed = config.seed if seed is None else int(seed)
    K = config.K if K_override is None else int(K_override)
    L = int(L_override if L_override is not None else mob.selected_L)
    delta_t_s = float(delta_t_override if delta_t_override is not None else mob.selected_delta_t_s)
    dt_s = float(mob.dt_s)
    t_sim_s = float(mob.t_sim_s)
    sigma_theta_deg = float(mob.sigma_theta_deg)
    array_type = mob.array_type

    n_steps = max(2, int(round(t_sim_s / dt_s)))
    t_s = np.arange(n_steps, dtype=float) * dt_s
    step_samples = max(1, int(round(delta_t_s / dt_s)))
    update_indices = np.arange(0, n_steps, step_samples)

    # Permite override limpo de velocidade sem mutar config original
    if velocity_override is not None and abs(velocity_override - mob.selected_velocity_kmh) > 1e-9:
        cfg_eff = SimulationConfig.from_dict(config.to_dict())
        cfg_eff.mobility.selected_velocity_kmh = float(velocity_override)
    else:
        cfg_eff = config

    ap_xy, ue_x, ue_y, ue_velocities_kmh, shadowing = generate_network_trajectories(
        cfg_eff, t_s, seed=snap_seed, K_override=K
    )
    M = ap_xy.shape[0]
    d_0 = config.channel.d_0
    k_const = config.channel.k_const
    n_exp = config.channel.pathloss_exp

    # Distâncias (T, K, M) e AP servidor em cada instante
    dx = ue_x[:, :, None] - ap_xy[None, None, :, 0]
    dy = ue_y[:, :, None] - ap_xy[None, None, :, 1]
    dists = np.maximum(np.hypot(dx, dy), d_0)
    pl_all = (k_const / (dists**n_exp)) * shadowing[None, :, :]
    serving_ap = np.argmax(pl_all, axis=2)  # (T, K)

    # Azimute real de cada UE em relação ao seu AP servidor
    t_idx = np.arange(n_steps)[:, None]
    k_idx = np.arange(K)[None, :]
    dx_srv = dx[t_idx, k_idx, serving_ap]
    dy_srv = dy[t_idx, k_idx, serving_ap]
    theta_real = np.arctan2(dy_srv, dx_srv)  # (T, K)

    # Ruído de estimação de AoA idêntico entre Hold e Preditivo (comparação justa)
    rng_err = np.random.default_rng(snap_seed + 909)
    n_upd = len(update_indices) + 2
    err_mat = (
        np.deg2rad(rng_err.normal(0.0, sigma_theta_deg, size=(n_upd, K)))
        if sigma_theta_deg > 0.0
        else np.zeros((n_upd, K), dtype=float)
    )

    theta_hold = np.empty_like(theta_real)
    theta_pred = np.empty_like(theta_real)
    for k in range(K):
        theta_hold[:, k] = _track_azimuth_segmented(
            theta_real[:, k],
            serving_ap[:, k],
            step_samples,
            dt_s,
            delta_t_s,
            sigma_theta_deg,
            err_mat[:, k],
            predictive=False,
        )
        theta_pred[:, k] = _track_azimuth_segmented(
            theta_real[:, k],
            serving_ap[:, k],
            step_samples,
            dt_s,
            delta_t_s,
            sigma_theta_deg,
            err_mat[:, k],
            predictive=True,
        )

    diff_hold = _wrap_angle_diff(theta_real - theta_hold)
    diff_pred = _wrap_angle_diff(theta_real - theta_pred)
    delta_theta_hold_deg = np.abs(np.rad2deg(diff_hold))
    delta_theta_pred_deg = np.abs(np.rad2deg(diff_pred))

    if array_type == "uca":
        g_hold_lin = mrc_gain(theta_hold, theta_real, L=L)
        g_pred_lin = mrc_gain(theta_pred, theta_real, L=L)
    else:
        # ULA orientado localmente ao broadside do setor do UE -> perda exata Eq. 2 vs Δθ
        g_hold_lin = calc_ula_gain(np.clip(diff_hold, -np.pi / 2.0, np.pi / 2.0), 0.0, L=L)
        g_pred_lin = calc_ula_gain(np.clip(diff_pred, -np.pi / 2.0, np.pi / 2.0), 0.0, L=L)

    gain_ideal_db = float(10.0 * np.log10(float(L)))
    gain_hold_db = 10.0 * np.log10(np.maximum(g_hold_lin, 1e-12))
    gain_pred_db = 10.0 * np.log10(np.maximum(g_pred_lin, 1e-12))
    loss_hold_db = np.maximum(gain_ideal_db - gain_hold_db, 0.0)
    loss_pred_db = np.maximum(gain_ideal_db - gain_pred_db, 0.0)

    # Avaliação multi-UE (interferência inter-usuário + SDMA MRC vs. ZF) em grade temporal rápida
    eval_stride = max(1, n_steps // 300)
    eval_idx = np.arange(0, n_steps, eval_stride)
    t_eval_s = t_s[eval_idx]
    Te = len(eval_idx)

    p_tx_w = config.power_control.p_max_w
    p_noise = config.noise_power_w
    overhead_fraction = float(min(1.0, mob.t_overhead_s / delta_t_s))

    # Alocação de canais inicial consistente com a política da simulação
    ch_rng = np.random.default_rng(snap_seed + 17)
    channels, is_agg = allocate_channels_vec(
        serving_ap[0], M=M, N=config.N, policy=config.allocation_policy, rng=ch_rng
    )
    same_ch = (channels[:, None] == channels[None, :]) | is_agg[:, None] | is_agg[None, :]

    def _eval_network_sinr(
        theta_tgt_mat: np.ndarray, use_zf: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sinr_db_out = np.zeros((Te, K), dtype=float)
        interf_dbm_out = np.zeros((Te, K), dtype=float)
        sum_se_out = np.zeros(Te, dtype=float)

        for ti, step_i in enumerate(eval_idx):
            srv = serving_ap[step_i]  # (K,)
            # Ângulos de TODOS os UEs i em relação ao AP servidor de cada UE k: phi_ki = angle(UE_i -> AP_{srv[k]})
            dx_ki = ue_x[step_i, None, :] - ap_xy[srv, 0][:, None]  # (K, K)
            dy_ki = ue_y[step_i, None, :] - ap_xy[srv, 1][:, None]  # (K, K)
            phi_ki = np.arctan2(dy_ki, dx_ki)  # (K, K)
            pl_ki = pl_all[step_i, :, :][:, srv].T  # pl_all[t, i, srv[k]] -> (K, K)

            tgt_k = theta_tgt_mat[step_i]  # (K,)
            if not use_zf:
                if array_type == "uca":
                    g_ant = mrc_gain(tgt_k[:, None], phi_ki, L=L)
                else:
                    ang_err_ki = _wrap_angle_diff(phi_ki - tgt_k[:, None])
                    g_ant = calc_ula_gain(np.clip(ang_err_ki, -np.pi / 2.0, np.pi / 2.0), 0.0, L=L)
                G_eff = pl_ki * g_ant * same_ch
            else:
                # Zero-Forcing espacial por AP servidor (cancela interferentes co-canal)
                G_eff = np.zeros((K, K), dtype=float)
                for m_id in range(M):
                    u_in_ap = np.flatnonzero(srv == m_id)
                    if u_in_ap.size == 0:
                        continue
                    # Vetores de direção reais de todos os K UEs vistos pelo AP m_id
                    phi_m_real = np.arctan2(
                        ue_y[step_i, :] - ap_xy[m_id, 1],
                        ue_x[step_i, :] - ap_xy[m_id, 0],
                    )
                    # Estimativa de direção: usa ângulo rastreado dos UEs do AP e ângulo estimado dos co-canal
                    phi_m_hat = phi_m_real.copy()
                    phi_m_hat[u_in_ap] = tgt_k[u_in_ap]
                    if array_type == "uca":
                        A_real_m = uca_steering(phi_m_real, L).T  # (L, K)
                        A_hat_m = uca_steering(phi_m_hat, L).T  # (L, K)
                    else:
                        # Mapeia azimute em [-pi/2, pi/2] relativo ao UE médio da célula
                        ref_ang = float(phi_m_hat[u_in_ap[0]])
                        A_real_m = ula_steering(
                            np.clip(_wrap_angle_diff(phi_m_real - ref_ang), -np.pi / 2.0, np.pi / 2.0),
                            L,
                        ).T
                        A_hat_m = ula_steering(
                            np.clip(_wrap_angle_diff(phi_m_hat - ref_ang), -np.pi / 2.0, np.pi / 2.0),
                            L,
                        ).T

                    for k_u in u_in_ap:
                        co_ch = np.flatnonzero(same_ch[k_u])
                        if co_ch.size <= 1:
                            w_k = A_hat_m[:, k_u] / np.sqrt(float(L))
                        else:
                            # Mantém o UE desejado + até L-1 interferentes co-canal mais fortes
                            if co_ch.size > L:
                                others = co_ch[co_ch != k_u]
                                top_o = others[np.argsort(pl_all[step_i, others, m_id])[-(L - 1) :]]
                                sel = np.concatenate(([k_u], top_o))
                            else:
                                sel = co_ch
                            A_sub = A_hat_m[:, sel]  # (L, S)
                            S_cnt = A_sub.shape[1]
                            gram = A_sub.conj().T @ A_sub + 1e-4 * float(L) * np.eye(S_cnt)
                            W_sub = A_sub @ np.linalg.inv(gram)
                            pos_k = int(np.flatnonzero(sel == k_u)[0])
                            w_raw = W_sub[:, pos_k]
                            w_k = w_raw / max(float(np.linalg.norm(w_raw)), 1e-12)

                        g_row = np.abs(w_k.conj() @ A_real_m) ** 2  # (K,)
                        G_eff[k_u, :] = pl_all[step_i, :, m_id] * g_row * same_ch[k_u]

            rx_mat = G_eff * p_tx_w
            useful = np.diag(rx_mat)
            interf = np.maximum(np.sum(rx_mat, axis=1) - useful, 0.0)
            sinr_lin = useful / (interf + p_noise)
            sinr_db_out[ti] = 10.0 * np.log10(np.maximum(sinr_lin, 1e-12))
            interf_dbm_out[ti] = 10.0 * np.log10(np.maximum(interf, 1e-18) * 1000.0)
            sum_se_out[ti] = (1.0 - overhead_fraction) * float(np.sum(np.log2(1.0 + sinr_lin)))

        return sinr_db_out, interf_dbm_out, sum_se_out

    sinr_hold_db, interf_mrc_dbm, sum_se_hold = _eval_network_sinr(theta_hold, use_zf=False)
    sinr_pred_db, _, sum_se_pred = _eval_network_sinr(theta_pred, use_zf=False)

    if compute_sdma:
        sinr_mrc_db = sinr_hold_db
        sum_se_mrc = sum_se_hold
        sinr_zf_db, interf_zf_dbm, sum_se_zf = _eval_network_sinr(theta_hold, use_zf=True)
    else:
        sinr_mrc_db = sinr_hold_db
        sum_se_mrc = sum_se_hold
        sinr_zf_db = sinr_hold_db
        interf_zf_dbm = interf_mrc_dbm
        sum_se_zf = sum_se_hold

    return NetworkMobilityResult(
        M=M,
        K=K,
        L=L,
        area_m=config.area_m,
        array_type=array_type,
        delta_t_s=delta_t_s,
        dt_s=dt_s,
        t_overhead_s=mob.t_overhead_s,
        sigma_theta_deg=sigma_theta_deg,
        seed=snap_seed,
        ap_xy=ap_xy,
        t_s=t_s,
        ue_x_m=ue_x,
        ue_y_m=ue_y,
        ue_velocities_kmh=ue_velocities_kmh,
        serving_ap=serving_ap,
        update_indices=update_indices,
        beam_angles_final_rad=theta_hold[-1],
        delta_theta_hold_deg=delta_theta_hold_deg,
        delta_theta_pred_deg=delta_theta_pred_deg,
        gain_ideal_db=gain_ideal_db,
        gain_hold_db=gain_hold_db,
        gain_pred_db=gain_pred_db,
        loss_hold_db=loss_hold_db,
        loss_pred_db=loss_pred_db,
        t_eval_s=t_eval_s,
        sinr_hold_db=sinr_hold_db,
        sinr_pred_db=sinr_pred_db,
        sum_se_hold=sum_se_hold,
        sum_se_pred=sum_se_pred,
        sinr_mrc_db=sinr_mrc_db,
        sinr_zf_db=sinr_zf_db,
        interf_mrc_dbm=interf_mrc_dbm,
        interf_zf_dbm=interf_zf_dbm,
        sum_se_mrc=sum_se_mrc,
        sum_se_zf=sum_se_zf,
        overhead_fraction=overhead_fraction,
        mean_delta_theta_hold_deg=float(np.mean(delta_theta_hold_deg)),
        max_delta_theta_hold_deg=float(np.max(delta_theta_hold_deg)),
        mean_delta_theta_pred_deg=float(np.mean(delta_theta_pred_deg)),
        mean_loss_hold_db=float(np.mean(loss_hold_db)),
        max_loss_hold_db=float(np.max(loss_hold_db)),
        mean_loss_pred_db=float(np.mean(loss_pred_db)),
        max_loss_pred_db=float(np.max(loss_pred_db)),
        mean_sum_se_hold=float(np.mean(sum_se_hold)),
        mean_sum_se_pred=float(np.mean(sum_se_pred)),
        mean_sum_se_mrc=float(np.mean(sum_se_mrc)),
        mean_sum_se_zf=float(np.mean(sum_se_zf)),
    )


def run_network_mobility_tradeoff(
    config: SimulationConfig,
    seed: Optional[int] = None,
    K_override: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Avalia o trade-off ΔT × L (Hold e Preditivo) para a REDE COMPLETA (M APs × K UEs).
    """
    mob = config.mobility
    rows: list[dict[str, Any]] = []
    for L_val in mob.L_list:
        for dt_val in mob.delta_t_list_s:
            res = run_network_mobility(
                config,
                seed=seed,
                K_override=K_override,
                L_override=int(L_val),
                delta_t_override=float(dt_val),
                compute_sdma=False,
            )
            rows.append(
                {
                    "tracking_mode": "hold",
                    "v_kmh": mob.selected_velocity_kmh,
                    "L": int(L_val),
                    "delta_t_ms": round(dt_val * 1000.0, 1),
                    "mean_loss_db": res.mean_loss_hold_db,
                    "max_loss_db": res.max_loss_hold_db,
                    "mean_se_net": res.mean_sum_se_hold,
                    "overhead_pct": res.overhead_fraction * 100.0,
                }
            )
            if mob.compare_predictive:
                rows.append(
                    {
                        "tracking_mode": "predictive",
                        "v_kmh": mob.selected_velocity_kmh,
                        "L": int(L_val),
                        "delta_t_ms": round(dt_val * 1000.0, 1),
                        "mean_loss_db": res.mean_loss_pred_db,
                        "max_loss_db": res.max_loss_pred_db,
                        "mean_se_net": res.mean_sum_se_pred,
                        "overhead_pct": res.overhead_fraction * 100.0,
                    }
                )
    return rows

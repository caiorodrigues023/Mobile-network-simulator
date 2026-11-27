# ---------------------------------------------------------------------------
# mobility.py — Motor Time-Driven (dt = 1 ms), Rastreamento Preditivo de Feixes
#               (Hold vs. Preditivo) e SDMA Multi-usuário Móvel (MRC vs. ZF)
# ---------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from .antennas import calc_ula_gain, mrc_gain, uca_steering, ula_steering
from .config import B_TOTAL, D_0, K_0, K_CONST, PATHLOSS_EXP
from .schemas import ChannelConfig, MobilityConfig, MobilityResult, SDMAResult

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

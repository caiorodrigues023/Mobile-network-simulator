# ---------------------------------------------------------------------------
# handover.py — Motor de Handover (Hard vs. Soft) e Detecção de Outage
#               Baseado na Parte 2 (Seção 8, Exercícios 12 e 13) com suporte
#               a múltiplos APs, margem de histerese, TTT e conjunto ativo
# ---------------------------------------------------------------------------

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .config import B_TOTAL, D_0, K_0, K_CONST, PATHLOSS_EXP, SIGMA_X
from .schemas import ChannelConfig, HandoverConfig, HandoverResult


def _eval_hard_handover_run(
    rx_power_w: np.ndarray,
    p_min_w: float,
    hysteresis_db: float,
    ttt_steps: int,
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

    curr_ap = 0
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

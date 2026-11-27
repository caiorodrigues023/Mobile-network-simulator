# ---------------------------------------------------------------------------
# metrics.py — SINR vetorizado, Shannon, limites QAM/Outage e KPIs consolidados
# ---------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from .config import B_TOTAL, GAMMA_MAX_DB, GAMMA_MIN_DB, K_0

# Compatibilidade com constantes antigas
_BT: float = B_TOTAL
_K0: float = K_0


# ---------------------------------------------------------------------------
# Núcleo Vetorizado (Data-Oriented Design)
# ---------------------------------------------------------------------------


def calculate_sinr_vector(
    G_eff: np.ndarray, p_watts: np.ndarray, p_noise: float
) -> np.ndarray:
    """
    Calcula o vetor linear de SINR para todos os K UEs simultaneamente via BLAS:
        rx_total = G_eff @ p_watts
        SINR_k = (G_eff[k,k] * p_k) / (rx_total[k] - G_eff[k,k]*p_k + p_noise)
    UEs desligados (p_k <= 0) recebem SINR = 0.0.
    """
    p_w = np.asarray(p_watts, dtype=float)
    if p_w.size == 0:
        return np.zeros(0, dtype=float)
    total_rx_power = G_eff @ p_w
    useful_signal = np.diagonal(G_eff) * p_w
    interference = np.maximum(total_rx_power - useful_signal, 0.0)
    return np.where(p_w > 0.0, useful_signal / (interference + p_noise), 0.0)


def evaluate_physical_kpis(
    sinr_lin: np.ndarray,
    p_watts: np.ndarray,
    bw_channel: float,
    gamma_min_db: float = GAMMA_MIN_DB,
    gamma_max_db: float = GAMMA_MAX_DB,
    apply_physical_limits: bool = True,
) -> tuple[np.ndarray, float, float, float, np.ndarray]:
    """
    Avalia KPIs por UE e agregados do snapshot.

    Quando `apply_physical_limits=True`:
      - UEs bloqueados (p <= 0) ou com SINR < gamma_min_db entram em Outage (SE = 0).
      - SINR acima de gamma_max_db (ex.: +20 dB) é saturada por limite de constelação QAM.
    Quando `apply_physical_limits=False`:
      - Capacidade de Shannon irrestrita log2(1 + SINR) para UEs ativos (outage ainda é reportado
        como indicador diagnóstico para SINR < gamma_min_db ou p <= 0).

    Retorna: (se_per_ue, sum_se, sum_cap_bps, total_power_w, outage_mask).
    """
    gamma_min_lin = 10.0 ** (gamma_min_db / 10.0)
    gamma_max_lin = 10.0 ** (gamma_max_db / 10.0)

    p_w = np.asarray(p_watts, dtype=float)
    s_lin = np.asarray(sinr_lin, dtype=float)

    outage_mask = (p_w <= 0.0) | (s_lin < gamma_min_lin)

    if apply_physical_limits:
        sinr_eff = np.where(outage_mask, 0.0, np.minimum(s_lin, gamma_max_lin))
    else:
        sinr_eff = np.where(p_w <= 0.0, 0.0, np.maximum(s_lin, 0.0))

    se_per_ue = np.log2(1.0 + sinr_eff)
    sum_se = float(np.sum(se_per_ue))
    sum_cap_bps = float(bw_channel * sum_se)
    total_power_w = float(np.sum(p_w))
    return se_per_ue, sum_se, sum_cap_bps, total_power_w, outage_mask


def jain_fairness(values: np.ndarray) -> float:
    """Índice de Justiça de Jain: (Σx)² / (K · Σx²) ∈ [0, 1]."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    s1 = float(np.sum(arr))
    s2 = float(np.sum(arr * arr))
    if s2 <= 0.0:
        return 0.0
    return (s1 * s1) / (arr.size * s2)


def area_spectrum_efficiency(
    avg_sum_cap_bps: float, b_total_hz: float, area_m2: float
) -> float:
    """Eficiência Espectral de Área (ASE) em [bps/Hz/km²] (Parte 2, Seção 2.4)."""
    if b_total_hz <= 0.0 or area_m2 <= 0.0:
        return 0.0
    area_km2 = area_m2 / 1e6
    return float(avg_sum_cap_bps / (b_total_hz * area_km2))


# ---------------------------------------------------------------------------
# Wrappers Legados (compatibilidade com main.py / scripts OO)
# ---------------------------------------------------------------------------


def SINR(
    ues: list,
    N: int,
    gains: np.ndarray,
    G: np.ndarray | None = None,
    b_total: float = _BT,
    k0: float = _K0,
) -> list[float]:
    """Calcula a SINR para antenas convencionais (Omni, Setorial, SMB) a partir de objetos UE."""
    _ = G
    K = len(ues)
    if K == 0:
        return []
    pn = k0 * b_total / N
    P = np.array([ue.power for ue in ues], dtype=float)
    A = np.array([ue.ap.id for ue in ues], dtype=int)
    B = np.array([ue.beam for ue in ues], dtype=int)
    C = np.array([ue.channel for ue in ues])

    G_raw = gains[:, A, B].T  # (K, K): G_raw[k, i] = gains[i, A[k], B[k]]
    same_ch = C[:, None] == C[None, :]
    G_eff = np.where(same_ch, G_raw, 0.0)
    return calculate_sinr_vector(G_eff, P, pn).tolist()


def SINR_adaptive(
    ues: list,
    N: int,
    gains: np.ndarray,
    Gm: np.ndarray,
    b_total: float = _BT,
    k0: float = _K0,
) -> list[float]:
    """Calcula a SINR com beamforming adaptativo MRC a partir de objetos UE."""
    K = len(ues)
    if K == 0:
        return []
    pn = k0 * b_total / N
    P = np.array([ue.power for ue in ues], dtype=float)
    A = np.array([ue.ap.id for ue in ues], dtype=int)
    C = np.array([ue.channel for ue in ues])

    G_raw = gains[:, A, 0].T * Gm
    same_ch = C[:, None] == C[None, :]
    G_eff = np.where(same_ch, G_raw, 0.0)
    return calculate_sinr_vector(G_eff, P, pn).tolist()


def channel_capacity(
    sinr_valores, N: int, b_total: float = _BT
) -> np.ndarray:
    """Equação de Shannon-Hartley: C = (B_total / N) · log₂(1 + SINR) em Mbps."""
    B_per_channel = b_total / N
    sinr_array = np.maximum(np.asarray(sinr_valores, dtype=float), 0.0)
    return B_per_channel * np.log2(1.0 + sinr_array) / 1e6

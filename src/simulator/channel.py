# ---------------------------------------------------------------------------
# channel.py — Modelagem de propagação vetorizada: Geometria, Path Loss,
#              Shadowing log-normal e Fast Fading (Rayleigh)
# ---------------------------------------------------------------------------

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import D_0, K_CONST, PATHLOSS_EXP, SIGMA_R, SIGMA_X
from .schemas import ChannelConfig


def _to_coords(arr) -> np.ndarray:
    """Converte lista de objetos (UE, AP) ou array de coordenadas para matriz (N, 2)."""
    if len(arr) == 0:
        return np.zeros((0, 2), dtype=float)
    if hasattr(arr[0], "x") and hasattr(arr[0], "y"):
        return np.array([[obj.x, obj.y] for obj in arr], dtype=float)
    arr_np = np.asarray(arr, dtype=float)
    if arr_np.ndim == 1:
        return arr_np.reshape(-1, 2)
    return arr_np


# ---------------------------------------------------------------------------
# Geometria espacial e angular
# ---------------------------------------------------------------------------


def distance_matrix(ue_xy, ap_xy, d_0: float = D_0) -> np.ndarray:
    """Retorna matriz (K, M) de distâncias Euclidianas com piso d_0."""
    ue_coords = _to_coords(ue_xy)
    ap_coords = _to_coords(ap_xy)
    dx = ue_coords[:, None, 0] - ap_coords[None, :, 0]
    dy = ue_coords[:, None, 1] - ap_coords[None, :, 1]
    return np.maximum(np.hypot(dx, dy), d_0)


def angle_matrix(ue_xy, ap_xy) -> np.ndarray:
    """
    Calcula matriz (K, M) de ângulos de azimute em [0, 2π) de cada UE em relação a cada AP.
    Aceita matrizes (N, 2) ou listas legadas de objetos UE/AP.
    """
    ue_coords = _to_coords(ue_xy)
    ap_coords = _to_coords(ap_xy)
    dx = ue_coords[:, None, 0] - ap_coords[None, :, 0]
    dy = ue_coords[:, None, 1] - ap_coords[None, :, 1]
    return np.mod(np.arctan2(dy, dx), 2.0 * np.pi)


# ---------------------------------------------------------------------------
# Shadowing log-normal e Fast Fading (Rayleigh)
# ---------------------------------------------------------------------------


def gerar_shadowing(
    K,
    M,
    sigma_x: float = SIGMA_X,
    use_shadowing: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sorteia shadowing log-normal exp(N(0, sigma_x)) independente por link (K, M).
    Se use_shadowing=False ou sigma_x <= 0, retorna matriz de 1.0.
    """
    n_k = len(K) if hasattr(K, "__len__") else int(K)
    n_m = len(M) if hasattr(M, "__len__") else int(M)
    if not use_shadowing or sigma_x <= 0.0:
        return np.ones((n_k, n_m), dtype=float)
    if rng is not None:
        return rng.lognormal(0.0, sigma_x, size=(n_k, n_m))
    return np.random.lognormal(0.0, sigma_x, size=(n_k, n_m))


def gerar_rayleigh(
    K,
    M,
    sigma_r: float = SIGMA_R,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sorteia amplitude de fading Rayleigh sqrt(X1² + X2²) independente por link (K, M).
    Com sigma_r = 1/sqrt(2), E[r²] = 1.0 (potência média unitária).
    """
    n_k = len(K) if hasattr(K, "__len__") else int(K)
    n_m = len(M) if hasattr(M, "__len__") else int(M)
    if rng is not None:
        x1 = rng.normal(0.0, sigma_r, size=(n_k, n_m))
        x2 = rng.normal(0.0, sigma_r, size=(n_k, n_m))
    else:
        x1 = np.random.normal(0.0, sigma_r, size=(n_k, n_m))
        x2 = np.random.normal(0.0, sigma_r, size=(n_k, n_m))
    return np.hypot(x1, x2)


# ---------------------------------------------------------------------------
# Matriz de ganho de canal (K, M, B)
# ---------------------------------------------------------------------------


def gain_matrix(
    ue_xy,
    ap_xy,
    G: np.ndarray,
    shadowing: Optional[np.ndarray] = None,
    use_rayleigh: bool = False,
    use_shadowing: bool = True,
    rayleigh: Optional[np.ndarray] = None,
    pathloss_exp: float = PATHLOSS_EXP,
    k_const: float = K_CONST,
    d_0: float = D_0,
    sigma_x: float = SIGMA_X,
    sigma_r: float = SIGMA_R,
    channel_cfg: Optional[ChannelConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Calcula a matriz de ganho de canal (K, M, B):
        g[k, m, b] = shadowing[k, m] * rayleigh[k, m]² * (k_const / d[k, m]^n) * G[k, m, b]

    Aceita `channel_cfg: ChannelConfig` ou parâmetros individuais.
    Permite passar `shadowing` e `rayleigh` pré-sorteados para garantir comparação
    justa sob realizações de canal idênticas entre modelos/algoritmos.
    """
    if channel_cfg is not None:
        use_shadowing = channel_cfg.use_shadowing
        sigma_x = channel_cfg.sigma_x
        use_rayleigh = channel_cfg.use_rayleigh
        sigma_r = channel_cfg.sigma_r
        pathloss_exp = channel_cfg.pathloss_exp
        k_const = channel_cfg.k_const
        d_0 = channel_cfg.d_0

    ue_coords = _to_coords(ue_xy)
    ap_coords = _to_coords(ap_xy)
    K, M = ue_coords.shape[0], ap_coords.shape[0]

    if shadowing is None:
        shadowing_arr = gerar_shadowing(
            K, M, sigma_x=sigma_x, use_shadowing=use_shadowing, rng=rng
        )
    else:
        shadowing_arr = np.asarray(shadowing, dtype=float)
    if shadowing_arr.ndim == 2:
        shadowing_arr = shadowing_arr[:, :, None]

    if rayleigh is not None:
        rayleigh_sq = np.asarray(rayleigh, dtype=float) ** 2
        if rayleigh_sq.ndim == 2:
            rayleigh_sq = rayleigh_sq[:, :, None]
    elif use_rayleigh:
        rayleigh_sq = gerar_rayleigh(K, M, sigma_r=sigma_r, rng=rng)[:, :, None] ** 2
    else:
        rayleigh_sq = 1.0

    d = distance_matrix(ue_coords, ap_coords, d_0=d_0)
    return shadowing_arr * rayleigh_sq * (k_const / (d[:, :, None] ** pathloss_exp)) * G


# ---------------------------------------------------------------------------
# Beamforming adaptativo (MRC UCA & ULA) — vetorizado + wrapper legado
# ---------------------------------------------------------------------------


def adaptive_gains_vec(
    phi: np.ndarray,
    serving_ap: np.ndarray,
    L: int = 8,
    aoa_err_deg: float = 0.0,
    array_type: str = "uca",
    rng: Optional[np.random.Generator] = None,
    aoa_noise_rad: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula de forma 100% vetorizada a matriz de ganhos espaciais Gm (K, K)
    e os ângulos efetivos de apontamento phi_hat (K,) para feixes adaptativos.

    Gm[k, i] = ganho do feixe do UE k (no AP serving_ap[k]) na direção real do UE i.
    """
    from .antennas import mrc_gain, ula_2d_gain

    K = phi.shape[0]
    if K == 0:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float)

    theta_true = phi[np.arange(K), serving_ap]
    if aoa_noise_rad is not None:
        phi_hat = theta_true + np.asarray(aoa_noise_rad, dtype=float)
    elif aoa_err_deg > 0.0:
        noise = rng.normal(0.0, 1.0, size=K) if rng is not None else np.random.randn(K)
        phi_hat = theta_true + np.deg2rad(aoa_err_deg) * noise
    else:
        phi_hat = theta_true.copy()

    # phi_dir[k, i] = azimute do UE i visto pelo AP servidor do UE k
    phi_dir = phi[:, serving_ap].T
    if array_type == "ula":
        Gm = ula_2d_gain(phi_hat[:, None], phi_dir, L=L)
    else:
        Gm = mrc_gain(phi_hat[:, None], phi_dir, L=L)
    return Gm, phi_hat


def adaptive_gains(
    ues: list,
    aps: list,
    L: int = 8,
    aoa_err_deg: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Wrapper compatível com objetos OO (UE/AP) para Beamforming adaptativo MRC (UCA).
    Retorna Gm (K, K).
    """
    ue_xy = _to_coords(ues)
    ap_xy = _to_coords(aps)
    phi = angle_matrix(ue_xy, ap_xy)
    serving_ap = np.array([ue.ap.id for ue in ues], dtype=int)
    Gm, _ = adaptive_gains_vec(phi, serving_ap, L=L, aoa_err_deg=aoa_err_deg, rng=rng)
    return Gm

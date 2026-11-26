# ---------------------------------------------------------------------------
# channel.py — Modelagem de propagação: Path Loss, Shadowing e Geometria
# ---------------------------------------------------------------------------

from typing import Optional

import numpy as np

from .config import D_0, K_CONST, PATHLOSS_EXP, SIGMA_R, SIGMA_X


def _to_coords(arr) -> np.ndarray:
    """Converte lista de objetos (UE, AP) ou array de coordenadas para matriz (N, 2)."""
    if len(arr) == 0:
        return np.zeros((0, 2), dtype=float)
    if hasattr(arr[0], "x") and hasattr(arr[0], "y"):
        return np.array([[obj.x, obj.y] for obj in arr], dtype=float)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 2)
    return arr


# ---------------------------------------------------------------------------
# Geometria angular
# ---------------------------------------------------------------------------
def angle_matrix(ue_xy, ap_xy) -> np.ndarray:
    """
    ue_xy: matriz (K, 2) ou lista de UEs.
    ap_xy: matriz (M, 2) ou lista de APs.
    Retorna matriz (K, M) de ângulos de azimute.
    """
    ue_coords = _to_coords(ue_xy)
    ap_coords = _to_coords(ap_xy)
    dx = ue_coords[:, None, 0] - ap_coords[None, :, 0]
    dy = ue_coords[:, None, 1] - ap_coords[None, :, 1]
    return np.mod(np.arctan2(dy, dx), 2 * np.pi)


# ---------------------------------------------------------------------------
# Shadowing log-normal e Fast Fading (Rayleigh)
# ---------------------------------------------------------------------------


def gerar_shadowing(K, M, sigma_x: float = SIGMA_X) -> np.ndarray:
    """
    Sorteia o shadowing log-normal de forma independente para cada link (UE, AP).
    Aceita inteiros K, M ou listas de entidades.
    Se sigma_x <= 0, retorna matriz de 1.0 (sem sombreamento).
    """
    n_k = len(K) if hasattr(K, "__len__") else int(K)
    n_m = len(M) if hasattr(M, "__len__") else int(M)
    if sigma_x <= 0.0:
        return np.ones((n_k, n_m), dtype=float)
    return np.random.lognormal(0, sigma_x, size=(n_k, n_m))


def gerar_rayleigh(K, M, sigma_r: float = SIGMA_R) -> np.ndarray:
    """
    Sorteia o fading de Rayleigh de forma independente para cada link (UE, AP).
    Aceita inteiros K, M ou listas de entidades.
    """
    n_k = len(K) if hasattr(K, "__len__") else int(K)
    n_m = len(M) if hasattr(M, "__len__") else int(M)
    x1 = np.random.normal(0, sigma_r, size=(n_k, n_m))
    x2 = np.random.normal(0, sigma_r, size=(n_k, n_m))
    return np.sqrt(x1**2 + x2**2)


# ---------------------------------------------------------------------------
# Matriz de ganho de canal
# ---------------------------------------------------------------------------


def gain_matrix(
    ue_xy,
    ap_xy,
    G: np.ndarray,
    shadowing: Optional[np.ndarray] = None,  # noqa
    use_rayleigh: bool = False,
    pathloss_exp: float = PATHLOSS_EXP,
    k_const: float = K_CONST,
    d_0: float = D_0,
    sigma_x: float = SIGMA_X,
    sigma_r: float = SIGMA_R,
) -> np.ndarray:
    """
    Calcula a matriz de ganho de canal combinando path loss (d⁻ⁿ),
    shadowing log-normal, fast fading opcional e ganho de antena G.
    Aceita matrizes de coordenadas ou listas de UEs/APs.
    """
    ue_coords = _to_coords(ue_xy)
    ap_coords = _to_coords(ap_xy)
    K, M = ue_coords.shape[0], ap_coords.shape[0]

    if shadowing is None:
        shadowing = gerar_shadowing(K, M, sigma_x=sigma_x)
    if shadowing.ndim == 2:
        shadowing = shadowing[:, :, None]

    rayleigh = 1.0
    if use_rayleigh:
        rayleigh = gerar_rayleigh(K, M, sigma_r=sigma_r)[:, :, None]

    # Distância Euclidiana (K, M)
    dx = ue_coords[:, None, 0] - ap_coords[None, :, 0]
    dy = ue_coords[:, None, 1] - ap_coords[None, :, 1]
    d = np.maximum(np.sqrt(dx**2 + dy**2), d_0)

    return shadowing * (rayleigh**2) * k_const / (d[:, :, None] ** pathloss_exp) * G


# ---------------------------------------------------------------------------
# Beamforming adaptativo MRC — cálculo dos ganhos por link
# ---------------------------------------------------------------------------


def adaptive_gains(
    ues: list, aps: list, L: int = 8, aoa_err_deg: float = 0.0
) -> np.ndarray:
    """
    Beamforming adaptativo MRC (Exercício 3; com aoa_err_deg > 0, Exercício 4).
    Requer o attachment já feito (ue.ap definido).

    Para cada UE k servido pelo AP m_k, forma um feixe MRC apontado para o
    AoA estimado:  φ̂_k = φ[k, m_k] + ε,  ε ~ N(0, aoa_err²).

    Retorna Gm (n_ues, n_ues) com:
        Gm[k, k]      = ganho no UE desejado   (= L se aoa_err_deg = 0)
        Gm[k, i≠k]   = ganho aplicado à interferência do UE i sobre o UE k
    """
    from .antennas import mrc_gain  # importação local para evitar ciclo

    n = len(ues)

    ue_xy = np.array([[ue.x, ue.y] for ue in ues], dtype=float)
    ap_xy = np.array([[ap.x, ap.y] for ap in aps], dtype=float)
    phi = angle_matrix(ue_xy, ap_xy)

    m = np.array([ue.ap.id for ue in ues])
    phi_hat = phi[np.arange(n), m] + np.deg2rad(aoa_err_deg) * np.random.randn(n)
    phi_dir = phi[:, m].T
    return mrc_gain(phi_hat[:, None], phi_dir, L)

# ---------------------------------------------------------------------------
# channel.py — Modelagem de propagação: Path Loss, Shadowing e Geometria
# ---------------------------------------------------------------------------

from typing import Optional

import numpy as np

from .config import D_0, K_CONST, PATHLOSS_EXP, SIGMA_R, SIGMA_X


# ---------------------------------------------------------------------------
# Geometria angular
# ---------------------------------------------------------------------------
def angle_matrix(ue_xy: np.ndarray, ap_xy: np.ndarray) -> np.ndarray:
    """
    ue_xy: matriz (K, 2) com coordenadas [x, y] dos UEs.
    ap_xy: matriz (M, 2) com coordenadas [x, y] dos APs.
    Retorna matriz (K, M) de ângulos de azimute.
    """
    dx = ue_xy[:, None, 0] - ap_xy[None, :, 0]
    dy = ue_xy[:, None, 1] - ap_xy[None, :, 1]
    return np.mod(np.arctan2(dy, dx), 2 * np.pi)


# ---------------------------------------------------------------------------
# Shadowing log-normal e Fast Fading (Rayleigh)
# ---------------------------------------------------------------------------


def gerar_shadowing(K: int, M: int) -> np.ndarray:
    """
    Sorteia o shadowing log-normal de forma independente para cada link (UE, AP).
    Variável aleatória log-normal(μ=0, σ=2).
    Retorna: matriz (n_ues, n_aps).
    """
    return np.random.lognormal(0, SIGMA_X, size=(K, M))


def gerar_rayleigh(K: int, M: int) -> np.ndarray:
    """
    Sorteia o fading de Rayleigh de forma independente para cada link (UE, AP).
    Variável construída a partir de duas normais (Parte 3 do projeto).
    Retorna: matriz (n_ues, n_aps).
    """
    x1 = np.random.normal(0, SIGMA_R, size=(K, M))
    x2 = np.random.normal(0, SIGMA_R, size=(K, M))
    return np.sqrt(x1**2 + x2**2)


# ---------------------------------------------------------------------------
# Matriz de ganho de canal
# ---------------------------------------------------------------------------


def gain_matrix(
    ue_xy: np.ndarray,
    ap_xy: np.ndarray,
    G: np.ndarray,
    shadowing: Optional[np.ndarray] = None,  # noqa
    use_rayleigh: bool = False,
) -> np.ndarray:
    """
    Calcula a matriz de ganho de canal combinando path loss (d⁻⁴),
    shadowing log-normal, fast fading opcional e ganho de antena G.
    """
    K, M = ue_xy.shape[0], ap_xy.shape[0]

    if shadowing is None:
        shadowing = gerar_shadowing(K, M)
    if shadowing.ndim == 2:
        shadowing = shadowing[:, :, None]

    rayleigh = 1.0
    if use_rayleigh:
        rayleigh = gerar_rayleigh(K, M)[:, :, None]

    # Distância Euclidiana (K, M)
    dx = ue_xy[:, None, 0] - ap_xy[None, :, 0]
    dy = ue_xy[:, None, 1] - ap_xy[None, :, 1]
    d = np.maximum(np.sqrt(dx**2 + dy**2), D_0)

    return shadowing * (rayleigh**2) * K_CONST / (d[:, :, None] ** PATHLOSS_EXP) * G


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

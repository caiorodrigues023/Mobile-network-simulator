# ---------------------------------------------------------------------------
# channel.py — Modelagem de propagação: Path Loss, Shadowing e Geometria
# ---------------------------------------------------------------------------

import numpy as np

# ---------------------------------------------------------------------------
# Geometria angular
# ---------------------------------------------------------------------------


def angle(ue, ap) -> float:
    """
    Ângulo de azimute (0 a 2π) do UE em relação ao AP, no sistema de
    coordenadas cartesianas (x positivo = 0 rad).
    """
    delta_x = ue.x - ap.x
    delta_y = ue.y - ap.y
    theta = np.arctan2(delta_y, delta_x)
    if theta < 0:
        theta += 2 * np.pi
    return float(theta)


def angle_matrix(ues: list, aps: list) -> np.ndarray:
    """
    Ângulo (0 a 2π) de cada UE i em relação a cada AP j.
    Retorna matriz (n_ues, n_aps).
    """
    ue_xy = np.array([[ue.x, ue.y] for ue in ues], dtype=float)
    ap_xy = np.array([[ap.x, ap.y] for ap in aps], dtype=float)
    dx = ue_xy[:, None, 0] - ap_xy[None, :, 0]
    dy = ue_xy[:, None, 1] - ap_xy[None, :, 1]
    return np.mod(np.arctan2(dy, dx), 2 * np.pi)


# ---------------------------------------------------------------------------
# Shadowing log-normal e Fast Fading (Rayleigh)
# ---------------------------------------------------------------------------


def gerar_shadowing(ues: list, aps: list) -> np.ndarray:
    """
    Sorteia o shadowing log-normal de forma independente para cada link (UE, AP).
    Variável aleatória log-normal(μ=0, σ=2).
    Retorna: matriz (n_ues, n_aps).
    """
    return np.random.lognormal(0, 2, size=(len(ues), len(aps)))


def gerar_rayleigh(ues: list, aps: list, sigma_r: float = 0.70710678) -> np.ndarray:
    """
    Sorteia o fading de Rayleigh de forma independente para cada link (UE, AP).
    Variável construída a partir de duas normais (Parte 3 do projeto).
    Retorna: matriz (n_ues, n_aps).
    """
    shape = (len(ues), len(aps))
    x1 = np.random.normal(0, sigma_r, size=shape)
    x2 = np.random.normal(0, sigma_r, size=shape)
    return np.sqrt(x1**2 + x2**2)


# ---------------------------------------------------------------------------
# Matriz de ganho de canal
# ---------------------------------------------------------------------------


def gain_matrix(
    ues: list,
    aps: list,
    G: np.ndarray,
    shadowing: np.ndarray = None,
    use_rayleigh: bool = False,
    sigma_r: float = 0.70710678,
) -> np.ndarray:
    """
    Calcula a matriz de ganho de canal combinando path loss (d⁻⁴),
    shadowing log-normal, fast fading opcional e ganho de antena G.
    """
    # 1. Tratamento do Shadowing
    if shadowing is None:
        shadowing = gerar_shadowing(ues, aps)
    shadowing = np.asarray(shadowing)
    if shadowing.ndim == 2:
        shadowing = shadowing[:, :, None]  # adiciona dimensão de feixe

    # 2. Tratamento do Rayleigh Fading (Nova Lógica)
    rayleigh = 1.0
    if use_rayleigh:
        rayleigh = gerar_rayleigh(ues, aps, sigma_r)
        if rayleigh.ndim == 2:
            rayleigh = rayleigh[:, :, None]  # adiciona dimensão de feixe

    # 3. Cálculo da Distância
    ue_xy = np.array([[ue.x, ue.y] for ue in ues], dtype=float)
    ap_xy = np.array([[ap.x, ap.y] for ap in aps], dtype=float)
    d = np.linalg.norm(ue_xy[:, None, :] - ap_xy[None, :, :], axis=2)  # (n_ues, n_aps)
    d = np.maximum(d, 1.0)

    # 4. Ganho Final (Multiplicando Rayleigh ao quadrado para virar potência)
    return shadowing * (rayleigh**2) * 1e-4 / (d[:, :, None] ** 4) * G


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
    phi = angle_matrix(ues, aps)  # (n_ues, n_aps)
    m = np.array([ue.ap.id for ue in ues])  # AP servidor de cada UE
    phi_hat = phi[np.arange(n), m] + np.deg2rad(aoa_err_deg) * np.random.randn(n)
    phi_dir = phi[:, m].T  # phi_dir[k, i] = azimute do UE i no AP m_k
    return mrc_gain(phi_hat[:, None], phi_dir, L)

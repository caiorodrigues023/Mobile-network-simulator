# ---------------------------------------------------------------------------
# antennas.py — Modelos de antena, arranjos (UCA/ULA) e Strategy Registry
# ---------------------------------------------------------------------------

from __future__ import annotations

from functools import cache
from typing import Callable

import numpy as np

from .config import _LAMBDA, SECTOR_BEAMWIDTH_DEG, SECTOR_G_MIN

# ---------------------------------------------------------------------------
# SMB — Switched Multi-Beam com Uniform Circular Array (UCA)
# ---------------------------------------------------------------------------


@cache
def _smb_weights(L: int, J: int) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pré-computa a matriz de pesos W (L × J) do SMB: cada coluna é o vetor
    de pesos MRC (conjugado do steering vector) apontado para θ_k = k·2π/J,
    normalizado por ‖w‖ = √L.
    """
    R = L * _LAMBDA / (4.0 * np.pi)
    phi_el = np.arange(L) * (2.0 * np.pi / L)
    theta_k = np.arange(J) * (2.0 * np.pi / J)
    a_tgt = np.exp(1j * 2.0 * np.pi / _LAMBDA * R * np.cos(theta_k[None, :] - phi_el[:, None]))
    W = np.conj(a_tgt) / np.linalg.norm(a_tgt, axis=0, keepdims=True)
    return W, phi_el, R


def smb_gains(phi, L: int = 8, J: int = 8) -> np.ndarray:
    """
    Ganho SMB para azimutes phi (escalar ou array de qualquer shape).
    Retorna |w_k^T a(phi)|² de todos os J feixes → shape = phi.shape + (J,).
    """
    phi_arr = np.asarray(phi, dtype=float)
    W, phi_el, R = _smb_weights(L, J)
    flat = phi_arr.ravel()
    A = np.exp(1j * 2.0 * np.pi / _LAMBDA * R * np.cos(flat[None, :] - phi_el[:, None]))
    y = W.T @ A  # (J, N)
    return (np.abs(y) ** 2).T.reshape(phi_arr.shape + (J,))


# ---------------------------------------------------------------------------
# Beamforming adaptativo MRC (UCA)
# ---------------------------------------------------------------------------


def uca_steering(phi, L: int) -> np.ndarray:
    """Steering vector a(φ) do UCA de L elementos. Retorna shape phi.shape + (L,)."""
    phi_arr = np.asarray(phi, dtype=float)
    R = L * _LAMBDA / (4.0 * np.pi)
    phi_el = np.arange(L) * (2.0 * np.pi / L)
    return np.exp(1j * 2.0 * np.pi / _LAMBDA * R * np.cos(phi_arr[..., None] - phi_el))


def mrc_gain(phi_beam, phi_user, L: int = 8) -> np.ndarray:
    """
    Ganho do feixe MRC (UCA) apontado para phi_beam, avaliado em phi_user:
        w = conj(a(phi_beam)) / √L,   G = |w^T a(phi_user)|²
    Com phi_user == phi_beam o ganho é exatamente L.
    """
    a_b = uca_steering(phi_beam, L)
    a_u = uca_steering(phi_user, L)
    y = np.sum(np.conj(a_b) / np.sqrt(L) * a_u, axis=-1)
    return np.abs(y) ** 2


# ---------------------------------------------------------------------------
# Uniform Linear Array (ULA, d = λ/2) — Atividade 2 e Massive MIMO
# ---------------------------------------------------------------------------


def ula_steering(theta, L: int) -> np.ndarray:
    """
    Steering vector a(θ) do ULA (d = λ/2) medido a partir do broadside (+Y):
        a_ℓ(θ) = exp(j π ℓ sin(θ)),   ℓ = 0, …, L-1.
    Retorna shape theta.shape + (L,).
    """
    theta_arr = np.asarray(theta, dtype=float)
    ell = np.arange(L, dtype=float)
    return np.exp(1j * np.pi * np.sin(theta_arr)[..., None] * ell)


def calc_ula_gain(theta_real: np.ndarray, theta_tgt: np.ndarray, L: int) -> np.ndarray:
    """
    Calcula o ganho diretivo do ULA (d = λ/2) de forma 100% vetorizada.
    `theta_real` e `theta_tgt` são ângulos (rad) medidos a partir do broadside (+Y).
    No alinhamento perfeito (theta_real == theta_tgt), o ganho é exatamente L.
    """
    t_real = np.asarray(theta_real, dtype=float)
    t_tgt = np.asarray(theta_tgt, dtype=float)
    ell = np.arange(L, dtype=float)
    d_sin = np.sin(t_real) - np.sin(t_tgt)
    fase = np.pi * d_sin[..., None] * ell
    return np.abs(np.sum(np.exp(1j * fase), axis=-1)) ** 2 / float(L)


def ula_2d_gain(phi_beam, phi_user, L: int = 8) -> np.ndarray:
    """
    Ganho de um ULA (d = λ/2) alinhado ao eixo X (broadside em +Y) no plano 2D
    usando azimutes padrão φ = arctan2(dy, dx).
    Como o ângulo a partir do broadside satisfaz sin(ψ) = cos(φ),
    a diferença de fase é π · ℓ · (cos(φ_user) - cos(φ_beam)).
    """
    p_beam = np.asarray(phi_beam, dtype=float)
    p_user = np.asarray(phi_user, dtype=float)
    ell = np.arange(L, dtype=float)
    d_cos = np.cos(p_user) - np.cos(p_beam)
    fase = np.pi * d_cos[..., None] * ell
    return np.abs(np.sum(np.exp(1j * fase), axis=-1)) ** 2 / float(L)


# ---------------------------------------------------------------------------
# Modelos básicos e Strategy Registry
# ---------------------------------------------------------------------------


def omni_gains(phi: np.ndarray, **_kwargs) -> np.ndarray:
    """Antena omnidirecional: ganho 1.0, shape (K, M, 1)."""
    return np.ones(phi.shape + (1,), dtype=float)


def sector_gains(
    phi: np.ndarray,
    beamwidth: float = np.deg2rad(SECTOR_BEAMWIDTH_DEG),
    g_min: float = SECTOR_G_MIN,
    **_kwargs,
) -> np.ndarray:
    """Antena setorial ideal de 3 setores (boresights 0°, 120°, 240°)."""
    theta_boresight = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    delta = np.abs(phi[:, :, None] - theta_boresight[None, None, :]) % (2.0 * np.pi)
    delta = np.minimum(delta, 2.0 * np.pi - delta)
    return np.where(delta <= beamwidth / 2.0, 1.0, g_min)


def _smb_strategy(phi: np.ndarray, L: int = 8, J: int = 8, **_kwargs) -> np.ndarray:
    return smb_gains(phi, L=L, J=J)


# Registry de estratégias de acesso inicial (mapeia chave da UI → função vetorizada)
ANTENNA_REGISTRY: dict[str, Callable[..., np.ndarray]] = {
    "omni": omni_gains,
    "sector": sector_gains,
    "smb": _smb_strategy,
    # Modelos adaptativos (MRC UCA e ULA) usam antena omni no initial access;
    # o ganho adaptativo espacial é aplicado em seguida sobre os UEs associados.
    "mrc": omni_gains,
    "adaptive": omni_gains,
    "adaptive_1": omni_gains,
    "adaptive_5": omni_gains,
    "adaptive_10": omni_gains,
    "ula": omni_gains,
}


def is_adaptive_model(model: str) -> bool:
    """Retorna True se o modelo forma feixes adaptativos por UE após o initial access."""
    return model in ("mrc", "ula") or model.startswith("adaptive")


def resolve_aoa_error_deg(model: str, default_err_deg: float = 0.0) -> float:
    """Extrai o erro de estimação AoA (graus) de nomes como 'adaptive_5', se presente."""
    if "_" in model and model.startswith("adaptive"):
        try:
            return float(model.split("_", 1)[1])
        except ValueError:
            pass
    return float(default_err_deg)


def compute_antenna_gain_matrix(
    phi: np.ndarray,
    model: str = "sector",
    beamwidth: float = 2.0 * np.pi / 3.0,
    g_min: float = SECTOR_G_MIN,
    L: int = 8,
    J: int = 8,
) -> np.ndarray:
    """Dispatcher puro em NumPy que recebe matriz de azimutes phi (K, M) → G (K, M, B)."""
    fn = ANTENNA_REGISTRY.get(model)
    if fn is None:
        if model.startswith("adaptive"):
            fn = omni_gains
        else:
            raise ValueError(f"Modelo de antena desconhecido: {model!r}")
    return fn(phi, beamwidth=beamwidth, g_min=g_min, L=L, J=J)


def antenna_gain(
    ues,
    aps,
    model: str = "sector",
    beamwidth: float = 2.0 * np.pi / 3.0,
    g_min: float = SECTOR_G_MIN,
    L: int = 8,
    J: int = 8,
) -> np.ndarray:
    """
    Dispatcher principal compatível tanto com arrays (K, 2)/(M, 2) quanto com listas de UE/AP.
    Retorna matriz de ganho de antena G[i, j, b].
    """
    from .channel import angle_matrix

    phi = angle_matrix(ues, aps)
    return compute_antenna_gain_matrix(
        phi, model=model, beamwidth=beamwidth, g_min=g_min, L=L, J=J
    )


# ---------------------------------------------------------------------------
# Padrões de radiação reais para visualização (2D/3D)
# ---------------------------------------------------------------------------


def compute_beam_lobe_xy(
    ap_x: float,
    ap_y: float,
    model: str,
    beam_angle_rad: float = 0.0,
    radius: float = 95.0,
    L: int = 8,
    J: int = 8,
    beam_idx: int = 0,
    beamwidth_deg: float = SECTOR_BEAMWIDTH_DEG,
    n_pts: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula coordenadas (x, y) exatas do lóbulo de radiação (normalizado para `radius`)
    para plotagem limpa no mapa 2D/3D sem aproximações artificiais.
    """
    th = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=True)
    if model == "omni":
        r = np.full_like(th, radius * 0.65)
        return ap_x + r * np.cos(th), ap_y + r * np.sin(th)

    if model == "sector":
        boresights = (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
        b0 = boresights[beam_idx % 3]
        half = np.deg2rad(beamwidth_deg) / 2.0
        arc = np.linspace(b0 - half, b0 + half, 40)
        xs = np.concatenate(([ap_x], ap_x + radius * np.cos(arc), [ap_x]))
        ys = np.concatenate(([ap_y], ap_y + radius * np.sin(arc), [ap_y]))
        return xs, ys

    if model == "smb":
        g = smb_gains(th, L=L, J=J)[:, beam_idx % J] / float(L)
        r = radius * g
        return ap_x + r * np.cos(th), ap_y + r * np.sin(th)

    if model == "ula":
        g = ula_2d_gain(beam_angle_rad, th, L=L) / float(L)
        r = radius * g
        return ap_x + r * np.cos(th), ap_y + r * np.sin(th)

    # MRC / Adaptive (UCA)
    g = mrc_gain(beam_angle_rad, th, L=L) / float(L)
    r = radius * g
    return ap_x + r * np.cos(th), ap_y + r * np.sin(th)

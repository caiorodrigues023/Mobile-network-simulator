# ---------------------------------------------------------------------------
# antennas.py — Modelos e diagramas de irradiação (Omni, Setorial, SMB, MRC)
# ---------------------------------------------------------------------------

from functools import lru_cache

import numpy as np

from .config import _LAMBDA


# ---------------------------------------------------------------------------
# SMB — Switched Multi-Beam com Uniform Circular Array (Apêndice A/B)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _smb_weights(L: int, J: int):
    """
    Pré-computa a matriz de pesos W (L × J) do SMB: cada coluna é o vetor
    de pesos MRC (conjugado do steering vector) apontado para θ_k = k·2π/J,
    normalizado por ‖w‖. Calculada uma única vez por (L, J).
    """
    R = L * _LAMBDA / (4 * np.pi)
    phi_el = np.arange(L) * 2 * np.pi / L                          # posição angular dos elementos
    theta_k = np.arange(J) * 2 * np.pi / J                         # direção de apontamento de cada feixe
    a_tgt = np.exp(1j * 2 * np.pi / _LAMBDA * R *
                   np.cos(theta_k[None, :] - phi_el[:, None]))      # (L, J)
    W = np.conj(a_tgt) / np.linalg.norm(a_tgt, axis=0, keepdims=True)
    return W, phi_el, R


def smb_gains(phi, L: int = 8, J: int = 8) -> np.ndarray:
    """
    Ganho SMB para azimutes phi (escalar ou array de qualquer shape).

    Retorna: |w_k^T a(phi)|² de todos os J feixes → shape = phi.shape + (J,).
    O pico de cada feixe vale L; o cruzamento entre feixes vizinhos ≈ 1,82
    para L = J = 8.
    """
    phi = np.asarray(phi, dtype=float)
    W, phi_el, R = _smb_weights(L, J)
    flat = phi.ravel()
    A = np.exp(1j * 2 * np.pi / _LAMBDA * R *
               np.cos(flat[None, :] - phi_el[:, None]))             # (L, N)
    y = W.T @ A                                                     # (J, N)
    return (np.abs(y) ** 2).T.reshape(phi.shape + (J,))


# ---------------------------------------------------------------------------
# Beamforming adaptativo MRC (Exercício 3)
# ---------------------------------------------------------------------------

def _uca_steering(phi, L: int) -> np.ndarray:
    """
    Steering vector a(φ) do UCA de L elementos (Apêndice A).
    Retorna shape phi.shape + (L,).
    """
    phi = np.asarray(phi, dtype=float)
    R = L * _LAMBDA / (4 * np.pi)
    phi_el = np.arange(L) * 2 * np.pi / L
    return np.exp(1j * 2 * np.pi / _LAMBDA * R *
                  np.cos(phi[..., None] - phi_el))


def mrc_gain(phi_beam, phi_user, L: int = 8) -> np.ndarray:
    """
    Ganho do feixe MRC apontado para phi_beam, avaliado na direção phi_user
    (broadcast entre os dois):
        w = conj(a(phi_beam)) / √L,   G = |w^T a(phi_user)|²

    Com phi_user == phi_beam o ganho é exatamente L (eq. 13 do roteiro).
    """
    a_b = _uca_steering(phi_beam, L)
    a_u = _uca_steering(phi_user, L)
    y = np.sum(np.conj(a_b) / np.sqrt(L) * a_u, axis=-1)
    return np.abs(y) ** 2


# ---------------------------------------------------------------------------
# Dispatcher principal de ganho de antena
# ---------------------------------------------------------------------------

def antenna_gain(ues: list, aps: list, model: str = 'sector',
                 beamwidth: float = 2 * np.pi / 3,
                 g_min: float = 0.01,
                 L: int = 8, J: int = 8) -> np.ndarray:
    """
    Matriz de ganho de antena G[i, j, b] para cada UE i, AP j e setor/feixe b.

    Modelos suportados
    ------------------
    'omni'      Antena omnidirecional — ganho unitário, 1 "feixe" por AP.
    'sector'    Antena setorial ideal — 3 setores de 'beamwidth' cada.
    'adaptive'* Beamforming adaptativo MRC. Retorna ganho unitário (omni)
                para o initial access; o ganho MRC real é aplicado depois,
                por UE, via channel.adaptive_gains().
    'smb'       Switched Multi-Beam com UCA de L elementos e J feixes.
    """
    from .channel import angle_matrix  # importação local para evitar ciclo

    phi = angle_matrix(ues, aps)       # (n_ues, n_aps)

    # Omni e todas as variações do adaptativo (adaptive_1, adaptive_5, …)
    if model == 'omni' or model.startswith('adaptive'):
        return np.ones((len(ues), len(aps), 1))

    if model == 'sector':
        theta_boresight = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
        delta = np.abs(phi[:, :, None] - theta_boresight[None, None, :]) % (2 * np.pi)
        delta = np.minimum(delta, 2 * np.pi - delta)
        return np.where(delta <= beamwidth / 2, 1.0, g_min)

    if model == 'smb':
        return smb_gains(phi, L=L, J=J)

    raise ValueError(f"Modelo de antena desconhecido: {model!r}")

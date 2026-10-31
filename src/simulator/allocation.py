# ---------------------------------------------------------------------------
# allocation.py — Attach, Alocação de Canais e DPC
# ---------------------------------------------------------------------------

import random

import numpy as np

from .config import (
    GAMMA_MAX_DB,
    GAMMA_MIN_DB,
    MU_STEP,
    P_MAX_DBM,
    P_MIN_DBM,
)

# ---------------------------------------------------------------------------
# Associação UE → AP (Initial Access)
# ---------------------------------------------------------------------------


def attach_AP_UE(gains: np.ndarray) -> tuple:
    """
    gains: matriz (K, M, B) de ganho total de canal.
    Retorna:
      - serving_ap: vetor (K,) com o índice do AP associado a cada UE.
      - serving_beam: vetor (K,) com o índice do setor/feixe associado.
    """
    K = gains.shape[0]
    # Encontra o índice linear do maior ganho para cada linha K
    flat_indices = np.argmax(gains.reshape(K, -1), axis=1)

    # Desempacota o índice linear para (M, B)
    serving_ap, serving_beam = np.unravel_index(flat_indices, gains.shape[1:])
    return serving_ap, serving_beam


# ---------------------------------------------------------------------------
# Alocação de canais ortogonais intra-célula
# ---------------------------------------------------------------------------


def alocar_canais_ortogonal(
    access_points: list, ues: list, number_channels: int, allocation: str
) -> None:
    """
    Aloca canais aos UEs de forma ortogonal dentro de cada célula (AP).

    Modos
    -----
    'random'    Cada UE recebe um canal aleatório (1 a number_channels).
    qualquer    Alocação ortogonal intra-AP: distribui canais sequencialmente
                por AP, minimizando reutilização intra-célula.

    Modifica os atributos 'channel' de cada UE in-place.
    """
    # 1. Zera os canais de todos os UEs
    for ue in ues:
        ue.channel = None

    if allocation == "random":
        for ue in ues:
            ue.channel = np.random.randint(1, number_channels + 1)
        return

    # 2. Contador global de uso de cada canal
    channels = {i: 0 for i in range(1, number_channels + 1)}

    # 3. Alocação intra-AP
    for ap in access_points:
        available_channels = list(channels.keys())
        random.shuffle(available_channels)

        for ue0 in ap.ues:
            if not available_channels:
                break  # mais UEs do que canais no AP

            channel = available_channels.pop()
            ues[ue0.id].channel = channel
            channels[channel] += 1

    # 4. UEs sem canal (AP com mais UEs do que canais) recebem o canal menos usado
    for ue in ues:
        if ue.channel is None:
            least_used = min(channels, key=channels.get)
            ue.channel = least_used
            channels[least_used] += 1


# ---------------------------------------------------------------------------
# Controle Distribuído de Potência (DPC)
# ---------------------------------------------------------------------------


def DPC(
    ues: list,
    N: int,
    t_max: int,
    G: np.ndarray,
    R: np.ndarray,
    y_tar: float,
    p_min: float,
    p_max: float,
    p_init: float,
) -> np.ndarray:
    """
    Algoritmo de Controle Distribuído de Potência.

    Parâmetros
    ----------
    N      : número de canais ortogonais (para cálculo do ruído).
    t_max  : número máximo de iterações.
    G      : ganho de antena do UE i no AP servidor (indexado por UE).
    R      : matriz de ganho de canal (n_ues, n_aps).
    y_tar  : SINR alvo.
    p_min  : potência mínima (W).
    p_max  : potência máxima (W).
    p_init : potência inicial (W).

    Retorna: matriz de potências p[:, 0:iteracoes].
    """
    p = np.ones((len(ues), t_max)) * p_init
    y = np.zeros((len(ues), t_max))
    pn = 1e8 * 1e-20 / N
    iteracoes = 0

    for t in range(t_max):
        for i, ue in enumerate(ues):
            interference = sum(
                p[k][t] * G[k][ue.ap.id] * R[k][ue.ap.id]
                for k in range(len(ues))
                if k != i and ues[k].channel == ue.channel
            )
            ue.interference.append(interference)

            sinr = (p[i][t] * ue.gain * R[i][ue.ap.id]) / (interference + pn)
            y[i][t] = sinr

            pt = min(max(p_min, p[i][t] * y_tar / sinr), p_max)
            if t < t_max - 1:
                p[i][t + 1] = pt

        iteracoes += 1
        if iteracoes > 5 and t < t_max - 1:
            soma_atual = np.sum(p[:, t + 1])
            soma_anterior = np.sum(p[:, t])
            if abs(soma_atual - soma_anterior) < 1e-3:
                break

    return p[:, 0:iteracoes]


def DPC_Dual_Loop(
    G_eff: np.ndarray,
    P_n: float,
    gamma_tar_db: np.ndarray,
    ue_active: np.ndarray,
    t_outer: int = 35,
    t_inner: int = 15,
    tol_db: float = 0.5,
) -> tuple:
    """
    Joint MRC & Dual-Loop Power Control.
    Utiliza as matrizes G (ganho antena) e R (ganho canal) extraídas dos UEs.
    """
    K = G_eff.shape[0]
    P_dbm = np.full(K, P_MAX_DBM)

    for outer in range(t_outer):
        if not np.any(ue_active):
            break

        gamma_tar_lin = 10.0 ** (gamma_tar_db / 10.0)
        inner_steps = 30 if outer == 0 else t_inner

        for _ in range(inner_steps):
            p_watts = np.where(ue_active, 10.0 ** (P_dbm / 10.0) / 1000.0, 0.0)
            rx_tot = G_eff @ p_watts
            rx_util = np.diagonal(G_eff) * p_watts
            interf = rx_tot - rx_util
            sinr_lin = np.where(p_watts > 0, rx_util / (interf + P_n), 0.0)

            direcao = np.sign(gamma_tar_lin - sinr_lin)
            P_dbm = np.where(
                ue_active,
                np.clip(P_dbm + MU_STEP * direcao, P_MIN_DBM, P_MAX_DBM),
                P_dbm,
            )

        # Outer Loop
        p_watts = np.where(ue_active, 10.0 ** (P_dbm / 10.0) / 1000.0, 0.0)
        rx_tot = G_eff @ p_watts
        rx_util = np.diagonal(G_eff) * p_watts
        interf = rx_tot - rx_util
        sinr_lin = np.where(p_watts > 0, rx_util / (interf + P_n), 0.0)
        sinr_db = 10.0 * np.log10(sinr_lin + 1e-30)

        target_changed = False
        for k in range(K):
            if not ue_active[k]:
                continue
            old_tar = gamma_tar_db[k]

            if (sinr_db[k] >= gamma_tar_db[k] - tol_db) and (P_dbm[k] < P_MAX_DBM):
                gamma_tar_db[k] = min(GAMMA_MAX_DB, gamma_tar_db[k] + 1.0)
            elif (P_dbm[k] >= P_MAX_DBM) and (sinr_db[k] < gamma_tar_db[k] - tol_db):
                gamma_tar_db[k] -= 1.0

            if gamma_tar_db[k] < GAMMA_MIN_DB:
                ue_active[k] = False
                target_changed = True
            elif gamma_tar_db[k] != old_tar:
                target_changed = True

        if not target_changed:
            break

    p_watts_final = np.where(ue_active, 10.0 ** (P_dbm / 10.0) / 1000.0, 0.0)
    return p_watts_final, sinr_lin

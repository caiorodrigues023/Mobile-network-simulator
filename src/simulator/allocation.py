# ---------------------------------------------------------------------------
# allocation.py — Attach, Alocação de Canais e DPC
# ---------------------------------------------------------------------------

import random

import numpy as np

from .channel import angle
from .config import (
    GAMMA_MAX_DB,
    GAMMA_MIN_DB,
    GAMMA_TAR_INIT,
    MU_STEP,
    P_MAX_DBM,
    P_MIN_DBM,
)

# ---------------------------------------------------------------------------
# Associação UE → AP (Initial Access)
# ---------------------------------------------------------------------------


def attach_AP_UE(ues: list, aps: list, gains: np.ndarray) -> None:
    """
    Initial access: cada UE se associa ao par (AP, setor/feixe) com o MAIOR
    ganho de canal, entre as M × B combinações (B = nº de setores/feixes por AP:
    1 no omni, 3 no setorizado, J no SMB).

    Preenche em cada UE: ue.ap, ue.beam, ue.gain, ue.dist e ue.angle.
    Adiciona cada UE à lista ue.ap.ues.
    """
    for i, ue in enumerate(ues):
        j, b = np.unravel_index(np.argmax(gains[i]), gains[i].shape)
        ue.ap = aps[j]
        ue.beam = int(b)
        ue.gain = gains[i, j, b]
        ue.dist = np.linalg.norm(np.array([ue.x, ue.y]) - np.array([ue.ap.x, ue.ap.y]))
        ue.angle = angle(ue, ue.ap)
        aps[j].ues.append(ue)


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
    ues: list,
    N: int,
    G: np.ndarray,
    R: np.ndarray,
    t_outer: int = 35,
    t_inner: int = 15,
    tol_db: float = 0.5,
) -> np.ndarray:
    """
    Joint MRC & Dual-Loop Power Control (Atividade 1 do SciChal4A).
    Utiliza as matrizes G (ganho antena) e R (ganho canal) extraídas dos UEs.
    """
    K = len(ues)
    P_dbm = np.full(K, P_MAX_DBM)
    gamma_tar_db = np.full(K, GAMMA_TAR_INIT)
    ue_active = np.ones(K, dtype=bool)

    # Ruído térmico
    pn = 1e8 * 1e-20 / N

    # Extrai matrizes puras para vetorização
    G_eff = np.zeros((K, K))
    for i, ue in enumerate(ues):
        for j, ue_interf in enumerate(ues):
            if ue.channel == ue_interf.channel:
                G_eff[i, j] = G[j, ue.ap.id] * R[j, ue.ap.id]

    for outer in range(t_outer):
        if not np.any(ue_active):
            break

        gamma_tar_lin = 10.0 ** (gamma_tar_db / 10.0)
        inner_steps = 30 if outer == 0 else t_inner

        for _ in range(inner_steps):
            p_watts = np.where(ue_active, 10.0 ** (P_dbm / 10.0) / 1000.0, 0.0)

            # Cálculo de SINR Vetorizado
            total_rx = G_eff @ p_watts
            util = np.diagonal(G_eff) * p_watts
            interf = total_rx - util
            sinr_lin = np.where(p_watts > 0, util / (interf + pn), 0.0)

            # UD-DPC Step
            direcao = np.sign(gamma_tar_lin - sinr_lin)
            P_next = np.clip(P_dbm + MU_STEP * direcao, P_MIN_DBM, P_MAX_DBM)
            P_dbm = np.where(ue_active, P_next, P_dbm)

        # Avaliação de Metas (Outer Loop)
        p_watts = np.where(ue_active, 10.0 ** (P_dbm / 10.0) / 1000.0, 0.0)
        total_rx = G_eff @ p_watts
        util = np.diagonal(G_eff) * p_watts
        interf = total_rx - util
        sinr_lin = np.where(p_watts > 0, util / (interf + pn), 0.0)
        sinr_db = 10.0 * np.log10(sinr_lin + 1e-30)

        target_changed = False
        for k in range(K):
            if not ue_active[k]:
                continue
            old_tar = gamma_tar_db[k]

            if (sinr_db[k] >= gamma_tar_db[k] - tol_db) and (P_dbm[k] < P_MAX_DBM):
                gamma_tar_db[k] = min(GAMMA_MAX_DB, gamma_tar_db[k] + 1.0)
            elif (P_dbm[k] >= P_MAX_DBM) and (sinr_db[k] < gamma_tar_db[k] - tol_db):
                gamma_tar_db[k] = gamma_tar_db[k] - 1.0

            if gamma_tar_db[k] < GAMMA_MIN_DB:
                ue_active[k] = False
                target_changed = True
            elif gamma_tar_db[k] != old_tar:
                target_changed = True

        if not target_changed:
            break

    # Salva potências finais nos UEs
    p_watts_final = np.where(ue_active, 10.0 ** (P_dbm / 10.0) / 1000.0, 0.0)
    for i, ue in enumerate(ues):
        ue.power = p_watts_final[i]

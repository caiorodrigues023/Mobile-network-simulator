(
    """"""
    """"""
    """"""
)  # ---------------------------------------------------------------------------
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
    G_eff: np.ndarray,  # (K, K) -> G_eff[k,i] = ganho de canal × ganho MRC do feixe do UE k sobre o UE i, visto no AP do UE k. A diagonal contém o ganho do próprio UE k.
    P_n: float,  # Potência de ruído (W)
    gamma_tar_db: np.ndarray,  # SINR alvo inicial (dB)
    ue_active: np.ndarray,  # Vetor de ativação dos UEs (1 para ativo, 0 para inativo)
    t_outer: int = 50,  # máximo de iterações do loop externo
    t_inner: int = 5000,  # máximo de iterações do loop interno
    tol_db: float = 0.5,  # tolerância de convergência do loop externo (dB)
    return_history: bool = False,  # Se True, retorna também o histórico de potências e SINRs (para plotagem e análise)
) -> tuple:
    """
    Joint MRC & Dual-Loop Power Control.

    Loop interno : DPC em passos de ±MU_STEP dB -> SINR persegue o alvo.
    Loop externo : renegocia o alvo de cada UE (sobe/desce 1 dB) e
                   desliga UEs cujo alvo caiu abaixo de GAMMA_MIN_DB.

    Se return_history=True, retorna também um dict com a evolução temporal:
      'power_w'    : (T, K) potência (W) a cada passo do loop interno
      'sinr_lin'   : (T, K) SINR linear a cada passo do loop interno
      'target_db'  : (n_outer+1, K) alvo de cada UE (linha 0 = alvo inicial)
      'outer_marks': índices (em T) onde cada iteração do loop externo termina
      'active'     : (n_outer+1, K) máscara de UEs ativos

    """
    K = G_eff.shape[0]

    # Todos os UEs começam em potência máxima (p_init = p_max, como no roteiro)
    P_dbm = np.full(K, P_MAX_DBM)

    # Estrutura do histórico
    hist = {
        "power_w": [],
        "sinr_lin": [],
        "target_db": [gamma_tar_db.copy()],
        "outer_marks": [],
        "active": [ue_active.copy()],
    }

    # ---------------- LOOP EXTERNO ----------------
    for outer in range(t_outer):
        # Se todos foram desligados, não há mais nada a ser feito
        if not np.any(ue_active):
            break

        # Alvos atuais convertidos para escala linear
        gamma_tar_lin = 10.0 ** (gamma_tar_db / 10.0)

        # histórico dos 2 últimos vetores de potência (dBm) desta rodada externa
        P_hist = []
        n_ciclo = 0  # passos consecutivos com P(t) == P(t-2)
        N_CICLO = 3  # confirmações exigidas

        # ---------------- LOOP INTERNO ----------------
        for _ in range(t_inner):
            # dBm -> Watts (UE desligado transmite 0 W)
            p_watts = np.where(ue_active, 10.0 ** (P_dbm / 10.0) / 1000.0, 0.0)

            # Potência total recebida no AP de cada UE k: sum_i G_eff[k,i]*p_i
            rx_tot = G_eff @ p_watts
            # Sinal útil: termo diagonal (o próprio UE k)
            rx_util = np.diagonal(G_eff) * p_watts
            # Interferência = total - útil (UEs de outros APs e do mesmo AP)
            interf = rx_tot - rx_util

            # SINR linear (0 para UEs desligados, evitando divisão inútil)
            sinr_lin = np.where(p_watts > 0, rx_util / (interf + P_n), 0.0)

            # Grava a potência usada neste passo e o SINR resultante
            if return_history:
                hist["power_w"].append(p_watts.copy())
                hist["sinr_lin"].append(sinr_lin.copy())

            # --- critério de parada: ciclo de período <= 2 ---
            if len(P_hist) >= 2 and np.array_equal(P_dbm, P_hist[-2]):
                n_ciclo += 1
                if n_ciclo >= N_CICLO:
                    break
            else:
                n_ciclo = 0
            P_hist.append(P_dbm.copy())
            if len(P_hist) > 2:
                P_hist.pop(0)

            # --- atualização fixa de ±MU_STEP ---
            direcao = np.sign(gamma_tar_lin - sinr_lin)
            P_dbm = np.where(
                ue_active,
                np.clip(P_dbm + MU_STEP * direcao, P_MIN_DBM, P_MAX_DBM),
                P_dbm,
            )
        # ---------------- FIM ----------------

        # ---------------- LOOP EXTERNO ----------------
        # Reavalia o SINR com as potências resultantes do loop interno
        p_watts = np.where(ue_active, 10.0 ** (P_dbm / 10.0) / 1000.0, 0.0)
        rx_tot = G_eff @ p_watts
        rx_util = np.diagonal(G_eff) * p_watts
        interf = rx_tot - rx_util
        sinr_lin = np.where(p_watts > 0, rx_util / (interf + P_n), 0.0)
        sinr_db = 10.0 * np.log10(sinr_lin + 1e-30)  # 1e-30 evita log(0)

        target_changed = False  # flag de convergência do loop externo

        for k in range(K):
            if not ue_active[k]:
                continue
            old_tar = gamma_tar_db[k]

            # Regra 1: atingiu o alvo (dentro da tolerância) e ainda tem
            # folga de potência -> eleva o alvo em 1 dB (até o teto de 20 dB,
            # saturação QAM) para explorar o ganho espacial.
            if (sinr_db[k] >= gamma_tar_db[k] - tol_db) and (P_dbm[k] < P_MAX_DBM):
                gamma_tar_db[k] = min(GAMMA_MAX_DB, gamma_tar_db[k] + 1.0)

            # Regra 2: está em p_max e não alcança o alvo -> baixa o alvo
            # em 1 dB (negociação), reduzindo a pressão por potência/interferência.
            elif (P_dbm[k] >= P_MAX_DBM) and (sinr_db[k] < gamma_tar_db[k] - tol_db):
                gamma_tar_db[k] -= 1.0

            # Regra 3: alvo negociado caiu abaixo do limiar -> blocking (outage)
            if gamma_tar_db[k] < GAMMA_MIN_DB:
                ue_active[k] = False  # potência vira 0 W nas próximas contas
                target_changed = True
            elif gamma_tar_db[k] != old_tar:
                target_changed = True

        # Grava o histórico do loop externo
        hist["outer_marks"].append(len(hist["power_w"]))
        hist["target_db"].append(gamma_tar_db.copy())
        hist["active"].append(ue_active.copy())

        # Se nenhum alvo mudou e ninguém foi desligado, o sistema estabilizou
        if not target_changed:
            break

    # Potência final (0 W para quem está em outage)
    p_watts_final = np.where(ue_active, 10.0 ** (P_dbm / 10.0) / 1000.0, 0.0)

    if return_history:
        hist["power_w"] = np.array(hist["power_w"])
        hist["sinr_lin"] = np.array(hist["sinr_lin"])
        hist["target_db"] = np.array(hist["target_db"])
        hist["active"] = np.array(hist["active"])
        return p_watts_final, sinr_lin, hist
    return p_watts_final, sinr_lin

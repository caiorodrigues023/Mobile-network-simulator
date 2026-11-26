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
    p_max_dbm: float = P_MAX_DBM,
    p_min_dbm: float = P_MIN_DBM,
    mu_step: float = MU_STEP,
    gamma_max_db: float = GAMMA_MAX_DB,
    gamma_min_db: float = GAMMA_MIN_DB,
) -> tuple:
    """
    Joint MRC & Dual-Loop Power Control.

    Loop interno : DPC em passos de ±mu_step dB -> SINR persegue o alvo.
    Loop externo : renegocia o alvo de cada UE (sobe/desce 1 dB) e
                   desliga UEs cujo alvo caiu abaixo de gamma_min_db.

    Se return_history=True, retorna também um dict com a evolução temporal:
      'power_w'    : (T, K) potência (W) a cada passo do loop interno
      'sinr_lin'   : (T, K) SINR linear a cada passo do loop interno
      'target_db'  : (n_outer+1, K) alvo de cada UE (linha 0 = alvo inicial)
      'outer_marks': índices (em T) onde cada iteração do loop externo termina
      'active'     : (n_outer+1, K) máscara de UEs ativos

    """
    K = G_eff.shape[0]
    gamma_tar_db = np.array(gamma_tar_db, dtype=float, copy=True)
    ue_active = np.array(ue_active, dtype=bool, copy=True)

    # Todos os UEs começam em potência máxima (p_init = p_max, como no roteiro)
    P_dbm = np.full(K, float(p_max_dbm), dtype=float)
    diag_G = np.diagonal(G_eff).copy()
    ln10_div10 = float(np.log(10.0) / 10.0)
    p_max_f = float(p_max_dbm)
    p_min_f = float(p_min_dbm)
    mu_f = float(mu_step)
    tol_f = float(tol_db)
    gmax_f = float(gamma_max_db)
    gmin_f = float(gamma_min_db)
    gamma_min_lin = 10.0 ** (gmin_f / 10.0)

    hist = None
    if return_history:
        hist = {
            "power_w": [],
            "sinr_lin": [],
            "target_db": [gamma_tar_db.copy()],
            "outer_marks": [],
            "active": [ue_active.copy()],
        }

    # ---------------- LOOP EXTERNO ----------------
    for _outer in range(t_outer):
        # Se todos foram desligados, não há mais nada a ser feito
        if not np.any(ue_active):
            break

        # Alvos atuais convertidos para escala linear
        gamma_tar_lin = np.exp(gamma_tar_db * ln10_div10)

        P_prev1 = None
        P_prev2 = None
        n_ciclo = 0  # passos consecutivos com P(t) == P(t-2)
        N_CICLO = 2  # confirmações exigidas (visita ambas as fases de eventual ciclo-2)
        last_sinr_lin = None
        inner_shutoff = False

        # ---------------- LOOP INTERNO ----------------
        for _ in range(t_inner):
            if not np.any(ue_active):
                last_sinr_lin = None
                break

            # dBm -> Watts (UE desligado transmite 0 W)
            p_watts = np.where(ue_active, np.exp((P_dbm - 30.0) * ln10_div10), 0.0)

            # Potência total recebida no AP de cada UE k: sum_i G_eff[k,i]*p_i
            rx_tot = G_eff @ p_watts
            rx_util = diag_G * p_watts
            interf = rx_tot - rx_util

            sinr_lin = np.where(ue_active & (p_watts > 0.0), rx_util / (interf + P_n), 0.0)
            last_sinr_lin = sinr_lin

            if return_history:
                hist["power_w"].append(p_watts.copy())
                hist["sinr_lin"].append(sinr_lin.copy())

            # Corte imediato de outage após o transiente inicial (Outer >= 1):
            # se UE já está em P_max e SINR real caiu abaixo de gamma_min_db, desliga no instante
            if _outer >= 1:
                sat_outage = ue_active & (P_dbm >= p_max_f - 1e-9) & (sinr_lin < gamma_min_lin)
                if np.any(sat_outage):
                    worst_k = int(np.argmin(np.where(sat_outage, sinr_lin, np.inf)))
                    ue_active[worst_k] = False
                    gamma_tar_db[worst_k] = gmin_f
                    P_prev1 = None
                    P_prev2 = None
                    n_ciclo = 0
                    last_sinr_lin = None
                    inner_shutoff = True
                    continue

            # --- critério de parada: ciclo de período <= 2 ---
            if P_prev2 is not None and np.array_equal(P_dbm, P_prev2):
                n_ciclo += 1
                if n_ciclo >= N_CICLO:
                    break
            else:
                n_ciclo = 0

            # --- atualização fixa de ±mu_step ---
            direcao = np.sign(gamma_tar_lin - sinr_lin)
            P_next = np.where(
                ue_active,
                np.clip(P_dbm + mu_f * direcao, p_min_f, p_max_f),
                P_dbm,
            )

            P_prev2 = P_prev1
            P_prev1 = P_dbm
            P_dbm = P_next
        else:
            last_sinr_lin = None
        # ---------------- FIM DO LOOP INTERNO ----------------

        # Reavalia o SINR com as potências resultantes do loop interno (reutiliza se recém-calculado)
        if last_sinr_lin is None:
            p_watts = np.where(ue_active, np.exp((P_dbm - 30.0) * ln10_div10), 0.0)
            rx_tot = G_eff @ p_watts
            rx_util = diag_G * p_watts
            interf = rx_tot - rx_util
            sinr_lin = np.where(ue_active & (p_watts > 0.0), rx_util / (interf + P_n), 0.0)
        else:
            sinr_lin = last_sinr_lin
        sinr_db = 10.0 * np.log10(sinr_lin + 1e-30)  # 1e-30 evita log(0)

        # Negociação vetorial do loop externo
        can_up = ue_active & (sinr_db >= gamma_tar_db - tol_f) & (P_dbm < p_max_f)
        need_down = (
            ue_active
            & (~can_up)
            & (P_dbm >= p_max_f - mu_f)
            & (sinr_db < gamma_tar_db - tol_f)
        )

        old_tar = gamma_tar_db.copy()
        if np.any(can_up):
            gamma_tar_db = np.where(can_up, np.minimum(gmax_f, gamma_tar_db + 1.0), gamma_tar_db)
        if np.any(need_down):
            gamma_tar_db = np.where(need_down, gamma_tar_db - 1.0, gamma_tar_db)

        outage_mask = ue_active & (gamma_tar_db < gmin_f)
        outer_shutoff = False
        if np.any(outage_mask):
            ue_active = ue_active & (~outage_mask)
            outer_shutoff = True

        # Ao fim de cada rodada externa (inclusive E1), desliga quem já está em P_max abaixo de gamma_min_db
        while np.any(ue_active):
            sat_outer = ue_active & (P_dbm >= p_max_f - mu_f) & (sinr_db < gmin_f)
            if not np.any(sat_outer):
                break
            worst_k = int(np.argmin(np.where(sat_outer, sinr_db, np.inf)))
            ue_active[worst_k] = False
            gamma_tar_db[worst_k] = gmin_f
            outer_shutoff = True
            p_watts = np.where(ue_active, np.exp((P_dbm - 30.0) * ln10_div10), 0.0)
            rx_tot = G_eff @ p_watts
            rx_util = diag_G * p_watts
            interf = rx_tot - rx_util
            sinr_lin = np.where(ue_active & (p_watts > 0.0), rx_util / (interf + P_n), 0.0)
            sinr_db = 10.0 * np.log10(sinr_lin + 1e-30)

        target_changed = (
            inner_shutoff
            or outer_shutoff
            or (not np.array_equal(gamma_tar_db, old_tar))
        )

        if return_history:
            hist["outer_marks"].append(len(hist["power_w"]))
            hist["target_db"].append(gamma_tar_db.copy())
            hist["active"].append(ue_active.copy())

        # Se nenhum alvo mudou e ninguém foi desligado, o sistema estabilizou
        if not target_changed:
            break

    # Garantia pós-loop: desliga iterativamente o pior UE ativo enquanto existir SINR < gamma_min_db
    post_shutoff_changed = False
    while np.any(ue_active):
        p_watts = np.where(ue_active, np.exp((P_dbm - 30.0) * ln10_div10), 0.0)
        rx_tot = G_eff @ p_watts
        rx_util = diag_G * p_watts
        interf = rx_tot - rx_util
        sinr_lin = np.where(ue_active & (p_watts > 0.0), rx_util / (interf + P_n), 0.0)
        sinr_db_chk = np.where(
            ue_active,
            10.0 * np.log10(np.maximum(sinr_lin, 1e-30)),
            np.inf,
        )
        worst_k = int(np.argmin(sinr_db_chk))
        if sinr_db_chk[worst_k] < gmin_f:
            ue_active[worst_k] = False
            post_shutoff_changed = True
        else:
            break

    # Potência e SINR finais consistentes com o estado pós-desligamento (0 W e 0.0 para inativos)
    p_watts_final = np.where(ue_active, np.exp((P_dbm - 30.0) * ln10_div10), 0.0)
    if np.any(ue_active):
        rx_tot = G_eff @ p_watts_final
        rx_util = diag_G * p_watts_final
        interf = rx_tot - rx_util
        sinr_lin = np.where(
            ue_active & (p_watts_final > 0.0),
            rx_util / (interf + P_n),
            0.0,
        )
    else:
        sinr_lin = np.zeros(K, dtype=float)

    if return_history:
        final_state_mismatch = (
            len(hist["power_w"]) == 0
            or (not np.array_equal(hist["power_w"][-1] > 0.0, ue_active))
        )
        if post_shutoff_changed or final_state_mismatch:
            hist["power_w"].append(p_watts_final.copy())
            hist["sinr_lin"].append(sinr_lin.copy())
            hist["outer_marks"].append(len(hist["power_w"]))
            hist["target_db"].append(gamma_tar_db.copy())
            hist["active"].append(ue_active.copy())
        hist["power_w"] = np.array(hist["power_w"])
        hist["sinr_lin"] = np.array(hist["sinr_lin"])
        hist["target_db"] = np.array(hist["target_db"])
        hist["active"] = np.array(hist["active"])
        hist["p_max_dbm"] = p_max_f
        hist["p_min_dbm"] = p_min_f
        hist["mu_step"] = mu_f
        hist["gamma_min_db"] = gmin_f
        hist["gamma_max_db"] = gmax_f
        hist["tol_db"] = tol_f
        return p_watts_final, sinr_lin, hist
    return p_watts_final, sinr_lin

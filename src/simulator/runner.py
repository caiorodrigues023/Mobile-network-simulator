# ---------------------------------------------------------------------------
# runner.py — Motor do experimento: orquestra rodadas e distribui APs
# ---------------------------------------------------------------------------

from typing import Optional

import numpy as np
from tqdm import tqdm

from .allocation import DPC_Dual_Loop, attach_AP_UE
from .antennas import antenna_gain
from .channel import adaptive_gains, angle_matrix, gain_matrix, gerar_shadowing
from .config import (
    B_TOTAL,
    GAMMA_TAR_INIT,
    K_0,
    MU_STEP,
    P_MAX_DBM,
    P_MIN_DBM,
)

# 1. RESTAURANDO OS IMPORTS APAGADOS
from .entities import AP, UE
from .metrics import SINR, SINR_adaptive, channel_capacity, evaluate_physical_kpis

# ---------------------------------------------------------------------------
# Posicionamento de APs em grade uniforme
# ---------------------------------------------------------------------------


def distribuir_AP(M: int) -> list:
    """Posiciona M APs em uma grade uniforme dentro da área 1000 × 1000 m."""
    APs = []
    dx = 1000 / (2 * np.sqrt(M))
    a = np.arange(dx, 1001 - dx, 2 * dx)
    x, y = np.meshgrid(a, a)
    for id_, (xi, yi) in enumerate(zip(x.ravel(), y.ravel())):
        APs.append(AP(xi, yi, id_))
    return APs


# ---------------------------------------------------------------------------
# Geração de cenário para visualização
# ---------------------------------------------------------------------------


def cenario_aleatorio(M: int, K: int, seed: Optional[int] = None) -> tuple:  # noqa
    """
    Gera um cenário (aps, ues, shadowing) reproduzível para visualização.

    Parâmetros
    ----------
    seed : semente do gerador aleatório (None = não fixa).

    Retorna: (aps, ues, shadowing).
    """
    if seed is not None:
        np.random.seed(seed)
    aps = distribuir_AP(M)
    UE.id_counter = 0
    ues = [UE(aps) for _ in range(K)]
    shadowing = gerar_shadowing(K, M)
    return aps, ues, shadowing


# ---------------------------------------------------------------------------
# Experimento de Monte Carlo
# ---------------------------------------------------------------------------


def simular_experimento(
    M: int,
    N: int,
    K: int,
    sim: int,
    allocation: str = "",
    antenna_models: tuple = (
        "omni",
        "adaptive",
        "adaptive_1",
        "adaptive_5",
        "adaptive_10",
    ),
    shadowing_params: Optional[dict] = None,  # noqa
    use_rayleigh=False,
    L: int = 8,
    J: int = 8,
) -> dict:
    """
    Executa 'sim' iterações de Monte Carlo para os modelos de antena indicados.

    Parâmetros
    ----------
    M               : número de APs.
    N               : número de canais ortogonais.
    K               : número de UEs por iteração.
    sim             : número de iterações de Monte Carlo.
    allocation      : modo de alocação de canal ('random' ou ortogonal).
    antenna_models  : tupla de modelos a simular.
    L, J            : parâmetros do arranjo SMB/MRC.
    use_rayleigh    : se True, utiliza fading de Rayleigh.

    Retorna dicionário:
        {model: {'power': [...], 'sinr': [...], 'cap': [...], 'sum_cap': média}}
    """
    aps = distribuir_AP(M)
    resultados = {
        model: {"power": [], "sinr": [], "cap": [], "sum_cap": []}
        for model in antenna_models
    }

    for _ in tqdm(range(sim)):
        UE.id_counter = 0
        ues = [UE(aps) for _ in range(K)]
        shadowing = gerar_shadowing(ues, aps)

        for model in antenna_models:
            # Reseta estado de associação
            for ue in ues:
                ue.ap, ue.beam, ue.channel = None, 0, 0
            for ap in aps:
                ap.ues = []

            G = antenna_gain(ues, aps, model=model, L=L, J=J)
            ue_xy = np.array([[ue.x, ue.y] for ue in ues], dtype=float)
            ap_xy = np.array([[ap.x, ap.y] for ap in aps], dtype=float)

            gains = gain_matrix(
                ue_xy, ap_xy, G, shadowing=shadowing, use_rayleigh=use_rayleigh
            )

            serving_ap, serving_beam = attach_AP_UE(gains)
            for i, ue in enumerate(ues):
                ue.ap = aps[serving_ap[i]]
                ue.beam = int(serving_beam[i])
                ue.gain = gains[i, serving_ap[i], serving_beam[i]]
                ue.ap.ues.append(ue)

            # --- Lógica adaptativa (com extração de erro de AoA) ---
            if model.startswith("adaptive"):
                err_deg = 0.0
                if "_" in model:
                    try:
                        err_deg = float(model.split("_")[1])
                    except ValueError:
                        pass

                Gm = adaptive_gains(ues, aps, L=L, aoa_err_deg=err_deg)
                s = SINR_adaptive(ues, N, gains, Gm)
                power = [ue.gain * ue.power * Gm[k, k] for k, ue in enumerate(ues)]
            else:
                s = SINR(ues, N, gains, G)
                power = [ue.gain * ue.power for ue in ues]

            cap = channel_capacity(s, N)
            sum_cap = float(np.sum(cap))

            resultados[model]["power"].extend(power)
            resultados[model]["sinr"].extend(s)
            resultados[model]["cap"].extend(cap)
            resultados[model]["sum_cap"].append(sum_cap)

            for ap in aps:
                ap.ues = []

    # Média da soma de capacidades ao longo das iterações
    for model in antenna_models:
        resultados[model]["sum_cap"] = float(np.mean(resultados[model]["sum_cap"]))

    return resultados


def simular_scichal4a(M: int, N: int, K: int, sim: int) -> dict:
    """
    Executa a simulação Monte Carlo para o SciChal4A usando Pure Array Computing.
    """
    aps = distribuir_AP(M)
    ap_xy = np.array([[ap.x, ap.y] for ap in aps])

    B_c = B_TOTAL / N
    P_n = B_c * K_0

    algorithms = ["full_power", "dpc_fixed", "dual_loop"]

    # ADICIONADO: Chaves 'sinr_lin' e 'power_dist' para as CDFs
    stats = {
        algo: {
            "se": [],
            "sum_cap": [],
            "power": [],
            "outage": [],
            "sinr_lin": [],
            "power_dist": [],
        }
        for algo in algorithms
    }

    for _ in tqdm(range(sim), desc=f"Monte Carlo K={K}"):
        ue_xy = np.random.uniform(0, 1000, size=(K, 2))

        G_omni = np.ones((K, M, 1))
        gains_base = gain_matrix(ue_xy, ap_xy, G_omni, use_rayleigh=True)
        serving_ap, _ = attach_AP_UE(gains_base)

        phi = angle_matrix(ue_xy, ap_xy)
        theta_tgt = phi[np.arange(K), serving_ap]
        theta_arr = phi[:, serving_ap].T

        L_elements = 8
        R_uca = L_elements * 1.0 / (4 * np.pi)
        phi_elements = np.arange(L_elements) * (2 * np.pi / L_elements)

        phase_tgt = (
            (2 * np.pi) * R_uca * np.cos(theta_tgt[:, None] - phi_elements[None, :])
        )
        phase_arr = (
            (2 * np.pi)
            * R_uca
            * np.cos(theta_arr[:, :, None] - phi_elements[None, None, :])
        )

        Gm = (
            np.abs(
                np.sum(np.exp(1j * (phase_arr - phase_tgt[:, None, :])), axis=-1)
                / np.sqrt(L_elements)
            )
            ** 2
        )

        pathloss_matrix = gains_base[:, serving_ap, 0].T
        G_eff = pathloss_matrix * Gm

        # -----------------------------------------------------------------
        # ALGORITMO 1: Full Power
        # -----------------------------------------------------------------
        p_watts_fp = np.full(K, 10 ** (P_MAX_DBM / 10) / 1000)
        rx_tot = G_eff @ p_watts_fp
        rx_util = np.diagonal(G_eff) * p_watts_fp
        sinr_fp = np.where(p_watts_fp > 0, rx_util / (rx_tot - rx_util + P_n), 0.0)

        se, _, sum_cap, tot_p, out = evaluate_physical_kpis(sinr_fp, p_watts_fp, B_c)
        stats["full_power"]["se"].extend(se)
        stats["full_power"]["sum_cap"].append(sum_cap)
        stats["full_power"]["power"].append(tot_p)
        stats["full_power"]["outage"].extend(out)

        # Salvando dados brutos para as CDFs
        stats["full_power"]["sinr_lin"].extend(sinr_fp)
        stats["full_power"]["power_dist"].extend(p_watts_fp)

        # -----------------------------------------------------------------
        # ALGORITMO 2: DPC Fixo (10 dB)
        # -----------------------------------------------------------------
        P_dbm_dpc = np.full(K, P_MAX_DBM)
        gamma_tar_lin = 10.0 ** (GAMMA_TAR_INIT / 10.0)
        for _ in range(150):
            p_w_dpc = 10 ** (P_dbm_dpc / 10.0) / 1000.0
            r_tot = G_eff @ p_w_dpc
            r_util = np.diagonal(G_eff) * p_w_dpc
            s_dpc = r_util / (r_tot - r_util + P_n)
            direcao = np.sign(gamma_tar_lin - s_dpc)
            P_dbm_dpc = np.clip(P_dbm_dpc + MU_STEP * direcao, P_MIN_DBM, P_MAX_DBM)

        p_watts_dpc = 10 ** (P_dbm_dpc / 10.0) / 1000.0
        rx_tot = G_eff @ p_watts_dpc
        rx_util = np.diagonal(G_eff) * p_watts_dpc
        sinr_dpc = rx_util / (rx_tot - rx_util + P_n)

        se, _, sum_cap, tot_p, out = evaluate_physical_kpis(sinr_dpc, p_watts_dpc, B_c)
        stats["dpc_fixed"]["se"].extend(se)
        stats["dpc_fixed"]["sum_cap"].append(sum_cap)
        stats["dpc_fixed"]["power"].append(tot_p)
        stats["dpc_fixed"]["outage"].extend(out)
        stats["dpc_fixed"]["sinr_lin"].extend(sinr_dpc)
        stats["dpc_fixed"]["power_dist"].extend(p_watts_dpc)

        # -----------------------------------------------------------------
        # ALGORITMO 3: Dual Loop PC Vetorizado
        # -----------------------------------------------------------------
        gamma_tar_db = np.full(K, GAMMA_TAR_INIT)
        ue_active = np.ones(K, dtype=bool)

        p_watts_dual, sinr_lin_dual = DPC_Dual_Loop(
            G_eff, P_n, gamma_tar_db, ue_active, t_outer=35, t_inner=15, tol_db=0.5
        )

        se, _, sum_cap, tot_p, out = evaluate_physical_kpis(
            sinr_lin_dual, p_watts_dual, B_c
        )
        stats["dual_loop"]["se"].extend(se)
        stats["dual_loop"]["sum_cap"].append(sum_cap)
        stats["dual_loop"]["power"].append(tot_p)
        stats["dual_loop"]["outage"].extend(out)
        stats["dual_loop"]["sinr_lin"].extend(sinr_lin_dual)
        stats["dual_loop"]["power_dist"].extend(p_watts_dual)

    return stats

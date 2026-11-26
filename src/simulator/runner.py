# ---------------------------------------------------------------------------
# runner.py — Motor do experimento: orquestra rodadas e distribui APs
# ---------------------------------------------------------------------------

import sys
from typing import Optional

import numpy as np
from tqdm import tqdm

from .allocation import DPC_Dual_Loop, alocar_canais_ortogonal, attach_AP_UE
from .antennas import antenna_gain
from .channel import adaptive_gains, angle_matrix, gain_matrix, gerar_shadowing
from .config import (
    B_TOTAL,
    D_0,
    GAMMA_MAX_DB,
    GAMMA_MIN_DB,
    GAMMA_TAR_INIT,
    K_0,
    K_CONST,
    MU_STEP,
    PATHLOSS_EXP,
    P_MAX_DBM,
    P_MIN_DBM,
    SIGMA_R,
    SIGMA_X,
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


def _build_effective_gain_matrix(
    ues: list, gains: np.ndarray, Gm: np.ndarray = None, is_adaptive: bool = False
) -> np.ndarray:
    """Constrói matriz K×K de ganho co-canal efetivo vista no AP servidor de cada UE."""
    K = len(ues)
    if K == 0:
        return np.zeros((0, 0), dtype=float)

    ap_ids = np.array([ue.ap.id for ue in ues], dtype=int)
    channels = np.array([ue.channel for ue in ues])
    if is_adaptive:
        G_raw = gains[:, ap_ids, 0].T * Gm
    else:
        beams = np.array([ue.beam for ue in ues], dtype=int)
        G_raw = gains[:, ap_ids, beams].T

    same_ch = channels[:, None] == channels[None, :]
    return np.where(same_ch, G_raw, 0.0)


def _fast_dpc_fixed(
    G_eff: np.ndarray,
    P_n: float,
    gamma_tar_db: float,
    p_max_dbm: float,
    p_min_dbm: float,
    mu_step: float,
    t_inner: int,
) -> np.ndarray:
    """Executa DPC de alvo fixo em ±mu_step dB com detecção rápida de ponto fixo ou ciclo-2."""
    K = G_eff.shape[0]
    if K == 0:
        return np.zeros(0, dtype=float)

    diag_G = np.diagonal(G_eff).copy()
    ln10_div10 = float(np.log(10.0) / 10.0)
    p_max_f = float(p_max_dbm)
    p_min_f = float(p_min_dbm)
    mu_f = float(mu_step)
    gamma_tar_lin = float(np.exp(float(gamma_tar_db) * ln10_div10))

    P_dbm = np.full(K, p_max_f, dtype=float)
    P_prev1 = None
    P_prev2 = None

    for step_idx in range(t_inner):
        if P_prev2 is not None and np.array_equal(P_dbm, P_prev2):
            if (t_inner - step_idx) % 2 == 1:
                P_dbm = P_prev1
            break

        p_w = np.exp((P_dbm - 30.0) * ln10_div10)
        rx_tot = G_eff @ p_w
        rx_util = diag_G * p_w
        s_est = rx_util / (rx_tot - rx_util + P_n)
        P_next = np.clip(P_dbm + mu_f * np.sign(gamma_tar_lin - s_est), p_min_f, p_max_f)
        if np.array_equal(P_next, P_dbm):
            break

        P_prev2 = P_prev1
        P_prev1 = P_dbm
        P_dbm = P_next

    return np.exp((P_dbm - 30.0) * ln10_div10)


def aplicar_controle_potencia(
    ues: list,
    N: int,
    gains: np.ndarray,
    G: np.ndarray,
    Gm: np.ndarray = None,
    is_adaptive: bool = False,
    power_mode: str = "full_power",
    b_total: float = B_TOTAL,
    k_0: float = K_0,
    p_max_dbm: float = P_MAX_DBM,
    p_min_dbm: float = P_MIN_DBM,
    mu_step: float = MU_STEP,
    gamma_tar_db: float = GAMMA_TAR_INIT,
    gamma_max_db: float = GAMMA_MAX_DB,
    gamma_min_db: float = GAMMA_MIN_DB,
    t_outer: int = 50,
    t_inner: int = 150,
    tol_db: float = 0.5,
    return_history: bool = False,
) -> np.ndarray | tuple:
    """
    Aplica `power_mode` ('full_power', 'dpc_fixed' ou 'dual_loop'), atualiza `ue.power`
    e retorna `sinr_lin` (0.0 apenas para UEs desligados no 'dual_loop').
    Se `return_history=True`, retorna `(sinr_lin, hist_or_none)`.
    """
    K = len(ues)
    if K == 0:
        empty = np.zeros(0, dtype=float)
        return (empty, None) if return_history else empty

    Pn = k_0 * (b_total / N)
    p_max_w = 10.0 ** (p_max_dbm / 10.0) / 1000.0

    if power_mode == "dpc_fixed":
        G_eff = _build_effective_gain_matrix(ues, gains, Gm=Gm, is_adaptive=is_adaptive)
        p_watts_dpc = _fast_dpc_fixed(
            G_eff, Pn, gamma_tar_db, p_max_dbm, p_min_dbm, mu_step, t_inner
        )
        for i, ue in enumerate(ues):
            ue.power = float(p_watts_dpc[i])

    elif power_mode == "dual_loop":
        G_eff = _build_effective_gain_matrix(ues, gains, Gm=Gm, is_adaptive=is_adaptive)
        gamma_tar_vec = np.full(K, gamma_tar_db, dtype=float)
        ue_active_init = np.ones(K, dtype=bool)
        dl_out = DPC_Dual_Loop(
            G_eff,
            Pn,
            gamma_tar_vec,
            ue_active_init,
            t_outer=t_outer,
            t_inner=t_inner,
            tol_db=tol_db,
            return_history=return_history,
            p_max_dbm=p_max_dbm,
            p_min_dbm=p_min_dbm,
            mu_step=mu_step,
            gamma_max_db=gamma_max_db,
            gamma_min_db=gamma_min_db,
        )
        if return_history:
            p_watts_dual, sinr_dual, hist = dl_out
        else:
            p_watts_dual, sinr_dual = dl_out
            hist = None
        for i, ue in enumerate(ues):
            ue.power = float(p_watts_dual[i])
        sinr_arr = np.asarray(sinr_dual, dtype=float).flatten()
        return (sinr_arr, hist) if return_history else sinr_arr

    else:
        for ue in ues:
            ue.power = float(p_max_w)

    sinr_arr = np.asarray(
        SINR_adaptive(ues, N, gains, Gm, b_total=b_total, k0=k_0)
        if is_adaptive
        else SINR(ues, N, gains, G, b_total=b_total, k0=k_0),
        dtype=float,
    ).flatten()
    return (sinr_arr, None) if return_history else sinr_arr


def simular_experimento(
    M: int,
    N: int,
    K: int,
    sim: int,
    allocation: str = "ortogonal",
    antenna_models: tuple = (
        "omni",
        "adaptive",
        "adaptive_1",
        "adaptive_5",
        "adaptive_10",
    ),
    shadowing_params: Optional[dict] = None,  # noqa
    use_rayleigh: bool = False,
    L: int = 8,
    J: int = 8,
    beamwidth_deg: float = 120.0,
    g_min: float = 0.01,
    aoa_err_deg: float = 0.0,
    b_total: float = B_TOTAL,
    k_0: float = K_0,
    pathloss_exp: float = PATHLOSS_EXP,
    k_const: float = K_CONST,
    d_0: float = D_0,
    sigma_x: float = SIGMA_X,
    sigma_r: float = SIGMA_R,
    p_max_dbm: float = P_MAX_DBM,
    power_mode: str = "full_power",
    p_min_dbm: float = P_MIN_DBM,
    mu_step: float = MU_STEP,
    gamma_tar_db: float = GAMMA_TAR_INIT,
    gamma_max_db: float = GAMMA_MAX_DB,
    gamma_min_db: float = GAMMA_MIN_DB,
    t_outer: int = 50,
    t_inner: int = 150,
    tol_db: float = 0.5,
    seed: Optional[int] = None,
) -> dict:
    """
    Executa 'sim' iterações de Monte Carlo para os modelos de antena indicados.
    Suporta 'full_power', 'dpc_fixed' e 'dual_loop' (único que desliga UEs em outage).
    """
    if seed is not None and seed >= 0:
        np.random.seed(seed)

    aps = distribuir_AP(M)
    p_max_w = 10.0 ** (p_max_dbm / 10.0) / 1000.0
    resultados = {
        model: {"power": [], "tx_power": [], "sinr": [], "cap": [], "sum_cap": []}
        for model in antenna_models
    }

    for _ in tqdm(range(sim)):
        UE.id_counter = 0
        ues = [UE(aps) for _ in range(K)]
        shadowing = gerar_shadowing(ues, aps, sigma_x=sigma_x)

        for model in antenna_models:
            # Reseta estado de associação e potência padrão
            for ue in ues:
                ue.ap, ue.beam, ue.channel, ue.power = None, 0, 0, p_max_w
            for ap in aps:
                ap.ues = []

            G = antenna_gain(
                ues,
                aps,
                model=model,
                beamwidth=np.deg2rad(beamwidth_deg),
                g_min=g_min,
                L=L,
                J=J,
            )
            ue_xy = np.array([[ue.x, ue.y] for ue in ues], dtype=float)
            ap_xy = np.array([[ap.x, ap.y] for ap in aps], dtype=float)

            gains = gain_matrix(
                ue_xy,
                ap_xy,
                G,
                shadowing=shadowing,
                use_rayleigh=use_rayleigh,
                pathloss_exp=pathloss_exp,
                k_const=k_const,
                d_0=d_0,
                sigma_x=sigma_x,
                sigma_r=sigma_r,
            )

            serving_ap, serving_beam = attach_AP_UE(gains)
            for i, ue in enumerate(ues):
                ue.ap = aps[serving_ap[i]]
                ue.beam = int(serving_beam[i])
                ue.gain = gains[i, serving_ap[i], serving_beam[i]]
                ue.ap.ues.append(ue)

            alocar_canais_ortogonal(aps, ues, N, allocation)

            is_adaptive = model.startswith("adaptive")
            if is_adaptive:
                err_deg = float(aoa_err_deg)
                if "_" in model:
                    try:
                        err_deg = float(model.split("_")[1])
                    except ValueError:
                        pass
                Gm = adaptive_gains(ues, aps, L=L, aoa_err_deg=err_deg)
            else:
                Gm = None

            s = aplicar_controle_potencia(
                ues=ues,
                N=N,
                gains=gains,
                G=G,
                Gm=Gm,
                is_adaptive=is_adaptive,
                power_mode=power_mode,
                b_total=b_total,
                k_0=k_0,
                p_max_dbm=p_max_dbm,
                p_min_dbm=p_min_dbm,
                mu_step=mu_step,
                gamma_tar_db=gamma_tar_db,
                gamma_max_db=gamma_max_db,
                gamma_min_db=gamma_min_db,
                t_outer=t_outer,
                t_inner=t_inner,
                tol_db=tol_db,
            )
            if is_adaptive:
                power = [
                    ue.gain * ue.power * Gm[k, k] if ue.power > 0.0 else 0.0
                    for k, ue in enumerate(ues)
                ]
            else:
                power = [
                    ue.gain * ue.power if ue.power > 0.0 else 0.0 for ue in ues
                ]

            tx_power_tot = float(sum(ue.power for ue in ues))
            cap = np.asarray(channel_capacity(s, N, b_total=b_total)).flatten()
            if power_mode == "dual_loop":
                cap = np.where(np.array([ue.power > 0.0 for ue in ues]), cap, 0.0)
            sum_cap = float(np.sum(cap))

            resultados[model]["power"].extend(power)
            resultados[model]["tx_power"].append(tx_power_tot)
            resultados[model]["sinr"].extend(s.tolist())
            resultados[model]["cap"].extend(cap.tolist())
            resultados[model]["sum_cap"].append(sum_cap)

            for ap in aps:
                ap.ues = []

    # Média da soma de capacidades e potência Tx ao longo das iterações
    for model in antenna_models:
        resultados[model]["sum_cap"] = float(np.mean(resultados[model]["sum_cap"]))
        resultados[model]["tx_power"] = float(np.mean(resultados[model]["tx_power"]))

    return resultados


def simular_scichal4a(
    M: int,
    N: int,
    K: int,
    sim: int,
    n_hist: int = 5,
    L: int = 8,
    use_rayleigh: bool = True,
    b_total: float = B_TOTAL,
    k_0: float = K_0,
    pathloss_exp: float = PATHLOSS_EXP,
    k_const: float = K_CONST,
    d_0: float = D_0,
    sigma_x: float = SIGMA_X,
    sigma_r: float = SIGMA_R,
    p_max_dbm: float = P_MAX_DBM,
    p_min_dbm: float = P_MIN_DBM,
    mu_step: float = MU_STEP,
    gamma_tar_init: float = GAMMA_TAR_INIT,
    gamma_max_db: float = GAMMA_MAX_DB,
    gamma_min_db: float = GAMMA_MIN_DB,
    t_outer: int = 50,
    t_inner: int = 150,
    tol_db: float = 0.5,
    aoa_err_deg: float = 0.0,
    seed: Optional[int] = None,
    progress_cb=None,
) -> dict:
    """
    Executa a simulação Monte Carlo para o SciChal4A usando Pure Array Computing.
    Apenas o algoritmo 'dual_loop' desliga usuários em outage; 'full_power' e
    'dpc_fixed' operam normalmente mantendo todos os transmissores ativos.
    """
    if seed is not None and seed >= 0:
        np.random.seed(seed)

    aps = distribuir_AP(M)
    ap_xy = np.array([[ap.x, ap.y] for ap in aps])

    B_c = b_total / N
    P_n = B_c * k_0

    algorithms = ["full_power", "dpc_fixed", "dual_loop"]

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
    stats["dual_loop_hist"] = []

    L_elements = int(L)
    R_uca = L_elements * 1.0 / (4 * np.pi)
    phi_elements = np.arange(L_elements) * (2 * np.pi / L_elements)

    tqdm_stream = sys.stdout if sys.stdout is not None else sys.stderr
    step_cb = max(1, sim // 10)
    for sim_idx in tqdm(
        range(sim),
        desc=f"Monte Carlo K={K}",
        disable=(tqdm_stream is None),
        file=tqdm_stream,
    ):
        ue_xy = np.random.uniform(0, 1000, size=(K, 2))

        G_omni = np.ones((K, M, 1))
        gains_base = gain_matrix(
            ue_xy,
            ap_xy,
            G_omni,
            use_rayleigh=use_rayleigh,
            pathloss_exp=pathloss_exp,
            k_const=k_const,
            d_0=d_0,
            sigma_x=sigma_x,
            sigma_r=sigma_r,
        )
        serving_ap, _ = attach_AP_UE(gains_base)

        phi = angle_matrix(ue_xy, ap_xy)
        theta_tgt = phi[np.arange(K), serving_ap]
        if aoa_err_deg > 0.0:
            theta_hat = theta_tgt + np.deg2rad(aoa_err_deg) * np.random.randn(K)
        else:
            theta_hat = theta_tgt
        theta_arr = phi[:, serving_ap].T

        phase_tgt = (
            (2 * np.pi) * R_uca * np.cos(theta_hat[:, None] - phi_elements[None, :])
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
        # ALGORITMO 1: Full Power (sem desligar transmissores)
        # -----------------------------------------------------------------
        p_watts_fp = np.full(K, 10.0 ** (p_max_dbm / 10.0) / 1000.0)
        rx_tot = G_eff @ p_watts_fp
        rx_util = np.diagonal(G_eff) * p_watts_fp
        sinr_fp = np.where(p_watts_fp > 0, rx_util / (rx_tot - rx_util + P_n), 0.0)

        se, _, sum_cap, tot_p, out = evaluate_physical_kpis(
            sinr_fp,
            p_watts_fp,
            B_c,
            gamma_min_db=gamma_min_db,
            gamma_max_db=gamma_max_db,
        )
        stats["full_power"]["se"].extend(se)
        stats["full_power"]["sum_cap"].append(sum_cap)
        stats["full_power"]["power"].append(tot_p)
        stats["full_power"]["outage"].extend(out)
        stats["full_power"]["sinr_lin"].extend(sinr_fp)
        stats["full_power"]["power_dist"].extend(p_watts_fp)

        # -----------------------------------------------------------------
        # ALGORITMO 2: DPC Fixo (sem desligar transmissores)
        # -----------------------------------------------------------------
        p_watts_dpc = _fast_dpc_fixed(
            G_eff, P_n, gamma_tar_init, p_max_dbm, p_min_dbm, mu_step, t_inner
        )
        rx_tot = G_eff @ p_watts_dpc
        rx_util = np.diagonal(G_eff) * p_watts_dpc
        sinr_dpc = rx_util / (rx_tot - rx_util + P_n)

        se, _, sum_cap, tot_p, out = evaluate_physical_kpis(
            sinr_dpc,
            p_watts_dpc,
            B_c,
            gamma_min_db=gamma_min_db,
            gamma_max_db=gamma_max_db,
        )
        stats["dpc_fixed"]["se"].extend(se)
        stats["dpc_fixed"]["sum_cap"].append(sum_cap)
        stats["dpc_fixed"]["power"].append(tot_p)
        stats["dpc_fixed"]["outage"].extend(out)
        stats["dpc_fixed"]["sinr_lin"].extend(sinr_dpc)
        stats["dpc_fixed"]["power_dist"].extend(p_watts_dpc)

        # -----------------------------------------------------------------
        # ALGORITMO 3: Dual Loop PC (Único que desliga usuários em outage)
        # -----------------------------------------------------------------
        gamma_tar_db = np.full(K, float(gamma_tar_init))
        ue_active = np.ones(K, dtype=bool)

        want_hist = sim_idx < n_hist
        out_dl = DPC_Dual_Loop(
            G_eff,
            P_n,
            gamma_tar_db,
            ue_active,
            t_outer=t_outer,
            t_inner=t_inner,
            tol_db=tol_db,
            return_history=want_hist,
            p_max_dbm=p_max_dbm,
            p_min_dbm=p_min_dbm,
            mu_step=mu_step,
            gamma_max_db=gamma_max_db,
            gamma_min_db=gamma_min_db,
        )
        if want_hist:
            p_watts_dual, sinr_lin_dual, hist = out_dl
            hist["K"] = K
            hist["snapshot"] = sim_idx
            stats["dual_loop_hist"].append(hist)
        else:
            p_watts_dual, sinr_lin_dual = out_dl

        se, _, sum_cap, tot_p, out = evaluate_physical_kpis(
            sinr_lin_dual,
            p_watts_dual,
            B_c,
            gamma_min_db=gamma_min_db,
            gamma_max_db=gamma_max_db,
        )
        stats["dual_loop"]["se"].extend(se)
        stats["dual_loop"]["sum_cap"].append(sum_cap)
        stats["dual_loop"]["power"].append(tot_p)
        stats["dual_loop"]["outage"].extend(out)
        stats["dual_loop"]["sinr_lin"].extend(sinr_lin_dual)
        stats["dual_loop"]["power_dist"].extend(p_watts_dual)

        if progress_cb is not None and ((sim_idx + 1) % step_cb == 0 or sim_idx == sim - 1):
            progress_cb(sim_idx + 1, sim)

    return stats

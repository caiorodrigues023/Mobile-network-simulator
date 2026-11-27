# ---------------------------------------------------------------------------
# allocation.py — Attach, Alocação de Canais (PAPOA / CA / Random) e
#                 Controle de Potência Vetorizado (Strategy Registry)
# ---------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from .config import (
    B_TOTAL,
    GAMMA_MAX_DB,
    GAMMA_MIN_DB,
    GAMMA_TAR_INIT,
    K_0,
    MU_STEP,
    P_MAX_DBM,
    P_MIN_DBM,
    T_INNER_DEFAULT,
    T_OUTER_DEFAULT,
    TOL_DB,
)
from .metrics import calculate_sinr_vector
from .schemas import PowerControlConfig

_LN10_DIV10: float = float(np.log(10.0) / 10.0)


def _dbm_to_w(p_dbm: np.ndarray) -> np.ndarray:
    return np.exp((p_dbm - 30.0) * _LN10_DIV10)


# ---------------------------------------------------------------------------
# Associação UE → AP (Initial Access — Máximo Path-Gain)
# ---------------------------------------------------------------------------


def attach_AP_UE(gains: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Associa cada UE ao par (AP, feixe/setor) com o MAIOR ganho de canal (Parte 2, Seção 5).

    gains : array (K, M, B)
    Retorna: (serving_ap (K,), serving_beam (K,))
    """
    K = gains.shape[0]
    if K == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    flat_indices = np.argmax(gains.reshape(K, -1), axis=1)
    serving_ap, serving_beam = np.unravel_index(flat_indices, gains.shape[1:])
    return np.asarray(serving_ap, dtype=int), np.asarray(serving_beam, dtype=int)


# ---------------------------------------------------------------------------
# Alocação de Canais (Reuso Total, Random, PAPOA e PAPOA + Carrier Aggregation)
# ---------------------------------------------------------------------------


def allocate_channels_vec(
    serving_ap: np.ndarray,
    M: int,
    N: int,
    policy: str = "papoa",
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aloca canais em 0..N-1 para K UEs de forma determinística (via `rng`) e vetorizada.

    Políticas suportadas (Parte 2, Exercícios 6 e 7):
      - 'full_reuse' : todos os UEs compartilham o canal 0.
      - 'random'     : sorteio uniforme independente em 0..N-1.
      - 'papoa'      : Per-AP Orthogonal Allocation (ortogonal intra-AP com balanceamento
                       por AP quando K_i > N).
      - 'papoa_ca'   : PAPOA + Carrier Aggregation (UE único em um AP com N > 1 agrega
                       todos os N canais com potência dividida P_t / N).

    Retorna: (channels (K,) int em 0..N-1, is_aggregated (K,) bool).
    """
    K = len(serving_ap)
    if K == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=bool)

    if rng is None:
        rng = np.random.default_rng()

    pol = "papoa" if policy == "ortogonal" else policy
    is_aggregated = np.zeros(K, dtype=bool)

    if N <= 1 or pol == "full_reuse":
        return np.zeros(K, dtype=int), is_aggregated

    if pol == "random":
        return rng.integers(0, N, size=K, dtype=int), is_aggregated

    if pol not in ("papoa", "papoa_ca"):
        raise ValueError(f"Política de alocação desconhecida: {policy!r}")

    channels = np.zeros(K, dtype=int)
    for ap_id in range(M):
        attached = np.flatnonzero(serving_ap == ap_id)
        k_i = attached.size
        if k_i == 0:
            continue
        if k_i == 1:
            if pol == "papoa_ca":
                is_aggregated[attached[0]] = True
                channels[attached[0]] = 0
            else:
                channels[attached[0]] = int(rng.integers(0, N))
        elif k_i <= N:
            perm = rng.permutation(N)
            channels[attached] = perm[:k_i]
        else:
            # Balanceamento intra-AP com desempate aleatório (Ex. 6)
            loads = np.zeros(N, dtype=int)
            order = rng.permutation(attached)
            for u in order:
                min_load = int(np.min(loads))
                candidates = np.flatnonzero(loads == min_load)
                chosen = int(candidates[int(rng.integers(0, candidates.size))])
                channels[u] = chosen
                loads[chosen] += 1

    return channels, is_aggregated


def alocar_canais_ortogonal(
    access_points: list,
    ues: list,
    number_channels: int,
    allocation: str,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """
    Wrapper compatível com entidades OO (atualiza `ue.channel` em 1..N in-place).
    """
    if not ues:
        return
    serving_ap = np.array([ue.ap.id for ue in ues], dtype=int)
    policy = "random" if allocation == "random" else "papoa"
    channels, _ = allocate_channels_vec(
        serving_ap, M=len(access_points), N=number_channels, policy=policy, rng=rng
    )
    for idx, ue in enumerate(ues):
        ue.channel = int(channels[idx]) + 1


# ---------------------------------------------------------------------------
# Algoritmos de Controle de Potência — Núcleo Vetorizado (Strategy Pattern)
# ---------------------------------------------------------------------------
# Contrato comum:
#   fn(G_eff, P_n, cfg, record_history=False) -> (p_watts, sinr_lin, gamma_tar_db, hist)


def pc_full_power(
    G_eff: np.ndarray,
    P_n: float,
    cfg: PowerControlConfig,
    record_history: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict[str, Any]]]:
    """Baseline sem controle de potência: todos os UEs transmitem em P_max."""
    _ = record_history
    K = G_eff.shape[0]
    p_w = np.full(K, cfg.p_max_w, dtype=float)
    sinr_lin = calculate_sinr_vector(G_eff, p_w, P_n)
    gamma_tar = np.full(K, cfg.gamma_tar_init_db, dtype=float)
    return p_w, sinr_lin, gamma_tar, None


def channel_inversion_power_control(
    G_eff: np.ndarray,
    P_n: float,
    gamma_tar_db: float = GAMMA_TAR_INIT,
    p_max_w: float = 1.0,
) -> np.ndarray:
    """
    Channel Inversion Power Control (CIPC — Parte 2, Eq. 10):
        p_k = min(gamma_tar * P_n / g_kk, p_max)
    """
    K = G_eff.shape[0]
    if K == 0:
        return np.zeros(0, dtype=float)
    diag_g = np.maximum(np.diagonal(G_eff), 1e-30)
    gamma_lin = 10.0 ** (gamma_tar_db / 10.0)
    p_w = (gamma_lin * P_n) / diag_g
    return np.clip(p_w, 1e-15, p_max_w)


def pc_cipc(
    G_eff: np.ndarray,
    P_n: float,
    cfg: PowerControlConfig,
    record_history: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict[str, Any]]]:
    """Estratégia CIPC (inversão de canal sem considerar interferência)."""
    _ = record_history
    K = G_eff.shape[0]
    p_w = channel_inversion_power_control(
        G_eff, P_n, gamma_tar_db=cfg.gamma_tar_init_db, p_max_w=cfg.p_max_w
    )
    sinr_lin = calculate_sinr_vector(G_eff, p_w, P_n)
    gamma_tar = np.full(K, cfg.gamma_tar_init_db, dtype=float)
    return p_w, sinr_lin, gamma_tar, None


def optimal_sinr_power_control(
    G_eff: np.ndarray,
    P_n: float,
    gamma_tar_db: float = GAMMA_TAR_INIT,
    p_max_w: float = 1.0,
) -> tuple[np.ndarray, bool]:
    """
    Optimal Target-SINR Power Control (Parte 2, Eq. 14–17 e Exercício 11):
      Resolve globalmente (I - A) p = b com A[i,j] = γ_t * G[i,j]/G[i,i] (i ≠ j),
      b[i] = γ_t * P_n / G[i,i].

    Verifica viabilidade via raio espectral ρ(A) < 1 e positividade p > 0.
    Conforme o Exercício 11 do roteiro, se o sistema for inviável (ou K == 1),
    faz fallback para CIPC e limita à potência máxima p_max_w.

    Retorna: (p_watts (K,), is_feasible_exact: bool).
    """
    K = G_eff.shape[0]
    if K == 0:
        return np.zeros(0, dtype=float), True

    p_cipc = channel_inversion_power_control(
        G_eff, P_n, gamma_tar_db=gamma_tar_db, p_max_w=p_max_w
    )
    if K == 1:
        return p_cipc, True

    diag_g = np.maximum(np.diagonal(G_eff), 1e-30)
    gamma_lin = 10.0 ** (gamma_tar_db / 10.0)

    A = gamma_lin * (G_eff / diag_g[:, None])
    np.fill_diagonal(A, 0.0)
    b = (gamma_lin * P_n) / diag_g

    try:
        eigvals = np.linalg.eigvals(A)
        rho = float(np.max(np.abs(eigvals)))
        if not np.isfinite(rho) or rho >= 1.0 - 1e-12:
            return p_cipc, False
        p_opt = np.linalg.solve(np.eye(K) - A, b)
        if np.any(p_opt <= 0.0) or not np.all(np.isfinite(p_opt)):
            return p_cipc, False
        return np.clip(p_opt, 1e-15, p_max_w), bool(np.all(p_opt <= p_max_w))
    except np.linalg.LinAlgError:
        return p_cipc, False


def pc_optimal(
    G_eff: np.ndarray,
    P_n: float,
    cfg: PowerControlConfig,
    record_history: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict[str, Any]]]:
    """Estratégia Optimal Target-SINR PC (solução matricial exata com fallback CIPC)."""
    _ = record_history
    K = G_eff.shape[0]
    p_w, _ = optimal_sinr_power_control(
        G_eff, P_n, gamma_tar_db=cfg.gamma_tar_init_db, p_max_w=cfg.p_max_w
    )
    sinr_lin = calculate_sinr_vector(G_eff, p_w, P_n)
    gamma_tar = np.full(K, cfg.gamma_tar_init_db, dtype=float)
    return p_w, sinr_lin, gamma_tar, None


def pc_dpc_fixed(
    G_eff: np.ndarray,
    P_n: float,
    cfg: PowerControlConfig,
    record_history: bool = False,
    n_iters: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict[str, Any]]]:
    """
    UD-DPC com meta universal fixa (SciChal4A Baseline 2):
        P_dbm(t+1) = clip(P_dbm(t) + μ · sign(γ_tar - γ(t)), P_min, P_max)
    """
    K = G_eff.shape[0]
    if K == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty, empty, None

    steps = int(n_iters) if n_iters is not None else max(150, cfg.t_outer * cfg.t_inner)
    diag_G = np.diagonal(G_eff).copy()
    gamma_tar_lin = float(np.exp(cfg.gamma_tar_init_db * _LN10_DIV10))
    P_dbm = np.full(K, float(cfg.p_max_dbm), dtype=float)

    hist: Optional[dict[str, Any]] = None
    if record_history:
        hist = {
            "algorithm": "dpc_fixed",
            "power_w": [],
            "sinr_lin": [],
            "target_db": [np.full(K, cfg.gamma_tar_init_db, dtype=float)],
            "outer_marks": [],
            "active": [np.ones(K, dtype=bool)],
        }

    P_prev1: Optional[np.ndarray] = None
    P_prev2: Optional[np.ndarray] = None
    cycle_hits = 0

    for _ in range(steps):
        p_w = _dbm_to_w(P_dbm)
        rx_tot = G_eff @ p_w
        rx_util = diag_G * p_w
        sinr_lin = rx_util / (np.maximum(rx_tot - rx_util, 0.0) + P_n)

        if hist is not None:
            hist["power_w"].append(p_w.copy())
            hist["sinr_lin"].append(sinr_lin.copy())

        if P_prev2 is not None and np.array_equal(P_dbm, P_prev2):
            cycle_hits += 1
            if cycle_hits >= 2:
                break
        else:
            cycle_hits = 0

        P_next = np.clip(
            P_dbm + cfg.mu_step_db * np.sign(gamma_tar_lin - sinr_lin),
            cfg.p_min_dbm,
            cfg.p_max_dbm,
        )
        if np.array_equal(P_next, P_dbm):
            break
        P_prev2 = P_prev1
        P_prev1 = P_dbm
        P_dbm = P_next

    p_final = _dbm_to_w(P_dbm)
    sinr_final = calculate_sinr_vector(G_eff, p_final, P_n)
    gamma_tar = np.full(K, cfg.gamma_tar_init_db, dtype=float)

    if hist is not None:
        hist["power_w"] = np.asarray(hist["power_w"], dtype=float)
        hist["sinr_lin"] = np.asarray(hist["sinr_lin"], dtype=float)
        hist["target_db"] = np.asarray(hist["target_db"], dtype=float)
        hist["active"] = np.asarray(hist["active"], dtype=bool)
        hist["p_max_dbm"] = float(cfg.p_max_dbm)
        hist["p_min_dbm"] = float(cfg.p_min_dbm)
        hist["mu_step"] = float(cfg.mu_step_db)
        hist["gamma_min_db"] = float(cfg.gamma_min_db)
        hist["gamma_max_db"] = float(cfg.gamma_max_db)
        hist["tol_db"] = float(cfg.tol_db)

    return p_final, sinr_final, gamma_tar, hist


def DPC_Dual_Loop_Vectorized(
    G_eff: np.ndarray,
    P_n: float,
    gamma_tar_db: np.ndarray,
    ue_active: np.ndarray,
    t_outer: int = T_OUTER_DEFAULT,
    t_inner: int = T_INNER_DEFAULT,
    tol_db: float = TOL_DB,
    record_history: bool = False,
    p_max_dbm: float = P_MAX_DBM,
    p_min_dbm: float = P_MIN_DBM,
    mu_step: float = MU_STEP,
    gamma_max_db: float = GAMMA_MAX_DB,
    gamma_min_db: float = GAMMA_MIN_DB,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict[str, Any]]]:
    """
    Joint MRC & Dual-Loop Power Control 100% vetorizado (fiel ao roteiro SciChal4A, Seção 1.2):
      - Loop Interno (rápido): UD-DPC em passos ±mu_step dB perseguindo γ_tar_k.
        UEs bloqueados (ue_active=False) transmitem 0 W (zero interferência).
      - Loop Externo (lento): negocia γ_tar_k em [γ_min, γ_max]:
          1. Target Increase : SINR_k >= γ_tar_k - tol_db e P_k < P_max ⇒ γ_tar_k += 1 dB (teto γ_max)
          2. Target Decrease : P_k == P_max e SINR_k < γ_tar_k - tol_db ⇒ γ_tar_k -= 1 dB
          3. UE Blocking     : γ_tar_k < γ_min ⇒ ue_active[k] = False (p_k = 0 W, Outage)

    Retorna: (p_watts_final, sinr_lin_final, gamma_tar_db_final, history_or_none).
    """
    K = G_eff.shape[0]
    if K == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty, empty, None

    gamma_tar = np.array(gamma_tar_db, dtype=float, copy=True)
    active = np.array(ue_active, dtype=bool, copy=True)
    P_dbm = np.full(K, float(p_max_dbm), dtype=float)
    diag_G = np.diagonal(G_eff).copy()

    p_max_f = float(p_max_dbm)
    p_min_f = float(p_min_dbm)
    mu_f = float(mu_step)
    tol_f = float(tol_db)
    gmax_f = float(gamma_max_db)
    gmin_f = float(gamma_min_db)

    # Na rodada externa 0, garante passos suficientes para cruzar de P_max até P_min se necessário
    min_full_sweep = int(np.ceil(max(1.0, (p_max_f - p_min_f) / mu_f))) + 2

    hist: Optional[dict[str, Any]] = None
    if record_history:
        hist = {
            "algorithm": "dual_loop",
            "power_w": [],
            "sinr_lin": [],
            "target_db": [gamma_tar.copy()],
            "outer_marks": [],
            "active": [active.copy()],
        }

    for outer_idx in range(t_outer):
        if not np.any(active):
            break

        gamma_tar_lin = np.exp(gamma_tar * _LN10_DIV10)
        inner_limit = max(t_inner, min_full_sweep) if outer_idx == 0 else t_inner

        P_prev1: Optional[np.ndarray] = None
        P_prev2: Optional[np.ndarray] = None
        n_ciclo = 0

        # ---------------- LOOP INTERNO (UD-DPC) ----------------
        for _ in range(inner_limit):
            p_watts = np.where(active, _dbm_to_w(P_dbm), 0.0)
            rx_tot = G_eff @ p_watts
            rx_util = diag_G * p_watts
            interf = np.maximum(rx_tot - rx_util, 0.0)
            sinr_lin = np.where(active & (p_watts > 0.0), rx_util / (interf + P_n), 0.0)

            if hist is not None:
                hist["power_w"].append(p_watts.copy())
                hist["sinr_lin"].append(sinr_lin.copy())

            if P_prev2 is not None and np.array_equal(P_dbm, P_prev2):
                n_ciclo += 1
                if n_ciclo >= 2:
                    break
            else:
                n_ciclo = 0

            direcao = np.sign(gamma_tar_lin - sinr_lin)
            P_next = np.where(
                active,
                np.clip(P_dbm + mu_f * direcao, p_min_f, p_max_f),
                P_dbm,
            )
            if np.array_equal(P_next, P_dbm):
                break
            P_prev2 = P_prev1
            P_prev1 = P_dbm
            P_dbm = P_next

        # ---------------- LOOP EXTERNO (NEGOCIAÇÃO DE ALVO) ----------------
        p_watts = np.where(active, _dbm_to_w(P_dbm), 0.0)
        rx_tot = G_eff @ p_watts
        rx_util = diag_G * p_watts
        interf = np.maximum(rx_tot - rx_util, 0.0)
        sinr_lin = np.where(active & (p_watts > 0.0), rx_util / (interf + P_n), 0.0)
        sinr_db = 10.0 * np.log10(np.maximum(sinr_lin, 1e-30))

        old_tar = gamma_tar.copy()
        old_active = active.copy()

        # Regra 1: Target Increase (atingiu meta com folga P < P_max)
        can_up = active & (sinr_db >= gamma_tar - tol_f) & (P_dbm < p_max_f - 1e-9)
        # Regra 2: Target Decrease (saturado em P_max sem atingir a meta)
        need_down = (
            active
            & (~can_up)
            & (P_dbm >= p_max_f - 1e-9)
            & (sinr_db < gamma_tar - tol_f)
        )

        gamma_tar = np.where(can_up, np.minimum(gmax_f, gamma_tar + 1.0), gamma_tar)
        gamma_tar = np.where(need_down, gamma_tar - 1.0, gamma_tar)

        # Regra 3: UE Blocking (meta negociada caiu abaixo de gamma_min_db)
        blocked_now = active & (gamma_tar < gmin_f)
        if np.any(blocked_now):
            active = active & (~blocked_now)

        if hist is not None:
            hist["outer_marks"].append(len(hist["power_w"]))
            hist["target_db"].append(gamma_tar.copy())
            hist["active"].append(active.copy())

        if np.array_equal(gamma_tar, old_tar) and np.array_equal(active, old_active):
            break

    # Avaliação final rigorosamente consistente com `active` (UEs bloqueados em 0 W)
    p_final = np.where(active, _dbm_to_w(P_dbm), 0.0)
    sinr_final = calculate_sinr_vector(G_eff, p_final, P_n)

    if hist is not None:
        # Registra estado final pós-desligamento se último passo interno ainda tinha UE bloqueado
        if len(hist["power_w"]) == 0 or not np.array_equal(hist["power_w"][-1] > 0.0, active):
            hist["power_w"].append(p_final.copy())
            hist["sinr_lin"].append(sinr_final.copy())
            hist["outer_marks"].append(len(hist["power_w"]))
            hist["target_db"].append(gamma_tar.copy())
            hist["active"].append(active.copy())
        hist["power_w"] = np.asarray(hist["power_w"], dtype=float)
        hist["sinr_lin"] = np.asarray(hist["sinr_lin"], dtype=float)
        hist["target_db"] = np.asarray(hist["target_db"], dtype=float)
        hist["active"] = np.asarray(hist["active"], dtype=bool)
        hist["p_max_dbm"] = p_max_f
        hist["p_min_dbm"] = p_min_f
        hist["mu_step"] = mu_f
        hist["gamma_min_db"] = gmin_f
        hist["gamma_max_db"] = gmax_f
        hist["tol_db"] = tol_f

    return p_final, sinr_final, gamma_tar, hist


def pc_dual_loop(
    G_eff: np.ndarray,
    P_n: float,
    cfg: PowerControlConfig,
    record_history: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict[str, Any]]]:
    """Estratégia Dual-Loop PC conforme o roteiro SciChal4A."""
    K = G_eff.shape[0]
    gamma_init = np.full(K, float(cfg.gamma_tar_init_db), dtype=float)
    ue_active = np.ones(K, dtype=bool)
    return DPC_Dual_Loop_Vectorized(
        G_eff=G_eff,
        P_n=P_n,
        gamma_tar_db=gamma_init,
        ue_active=ue_active,
        t_outer=cfg.t_outer,
        t_inner=cfg.t_inner,
        tol_db=cfg.tol_db,
        record_history=record_history,
        p_max_dbm=cfg.p_max_dbm,
        p_min_dbm=cfg.p_min_dbm,
        mu_step=cfg.mu_step_db,
        gamma_max_db=cfg.gamma_max_db,
        gamma_min_db=cfg.gamma_min_db,
    )


def pc_max_sum(
    G_eff: np.ndarray,
    P_n: float,
    cfg: PowerControlConfig,
    record_history: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict[str, Any]]]:
    """
    MaxSum Power Control via gradiente projetado (Parte 3 — rrm.py, Eq. 17):
        ∂(Σ log(1 + SINR_i)) / ∂p_k ∝ G[k,k] / (I_k + P_n + G[k,k] p_k)
                                      - Σ_{i≠k} SINR_i · G[i,k] / (I_i + P_n + G[i,i] p_i)
    Usa passo normalizado para estabilidade em regimes limitados por ruído e interferência.
    """
    K = G_eff.shape[0]
    if K == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty, empty, None

    p_min_w = cfg.p_min_w
    p_max_w = cfg.p_max_w
    p = np.full(K, 0.5 * (p_min_w + p_max_w), dtype=float)
    diag_G = np.diagonal(G_eff).copy()
    G_off = G_eff.copy()
    np.fill_diagonal(G_off, 0.0)

    hist: Optional[dict[str, Any]] = None
    if record_history:
        hist = {
            "algorithm": "max_sum",
            "power_w": [],
            "sinr_lin": [],
            "target_db": [np.full(K, cfg.gamma_tar_init_db, dtype=float)],
            "outer_marks": [],
            "active": [np.ones(K, dtype=bool)],
        }

    for it in range(cfg.sgd_iters):
        rx_tot = G_eff @ p
        rx_util = diag_G * p
        interf_noise = np.maximum(rx_tot - rx_util, 0.0) + P_n
        sinr = rx_util / interf_noise

        if hist is not None:
            hist["power_w"].append(p.copy())
            hist["sinr_lin"].append(sinr.copy())

        denom = rx_tot + P_n
        benefit = diag_G / np.maximum(denom, 1e-30)
        # cost_k = sum_{i != k} (sinr_i * G_eff[i, k]) / (rx_tot[i] + P_n)
        weights = sinr / np.maximum(denom, 1e-30)
        cost = G_off.T @ weights
        grad = benefit - cost

        max_g = float(np.max(np.abs(grad)))
        if max_g < 1e-12:
            break
        step = (0.15 * p_max_w) / ((1.0 + 0.08 * it) * max_g)
        p_next = np.clip(p + step * grad, p_min_w, p_max_w)
        if np.max(np.abs(p_next - p)) < 1e-5 * p_max_w:
            p = p_next
            break
        p = p_next

    sinr_final = calculate_sinr_vector(G_eff, p, P_n)
    gamma_tar = np.full(K, cfg.gamma_tar_init_db, dtype=float)

    if hist is not None:
        hist["power_w"] = np.asarray(hist["power_w"], dtype=float)
        hist["sinr_lin"] = np.asarray(hist["sinr_lin"], dtype=float)
        hist["target_db"] = np.asarray(hist["target_db"], dtype=float)
        hist["active"] = np.asarray(hist["active"], dtype=bool)
        hist["p_max_dbm"] = float(cfg.p_max_dbm)
        hist["p_min_dbm"] = float(cfg.p_min_dbm)
        hist["mu_step"] = float(cfg.mu_step_db)
        hist["gamma_min_db"] = float(cfg.gamma_min_db)
        hist["gamma_max_db"] = float(cfg.gamma_max_db)
        hist["tol_db"] = float(cfg.tol_db)

    return p, sinr_final, gamma_tar, hist


def pc_max_prod(
    G_eff: np.ndarray,
    P_n: float,
    cfg: PowerControlConfig,
    record_history: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict[str, Any]]]:
    """
    MaxProd (Proportional Fairness) via gradiente projetado (Parte 3 — rrm.py, Eq. 20):
        ∂(Σ log(SINR_i)) / ∂p_k = 1 / p_k - Σ_{i≠k} G[i,k] / (I_i + P_n)
    """
    K = G_eff.shape[0]
    if K == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty, empty, None

    p_min_w = max(cfg.p_min_w, 1e-6)
    p_max_w = cfg.p_max_w
    p = np.full(K, 0.5 * (p_min_w + p_max_w), dtype=float)
    diag_G = np.diagonal(G_eff).copy()
    G_off = G_eff.copy()
    np.fill_diagonal(G_off, 0.0)

    hist: Optional[dict[str, Any]] = None
    if record_history:
        hist = {
            "algorithm": "max_prod",
            "power_w": [],
            "sinr_lin": [],
            "target_db": [np.full(K, cfg.gamma_tar_init_db, dtype=float)],
            "outer_marks": [],
            "active": [np.ones(K, dtype=bool)],
        }

    for it in range(cfg.sgd_iters):
        rx_tot = G_eff @ p
        rx_util = diag_G * p
        interf_noise = np.maximum(rx_tot - rx_util, 0.0) + P_n
        sinr = rx_util / interf_noise

        if hist is not None:
            hist["power_w"].append(p.copy())
            hist["sinr_lin"].append(sinr.copy())

        benefit = 1.0 / np.maximum(p, 1e-12)
        cost = G_off.T @ (1.0 / np.maximum(interf_noise, 1e-30))
        grad = benefit - cost

        max_g = float(np.max(np.abs(grad)))
        if max_g < 1e-12:
            break
        step = (0.15 * p_max_w) / ((1.0 + 0.08 * it) * max_g)
        p_next = np.clip(p + step * grad, p_min_w, p_max_w)
        if np.max(np.abs(p_next - p)) < 1e-5 * p_max_w:
            p = p_next
            break
        p = p_next

    sinr_final = calculate_sinr_vector(G_eff, p, P_n)
    gamma_tar = np.full(K, cfg.gamma_tar_init_db, dtype=float)

    if hist is not None:
        hist["power_w"] = np.asarray(hist["power_w"], dtype=float)
        hist["sinr_lin"] = np.asarray(hist["sinr_lin"], dtype=float)
        hist["target_db"] = np.asarray(hist["target_db"], dtype=float)
        hist["active"] = np.asarray(hist["active"], dtype=bool)
        hist["p_max_dbm"] = float(cfg.p_max_dbm)
        hist["p_min_dbm"] = float(cfg.p_min_dbm)
        hist["mu_step"] = float(cfg.mu_step_db)
        hist["gamma_min_db"] = float(cfg.gamma_min_db)
        hist["gamma_max_db"] = float(cfg.gamma_max_db)
        hist["tol_db"] = float(cfg.tol_db)

    return p, sinr_final, gamma_tar, hist


# Strategy Registry de Controle de Potência
POWER_CONTROL_REGISTRY: dict[
    str,
    Callable[
        [np.ndarray, float, PowerControlConfig, bool],
        tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict[str, Any]]],
    ],
] = {
    "full_power": pc_full_power,
    "cipc": pc_cipc,
    "optimal": pc_optimal,
    "dpc_fixed": pc_dpc_fixed,
    "dual_loop": pc_dual_loop,
    "max_sum": pc_max_sum,
    "max_prod": pc_max_prod,
}


def run_power_control(
    algorithm: str,
    G_eff: np.ndarray,
    P_n: float,
    cfg: Optional[PowerControlConfig] = None,
    record_history: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict[str, Any]]]:
    """Executa a estratégia de controle de potência registrada em POWER_CONTROL_REGISTRY."""
    fn = POWER_CONTROL_REGISTRY.get(algorithm)
    if fn is None:
        raise ValueError(f"Algoritmo de controle de potência desconhecido: {algorithm!r}")
    pc_cfg = cfg if cfg is not None else PowerControlConfig()
    return fn(G_eff, P_n, pc_cfg, record_history)


# ---------------------------------------------------------------------------
# Wrappers de Retrocompatibilidade (CLI / scripts legados)
# ---------------------------------------------------------------------------


def DPC_Dual_Loop(
    G_eff: np.ndarray,
    P_n: float,
    gamma_tar_db: np.ndarray,
    ue_active: np.ndarray,
    t_outer: int = T_OUTER_DEFAULT,
    t_inner: int = T_INNER_DEFAULT,
    tol_db: float = TOL_DB,
    return_history: bool = False,
    p_max_dbm: float = P_MAX_DBM,
    p_min_dbm: float = P_MIN_DBM,
    mu_step: float = MU_STEP,
    gamma_max_db: float = GAMMA_MAX_DB,
    gamma_min_db: float = GAMMA_MIN_DB,
) -> tuple:
    """Wrapper retrocompatível para `research-activity-01.py` e chamadas existentes."""
    p_w, sinr_lin, _, hist = DPC_Dual_Loop_Vectorized(
        G_eff=G_eff,
        P_n=P_n,
        gamma_tar_db=gamma_tar_db,
        ue_active=ue_active,
        t_outer=t_outer,
        t_inner=t_inner,
        tol_db=tol_db,
        record_history=return_history,
        p_max_dbm=p_max_dbm,
        p_min_dbm=p_min_dbm,
        mu_step=mu_step,
        gamma_max_db=gamma_max_db,
        gamma_min_db=gamma_min_db,
    )
    if return_history:
        return p_w, sinr_lin, hist
    return p_w, sinr_lin


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
    """Wrapper legado do DPC multiplicativo (compatibilidade com scripts antigos)."""
    K = len(ues)
    p = np.ones((K, t_max), dtype=float) * p_init
    pn = B_TOTAL * K_0 / N
    iteracoes = 0

    for t in range(t_max):
        for i, ue in enumerate(ues):
            interference = sum(
                p[k, t]
                * (G[k, ue.ap.id, ue.beam] if G.ndim == 3 else G[k, ue.ap.id])
                * R[k, ue.ap.id]
                for k in range(K)
                if k != i and ues[k].channel == ue.channel
            )
            ue.interference.append(interference)
            sinr = (p[i, t] * ue.gain * R[i, ue.ap.id]) / (interference + pn)
            pt = min(max(p_min, p[i, t] * y_tar / max(sinr, 1e-30)), p_max)
            if t < t_max - 1:
                p[i, t + 1] = pt

        iteracoes += 1
        if iteracoes > 5 and t < t_max - 1:
            if abs(np.sum(p[:, t + 1]) - np.sum(p[:, t])) < 1e-3:
                break

    return p[:, :iteracoes]

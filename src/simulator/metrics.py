# ---------------------------------------------------------------------------
# metrics.py — SINR (estático e adaptativo) e Capacidade de Shannon
# ---------------------------------------------------------------------------

import numpy as np

# Constantes de ruído (compartilhadas entre as funções)
_BT: float = 1e8  # largura de banda total disponível (Hz)
_K0: float = 1e-20  # constante de densidade espectral de ruído


def SINR(ues: list, N: int, gains: np.ndarray, G: np.ndarray = None) -> list:
    """
    Calcula a SINR para antenas convencionais (Omni, Setorial, SMB).

    Parâmetros
    ----------
    N     : número de canais ortogonais.
    gains : matriz (n_ues, n_aps, n_beams) de ganho de canal.
    G     : ignorado (mantido por compatibilidade).

    Retorna lista de valores SINR lineares, um por UE.
    """
    pn = _K0 * _BT / N

    P = np.array([ue.power for ue in ues])
    A = np.array([ue.ap.id for ue in ues])
    B = np.array([ue.beam for ue in ues])
    C = np.array([ue.channel for ue in ues])

    sinr_list = []
    num_ues = len(ues)

    for k, ue in enumerate(ues):
        m = A[k]  # AP servidor
        b = B[k]  # feixe ativo do UE k

        # Sinal desejado
        S = gains[k, m, b] * P[k]

        # Interferência: demais UEs no mesmo canal recebidos no feixe ativo b
        I = sum(
            gains[i, m, b] * P[i] for i in range(num_ues) if i != k and C[i] == C[k]
        )

        sinr_list.append(S / (I + pn))

    return sinr_list


def SINR_adaptive(ues: list, N: int, gains: np.ndarray, Gm: np.ndarray) -> list:
    """
    Calcula a SINR com beamforming adaptativo MRC.

    Parâmetros
    ----------
    gains : (n_ues, n_aps, 1) — path loss + shadowing SEM ganho de antena.
    Gm    : saída de channel.adaptive_gains() — (n_ues, n_ues).

    Retorna lista de valores SINR lineares, um por UE.
    """
    pn = _K0 * _BT / N

    P = np.array([ue.power for ue in ues])
    A = np.array([ue.ap.id for ue in ues])
    C = np.array([ue.channel for ue in ues])

    sinr_list = []

    for k in range(len(ues)):
        m = A[k]
        S = gains[k, m, 0] * P[k] * Gm[k, k]
        mask = C == C[k]
        mask[k] = False
        I = np.sum(gains[mask, m, 0] * P[mask] * Gm[k, mask])
        sinr_list.append(S / (I + pn))

    return sinr_list


def channel_capacity(sinr_valores: list, N: int) -> np.ndarray:
    """
    Aplica a equação de Shannon-Hartley: C = B · log₂(1 + SINR).

    Parâmetros
    ----------
    sinr_valores : lista ou array de valores SINR lineares.
    N            : número de canais ortogonais (define a largura de banda por canal).

    Retorna array de capacidades em Mbps, um valor por UE.
    """
    B_per_channel = _BT / N
    sinr_array = np.asarray(sinr_valores)
    return B_per_channel * np.log2(1 + sinr_array) / 1e6


# ---------------------------------------------------------------------------
# Métricas Avançadas e Físicas
# ---------------------------------------------------------------------------


def calculate_sinr_vector(
    G_eff: np.ndarray, p_watts: np.ndarray, p_noise: float
) -> np.ndarray:
    """
    Calcula o vetor linear de SINR para todos os K UEs simultaneamente
    usando produto matricial rápido (G_eff @ p_watts).
    """
    total_rx_power = G_eff @ p_watts
    useful_signal = np.diagonal(G_eff) * p_watts
    interference = total_rx_power - useful_signal

    # Evita divisão por zero e zera SINR de UEs desligados
    return np.where(p_watts > 0.0, useful_signal / (interference + p_noise), 0.0)


def evaluate_physical_kpis(
    sinr_lin, p_watts, bw_channel, gamma_min_db=-10.0, gamma_max_db=20.0
):
    """
    Avalia KPIs aplicando a saturação de hardware QAM (20 dB)
    e o limiar de Outage (-10 dB).
    Retorna: (se_per_ue, sum_se, sum_cap_bps, total_power_w, outage_mask)
    """
    gamma_min_lin = 10.0 ** (gamma_min_db / 10.0)
    gamma_max_lin = 10.0 ** (gamma_max_db / 10.0)

    p_watts = np.asarray(p_watts)
    sinr_lin = np.asarray(sinr_lin)

    # Identifica UEs em Outage ou bloqueados pelo DPC
    outage_mask = (p_watts <= 0.0) | (sinr_lin < gamma_min_lin)

    # Limita o teto em 20 dB e zera o de quem está em Outage
    sinr_capped = np.where(outage_mask, 0.0, np.minimum(sinr_lin, gamma_max_lin))

    se_per_ue = np.log2(1.0 + sinr_capped)  # Eficiência Espectral [bps/Hz]
    sum_se = np.sum(se_per_ue)  # Sum-Capacity SE
    sum_cap_bps = bw_channel * sum_se  # Sum-Capacity [bps]
    total_power_w = np.sum(p_watts)  # Potência total gasta [W]

    return se_per_ue, sum_se, sum_cap_bps, total_power_w, outage_mask

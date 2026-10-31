# ---------------------------------------------------------------------------
# metrics.py — SINR (estático e adaptativo), Capacidade de Shannon e KPIs
# ---------------------------------------------------------------------------

import numpy as np


# Constantes de ruído (compartilhadas entre as funções)
_BT: float = 1e8    # largura de banda total disponível (Hz)
_K0: float = 1e-20  # constante de densidade espectral de ruído


def SINR(ues: list, N: int, gains: np.ndarray,
         G: np.ndarray = None) -> list:
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

    P = np.array([ue.power   for ue in ues])
    A = np.array([ue.ap.id   for ue in ues])
    B = np.array([ue.beam    for ue in ues])
    C = np.array([ue.channel for ue in ues])

    sinr_list = []
    num_ues = len(ues)

    for k, ue in enumerate(ues):
        m = A[k]    # AP servidor
        b = B[k]    # feixe ativo do UE k

        # Sinal desejado
        S = gains[k, m, b] * P[k]

        # Interferência: demais UEs no mesmo canal recebidos no feixe ativo b
        I = sum(
            gains[i, m, b] * P[i]
            for i in range(num_ues)
            if i != k and C[i] == C[k]
        )

        sinr_list.append(S / (I + pn))

    return sinr_list


def SINR_adaptive(ues: list, N: int, gains: np.ndarray,
                  Gm: np.ndarray) -> list:
    """
    Calcula a SINR com beamforming adaptativo MRC.

    Parâmetros
    ----------
    gains : (n_ues, n_aps, 1) — path loss + shadowing SEM ganho de antena.
    Gm    : saída de channel.adaptive_gains() — (n_ues, n_ues).

    Retorna lista de valores SINR lineares, um por UE.
    """
    pn = _K0 * _BT / N

    P = np.array([ue.power   for ue in ues])
    A = np.array([ue.ap.id   for ue in ues])
    C = np.array([ue.channel for ue in ues])

    sinr_list = []

    for k in range(len(ues)):
        m = A[k]
        S = gains[k, m, 0] * P[k] * Gm[k, k]
        mask = (C == C[k])
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


def tabela_kpis(res: dict) -> None:
    """
    Imprime os KPIs médios de cada cenário de mobilidade.

    res : saída de mobility.estudo_mobilidade(),
          {(nome_vel, T_ms, L, d_perp): resultado_de_simular_trajetoria}.
    """
    print(f"{'Mobilidade':20s} {'T(ms)':>6s} {'L':>3s} {'d(m)':>6s} "
          f"{'Perda(dB)':>10s} {'Pico(dB)':>9s} {'SE bruta':>9s} "
          f"{'SE líq.':>8s}")
    for (nome, T, L, d), r in res.items():
        print(f"{nome:20s} {T:6.0f} {L:3d} {d:6.0f} "
              f"{r['loss_medio_db']:10.2f} {r['loss_max_db']:9.2f} "
              f"{r['se_gross_medio']:9.3f} {r['se_net_medio']:8.3f}")

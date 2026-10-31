# ---------------------------------------------------------------------------
# mobility.py — Impacto da mobilidade no beamforming adaptativo (Cap. 2)
# ---------------------------------------------------------------------------
# Um único UE atravessa a célula tangencialmente ao broadside do arranjo
# (trajetória retilínea paralela ao eixo y, a uma distância perpendicular
# d_perp do AP). O AP atualiza os pesos MRC a cada T ms; entre atualizações o
# UE se move e surge o erro de desalinhamento (lag):
#
#     φ_total(t) = ε_est + φ(t) − φ(t_últ_atualização)
#
# Os gráficos deste módulo ficam em visualization.py.
#
# O feixe aponta para φ̂ = φ(t_upd) + ε_est e o ganho real é
# G(t) = |w(φ̂)ᵀ a(φ(t))|²  (mrc_gain, UCA de L elementos; máx = L).
# ---------------------------------------------------------------------------

import numpy as np

from .config import VELOCIDADES_KMH, PERIODOS_MS, ARRANJOS_L, DISTANCIAS_M, _K0, _BT, SINR_CAP_DB
from .antennas import mrc_gain

# ---------------------------------------------------------------------------
# Simulação de uma trajetória
# ---------------------------------------------------------------------------

def simular_trajetoria(v_kmh: float, d_perp: float, T_ms: float, L: int = 8,
                       dt_ms: float = 1.0, duracao_s: float = 2.0,
                       sigma_est_deg: float = 1.0, seed: int = 0,
                       N: int = 1, t_pilot_ms: float = 0.5,
                       p_tx: float = 1.0) -> dict:
    """
    Simula um UE cruzando a célula tangencialmente ao broadside do AP.

    O AP está na origem e o broadside aponta para +x. O UE anda em
    x = d_perp, com y(t) variando de −v·duracao/2 a +v·duracao/2
    (cruza o broadside no instante central).

    Parâmetros
    ----------
    v_kmh        : velocidade do UE (km/h).
    d_perp       : distância perpendicular ao AP (m).
    T_ms         : período de atualização dos pesos MRC (ms).
    L            : nº de elementos do arranjo.
    dt_ms        : passo de simulação (ms).
    sigma_est_deg: desvio-padrão do erro de estimação de AoA (graus).
    seed         : semente — é reiniciada no início de CADA cenário, para que
                   todos enfrentem as mesmas realizações de erro.
    t_pilot_ms   : tempo de sinalização/treino gasto em cada atualização
                   (define o overhead = t_pilot / T).

    Retorna dict com séries temporais (t, y, phi, phi_hat, gain, loss_db,
    snr_db, se_gross, se_net) e KPIs escalares (média no tempo).
    """
    np.random.seed(seed)                                  # mesma semente por cenário

    n = int(round(duracao_s * 1e3 / dt_ms))
    t_ms = np.arange(n) * dt_ms
    v = v_kmh / 3.6                                        # m/s
    y = v * (t_ms / 1e3 - duracao_s / 2)                   # posição ao longo da trajetória
    phi = np.arctan2(y, d_perp)                            # AoA verdadeiro (rad)

    # --- Atualizações de feixe: a cada T ms, com erro de estimação gaussiano ---
    passos_upd = max(int(round(T_ms / dt_ms)), 1)
    idx_upd = (np.arange(n) // passos_upd) * passos_upd    # índice da última atualização
    n_upd = int(np.ceil(n / passos_upd))
    eps = np.deg2rad(sigma_est_deg) * np.random.randn(n_upd)
    phi_hat = phi[idx_upd] + eps[np.arange(n) // passos_upd]

    # --- Ganho de arranjo: real (com lag) vs ideal (= L) ---
    gain = mrc_gain(phi_hat, phi, L)
    gain_db = 10 * np.log10(np.maximum(gain, 1e-6))
    loss_db = 10 * np.log10(L / np.maximum(gain, 1e-6))    # perda de desalinhamento
    erro_total_deg = np.rad2deg(phi - phi_hat)

    # --- Enlace: path loss d⁻⁴ (sem shadowing), ruído térmico, SNR com teto ---
    d = np.maximum(np.hypot(d_perp, y), 1.0)
    pn = _K0 * _BT / N
    snr = p_tx * 1e-4 / d ** 4 * gain / pn
    snr_db = np.minimum(10 * np.log10(snr), SINR_CAP_DB)
    snr_cap = 10 ** (snr_db / 10)

    # --- Eficiência espectral (C/B) e overhead de atualização ---
    se_gross = np.log2(1 + snr_cap)                        # bit/s/Hz
    overhead = min(t_pilot_ms / T_ms, 1.0)
    se_net = (1 - overhead) * se_gross

    return {
        't_ms': t_ms, 'y': y, 'phi': phi, 'phi_hat': phi_hat,
        'erro_total_deg': erro_total_deg,
        'gain': gain, 'gain_db': gain_db, 'loss_db': loss_db,
        'snr_db': snr_db, 'se_gross': se_gross, 'se_net': se_net,
        # KPIs escalares
        'loss_medio_db': float(np.mean(loss_db)),
        'loss_max_db': float(np.max(loss_db)),
        'se_gross_medio': float(np.mean(se_gross)),
        'se_net_medio': float(np.mean(se_net)),
        'overhead': overhead,
        'params': dict(v_kmh=v_kmh, d_perp=d_perp, T_ms=T_ms, L=L),
    }


# ---------------------------------------------------------------------------
# Estudo completo: velocidades × períodos × arranjos × distâncias
# ---------------------------------------------------------------------------

def estudo_mobilidade(velocidades: dict = None, periodos: tuple = PERIODOS_MS,
                      arranjos: tuple = ARRANJOS_L,
                      distancias: tuple = DISTANCIAS_M, **kwargs) -> dict:
    """
    Executa todas as combinações. Retorna
        {(nome_vel, T_ms, L, d_perp): resultado_de_simular_trajetoria}
    A semente é reiniciada em cada cenário (via simular_trajetoria), garantindo
    realizações de erro de estimação idênticas entre configurações.
    """
    velocidades = velocidades or VELOCIDADES_KMH
    res = {}
    for nome, v in velocidades.items():
        for T in periodos:
            for L in arranjos:
                for d in distancias:
                    res[(nome, T, L, d)] = simular_trajetoria(
                        v, d, T, L, **kwargs)
    return res

# ---------------------------------------------------------------------------
# runner.py — Motor do experimento: orquestra rodadas e distribui APs
# ---------------------------------------------------------------------------

import numpy as np
from tqdm import tqdm

from .allocation import attach_AP_UE
from .antennas import antenna_gain
from .channel import adaptive_gains, gain_matrix, gerar_shadowing
from .entities import AP, UE
from .metrics import SINR, SINR_adaptive, channel_capacity

# ---------------------------------------------------------------------------
# Posicionamento de APs em grade uniforme
# ---------------------------------------------------------------------------


def distribuir_AP(M: int) -> list:
    """
    Posiciona M APs em uma grade uniforme dentro da área 1000 × 1000 m.

    Retorna lista de objetos AP.
    """
    APs: list = []
    dx = 1000 / (2 * np.sqrt(M))
    a = np.arange(dx, 1001 - dx, 2 * dx)
    x, y = np.meshgrid(a, a)
    for id_, (xi, yi) in enumerate(zip(x.ravel(), y.ravel())):
        APs.append(AP(xi, yi, id_))
    return APs


# ---------------------------------------------------------------------------
# Geração de cenário para visualização
# ---------------------------------------------------------------------------


def cenario_aleatorio(M: int, K: int, seed: int = None) -> tuple:
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
    shadowing = gerar_shadowing(ues, aps)
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
    shadowing_params: dict = None,
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
            gains = gain_matrix(
                ues, aps, G, shadowing=shadowing, use_rayleigh=use_rayleigh
            )
            attach_AP_UE(ues, aps, gains)

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

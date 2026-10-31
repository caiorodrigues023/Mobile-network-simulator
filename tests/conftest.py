# ---------------------------------------------------------------------------
# conftest.py — Fixtures compartilhadas entre todos os módulos de teste
# ---------------------------------------------------------------------------
#
# Fixtures de escopo 'session' são instanciadas UMA VEZ por execução do pytest,
# reduzindo tempo total. Fixtures de escopo 'function' recalculam a cada teste,
# garantindo isolamento quando há estado mutável (ex.: ue.ap, ap.ues).
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")   # backend sem janela: testes de gráficos rodam sem display

import numpy as np
import pytest

from simulator.entities import AP, UE
from simulator.channel import gerar_shadowing, gain_matrix
from simulator.antennas import antenna_gain
from simulator.allocation import attach_AP_UE, alocar_canais_ortogonal
from simulator.runner import distribuir_AP


# ---------------------------------------------------------------------------
# Semente global para reproducibilidade
# ---------------------------------------------------------------------------

SEED = 42


# ---------------------------------------------------------------------------
# APs em grade (compartilhado — não tem estado mutável)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def aps_grade_4() -> list:
    """4 APs em grade 2×2 dentro de 1000×1000 m."""
    return distribuir_AP(4)


@pytest.fixture(scope="session")
def aps_grade_9() -> list:
    """9 APs em grade 3×3 dentro de 1000×1000 m."""
    return distribuir_AP(9)


# ---------------------------------------------------------------------------
# Cenários completos com UEs (escopo 'function' por causa de ue.ap, ap.ues)
# ---------------------------------------------------------------------------

@pytest.fixture
def cenario_basico(aps_grade_4) -> dict:
    """
    Cenário reproduzível: 4 APs, 4 UEs, shadowing fixo.
    Retorna dicionário com 'aps', 'ues' e 'shadowing'.
    """
    np.random.seed(SEED)
    UE.id_counter = 0

    # Copia a lista para não poluir o fixture de sessão
    aps = [AP(ap.x, ap.y, ap.id) for ap in aps_grade_4]
    ues = [UE(aps) for _ in range(4)]
    shadowing = gerar_shadowing(ues, aps)
    return {"aps": aps, "ues": ues, "shadowing": shadowing}


@pytest.fixture
def cenario_com_attach(cenario_basico) -> dict:
    """
    Cenário com attach já executado para o modelo 'omni'.
    Útil para testes de métricas e alocação de canal.
    """
    aps = cenario_basico["aps"]
    ues = cenario_basico["ues"]
    shadowing = cenario_basico["shadowing"]

    G = antenna_gain(ues, aps, model="omni")
    gains = gain_matrix(ues, aps, G, shadowing=shadowing)
    attach_AP_UE(ues, aps, gains)
    alocar_canais_ortogonal(aps, ues, number_channels=4, allocation="ortogonal")

    return {"aps": aps, "ues": ues, "shadowing": shadowing, "gains": gains}

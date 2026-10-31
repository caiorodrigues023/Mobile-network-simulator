# ---------------------------------------------------------------------------
# test_channel.py — Testes de geometria, shadowing e matriz de ganho
# ---------------------------------------------------------------------------

import numpy as np
import pytest

from simulator.channel import angle, angle_matrix, gerar_shadowing, gain_matrix
from simulator.antennas import antenna_gain
from simulator.entities import AP, UE
from simulator.runner import distribuir_AP


# ---------------------------------------------------------------------------
# Fixtures locais
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_ue_counter():
    UE.id_counter = 0


@pytest.fixture
def aps():
    return distribuir_AP(4)


@pytest.fixture
def ues(aps):
    np.random.seed(42)
    return [UE(aps) for _ in range(6)]


# ---------------------------------------------------------------------------
# angle()
# ---------------------------------------------------------------------------

class TestAngle:
    def _make_ue_ap(self, ue_xy, ap_xy):
        """Cria objetos simples sem usar UE.__init__ (que posiciona aleatoriamente)."""
        ue = object.__new__(UE)
        ue.x, ue.y = ue_xy
        ap = AP(*ap_xy, id=0)
        return ue, ap

    @pytest.mark.parametrize("ue_xy,ap_xy,expected_deg", [
        ((2, 0), (0, 0), 0),          # Leste → 0°
        ((0, 2), (0, 0), 90),         # Norte → 90°
        ((-2, 0), (0, 0), 180),       # Oeste → 180°
        ((0, -2), (0, 0), 270),       # Sul → 270°
    ])
    def test_quadrantes_cardinais(self, ue_xy, ap_xy, expected_deg):
        ue, ap = self._make_ue_ap(ue_xy, ap_xy)
        result_deg = np.degrees(angle(ue, ap))
        assert abs(result_deg - expected_deg) < 1e-6

    def test_retorna_float(self):
        ue, ap = self._make_ue_ap((100, 200), (50, 50))
        assert isinstance(angle(ue, ap), float)

    def test_intervalo_0_a_2pi(self):
        aps_test = distribuir_AP(4)
        UE.id_counter = 0
        np.random.seed(10)
        for _ in range(50):
            ue = UE(aps_test)
            for ap in aps_test:
                theta = angle(ue, ap)
                assert 0 <= theta <= 2 * np.pi, f"theta={theta} fora de [0, 2π]"


# ---------------------------------------------------------------------------
# angle_matrix()
# ---------------------------------------------------------------------------

class TestAngleMatrix:
    def test_shape(self, ues, aps):
        phi = angle_matrix(ues, aps)
        assert phi.shape == (len(ues), len(aps))

    def test_intervalo(self, ues, aps):
        phi = angle_matrix(ues, aps)
        assert np.all(phi >= 0)
        assert np.all(phi <= 2 * np.pi)

    def test_consistente_com_angle(self, ues, aps):
        """angle_matrix deve concordar com a função escalar angle()."""
        phi_mat = angle_matrix(ues, aps)
        for i, ue in enumerate(ues):
            for j, ap in enumerate(aps):
                expected = angle(ue, ap)
                assert abs(phi_mat[i, j] - expected) < 1e-10


# ---------------------------------------------------------------------------
# gerar_shadowing()
# ---------------------------------------------------------------------------

class TestGerarShadowing:
    def test_shape(self, ues, aps):
        s = gerar_shadowing(ues, aps)
        assert s.shape == (len(ues), len(aps))

    def test_valores_positivos(self, ues, aps):
        """Shadowing log-normal deve ser sempre positivo."""
        s = gerar_shadowing(ues, aps)
        assert np.all(s > 0)

    def test_reproducivel_com_semente(self, ues, aps):
        np.random.seed(0)
        s1 = gerar_shadowing(ues, aps)
        np.random.seed(0)
        s2 = gerar_shadowing(ues, aps)
        np.testing.assert_array_equal(s1, s2)

    def test_distribuicao_log_normal(self, aps):
        """log(shadowing) deve ter distribuição próxima de N(0, σ²) com σ=2."""
        UE.id_counter = 0
        np.random.seed(99)
        ues_big = [UE(aps) for _ in range(200)]
        s = gerar_shadowing(ues_big, aps)
        log_s = np.log(s.ravel())
        assert abs(np.mean(log_s)) < 0.3          # média de log ≈ 0
        assert abs(np.std(log_s) - 2.0) < 0.3     # desvio ≈ 2


# ---------------------------------------------------------------------------
# gain_matrix()
# ---------------------------------------------------------------------------

class TestGainMatrix:
    def test_shape_omni(self, ues, aps):
        G = antenna_gain(ues, aps, model="omni")
        shadowing = gerar_shadowing(ues, aps)
        gains = gain_matrix(ues, aps, G, shadowing=shadowing)
        assert gains.shape == (len(ues), len(aps), 1)

    def test_shape_smb(self, ues, aps):
        G = antenna_gain(ues, aps, model="smb", L=8, J=8)
        shadowing = gerar_shadowing(ues, aps)
        gains = gain_matrix(ues, aps, G, shadowing=shadowing)
        assert gains.shape == (len(ues), len(aps), 8)

    def test_valores_positivos(self, ues, aps):
        G = antenna_gain(ues, aps, model="omni")
        shadowing = gerar_shadowing(ues, aps)
        gains = gain_matrix(ues, aps, G, shadowing=shadowing)
        assert np.all(gains > 0)

    def test_path_loss_decrescente_com_distancia(self, aps):
        """
        UE perto de um AP deve ter maior ganho naquele AP do que um UE mais distante.
        Usando shadowing=1 para isolar apenas o path loss.
        """
        # Cria dois UEs manualmente em distâncias controladas do AP[0]
        ap = aps[0]
        ue_perto = object.__new__(UE)
        ue_perto.x, ue_perto.y = ap.x + 10, ap.y
        ue_longe = object.__new__(UE)
        ue_longe.x, ue_longe.y = ap.x + 400, ap.y

        ue_list = [ue_perto, ue_longe]
        G = np.ones((2, 4, 1))
        shadowing = np.ones((2, 4))

        gains = gain_matrix(ue_list, aps, G, shadowing=shadowing)
        assert gains[0, 0, 0] > gains[1, 0, 0]

    def test_shadowing_none_usa_sorteio(self, ues, aps):
        """Passar shadowing=None não deve lançar exceção."""
        G = antenna_gain(ues, aps, model="omni")
        np.random.seed(5)
        gains = gain_matrix(ues, aps, G, shadowing=None)
        assert gains.shape == (len(ues), len(aps), 1)

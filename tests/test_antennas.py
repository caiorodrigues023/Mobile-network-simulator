# ---------------------------------------------------------------------------
# test_antennas.py — Testes dos modelos de antena
# ---------------------------------------------------------------------------

import numpy as np
import pytest

from simulator.antennas import smb_gains, mrc_gain, antenna_gain
from simulator.runner import distribuir_AP
from simulator.entities import UE


# ---------------------------------------------------------------------------
# Helpers / fixtures locais
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_ue_counter():
    UE.id_counter = 0


@pytest.fixture
def aps():
    return distribuir_AP(4)


@pytest.fixture
def ues_4(aps):
    np.random.seed(42)
    return [UE(aps) for _ in range(4)]


# ---------------------------------------------------------------------------
# SMB gains
# ---------------------------------------------------------------------------

class TestSmbGains:
    def test_shape_escalar(self):
        g = smb_gains(0.0, L=8, J=8)
        assert g.shape == (8,)

    def test_shape_array_1d(self):
        phi = np.linspace(0, 2 * np.pi, 10)
        g = smb_gains(phi, L=8, J=8)
        assert g.shape == (10, 8)

    def test_shape_array_2d(self):
        phi = np.zeros((3, 5))
        g = smb_gains(phi, L=8, J=8)
        assert g.shape == (3, 5, 8)

    def test_ganho_pico_igual_L(self):
        """O ganho de um feixe alinhado com φ deve valer L."""
        L, J = 8, 8
        theta_k = np.arange(J) * 2 * np.pi / J
        for k, phi in enumerate(theta_k):
            g = smb_gains(phi, L=L, J=J)
            assert abs(g[k] - L) < 1e-6, f"Feixe {k}: ganho={g[k]:.4f}, esperado {L}"

    def test_ganho_nao_negativo(self):
        phi = np.random.uniform(0, 2 * np.pi, 50)
        assert np.all(smb_gains(phi, L=8, J=8) >= 0)

    def test_cache_nao_muta_resultado(self):
        """Chamadas repetidas com os mesmos parâmetros devem retornar o mesmo valor."""
        phi = np.pi / 4
        g1 = smb_gains(phi, L=8, J=8).copy()
        g2 = smb_gains(phi, L=8, J=8).copy()
        np.testing.assert_array_equal(g1, g2)


# ---------------------------------------------------------------------------
# MRC gain
# ---------------------------------------------------------------------------

class TestMrcGain:
    def test_ganho_perfeito_igual_L(self):
        """Feixe alinhado com o usuário → ganho == L."""
        L = 8
        phi = np.pi / 3
        g = mrc_gain(phi, phi, L=L)
        assert abs(float(g) - L) < 1e-6

    def test_ganho_nao_negativo(self):
        L = 8
        phi_b = np.random.uniform(0, 2 * np.pi, 30)
        phi_u = np.random.uniform(0, 2 * np.pi, 30)
        assert np.all(mrc_gain(phi_b, phi_u, L=L) >= 0)

    def test_simetria_do_ganho(self):
        """mrc_gain(a, b) deve ser igual a mrc_gain(b, a)."""
        a, b = 0.5, 1.2
        g_ab = mrc_gain(a, b, L=8)
        g_ba = mrc_gain(b, a, L=8)
        assert abs(float(g_ab) - float(g_ba)) < 1e-6


# ---------------------------------------------------------------------------
# antenna_gain dispatcher
# ---------------------------------------------------------------------------

class TestAntennaGain:
    def test_omni_shape(self, ues_4, aps):
        G = antenna_gain(ues_4, aps, model="omni")
        assert G.shape == (4, 4, 1)

    def test_omni_todos_unitarios(self, ues_4, aps):
        G = antenna_gain(ues_4, aps, model="omni")
        np.testing.assert_array_equal(G, 1.0)

    def test_sector_shape(self, ues_4, aps):
        G = antenna_gain(ues_4, aps, model="sector")
        assert G.shape == (4, 4, 3)

    def test_sector_valores_binarios(self, ues_4, aps):
        """Modelo setorial retorna apenas 1.0 (inside) ou g_min (outside)."""
        G = antenna_gain(ues_4, aps, model="sector", g_min=0.01)
        unique = np.unique(G)
        for v in unique:
            assert v in (1.0, 0.01), f"Valor inesperado: {v}"

    def test_smb_shape(self, ues_4, aps):
        G = antenna_gain(ues_4, aps, model="smb", L=8, J=8)
        assert G.shape == (4, 4, 8)

    def test_smb_ganho_nao_negativo(self, ues_4, aps):
        G = antenna_gain(ues_4, aps, model="smb", L=8, J=8)
        assert np.all(G >= 0)

    @pytest.mark.parametrize("model", ["adaptive", "adaptive_1", "adaptive_5", "adaptive_10"])
    def test_adaptive_retorna_uns(self, model, ues_4, aps):
        """Variantes adaptativas retornam matrix de uns no initial access."""
        G = antenna_gain(ues_4, aps, model=model)
        np.testing.assert_array_equal(G, 1.0)

    def test_modelo_desconhecido_lanca_erro(self, ues_4, aps):
        with pytest.raises(ValueError, match="desconhecido"):
            antenna_gain(ues_4, aps, model="inexistente")

# ---------------------------------------------------------------------------
# test_runner.py — cenario_aleatorio e simular_experimento (Monte Carlo)
# ---------------------------------------------------------------------------

import numpy as np
import pytest

from simulator.entities import UE
from simulator.runner import cenario_aleatorio, simular_experimento


@pytest.fixture(autouse=True)
def reset_ue_counter():
    UE.id_counter = 0


# ---------------------------------------------------------------------------
# cenario_aleatorio
# ---------------------------------------------------------------------------

class TestCenarioAleatorio:
    def test_tamanhos(self):
        aps, ues, sh = cenario_aleatorio(M=9, K=5, seed=1)
        assert len(aps) == 9 and len(ues) == 5
        assert sh.shape == (5, 9)

    def test_ids_dos_ues_recomecam_em_zero(self):
        UE.id_counter = 99
        _, ues, _ = cenario_aleatorio(M=4, K=3, seed=1)
        assert [u.id for u in ues] == [0, 1, 2]

    def test_reproducivel_com_semente(self):
        _, ues1, sh1 = cenario_aleatorio(M=4, K=6, seed=7)
        _, ues2, sh2 = cenario_aleatorio(M=4, K=6, seed=7)
        assert [(u.x, u.y) for u in ues1] == [(u.x, u.y) for u in ues2]
        np.testing.assert_array_equal(sh1, sh2)

    def test_sementes_diferentes_geram_cenarios_diferentes(self):
        _, ues1, _ = cenario_aleatorio(M=4, K=6, seed=1)
        _, ues2, _ = cenario_aleatorio(M=4, K=6, seed=2)
        assert [(u.x, u.y) for u in ues1] != [(u.x, u.y) for u in ues2]

    def test_shadowing_positivo(self):
        _, _, sh = cenario_aleatorio(M=4, K=4, seed=3)
        assert np.all(sh > 0)


# ---------------------------------------------------------------------------
# simular_experimento
# ---------------------------------------------------------------------------

MODELOS = ("omni", "sector", "smb")


def _rodar(K=3, sim=4, modelos=MODELOS, seed=11, **kw):
    np.random.seed(seed)
    return simular_experimento(M=4, N=4, K=K, sim=sim,
                               antenna_models=modelos, **kw)


class TestSimularExperimento:
    def test_chaves_por_modelo(self):
        res = _rodar()
        assert set(res) == set(MODELOS)
        for m in MODELOS:
            assert set(res[m]) == {"power", "sinr", "cap", "sum_cap"}

    def test_tamanho_das_listas_e_sim_vezes_k(self):
        res = _rodar(K=3, sim=4)
        for m in MODELOS:
            assert len(res[m]["power"]) == 12
            assert len(res[m]["sinr"]) == 12
            assert len(res[m]["cap"]) == 12

    def test_sum_cap_e_escalar_positivo(self):
        res = _rodar()
        for m in MODELOS:
            assert isinstance(res[m]["sum_cap"], float)
            assert res[m]["sum_cap"] > 0

    def test_valores_finitos_e_positivos(self):
        res = _rodar()
        for m in MODELOS:
            for k in ("power", "sinr", "cap"):
                v = np.asarray(res[m][k])
                assert np.all(np.isfinite(v)) and np.all(v > 0)

    def test_reproducivel_com_semente(self):
        r1, r2 = _rodar(seed=5), _rodar(seed=5)
        for m in MODELOS:
            np.testing.assert_array_equal(r1[m]["sinr"], r2[m]["sinr"])

    def test_sum_cap_e_media_das_somas_por_iteracao(self):
        """Com K UEs/iteração, sum_cap·sim == soma de todas as capacidades."""
        res = _rodar(K=3, sim=4)
        for m in MODELOS:
            assert res[m]["sum_cap"] * 4 == pytest.approx(
                float(np.sum(res[m]["cap"])))

    def test_k1_setorial_igual_omni(self):
        """Com 1 UE sempre há um setor ativo com ganho 1 → mesmo SINR do omni."""
        res = _rodar(K=1, sim=10)
        np.testing.assert_allclose(res["sector"]["sinr"], res["omni"]["sinr"])

    def test_k1_smb_nunca_pior_que_omni(self):
        """Sem interferência, o melhor feixe SMB tem ganho ≥ 1 (cruzamento ≈ 1,82)."""
        res = _rodar(K=1, sim=10)
        assert np.all(np.asarray(res["smb"]["sinr"])
                      >= np.asarray(res["omni"]["sinr"]) * (1 - 1e-9))

    def test_k1_adaptativo_ideal_ganha_L_sobre_omni(self):
        res = _rodar(K=1, sim=10, modelos=("omni", "adaptive"), L=8)
        np.testing.assert_allclose(
            np.asarray(res["adaptive"]["sinr"]),
            8 * np.asarray(res["omni"]["sinr"]), rtol=1e-6)

    def test_adaptativo_com_erro_nao_excede_ideal(self):
        res = _rodar(K=1, sim=10, modelos=("adaptive", "adaptive_10"))
        assert np.all(np.asarray(res["adaptive_10"]["sinr"])
                      <= np.asarray(res["adaptive"]["sinr"]) * (1 + 1e-9))

    def test_modelo_desconhecido_lanca_erro(self):
        with pytest.raises(ValueError):
            _rodar(modelos=("inexistente",))

    def test_aps_ficam_sem_ues_ao_final(self, monkeypatch):
        """O runner limpa ap.ues ao fim de cada modelo (sem vazar estado)."""
        import simulator.runner as runner
        capturados = []
        orig = runner.distribuir_AP
        monkeypatch.setattr(runner, "distribuir_AP",
                            lambda M: capturados.append(orig(M)) or capturados[-1])
        _rodar(sim=2)
        assert all(len(ap.ues) == 0 for ap in capturados[0])

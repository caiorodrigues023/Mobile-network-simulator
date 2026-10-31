# ---------------------------------------------------------------------------
# test_entities.py — Testes das classes AP e UE
# ---------------------------------------------------------------------------

import numpy as np
import pytest

from simulator.entities import AP, UE
from simulator.runner import distribuir_AP


class TestAP:
    def test_atributos_basicos(self):
        ap = AP(100.0, 200.0, 0)
        assert ap.x == 100.0
        assert ap.y == 200.0
        assert ap.id == 0
        assert ap.channel is None
        assert ap.ues == []

    def test_repr_e_str(self):
        ap = AP(100.0, 200.0, 3)
        assert "AP[3]" in str(ap)
        assert "100.0" in str(ap)

    def test_lista_ues_independente(self):
        """Cada AP deve ter sua própria lista de UEs, não compartilhada."""
        ap1 = AP(0, 0, 0)
        ap2 = AP(500, 500, 1)
        ap1.ues.append("x")
        assert "x" not in ap2.ues


class TestUE:
    @pytest.fixture(autouse=True)
    def reset_counter(self):
        """Garante que o contador de IDs começa do zero em cada teste."""
        UE.id_counter = 0

    def test_ids_sequenciais(self):
        aps = distribuir_AP(4)
        np.random.seed(0)
        ues = [UE(aps) for _ in range(5)]
        assert [ue.id for ue in ues] == [0, 1, 2, 3, 4]

    def test_posicao_dentro_da_area(self):
        aps = distribuir_AP(4)
        np.random.seed(1)
        for _ in range(20):
            ue = UE(aps)
            assert 0 <= ue.x <= 1000
            assert 0 <= ue.y <= 1000

    def test_distancia_minima_dos_aps(self):
        """Todo UE deve estar a pelo menos 1 m de cada AP."""
        aps = distribuir_AP(4)
        np.random.seed(2)
        ues = [UE(aps) for _ in range(30)]
        for ue in ues:
            for ap in aps:
                d = np.hypot(ue.x - ap.x, ue.y - ap.y)
                assert d >= 1.0, f"UE {ue.id} está a {d:.2f} m do AP {ap.id}"

    def test_igualdade_por_id(self):
        aps = distribuir_AP(4)
        np.random.seed(3)
        ue_a = UE(aps)
        ue_b = UE(aps)
        assert ue_a == ue_a
        assert ue_a != ue_b
        assert ue_a != "nao_e_um_ue"

    def test_estado_inicial_de_associacao(self):
        aps = distribuir_AP(4)
        np.random.seed(4)
        ue = UE(aps)
        assert ue.ap is None
        assert ue.beam == 0
        assert ue.channel == 0
        assert ue.power == 1.0
        assert ue.interference == []


class TestDistribuirAP:
    @pytest.mark.parametrize("M", [1, 4, 9, 16])
    def test_quantidade_correta(self, M):
        aps = distribuir_AP(M)
        assert len(aps) == M

    def test_ids_sequenciais(self):
        aps = distribuir_AP(4)
        assert [ap.id for ap in aps] == list(range(4))

    def test_aps_dentro_da_area(self):
        for ap in distribuir_AP(9):
            assert 0 < ap.x < 1000
            assert 0 < ap.y < 1000

    def test_aps_com_espacamento_uniforme(self):
        """Para M=4 (grade 2×2) os espaçamentos devem ser iguais."""
        aps = distribuir_AP(4)
        xs = sorted({ap.x for ap in aps})
        ys = sorted({ap.y for ap in aps})
        assert len(xs) == 2
        assert len(ys) == 2
        # Diferença entre as duas colunas deve ser igual à diferença entre as duas linhas
        assert abs((xs[1] - xs[0]) - (ys[1] - ys[0])) < 1e-6

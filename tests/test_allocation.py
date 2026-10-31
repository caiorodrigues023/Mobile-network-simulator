# ---------------------------------------------------------------------------
# test_allocation.py — Testes de attach UE↔AP e alocação de canais
# ---------------------------------------------------------------------------

import numpy as np
import pytest

from simulator.entities import AP, UE
from simulator.runner import distribuir_AP
from simulator.channel import gerar_shadowing, gain_matrix
from simulator.antennas import antenna_gain
from types import SimpleNamespace

from simulator.allocation import attach_AP_UE, alocar_canais_ortogonal, DPC


# ---------------------------------------------------------------------------
# Fixtures locais
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_ue_counter():
    UE.id_counter = 0


@pytest.fixture
def cenario():
    """Cenário com gain matrix pré-calculada, sem attach."""
    aps = distribuir_AP(4)
    np.random.seed(42)
    ues = [UE(aps) for _ in range(8)]
    shadowing = gerar_shadowing(ues, aps)
    G = antenna_gain(ues, aps, model="omni")
    gains = gain_matrix(ues, aps, G, shadowing=shadowing)
    return {"aps": aps, "ues": ues, "gains": gains}


@pytest.fixture
def cenario_attachado(cenario):
    """Cenário com attach e alocação ortogonal já executados."""
    attach_AP_UE(cenario["ues"], cenario["aps"], cenario["gains"])
    alocar_canais_ortogonal(
        cenario["aps"], cenario["ues"],
        number_channels=4, allocation="ortogonal"
    )
    return cenario


# ---------------------------------------------------------------------------
# attach_AP_UE
# ---------------------------------------------------------------------------

class TestAttachAPUE:
    def test_todo_ue_tem_ap(self, cenario):
        attach_AP_UE(cenario["ues"], cenario["aps"], cenario["gains"])
        for ue in cenario["ues"]:
            assert ue.ap is not None

    def test_ap_valido(self, cenario):
        attach_AP_UE(cenario["ues"], cenario["aps"], cenario["gains"])
        ap_ids = {ap.id for ap in cenario["aps"]}
        for ue in cenario["ues"]:
            assert ue.ap.id in ap_ids

    def test_dist_e_angle_preenchidos(self, cenario):
        attach_AP_UE(cenario["ues"], cenario["aps"], cenario["gains"])
        for ue in cenario["ues"]:
            assert ue.dist > 0
            assert 0 <= ue.angle <= 2 * np.pi

    def test_ue_na_lista_do_ap(self, cenario):
        """Após o attach, cada UE deve aparecer na lista ues do seu AP."""
        attach_AP_UE(cenario["ues"], cenario["aps"], cenario["gains"])
        for ue in cenario["ues"]:
            assert ue in ue.ap.ues

    def test_attach_seleciona_maior_ganho(self, cenario):
        """Para omni (G=1 em tudo), o UE deve ir para o AP de maior ganho de canal."""
        attach_AP_UE(cenario["ues"], cenario["aps"], cenario["gains"])
        gains = cenario["gains"]
        for i, ue in enumerate(cenario["ues"]):
            # Ganho do AP escolhido deve ser o máximo da linha i
            chosen_gain = gains[i, ue.ap.id, ue.beam]
            max_gain = gains[i].max()
            assert abs(chosen_gain - max_gain) < 1e-12


# ---------------------------------------------------------------------------
# alocar_canais_ortogonal
# ---------------------------------------------------------------------------

class TestAlocarCanais:
    def test_todos_ues_tem_canal(self, cenario_attachado):
        for ue in cenario_attachado["ues"]:
            assert ue.channel is not None

    def test_canais_dentro_do_range(self, cenario_attachado):
        N = 4
        for ue in cenario_attachado["ues"]:
            assert 1 <= ue.channel <= N, f"Canal {ue.channel} fora de [1, {N}]"

    def test_no_max_n_canais_iguais_por_ap(self, cenario_attachado):
        """Dentro de um mesmo AP, cada canal deve aparecer no máximo uma vez."""
        for ap in cenario_attachado["aps"]:
            canais = [ue.channel for ue in ap.ues if ue.channel is not None]
            assert len(canais) == len(set(canais)), \
                f"AP {ap.id} tem canais duplicados: {canais}"

    def test_modo_random_usa_canais_validos(self, cenario):
        """Modo 'random' deve alocar canais válidos para todos os UEs."""
        attach_AP_UE(cenario["ues"], cenario["aps"], cenario["gains"])
        alocar_canais_ortogonal(
            cenario["aps"], cenario["ues"],
            number_channels=4, allocation="random"
        )
        for ue in cenario["ues"]:
            assert 1 <= ue.channel <= 4


# ---------------------------------------------------------------------------
# DPC — Controle Distribuído de Potência
# ---------------------------------------------------------------------------

def _ue_dpc(ap_id=0, gain=1.0, channel=1):
    """UE mínimo para o DPC (só os atributos que o algoritmo lê/escreve)."""
    return SimpleNamespace(ap=SimpleNamespace(id=ap_id), gain=gain,
                           channel=channel, interference=[])


class TestDPC:
    # Parâmetros em que o alvo é atingível: p* = y_tar·pn/(gain·R) = 0.1 W
    R_OK = np.full((1, 1), 1e-10)
    G_1 = np.ones((1, 1))

    def _rodar(self, ues, G, R, N=1, t_max=50, y_tar=10.0,
               p_min=1e-3, p_max=1.0, p_init=1.0):
        return DPC(ues, N, t_max, G, R, y_tar, p_min, p_max, p_init)

    def test_shape_e_potencia_inicial(self):
        p = self._rodar([_ue_dpc()], self.G_1, self.R_OK)
        assert p.ndim == 2 and p.shape[0] == 1
        assert 1 <= p.shape[1] <= 50
        assert p[0, 0] == 1.0                      # primeira coluna = p_init

    def test_converge_para_potencia_do_alvo(self):
        """UE isolado: p* = y_tar · pn / (gain · R) = 0.1 W."""
        p = self._rodar([_ue_dpc()], self.G_1, self.R_OK)
        assert p[0, -1] == pytest.approx(0.1, rel=1e-6)

    def test_alvo_atingido_no_sinr_final(self):
        p = self._rodar([_ue_dpc()], self.G_1, self.R_OK)
        sinr = p[0, -1] * 1.0 * 1e-10 / (1e8 * 1e-20)
        assert sinr == pytest.approx(10.0, rel=1e-6)

    def test_satura_em_p_max(self):
        """Enlace fraco: a potência necessária excede p_max."""
        R = np.full((1, 1), 1e-14)
        p = self._rodar([_ue_dpc()], self.G_1, R, p_max=1.0)
        assert p[0, -1] == pytest.approx(1.0)

    def test_satura_em_p_min(self):
        """Enlace muito forte: a potência necessária fica abaixo de p_min."""
        R = np.full((1, 1), 1e-5)
        p = self._rodar([_ue_dpc()], self.G_1, R, p_min=1e-3)
        assert p[0, -1] == pytest.approx(1e-3)

    def test_potencias_dentro_dos_limites(self):
        ues = [_ue_dpc(), _ue_dpc()]
        G = np.ones((2, 1))
        R = np.array([[1e-10], [1e-11]])
        p = self._rodar(ues, G, R, p_min=1e-3, p_max=1.0)
        assert np.all(p[:, 1:] >= 1e-3 - 1e-12)
        assert np.all(p <= 1.0 + 1e-12)

    def test_respeita_t_max(self):
        p = self._rodar([_ue_dpc()], self.G_1, self.R_OK, t_max=3)
        assert p.shape[1] <= 3

    def test_interferencia_registrada_a_cada_iteracao(self):
        ue = _ue_dpc()
        p = self._rodar([ue], self.G_1, self.R_OK)
        assert len(ue.interference) == p.shape[1]
        assert all(i == 0 for i in ue.interference)   # sem outros UEs

    def test_interferencia_so_do_mesmo_canal(self):
        G = np.ones((2, 1))
        R = np.array([[1e-10], [1e-10]])
        mesmo = [_ue_dpc(channel=1), _ue_dpc(channel=1)]
        outro = [_ue_dpc(channel=1), _ue_dpc(channel=2)]
        self._rodar(mesmo, G, R, t_max=1)
        self._rodar(outro, G, R, t_max=1)
        # t=0: p = p_init = 1 → I = p·G·R = 1e-10 (mesmo canal) ou 0
        assert mesmo[0].interference[0] == pytest.approx(1e-10)
        assert outro[0].interference[0] == 0.0

    def test_interferencia_reduz_sinr_e_eleva_potencia(self):
        """Com interferente no mesmo canal, o UE precisa de mais potência."""
        R = np.array([[1e-10], [1e-10]])
        G = np.ones((2, 1))
        p_isolado = self._rodar([_ue_dpc()], self.G_1, self.R_OK, p_init=0.05)
        p_interf = self._rodar([_ue_dpc(), _ue_dpc()], G, R, p_init=0.05)
        assert p_interf[0, -1] > p_isolado[0, -1]

# ---------------------------------------------------------------------------
# test_metrics.py — SINR, SINR adaptativo, capacidade de Shannon e KPIs
# ---------------------------------------------------------------------------

from types import SimpleNamespace

import numpy as np
import pytest

from simulator.metrics import (SINR, SINR_adaptive, channel_capacity,
                               tabela_kpis, _BT, _K0)
from simulator.mobility import estudo_mobilidade


def _ue(ap_id=0, beam=0, channel=1, power=1.0):
    """UE mínimo com apenas os atributos lidos pelas métricas."""
    return SimpleNamespace(ap=SimpleNamespace(id=ap_id), beam=beam,
                           channel=channel, power=power)


def _pn(N):
    return _K0 * _BT / N


# ---------------------------------------------------------------------------
# SINR
# ---------------------------------------------------------------------------

class TestSINR:
    def test_ue_isolado_e_snr(self):
        gains = np.full((1, 1, 1), 1e-9)
        s = SINR([_ue()], N=4, gains=gains)
        assert s[0] == pytest.approx(1e-9 / _pn(4))

    def test_ruido_escala_com_n(self):
        """pn = k0·B/N: dobrar N dobra o SNR de um UE isolado."""
        gains = np.full((1, 1, 1), 1e-9)
        s1 = SINR([_ue()], N=1, gains=gains)[0]
        s2 = SINR([_ue()], N=2, gains=gains)[0]
        assert s2 == pytest.approx(2 * s1)

    def test_potencia_escala_o_sinal(self):
        gains = np.full((1, 1, 1), 1e-9)
        s1 = SINR([_ue(power=1.0)], 1, gains)[0]
        s2 = SINR([_ue(power=2.0)], 1, gains)[0]
        assert s2 == pytest.approx(2 * s1)

    def test_interferencia_no_mesmo_canal(self):
        gains = np.zeros((2, 1, 1))
        gains[0, 0, 0], gains[1, 0, 0] = 1e-9, 4e-10
        ues = [_ue(channel=1), _ue(channel=1)]
        s = SINR(ues, 1, gains)
        assert s[0] == pytest.approx(1e-9 / (4e-10 + _pn(1)))
        assert s[1] == pytest.approx(4e-10 / (1e-9 + _pn(1)))

    def test_canais_distintos_nao_interferem(self):
        gains = np.zeros((2, 1, 1))
        gains[0, 0, 0], gains[1, 0, 0] = 1e-9, 4e-10
        ues = [_ue(channel=1), _ue(channel=2)]
        s = SINR(ues, 1, gains)
        assert s[0] == pytest.approx(1e-9 / _pn(1))
        assert s[1] == pytest.approx(4e-10 / _pn(1))

    def test_interferencia_medida_no_feixe_ativo_do_desejado(self):
        """Interferente é recebido pelo feixe b do UE desejado (filtro espacial)."""
        gains = np.zeros((2, 1, 2))
        gains[0, 0, 1] = 1e-9          # UE 0 serve-se do feixe 1
        gains[1, 0, 0] = 1e-6          # interferente forte no feixe 0 (ignorado)
        gains[1, 0, 1] = 1e-12         # interferente fraco no feixe 1 (conta)
        ues = [_ue(beam=1, channel=1), _ue(beam=0, channel=1)]
        s = SINR(ues, 1, gains)
        assert s[0] == pytest.approx(1e-9 / (1e-12 + _pn(1)))

    def test_ue_em_outro_ap_so_interfere_via_ganho_no_ap_servidor(self):
        gains = np.zeros((2, 2, 1))
        gains[0, 0, 0] = 1e-9
        gains[1, 1, 0] = 1e-9
        gains[1, 0, 0] = 1e-11         # UE 1 recebido no AP 0
        ues = [_ue(ap_id=0), _ue(ap_id=1)]
        s = SINR(ues, 1, gains)
        assert s[0] == pytest.approx(1e-9 / (1e-11 + _pn(1)))

    def test_cenario_real(self, cenario_com_attach):
        ues, gains = cenario_com_attach["ues"], cenario_com_attach["gains"]
        s = np.asarray(SINR(ues, 4, gains))
        assert s.shape == (len(ues),)
        assert np.all(np.isfinite(s)) and np.all(s > 0)


# ---------------------------------------------------------------------------
# SINR adaptativo
# ---------------------------------------------------------------------------

class TestSINRAdaptive:
    def test_ue_isolado_multiplica_pelo_ganho_mrc(self):
        gains = np.full((1, 1, 1), 1e-9)
        Gm = np.array([[8.0]])
        s = SINR_adaptive([_ue()], 1, gains, Gm)
        assert s[0] == pytest.approx(8.0 * 1e-9 / _pn(1))

    def test_interferencia_usa_ganho_do_feixe_do_desejado(self):
        """I = g[i]·P[i]·Gm[k, i]: o feixe do UE k atenua o UE interferente i."""
        gains = np.zeros((2, 1, 1))
        gains[0, 0, 0], gains[1, 0, 0] = 1e-9, 1e-9
        Gm = np.array([[8.0, 0.5],
                       [0.2, 8.0]])
        ues = [_ue(channel=1), _ue(channel=1)]
        s = SINR_adaptive(ues, 1, gains, Gm)
        assert s[0] == pytest.approx(8e-9 / (0.5e-9 + _pn(1)))
        assert s[1] == pytest.approx(8e-9 / (0.2e-9 + _pn(1)))

    def test_canais_distintos_sem_interferencia(self):
        gains = np.full((2, 1, 1), 1e-9)
        Gm = np.array([[8.0, 1.0], [1.0, 8.0]])
        ues = [_ue(channel=1), _ue(channel=2)]
        s = SINR_adaptive(ues, 1, gains, Gm)
        assert s[0] == pytest.approx(8e-9 / _pn(1))

    def test_feixe_nulo_na_interferencia_equivale_a_isolado(self):
        gains = np.full((2, 1, 1), 1e-9)
        Gm = np.array([[8.0, 0.0], [0.0, 8.0]])
        ues = [_ue(channel=1), _ue(channel=1)]
        s = SINR_adaptive(ues, 1, gains, Gm)
        assert s[0] == pytest.approx(8e-9 / _pn(1))


# ---------------------------------------------------------------------------
# channel_capacity
# ---------------------------------------------------------------------------

class TestChannelCapacity:
    def test_sinr_zero_capacidade_zero(self):
        assert channel_capacity([0.0], N=1)[0] == 0.0

    def test_valor_conhecido(self):
        """SINR=1, N=1: 1e8 · log2(2) / 1e6 = 100 Mbps."""
        assert channel_capacity([1.0], N=1)[0] == pytest.approx(100.0)

    def test_banda_por_canal_dividida_por_n(self):
        c1 = channel_capacity([3.0], N=1)[0]
        c4 = channel_capacity([3.0], N=4)[0]
        assert c4 == pytest.approx(c1 / 4)

    def test_monotonica_no_sinr(self):
        c = channel_capacity(np.array([0.1, 1.0, 10.0, 100.0]), N=1)
        assert np.all(np.diff(c) > 0)

    def test_retorna_array_do_mesmo_tamanho(self):
        c = channel_capacity([1.0, 2.0, 3.0], N=2)
        assert isinstance(c, np.ndarray) and c.shape == (3,)


# ---------------------------------------------------------------------------
# tabela_kpis
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def res():
    return estudo_mobilidade(
        velocidades={"Teste": 50}, periodos=(10, 50), arranjos=(8, 16),
        distancias=(50.0,), duracao_s=0.2)


class TestTabelaKpis:
    def test_imprime_cabecalho_e_uma_linha_por_cenario(self, res, capsys):
        tabela_kpis(res)
        linhas = capsys.readouterr().out.strip().splitlines()
        assert "Mobilidade" in linhas[0] and "Perda(dB)" in linhas[0]
        assert len(linhas) == 1 + len(res)

    def test_linhas_contem_nome_e_parametros(self, res, capsys):
        tabela_kpis(res)
        out = capsys.readouterr().out
        assert "Teste" in out
        assert "50" in out

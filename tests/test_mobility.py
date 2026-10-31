# ---------------------------------------------------------------------------
# test_mobility.py — Trajetória, beam tracking e estudo de mobilidade
# ---------------------------------------------------------------------------

import numpy as np
import pytest

from simulator.mobility import (simular_trajetoria, estudo_mobilidade,
                                SINR_CAP_DB, VELOCIDADES_KMH, PERIODOS_MS,
                                ARRANJOS_L, DISTANCIAS_M)


def traj(**kw):
    base = dict(v_kmh=120, d_perp=25.0, T_ms=50, L=16, seed=0)
    base.update(kw)
    return simular_trajetoria(**base)


# ---------------------------------------------------------------------------
# Estrutura e geometria
# ---------------------------------------------------------------------------

class TestEstrutura:
    def test_chaves_e_tamanho(self):
        r = traj()
        n = 2000                                   # 2 s com dt = 1 ms
        for k in ("t_ms", "y", "phi", "phi_hat", "gain", "gain_db", "loss_db",
                  "snr_db", "se_gross", "se_net", "erro_total_deg"):
            assert len(r[k]) == n, k
        for k in ("loss_medio_db", "loss_max_db", "se_gross_medio",
                  "se_net_medio", "overhead"):
            assert np.isscalar(r[k]), k
        assert r["params"] == dict(v_kmh=120, d_perp=25.0, T_ms=50, L=16)

    def test_dt_e_duracao_definem_n(self):
        r = traj(dt_ms=2.0, duracao_s=1.0)
        assert len(r["t_ms"]) == 500
        assert r["t_ms"][1] - r["t_ms"][0] == 2.0

    def test_ue_cruza_o_broadside_no_instante_central(self):
        r = traj()
        assert abs(r["phi"][1000]) < 1e-12
        assert r["phi"][0] < 0 < r["phi"][-1]

    def test_trajetoria_simetrica(self):
        r = traj()
        np.testing.assert_allclose(r["phi"][500], -r["phi"][1500], atol=1e-12)

    def test_deslocamento_compativel_com_velocidade(self):
        """Em 1 s o UE a 36 km/h (10 m/s) percorre 10 m."""
        r = traj(v_kmh=36)
        assert r["y"][1500] - r["y"][500] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Ganho e perda de desalinhamento
# ---------------------------------------------------------------------------

class TestGanhoEPerda:
    def test_ganho_nunca_excede_L(self):
        r = traj(L=16)
        assert np.all(r["gain"] <= 16 + 1e-9)

    def test_perda_nao_negativa(self):
        r = traj()
        assert np.all(r["loss_db"] >= -1e-9)

    def test_perda_e_ideal_menos_real_em_db(self):
        r = traj(L=16)
        np.testing.assert_allclose(
            r["loss_db"], 10 * np.log10(16) - r["gain_db"], atol=1e-6)

    def test_sem_erro_e_atualizacao_a_cada_passo_perda_nula(self):
        """T = dt e σ = 0 → feixe sempre perfeito."""
        r = traj(T_ms=1, sigma_est_deg=0.0)
        np.testing.assert_allclose(r["loss_db"], 0.0, atol=1e-6)
        np.testing.assert_allclose(r["erro_total_deg"], 0.0, atol=1e-9)

    def test_ganho_perfeito_logo_apos_atualizacao_sem_ruido(self):
        """σ = 0: no instante da atualização o feixe está exatamente no UE."""
        r = traj(T_ms=100, sigma_est_deg=0.0)
        idx = np.arange(0, 2000, 100)
        np.testing.assert_allclose(r["gain"][idx], 16.0, rtol=1e-9)

    def test_perda_cresce_com_o_periodo_de_atualizacao(self):
        lentos = traj(T_ms=100, L=32, sigma_est_deg=0.0)["loss_medio_db"]
        rapidos = traj(T_ms=10, L=32, sigma_est_deg=0.0)["loss_medio_db"]
        assert lentos > rapidos

    def test_perda_cresce_com_a_velocidade(self):
        ped = traj(v_kmh=3, L=32, T_ms=100, sigma_est_deg=0.0)["loss_medio_db"]
        rod = traj(v_kmh=120, L=32, T_ms=100, sigma_est_deg=0.0)["loss_medio_db"]
        assert rod > ped

    def test_arranjo_maior_e_mais_sensivel_ao_lag(self):
        l8 = traj(L=8, T_ms=100, sigma_est_deg=0.0)["loss_medio_db"]
        l32 = traj(L=32, T_ms=100, sigma_est_deg=0.0)["loss_medio_db"]
        assert l32 > l8

    def test_pedestre_tem_perda_desprezivel(self):
        assert traj(v_kmh=3, L=8, T_ms=100)["loss_medio_db"] < 0.1

    def test_ue_parado_erro_e_so_estimacao(self):
        """v = 0: sem lag; erro total = −ε_est, constante dentro de cada janela T."""
        r = traj(v_kmh=0, T_ms=10, L=8)
        blocos = np.rad2deg(-r["phi_hat"] + r["phi"]).reshape(-1, 10)
        np.testing.assert_allclose(blocos, np.repeat(blocos[:, :1], 10, axis=1),
                                   atol=1e-12)

    def test_desvio_do_erro_de_estimacao_e_sigma(self):
        r = traj(v_kmh=0, T_ms=10, duracao_s=50.0, sigma_est_deg=1.0)
        amostras = r["erro_total_deg"][::10]       # uma por atualização
        assert np.std(amostras) == pytest.approx(1.0, abs=0.1)
        assert abs(np.mean(amostras)) < 0.1


# ---------------------------------------------------------------------------
# Enlace e eficiência espectral
# ---------------------------------------------------------------------------

class TestEnlace:
    def test_snr_respeita_teto_de_20_db(self):
        r = traj(d_perp=1.0)                       # muito perto → SNR altíssimo
        assert np.max(r["snr_db"]) <= SINR_CAP_DB + 1e-9
        assert np.max(r["snr_db"]) == pytest.approx(SINR_CAP_DB)

    def test_se_bruta_limitada_pelo_teto(self):
        r = traj(d_perp=1.0)
        assert np.max(r["se_gross"]) <= np.log2(1 + 10 ** (SINR_CAP_DB / 10)) + 1e-9

    def test_snr_cai_com_a_distancia(self):
        perto = traj(d_perp=50.0)["snr_db"].mean()
        longe = traj(d_perp=250.0)["snr_db"].mean()
        assert perto > longe

    def test_overhead_e_pilot_sobre_T(self):
        r = traj(T_ms=10, t_pilot_ms=0.5)
        assert r["overhead"] == pytest.approx(0.05)
        np.testing.assert_allclose(r["se_net"], 0.95 * r["se_gross"])

    def test_se_liquida_nunca_excede_bruta(self):
        r = traj()
        assert np.all(r["se_net"] <= r["se_gross"] + 1e-12)

    def test_overhead_satura_em_um(self):
        r = traj(T_ms=0.25, t_pilot_ms=0.5)
        assert r["overhead"] == 1.0
        np.testing.assert_allclose(r["se_net"], 0.0)

    def test_menor_T_aumenta_overhead(self):
        assert traj(T_ms=10)["overhead"] > traj(T_ms=100)["overhead"]

    def test_se_liquida_media_e_media_da_serie(self):
        r = traj()
        assert r["se_net_medio"] == pytest.approx(np.mean(r["se_net"]))


# ---------------------------------------------------------------------------
# Reprodutibilidade (semente reiniciada por cenário)
# ---------------------------------------------------------------------------

class TestSemente:
    def test_mesma_semente_mesmo_resultado(self):
        np.testing.assert_array_equal(traj(seed=3)["gain"], traj(seed=3)["gain"])

    def test_sementes_diferentes_mudam_o_erro(self):
        assert not np.array_equal(traj(seed=1)["phi_hat"], traj(seed=2)["phi_hat"])

    def test_semente_global_nao_afeta_o_resultado(self):
        """A semente é reiniciada dentro de cada cenário."""
        np.random.seed(111)
        a = traj(seed=5)["gain"]
        np.random.seed(222)
        b = traj(seed=5)["gain"]
        np.testing.assert_array_equal(a, b)

    def test_erro_de_estimacao_identico_entre_velocidades(self):
        """Mesma semente e mesmo T → mesmas ε_est (comparação justa)."""
        a = traj(v_kmh=3, T_ms=50, seed=9)
        b = traj(v_kmh=120, T_ms=50, seed=9)
        eps_a = (a["phi_hat"] - a["phi"][(np.arange(2000) // 50) * 50])
        eps_b = (b["phi_hat"] - b["phi"][(np.arange(2000) // 50) * 50])
        np.testing.assert_allclose(eps_a, eps_b, atol=1e-12)


# ---------------------------------------------------------------------------
# estudo_mobilidade
# ---------------------------------------------------------------------------

class TestEstudoMobilidade:
    def test_grid_padrao(self):
        assert len(VELOCIDADES_KMH) == 3 and PERIODOS_MS == (100, 50, 10)
        assert ARRANJOS_L == (8, 16, 32) and len(DISTANCIAS_M) == 3
        res = estudo_mobilidade(duracao_s=0.1)
        assert len(res) == 3 * 3 * 3 * 3

    def test_chaves_e_parametros(self):
        res = estudo_mobilidade(velocidades={"a": 3, "b": 50}, periodos=(10,),
                                arranjos=(8, 16), distancias=(25.0,),
                                duracao_s=0.1)
        assert set(res) == {(n, 10, L, 25.0) for n in "ab" for L in (8, 16)}
        r = res[("b", 10, 16, 25.0)]
        assert r["params"] == dict(v_kmh=50, d_perp=25.0, T_ms=10, L=16)

    def test_kwargs_repassados(self):
        res = estudo_mobilidade(velocidades={"a": 3}, periodos=(10,),
                                arranjos=(8,), distancias=(25.0,),
                                duracao_s=0.1, dt_ms=2.0)
        assert len(res[("a", 10, 8, 25.0)]["t_ms"]) == 50

    def test_cenarios_compartilham_a_mesma_semente(self):
        """Mesmo T e mesma semente → mesmo ε_est em cenários diferentes."""
        res = estudo_mobilidade(velocidades={"a": 0}, periodos=(10,),
                                arranjos=(8,), distancias=(25.0, 100.0),
                                duracao_s=0.1, seed=4)
        a, b = res[("a", 10, 8, 25.0)], res[("a", 10, 8, 100.0)]
        np.testing.assert_allclose(a["erro_total_deg"], b["erro_total_deg"])

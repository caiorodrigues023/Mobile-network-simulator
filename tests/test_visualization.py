# ---------------------------------------------------------------------------
# test_visualization.py — Testes de fumaça dos gráficos (backend Agg)
# ---------------------------------------------------------------------------
# Verificam que cada função executa sem erro e grava o arquivo esperado.
# ---------------------------------------------------------------------------

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import simulator.visualization as viz
from simulator.entities import UE
from simulator.mobility import estudo_mobilidade
from simulator.runner import simular_experimento

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(autouse=True)
def limpar_figuras():
    UE.id_counter = 0
    yield
    plt.close("all")


@pytest.fixture
def figs_dir(tmp_path, monkeypatch):
    """Redireciona FIGURES_DIR para uma pasta temporária."""
    monkeypatch.setattr(viz, "FIGURES_DIR", tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def res_mob():
    return estudo_mobilidade(periodos=(100, 10), arranjos=(8, 32),
                             distancias=(25.0, 100.0), duracao_s=0.5)


@pytest.fixture(scope="module")
def res_mc():
    np.random.seed(0)
    modelos = ("omni", "smb")
    res = simular_experimento(M=4, N=4, K=3, sim=3, antenna_models=modelos)
    return {m: {"K=3": res[m]} for m in modelos}


# ---------------------------------------------------------------------------
# _salvar
# ---------------------------------------------------------------------------

class TestSalvar:
    def test_relativo_vai_para_figures_dir(self, figs_dir):
        fig = plt.figure()
        p = viz._salvar(fig, "a.png")
        assert p == figs_dir / "a.png" and p.exists()

    def test_cria_subpastas(self, figs_dir):
        fig = plt.figure()
        p = viz._salvar(fig, "x/y/z.png")
        assert p == figs_dir / "x" / "y" / "z.png" and p.exists()

    def test_absoluto_e_respeitado(self, figs_dir, tmp_path_factory):
        outro = tmp_path_factory.mktemp("outro") / "f.png"
        p = viz._salvar(plt.figure(), str(outro))
        assert p == outro and outro.exists()
        assert not (figs_dir / "f.png").exists()


# ---------------------------------------------------------------------------
# Gráficos do Monte Carlo
# ---------------------------------------------------------------------------

class TestGraficosMonteCarlo:
    def test_plotar_mapa(self, cenario_basico, figs_dir):
        c = cenario_basico
        viz.plotar_mapa(c["aps"], c["ues"], shadowing=c["shadowing"],
                        models=["omni", "sector", "smb", "adaptive"],
                        savepath="mapas/mapa.png", show=False)
        assert (figs_dir / "mapas" / "mapa.png").exists()

    def test_plotar_mapa_sem_shadowing_dado(self, cenario_basico, figs_dir):
        c = cenario_basico
        viz.plotar_mapa(c["aps"], c["ues"], models=["omni"],
                        savepath="m.png", show=False)
        assert (figs_dir / "m.png").exists()

    def test_plot_padroes_smb(self, figs_dir):
        fig = viz.plot_padroes_smb(configs=[(8, 8), (8, 16), (4, 8)],
                                   savepath="smb.png", show=False)
        assert fig is not None and (figs_dir / "smb.png").exists()
        assert sum(ax.get_visible() for ax in fig.axes) == 3   # 4º painel oculto

    def test_plot_all_cdfs_e_compare_kpis(self, res_mc):
        viz.plot_all_cdfs(res_mc, ["K=3"], models=["omni", "smb"])
        viz.compare_kpis(res_mc, ["K=3"], models=["omni", "smb"])
        assert len(plt.get_fignums()) == 2

    def test_plot_cdfs_legado(self):
        sinr = [list(np.random.uniform(0.1, 10, 20)) for _ in range(2)]
        cap = [list(np.random.uniform(1, 50, 20)) for _ in range(2)]
        viz.plot_cdfs(sinr, cap)
        assert len(plt.get_fignums()) == 1


# ---------------------------------------------------------------------------
# Gráficos de mobilidade
# ---------------------------------------------------------------------------

class TestGraficosMobilidade:
    def test_ganho_vs_tempo(self, res_mob, figs_dir):
        fig = viz.plot_ganho_vs_tempo(res_mob, L=32, d_perp=25.0,
                                      periodos=(100, 10), janela_ms=300,
                                      savepath="mob/ganho.png", show=False)
        assert (figs_dir / "mob" / "ganho.png").exists()
        assert len(fig.axes) == 3                              # 1 por velocidade

    def test_ganho_vs_tempo_curvas_e_ideal(self, res_mob, figs_dir):
        fig = viz.plot_ganho_vs_tempo(res_mob, L=32, d_perp=25.0,
                                      periodos=(100, 10), show=False)
        ax = fig.axes[0]
        assert len(ax.lines) == 3                              # 2 curvas + ideal
        ideal = [l for l in ax.lines if l.get_linestyle() == "--"][0]
        assert ideal.get_ydata()[0] == pytest.approx(10 * np.log10(32))

    def test_perda_desalinhamento(self, res_mob, figs_dir):
        fig = viz.plot_perda_desalinhamento(
            res_mob, L=8, d_perp=100.0, periodos=(100, 10), janela_ms=200,
            savepath="mob/perda.png", show=False)
        assert (figs_dir / "mob" / "perda.png").exists()
        assert len(fig.axes[0].lines) == 2

    def test_perda_por_arranjo(self, res_mob, figs_dir):
        fig = viz.plot_perda_por_arranjo(
            res_mob, d_perp=25.0, periodos=(100, 10), arranjos=(8, 32),
            savepath="mob/arranjo.png", show=False)
        assert (figs_dir / "mob" / "arranjo.png").exists()
        assert len(fig.axes) == 3 and len(fig.axes[0].lines) == 2

    def test_tradeoff_se(self, res_mob, figs_dir):
        fig = viz.plot_tradeoff_se(
            res_mob, d_perp=100.0, periodos=(100, 10), arranjos=(8, 32),
            savepath="mob/tradeoff.png", show=False)
        assert (figs_dir / "mob" / "tradeoff.png").exists()
        assert len(fig.axes[0].lines) == 4                     # líquida + bruta × 2 L

    def test_tradeoff_liquida_abaixo_da_bruta(self, res_mob):
        fig = viz.plot_tradeoff_se(res_mob, d_perp=100.0, periodos=(100, 10),
                                   arranjos=(8,), show=False)
        liq, bruta = fig.axes[0].lines
        assert np.all(np.asarray(liq.get_ydata())
                      <= np.asarray(bruta.get_ydata()) + 1e-12)

    def test_sem_savepath_nao_grava(self, res_mob, figs_dir):
        viz.plot_tradeoff_se(res_mob, d_perp=100.0, periodos=(100, 10),
                             arranjos=(8, 32), show=False)
        assert list(Path(figs_dir).iterdir()) == []

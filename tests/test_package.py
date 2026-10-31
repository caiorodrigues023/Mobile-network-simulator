# ---------------------------------------------------------------------------
# test_package.py — API pública (__init__) e configuração
# ---------------------------------------------------------------------------

import importlib

import pytest

import simulator
from simulator import config


class TestExports:
    def test_all_e_importavel(self):
        for nome in simulator.__all__:
            assert hasattr(simulator, nome), f"'{nome}' em __all__ mas ausente"

    def test_all_sem_duplicatas(self):
        assert len(simulator.__all__) == len(set(simulator.__all__))

    @pytest.mark.parametrize("nome", [
        "AP", "UE", "simular_experimento", "cenario_aleatorio", "plotar_mapa",
        "plot_all_cdfs", "compare_kpis", "plot_padroes_smb", "SINR",
        "SINR_adaptive", "channel_capacity", "tabela_kpis", "DPC",
        "simular_trajetoria", "estudo_mobilidade", "plot_ganho_vs_tempo",
        "plot_perda_desalinhamento", "plot_perda_por_arranjo",
        "plot_tradeoff_se",
    ])
    def test_simbolos_publicos(self, nome):
        assert nome in simulator.__all__

    @pytest.mark.parametrize("nome,modulo", [
        ("tabela_kpis", "metrics"),
        ("plot_ganho_vs_tempo", "visualization"),
        ("plot_tradeoff_se", "visualization"),
        ("estudo_mobilidade", "mobility"),
    ])
    def test_funcoes_vivem_no_modulo_esperado(self, nome, modulo):
        assert getattr(simulator, nome).__module__ == f"simulator.{modulo}"

    def test_mobility_nao_depende_de_matplotlib(self):
        src = importlib.import_module("simulator.mobility")
        assert "plt" not in vars(src)


class TestConfig:
    def test_labels_e_cores_cobrem_os_mesmos_modelos(self):
        assert set(config.MODEL_LABELS) == set(config.MODEL_COLORS)

    def test_modelos_base_presentes(self):
        for m in ("omni", "sector", "smb", "adaptive"):
            assert m in config.MODEL_LABELS

    def test_lambda_unitario(self):
        assert config._LAMBDA == 1.0

    def test_figures_dir(self):
        assert config.FIGURES_DIR.name == "figures"
        assert config.FIGURES_DIR.is_absolute()

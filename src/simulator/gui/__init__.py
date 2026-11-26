# ---------------------------------------------------------------------------
# src/simulator/gui/__init__.py
# ---------------------------------------------------------------------------
"""Pacote da Interface Gráfica Interativa do Simulador de Redes Móveis."""

__all__ = ["SimulatorMainWindow"]


def __getattr__(name: str):
    if name == "SimulatorMainWindow":
        from .main_window import SimulatorMainWindow

        return SimulatorMainWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

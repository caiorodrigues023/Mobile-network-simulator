# ---------------------------------------------------------------------------
# __init__.py — Transforma src/simulator/ em um pacote Python exportável
# ---------------------------------------------------------------------------
# Re-exporta os símbolos mais usados para facilitar imports diretos:
#   from simulator import AP, UE, simular_experimento, plotar_mapa

from .config import MODEL_LABELS, MODEL_COLORS
from .entities import AP, UE
from .antennas import smb_gains, mrc_gain, antenna_gain
from .channel import angle_matrix, gerar_shadowing, gain_matrix, adaptive_gains
from .allocation import attach_AP_UE, alocar_canais_ortogonal, DPC
from .metrics import SINR, SINR_adaptive, channel_capacity
from .runner import distribuir_AP, cenario_aleatorio, simular_experimento
from .visualization import (
    plotar_mapa,
    plot_cdfs,
    plot_all_cdfs,
    compare_kpis,
    plot_padroes_smb,
)

__all__ = [
    # config
    'MODEL_LABELS', 'MODEL_COLORS',
    # entities
    'AP', 'UE',
    # antennas
    'smb_gains', 'mrc_gain', 'antenna_gain',
    # channel
    'angle_matrix', 'gerar_shadowing', 'gain_matrix', 'adaptive_gains',
    # allocation
    'attach_AP_UE', 'alocar_canais_ortogonal', 'DPC',
    # metrics
    'SINR', 'SINR_adaptive', 'channel_capacity',
    # runner
    'distribuir_AP', 'cenario_aleatorio', 'simular_experimento',
    # visualization
    'plotar_mapa', 'plot_cdfs', 'plot_all_cdfs', 'compare_kpis',
    'plot_padroes_smb',
]

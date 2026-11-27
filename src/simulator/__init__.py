# ---------------------------------------------------------------------------
# __init__.py — Exportação pública do pacote simulator (API Híbrida + CLI)
# ---------------------------------------------------------------------------

from .allocation import (
    DPC,
    POWER_CONTROL_REGISTRY,
    DPC_Dual_Loop,
    DPC_Dual_Loop_Vectorized,
    allocate_channels_vec,
    alocar_canais_ortogonal,
    attach_AP_UE,
    channel_inversion_power_control,
    optimal_sinr_power_control,
    run_power_control,
)
from .antennas import (
    ANTENNA_REGISTRY,
    antenna_gain,
    calc_ula_gain,
    compute_antenna_gain_matrix,
    compute_beam_lobe_xy,
    mrc_gain,
    smb_gains,
    ula_2d_gain,
)
from .channel import (
    adaptive_gains,
    adaptive_gains_vec,
    angle_matrix,
    distance_matrix,
    gain_matrix,
    gerar_rayleigh,
    gerar_shadowing,
)
from .config import (
    ALLOCATION_LABELS,
    AP_PALETTE_HEX,
    MODEL_COLORS,
    MODEL_LABELS,
    PLOTLY_COLORS,
)
from .entities import AP, UE
from .handover import HANDOVER_REGISTRY, run_handover_simulation
from .metrics import (
    SINR,
    SINR_adaptive,
    area_spectrum_efficiency,
    calculate_sinr_vector,
    channel_capacity,
    evaluate_physical_kpis,
    jain_fairness,
)
from .mobility import (
    TRACKING_REGISTRY,
    run_mobile_sdma,
    run_mobility_scenario,
    run_mobility_tradeoff_study,
)
from .runner import (
    aplicar_controle_potencia,
    cenario_aleatorio,
    compute_coverage_heatmap,
    distribuir_AP,
    distribute_aps_xy,
    mc_round_seed,
    run_monte_carlo,
    run_single_snapshot,
    simular_experimento,
    simular_scichal4a,
    simular_scichal4a_vetorizado,
)
from .schemas import (
    AntennaConfig,
    APSnapshot,
    ChannelConfig,
    HandoverConfig,
    HandoverResult,
    MobilityConfig,
    MobilityResult,
    MonteCarloResult,
    PowerControlConfig,
    ScenarioMetrics,
    SDMAResult,
    SimulationConfig,
    SnapshotResult,
    UESnapshot,
)
from .visualization import (
    compare_kpis,
    plot_all_cdfs,
    plot_cdfs,
    plot_cenario_mobilidade,
    plot_padroes_smb,
    plot_scichal4a_kpis,
    plotar_convergencia_dual_loop,
    plotar_mapa,
)

__all__ = [
    # config
    "MODEL_LABELS",
    "MODEL_COLORS",
    "PLOTLY_COLORS",
    "AP_PALETTE_HEX",
    "ALLOCATION_LABELS",
    # schemas
    "ChannelConfig",
    "AntennaConfig",
    "PowerControlConfig",
    "MobilityConfig",
    "HandoverConfig",
    "SimulationConfig",
    "UESnapshot",
    "APSnapshot",
    "SnapshotResult",
    "ScenarioMetrics",
    "MonteCarloResult",
    "MobilityResult",
    "SDMAResult",
    "HandoverResult",
    # entities
    "AP",
    "UE",
    # antennas
    "ANTENNA_REGISTRY",
    "smb_gains",
    "mrc_gain",
    "calc_ula_gain",
    "ula_2d_gain",
    "antenna_gain",
    "compute_antenna_gain_matrix",
    "compute_beam_lobe_xy",
    # channel
    "distance_matrix",
    "angle_matrix",
    "gerar_shadowing",
    "gerar_rayleigh",
    "gain_matrix",
    "adaptive_gains",
    "adaptive_gains_vec",
    # allocation
    "POWER_CONTROL_REGISTRY",
    "attach_AP_UE",
    "allocate_channels_vec",
    "alocar_canais_ortogonal",
    "channel_inversion_power_control",
    "optimal_sinr_power_control",
    "run_power_control",
    "DPC",
    "DPC_Dual_Loop",
    "DPC_Dual_Loop_Vectorized",
    # metrics
    "SINR",
    "SINR_adaptive",
    "channel_capacity",
    "calculate_sinr_vector",
    "evaluate_physical_kpis",
    "jain_fairness",
    "area_spectrum_efficiency",
    # mobility & handover
    "TRACKING_REGISTRY",
    "run_mobility_scenario",
    "run_mobility_tradeoff_study",
    "run_mobile_sdma",
    "HANDOVER_REGISTRY",
    "run_handover_simulation",
    # runner
    "mc_round_seed",
    "distribute_aps_xy",
    "compute_coverage_heatmap",
    "run_single_snapshot",
    "run_monte_carlo",
    "distribuir_AP",
    "cenario_aleatorio",
    "aplicar_controle_potencia",
    "simular_experimento",
    "simular_scichal4a",
    "simular_scichal4a_vetorizado",
    # visualization
    "plotar_mapa",
    "plot_cdfs",
    "plot_all_cdfs",
    "compare_kpis",
    "plot_padroes_smb",
    "plot_scichal4a_kpis",
    "plot_cenario_mobilidade",
    "plotar_convergencia_dual_loop",
]

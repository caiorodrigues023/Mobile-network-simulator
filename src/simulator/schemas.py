# ---------------------------------------------------------------------------
# schemas.py — Camada de Contratos e Configuração (@dataclass + validação JSON)
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .config import (
    AREA_SIZE,
    B_TOTAL,
    D_0,
    GAMMA_MAX_DB,
    GAMMA_MIN_DB,
    GAMMA_TAR_INIT,
    HANDOVER_ACTIVE_SET_MARGIN_DB,
    HANDOVER_DIST_TOTAL_M,
    HANDOVER_HYSTERESIS_DB,
    HANDOVER_NUM_APS,
    HANDOVER_P_MIN_W,
    HANDOVER_START_MARGIN_M,
    HANDOVER_TS_S,
    HANDOVER_TTT_MS,
    HANDOVER_VELOCITY_MS,
    K_0,
    K_CONST,
    MOBILITY_DISTANCES_M,
    MOBILITY_DT,
    MOBILITY_SIGMA_THETA_DEG,
    MOBILITY_T_OVERHEAD,
    MOBILITY_T_SIM,
    MOBILITY_ULA_SIZES,
    MOBILITY_UPDATE_PERIODS_S,
    MU_STEP,
    P_MAX_DBM,
    P_MIN_DBM,
    PATHLOSS_EXP,
    SECTOR_BEAMWIDTH_DEG,
    SECTOR_G_MIN,
    SIGMA_R,
    SIGMA_X,
    T_INNER_DEFAULT,
    T_OUTER_DEFAULT,
    TOL_DB,
)

VALID_ANTENNA_MODELS = frozenset(
    {"omni", "sector", "smb", "mrc", "adaptive", "adaptive_1", "adaptive_5", "adaptive_10", "ula"}
)
VALID_POWER_ALGORITHMS = frozenset(
    {"full_power", "cipc", "optimal", "dpc_fixed", "dual_loop", "max_sum", "max_prod"}
)
VALID_ALLOCATION_POLICIES = frozenset({"full_reuse", "papoa", "papoa_ca", "random", "ortogonal"})
VALID_TRACKING_MODES = frozenset({"hold", "predictive"})
VALID_HANDOVER_MODES = frozenset({"hard", "soft"})
VALID_COMPARISON_MODES = frozenset({"auto", "antennas", "power", "both"})


def _is_perfect_square(n: int) -> bool:
    if n < 1:
        return False
    root = math.isqrt(n)
    return root * root == n


# ---------------------------------------------------------------------------
# Dataclasses de Configuração (Entrada)
# ---------------------------------------------------------------------------


@dataclass
class ChannelConfig:
    """Parâmetros físicos do canal de propagação e ruído térmico."""

    use_shadowing: bool = True
    sigma_x: float = SIGMA_X
    use_rayleigh: bool = False
    sigma_r: float = SIGMA_R
    pathloss_exp: float = PATHLOSS_EXP
    k_const: float = K_CONST
    d_0: float = D_0
    b_total_hz: float = B_TOTAL
    k_0: float = K_0

    def __post_init__(self) -> None:
        if self.sigma_x < 0.0:
            raise ValueError(f"sigma_x deve ser >= 0, recebido {self.sigma_x}")
        if self.sigma_r <= 0.0:
            raise ValueError(f"sigma_r deve ser > 0, recebido {self.sigma_r}")
        if self.pathloss_exp <= 0.0:
            raise ValueError(f"pathloss_exp deve ser > 0, recebido {self.pathloss_exp}")
        if self.k_const <= 0.0:
            raise ValueError(f"k_const deve ser > 0, recebido {self.k_const}")
        if self.d_0 <= 0.0:
            raise ValueError(f"d_0 deve ser > 0, recebido {self.d_0}")
        if self.b_total_hz <= 0.0:
            raise ValueError(f"b_total_hz deve ser > 0, recebido {self.b_total_hz}")
        if self.k_0 <= 0.0:
            raise ValueError(f"k_0 deve ser > 0, recebido {self.k_0}")


@dataclass
class AntennaConfig:
    """Parâmetros dos modelos de antena e arranjos MIMO (UCA/ULA/SMB)."""

    models: tuple[str, ...] = ("omni", "sector", "smb", "mrc", "ula")
    L: int = 8
    J: int = 8
    aoa_err_deg: float = 0.0
    beamwidth_deg: float = SECTOR_BEAMWIDTH_DEG
    g_min: float = SECTOR_G_MIN

    def __post_init__(self) -> None:
        self.models = tuple(self.models)
        if not self.models:
            raise ValueError("Selecione pelo menos um modelo de antena.")
        for m in self.models:
            if m not in VALID_ANTENNA_MODELS:
                raise ValueError(f"Modelo de antena desconhecido: {m!r}")
        if self.L < 1:
            raise ValueError(f"Número de antenas L deve ser >= 1, recebido {self.L}")
        if self.J < 1:
            raise ValueError(f"Número de feixes SMB J deve ser >= 1, recebido {self.J}")
        if self.aoa_err_deg < 0.0:
            raise ValueError(f"aoa_err_deg deve ser >= 0, recebido {self.aoa_err_deg}")
        if not (0.0 < self.beamwidth_deg <= 360.0):
            raise ValueError(f"beamwidth_deg deve estar em (0, 360], recebido {self.beamwidth_deg}")
        if not (0.0 < self.g_min <= 1.0):
            raise ValueError(f"g_min deve estar em (0, 1], recebido {self.g_min}")


@dataclass
class PowerControlConfig:
    """Parâmetros dos algoritmos de controle de potência e restrições físicas."""

    algorithms: tuple[str, ...] = ("full_power", "dpc_fixed", "dual_loop")
    p_max_dbm: float = P_MAX_DBM
    p_min_dbm: float = P_MIN_DBM
    mu_step_db: float = MU_STEP
    gamma_tar_init_db: float = GAMMA_TAR_INIT
    gamma_max_db: float = GAMMA_MAX_DB
    gamma_min_db: float = GAMMA_MIN_DB
    t_outer: int = T_OUTER_DEFAULT
    t_inner: int = T_INNER_DEFAULT
    tol_db: float = TOL_DB
    apply_physical_limits: bool = True
    sgd_iters: int = 60

    def __post_init__(self) -> None:
        self.algorithms = tuple(self.algorithms)
        if not self.algorithms:
            raise ValueError("Selecione pelo menos um algoritmo de controle de potência.")
        for a in self.algorithms:
            if a not in VALID_POWER_ALGORITHMS:
                raise ValueError(f"Algoritmo de potência desconhecido: {a!r}")
        if self.p_min_dbm > self.p_max_dbm:
            raise ValueError(
                f"p_min_dbm ({self.p_min_dbm}) deve ser <= p_max_dbm ({self.p_max_dbm})"
            )
        if self.gamma_min_db >= self.gamma_max_db:
            raise ValueError(
                f"gamma_min_db ({self.gamma_min_db}) deve ser < gamma_max_db ({self.gamma_max_db})"
            )
        if self.mu_step_db <= 0.0:
            raise ValueError(f"mu_step_db deve ser > 0, recebido {self.mu_step_db}")
        if self.t_outer < 1 or self.t_inner < 1:
            raise ValueError("t_outer e t_inner devem ser >= 1")
        if self.tol_db < 0.0:
            raise ValueError(f"tol_db deve ser >= 0, recebido {self.tol_db}")

    @property
    def p_max_w(self) -> float:
        return 10.0 ** ((self.p_max_dbm - 30.0) / 10.0)

    @property
    def p_min_w(self) -> float:
        return 10.0 ** ((self.p_min_dbm - 30.0) / 10.0)


@dataclass
class MobilityConfig:
    """Parâmetros da simulação time-driven de mobilidade, tracking preditivo e SDMA móvel."""

    velocities_kmh: tuple[float, ...] = (3.0, 50.0, 120.0)
    selected_velocity_kmh: float = 120.0
    delta_t_list_s: tuple[float, ...] = MOBILITY_UPDATE_PERIODS_S
    selected_delta_t_s: float = 0.100
    distances_m: tuple[float, ...] = MOBILITY_DISTANCES_M
    selected_distance_m: float = 50.0
    L_list: tuple[int, ...] = MOBILITY_ULA_SIZES
    selected_L: int = 32
    dt_s: float = MOBILITY_DT
    t_sim_s: float = MOBILITY_T_SIM
    t_overhead_s: float = MOBILITY_T_OVERHEAD
    sigma_theta_deg: float = MOBILITY_SIGMA_THETA_DEG
    array_type: str = "ula"  # "ula" ou "uca"
    compare_predictive: bool = True
    enable_sdma: bool = True
    sdma_k_ues: int = 4
    sdma_combiner: str = "both"  # "mrc", "zf", "both"
    mixed_speeds: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        self.velocities_kmh = tuple(float(v) for v in self.velocities_kmh)
        self.delta_t_list_s = tuple(float(d) for d in self.delta_t_list_s)
        self.distances_m = tuple(float(d) for d in self.distances_m)
        self.L_list = tuple(int(v) for v in self.L_list)
        if self.dt_s <= 0.0 or self.t_sim_s <= self.dt_s:
            raise ValueError("dt_s deve ser > 0 e t_sim_s > dt_s.")
        if self.selected_delta_t_s < self.dt_s:
            raise ValueError("selected_delta_t_s deve ser >= dt_s.")
        if self.selected_distance_m < 1.0:
            raise ValueError("selected_distance_m deve ser >= 1 m.")
        if self.selected_L < 2:
            raise ValueError("selected_L deve ser >= 2.")
        if self.sdma_k_ues < 1:
            raise ValueError("sdma_k_ues deve ser >= 1.")


@dataclass
class HandoverConfig:
    """Parâmetros do motor de Handover (Hard vs. Soft) ao longo da rodovia."""

    num_aps: int = HANDOVER_NUM_APS
    dist_total_m: float = HANDOVER_DIST_TOTAL_M
    start_margin_m: float = HANDOVER_START_MARGIN_M
    velocity_ms: float = HANDOVER_VELOCITY_MS
    t_sample_s: float = HANDOVER_TS_S
    p_tx_w: float = 1.0
    p_min_w: float = HANDOVER_P_MIN_W
    hysteresis_db: float = HANDOVER_HYSTERESIS_DB
    active_set_margin_db: float = HANDOVER_ACTIVE_SET_MARGIN_DB
    ttt_ms: float = HANDOVER_TTT_MS
    soft_combining: str = "selection"  # "selection" (Ex. 13) ou "macro_diversity"
    compare_hard_soft: bool = True
    mc_runs: int = 500
    t_ho_s: float = 20.0
    ho_step_s: float = 0.2
    p_min_dbm_network: float = -102.0
    shadowing_corr_dist_m: float = 50.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.num_aps < 2:
            raise ValueError(f"Handover requer num_aps >= 2, recebido {self.num_aps}")
        if self.dist_total_m <= 2.0 * self.start_margin_m:
            raise ValueError("dist_total_m deve ser > 2 * start_margin_m.")
        if self.velocity_ms <= 0.0 or self.t_sample_s <= 0.0:
            raise ValueError("velocity_ms e t_sample_s devem ser > 0.")
        if self.p_min_w <= 0.0 or self.p_tx_w <= 0.0:
            raise ValueError("p_min_w e p_tx_w devem ser > 0.")
        if self.hysteresis_db < 0.0 or self.active_set_margin_db < 0.0 or self.ttt_ms < 0.0:
            raise ValueError("Margens de histerese, conjunto ativo e TTT devem ser >= 0.")
        if self.mc_runs < 1:
            raise ValueError("mc_runs deve ser >= 1.")


@dataclass
class SimulationConfig:
    """Configuração mestre unificada da simulação (serializável em JSON)."""

    M: int = 9
    K: int = 4
    k_scenarios: tuple[int, ...] = (1, 4, 8)
    N: int = 1
    area_m: float = AREA_SIZE
    allocation_policy: str = "papoa"
    monte_carlo_runs: int = 500
    seed: int = 42
    ue_mode: str = "static"  # "static" ou "mobile"
    comparison_mode: str = "auto"  # "auto", "antennas", "power", "both"
    snapshot_antenna_model: str = "mrc"
    snapshot_power_mode: str = "dual_loop"
    h_ap_m: float = 25.0
    h_ue_m: float = 1.5
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    antenna: AntennaConfig = field(default_factory=AntennaConfig)
    power_control: PowerControlConfig = field(default_factory=PowerControlConfig)
    mobility: MobilityConfig = field(default_factory=MobilityConfig)
    handover: HandoverConfig = field(default_factory=HandoverConfig)

    def __post_init__(self) -> None:
        if not _is_perfect_square(self.M):
            raise ValueError(
                f"M deve ser um quadrado perfeito (1, 4, 9, 16, 25...), recebido {self.M}"
            )
        if self.K < 1:
            raise ValueError(f"K deve ser >= 1, recebido {self.K}")
        self.k_scenarios = tuple(sorted({int(k) for k in self.k_scenarios if int(k) >= 1}))
        if not self.k_scenarios:
            self.k_scenarios = (self.K,)
        if self.N < 1:
            raise ValueError(f"N deve ser >= 1, recebido {self.N}")
        if self.area_m <= 0.0:
            raise ValueError(f"area_m deve ser > 0, recebido {self.area_m}")
        if self.allocation_policy == "ortogonal":
            self.allocation_policy = "papoa"
        if self.allocation_policy not in VALID_ALLOCATION_POLICIES:
            raise ValueError(f"Política de alocação desconhecida: {self.allocation_policy!r}")
        if self.monte_carlo_runs < 1:
            raise ValueError("monte_carlo_runs deve ser >= 1")
        if self.comparison_mode not in VALID_COMPARISON_MODES:
            raise ValueError(f"comparison_mode desconhecido: {self.comparison_mode!r}")
        if self.snapshot_antenna_model not in VALID_ANTENNA_MODELS:
            raise ValueError(
                f"snapshot_antenna_model desconhecido: {self.snapshot_antenna_model!r}"
            )
        if self.snapshot_power_mode not in VALID_POWER_ALGORITHMS:
            raise ValueError(f"snapshot_power_mode desconhecido: {self.snapshot_power_mode!r}")

    @property
    def bw_channel_hz(self) -> float:
        return self.channel.b_total_hz / self.N

    @property
    def noise_power_w(self) -> float:
        return self.bw_channel_hz * self.channel.k_0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationConfig:
        d = dict(data)
        if isinstance(d.get("channel"), dict):
            d["channel"] = ChannelConfig(**d["channel"])
        if isinstance(d.get("antenna"), dict):
            d["antenna"] = AntennaConfig(**d["antenna"])
        if isinstance(d.get("power_control"), dict):
            d["power_control"] = PowerControlConfig(**d["power_control"])
        if isinstance(d.get("mobility"), dict):
            d["mobility"] = MobilityConfig(**d["mobility"])
        if isinstance(d.get("handover"), dict):
            d["handover"] = HandoverConfig(**d["handover"])
        return cls(**d)

    @classmethod
    def from_json(cls, raw: str) -> SimulationConfig:
        return cls.from_dict(json.loads(raw))


# ---------------------------------------------------------------------------
# Dataclasses de Resultado (Saída Padronizada)
# ---------------------------------------------------------------------------


@dataclass
class UESnapshot:
    """Estado detalhado de um UE no snapshot (alimenta hover e tabela)."""

    ue_id: int
    x: float
    y: float
    z: float
    serving_ap: int
    ap_x: float
    ap_y: float
    distance_m: float
    aoa_deg: float
    channel: int
    serving_beam: int
    tx_power_w: float
    tx_power_dbm: float
    rx_gain_db: float
    snr_db: float
    sinr_lin: float
    sinr_db: float
    se_bps_hz: float
    capacity_mbps: float
    is_outage: bool
    is_blocked: bool
    status: str
    gamma_tar_db: float | None = None
    is_aggregated: bool = False


@dataclass
class APSnapshot:
    """Resumo agregado de um AP no snapshot (alimenta hover de AP)."""

    ap_id: int
    x: float
    y: float
    z: float
    connected_ue_ids: list[int]
    num_ues: int
    sum_capacity_mbps: float
    num_outage: int


@dataclass
class SnapshotResult:
    """Contrato completo de saída de um único snapshot da rede."""

    seed: int
    area_m: float
    antenna_model: str
    power_mode: str
    allocation_policy: str
    L: int
    J: int
    N: int
    beamwidth_deg: float
    g_min: float
    ap_xy: np.ndarray  # (M, 2)
    ue_xy: np.ndarray  # (K, 2)
    serving_ap: np.ndarray  # (K,)
    serving_beam: np.ndarray  # (K,)
    channels: np.ndarray  # (K,)
    distances_m: np.ndarray  # (K,)
    aoa_rad: np.ndarray  # (K,)
    beam_angles_rad: np.ndarray  # (K,) direção efetiva do feixe (inclui erro AoA)
    G_eff: np.ndarray  # (K, K)
    ues: list[UESnapshot]
    aps: list[APSnapshot]
    avg_sinr_db: float
    p10_sinr_db: float
    sum_se_bps_hz: float
    sum_capacity_mbps: float
    total_tx_power_w: float
    energy_efficiency_mbit_j: float
    outage_rate_pct: float
    jain_fairness: float
    ase_bps_hz_km2: float
    convergence_history: dict[str, Any] | None = None
    heatmap: dict[str, Any] | None = None


@dataclass
class ScenarioMetrics:
    """Distribuições brutas e KPIs consolidados para um par (modelo/algoritmo, carga K)."""

    key: str
    label: str
    K: int
    sinr_lin: np.ndarray
    sinr_db: np.ndarray
    se_bps_hz: np.ndarray
    capacity_mbps: np.ndarray
    tx_power_w: np.ndarray
    rx_power_w: np.ndarray
    outage_flags: np.ndarray
    sum_cap_per_run_mbps: np.ndarray
    total_power_per_run_w: np.ndarray
    # KPIs consolidados
    border_se_p10: float
    sum_se_mean: float
    sum_cap_mean_mbps: float
    sinr_p10_db: float
    sinr_p50_db: float
    sinr_p90_db: float
    cap_p10_mbps: float
    mean_tx_power_w: float
    energy_efficiency_mbit_j: float
    outage_rate_pct: float
    jain_fairness: float
    ase_bps_hz_km2: float


@dataclass
class MonteCarloResult:
    """Contrato padronizado de saída de uma bateria Monte Carlo multi-carga."""

    config: SimulationConfig
    comparison_mode: str
    k_scenarios: tuple[int, ...]
    elapsed_s: float
    representative_snapshot: SnapshotResult
    # {model_key: {K_int: ScenarioMetrics}} — sem inconsistências K vs 'K=1'
    antenna_metrics: dict[str, dict[int, ScenarioMetrics]] = field(default_factory=dict)
    power_metrics: dict[str, dict[int, ScenarioMetrics]] = field(default_factory=dict)
    # Históricos de convergência Dual-Loop / iterativos salvos para inspeção
    convergence_histories: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    # Metadados determinísticos das rodadas para o Inspetor de Rodadas (K do snapshot)
    round_inspector: dict[str, Any] | None = None


@dataclass
class MobilityResult:
    """Saída da simulação time-driven de um UE móvel (Hold ou Preditivo)."""

    tracking_mode: str
    L: int
    v_kmh: float
    delta_t_s: float
    D_m: float
    sigma_theta_deg: float
    t_s: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    dist_m: np.ndarray
    theta_real_rad: np.ndarray
    theta_tgt_rad: np.ndarray
    gain_actual_lin: np.ndarray
    gain_actual_db: np.ndarray
    gain_ideal_db: np.ndarray
    misalignment_loss_db: np.ndarray
    snr_db: np.ndarray
    se_gross: np.ndarray
    se_net: np.ndarray
    mean_loss_db: float
    max_loss_db: float
    mean_se_gross: float
    mean_se_net: float
    overhead_fraction: float


@dataclass
class SDMAResult:
    """Saída da simulação de SDMA Multi-usuário Móvel (MRC vs. Zero-Forcing)."""

    K: int
    L: int
    delta_t_s: float
    tracking_mode: str
    t_s: np.ndarray
    ue_x_m: np.ndarray  # (T, K)
    ue_y_m: np.ndarray  # (T, K)
    ue_velocities_kmh: np.ndarray  # (K,)
    sinr_mrc_db: np.ndarray  # (T, K)
    sinr_zf_db: np.ndarray  # (T, K)
    interf_mrc_dbm: np.ndarray  # (T, K)
    interf_zf_dbm: np.ndarray  # (T, K)
    sum_se_mrc: np.ndarray  # (T,)
    sum_se_zf: np.ndarray  # (T,)
    mean_sum_se_mrc: float
    mean_sum_se_zf: float


@dataclass
class HandoverResult:
    """Saída comparativa do motor de Handover (Hard vs. Soft)."""

    config: HandoverConfig
    ap_positions_m: np.ndarray  # (M,)
    ue_positions_m: np.ndarray  # (S,)
    t_s: np.ndarray  # (S,)
    # Trajetória representativa (realização 0)
    rx_power_w: np.ndarray  # (M, S)
    rx_power_dbm: np.ndarray  # (M, S)
    hard_serving_ap: np.ndarray  # (S,) índices 1..M
    soft_serving_ap: np.ndarray  # (S,) índices 1..M
    hard_outage_mask: np.ndarray  # (S,)
    soft_outage_mask: np.ndarray  # (S,)
    hard_rx_power_w: np.ndarray  # (S,)
    soft_rx_power_w: np.ndarray  # (S,)
    hard_snr_db: np.ndarray  # (S,)
    soft_snr_db: np.ndarray  # (S,)
    # Estatísticas (trajetória + Monte Carlo)
    hard_handovers_sample: int
    soft_handovers_sample: int
    hard_handovers_mean: float
    soft_handovers_mean: float
    hard_outage_rate_pct: float
    soft_outage_rate_pct: float
    mean_snr_gain_soft_vs_hard_db: float


@dataclass
class NetworkMobilityResult:
    """
    Saída da simulação time-driven de mobilidade e beam tracking na REDE COMPLETA
    (todos os M APs × K UEs em movimento sobre a área da simulação).
    """

    M: int
    K: int
    L: int
    area_m: float
    array_type: str
    delta_t_s: float
    dt_s: float
    t_overhead_s: float
    sigma_theta_deg: float
    seed: int
    ap_xy: np.ndarray  # (M, 2)
    t_s: np.ndarray  # (T,)
    ue_x_m: np.ndarray  # (T, K)
    ue_y_m: np.ndarray  # (T, K)
    ue_velocities_kmh: np.ndarray  # (K,)
    serving_ap: np.ndarray  # (T, K) índices 0..M-1
    update_indices: np.ndarray  # índices temporais onde ocorre medição ΔT
    beam_angles_final_rad: np.ndarray  # (K,)
    # Métricas de Beam Tracking (Hold vs. Preditivo) — todos os K UEs
    delta_theta_hold_deg: np.ndarray  # (T, K)
    delta_theta_pred_deg: np.ndarray  # (T, K)
    gain_ideal_db: float
    gain_hold_db: np.ndarray  # (T, K)
    gain_pred_db: np.ndarray  # (T, K)
    loss_hold_db: np.ndarray  # (T, K)
    loss_pred_db: np.ndarray  # (T, K)
    # Métricas de Rede / SDMA (MRC vs. Zero-Forcing) — avaliadas em t_eval_s
    t_eval_s: np.ndarray  # (Te,)
    sinr_hold_db: np.ndarray  # (Te, K)
    sinr_pred_db: np.ndarray  # (Te, K)
    sum_se_hold: np.ndarray  # (Te,)
    sum_se_pred: np.ndarray  # (Te,)
    sinr_mrc_db: np.ndarray  # (Te, K)
    sinr_zf_db: np.ndarray  # (Te, K)
    interf_mrc_dbm: np.ndarray  # (Te, K)
    interf_zf_dbm: np.ndarray  # (Te, K)
    sum_se_mrc: np.ndarray  # (Te,)
    sum_se_zf: np.ndarray  # (Te,)
    # KPIs agregados de toda a rede
    overhead_fraction: float
    mean_delta_theta_hold_deg: float
    max_delta_theta_hold_deg: float
    mean_delta_theta_pred_deg: float
    mean_loss_hold_db: float
    max_loss_hold_db: float
    mean_loss_pred_db: float
    max_loss_pred_db: float
    mean_sum_se_hold: float
    mean_sum_se_pred: float
    mean_sum_se_mrc: float
    mean_sum_se_zf: float


@dataclass
class NetworkHandoverResult:
    """
    Saída comparativa do motor de Handover (Hard vs. Soft) na REDE COMPLETA
    (todos os M APs × K UEs percorrendo a área da simulação).
    """

    config: HandoverConfig
    M: int
    K: int
    area_m: float
    seed: int
    p_min_dbm: float
    ap_xy: np.ndarray  # (M, 2)
    t_s: np.ndarray  # (T,)
    ue_x_m: np.ndarray  # (T, K)
    ue_y_m: np.ndarray  # (T, K)
    ue_velocities_kmh: np.ndarray  # (K,)
    rx_power_dbm: np.ndarray  # (T, K, M)
    hard_serving_ap: np.ndarray  # (T, K) índices 0..M-1
    soft_serving_ap: np.ndarray  # (T, K) índices 0..M-1
    hard_outage_mask: np.ndarray  # (T, K)
    soft_outage_mask: np.ndarray  # (T, K)
    hard_snr_db: np.ndarray  # (T, K)
    soft_snr_db: np.ndarray  # (T, K)
    # Agregados por UE e de toda a rede
    hard_handovers_per_ue: np.ndarray  # (K,)
    soft_handovers_per_ue: np.ndarray  # (K,)
    hard_outage_pct_per_ue: np.ndarray  # (K,)
    soft_outage_pct_per_ue: np.ndarray  # (K,)
    hard_outage_rate_pct: float
    soft_outage_rate_pct: float
    hard_handovers_total: int
    soft_handovers_total: int
    hard_handovers_mean: float
    soft_handovers_mean: float
    mean_snr_gain_soft_vs_hard_db: float

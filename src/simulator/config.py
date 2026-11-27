# ---------------------------------------------------------------------------
# config.py — Constantes físicas globais, rótulos, paletas e presets
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PARÂMETROS FÍSICOS E DE REDE (Partes 1–4 & SciChal4A)
# ---------------------------------------------------------------------------
AREA_SIZE: float = 1000.0  # Lado da área de cobertura quadrada (m)
B_TOTAL: float = 100e6  # Banda total = 100 MHz
K_0: float = 1e-20  # Densidade espectral de ruído térmico [W/Hz]
PATHLOSS_EXP: float = 4.0  # Expoente de perda de percurso (n)
D_0: float = 1.0  # Distância mínima de referência (m)
K_CONST: float = 1e-4  # Constante de propagação (k)
SIGMA_X: float = 2.0  # Desvio padrão do sombreamento log-normal (Np)
SIGMA_R: float = 0.7071067811865476  # Parâmetro de escala Rayleigh (1/sqrt(2))

# Controle de Potência (UD-DPC, Dual-Loop, CIPC, Optimal, MaxSum, MaxProd)
P_MAX_DBM: float = 30.0  # Potência máxima = 30 dBm (1 W)
P_MIN_DBM: float = 0.0  # Potência mínima para UE ativo = 0 dBm (1 mW)
MU_STEP: float = 1.0  # Passo do UD-DPC = 1 dBm
GAMMA_TAR_INIT: float = 10.0  # Meta inicial de SINR = 10 dB
GAMMA_MAX_DB: float = 20.0  # Teto físico de SINR (saturação QAM) = 20 dB
GAMMA_MIN_DB: float = -10.0  # Limiar de Outage / Bloqueio = -10 dB
TOL_DB: float = 0.5  # Margem de tolerância do loop externo (dB)
T_OUTER_DEFAULT: int = 35  # Iterações do loop externo (SciChal4A)
T_INNER_DEFAULT: int = 25  # Iterações do loop interno por rodada externa

# Antenas (comprimento de onda normalizado: R ∝ λ ⇒ ganho adimensional)
_LAMBDA: float = 1.0
SECTOR_BEAMWIDTH_DEG: float = 120.0
SECTOR_G_MIN: float = 0.01  # -20 dB

# ---------------------------------------------------------------------------
# Rótulos e Paletas (Matplotlib + Plotly Hex)
# ---------------------------------------------------------------------------
MODEL_LABELS: dict[str, str] = {
    # Antenas
    "omni": "Omni",
    "sector": "Setorial (3×120°)",
    "smb": "SMB (UCA)",
    "mrc": "MRC Adaptativo (UCA)",
    "adaptive": "MRC Adaptativo (Ideal 0°)",
    "adaptive_1": "MRC Adaptativo (Erro 1°)",
    "adaptive_5": "MRC Adaptativo (Erro 5°)",
    "adaptive_10": "MRC Adaptativo (Erro 10°)",
    "ula": "ULA / Massive MIMO",
    # Controle de Potência
    "full_power": "Full Power (P_max)",
    "cipc": "CIPC (Inversão de Canal)",
    "optimal": "Controle Ótimo (Target-SINR)",
    "dpc_fixed": "DPC Fixo (UD-DPC)",
    "dual_loop": "Dual-Loop PC (Outage)",
    "max_sum": "MaxSum (SGD)",
    "max_prod": "MaxProd (Proportional Fairness)",
}

MODEL_COLORS: dict[str, str] = {
    "omni": "tab:blue",
    "sector": "tab:orange",
    "smb": "tab:green",
    "mrc": "tab:red",
    "adaptive": "tab:red",
    "adaptive_1": "tab:purple",
    "adaptive_5": "tab:brown",
    "adaptive_10": "tab:pink",
    "ula": "tab:cyan",
    "full_power": "tab:gray",
    "cipc": "tab:olive",
    "optimal": "tab:purple",
    "dpc_fixed": "tab:blue",
    "dual_loop": "tab:orange",
    "max_sum": "tab:green",
    "max_prod": "tab:cyan",
}

PLOTLY_COLORS: dict[str, str] = {
    "omni": "#38bdf8",
    "sector": "#f59e0b",
    "smb": "#10b981",
    "mrc": "#f43f5e",
    "adaptive": "#f43f5e",
    "adaptive_1": "#8b5cf6",
    "adaptive_5": "#ec4899",
    "adaptive_10": "#a855f7",
    "ula": "#06b6d4",
    "full_power": "#94a3b8",
    "cipc": "#eab308",
    "optimal": "#a855f7",
    "dpc_fixed": "#38bdf8",
    "dual_loop": "#f97316",
    "max_sum": "#10b981",
    "max_prod": "#06b6d4",
}

AP_PALETTE_HEX: list[str] = [
    "#00b4d8",
    "#ff006e",
    "#10b981",
    "#f59e0b",
    "#8b5cf6",
    "#3b82f6",
    "#ef4444",
    "#84cc16",
    "#eab308",
    "#06b6d4",
    "#ec4899",
    "#14b8a6",
    "#6366f1",
    "#fb7185",
    "#22c55e",
    "#a855f7",
]

ALLOCATION_LABELS: dict[str, str] = {
    "full_reuse": "Reuso Total (Canal Único)",
    "papoa": "Ortogonal Intra-AP (PAPOA)",
    "papoa_ca": "PAPOA + Carrier Aggregation",
    "random": "Aleatória (Random)",
}

# ---------------------------------------------------------------------------
# Atividade 2 — Mobilidade, Beam Tracking e SDMA (SciChal4A Seção 2)
# ---------------------------------------------------------------------------
MOBILITY_DT: float = 1e-3  # passo de tempo (1 ms)
MOBILITY_T_SIM: float = 4.0  # janela de observação centrada no broadside (s)
MOBILITY_T_OVERHEAD: float = 1.5e-3  # overhead piloto por atualização (1.5 ms)
MOBILITY_SIGMA_THETA_DEG: float = 1.0  # erro padrão de estimação AoA no roteiro (1°)
MOBILITY_PROFILES_KMH: dict[str, float] = {
    "Pedestrian": 3.0,
    "Urban": 50.0,
    "Highway": 120.0,
}
MOBILITY_UPDATE_PERIODS_S: tuple[float, ...] = (0.100, 0.050, 0.010)  # ΔT: Slow, Fair, High
MOBILITY_ULA_SIZES: tuple[int, ...] = (8, 16, 32)  # L: conventional → massive
MOBILITY_DISTANCES_M: tuple[float, ...] = (25.0, 50.0, 75.0, 150.0)  # D: perto → borda

# ---------------------------------------------------------------------------
# Parte 2 (Exercícios 12 e 13) — Handover (Rodovia)
# ---------------------------------------------------------------------------
HANDOVER_DIST_TOTAL_M: float = 3100.0
HANDOVER_START_MARGIN_M: float = 50.0
HANDOVER_VELOCITY_MS: float = 20.0
HANDOVER_TS_S: float = 1.0
HANDOVER_P_MIN_W: float = 1e-17
HANDOVER_NUM_APS: int = 2
HANDOVER_HYSTERESIS_DB: float = 0.0  # 0.0 = modo reativo fiel ao Exercício 13
HANDOVER_ACTIVE_SET_MARGIN_DB: float = 0.0  # 0.0 = seleção pura fiel ao Exercício 13
HANDOVER_TTT_MS: float = 0.0

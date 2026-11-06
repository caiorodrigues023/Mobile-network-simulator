# ---------------------------------------------------------------------------
# config.py — Constantes físicas globais e paletas visuais
# ---------------------------------------------------------------------------

from pathlib import Path

# Pasta onde todas as figuras geradas são salvas (<raiz do projeto>/figures).
# Caminhos relativos passados em 'savepath' são resolvidos a partir dela.
FIGURES_DIR: Path = Path(__file__).resolve().parents[2] / 'figures'

# ---------------------------------------------------------------------------
# PARÂMETROS FÍSICOS E DE REDE
# ---------------------------------------------------------------------------
B_TOTAL = 100e6        # Banda total = 100 MHz
K_0 = 1e-20            # Densidade espectral de ruído térmico [W/Hz]
PATHLOSS_EXP = 4.0     # Expoente de perda de percurso (n)
D_0 = 1.0              # Distância mínima de referência (metros)
K_CONST = 1e-4         # Constante de propagação
SIGMA_X = 2.0          # Desvio padrão do sombreamento lognormal
SIGMA_R = 0.707106     # Parâmetro de escala do fading Rayleigh (1/sqrt(2))

# Controle de Potência (UD-DPC e Dual-Loop)
P_MAX_DBM = 30.0       # Potência máxima = 30 dBm (1 W)
P_MIN_DBM = 0.0        # Potência mínima para UE ativo = 0 dBm (1 mW)
MU_STEP = 1.0          # Passo do UD-DPC = 1 dBm
GAMMA_TAR_INIT = 10.0  # Meta inicial = 10 dB
GAMMA_MAX_DB = 20.0    # Teto físico de SINR (saturação QAM) = 20 dB
GAMMA_MIN_DB = -10.0   # Limiar de Outage / Bloqueio = -10 dB

_LAMBDA: float = 1.0

# Parâmetros de enlace (mesmos de metrics.py / channel.py)
_BT = 1e8            # banda total (Hz)
_K0 = 1e-20          # densidade espectral de ruído
SINR_CAP_DB = 20.0   # limite físico das constelações QAM (rodapé do roteiro)

VELOCIDADES_KMH = {'Pedestre (3 km/h)': 3, 'Urbano (50 km/h)': 50,
                   'Rodovia (120 km/h)': 120}
PERIODOS_MS = (100, 50, 10)
ARRANJOS_L = (8, 16, 32)
DISTANCIAS_M = (25.0, 100.0, 250.0)   # perto do AP → borda da célula (M=4)

# Rótulos legíveis e cores para cada modelo de antena nos gráficos
MODEL_LABELS: dict = {
    'omni':        'Omni',
    'sector':      'Setorizado',
    'smb':         'SMB',
    'adaptive':    'Adaptativo (Ideal 0°)',
    'adaptive_1':  'Adaptativo (Erro 1°)',
    'adaptive_5':  'Adaptativo (Erro 5°)',
    'adaptive_10': 'Adaptativo (Erro 10°)',
}

MODEL_COLORS: dict = {
    'omni':        'tab:blue',
    'sector':      'tab:orange',
    'smb':         'tab:green',
    'adaptive':    'tab:red',
    'adaptive_1':  'tab:purple',
    'adaptive_5':  'tab:brown',
    'adaptive_10': 'tab:pink',
}

# ---------------------------------------------------------------------------
# Atividade 2 — Mobilidade e beam tracking (report_scichal4A, Seção 2)
# ---------------------------------------------------------------------------
MOBILITY_DT: float = 1e-3          # passo de tempo da simulação (s)
MOBILITY_T_SIM: float = 4.0        # janela de observação centrada no broadside (s)
MOBILITY_PROFILES_KMH: dict = {    # perfil → velocidade (km/h)
    'Pedestrian': 3.0,
    'Urban': 50.0,
    'Highway': 120.0,
}
MOBILITY_UPDATE_PERIODS_S: tuple = (0.100, 0.050, 0.010)   # ΔT: Slow, Fair, High
MOBILITY_ULA_SIZES: tuple = (8, 16, 32)                    # L: conventional → massive
MOBILITY_DISTANCES_M: tuple = (25.0, 50.0, 75.0, 150.0)    # D: perto → borda da célula
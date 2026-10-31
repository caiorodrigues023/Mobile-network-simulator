# ---------------------------------------------------------------------------
# config.py — Constantes físicas globais e paletas visuais
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PARÂMETROS FÍSICOS E DE REDE (SciChal4A)
# ---------------------------------------------------------------------------
B_TOTAL = 100e6  # Banda total = 100 MHz
K_0 = 1e-20  # Densidade espectral de ruído térmico [W/Hz]
PATHLOSS_EXP = 4.0  # Expoente de perda de percurso (n)
D_0 = 1.0  # Distância mínima de referência (metros)
K_CONST = 1e-4  # Constante de propagação
SIGMA_X = 2.0  # Desvio padrão do sombreamento lognormal
SIGMA_R = 0.707106  # Parâmetro de escala do fading Rayleigh (1/sqrt(2))

# Controle de Potência (UD-DPC e Dual-Loop)
P_MAX_DBM = 30.0  # Potência máxima = 30 dBm (1 W)
P_MIN_DBM = 0.0  # Potência mínima para UE ativo = 0 dBm (1 mW)
MU_STEP = 1.0  # Passo do UD-DPC = 1 dBm
GAMMA_TAR_INIT = 10.0  # Meta inicial = 10 dB
GAMMA_MAX_DB = 20.0  # Teto físico de SINR (saturação QAM) = 20 dB
GAMMA_MIN_DB = -10.0  # Limiar de Outage / Bloqueio = -10 dB


# Comprimento de onda (resultado do ganho SMB não depende do valor absoluto:
# R é proporcional a lambda, portanto o ganho normalizado é adimensional).
_LAMBDA: float = 1.0

# Rótulos legíveis e cores para cada modelo de antena nos gráficos
# Rótulos legíveis e cores para cada modelo nos gráficos
MODEL_LABELS: dict = {
    "omni": "Omni",
    "sector": "Setorizado",
    "smb": "SMB",
    "adaptive": "Adaptativo (Ideal 0°)",
    "adaptive_1": "Adaptativo (Erro 1°)",
    "adaptive_5": "Adaptativo (Erro 5°)",
    "adaptive_10": "Adaptativo (Erro 10°)",
    # --- (Controle de Potência) ---
    "full_power": "Full Power (1W)",
    "dpc_fixed": "DPC Fixo (Meta 10 dB)",
    "dual_loop": "Dual-Loop PC (Outage)",
}

MODEL_COLORS: dict = {
    "omni": "tab:blue",
    "sector": "tab:orange",
    "smb": "tab:green",
    "adaptive": "tab:red",
    "adaptive_1": "tab:purple",
    "adaptive_5": "tab:brown",
    "adaptive_10": "tab:pink",
    # ---
    "full_power": "tab:gray",
    "dpc_fixed": "tab:blue",
    "dual_loop": "tab:orange",
}

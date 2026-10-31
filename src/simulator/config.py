# ---------------------------------------------------------------------------
# config.py — Constantes físicas globais e paletas visuais
# ---------------------------------------------------------------------------

from pathlib import Path

# Pasta onde todas as figuras geradas são salvas (<raiz do projeto>/figures).
# Caminhos relativos passados em 'savepath' são resolvidos a partir dela.
FIGURES_DIR: Path = Path(__file__).resolve().parents[2] / 'figures'

# Comprimento de onda (resultado do ganho SMB não depende do valor absoluto:
# R é proporcional a lambda, portanto o ganho normalizado é adimensional).
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

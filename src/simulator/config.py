# ---------------------------------------------------------------------------
# config.py — Constantes físicas globais e paletas visuais
# ---------------------------------------------------------------------------

# Comprimento de onda (resultado do ganho SMB não depende do valor absoluto:
# R é proporcional a lambda, portanto o ganho normalizado é adimensional).
_LAMBDA: float = 1.0

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

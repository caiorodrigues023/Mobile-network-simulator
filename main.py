"""
main.py — Ponto de entrada do simulador de redes móveis.

Executa um experimento de Monte Carlo rápido (para validação) e, opcionalmente,
plota o mapa do cenário e as CDFs dos KPIs. Também pode rodar o estudo de
mobilidade (beam tracking) ativando RUN_MOBILITY.

Para uma simulação completa com comparação entre cargas, consulte o ARCHITECTURE.md.
"""

import matplotlib
matplotlib.use('Agg')  # backend sem janela — remove esta linha para exibir gráficos interativos

import numpy as np
from simulator import (
    simular_experimento,
    cenario_aleatorio,
    plotar_mapa,
    plot_all_cdfs,
    compare_kpis,
    estudo_mobilidade,
    tabela_kpis,
    plot_ganho_vs_tempo,
    plot_perda_desalinhamento,
    plot_perda_por_arranjo,
    plot_tradeoff_se,
)

# ---------------------------------------------------------------------------
# Parâmetros do experimento
# ---------------------------------------------------------------------------
M   = 4       # número de APs (deve ser quadrado perfeito: 4, 9, 16 …)
N   = 4       # canais ortogonais
SIM = 5       # iterações de Monte Carlo (aumente para resultados estatísticos)

ANTENNA_MODELS = ('omni', 'sector', 'smb')
LOAD_CONFIGS   = {'K=1': 1, 'K=4': 4}   # label → nº de UEs

RUN_MOBILITY = True    # True → executa o estudo de mobilidade (Cap. 2)

# ---------------------------------------------------------------------------
# Experimento Monte Carlo por cenário de carga
# ---------------------------------------------------------------------------
print("=== Simulador de Redes Móveis ===\n")

resultados_por_carga: dict = {model: {} for model in ANTENNA_MODELS}

for label, K in LOAD_CONFIGS.items():
    print(f"[{label}] Simulando {SIM} iterações com M={M} APs, N={N} canais …")
    res = simular_experimento(
        M=M, N=N, K=K, sim=SIM,
        allocation='ortogonal',
        antenna_models=ANTENNA_MODELS,
    )
    for model in ANTENNA_MODELS:
        resultados_por_carga[model][label] = res[model]

print("\nSimulação concluída.\n")

# ---------------------------------------------------------------------------
# Resumo dos resultados
# ---------------------------------------------------------------------------
for label in LOAD_CONFIGS:
    print(f"--- {label} ---")
    for model in ANTENNA_MODELS:
        r = resultados_por_carga[model][label]
        sinr_p10 = 10 * np.log10(np.percentile(r['sinr'], 10))
        cap_p10  = np.percentile(r['cap'], 10)
        sum_cap  = r['sum_cap']
        print(f"  {model:12s}  SINR 10th={sinr_p10:+.1f} dB  "
              f"Cap 10th={cap_p10:.2f} Mbps  SumCap={sum_cap:.2f} Mbps")
    print()

# ---------------------------------------------------------------------------
# Estudo de mobilidade: lag de beam tracking (velocidade × T × L × distância)
# ---------------------------------------------------------------------------
if RUN_MOBILITY:
    print("=== Estudo de Mobilidade ===\n")
    res_mob = estudo_mobilidade(seed=42)
    tabela_kpis(res_mob)
    plot_ganho_vs_tempo(res_mob, L=32, d_perp=25, janela_ms=600,
                        savepath='mobilidade/ganho_vs_tempo.png', show=False)
    plot_perda_desalinhamento(res_mob, L=32, d_perp=25, janela_ms=600,
                              savepath='mobilidade/perda_desalinhamento.png', show=False)
    plot_perda_por_arranjo(res_mob, savepath='mobilidade/perda_por_arranjo.png', show=False)
    plot_tradeoff_se(res_mob, savepath='mobilidade/tradeoff_se.png', show=False)

# ---------------------------------------------------------------------------
# Gráficos (descomente para exibir / salvar)
# ---------------------------------------------------------------------------
# plot_all_cdfs(resultados_por_carga, list(LOAD_CONFIGS.keys()), models=list(ANTENNA_MODELS))
# compare_kpis(resultados_por_carga, list(LOAD_CONFIGS.keys()), models=list(ANTENNA_MODELS))

# Mapa de um cenário representativo
# aps, ues, shadowing = cenario_aleatorio(M=M, K=4, seed=42)
# plotar_mapa(aps, ues, shadowing=shadowing, models=list(ANTENNA_MODELS), show=True)

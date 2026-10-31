"""
run_scichal4a.py — Ponto de Entrada da Simulação (SciChal4A)

"""

import time

import numpy as np

# Importa as funções modulares da sua arquitetura
from src.simulator.config import B_TOTAL
from src.simulator.runner import simular_scichal4a
from src.simulator.visualization import plot_all_cdfs, plot_scichal4a_kpis


def run_experiment():
    print("======================================================")
    print(" Iniciando Simulação ")
    print("======================================================")

    # Parâmetros do sistema
    M = 16
    N = 1
    SIM = 100  # 10 mil snapshots Monte Carlo!
    K_scenarios = [1, 4, 8]

    # Algoritmos mapeados no seu simular_scichal4a_vetorizado
    algorithms = ["full_power", "dpc_fixed", "dual_loop"]
    resultados_consolidados = {
        algo: {K: {} for K in K_scenarios} for algo in algorithms
    }

    B_c = B_TOTAL / N
    start_time = time.time()

    for K in K_scenarios:
        # 1. Aciona o "Motor de Fórmula 1" matricial no runner.py
        stats = simular_scichal4a(M=M, N=N, K=K, sim=SIM)

        # 2. Consolida as estatísticas exigidas para o KPI daquele cenário
        for algo in algorithms:
            resultados_consolidados[algo][K]["border_se"] = np.percentile(
                stats[algo]["se"], 10
            )
            resultados_consolidados[algo][K]["sum_se"] = (
                np.mean(stats[algo]["sum_cap"]) / B_c
            )

            mean_power = np.mean(stats[algo]["power"])
            # Calcula Eficiência Energética (prevenindo divisão por zero)
            if mean_power > 0:
                resultados_consolidados[algo][K]["energy_eff"] = (
                    np.mean(stats[algo]["sum_cap"]) / mean_power
                ) / 1e6
            else:
                resultados_consolidados[algo][K]["energy_eff"] = 0.0

            resultados_consolidados[algo][K]["outage_rate"] = (
                np.mean(stats[algo]["outage"]) * 100.0
            )

            # Repassando dados brutos para CDFs
            resultados_consolidados[algo][K]["sinr"] = stats[algo]["sinr_lin"]
            resultados_consolidados[algo][K]["cap"] = (
                np.array(stats[algo]["se"]) * B_c / 1e6
            )
            resultados_consolidados[algo][K]["power"] = stats[algo]["power_dist"]

    print(
        f"\nSimulação de {SIM * len(K_scenarios)} rodadas concluída em {time.time() - start_time:.2f} segundos!"
    )

    # 3. Dispara a renderização do Painel Executivo
    plot_scichal4a_kpis(
        resultados_consolidados,
        K_scenarios,
        algorithms,
        title="Joint MRC & Dual-Loop PC",
    )

    # 4. Dispara a renderização das CDFs
    labels_k = [f"K={k}" for k in K_scenarios]
    resultados_cdfs = {
        algo: {f"K={k}": resultados_consolidados[algo][k] for k in K_scenarios}
        for algo in algorithms
    }

    plot_all_cdfs(resultados_cdfs, labels_k, algorithms)


if __name__ == "__main__":
    run_experiment()

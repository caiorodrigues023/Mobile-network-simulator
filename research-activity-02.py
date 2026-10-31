# """
# run_mobility.py — Simulação Cinemática e Beam Tracking (Atividade 2)

# Avalia o impacto da mobilidade (carro a 20 m/s) e do atraso de
# sinalização (100 ms) no alinhamento de feixes de um arranjo ULA.
# Implementação orientada à linha do tempo (time-driven).
# """

# import matplotlib.pyplot as plt
# import numpy as np


# def calc_ula_gain(theta_real, theta_tgt, L):
#     """
#     Calcula o ganho diretivo do ULA (Linear) para toda a linha do tempo de uma vez.
#     d = lambda / 2 -> O termo (2*pi/lambda * d) simplifica para pi.
#     """
#     ell = np.arange(L)

#     # Broadcasting: ell tem shape (L, 1), a diferença de senos tem shape (1, N_snapshots)
#     d_sin = np.sin(theta_real) - np.sin(theta_tgt)
#     fase = np.pi * ell[:, None] * d_sin[None, :]

#     # Soma a fase de todos os L elementos e calcula o ganho normalizado
#     G_linear = np.abs(np.sum(np.exp(1j * fase), axis=0)) ** 2 / L
#     return G_linear


# def simulate_beam_tracking(L, v=20.0, D=50.0, T_update=0.1, sigma_deg=0.0):
#     """
#     Simula o carro cruzando a antena tangencialmente.
#     - v: velocidade do carro (m/s) -> padrão 20 m/s (72 km/h)
#     - D: distância perpendicular (m) -> padrão 50 m
#     - T_update: intervalo de atualização do feixe (s) -> padrão 100 ms
#     - sigma_deg: erro gaussiano de estimação do AoA (graus)
#     """
#     dt = 0.001  # Resolução de 1 milissegundo
#     X_start, X_end = -100.0, 100.0  # O carro anda de X=-100 até X=100

#     t_total = (X_end - X_start) / v
#     t = np.arange(0, t_total, dt)

#     # 1. Cinemática Tangencial
#     X = X_start + v * t
#     Y = np.full_like(X, D)

#     # 2. Ângulo Real de Chegada (Broadside Y-axis = 0 radianos)
#     theta_real = np.arctan2(X, Y)

#     # 3. Simulação do Atraso de Rastreamento (Janelas de Atualização)
#     theta_tgt = np.zeros_like(theta_real)
#     passos_janela = int(T_update / dt)

#     np.random.seed(42)  # Fixado para gerar ruídos comparáveis nos gráficos

#     for i in range(0, len(t), passos_janela):
#         fim = min(i + passos_janela, len(t))
#         # O AP mede o ângulo no instante 'i', comete um erro e trava a mira
#         erro_rad = np.deg2rad(np.random.normal(0, sigma_deg))
#         theta_tgt[i:fim] = theta_real[i] + erro_rad

#     # 4. Cálculo de Ganho ULA Instantâneo
#     ganho_linear = calc_ula_gain(theta_real, theta_tgt, L)
#     ganho_db = 10 * np.log10(ganho_linear + 1e-12)

#     return X, theta_real, theta_tgt, ganho_db


# def plot_mobility_impact():
#     print("======================================================")
#     print(" Iniciando Simulação de Mobilidade (Atividade 2)      ")
#     print("======================================================")

#     fig, axes = plt.subplots(2, 1, figsize=(12, 10))
#     fig.suptitle(
#         "Impacto da Mobilidade Tangencial no Rastreamento de Feixes (ULA)\n"
#         "Carro a 20 m/s (72 km/h) | Atualização a cada 100 ms",
#         fontsize=14,
#         fontweight="bold",
#     )

#     # --- CENÁRIO 1: Impacto do Tamanho do Arranjo (L) ---
#     ax1 = axes[0]
#     L_scenarios = [8, 16, 32]
#     cores_L = ["tab:green", "tab:orange", "tab:red"]

#     for L, cor in zip(L_scenarios, cores_L):
#         X, _, _, ganho_db = simulate_beam_tracking(L=L, sigma_deg=0.0)
#         ax1.plot(X, ganho_db, label=f"L = {L} antenas", color=cor, lw=1.5)

#     ax1.set_title("Cenário A: Variação de L (Sem Erro de Estimação)", fontweight="bold")
#     ax1.set_ylabel("Ganho do Arranjo [dB]")
#     ax1.set_xlabel("Posição Tangencial do Carro - Eixo X [m]")
#     ax1.set_xlim(-100, 100)
#     ax1.set_ylim(-5, 20)
#     ax1.grid(True, linestyle="--", alpha=0.6)
#     ax1.legend()
#     # Adicionando anotação na origem (onde a velocidade angular é máxima)
#     ax1.axvline(0, color="gray", linestyle=":", alpha=0.7)
#     ax1.text(
#         0,
#         -3,
#         "Cruzamento Broadside\n(Maior Velocidade Angular)",
#         ha="center",
#         va="bottom",
#         fontsize=9,
#     )

#     # --- CENÁRIO 2: Impacto do Erro de Estimação (Sigma) ---
#     ax2 = axes[1]
#     Sigma_scenarios = [0.0, 2.0, 5.0]
#     cores_S = ["tab:blue", "tab:purple", "tab:brown"]

#     for sigma, cor in zip(Sigma_scenarios, cores_S):
#         X, _, _, ganho_db = simulate_beam_tracking(L=32, sigma_deg=sigma)
#         ax2.plot(
#             X, ganho_db, label=rf"Erro $\sigma = {sigma}^\circ$", color=cor, lw=1.5
#         )

#     ax2.set_title(
#         r"Cenário B: Variação do Ruído de Estimação $\sigma$ (Fixo em L = 32)",
#         fontweight="bold",
#     )
#     ax2.set_ylabel("Ganho do Arranjo [dB]")
#     ax2.set_xlabel("Posição Tangencial do Carro - Eixo X [m]")
#     ax2.set_xlim(-100, 100)
#     ax2.set_ylim(-10, 20)
#     ax2.grid(True, linestyle="--", alpha=0.6)
#     ax2.legend()
#     ax2.axvline(0, color="gray", linestyle=":", alpha=0.7)

#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.show()


# if __name__ == "__main__":
#     plot_mobility_impact()


"""
run_mobility.py — Simulação Cinemática e Beam Tracking (Seção 2.2 - Atividade 2)

Avalia o sistema sob:
- 3 Perfis de Mobilidade: Pedestrian (3 km/h), Urban (50 km/h), Highway (120 km/h)
- 3 Taxas de Atualização (Delta T): 100 ms, 50 ms, 10 ms (passo dt = 1 ms)
- 3 Configurações ULA: L = 8, L = 16, L = 32 antenas
- Variação de Distância Tangencial (D): Perto do AP até a borda da célula
- Métricas: Array Gain vs. Time, Beam Misalignment Loss (dB) e Net Spectral Efficiency
"""

import matplotlib.pyplot as plt
import numpy as np

# Tentativa de importação modular do seu pacote simulator
try:
    from simulator.antennas import calc_ula_gain_vectorized as calc_ula_gain
except ImportError:

    def calc_ula_gain(
        theta_real: np.ndarray, theta_tgt: np.ndarray, L: int
    ) -> np.ndarray:
        """Calcula o ganho diretivo do ULA (d = lambda/2) de forma 100% vetorizada."""
        ell = np.arange(L)
        d_sin = np.sin(theta_real) - np.sin(theta_tgt)
        fase = np.pi * ell[:, None] * d_sin[None, :]
        return np.abs(np.sum(np.exp(1j * fase), axis=0)) ** 2 / L


# ==============================================================================
# 1. PARÂMETROS DA SEÇÃO 2.2 E FÍSICA DO CANAL
# ==============================================================================
DT = 0.001  # Passo de tempo exigido: dt = 1 ms (0.001 s)
T_SIM = 4.0  # Janela temporal de observação (4 segundos centralizados no broadside)
SEED = 42  # Semente fixa para garantir realizações idênticas (Methodology Remark)

# Perfis de velocidade (convertidos de km/h para m/s)
VELOCITIES_KMH = {
    "Pedestrian (3 km/h)": 3.0,
    "Urban (50 km/h)": 50.0,
    "Highway (120 km/h)": 120.0,
}

# Taxas de atualização de feixe (Delta T em segundos)
UPDATE_RATES_MS = {
    "Slow (100 ms)": 0.100,
    "Fair (50 ms)": 0.050,
    "High (10 ms)": 0.010,
}

# Configurações de antena ULA e distâncias tangenciais (Perto -> Borda)
L_CONFIGS = [8, 16, 32]
DISTANCES_M = [25.0, 75.0, 150.0]  # Perto (25m), Médio (75m), Borda (150m)

# Parâmetros físicos para cálculo de SNR e Eficiência Espectral Líquida
P_TX = 1.0  # 1 W
K_CONST = 1e-4  # Constante de propagação
PATHLOSS_EXP = 4.0  # Expoente de perda de percurso
B_C = 100e6  # Largura de banda (100 MHz)
K_0 = 1e-20  # Densidade espectral de ruído
P_N = B_C * K_0  # Potência de ruído térmico
T_OVERHEAD = 0.0015  # Overhead de treinamento/piloto por atualização de feixe (1.5 ms)


# ==============================================================================
# 2. MOTOR DE SIMULAÇÃO TIME-DRIVEN
# ==============================================================================
def simulate_scenario(
    L: int,
    v_kmh: float,
    delta_t: float,
    D: float = 50.0,
    sigma_deg: float = 0.0,
    t_sim: float = T_SIM,
) -> dict:
    """
    Executa a simulação temporal (dt = 1 ms) de um UE cruzando tangencialmente
    o broadside do AP na distância perpendicular D.
    """
    # Methodology remark: reseta a semente antes de cada bateria de snapshots
    np.random.seed(SEED)

    v_ms = v_kmh / 3.6
    t = np.arange(0, t_sim, DT)

    # Trajetória centralizada: em t = t_sim / 2, o UE cruza exatamente X = 0 (broadside)
    X = v_ms * (t - (t_sim / 2.0))
    Y = np.full_like(X, D)
    dist_2d = np.sqrt(X**2 + Y**2)

    # Ângulo real instantâneo em relação ao broadside (Eixo Y -> theta = 0 rad)
    theta_real = np.arctan2(X, Y)

    # Rastreamento discreto com atraso Delta T
    theta_tgt = np.zeros_like(theta_real)
    step_samples = max(1, int(round(delta_t / DT)))

    for i in range(0, len(t), step_samples):
        end_idx = min(i + step_samples, len(t))
        err_rad = np.deg2rad(np.random.normal(0, sigma_deg)) if sigma_deg > 0 else 0.0
        theta_tgt[i:end_idx] = theta_real[i] + err_rad

    # Ganhos Linear e em dB (Real vs. Ideal)
    gain_actual_lin = calc_ula_gain(theta_real, theta_tgt, L)
    gain_ideal_lin = np.full_like(gain_actual_lin, float(L))

    gain_actual_db = 10.0 * np.log10(gain_actual_lin + 1e-12)
    gain_ideal_db = 10.0 * np.log10(gain_ideal_lin)

    # Beam Misalignment Loss (dB): Diferença entre o ganho ideal e o rastreamento Delta T
    misalignment_loss_db = gain_ideal_db - gain_actual_db

    # Eficiência Espectral Bruta vs. Líquida (descontando overhead de atualização)
    path_gain = K_CONST / (np.maximum(dist_2d, 1.0) ** PATHLOSS_EXP)
    snr_actual = (P_TX * path_gain * gain_actual_lin) / P_N
    se_gross = np.log2(1.0 + snr_actual)

    overhead_fraction = min(1.0, T_OVERHEAD / delta_t)
    se_net = (1.0 - overhead_fraction) * se_gross

    return {
        "t": t,
        "X": X,
        "gain_actual_db": gain_actual_db,
        "gain_ideal_db": gain_ideal_db,
        "loss_db": misalignment_loss_db,
        "se_gross": se_gross,
        "se_net": se_net,
    }


# ==============================================================================
# 3. PLOTAGEM DOS PAINÉIS EXIGIDOS PELA SEÇÃO 2.2
# ==============================================================================
def run_all_evaluations():
    print("======================================================================")
    print(" Simulação Seção 2.2: Perfis de Mobilidade, ULA e RRM (dt = 1 ms)     ")
    print("======================================================================\n")

    # --------------------------------------------------------------------------
    # FIGURA 1: Array Gain vs. Time & Beam Misalignment Loss (Efeito de Delta T e L)
    # --------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    fig1.suptitle(
        "Seção 2.2: Array Gain vs. Time e Beam Misalignment Loss (D = 50 m)",
        fontsize=14,
        fontweight="bold",
    )

    # Coluna Esquerda: Fixando Highway (120 km/h) e L=32, variando Delta T
    ax_gain_dt, ax_loss_dt = axes1[0, 0], axes1[1, 0]
    colors_dt = ["tab:red", "tab:orange", "tab:green"]

    for (label_dt, dt_val), color in zip(UPDATE_RATES_MS.items(), colors_dt):
        res = simulate_scenario(L=32, v_kmh=120.0, delta_t=dt_val, D=50.0)
        ax_gain_dt.plot(
            res["t"], res["gain_actual_db"], label=label_dt, color=color, lw=1.3
        )
        ax_loss_dt.plot(res["t"], res["loss_db"], label=label_dt, color=color, lw=1.3)

    ax_gain_dt.axhline(
        10 * np.log10(32), color="black", ls="--", lw=1, label="Ideal (L=32)"
    )
    ax_gain_dt.set_title(
        "Array Gain vs. Time (Highway 120 km/h | L=32)", fontweight="bold"
    )
    ax_gain_dt.set_ylabel("Array Gain [dB]")
    ax_gain_dt.set_ylim(-10, 18)
    ax_gain_dt.grid(True, ls="--", alpha=0.6)
    ax_gain_dt.legend(loc="lower left")

    ax_loss_dt.set_title(
        "Beam Misalignment Loss vs. Time (Highway 120 km/h | L=32)", fontweight="bold"
    )
    ax_loss_dt.set_ylabel("Misalignment Loss [dB]")
    ax_loss_dt.set_xlabel("Tempo de Simulação [s] (Cruzamento Broadside em t = 2.0 s)")
    ax_loss_dt.set_ylim(0, 25)
    ax_loss_dt.grid(True, ls="--", alpha=0.6)
    ax_loss_dt.legend(loc="upper left")

    # Coluna Direita: Fixando Highway (120 km/h) e Delta T = 100 ms, variando L (8, 16, 32)
    ax_gain_L, ax_loss_L = axes1[0, 1], axes1[1, 1]
    colors_L = ["tab:blue", "tab:purple", "tab:brown"]

    for L_val, color in zip(L_CONFIGS, colors_L):
        res = simulate_scenario(L=L_val, v_kmh=120.0, delta_t=0.100, D=50.0)
        ax_gain_L.plot(
            res["t"],
            res["gain_actual_db"],
            label=f"L = {L_val} antenas",
            color=color,
            lw=1.3,
        )
        ax_loss_L.plot(
            res["t"], res["loss_db"], label=f"L = {L_val} antenas", color=color, lw=1.3
        )

    ax_gain_L.set_title(
        r"Array Gain vs. Time (Highway 120 km/h | $\Delta T = 100$ ms)",
        fontweight="bold",
    )
    ax_gain_L.set_ylabel("Array Gain [dB]")
    ax_gain_L.set_ylim(-10, 18)
    ax_gain_L.grid(True, ls="--", alpha=0.6)
    ax_gain_L.legend(loc="lower left")

    ax_loss_L.set_title(
        r"Beam Misalignment Loss vs. Time (Highway 120 km/h | $\Delta T = 100$ ms)",
        fontweight="bold",
    )
    ax_loss_L.set_ylabel("Misalignment Loss [dB]")
    ax_loss_L.set_xlabel("Tempo de Simulação [s] (Cruzamento Broadside em t = 2.0 s)")
    ax_loss_L.set_ylim(0, 25)
    ax_loss_L.grid(True, ls="--", alpha=0.6)
    ax_loss_L.legend(loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # --------------------------------------------------------------------------
    # FIGURA 2: Efeito da Velocidade (Pedestrian, Urban, Highway) e Distância (D)
    # --------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5.5))
    fig2.suptitle(
        r"Impacto dos Perfis de Velocidade e da Distância ao AP (L = 32, $\Delta T = 100$ ms)",
        fontsize=14,
        fontweight="bold",
    )

    # Painel Esquerdo: Comparação dos 3 Perfis de Mobilidade
    ax_vel = axes2[0]
    for (v_label, v_val), color in zip(
        VELOCITIES_KMH.items(), ["tab:green", "tab:blue", "tab:red"]
    ):
        res = simulate_scenario(L=32, v_kmh=v_val, delta_t=0.100, D=50.0)
        ax_vel.plot(res["t"], res["loss_db"], label=v_label, color=color, lw=1.5)

    ax_vel.set_title(
        "Beam Misalignment Loss por Perfil de Mobilidade (D = 50 m)", fontweight="bold"
    )
    ax_vel.set_xlabel("Tempo de Simulação [s]")
    ax_vel.set_ylabel("Misalignment Loss [dB]")
    ax_vel.grid(True, ls="--", alpha=0.6)
    ax_vel.legend()

    # Painel Direito: Comparação das Distâncias Tangenciais (Perto vs. Borda)
    ax_dist = axes2[1]
    for D_val, color in zip(DISTANCES_M, ["tab:red", "tab:orange", "tab:blue"]):
        res = simulate_scenario(L=32, v_kmh=50.0, delta_t=0.100, D=D_val)
        ax_dist.plot(
            res["t"],
            res["loss_db"],
            label=f"Distância D = {D_val:.0f} m",
            color=color,
            lw=1.5,
        )

    ax_dist.set_title(
        "Beam Misalignment Loss por Distância ao AP (Urban 50 km/h)", fontweight="bold"
    )
    ax_dist.set_xlabel("Tempo de Simulação [s]")
    ax_dist.set_ylabel("Misalignment Loss [dB]")
    ax_dist.grid(True, ls="--", alpha=0.6)
    ax_dist.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # --------------------------------------------------------------------------
    # FIGURA 3: Avaliação Cruzada e Trade-off de Net Spectral Efficiency
    # --------------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig3.suptitle(
        r"Trade-off: Taxa de Atualização ($\Delta T$) vs. Eficiência Espectral Líquida (D = 50 m)",
        fontsize=14,
        fontweight="bold",
    )

    x_pos = np.arange(len(UPDATE_RATES_MS))
    width = 0.25

    print(
        f"{'Perfil de Mobilidade':<22} | {'L':<4} | {'Delta T':<12} | {'Perda Média (dB)':<16} | {'SE Líquida (bps/Hz)':<20}"
    )
    print("-" * 85)

    for idx, (v_label, v_val) in enumerate(VELOCITIES_KMH.items()):
        ax = axes3[idx]
        for l_idx, L_val in enumerate(L_CONFIGS):
            net_se_means = []
            for dt_label, dt_val in UPDATE_RATES_MS.items():
                res = simulate_scenario(L=L_val, v_kmh=v_val, delta_t=dt_val, D=50.0)
                mean_loss = float(np.mean(res["loss_db"]))
                mean_net_se = float(np.mean(res["se_net"]))
                net_se_means.append(mean_net_se)

                print(
                    f"{v_label:<22} | {L_val:<4} | {dt_label:<12} | {mean_loss:<16.2f} | {mean_net_se:<20.2f}"
                )

            ax.bar(
                x_pos + (l_idx - 1) * width,
                net_se_means,
                width,
                label=f"L = {L_val}",
            )
        print("-" * 85)

        ax.set_title(v_label, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(list(UPDATE_RATES_MS.keys()))
        ax.set_xlabel(r"Intervalo de Atualização ($\Delta T$)")
        if idx == 0:
            ax.set_ylabel("Eficiência Espectral Líquida Média [bps/Hz]")
            ax.legend()
        ax.grid(True, axis="y", ls="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


if __name__ == "__main__":
    run_all_evaluations()

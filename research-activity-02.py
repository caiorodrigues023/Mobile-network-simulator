"""
run_mobility.py — Simulação Cinemática e Beam Tracking (Atividade 2)

Avalia o impacto da mobilidade (carro a 20 m/s) e do atraso de
sinalização (100 ms) no alinhamento de feixes de um arranjo ULA.
Implementação orientada à linha do tempo (time-driven).
"""

import matplotlib.pyplot as plt
import numpy as np


def calc_ula_gain(theta_real, theta_tgt, L):
    """
    Calcula o ganho diretivo do ULA (Linear) para toda a linha do tempo de uma vez.
    d = lambda / 2 -> O termo (2*pi/lambda * d) simplifica para pi.
    """
    ell = np.arange(L)

    # Broadcasting: ell tem shape (L, 1), a diferença de senos tem shape (1, N_snapshots)
    d_sin = np.sin(theta_real) - np.sin(theta_tgt)
    fase = np.pi * ell[:, None] * d_sin[None, :]

    # Soma a fase de todos os L elementos e calcula o ganho normalizado
    G_linear = np.abs(np.sum(np.exp(1j * fase), axis=0)) ** 2 / L
    return G_linear


def simulate_beam_tracking(L, v=20.0, D=50.0, T_update=0.1, sigma_deg=0.0):
    """
    Simula o carro cruzando a antena tangencialmente.
    - v: velocidade do carro (m/s) -> padrão 20 m/s (72 km/h)
    - D: distância perpendicular (m) -> padrão 50 m
    - T_update: intervalo de atualização do feixe (s) -> padrão 100 ms
    - sigma_deg: erro gaussiano de estimação do AoA (graus)
    """
    dt = 0.001  # Resolução de 1 milissegundo
    X_start, X_end = -100.0, 100.0  # O carro anda de X=-100 até X=100

    t_total = (X_end - X_start) / v
    t = np.arange(0, t_total, dt)

    # 1. Cinemática Tangencial
    X = X_start + v * t
    Y = np.full_like(X, D)

    # 2. Ângulo Real de Chegada (Broadside Y-axis = 0 radianos)
    theta_real = np.arctan2(X, Y)

    # 3. Simulação do Atraso de Rastreamento (Janelas de Atualização)
    theta_tgt = np.zeros_like(theta_real)
    passos_janela = int(T_update / dt)

    np.random.seed(42)  # Fixado para gerar ruídos comparáveis nos gráficos

    for i in range(0, len(t), passos_janela):
        fim = min(i + passos_janela, len(t))
        # O AP mede o ângulo no instante 'i', comete um erro e trava a mira
        erro_rad = np.deg2rad(np.random.normal(0, sigma_deg))
        theta_tgt[i:fim] = theta_real[i] + erro_rad

    # 4. Cálculo de Ganho ULA Instantâneo
    ganho_linear = calc_ula_gain(theta_real, theta_tgt, L)
    ganho_db = 10 * np.log10(ganho_linear + 1e-12)

    return X, theta_real, theta_tgt, ganho_db


def plot_mobility_impact():
    print("======================================================")
    print(" Iniciando Simulação de Mobilidade (Atividade 2)      ")
    print("======================================================")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(
        "Impacto da Mobilidade Tangencial no Rastreamento de Feixes (ULA)\n"
        "Carro a 20 m/s (72 km/h) | Atualização a cada 100 ms",
        fontsize=14,
        fontweight="bold",
    )

    # --- CENÁRIO 1: Impacto do Tamanho do Arranjo (L) ---
    ax1 = axes[0]
    L_scenarios = [8, 16, 32]
    cores_L = ["tab:green", "tab:orange", "tab:red"]

    for L, cor in zip(L_scenarios, cores_L):
        X, _, _, ganho_db = simulate_beam_tracking(L=L, sigma_deg=0.0)
        ax1.plot(X, ganho_db, label=f"L = {L} antenas", color=cor, lw=1.5)

    ax1.set_title("Cenário A: Variação de L (Sem Erro de Estimação)", fontweight="bold")
    ax1.set_ylabel("Ganho do Arranjo [dB]")
    ax1.set_xlabel("Posição Tangencial do Carro - Eixo X [m]")
    ax1.set_xlim(-100, 100)
    ax1.set_ylim(-5, 20)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()
    # Adicionando anotação na origem (onde a velocidade angular é máxima)
    ax1.axvline(0, color="gray", linestyle=":", alpha=0.7)
    ax1.text(
        0,
        -3,
        "Cruzamento Broadside\n(Maior Velocidade Angular)",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    # --- CENÁRIO 2: Impacto do Erro de Estimação (Sigma) ---
    ax2 = axes[1]
    Sigma_scenarios = [0.0, 2.0, 5.0]
    cores_S = ["tab:blue", "tab:purple", "tab:brown"]

    for sigma, cor in zip(Sigma_scenarios, cores_S):
        X, _, _, ganho_db = simulate_beam_tracking(L=32, sigma_deg=sigma)
        ax2.plot(
            X, ganho_db, label=rf"Erro $\sigma = {sigma}^\circ$", color=cor, lw=1.5
        )

    ax2.set_title(
        r"Cenário B: Variação do Ruído de Estimação $\sigma$ (Fixo em L = 32)",
        fontweight="bold",
    )
    ax2.set_ylabel("Ganho do Arranjo [dB]")
    ax2.set_xlabel("Posição Tangencial do Carro - Eixo X [m]")
    ax2.set_xlim(-100, 100)
    ax2.set_ylim(-10, 20)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    ax2.axvline(0, color="gray", linestyle=":", alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    plot_mobility_impact()

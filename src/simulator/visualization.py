# ---------------------------------------------------------------------------
# visualization.py — Gráficos Matplotlib (uso CLI) retornando `fig`
#                    sem forçar plt.show() por padrão
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Wedge

from .allocation import attach_AP_UE
from .antennas import antenna_gain, calc_ula_gain, mrc_gain, smb_gains
from .channel import adaptive_gains, angle_matrix, gain_matrix, gerar_shadowing
from .config import (
    GAMMA_MAX_DB,
    GAMMA_MIN_DB,
    MOBILITY_DISTANCES_M,
    MOBILITY_DT,
    MOBILITY_PROFILES_KMH,
    MOBILITY_T_SIM,
    MOBILITY_ULA_SIZES,
    MOBILITY_UPDATE_PERIODS_S,
    MODEL_COLORS,
    MODEL_LABELS,
    P_MAX_DBM,
    P_MIN_DBM,
)
from .metrics import SINR, SINR_adaptive

# ---------------------------------------------------------------------------
# Mapa 2D do cenário
# ---------------------------------------------------------------------------


def plotar_mapa(
    aps: list,
    ues: list,
    shadowing: Optional[np.ndarray] = None,
    models: Sequence[str] = ("omni", "sector", "smb", "adaptive"),
    r_ap: float = 110.0,
    L: int = 8,
    J: int = 8,
    show_sinr: bool = True,
    N: int = 1,
    savepath: Optional[str] = None,
    show: bool = False,
) -> Figure:
    """
    Desenha o mapa 2D da área de cobertura, um painel por modelo de antena,
    todos com OS MESMOS APs, UEs e shadowing (comparação justa).
    """
    if shadowing is None:
        shadowing = gerar_shadowing(ues, aps)

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(7.2 * n, 7.6), squeeze=False)
    axes_row = axes[0]
    cmap = plt.get_cmap("tab20")
    sector_colors = ["tab:blue", "tab:red", "tab:green"]
    theta_b = [0, 120, 240]
    th = np.linspace(0, 2 * np.pi, 721)
    curves = smb_gains(th, L=L, J=J) / L if "smb" in models else None
    phi_all = angle_matrix(ues, aps)

    for ax, model in zip(axes_row, models):
        for ue in ues:
            ue.ap, ue.beam, ue.channel = None, 0, 0
        for ap in aps:
            ap.ues = []

        G = antenna_gain(ues, aps, model=model, L=L, J=J)
        gains = gain_matrix(ues, aps, G, shadowing=shadowing)
        serving_ap, serving_beam = attach_AP_UE(gains)
        for i, ue in enumerate(ues):
            ap_idx = int(serving_ap[i])
            b_idx = int(serving_beam[i])
            ue.ap = aps[ap_idx]
            ue.beam = b_idx
            ue.gain = float(gains[i, ap_idx, b_idx])
            ue.angle = float(phi_all[i, ap_idx])
            ue.dist = float(np.hypot(ue.x - ue.ap.x, ue.y - ue.ap.y))
            ue.ap.ues.append(ue)

        if model.startswith("adaptive") or model == "mrc":
            err_deg = 0.0
            if "_" in model:
                try:
                    err_deg = float(model.split("_")[1])
                except ValueError:
                    pass
            sinr = np.asarray(
                SINR_adaptive(
                    ues, N, gains, adaptive_gains(ues, aps, L=L, aoa_err_deg=err_deg)
                ),
                dtype=float,
            ).flatten()
        else:
            sinr = np.asarray(SINR(ues, N, gains, G), dtype=float).flatten()

        sinr_db = 10.0 * np.log10(np.maximum(sinr, 1e-12))
        ativos = {(ue.ap.id, ue.beam) for ue in ues}

        for ap in aps:
            if model == "omni":
                ax.add_patch(
                    Circle(
                        (ap.x, ap.y),
                        r_ap * 0.7,
                        fc="tab:blue",
                        alpha=0.10,
                        ec="tab:blue",
                        lw=0.8,
                        zorder=1,
                    )
                )
            elif model == "sector":
                for b in range(3):
                    ativo = (ap.id, b) in ativos
                    ax.add_patch(
                        Wedge(
                            (ap.x, ap.y),
                            r_ap,
                            theta_b[b] - 60,
                            theta_b[b] + 60,
                            fc=sector_colors[b],
                            alpha=0.50 if ativo else 0.12,
                            ec=sector_colors[b],
                            lw=1.0,
                            zorder=1,
                        )
                    )
            elif model == "smb" and curves is not None:
                for k in range(J):
                    r = r_ap * curves[:, k]
                    x_k = ap.x + r * np.cos(th)
                    y_k = ap.y + r * np.sin(th)
                    if (ap.id, k) in ativos:
                        c = plt.get_cmap("hsv")(k / J)
                        ax.fill(x_k, y_k, color=c, alpha=0.45, zorder=2)
                        ax.plot(x_k, y_k, color=c, lw=1.6, zorder=2)
                    else:
                        ax.plot(
                            x_k, y_k, color="gray", lw=0.6, ls="--", alpha=0.7, zorder=1
                        )

        if model.startswith("adaptive") or model == "mrc":
            for ue in ues:
                r = r_ap * mrc_gain(ue.angle, th, L) / L
                x_u = ue.ap.x + r * np.cos(th)
                y_u = ue.ap.y + r * np.sin(th)
                c = cmap(ue.ap.id % 20)
                ax.fill(x_u, y_u, color=c, alpha=0.45, zorder=2)
                ax.plot(x_u, y_u, color=c, lw=1.6, zorder=2)

        for ue, sdb in zip(ues, sinr_db):
            c = cmap(ue.ap.id % 20)
            ax.plot([ue.x, ue.ap.x], [ue.y, ue.ap.y], color=c, lw=1.2, alpha=0.9, zorder=3)
            ax.scatter(ue.x, ue.y, s=55, color=c, edgecolor="k", linewidth=0.8, zorder=6)
            if show_sinr:
                ax.annotate(
                    f"{sdb:.1f} dB",
                    (ue.x, ue.y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=7,
                    zorder=7,
                )

        for ap in aps:
            ax.scatter(ap.x, ap.y, marker="^", s=70, color="k", zorder=8)
            ax.annotate(
                str(ap.id),
                (ap.x, ap.y),
                xytext=(4, -11),
                textcoords="offset points",
                fontsize=7,
                color="k",
                zorder=8,
            )

        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.set_aspect("equal")
        ax.set_facecolor("#f1f1f1")
        ax.grid(True, linestyle=":", color="gray", alpha=0.5)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(
            f"{MODEL_LABELS.get(model, model)}  |  SINR médio: {np.mean(sinr_db):.1f} dB"
        )

    handles = [
        Line2D([], [], marker="^", color="k", ls="", label="AP"),
        Line2D(
            [],
            [],
            marker="o",
            color="gray",
            mfc="gray",
            mec="k",
            ls="",
            label="UE (cor = AP servidor)",
        ),
    ]
    if "sector" in models:
        handles.append(Line2D([], [], color="tab:red", lw=6, alpha=0.5, label="Setor ativo"))
    if "smb" in models:
        handles.append(
            Line2D([], [], color="tab:green", lw=2, alpha=0.5, label="Feixe SMB ativo")
        )
    if any(m.startswith("adaptive") or m == "mrc" for m in models):
        handles.append(
            Line2D([], [], color="tab:purple", lw=4, alpha=0.5, label="Feixe MRC (Adaptativo)")
        )

    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    plt.tight_layout(rect=(0, 0.04, 1, 1))

    if savepath:
        fig.savefig(savepath, dpi=150)
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# CDFs de Potência, SINR e Capacidade
# ---------------------------------------------------------------------------


def plot_cdfs(cdf_sinr: list, cdf_capacity: list, show: bool = False) -> Figure:
    """Plota CDFs básicas de SINR e Capacidade (compatibilidade legada)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, data in enumerate(cdf_sinr):
        arr = np.asarray(data, dtype=float)
        db_data = np.sort(10.0 * np.log10(np.maximum(arr, 1e-12)))
        percentis = np.linspace(0, 1, len(db_data))
        axes[0].plot(
            db_data,
            percentis,
            label=f"SINR {'Per-AP' if i == 1 else 'Random'} channel allocation",
        )

    axes[0].axhline(y=0.10, color="r", linewidth=0.7, linestyle="--", label="10th Percentil")
    axes[0].axhline(y=0.50, color="b", linewidth=0.7, linestyle="--", label="50th Percentil")
    axes[0].set_title("CDF do SINR por UE")
    axes[0].set_xlabel("SINR (dB)")
    axes[0].set_ylabel("Percentil")
    axes[0].legend()
    axes[0].grid(True)

    for i, data in enumerate(cdf_capacity):
        data_sorted = np.sort(np.asarray(data, dtype=float))
        percentis = np.linspace(0, 1, len(data_sorted))
        axes[1].plot(
            data_sorted,
            percentis,
            label=f"Cap. do Canal {'Per-AP' if i == 1 else 'Random'} channel allocation",
        )

    axes[1].axhline(y=0.10, color="r", linewidth=0.7, linestyle="--", label="10th Percentil")
    axes[1].axhline(y=0.50, color="b", linewidth=0.7, linestyle="--", label="50th Percentil")
    axes[1].set_title("CDF da Capacidade do Canal por UE")
    axes[1].set_xlabel("Capacidade do Canal (Mbps)")
    axes[1].set_ylabel("Percentil")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_all_cdfs(
    resultados_por_carga: dict,
    load_labels: list,
    models: Optional[list] = None,
    show: bool = False,
) -> Figure:
    """CDFs de potência, SINR e capacidade para cada cenário de carga."""
    if models is None:
        models = list(resultados_por_carga.keys())

    n_loads = len(load_labels)
    fig, axes = plt.subplots(n_loads, 3, figsize=(18, 5 * n_loads))
    if n_loads == 1:
        axes = axes.reshape(1, 3)

    kpi_specs = [
        ("power", "Potência Recebida", "Potência Recebida (dBW)", True),
        ("sinr", "SINR", "SINR (dB)", True),
        ("cap", "Capacidade do Canal", "Capacidade do Canal (Mbps)", False),
    ]

    for row, load in enumerate(load_labels):
        for col, (key, title, xlabel, to_db) in enumerate(kpi_specs):
            ax = axes[row, col]
            for model in models:
                data = np.asarray(resultados_por_carga[model][load][key], dtype=float).flatten()
                if to_db:
                    data = 10.0 * np.log10(np.maximum(data, 1e-18))
                data = np.sort(data)
                percentis = np.linspace(0, 1, len(data))
                ax.plot(
                    data,
                    percentis,
                    label=MODEL_LABELS.get(model, model),
                    color=MODEL_COLORS.get(model),
                )

            ax.axhline(y=0.10, color="r", linewidth=0.7, linestyle="--", label="10th Percentil")
            ax.axhline(y=0.50, color="k", linewidth=0.7, linestyle="--", label="50th Percentil")
            ax.set_title(f"CDF {title} ({load})")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Percentil")
            ax.legend(fontsize=8)
            ax.grid(True)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Comparação de KPIs entre modelos (Remark 2)
# ---------------------------------------------------------------------------


def compare_kpis(
    resultados: dict,
    load_labels: list,
    models: Optional[list] = None,
    show: bool = False,
) -> Figure:
    """Compara SINR 10th pct (ganho dB vs omni) e capacidade normalizada."""
    if models is None:
        models = list(resultados.keys())

    labels = [MODEL_LABELS.get(m, m) + (" (baseline)" if m == "omni" else "") for m in models]
    colors = [MODEL_COLORS.get(m) for m in models]

    fig, axes = plt.subplots(2, len(load_labels), figsize=(6 * len(load_labels), 9))
    if len(load_labels) == 1:
        axes = axes.reshape(2, 1)

    base_model = "omni" if "omni" in resultados else models[0]
    for col, load in enumerate(load_labels):
        base_sinr_db = 10.0 * np.log10(
            max(
                float(np.percentile(np.asarray(resultados[base_model][load]["sinr"]).flatten(), 10)),
                1e-12,
            )
        )
        sinr_db = [
            10.0
            * np.log10(
                max(
                    float(np.percentile(np.asarray(resultados[m][load]["sinr"]).flatten(), 10)),
                    1e-12,
                )
            )
            - base_sinr_db
            for m in models
        ]
        ax_sinr = axes[0, col]
        ax_sinr.bar(labels, sinr_db, color=colors)
        ax_sinr.set_title(f"Ganho de SINR 10th pct ({load})")
        ax_sinr.set_ylabel("Ganho vs Omni (dB)")
        ax_sinr.axhline(y=0, color="k", linewidth=0.8, linestyle="--")
        ax_sinr.grid(True, axis="y")
        ax_sinr.tick_params(axis="x", rotation=15)

        base = resultados[base_model][load]
        cap_base = max(float(np.percentile(np.asarray(base["cap"]).flatten(), 10)), 1e-9)
        sumcap_base = max(float(base["sum_cap"]), 1e-9)

        metrics = ["Capacidade (10th pct)", "Soma-capacidade média"]
        x = np.arange(len(metrics))
        width = 0.8 / len(models)

        ax = axes[1, col]
        for idx, m in enumerate(models):
            res = resultados[m][load]
            vals = [
                float(np.percentile(np.asarray(res["cap"]).flatten(), 10)) / cap_base,
                float(res["sum_cap"]) / sumcap_base,
            ]
            offset = (idx - (len(models) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=labels[idx], color=colors[idx])

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=15)
        ax.set_title(f"Capacidade normalizada ({load})")
        ax.axhline(y=1.0, color="k", linewidth=0.8, linestyle="--")
        ax.legend()
        ax.grid(True, axis="y")

    plt.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Padrões polares SMB
# ---------------------------------------------------------------------------


def plot_padroes_smb(
    configs: Sequence[tuple[int, int]] = ((8, 4), (8, 16), (4, 8), (16, 8)),
    savepath: Optional[str] = None,
    show: bool = False,
) -> Figure:
    """Padrões de antena SMB em coordenadas polares para combinações (L, J)."""
    n = len(configs)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        subplot_kw={"projection": "polar"},
        figsize=(5 * cols, 5 * rows),
        squeeze=False,
    )
    th = np.linspace(0, 2 * np.pi, 1000)

    for ax, (L_, J_) in zip(axes.ravel(), configs):
        g = smb_gains(th, L=L_, J=J_)
        for k in range(J_):
            ax.plot(th, g[:, k], color=plt.get_cmap("hsv")(k / J_), lw=1.6)
        ax.set_ylim(0, L_)
        ax.set_yticks([L_ / 2, L_])
        ax.set_yticklabels([f"{L_ / 2:g}", f"{L_}"], fontsize=7)
        ax.set_title(f"L={L_}, J={J_}  (G = {L_})", fontsize=10)

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Painel 2x2 SciChal4A
# ---------------------------------------------------------------------------


def plot_scichal4a_kpis(
    resultados_consolidados: dict,
    K_scenarios: list,
    models: list,
    title: str = "Joint MRC & Dual-Loop PC",
    show: bool = False,
) -> Figure:
    """Gera o painel 2x2 com Border SE, Sum-Capacity SE, Eficiência Energética e Outage."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Comparativo de KPIs — {title}", fontsize=14, fontweight="bold")

    kpi_configs = [
        ("border_se", "1. Border Spectrum Efficiency (10º Pct)", "SE [bps/Hz]", axes[0, 0], False),
        ("sum_se", "2. Sum-Capacity SE (Média)", "SE Soma [bps/Hz]", axes[0, 1], False),
        ("energy_eff", "3. Energy Efficiency (Log)", "Eficiência [Mbits/Joule]", axes[1, 0], True),
        ("outage_rate", "4. Taxa de Outage", "Outage [%]", axes[1, 1], False),
    ]

    x = np.arange(len(K_scenarios))
    width = 0.8 / max(len(models), 1)

    for kpi_key, subplot_title, ylabel, ax, use_log in kpi_configs:
        for idx, model in enumerate(models):
            vals = [resultados_consolidados[model][K][kpi_key] for K in K_scenarios]
            offset = (idx - (len(models) - 1) / 2) * width
            rects = ax.bar(
                x + offset,
                vals,
                width,
                label=MODEL_LABELS.get(model, model),
                color=MODEL_COLORS.get(model, "tab:gray"),
                alpha=0.85,
            )
            for rect in rects:
                h = rect.get_height()
                if h > 0:
                    ax.annotate(
                        f"{h:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )

        ax.set_title(subplot_title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={K} UEs" for K in K_scenarios])
        ax.set_ylabel(ylabel)
        if use_log:
            ax.set_yscale("log")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Atividade 2 — Cenário de mobilidade e beam tracking
# ---------------------------------------------------------------------------


def _desenhar_lobo_ula(
    ax, psi_tgt: float, L: int, raio: float, color, alpha: float = 0.25, label: Optional[str] = None
) -> None:
    psi = np.linspace(-np.pi / 2, np.pi / 2, 2000)
    r = raio * calc_ula_gain(psi, np.full_like(psi, psi_tgt), L) / L
    x, y = r * np.sin(psi), r * np.cos(psi)
    ax.fill(x, y, color=color, alpha=alpha, label=label)
    ax.plot(x, y, color=color, lw=1.2)


def plot_cenario_mobilidade(
    distancias: tuple = MOBILITY_DISTANCES_M,
    perfis_kmh: Optional[dict] = None,
    t_sim: float = MOBILITY_T_SIM,
    dt: float = MOBILITY_DT,
    periodos_update: tuple = MOBILITY_UPDATE_PERIODS_S,
    L_list: tuple = MOBILITY_ULA_SIZES,
    L_beam: int = 16,
    t_beam: float = 1.0,
    D_zoom: float = 50.0,
    v_zoom_kmh: float = 120.0,
    T_zoom: float = 0.100,
    r_cell: float = 150.0,
    savepath: Optional[str] = None,
    show: bool = False,
) -> Figure:
    """Ilustra o cenário da Atividade 2 (impacto da mobilidade no beamforming ULA)."""
    if perfis_kmh is None:
        perfis_kmh = MOBILITY_PROFILES_KMH

    d_palette = ["tab:red", "tab:green", "tab:orange", "tab:blue", "tab:purple", "tab:brown"]
    d_color = {D: d_palette[i % len(d_palette)] for i, D in enumerate(distancias)}
    v_palette = ["tab:green", "tab:blue", "tab:red", "tab:purple", "tab:brown"]
    v_color = {n: v_palette[i % len(v_palette)] for i, n in enumerate(perfis_kmh)}
    v_max = max(perfis_kmh.values()) / 3.6
    x_max = v_max * t_sim / 2

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(17, 8.4), gridspec_kw={"width_ratios": [1.15, 1]}
    )

    axA.axhspan(0, r_cell, color="#f1f1f1", zorder=0)
    axA.axhline(r_cell, color="k", ls="--", lw=1)
    axA.text(-x_max - 45, r_cell + 4, f"borda da célula (D = {r_cell:g} m)", fontsize=9)
    y_top = 1.33 * max(distancias)
    axA.annotate(
        "", xy=(0, y_top * 0.93), xytext=(0, 15), arrowprops=dict(arrowstyle="->", color="gray", ls="--")
    )
    axA.text(3, y_top * 0.94, "broadside (θ = 0°, +Y)", color="gray", fontsize=9)

    xs = (np.arange(8) - 3.5) * 2.0
    axA.scatter(xs, np.zeros(8), marker="s", s=26, color="k", zorder=9)
    axA.text(0, -14, "AP na origem – ULA (eixo ∥ X, d = λ/2)", ha="center", fontsize=10, weight="bold")

    for D in distancias:
        c = d_color[D]
        for nome, kmh in sorted(perfis_kmh.items(), key=lambda kv: -kv[1]):
            xe = kmh / 3.6 * t_sim / 2
            axA.plot(
                [-xe, xe],
                [D, D],
                color=v_color[nome],
                zorder=3,
                lw=1.5 + 3.0 * (1 - kmh / (v_max * 3.6)) ** 0.5 * 1.5,
                alpha=0.9,
                solid_capstyle="butt",
            )
        axA.annotate(
            "", xy=(x_max + 2, D), xytext=(x_max - 14, D), arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=1.5)
        )
        axA.scatter([-x_max], [D], s=60, color="white", edgecolor="k", zorder=5)
        axA.scatter([0], [D], s=70, color=c, edgecolor="k", zorder=6)
        axA.text(x_max + 6, D - 3, f"D = {D:g} m", color=c, fontsize=10, weight="bold")
        axA.plot([0, 0], [0, D], color=c, lw=0.8, ls=":")

    x_b = v_max * (t_beam - t_sim / 2)
    for D in distancias:
        psi_b = float(np.arctan2(x_b, D))
        _desenhar_lobo_ula(axA, psi_b, L_beam, 1.12 * float(np.hypot(x_b, D)), d_color[D])
        axA.plot([0, x_b], [0, D], color=d_color[D], lw=0.9, ls="--")
        axA.scatter([x_b], [D], s=60, color=d_color[D], edgecolor="k", zorder=8)
    axA.text(x_b - 4, 12, f"feixe MRC (L = {L_beam})\napontado p/ o UE em t = {t_beam:g} s", ha="right", fontsize=8.5)

    t_upd = np.arange(0, t_sim + 1e-9, T_zoom)
    axA.scatter(v_max * (t_upd - t_sim / 2), np.full_like(t_upd, D_zoom), marker="|", s=90, color="k", zorder=7)
    axA.set_xlim(-x_max - 75, x_max + 75)
    axA.set_ylim(-25, y_top)
    axA.set_aspect("equal")
    axA.grid(ls=":", alpha=0.5)
    axA.set_xlabel("x (m)")
    axA.set_ylabel("y (m)")
    axA.set_title(
        f"A) Cenário: UE cruza o broadside em y = D, x(t) = v·(t − Tsim/2), Tsim = {t_sim:g} s",
        fontsize=11.5,
    )

    v = v_zoom_kmh / 3.6
    x_old, x_now = -v * T_zoom, 0.0
    th_old, th_now = float(np.arctan2(x_old, D_zoom)), float(np.arctan2(x_now, D_zoom))
    t_old = t_sim / 2 - T_zoom
    l_palette = ["tab:cyan", "tab:orange", "tab:purple", "tab:green", "tab:brown"]
    for i, L in enumerate(L_list):
        _desenhar_lobo_ula(
            axB, th_old, L, 80, l_palette[i % len(l_palette)], alpha=0.22, label=f"Feixe L = {L}"
        )
    axB.plot([-20, 20], [D_zoom, D_zoom], color="tab:red", lw=2)
    axB.plot([0, x_old], [0, D_zoom], "k--", lw=1)
    axB.plot([0, x_now], [0, D_zoom], color="tab:red", ls="--", lw=1)
    axB.scatter(x_old, D_zoom, s=110, color="white", edgecolor="k", lw=2, zorder=6)
    axB.scatter(x_now, D_zoom, s=110, color="tab:red", edgecolor="k", zorder=6)
    axB.add_patch(Arc((0, 0), 40, 40, theta1=90, theta2=90 + np.rad2deg(-th_old), color="k", lw=2))
    axB.scatter(0, 0, marker="s", s=60, color="k", zorder=9)

    perdas = {
        L: 10.0 * np.log10(float(calc_ula_gain(np.array([th_now]), np.array([th_old]), L)[0]) / L)
        for L in L_list
    }
    axB.text(
        -19,
        1.25 * D_zoom * 2 - 22,
        f"Perda de desalinhamento (t → {t_sim/2:g} s⁻):\n"
        + "\n".join(f"  L = {L:2d}: {perdas[L]:6.1f} dB" for L in L_list),
        fontsize=9.5,
        family="monospace",
        va="top",
        bbox=dict(fc="white", ec="gray", boxstyle="round"),
    )
    axB.set_xlim(-20, 20)
    axB.set_ylim(-8, 2.1 * D_zoom)
    axB.set_aspect("equal")
    axB.grid(ls=":", alpha=0.5)
    axB.legend(loc="lower right", fontsize=8)
    axB.set_xlabel("x (m)")
    axB.set_ylabel("y (m)")
    axB.set_title(
        f"B) Zoom no broadside: {v_zoom_kmh:g} km/h, D = {D_zoom:g} m, ΔT = {T_zoom*1e3:g} ms",
        fontsize=11.5,
    )

    _ = (dt, periodos_update, t_old)
    fig.suptitle("Atividade 2 – Impacto da mobilidade no beamforming (MRC com ULA)", fontsize=13.5, weight="bold")
    plt.tight_layout(rect=(0, 0.04, 1, 0.96))

    if savepath:
        fig.savefig(savepath, dpi=160)
    if show:
        plt.show()
    return fig


def _cores_auto(n: int) -> list:
    if n <= 10:
        return [plt.get_cmap("tab10")(i) for i in range(n)]
    return list(plt.get_cmap("turbo")(np.linspace(0.0, 1.0, n)))


def plotar_convergencia_dual_loop(
    hist: dict,
    ue_labels: Optional[list] = None,
    p_max_dbm: Optional[float] = None,
    p_min_dbm: Optional[float] = None,
    savepath: Optional[str] = None,
    show: bool = False,
    titulo_extra: str = "",
) -> Figure:
    """Plota a convergência do Dual-Loop PC para uma realização (snapshot)."""
    p_max_dbm = P_MAX_DBM if p_max_dbm is None else p_max_dbm
    p_min_dbm = P_MIN_DBM if p_min_dbm is None else p_min_dbm

    p_w = hist["power_w"]
    T, K = p_w.shape
    if ue_labels is None:
        ue_labels = [f"UE{k + 1}" for k in range(K)]

    with np.errstate(divide="ignore"):
        p_dbm = np.where(p_w > 0, 10.0 * np.log10(p_w * 1000.0), np.nan)

    it_inner = np.arange(T)
    cores = _cores_auto(K)

    fig, (ax_p, ax_t) = plt.subplots(
        1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.6, 1]}
    )

    for k in range(K):
        ax_p.plot(it_inner, p_dbm[:, k], color=cores[k], lw=1.4, label=ue_labels[k])

    for m in hist["outer_marks"]:
        ax_p.axvline(m - 0.5, color="gray", ls=":", lw=0.6, alpha=0.7)

    ax_p.axhline(p_max_dbm, color="r", ls="--", lw=0.9, label=f"P_max ({p_max_dbm:g} dBm)")
    ax_p.axhline(p_min_dbm, color="k", ls="--", lw=0.9, label=f"P_min ({p_min_dbm:g} dBm)")
    ax_p.set_title(f"Convergência da Potência de Transmissão (Dual-Loop PC) {titulo_extra}", fontweight="bold")
    ax_p.set_xlabel("Iteração do loop interno")
    ax_p.set_ylabel("Potência (dBm)")
    ax_p.set_ylim(p_min_dbm - 2, p_max_dbm + 2)
    ax_p.grid(True, ls="--", alpha=0.6)
    ax_p.legend(fontsize=8, ncol=2)

    tgt = hist["target_db"]
    it_outer = np.arange(tgt.shape[0])
    for k in range(K):
        ax_t.step(it_outer, tgt[:, k], where="post", color=cores[k], lw=1.6, label=ue_labels[k])

    ax_t.axhline(GAMMA_MAX_DB, color="r", ls="--", lw=0.9, label=f"Teto ({GAMMA_MAX_DB:g} dB)")
    ax_t.axhline(GAMMA_MIN_DB, color="k", ls="--", lw=0.9, label=f"Outage ({GAMMA_MIN_DB:g} dB)")
    ax_t.set_title("Negociação do Alvo de SINR (Loop Externo)", fontweight="bold")
    ax_t.set_xlabel("Iteração do loop externo")
    ax_t.set_ylabel(r"$\gamma_{tar}$ (dB)")
    ax_t.grid(True, ls="--", alpha=0.6)
    ax_t.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    if savepath:
        pasta = os.path.dirname(savepath)
        if pasta:
            os.makedirs(pasta, exist_ok=True)
        fig.savefig(savepath, dpi=150)
    if show:
        plt.show()
    return fig

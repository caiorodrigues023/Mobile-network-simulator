# ---------------------------------------------------------------------------
# visualization.py — Mapas 2D, CDFs e comparações de KPI
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.lines import Line2D

from pathlib import Path

from .config import MODEL_LABELS, MODEL_COLORS, FIGURES_DIR
from .antennas import antenna_gain, smb_gains, mrc_gain
from .channel import gerar_shadowing, gain_matrix, adaptive_gains
from .allocation import attach_AP_UE
from .metrics import SINR, SINR_adaptive
from .mobility import PERIODOS_MS, ARRANJOS_L


def _salvar(fig, savepath) -> Path:
    """
    Salva a figura em 'savepath'. Caminhos relativos são resolvidos a partir de
    config.FIGURES_DIR (subpastas são criadas); caminhos absolutos são respeitados.
    """
    p = Path(savepath)
    if not p.is_absolute():
        p = FIGURES_DIR / p
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150)
    return p


# ---------------------------------------------------------------------------
# Mapa 2D do cenário
# ---------------------------------------------------------------------------

def plotar_mapa(aps: list, ues: list, shadowing: np.ndarray = None,
                models: list = ('omni', 'sector', 'smb', 'adaptive'),
                r_ap: float = 110.0, L: int = 8, J: int = 8,
                show_sinr: bool = True, N: int = 1,
                savepath: str = None, show: bool = True) -> None:
    """
    Desenha o mapa 2D da área de cobertura, um painel por modelo de antena,
    todos com OS MESMOS APs, UEs e shadowing (comparação justa).
    """
    if shadowing is None:
        shadowing = gerar_shadowing(ues, aps)

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(7.2 * n, 7.6), squeeze=False)
    axes = axes[0]
    cmap = plt.get_cmap('tab20')
    sector_colors = ['tab:blue', 'tab:red', 'tab:green']
    theta_b = [0, 120, 240]                                     # graus, 3 setores
    th = np.linspace(0, 2 * np.pi, 721)
    curves = smb_gains(th, L=L, J=J) / L if 'smb' in models else None

    for ax, model in zip(axes, models):
        # Reseta e refaz o attachment para este modelo
        for ue in ues:
            ue.ap, ue.beam, ue.channel = None, 0, 0
        for ap in aps:
            ap.ues = []

        G = antenna_gain(ues, aps, model=model, L=L, J=J)
        gains = gain_matrix(ues, aps, G, shadowing=shadowing)
        attach_AP_UE(ues, aps, gains)

        if model.startswith('adaptive'):
            err_deg = 0.0
            if '_' in model:
                try:
                    err_deg = float(model.split('_')[1])
                except ValueError:
                    pass
            sinr = np.asarray(
                SINR_adaptive(ues, N, gains,
                              adaptive_gains(ues, aps, L=L, aoa_err_deg=err_deg))
            ).flatten()
        else:
            sinr = np.asarray(SINR(ues, N, gains, G)).flatten()

        sinr_db = 10 * np.log10(sinr)
        ativos = {(ue.ap.id, ue.beam) for ue in ues}

        # --- padrões de antena ---
        for ap in aps:
            if model == 'omni':
                ax.add_patch(Circle((ap.x, ap.y), r_ap * 0.7,
                                    fc='tab:blue', alpha=0.10,
                                    ec='tab:blue', lw=0.8, zorder=1))
            elif model == 'sector':
                for b in range(3):
                    ativo = (ap.id, b) in ativos
                    ax.add_patch(Wedge(
                        (ap.x, ap.y), r_ap,
                        theta_b[b] - 60, theta_b[b] + 60,
                        fc=sector_colors[b],
                        alpha=0.50 if ativo else 0.12,
                        ec=sector_colors[b], lw=1.0, zorder=1,
                    ))
            elif model == 'smb':
                for k in range(J):
                    r = r_ap * curves[:, k]
                    x_k = ap.x + r * np.cos(th)
                    y_k = ap.y + r * np.sin(th)
                    if (ap.id, k) in ativos:
                        c = plt.cm.hsv(k / J)
                        ax.fill(x_k, y_k, color=c, alpha=0.45, zorder=2)
                        ax.plot(x_k, y_k, color=c, lw=1.6, zorder=2)
                    else:
                        ax.plot(x_k, y_k, color='gray', lw=0.6,
                                ls='--', alpha=0.7, zorder=1)

        # --- feixes MRC adaptativos ---
        if model.startswith('adaptive'):
            for ue in ues:
                r = r_ap * mrc_gain(ue.angle, th, L) / L
                x_u = ue.ap.x + r * np.cos(th)
                y_u = ue.ap.y + r * np.sin(th)
                c = cmap(ue.ap.id % 20)
                ax.fill(x_u, y_u, color=c, alpha=0.45, zorder=2)
                ax.plot(x_u, y_u, color=c, lw=1.6, zorder=2)

        # --- ligações e UEs ---
        for ue, sdb in zip(ues, sinr_db):
            c = cmap(ue.ap.id % 20)
            ax.plot([ue.x, ue.ap.x], [ue.y, ue.ap.y],
                    color=c, lw=1.2, alpha=0.9, zorder=3)
            ax.scatter(ue.x, ue.y, s=55, color=c,
                       edgecolor='k', linewidth=0.8, zorder=6)
            if show_sinr:
                ax.annotate(f'{sdb:.1f} dB', (ue.x, ue.y),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=7, zorder=7)

        # --- APs ---
        for ap in aps:
            ax.scatter(ap.x, ap.y, marker='^', s=70, color='k', zorder=8)
            ax.annotate(str(ap.id), (ap.x, ap.y),
                        xytext=(4, -11), textcoords='offset points',
                        fontsize=7, color='k', zorder=8)

        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.set_aspect('equal')
        ax.set_facecolor('#f1f1f1')
        ax.grid(True, linestyle=':', color='gray', alpha=0.5)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(
            f'{MODEL_LABELS.get(model, model)}  |  '
            f'SINR médio: {np.mean(sinr_db):.1f} dB'
        )

    # --- Legenda ---
    handles = [
        Line2D([], [], marker='^', color='k', ls='', label='AP'),
        Line2D([], [], marker='o', color='gray', mfc='gray', mec='k',
               ls='', label='UE (cor = AP servidor)'),
    ]
    last_model = models[-1]
    if 'sector' in models:
        handles.append(Line2D([], [], color='tab:red', lw=6,
                              alpha=0.5, label='Setor ativo'))
    if 'smb' in models:
        handles.append(Line2D([], [], color='tab:green', lw=2,
                              alpha=0.5, label='Feixe SMB ativo'))
    if last_model.startswith('adaptive'):
        handles.append(Line2D([], [], color='tab:purple', lw=4,
                              alpha=0.5, label='Feixe MRC (Adaptativo)'))

    fig.legend(handles=handles, loc='lower center',
               ncol=len(handles), frameon=False)
    plt.tight_layout(rect=(0, 0.04, 1, 1))

    if savepath:
        _salvar(fig, savepath)
    if show:
        plt.show()


# ---------------------------------------------------------------------------
# CDFs de Potência, SINR e Capacidade
# ---------------------------------------------------------------------------

def plot_cdfs(cdf_sinr: list, cdf_capacity: list) -> None:
    """
    Plota CDFs básicas de SINR e Capacidade para dois cenários de alocação.
    (Compatibilidade com chamadas legadas do scichal.py.)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, data in enumerate(cdf_sinr):
        db_data = sorted(10 * np.log10(a) for a in data)
        percentis = np.linspace(0, 1, len(db_data))
        axes[0].plot(db_data, percentis,
                     label=f'SINR {"Per-AP" if i == 1 else "Random"} channel allocation')

    axes[0].axhline(y=0.10, color='r', linewidth=0.7, linestyle='--', label='10th Percentil')
    axes[0].axhline(y=0.50, color='b', linewidth=0.7, linestyle='--', label='50th Percentil')
    axes[0].set_title('CDF do SINR por UE')
    axes[0].set_xlabel('SINR (dB)')
    axes[0].set_ylabel('Percentil')
    axes[0].legend()
    axes[0].grid(True)

    for i, data in enumerate(cdf_capacity):
        data_sorted = sorted(data)
        percentis = np.linspace(0, 1, len(data_sorted))
        axes[1].plot(data_sorted, percentis,
                     label=f'Cap. do Canal {"Per-AP" if i == 1 else "Random"} channel allocation')

    axes[1].axhline(y=0.10, color='r', linewidth=0.7, linestyle='--', label='10th Percentil')
    axes[1].axhline(y=0.50, color='b', linewidth=0.7, linestyle='--', label='50th Percentil')
    axes[1].set_title('CDF da Capacidade do Canal por UE')
    axes[1].set_xlabel('Capacidade do Canal (Mbps)')
    axes[1].set_ylabel('Percentil')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_all_cdfs(resultados_por_carga: dict, load_labels: list,
                  models: list = None) -> None:
    """
    CDFs de potência recebida, SINR e capacidade para cada cenário de carga,
    comparando os modelos de antena com marcação dos percentis 10 e 50.

    resultados_por_carga : {model: {label: {'power', 'sinr', 'cap', ...}, ...}, ...}
    load_labels          : lista de labels (ex: ['K=1', 'K=4', 'K=8']).
    """
    if models is None:
        models = list(resultados_por_carga.keys())

    n_loads = len(load_labels)
    fig, axes = plt.subplots(n_loads, 3, figsize=(18, 5 * n_loads))
    if n_loads == 1:
        axes = axes.reshape(1, 3)

    kpi_specs = [
        ('power', 'Potência Recebida',    'Potência Recebida (dBW)',      True),
        ('sinr',  'SINR',                 'SINR (dB)',                    True),
        ('cap',   'Capacidade do Canal',  'Capacidade do Canal (Mbps)',   False),
    ]

    for row, load in enumerate(load_labels):
        for col, (key, title, xlabel, to_db) in enumerate(kpi_specs):
            ax = axes[row, col]
            for model in models:
                data = np.asarray(
                    resultados_por_carga[model][load][key]
                ).flatten()
                if to_db:
                    data = 10 * np.log10(data)
                data = np.sort(data)
                percentis = np.linspace(0, 1, len(data))
                ax.plot(data, percentis,
                        label=MODEL_LABELS.get(model, model),
                        color=MODEL_COLORS.get(model))

            ax.axhline(y=0.10, color='r', linewidth=0.7, linestyle='--',
                       label='10th Percentil')
            ax.axhline(y=0.50, color='k', linewidth=0.7, linestyle='--',
                       label='50th Percentil')
            ax.set_title(f'CDF {title} ({load})')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Percentil')
            ax.legend(fontsize=8)
            ax.grid(True)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Comparação de KPIs entre modelos (Remark 2)
# ---------------------------------------------------------------------------

def compare_kpis(resultados: dict, load_labels: list,
                 models: list = None) -> None:
    """
    Compara SINR (10th pct, ganho em dB vs omni) e capacidade
    (10th pct e soma-capacidade, normalizados pelo omni) para cada cenário.
    """
    if models is None:
        models = list(resultados.keys())

    labels = [
        MODEL_LABELS.get(m, m) + (' (baseline)' if m == 'omni' else '')
        for m in models
    ]
    colors = [MODEL_COLORS.get(m) for m in models]

    fig, axes = plt.subplots(2, len(load_labels),
                              figsize=(6 * len(load_labels), 9))
    if len(load_labels) == 1:
        axes = axes.reshape(2, 1)

    for col, load in enumerate(load_labels):
        # SINR 10th percentil — diferença em dB em relação ao omni
        base_sinr_db = 10 * np.log10(
            np.percentile(np.asarray(resultados['omni'][load]['sinr']).flatten(), 10)
        )
        sinr_db = [
            10 * np.log10(
                np.percentile(np.asarray(resultados[m][load]['sinr']).flatten(), 10)
            ) - base_sinr_db
            for m in models
        ]
        ax_sinr = axes[0, col]
        ax_sinr.bar(labels, sinr_db, color=colors)
        ax_sinr.set_title(f'Ganho de SINR 10th pct ({load})')
        ax_sinr.set_ylabel('Ganho vs Omni (dB)')
        ax_sinr.axhline(y=0, color='k', linewidth=0.8, linestyle='--')
        ax_sinr.grid(True, axis='y')
        ax_sinr.tick_params(axis='x', rotation=15)

        # Capacidade normalizada
        base = resultados['omni'][load]
        cap_base = np.percentile(np.asarray(base['cap']).flatten(), 10)
        sumcap_base = base['sum_cap']

        metrics = ['Capacidade (10th pct)', 'Soma-capacidade média']
        x = np.arange(len(metrics))
        width = 0.8 / len(models)

        ax = axes[1, col]
        for idx, m in enumerate(models):
            res = resultados[m][load]
            vals = [
                np.percentile(np.asarray(res['cap']).flatten(), 10) / cap_base,
                res['sum_cap'] / sumcap_base,
            ]
            offset = (idx - (len(models) - 1) / 2) * width
            ax.bar(x + offset, vals, width,
                   label=labels[idx], color=colors[idx])

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=15)
        ax.set_title(f'Capacidade normalizada ({load})')
        ax.axhline(y=1.0, color='k', linewidth=0.8, linestyle='--')
        ax.legend()
        ax.grid(True, axis='y')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Padrões polares SMB
# ---------------------------------------------------------------------------

def plot_padroes_smb(configs: list = ((8, 4), (8, 16), (4, 8), (16, 8)),
                     savepath: str = None, show: bool = True):
    """
    Padrões de antena SMB em coordenadas polares para várias combinações (L, J),
    como na Fig. 9 do roteiro. O círculo externo corresponde ao ganho de pico G = L.
    """
    n = len(configs)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, subplot_kw={'projection': 'polar'},
                              figsize=(5 * cols, 5 * rows), squeeze=False)
    th = np.linspace(0, 2 * np.pi, 1000)

    for ax, (L_, J_) in zip(axes.ravel(), configs):
        g = smb_gains(th, L=L_, J=J_)
        for k in range(J_):
            ax.plot(th, g[:, k], color=plt.cm.hsv(k / J_), lw=1.6)
        ax.set_ylim(0, L_)
        ax.set_yticks([L_ / 2, L_])
        ax.set_yticklabels([f'{L_/2:g}', f'{L_}'], fontsize=7)
        ax.set_title(f'L={L_}, J={J_}  (G = {L_})', fontsize=10)

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    if savepath:
        _salvar(fig, savepath)
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Mobilidade (beam tracking) — consome os resultados de mobility.estudo_mobilidade()
# ---------------------------------------------------------------------------

def _finish(fig, savepath: str = None, show: bool = True):
    plt.tight_layout()
    if savepath:
        _salvar(fig, savepath)
    if show:
        plt.show()
    return fig


def plot_ganho_vs_tempo(res: dict, L: int = 16, d_perp: float = 100.0,
                        periodos: tuple = PERIODOS_MS, janela_ms: float = None,
                        savepath: str = None, show: bool = True):
    """
    Array Gain (dB) vs tempo: uma linha de gráficos por velocidade, uma curva
    por T. A linha tracejada é o ganho ideal (10·log10 L).
    Use 'janela_ms' (ex.: 400) para dar zoom ao redor do broadside.
    """
    nomes = list(dict.fromkeys(k[0] for k in res))
    fig, axes = plt.subplots(len(nomes), 1, figsize=(11, 3.2 * len(nomes)),
                             sharex=True, squeeze=False)
    for ax, nome in zip(axes[:, 0], nomes):
        for T in periodos:
            r = res[(nome, T, L, d_perp)]
            ax.plot(r['t_ms'], r['gain_db'], lw=1.2, label=f'T = {T} ms')
        ax.axhline(10 * np.log10(L), color='k', ls='--', lw=0.8,
                   label=f'Ideal ({10 * np.log10(L):.1f} dB)')
        ax.set_ylim(10 * np.log10(L) - 25, 10 * np.log10(L) + 1)
        ax.set_ylabel('Ganho de arranjo (dB)')
        ax.set_title(f'{nome} — L={L}, d⊥={d_perp:.0f} m')
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8, ncol=4, loc='lower center')
        if janela_ms:
            c = r['t_ms'][-1] / 2
            ax.set_xlim(c - janela_ms / 2, c + janela_ms / 2)
    axes[-1, 0].set_xlabel('Tempo (ms)')
    return _finish(fig, savepath, show)


def plot_perda_desalinhamento(res: dict, L: int = 16, d_perp: float = 100.0,
                              periodos: tuple = PERIODOS_MS,
                              janela_ms: float = None,
                              savepath: str = None, show: bool = True):
    """Beam Misalignment Loss (dB) = ganho ideal − ganho real, vs tempo."""
    nomes = list(dict.fromkeys(k[0] for k in res))
    fig, axes = plt.subplots(len(nomes), 1, figsize=(11, 3.2 * len(nomes)),
                             sharex=True, squeeze=False)
    for ax, nome in zip(axes[:, 0], nomes):
        for T in periodos:
            r = res[(nome, T, L, d_perp)]
            ax.plot(r['t_ms'], r['loss_db'], lw=1.2,
                    label=f"T = {T} ms (média {r['loss_medio_db']:.1f} dB)")
        ax.set_ylabel('Perda de desalinhamento (dB)')
        ax.set_title(f'{nome} — L={L}, d⊥={d_perp:.0f} m')
        ax.set_ylim(bottom=0, top=25)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8, ncol=3, loc='upper center')
        if janela_ms:
            c = r['t_ms'][-1] / 2
            ax.set_xlim(c - janela_ms / 2, c + janela_ms / 2)
    axes[-1, 0].set_xlabel('Tempo (ms)')
    return _finish(fig, savepath, show)


def plot_perda_por_arranjo(res: dict, d_perp: float = 100.0,
                           periodos: tuple = PERIODOS_MS,
                           arranjos: tuple = ARRANJOS_L,
                           savepath: str = None, show: bool = True):
    """Perda média de desalinhamento (dB) vs T, uma curva por L, por mobilidade."""
    nomes = list(dict.fromkeys(k[0] for k in res))
    fig, axes = plt.subplots(1, len(nomes), figsize=(5 * len(nomes), 4),
                             squeeze=False)
    for ax, nome in zip(axes[0], nomes):
        for L in arranjos:
            ax.plot(periodos,
                    [res[(nome, T, L, d_perp)]['loss_medio_db']
                     for T in periodos], 'o-', label=f'L = {L}')
        ax.set_xlabel('Período de atualização T (ms)')
        ax.set_ylabel('Perda média (dB)')
        ax.set_title(f'{nome}, d⊥={d_perp:.0f} m')
        ax.grid(True, alpha=0.4)
        ax.legend()
    return _finish(fig, savepath, show)


def plot_tradeoff_se(res: dict, d_perp: float = 100.0,
                     periodos: tuple = PERIODOS_MS, arranjos: tuple = ARRANJOS_L,
                     savepath: str = None, show: bool = True):
    """
    Trade-off taxa de atualização × eficiência espectral.
    Linhas cheias: SE líquida (descontado o overhead t_pilot/T).
    Tracejadas: SE bruta (sem overhead).
    """
    nomes = list(dict.fromkeys(k[0] for k in res))
    fig, axes = plt.subplots(1, len(nomes), figsize=(5 * len(nomes), 4.2),
                             squeeze=False)
    for ax, nome in zip(axes[0], nomes):
        for i, L in enumerate(arranjos):
            c = f'C{i}'
            ax.plot(periodos, [res[(nome, T, L, d_perp)]['se_net_medio']
                               for T in periodos], 'o-', color=c,
                    label=f'L={L} líquida')
            ax.plot(periodos, [res[(nome, T, L, d_perp)]['se_gross_medio']
                               for T in periodos], 's--', color=c, alpha=0.5,
                    label=f'L={L} bruta')
        ax.set_xlabel('Período de atualização T (ms)')
        ax.set_ylabel('Eficiência espectral C/B (bit/s/Hz)')
        ax.set_title(f'{nome}, d⊥={d_perp:.0f} m')
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=7)
    return _finish(fig, savepath, show)

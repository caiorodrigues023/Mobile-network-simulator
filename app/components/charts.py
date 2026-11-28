# ---------------------------------------------------------------------------
# app/components/charts.py — Fábrica de Gráficos Interativos em Plotly
# ---------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulator.antennas import calc_ula_gain, compute_beam_lobe_xy
from simulator.config import AP_PALETTE_HEX, MODEL_LABELS, PLOTLY_COLORS
from simulator.schemas import (
    HandoverResult,
    MobilityResult,
    NetworkHandoverResult,
    NetworkMobilityResult,
    ScenarioMetrics,
    SDMAResult,
    SnapshotResult,
    UESnapshot,
)

_UE_TRACE_PALETTE = [
    "#38bdf8",
    "#f97316",
    "#10b981",
    "#f43f5e",
    "#a855f7",
    "#eab308",
    "#06b6d4",
    "#ec4899",
    "#84cc16",
    "#6366f1",
]


def _base_layout(fig: go.Figure, height: int = 500, title: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left") if title else None,
        height=height,
        margin=dict(l=48, r=30, t=54 if title else 36, b=44),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.06)",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.22)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.22)")
    return fig


def _ue_hover_row(u: UESnapshot) -> list[Any]:
    p_dbm_str = "Desligado (0 W)" if u.is_blocked else f"{u.tx_power_dbm:.1f} dBm ({u.tx_power_w:.4f} W)"
    sinr_str = "Desligado (Outage)" if u.is_blocked else f"{u.sinr_db:+.2f} dB"
    ch_str = "CA (Todos)" if u.is_aggregated else f"Canal {u.channel}"
    tar_line = (
        f"<br>Meta γ_tar: <b>{u.gamma_tar_db:+.1f} dB</b>" if u.gamma_tar_db is not None else ""
    )
    return [
        u.ue_id,
        u.x,
        u.y,
        u.serving_ap,
        u.distance_m,
        u.aoa_deg,
        ch_str,
        u.serving_beam,
        p_dbm_str,
        u.rx_gain_db,
        sinr_str,
        u.se_bps_hz,
        u.capacity_mbps,
        u.status,
        tar_line,
    ]


_UE_HOVERTEMPLATE = (
    "<b>📱 UE %{customdata[0]}</b> — (%{customdata[1]:.1f}, %{customdata[2]:.1f}) m<br>"
    "AP Conectado: <b>AP %{customdata[3]}</b> | Distância: <b>%{customdata[4]:.1f} m</b><br>"
    "Ângulo AoA: %{customdata[5]:.1f}° | %{customdata[6]} | Feixe/Setor: %{customdata[7]}<br>"
    "Potência Tx: %{customdata[8]}<br>"
    "Ganho Efetivo: %{customdata[9]:.1f} dB | SINR: <b>%{customdata[10]}</b><br>"
    "SE: %{customdata[11]:.2f} bps/Hz | Capacidade: <b>%{customdata[12]:.2f} Mbps</b><br>"
    "Status: <b>%{customdata[13]}</b>%{customdata[14]}<extra></extra>"
)

_AP_HOVERTEMPLATE = (
    "<b>📡 AP %{customdata[0]}</b> — (%{customdata[1]:.1f}, %{customdata[2]:.1f}) m<br>"
    "UEs Conectados: <b>%{customdata[3]}</b><br>"
    "Lista de UEs: <b>%{customdata[4]}</b><br>"
    "Sum-Capacity da Célula: <b>%{customdata[5]:.2f} Mbps</b><br>"
    "UEs em Outage neste AP: <b>%{customdata[6]}</b><extra></extra>"
)


# ---------------------------------------------------------------------------
# 1. Mapa 2D Interativo do Snapshot (Tela 1)
# ---------------------------------------------------------------------------


def build_snapshot_map_2d(
    snap: SnapshotResult,
    show_beams: bool = True,
    show_links: bool = True,
    show_heatmap: bool = False,
    show_interference: bool = False,
    victim_ue_id: Optional[int] = None,
) -> go.Figure:
    """
    Mapa 2D interativo em Plotly com grade de células, lóbulos de antena reais,
    linhas UE↔AP (`hoverinfo='skip'`) e hover rico em cada UE e cada AP.
    """
    fig = go.Figure()
    area = snap.area_m
    M = len(snap.aps)
    grid_dim = max(1, math.isqrt(M))
    cell_size = area / grid_dim
    r_lobe = min(cell_size * 0.42, area * 0.14)

    # 1. Heatmap de cobertura 2D opcional (hoverinfo="skip" para não bloquear UE/AP!)
    if show_heatmap and snap.heatmap is not None:
        hm = snap.heatmap
        fig.add_trace(
            go.Heatmap(
                x=hm["x"],
                y=hm["y"],
                z=hm["z"],
                colorscale="Turbo",
                opacity=0.45,
                colorbar=dict(title=hm["label"], len=0.75, thickness=14),
                hoverinfo="skip",
                showscale=True,
                name="Mapa de Cobertura",
            )
        )

    # 2. Grade das células (hoverinfo="skip")
    gx_lines: list[Optional[float]] = []
    gy_lines: list[Optional[float]] = []
    for k in range(grid_dim + 1):
        pos = k * cell_size
        gx_lines.extend([pos, pos, None, 0.0, area, None])
        gy_lines.extend([0.0, area, None, pos, pos, None])
    fig.add_trace(
        go.Scatter(
            x=gx_lines,
            y=gy_lines,
            mode="lines",
            line=dict(color="rgba(148,163,184,0.35)", width=1, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # 3. Lóbulos de feixe reais (hoverinfo="skip")
    if show_beams:
        active_pairs = {(u.serving_ap, u.serving_beam) for u in snap.ues if not u.is_blocked}
        for ap in snap.aps:
            col = AP_PALETTE_HEX[ap.ap_id % len(AP_PALETTE_HEX)]
            if snap.antenna_model == "omni":
                lx, ly = compute_beam_lobe_xy(ap.x, ap.y, "omni", radius=r_lobe)
                fig.add_trace(
                    go.Scatter(
                        x=lx,
                        y=ly,
                        mode="lines",
                        fill="toself",
                        fillcolor="rgba(56,189,248,0.08)",
                        line=dict(color=col, width=1, dash="dot"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            elif snap.antenna_model == "sector":
                for b in range(3):
                    is_act = (ap.ap_id, b) in active_pairs
                    lx, ly = compute_beam_lobe_xy(
                        ap.x,
                        ap.y,
                        "sector",
                        radius=r_lobe,
                        beam_idx=b,
                        beamwidth_deg=snap.beamwidth_deg,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=lx,
                            y=ly,
                            mode="lines",
                            fill="toself",
                            fillcolor="rgba(245,158,11,0.22)" if is_act else "rgba(148,163,184,0.05)",
                            line=dict(
                                color=col if is_act else "rgba(148,163,184,0.30)",
                                width=1.6 if is_act else 0.8,
                            ),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
            elif snap.antenna_model == "smb":
                for b in range(snap.J):
                    is_act = (ap.ap_id, b) in active_pairs
                    if not is_act and M > 9:
                        continue
                    lx, ly = compute_beam_lobe_xy(
                        ap.x, ap.y, "smb", radius=r_lobe, L=snap.L, J=snap.J, beam_idx=b, n_pts=120
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=lx,
                            y=ly,
                            mode="lines",
                            fill="toself" if is_act else None,
                            fillcolor="rgba(16,185,129,0.20)" if is_act else None,
                            line=dict(
                                color=col if is_act else "rgba(148,163,184,0.22)",
                                width=1.6 if is_act else 0.6,
                                dash="solid" if is_act else "dot",
                            ),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
            else:
                # Adaptativo MRC (UCA) ou ULA — um lóbulo por UE ativo conectado ao AP
                for u in snap.ues:
                    if u.serving_ap != ap.ap_id or u.is_blocked:
                        continue
                    b_ang = float(snap.beam_angles_rad[u.ue_id])
                    lx, ly = compute_beam_lobe_xy(
                        ap.x,
                        ap.y,
                        snap.antenna_model,
                        beam_angle_rad=b_ang,
                        radius=min(r_lobe * 1.25, max(u.distance_m * 0.9, r_lobe * 0.7)),
                        L=snap.L,
                        n_pts=180,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=lx,
                            y=ly,
                            mode="lines",
                            fill="toself",
                            fillcolor="rgba(244,63,94,0.16)",
                            line=dict(color=col, width=1.4),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

    # 4. Linhas de conexão UE ↔ AP (hoverinfo="skip")
    if show_links and snap.ues:
        # Agrupa por AP para colorir cada link com a cor do AP servidor
        for ap in snap.aps:
            conn_ues = [u for u in snap.ues if u.serving_ap == ap.ap_id]
            if not conn_ues:
                continue
            col = AP_PALETTE_HEX[ap.ap_id % len(AP_PALETTE_HEX)]
            lx_act: list[Optional[float]] = []
            ly_act: list[Optional[float]] = []
            lx_out: list[Optional[float]] = []
            ly_out: list[Optional[float]] = []
            for u in conn_ues:
                if u.is_outage:
                    lx_out.extend([u.x, ap.x, None])
                    ly_out.extend([u.y, ap.y, None])
                else:
                    lx_act.extend([u.x, ap.x, None])
                    ly_act.extend([u.y, ap.y, None])
            if lx_act:
                fig.add_trace(
                    go.Scatter(
                        x=lx_act,
                        y=ly_act,
                        mode="lines",
                        line=dict(color=col, width=1.7),
                        opacity=0.80,
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            if lx_out:
                fig.add_trace(
                    go.Scatter(
                        x=lx_out,
                        y=ly_out,
                        mode="lines",
                        line=dict(color="#ef4444", width=1.2, dash="dot"),
                        opacity=0.65,
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    # 5. Linhas de interferência co-canal (Parte 1 Ex. 15) — opcional
    if show_interference and snap.ues:
        if victim_ue_id is None or not (0 <= victim_ue_id < len(snap.ues)):
            victim = min(snap.ues, key=lambda u: (-1e9 if u.is_blocked else u.sinr_db))
        else:
            victim = snap.ues[victim_ue_id]
        ix: list[Optional[float]] = []
        iy: list[Optional[float]] = []
        for u in snap.ues:
            if u.ue_id == victim.ue_id or u.is_blocked:
                continue
            if u.channel == victim.channel or u.is_aggregated or victim.is_aggregated:
                ix.extend([u.x, victim.ap_x, None])
                iy.extend([u.y, victim.ap_y, None])
        if ix:
            fig.add_trace(
                go.Scatter(
                    x=ix,
                    y=iy,
                    mode="lines",
                    line=dict(color="#dc2626", width=1.4, dash="dash"),
                    opacity=0.75,
                    hoverinfo="skip",
                    name=f"Interferência CCI → AP {victim.serving_ap} (UE {victim.ue_id})",
                )
            )

    # 6. UEs Ativos (círculo colorido pelo AP servidor) + UEs em Outage (X vermelho/escuro)
    active_ues = [u for u in snap.ues if not u.is_outage]
    outage_ues = [u for u in snap.ues if u.is_outage]

    if active_ues:
        fig.add_trace(
            go.Scatter(
                x=[u.x for u in active_ues],
                y=[u.y for u in active_ues],
                mode="markers+text",
                text=[f"UE {u.ue_id}" for u in active_ues],
                textposition="top center",
                textfont=dict(size=10),
                marker=dict(
                    symbol="circle",
                    size=12,
                    color=[AP_PALETTE_HEX[u.serving_ap % len(AP_PALETTE_HEX)] for u in active_ues],
                    line=dict(color="#0f172a", width=1.5),
                ),
                customdata=[_ue_hover_row(u) for u in active_ues],
                hovertemplate=_UE_HOVERTEMPLATE,
                name="UEs Ativos",
            )
        )

    if outage_ues:
        fig.add_trace(
            go.Scatter(
                x=[u.x for u in outage_ues],
                y=[u.y for u in outage_ues],
                mode="markers+text",
                text=[f"UE {u.ue_id}" for u in outage_ues],
                textposition="top center",
                textfont=dict(size=10, color="#ef4444"),
                marker=dict(
                    symbol="x",
                    size=13,
                    color="#dc2626",
                    line=dict(color="#7f1d1d", width=1.8),
                ),
                customdata=[_ue_hover_row(u) for u in outage_ues],
                hovertemplate=_UE_HOVERTEMPLATE,
                name="UEs em Outage",
            )
        )

    # 7. APs (triângulo ^ com hover rico contendo lista exata de UEs conectados e outage)
    ap_custom = []
    for ap in snap.aps:
        ue_list_str = (
            ", ".join(f"UE {uid}" for uid in ap.connected_ue_ids)
            if ap.connected_ue_ids
            else "Nenhum"
        )
        ap_custom.append(
            [
                ap.ap_id,
                ap.x,
                ap.y,
                f"{ap.num_ues} UE{'s' if ap.num_ues != 1 else ''}",
                ue_list_str,
                ap.sum_capacity_mbps,
                ap.num_outage,
            ]
        )

    fig.add_trace(
        go.Scatter(
            x=[ap.x for ap in snap.aps],
            y=[ap.y for ap in snap.aps],
            mode="markers+text",
            text=[f"AP {ap.ap_id}" for ap in snap.aps],
            textposition="bottom center",
            textfont=dict(size=11),
            marker=dict(
                symbol="triangle-up",
                size=16,
                color=[AP_PALETTE_HEX[ap.ap_id % len(AP_PALETTE_HEX)] for ap in snap.aps],
                line=dict(color="#0f172a", width=2),
            ),
            customdata=ap_custom,
            hovertemplate=_AP_HOVERTEMPLATE,
            name="Access Points (APs)",
        )
    )

    ant_lbl = MODEL_LABELS.get(snap.antenna_model, snap.antenna_model)
    pwr_lbl = MODEL_LABELS.get(snap.power_mode, snap.power_mode)
    _base_layout(
        fig,
        height=620,
        title=f"Snapshot Interativo 2D — {ant_lbl} | {pwr_lbl} | Seed={snap.seed}",
    )
    fig.update_xaxes(title_text="x (m)", range=[-area * 0.03, area * 1.03])
    fig.update_yaxes(
        title_text="y (m)",
        range=[-area * 0.03, area * 1.03],
        scaleanchor="x",
        scaleratio=1.0,
    )
    return fig


# ---------------------------------------------------------------------------
# 1B. Mapa 3D Interativo (Bônus opcional na Tela 1)
# ---------------------------------------------------------------------------


def build_snapshot_map_3d(
    snap: SnapshotResult,
    show_beams: bool = True,
    show_links: bool = True,
    show_heatmap: bool = True,
) -> go.Figure:
    """Visualização 3D WebGL opcional com torres dos APs, feixes reais, heatmap e hover rico."""
    fig = go.Figure()
    area = snap.area_m
    M = len(snap.aps)
    r_lobe = min((area / max(1, math.isqrt(M))) * 0.42, area * 0.14)

    if show_heatmap and snap.heatmap is not None:
        hm = snap.heatmap
        z_vals = np.asarray(hm["z"], dtype=float)
        fig.add_trace(
            go.Surface(
                x=hm["x"],
                y=hm["y"],
                z=np.full_like(z_vals, 0.3),
                surfacecolor=z_vals,
                colorscale="Turbo",
                opacity=0.65,
                colorbar=dict(title=hm["label"], len=0.65, thickness=14),
                hoverinfo="skip",
                name="Mapa de Cobertura",
            )
        )

    # Mastros das torres (hoverinfo="skip")
    tx: list[Optional[float]] = []
    ty: list[Optional[float]] = []
    tz: list[Optional[float]] = []
    for ap in snap.aps:
        tx.extend([ap.x, ap.x, None])
        ty.extend([ap.y, ap.y, None])
        tz.extend([0.0, ap.z, None])
    fig.add_trace(
        go.Scatter3d(
            x=tx,
            y=ty,
            z=tz,
            mode="lines",
            line=dict(color="#64748b", width=6),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    if show_beams:
        for u in snap.ues:
            if u.is_blocked:
                continue
            ap = snap.aps[u.serving_ap]
            col = AP_PALETTE_HEX[ap.ap_id % len(AP_PALETTE_HEX)]
            lx, ly = compute_beam_lobe_xy(
                ap.x,
                ap.y,
                snap.antenna_model,
                beam_angle_rad=float(snap.beam_angles_rad[u.ue_id]),
                radius=r_lobe,
                L=snap.L,
                J=snap.J,
                beam_idx=u.serving_beam,
                beamwidth_deg=snap.beamwidth_deg,
                n_pts=90,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=lx,
                    y=ly,
                    z=np.full_like(lx, ap.z - 2.0),
                    mode="lines",
                    line=dict(color=col, width=3),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    if show_links and snap.ues:
        lx_l: list[Optional[float]] = []
        ly_l: list[Optional[float]] = []
        lz_l: list[Optional[float]] = []
        for u in snap.ues:
            if u.is_blocked:
                continue
            ap = snap.aps[u.serving_ap]
            lx_l.extend([u.x, ap.x, None])
            ly_l.extend([u.y, ap.y, None])
            lz_l.extend([u.z, ap.z, None])
        if lx_l:
            fig.add_trace(
                go.Scatter3d(
                    x=lx_l,
                    y=ly_l,
                    z=lz_l,
                    mode="lines",
                    line=dict(color="rgba(56,189,248,0.65)", width=2, dash="dash"),
                    hoverinfo="skip",
                    name="Enlaces UE ↔ AP",
                )
            )

    # UEs 3D com o mesmo hover completo
    active_ues = [u for u in snap.ues if not u.is_outage]
    outage_ues = [u for u in snap.ues if u.is_outage]
    if active_ues:
        fig.add_trace(
            go.Scatter3d(
                x=[u.x for u in active_ues],
                y=[u.y for u in active_ues],
                z=[u.z for u in active_ues],
                mode="markers+text",
                text=[f"UE {u.ue_id}" for u in active_ues],
                marker=dict(
                    size=7,
                    color=[AP_PALETTE_HEX[u.serving_ap % len(AP_PALETTE_HEX)] for u in active_ues],
                    line=dict(color="#ffffff", width=1.5),
                ),
                customdata=[_ue_hover_row(u) for u in active_ues],
                hovertemplate=_UE_HOVERTEMPLATE,
                name="UEs Ativos",
            )
        )
    if outage_ues:
        fig.add_trace(
            go.Scatter3d(
                x=[u.x for u in outage_ues],
                y=[u.y for u in outage_ues],
                z=[u.z for u in outage_ues],
                mode="markers+text",
                text=[f"UE {u.ue_id} (OFF)" for u in outage_ues],
                marker=dict(size=7, symbol="x", color="#ef4444"),
                customdata=[_ue_hover_row(u) for u in outage_ues],
                hovertemplate=_UE_HOVERTEMPLATE,
                name="UEs em Outage",
            )
        )

    ap_custom = [
        [
            ap.ap_id,
            ap.x,
            ap.y,
            f"{ap.num_ues} UE{'s' if ap.num_ues != 1 else ''}",
            ", ".join(f"UE {uid}" for uid in ap.connected_ue_ids) if ap.connected_ue_ids else "Nenhum",
            ap.sum_capacity_mbps,
            ap.num_outage,
        ]
        for ap in snap.aps
    ]
    fig.add_trace(
        go.Scatter3d(
            x=[ap.x for ap in snap.aps],
            y=[ap.y for ap in snap.aps],
            z=[ap.z for ap in snap.aps],
            mode="markers+text",
            text=[f"AP {ap.ap_id}" for ap in snap.aps],
            marker=dict(
                symbol="diamond",
                size=10,
                color=[AP_PALETTE_HEX[ap.ap_id % len(AP_PALETTE_HEX)] for ap in snap.aps],
                line=dict(color="#ffffff", width=1.5),
            ),
            customdata=ap_custom,
            hovertemplate=_AP_HOVERTEMPLATE,
            name="Access Points (APs)",
        )
    )

    fig.update_layout(
        height=640,
        margin=dict(l=0, r=0, t=42, b=0),
        title=f"Snapshot Interativo 3D (Bônus) — {MODEL_LABELS.get(snap.antenna_model, snap.antenna_model)} | Seed={snap.seed}",
        scene=dict(
            xaxis=dict(title="X (m)", range=[0, area]),
            yaxis=dict(title="Y (m)", range=[0, area]),
            zaxis=dict(title="Z (m)", range=[0, max(50.0, snap.aps[0].z * 2.2) if snap.aps else 60.0]),
            aspectratio=dict(x=1.0, y=1.0, z=0.32),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Painel 2x2 de KPIs Comparativos (Tela 1 + Aba 3)
# ---------------------------------------------------------------------------


def build_kpi_2x2_figure(
    metrics_dict: dict[str, dict[int, ScenarioMetrics]],
    k_scenarios: tuple[int, ...],
    title: str = "Comparativo de KPIs (Monte Carlo)",
) -> go.Figure:
    """
    Painel 2x2:
      1. Border Spectrum Efficiency (10º Pct) [bps/Hz]
      2. Sum-Capacity Spectrum Efficiency [bps/Hz]
      3. Eficiência Energética em escala log [Mbits/Joule]
      4. Taxa de Outage [%]
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "1. Border SE (10º Percentil)",
            "2. Sum-Capacity SE (Média)",
            "3. Eficiência Energética (Escala Log)",
            "4. Taxa de Outage (%)",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.09,
    )

    x_labels = [f"K = {k} UEs" for k in k_scenarios]
    keys = list(metrics_dict.keys())

    specs = [
        ("border_se_p10", 1, 1, "{:.2f}"),
        ("sum_se_mean", 1, 2, "{:.2f}"),
        ("energy_efficiency_mbit_j", 2, 1, "{:.1f}"),
        ("outage_rate_pct", 2, 2, "{:.1f}%"),
    ]

    for key in keys:
        lbl = MODEL_LABELS.get(key, key)
        col = PLOTLY_COLORS.get(key, "#38bdf8")
        for attr, row, c_idx, fmt in specs:
            vals = [getattr(metrics_dict[key][k], attr) for k in k_scenarios]
            plot_vals = [max(v, 1e-3) if attr == "energy_efficiency_mbit_j" else v for v in vals]
            fig.add_trace(
                go.Bar(
                    x=x_labels,
                    y=plot_vals,
                    name=lbl,
                    legendgroup=key,
                    showlegend=(row == 1 and c_idx == 1),
                    marker_color=col,
                    text=[fmt.format(v) for v in vals],
                    textposition="auto",
                ),
                row=row,
                col=c_idx,
            )

    _base_layout(fig, height=680, title=title)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="SE [bps/Hz]", row=1, col=1)
    fig.update_yaxes(title_text="Sum-SE [bps/Hz]", row=1, col=2)
    fig.update_yaxes(title_text="Mbits / Joule (Log)", type="log", row=2, col=1)
    fig.update_yaxes(title_text="Outage [%]", row=2, col=2)
    return fig


# ---------------------------------------------------------------------------
# 3. CDFs Comparativas (Tela 1 + Aba 2)
# ---------------------------------------------------------------------------


def build_cdfs_figure(
    metrics_dict: dict[str, dict[int, ScenarioMetrics]],
    K: int,
    gamma_min_db: float = -10.0,
    gamma_tar_db: float = 10.0,
    gamma_max_db: float = 20.0,
    show_ref_lines: bool = True,
    power_metric: str = "tx_dbm",  # "tx_dbm" ou "rx_dbm"
    cap_metric: str = "se",  # "se" ou "mbps"
    title: Optional[str] = None,
) -> go.Figure:
    """Painel 1x3 de CDFs: Potência, SINR (dB) e Capacidade/SE para uma carga K."""
    p_title = "Potência Transmitida (dBm)" if power_metric == "tx_dbm" else "Potência Recebida (dBm)"
    c_title = "Eficiência Espectral (bps/Hz)" if cap_metric == "se" else "Capacidade (Mbps)"
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(f"CDF {p_title}", "CDF SINR (dB)", f"CDF {c_title}"),
        horizontal_spacing=0.07,
    )

    pct_grid = np.linspace(0.0, 100.0, 250)
    y_prob = pct_grid / 100.0

    for key, by_k in metrics_dict.items():
        if K not in by_k:
            continue
        sm = by_k[K]
        lbl = MODEL_LABELS.get(key, key)
        col = PLOTLY_COLORS.get(key, "#38bdf8")

        # 1. Potência
        raw_p = sm.tx_power_w if power_metric == "tx_dbm" else sm.rx_power_w
        if power_metric == "tx_dbm":
            p_dbm = np.where(raw_p > 0.0, 10.0 * np.log10(np.maximum(raw_p, 1e-12) * 1000.0), -5.0)
        else:
            pos_p = raw_p[raw_p > 0.0]
            p_dbm = 10.0 * np.log10(np.maximum(pos_p, 1e-18) * 1000.0) if pos_p.size else np.array([-150.0])

        fig.add_trace(
            go.Scatter(
                x=np.percentile(p_dbm, pct_grid),
                y=y_prob,
                mode="lines",
                name=lbl,
                legendgroup=key,
                line=dict(color=col, width=2.4),
            ),
            row=1,
            col=1,
        )

        # 2. SINR (dB)
        fig.add_trace(
            go.Scatter(
                x=np.percentile(sm.sinr_db, pct_grid),
                y=y_prob,
                mode="lines",
                name=lbl,
                legendgroup=key,
                showlegend=False,
                line=dict(color=col, width=2.4),
            ),
            row=1,
            col=2,
        )

        # 3. Capacidade / SE
        c_arr = sm.se_bps_hz if cap_metric == "se" else sm.capacity_mbps
        fig.add_trace(
            go.Scatter(
                x=np.percentile(c_arr, pct_grid),
                y=y_prob,
                mode="lines",
                name=lbl,
                legendgroup=key,
                showlegend=False,
                line=dict(color=col, width=2.4),
            ),
            row=1,
            col=3,
        )

    if show_ref_lines:
        for col_i in (1, 2, 3):
            fig.add_hline(y=0.10, line_dash="dot", line_color="#ef4444", line_width=1, row=1, col=col_i)
            fig.add_hline(y=0.50, line_dash="dot", line_color="#64748b", line_width=1, row=1, col=col_i)
        fig.add_vline(
            x=gamma_min_db,
            line_dash="dash",
            line_color="#ef4444",
            line_width=1.2,
            annotation_text=f"Outage ({gamma_min_db:g} dB)",
            annotation_position="top left",
            row=1,
            col=2,
        )
        fig.add_vline(
            x=gamma_tar_db,
            line_dash="dash",
            line_color="#38bdf8",
            line_width=1.2,
            annotation_text=f"Meta ({gamma_tar_db:g} dB)",
            row=1,
            col=2,
        )
        fig.add_vline(
            x=gamma_max_db,
            line_dash="dash",
            line_color="#10b981",
            line_width=1.2,
            annotation_text=f"QAM ({gamma_max_db:g} dB)",
            row=1,
            col=2,
        )

    _base_layout(fig, height=430, title=title or f"CDFs Comparativas — Carga K = {K} UEs")
    fig.update_xaxes(title_text=p_title, row=1, col=1)
    fig.update_xaxes(title_text="SINR (dB)", row=1, col=2)
    fig.update_xaxes(title_text=c_title, row=1, col=3)
    fig.update_yaxes(title_text="Probabilidade Acumulada", range=[0, 1.02], row=1, col=1)
    fig.update_yaxes(range=[0, 1.02], row=1, col=2)
    fig.update_yaxes(range=[0, 1.02], row=1, col=3)
    return fig


# ---------------------------------------------------------------------------
# 4. Mobilidade, Tracking Preditivo & SDMA (Aba 4)
# ---------------------------------------------------------------------------


def build_mobility_tracking_figure(
    res_hold: MobilityResult,
    res_pred: Optional[MobilityResult] = None,
) -> go.Figure:
    """Painel 2x1: Array Gain vs. Time (Sawtooth vs. Preditivo vs. Ideal) e Misalignment Loss."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            f"Array Gain vs. Tempo (v={res_hold.v_kmh:g} km/h | L={res_hold.L} | ΔT={res_hold.delta_t_s*1e3:g} ms | D={res_hold.D_m:g} m)",
            "Beam Misalignment Loss vs. Tempo (dB)",
        ),
        vertical_spacing=0.12,
    )

    # Downsample suave se > 2000 pontos apenas para fluidez visual mantendo picos
    stride = max(1, len(res_hold.t_s) // 2000)
    t = res_hold.t_s[::stride]

    fig.add_trace(
        go.Scatter(
            x=t,
            y=res_hold.gain_ideal_db[::stride],
            mode="lines",
            name=f"Ideal 10·log₁₀(L) = {res_hold.gain_ideal_db[0]:.1f} dB",
            line=dict(color="#94a3b8", width=1.6, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=res_hold.gain_actual_db[::stride],
            mode="lines",
            name=f"Padrão (Hold ΔT) — Perda média {res_hold.mean_loss_db:.2f} dB",
            legendgroup="hold",
            line=dict(color="#ef4444", width=1.7),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=res_hold.misalignment_loss_db[::stride],
            mode="lines",
            name="Perda Hold (dB)",
            legendgroup="hold",
            showlegend=False,
            line=dict(color="#ef4444", width=1.7),
        ),
        row=2,
        col=1,
    )

    if res_pred is not None:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=res_pred.gain_actual_db[::stride],
                mode="lines",
                name=f"Tracking Preditivo — Perda média {res_pred.mean_loss_db:.2f} dB",
                legendgroup="pred",
                line=dict(color="#10b981", width=2.2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=res_pred.misalignment_loss_db[::stride],
                mode="lines",
                name="Perda Preditivo (dB)",
                legendgroup="pred",
                showlegend=False,
                line=dict(color="#10b981", width=2.2),
            ),
            row=2,
            col=1,
        )

    _base_layout(fig, height=580)
    fig.update_yaxes(title_text="Array Gain [dB]", row=1, col=1)
    fig.update_yaxes(title_text="Misalignment Loss [dB]", row=2, col=1)
    fig.update_xaxes(title_text="Tempo de Simulação [s] (Broadside no centro)", row=2, col=1)
    return fig


def build_mobility_sweeps_figure(study: dict[str, Any]) -> go.Figure:
    """Painel 1x3 comparando perda de desalinhamento por Velocidade, L e Distância D."""
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Perda por Velocidade",
            "Perda por Tamanho do Arranjo (L)",
            "Perda por Distância Tangencial (D)",
        ),
        horizontal_spacing=0.07,
    )

    vel_colors = ["#10b981", "#38bdf8", "#ef4444", "#a855f7"]
    for idx, (v_val, res) in enumerate(study["by_velocity"].items()):
        s = max(1, len(res.t_s) // 1200)
        fig.add_trace(
            go.Scatter(
                x=res.t_s[::s],
                y=res.misalignment_loss_db[::s],
                mode="lines",
                name=f"v = {v_val:g} km/h",
                line=dict(color=vel_colors[idx % len(vel_colors)], width=1.8),
            ),
            row=1,
            col=1,
        )

    l_colors = ["#38bdf8", "#f59e0b", "#f43f5e", "#a855f7"]
    for idx, (L_val, res) in enumerate(study["by_L"].items()):
        s = max(1, len(res.t_s) // 1200)
        fig.add_trace(
            go.Scatter(
                x=res.t_s[::s],
                y=res.misalignment_loss_db[::s],
                mode="lines",
                name=f"L = {L_val}",
                line=dict(color=l_colors[idx % len(l_colors)], width=1.8),
            ),
            row=1,
            col=2,
        )

    d_colors = ["#ef4444", "#f97316", "#38bdf8", "#10b981"]
    for idx, (d_val, res) in enumerate(study["by_distance"].items()):
        s = max(1, len(res.t_s) // 1200)
        fig.add_trace(
            go.Scatter(
                x=res.t_s[::s],
                y=res.misalignment_loss_db[::s],
                mode="lines",
                name=f"D = {d_val:g} m",
                line=dict(color=d_colors[idx % len(d_colors)], width=1.8),
            ),
            row=1,
            col=3,
        )

    _base_layout(fig, height=400, title="Sensibilidade do Beam Misalignment Loss")
    for c_i in (1, 2, 3):
        fig.update_xaxes(title_text="Tempo [s]", row=1, col=c_i)
    fig.update_yaxes(title_text="Perda [dB]", row=1, col=1)
    return fig


def build_mobility_tradeoff_figure(
    tradeoff_rows: list[dict[str, Any]],
    selected_mode: str = "hold",
) -> go.Figure:
    """Gráfico de barras de Trade-off ΔT vs. Eficiência Espectral Líquida Média por Velocidade e L."""
    rows = [r for r in tradeoff_rows if r["tracking_mode"] == selected_mode]
    velocities = sorted({r["v_kmh"] for r in rows})
    L_vals = sorted({r["L"] for r in rows})
    dt_vals = sorted({r["delta_t_ms"] for r in rows}, reverse=True)

    fig = make_subplots(
        rows=1,
        cols=len(velocities),
        subplot_titles=[f"v = {v:g} km/h" for v in velocities],
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )
    l_colors = ["#38bdf8", "#f59e0b", "#10b981", "#f43f5e"]

    for col_i, v_val in enumerate(velocities, start=1):
        for l_idx, L_val in enumerate(L_vals):
            y_vals = []
            for dt_ms in dt_vals:
                match = next(
                    (
                        r["mean_se_net"]
                        for r in rows
                        if r["v_kmh"] == v_val and r["L"] == L_val and r["delta_t_ms"] == dt_ms
                    ),
                    0.0,
                )
                y_vals.append(match)
            fig.add_trace(
                go.Bar(
                    x=[f"{dt_ms:g} ms" for dt_ms in dt_vals],
                    y=y_vals,
                    name=f"L = {L_val}",
                    legendgroup=f"L_{L_val}",
                    showlegend=(col_i == 1),
                    marker_color=l_colors[l_idx % len(l_colors)],
                    text=[f"{y:.2f}" for y in y_vals],
                    textposition="auto",
                ),
                row=1,
                col=col_i,
            )
        fig.update_xaxes(title_text="ΔT (ms)", row=1, col=col_i)

    mode_lbl = "Tracking Preditivo" if selected_mode == "predictive" else "Rastreamento Padrão (Hold)"
    _base_layout(fig, height=420, title=f"Trade-off ΔT vs. Eficiência Espectral Líquida Média — {mode_lbl}")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="SE Líquida Média [bps/Hz]", row=1, col=1)
    return fig


def build_sdma_figures(sdma: SDMAResult, combiner: str = "both") -> tuple[go.Figure, go.Figure]:
    """Gera (1) mapa de trajetórias 2D dos K UEs móveis e (2) SINR individual + Sum-SE (MRC vs. ZF)."""
    stride = max(1, len(sdma.t_s) // 1000)
    t = sdma.t_s[::stride]

    # Figura 1: Trajetórias 2D
    fig_traj = go.Figure()
    fig_traj.add_trace(
        go.Scatter(
            x=[0.0],
            y=[0.0],
            mode="markers+text",
            text=["AP (MIMO)"],
            textposition="bottom center",
            marker=dict(symbol="triangle-up", size=16, color="#10b981", line=dict(color="#0f172a", width=2)),
            name="AP na Origem",
        )
    )
    for k in range(sdma.K):
        col = _UE_TRACE_PALETTE[k % len(_UE_TRACE_PALETTE)]
        fig_traj.add_trace(
            go.Scatter(
                x=sdma.ue_x_m[::stride, k],
                y=sdma.ue_y_m[::stride, k],
                mode="lines",
                line=dict(color=col, width=2.4),
                name=f"UE {k} ({sdma.ue_velocities_kmh[k]:.0f} km/h, D={sdma.ue_y_m[0, k]:.0f} m)",
            )
        )
        fig_traj.add_trace(
            go.Scatter(
                x=[sdma.ue_x_m[0, k], sdma.ue_x_m[-1, k]],
                y=[sdma.ue_y_m[0, k], sdma.ue_y_m[-1, k]],
                mode="markers",
                marker=dict(symbol=["circle-open", "diamond"], size=9, color=col),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    _base_layout(fig_traj, height=380, title=f"Trajetórias 2D — SDMA Multi-usuário Móvel (K={sdma.K}, L={sdma.L})")
    fig_traj.update_xaxes(title_text="x (m)")
    fig_traj.update_yaxes(title_text="y (m)")

    # Figura 2: SINR por UE, Interferência Inter-usuário e Sum-SE (MRC vs. ZF)
    fig_metrics = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"SINR por UE ({'ZF' if combiner == 'zf' else 'MRC'}) [dB]",
            f"Interferência Inter-usuário ({'ZF' if combiner == 'zf' else 'MRC'}) [dBm]",
            "Sum-SE Instantânea: MRC vs. Zero-Forcing",
        ),
        horizontal_spacing=0.07,
    )
    sinr_mat = sdma.sinr_zf_db if combiner == "zf" else sdma.sinr_mrc_db
    int_mat = sdma.interf_zf_dbm if combiner == "zf" else sdma.interf_mrc_dbm
    for k in range(sdma.K):
        col = _UE_TRACE_PALETTE[k % len(_UE_TRACE_PALETTE)]
        fig_metrics.add_trace(
            go.Scatter(
                x=t,
                y=sinr_mat[::stride, k],
                mode="lines",
                name=f"UE {k}",
                legendgroup=f"sdma_ue_{k}",
                line=dict(color=col, width=1.8),
            ),
            row=1,
            col=1,
        )
        fig_metrics.add_trace(
            go.Scatter(
                x=t,
                y=int_mat[::stride, k],
                mode="lines",
                name=f"Interf UE {k}",
                legendgroup=f"sdma_ue_{k}",
                showlegend=False,
                line=dict(color=col, width=1.6),
            ),
            row=1,
            col=2,
        )

    if combiner in ("both", "mrc"):
        fig_metrics.add_trace(
            go.Scatter(
                x=t,
                y=sdma.sum_se_mrc[::stride],
                mode="lines",
                name=f"MRC (média {sdma.mean_sum_se_mrc:.2f} bps/Hz)",
                line=dict(color="#f97316", width=2.2),
            ),
            row=1,
            col=3,
        )
    if combiner in ("both", "zf"):
        fig_metrics.add_trace(
            go.Scatter(
                x=t,
                y=sdma.sum_se_zf[::stride],
                mode="lines",
                name=f"Zero-Forcing (média {sdma.mean_sum_se_zf:.2f} bps/Hz)",
                line=dict(color="#10b981", width=2.2),
            ),
            row=1,
            col=3,
        )

    _base_layout(fig_metrics, height=410)
    for c_i in (1, 2, 3):
        fig_metrics.update_xaxes(title_text="Tempo [s]", row=1, col=c_i)
    fig_metrics.update_yaxes(title_text="SINR [dB]", row=1, col=1)
    fig_metrics.update_yaxes(title_text="Interferência [dBm]", row=1, col=2)
    fig_metrics.update_yaxes(title_text="Sum-SE [bps/Hz]", row=1, col=3)
    return fig_traj, fig_metrics


# ---------------------------------------------------------------------------
# 5. Handover: Hard vs. Soft (Aba 5)
# ---------------------------------------------------------------------------


def build_handover_figures(res: HandoverResult) -> tuple[go.Figure, go.Figure]:
    """
    Figura 1: Step plot de AP conectado (Hard vs. Soft) marcando com 'X' cada Outage.
    Figura 2: Nível de potência recebida por AP + curva de SNR recebida (Hard vs. Soft).
    """
    d = res.ue_positions_m
    p_min_dbm = 10.0 * np.log10(res.config.p_min_w * 1000.0)

    # Figura 1: Decisão de AP (step plot) + Outages marcados com 'X'
    fig_dec = go.Figure()
    fig_dec.add_trace(
        go.Scatter(
            x=d,
            y=res.hard_serving_ap,
            mode="lines",
            line_shape="hv",
            line=dict(color="#a855f7", width=2.4),
            name=f"Hard Handover ({res.hard_handovers_sample} trocas)",
        )
    )
    if res.config.compare_hard_soft:
        fig_dec.add_trace(
            go.Scatter(
                x=d,
                y=res.soft_serving_ap + 0.06,
                mode="lines",
                line_shape="hv",
                line=dict(color="#10b981", width=2.2),
                name=f"Soft Handover (+0.06 offset | {res.soft_handovers_sample} trocas)",
            )
        )

    if np.any(res.hard_outage_mask):
        fig_dec.add_trace(
            go.Scatter(
                x=d[res.hard_outage_mask],
                y=res.hard_serving_ap[res.hard_outage_mask],
                mode="markers",
                marker=dict(symbol="x", size=11, color="#ef4444", line=dict(width=1.5)),
                name=f"Outage Hard ({int(np.sum(res.hard_outage_mask))})",
            )
        )
    if res.config.compare_hard_soft and np.any(res.soft_outage_mask):
        fig_dec.add_trace(
            go.Scatter(
                x=d[res.soft_outage_mask],
                y=res.soft_serving_ap[res.soft_outage_mask] + 0.06,
                mode="markers",
                marker=dict(symbol="x", size=11, color="#f59e0b", line=dict(width=1.5)),
                name=f"Outage Soft ({int(np.sum(res.soft_outage_mask))})",
            )
        )

    _base_layout(fig_dec, height=400, title="Decisão de AP Conectado e Eventos de Outage (X) — Hard vs. Soft")
    fig_dec.update_xaxes(title_text="Posição do UE na Rodovia (m)")
    fig_dec.update_yaxes(
        title_text="ID do Access Point",
        tickmode="array",
        tickvals=list(range(1, res.config.num_aps + 1)),
    )

    # Figura 2: Potência recebida de cada AP e SNR recebida (Hard vs. Soft)
    fig_sig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Potência Recebida por AP (dBm)", "SNR Recebida: Hard vs. Soft (dB)"),
        horizontal_spacing=0.08,
    )
    for m in range(res.config.num_aps):
        col = AP_PALETTE_HEX[m % len(AP_PALETTE_HEX)]
        fig_sig.add_trace(
            go.Scatter(
                x=d,
                y=res.rx_power_dbm[m],
                mode="lines",
                line=dict(color=col, width=1.6),
                name=f"Sinal AP {m + 1}",
            ),
            row=1,
            col=1,
        )
    fig_sig.add_hline(
        y=p_min_dbm,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"P_min ({p_min_dbm:.0f} dBm)",
        row=1,
        col=1,
    )

    fig_sig.add_trace(
        go.Scatter(
            x=d,
            y=res.hard_snr_db,
            mode="lines",
            line=dict(color="#a855f7", width=1.9),
            name="SNR Hard Handover",
        ),
        row=1,
        col=2,
    )
    if res.config.compare_hard_soft:
        fig_sig.add_trace(
            go.Scatter(
                x=d,
                y=res.soft_snr_db,
                mode="lines",
                line=dict(color="#10b981", width=2.0),
                name="SNR Soft Handover",
            ),
            row=1,
            col=2,
        )

    _base_layout(fig_sig, height=410)
    fig_sig.update_xaxes(title_text="Posição (m)", row=1, col=1)
    fig_sig.update_xaxes(title_text="Posição (m)", row=1, col=2)
    fig_sig.update_yaxes(title_text="Potência Recebida (dBm)", row=1, col=1)
    fig_sig.update_yaxes(title_text="SNR (dB)", row=1, col=2)
    return fig_dec, fig_sig


# ---------------------------------------------------------------------------
# 6. Convergência Passo a Passo do Snapshot (Aba 6)
# ---------------------------------------------------------------------------


def build_convergence_figures(
    hist: dict[str, Any],
    power_unit: str = "dbm",
    ue_filter: str = "all",
) -> tuple[go.Figure, go.Figure, go.Figure]:
    """
    Retorna 3 figuras Plotly sincronizadas da convergência do Snapshot:
      1. Evolução da Potência Tx P_k (dBm ou mW) com marcações de Outage ('X') e loop externo;
      2. Evolução da SINR real (dB) por iteração do loop interno;
      3. Evolução em degrau da meta negociada γ_tar_k (dB) no loop externo.
    """
    pw = np.asarray(hist["power_w"], dtype=float)
    sinr_lin = np.asarray(hist["sinr_lin"], dtype=float)
    tgt = np.asarray(hist["target_db"], dtype=float)
    outer_marks = [int(m) for m in hist.get("outer_marks", [])]

    T, K = pw.shape
    t_axis = np.arange(T)
    final_active = pw[-1] > 0.0 if T > 0 else np.ones(K, dtype=bool)

    if ue_filter == "active":
        ue_indices = [k for k in range(K) if final_active[k]]
    elif ue_filter == "off":
        ue_indices = [k for k in range(K) if not final_active[k]]
    else:
        ue_indices = list(range(K))

    p_max_dbm = float(hist.get("p_max_dbm", 30.0))
    p_min_dbm = float(hist.get("p_min_dbm", 0.0))
    g_max_db = float(hist.get("gamma_max_db", 20.0))
    g_min_db = float(hist.get("gamma_min_db", -10.0))

    fig_p = go.Figure()
    fig_s = go.Figure()
    fig_t = go.Figure()

    off_p_x: list[int] = []
    off_p_y: list[float] = []
    off_p_txt: list[str] = []
    off_s_x: list[int] = []
    off_s_y: list[float] = []
    off_s_txt: list[str] = []

    for k in ue_indices:
        col = _UE_TRACE_PALETTE[k % len(_UE_TRACE_PALETTE)]
        p_col = pw[:, k]
        s_col = sinr_lin[:, k]
        is_off = not bool(final_active[k])
        lbl = f"UE {k}" + (" (OFF)" if is_off else "")

        if power_unit == "mw":
            y_p = [float(v * 1000.0) if v > 0.0 else None for v in p_col]
        else:
            y_p = [float(10.0 * np.log10(v * 1000.0)) if v > 0.0 else None for v in p_col]

        y_s = [
            float(10.0 * np.log10(max(sv, 1e-12))) if pv > 0.0 else None
            for pv, sv in zip(p_col, s_col)
        ]

        fig_p.add_trace(
            go.Scatter(
                x=t_axis,
                y=y_p,
                mode="lines+markers" if T <= 80 else "lines",
                marker=dict(size=4),
                line=dict(color=col, width=2.0),
                name=lbl,
            )
        )
        fig_s.add_trace(
            go.Scatter(
                x=t_axis,
                y=y_s,
                mode="lines+markers" if T <= 80 else "lines",
                marker=dict(size=4),
                line=dict(color=col, width=2.0),
                name=lbl,
            )
        )

        # Marca o instante exato de bloqueio (Outage)
        off_idx = np.flatnonzero(p_col <= 0.0)
        if off_idx.size > 0 and off_idx[0] > 0:
            t_cut = int(off_idx[0] - 1)
            last_p = y_p[t_cut]
            last_s = y_s[t_cut]
            if last_p is not None:
                off_p_x.append(t_cut)
                off_p_y.append(last_p)
                off_p_txt.append(f"UE {k} bloqueado (Outage) no passo {int(off_idx[0])}")
            if last_s is not None:
                off_s_x.append(t_cut)
                off_s_y.append(last_s)
                off_s_txt.append(f"UE {k} bloqueado (Outage) no passo {int(off_idx[0])}")

        if tgt.ndim == 2 and tgt.shape[1] == K:
            fig_t.add_trace(
                go.Scatter(
                    x=np.arange(tgt.shape[0]),
                    y=tgt[:, k],
                    mode="lines+markers",
                    line_shape="hv",
                    line=dict(color=col, width=2.2),
                    name=lbl,
                )
            )

    if off_p_x:
        fig_p.add_trace(
            go.Scatter(
                x=off_p_x,
                y=off_p_y,
                mode="markers",
                marker=dict(symbol="x", size=12, color="#ef4444", line=dict(width=1.8)),
                text=off_p_txt,
                hovertemplate="%{text}<extra></extra>",
                name="Bloqueio (Outage → 0 W)",
            )
        )
    if off_s_x:
        fig_s.add_trace(
            go.Scatter(
                x=off_s_x,
                y=off_s_y,
                mode="markers",
                marker=dict(symbol="x", size=12, color="#ef4444", line=dict(width=1.8)),
                text=off_s_txt,
                hovertemplate="%{text}<extra></extra>",
                name="Bloqueio (Outage)",
            )
        )

    for mk in outer_marks[:35]:
        fig_p.add_vline(x=mk - 0.5, line_dash="dot", line_color="rgba(148,163,184,0.35)", line_width=1)
        fig_s.add_vline(x=mk - 0.5, line_dash="dot", line_color="rgba(148,163,184,0.35)", line_width=1)

    p_max_ref = 10.0 ** (p_max_dbm / 10.0) if power_unit == "mw" else p_max_dbm
    p_min_ref = 10.0 ** (p_min_dbm / 10.0) if power_unit == "mw" else p_min_dbm
    unit_lbl = "mW" if power_unit == "mw" else "dBm"
    fig_p.add_hline(y=p_max_ref, line_dash="dash", line_color="#38bdf8", annotation_text=f"P_max ({p_max_ref:.1f} {unit_lbl})")
    fig_p.add_hline(y=p_min_ref, line_dash="dot", line_color="#94a3b8", annotation_text=f"P_min ({p_min_ref:.1f} {unit_lbl})")

    fig_s.add_hline(y=g_max_db, line_dash="dash", line_color="#10b981", annotation_text=f"Teto QAM ({g_max_db:g} dB)")
    fig_s.add_hline(y=g_min_db, line_dash="dash", line_color="#ef4444", annotation_text=f"Outage ({g_min_db:g} dB)")

    fig_t.add_hline(y=g_max_db, line_dash="dash", line_color="#10b981", annotation_text=f"Teto ({g_max_db:g} dB)")
    fig_t.add_hline(y=g_min_db, line_dash="dash", line_color="#ef4444", annotation_text=f"Bloqueio ({g_min_db:g} dB)")

    _base_layout(fig_p, height=380, title=f"1. Convergência da Potência de Transmissão ({unit_lbl})")
    fig_p.update_xaxes(title_text="Iteração do Loop Interno (linhas pontilhadas = fim do Loop Externo)")
    fig_p.update_yaxes(title_text=f"Potência Tx ({unit_lbl})")

    _base_layout(fig_s, height=380, title="2. Evolução da SINR Real por UE (dB)")
    fig_s.update_xaxes(title_text="Iteração do Loop Interno")
    fig_s.update_yaxes(title_text="SINR Real (dB)")

    _base_layout(fig_t, height=340, title="3. Negociação da Meta de SINR γ_tar (Loop Externo)")
    fig_t.update_xaxes(title_text="Iteração do Loop Externo")
    fig_t.update_yaxes(title_text="γ_tar (dB)")

    return fig_p, fig_s, fig_t


# ---------------------------------------------------------------------------
# 7. KPIs Instantâneos do Snapshot (Grid 2x2 da branch `interface`) & Baseline
# ---------------------------------------------------------------------------


def build_snapshot_kpis_2x2_figure(
    snap: SnapshotResult,
    gamma_min_db: float = -10.0,
    gamma_tar_db: float = 10.0,
) -> go.Figure:
    """
    Painel 2x2 de KPIs instantâneos do Snapshot atual:
      1. SINR por UE (dB) com linhas de referência de Outage e Meta;
      2. Capacidade de Shannon por UE (Mbps);
      3. Potência de Transmissão por UE (dBm);
      4. Carga (nº UEs) e Vazão Agregada por AP (eixo duplo).
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "1. SINR por UE (dB)",
            "2. Capacidade por UE (Mbps)",
            "3. Potência de Transmissão por UE (dBm)",
            "4. Carga e Vazão Agregada por AP",
        ),
        specs=[[{}, {}], [{}, {"secondary_y": True}]],
        vertical_spacing=0.16,
        horizontal_spacing=0.09,
    )

    ue_labels = [f"UE {u.ue_id}" for u in snap.ues]
    sinr_vals = [gamma_min_db - 5.0 if u.is_blocked else u.sinr_db for u in snap.ues]
    sinr_colors = [
        "#ef4444"
        if u.is_outage
        else ("#10b981" if u.sinr_db >= gamma_tar_db else "#38bdf8")
        for u in snap.ues
    ]
    sinr_txt = ["OFF" if u.is_blocked else f"{u.sinr_db:+.1f}" for u in snap.ues]

    fig.add_trace(
        go.Bar(
            x=ue_labels,
            y=sinr_vals,
            marker_color=sinr_colors,
            text=sinr_txt,
            textposition="auto",
            name="SINR (dB)",
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>SINR: %{text} dB<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=gamma_min_db,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Outage ({gamma_min_db:g} dB)",
        row=1,
        col=1,
    )
    fig.add_hline(
        y=gamma_tar_db,
        line_dash="dot",
        line_color="#10b981",
        annotation_text=f"Meta ({gamma_tar_db:g} dB)",
        row=1,
        col=1,
    )

    cap_vals = [u.capacity_mbps for u in snap.ues]
    ap_ue_colors = [AP_PALETTE_HEX[u.serving_ap % len(AP_PALETTE_HEX)] for u in snap.ues]
    fig.add_trace(
        go.Bar(
            x=ue_labels,
            y=cap_vals,
            marker_color=ap_ue_colors,
            text=[f"{v:.1f}" for v in cap_vals],
            textposition="auto",
            name="Capacidade (Mbps)",
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Capacidade: %{y:.2f} Mbps<extra></extra>",
        ),
        row=1,
        col=2,
    )

    p_vals = [0.0 if u.is_blocked else u.tx_power_dbm for u in snap.ues]
    p_txt = ["OFF (0 W)" if u.is_blocked else f"{u.tx_power_dbm:.1f}" for u in snap.ues]
    p_cols = ["#64748b" if u.is_blocked else "#f59e0b" for u in snap.ues]
    fig.add_trace(
        go.Bar(
            x=ue_labels,
            y=p_vals,
            marker_color=p_cols,
            text=p_txt,
            textposition="auto",
            name="Potência Tx (dBm)",
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Potência Tx: %{text} dBm<extra></extra>",
        ),
        row=2,
        col=1,
    )

    ap_labels = [f"AP {ap.ap_id}" for ap in snap.aps]
    ap_caps = [ap.sum_capacity_mbps for ap in snap.aps]
    ap_loads = [ap.num_ues for ap in snap.aps]
    fig.add_trace(
        go.Bar(
            x=ap_labels,
            y=ap_caps,
            marker_color=[AP_PALETTE_HEX[ap.ap_id % len(AP_PALETTE_HEX)] for ap in snap.aps],
            text=[f"{v:.1f}" for v in ap_caps],
            textposition="auto",
            name="Vazão do AP (Mbps)",
            hovertemplate="<b>%{x}</b><br>Sum-Capacity: %{y:.2f} Mbps<extra></extra>",
        ),
        row=2,
        col=2,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=ap_labels,
            y=ap_loads,
            mode="lines+markers",
            line=dict(color="#f43f5e", width=2.2),
            marker=dict(size=8, symbol="diamond"),
            name="Nº UEs Conectados",
            hovertemplate="<b>%{x}</b><br>UEs conectados: %{y}<extra></extra>",
        ),
        row=2,
        col=2,
        secondary_y=True,
    )

    ant_lbl = MODEL_LABELS.get(snap.antenna_model, snap.antenna_model)
    pwr_lbl = MODEL_LABELS.get(snap.power_mode, snap.power_mode)
    _base_layout(
        fig,
        height=640,
        title=f"KPIs Instantâneos do Snapshot — {ant_lbl} | {pwr_lbl} | Seed={snap.seed}",
    )
    fig.update_yaxes(title_text="SINR [dB]", row=1, col=1)
    fig.update_yaxes(title_text="Capacidade [Mbps]", row=1, col=2)
    fig.update_yaxes(title_text="Potência Tx [dBm]", row=2, col=1)
    fig.update_yaxes(title_text="Vazão [Mbps]", row=2, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Nº UEs", rangemode="tozero", row=2, col=2, secondary_y=True)
    return fig


def build_snapshot_comparison_figure(
    snaps_dict: dict[str, SnapshotResult],
    title: str = "Comparativo Instantâneo no Mesmo Snapshot",
) -> go.Figure:
    """Compara múltiplas antenas ou múltiplos controles de potência na mesma realização (seed)."""
    fig = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=(
            "Sum-Capacity [Mbps]",
            "SINR Média (Ativos) [dB]",
            "Potência Tx Total [W]",
            "Taxa de Outage [%]",
        ),
        horizontal_spacing=0.07,
    )
    keys = list(snaps_dict.keys())
    labels = [MODEL_LABELS.get(k, k) for k in keys]
    colors = [PLOTLY_COLORS.get(k, "#38bdf8") for k in keys]

    specs = [
        ("sum_capacity_mbps", 1, "{:.1f}"),
        ("avg_sinr_db", 2, "{:+.1f}"),
        ("total_tx_power_w", 3, "{:.3f}"),
        ("outage_rate_pct", 4, "{:.1f}%"),
    ]
    for attr, col_i, fmt in specs:
        vals = [getattr(snaps_dict[k], attr) for k in keys]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=vals,
                marker_color=colors,
                text=[fmt.format(v) for v in vals],
                textposition="auto",
                showlegend=False,
            ),
            row=1,
            col=col_i,
        )

    _base_layout(fig, height=340, title=title)
    return fig


def build_antenna_baseline_gain_figure(
    antenna_metrics: dict[str, dict[int, ScenarioMetrics]],
    k_scenarios: tuple[int, ...],
    k_focus: Optional[int] = None,
) -> go.Figure:
    """
    Gráficos comparativos de Antenas vs. Baseline (trazidos da branch `interface`):
      1. Ganho de SINR 10º Pct (Borda) vs. Baseline [dB];
      2. Razão de Capacidade Agregada vs. Baseline [×];
      3. Vazão Agregada [Mbps] & Taxa de Outage [%] na carga selecionada.
    """
    keys = list(antenna_metrics.keys())
    base_key = "omni" if "omni" in antenna_metrics else keys[0]
    base_lbl = MODEL_LABELS.get(base_key, base_key)
    k_sel = int(k_focus) if k_focus in k_scenarios else int(k_scenarios[-1])

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"Ganho SINR 10º Pct (Borda) vs. {base_lbl} [dB]",
            f"Razão Sum-Capacity vs. {base_lbl} [×]",
            f"Vazão Agregada & Outage (K = {k_sel} UEs)",
        ),
        specs=[[{}, {}, {"secondary_y": True}]],
        horizontal_spacing=0.08,
    )

    x_k = [f"K = {k}" for k in k_scenarios]
    for key in keys:
        lbl = MODEL_LABELS.get(key, key)
        col = PLOTLY_COLORS.get(key, "#38bdf8")

        gains_db = [
            antenna_metrics[key][k].sinr_p10_db - antenna_metrics[base_key][k].sinr_p10_db
            for k in k_scenarios
        ]
        ratios = [
            (
                antenna_metrics[key][k].sum_cap_mean_mbps
                / max(antenna_metrics[base_key][k].sum_cap_mean_mbps, 1e-9)
            )
            for k in k_scenarios
        ]
        fig.add_trace(
            go.Bar(
                x=x_k,
                y=gains_db,
                name=lbl,
                legendgroup=key,
                marker_color=col,
                text=[f"{g:+.1f}" for g in gains_db],
                textposition="auto",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=x_k,
                y=ratios,
                name=lbl,
                legendgroup=key,
                showlegend=False,
                marker_color=col,
                text=[f"{r:.2f}×" for r in ratios],
                textposition="auto",
            ),
            row=1,
            col=2,
        )

    fig.add_hline(y=0.0, line_dash="dot", line_color="#94a3b8", row=1, col=1)
    fig.add_hline(y=1.0, line_dash="dot", line_color="#94a3b8", row=1, col=2)

    # Painel 3: Sum-Capacity + Outage (%) na carga k_sel
    m_labels = [MODEL_LABELS.get(k, k) for k in keys]
    sum_caps = [antenna_metrics[k][k_sel].sum_cap_mean_mbps for k in keys]
    outages = [antenna_metrics[k][k_sel].outage_rate_pct for k in keys]
    fig.add_trace(
        go.Bar(
            x=m_labels,
            y=sum_caps,
            marker_color=[PLOTLY_COLORS.get(k, "#38bdf8") for k in keys],
            text=[f"{v:.1f}" for v in sum_caps],
            textposition="auto",
            name="Sum-Capacity (Mbps)",
            showlegend=False,
        ),
        row=1,
        col=3,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=m_labels,
            y=outages,
            mode="lines+markers",
            line=dict(color="#ef4444", width=2.2),
            marker=dict(size=8, symbol="x"),
            name="Outage (%)",
        ),
        row=1,
        col=3,
        secondary_y=True,
    )

    _base_layout(fig, height=420, title="Comparativo de Antenas vs. Baseline (Monte Carlo)")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Ganho [dB]", row=1, col=1)
    fig.update_yaxes(title_text="Razão [×]", row=1, col=2)
    fig.update_yaxes(title_text="Sum-Capacity [Mbps]", row=1, col=3, secondary_y=False)
    fig.update_yaxes(title_text="Outage [%]", range=[0, 105], row=1, col=3, secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# 8. Mobilidade & Beam Tracking na REDE COMPLETA (M APs × K UEs)
# ---------------------------------------------------------------------------


def build_network_mobility_overview_figure(
    res: NetworkMobilityResult,
    compare_predictive: bool = True,
) -> go.Figure:
    """
    Painel 2 colunas unindo o mapa da REDE COMPLETA (M APs × K UEs móveis)
    com a curva clássica de Atenuação por Desalinhamento Angular Δθ (L = 8, 16, 32)
    ancorada nos erros angulares reais da simulação.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.56, 0.44],
        horizontal_spacing=0.09,
        subplot_titles=(
            f"Trajetórias 2D da Rede Completa ({res.M} APs × {res.K} UEs | ΔT={res.delta_t_s*1e3:g} ms)",
            "Perda por Desalinhamento Angular vs. Δθ (Comparativo L)",
        ),
    )

    area = res.area_m
    grid_dim = max(1, math.isqrt(res.M))
    cell_size = area / grid_dim

    # Grade de células (hoverinfo="skip")
    gx: list[Optional[float]] = []
    gy: list[Optional[float]] = []
    for i in range(grid_dim + 1):
        pos = i * cell_size
        gx.extend([pos, pos, None, 0.0, area, None])
        gy.extend([0.0, area, None, pos, pos, None])
    fig.add_trace(
        go.Scatter(
            x=gx,
            y=gy,
            mode="lines",
            line=dict(color="rgba(148,163,184,0.28)", width=1, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Trajetórias dos K UEs + pontos de atualização ΔT + lóbulo final
    stride = max(1, len(res.t_s) // 400)
    r_lobe = min(cell_size * 0.36, area * 0.12)
    upd_stride = max(1, len(res.update_indices) // 25)
    upd_show = res.update_indices[::upd_stride]

    for k in range(res.K):
        col = _UE_TRACE_PALETTE[k % len(_UE_TRACE_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=res.ue_x_m[::stride, k],
                y=res.ue_y_m[::stride, k],
                mode="lines",
                line=dict(color=col, width=2.2),
                name=f"UE {k} ({res.ue_velocities_kmh[k]:.0f} km/h)",
                hovertemplate=(
                    f"<b>UE {k}</b> ({res.ue_velocities_kmh[k]:.0f} km/h)<br>"
                    "X: %{x:.1f} m | Y: %{y:.1f} m<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        # Início (○) e Fim (◆)
        fig.add_trace(
            go.Scatter(
                x=[res.ue_x_m[0, k], res.ue_x_m[-1, k]],
                y=[res.ue_y_m[0, k], res.ue_y_m[-1, k]],
                mode="markers",
                marker=dict(symbol=["circle-open", "diamond"], size=9, color=col, line=dict(width=1.5)),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        # Pontos de atualização ΔT
        fig.add_trace(
            go.Scatter(
                x=res.ue_x_m[upd_show, k],
                y=res.ue_y_m[upd_show, k],
                mode="markers",
                marker=dict(symbol="circle-open", size=5, color="#f59e0b", line=dict(width=1)),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        # Lóbulo apontado pelo AP servidor no instante final
        ap_f = int(res.serving_ap[-1, k])
        lx, ly = compute_beam_lobe_xy(
            float(res.ap_xy[ap_f, 0]),
            float(res.ap_xy[ap_f, 1]),
            "mrc" if res.array_type == "uca" else "ula",
            beam_angle_rad=float(res.beam_angles_final_rad[k]),
            radius=r_lobe,
            L=res.L,
            n_pts=100,
        )
        fig.add_trace(
            go.Scatter(
                x=lx,
                y=ly,
                mode="lines",
                fill="toself",
                fillcolor="rgba(16,185,129,0.12)",
                line=dict(color=col, width=1.1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # APs da rede
    fig.add_trace(
        go.Scatter(
            x=res.ap_xy[:, 0],
            y=res.ap_xy[:, 1],
            mode="markers+text",
            text=[f"AP {m}" for m in range(res.M)],
            textposition="bottom center",
            marker=dict(
                symbol="triangle-up",
                size=14,
                color=[AP_PALETTE_HEX[m % len(AP_PALETTE_HEX)] for m in range(res.M)],
                line=dict(color="#0f172a", width=1.6),
            ),
            name="Access Points (APs)",
            hovertemplate="<b>%{text}</b><br>(%{x:.1f}, %{y:.1f}) m<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Coluna 2: Atenuação por Desalinhamento Angular (dB) vs Δθ (graus) — estilo branch `interface`
    max_deg = max(12.0, min(45.0, float(np.percentile(res.delta_theta_hold_deg, 99)) * 1.6))
    theta_deg = np.linspace(0.0, max_deg, 400)
    theta_rad = np.deg2rad(theta_deg)

    l_candidates = sorted({8, 16, 32, int(res.L)})
    l_palette = {8: "#38bdf8", 16: "#f59e0b", 32: "#f43f5e", 64: "#a855f7", 4: "#10b981"}
    for l_val in l_candidates:
        g_norm = calc_ula_gain(theta_rad, np.zeros_like(theta_rad), l_val) / float(l_val)
        atten_db = 10.0 * np.log10(np.maximum(g_norm, 1e-4))
        is_sel = l_val == res.L
        fig.add_trace(
            go.Scatter(
                x=theta_deg,
                y=atten_db,
                mode="lines",
                name=f"L = {l_val}" + (" (Ativo)" if is_sel else ""),
                line=dict(
                    color=l_palette.get(l_val, "#94a3b8"),
                    width=2.8 if is_sel else 1.5,
                    dash="solid" if is_sel else "dot",
                ),
                hovertemplate=f"<b>L = {l_val}</b><br>Δθ: %{{x:.2f}}°<br>Atenuação: %{{y:.2f}} dB<extra></extra>",
            ),
            row=1,
            col=2,
        )

    # Linhas verticais marcando o Δθ real observado na rede completa (Hold médio, P95 e Preditivo)
    p95_hold = float(np.percentile(res.delta_theta_hold_deg, 95))
    fig.add_vline(
        x=res.mean_delta_theta_hold_deg,
        line_dash="dash",
        line_color="#ef4444",
        line_width=2.0,
        annotation_text=f"Médio Hold: {res.mean_delta_theta_hold_deg:.2f}°",
        annotation_position="top right",
        row=1,
        col=2,
    )
    if p95_hold > res.mean_delta_theta_hold_deg + 0.3:
        fig.add_vline(
            x=p95_hold,
            line_dash="dot",
            line_color="#f97316",
            line_width=1.6,
            annotation_text=f"P95 Hold: {p95_hold:.2f}°",
            annotation_position="bottom right",
            row=1,
            col=2,
        )
    if compare_predictive:
        fig.add_vline(
            x=res.mean_delta_theta_pred_deg,
            line_dash="dash",
            line_color="#10b981",
            line_width=2.0,
            annotation_text=f"Médio Pred.: {res.mean_delta_theta_pred_deg:.2f}°",
            annotation_position="top left",
            row=1,
            col=2,
        )

    _base_layout(fig, height=520)
    fig.update_xaxes(title_text="X (m)", range=[-area * 0.03, area * 1.03], row=1, col=1)
    fig.update_yaxes(
        title_text="Y (m)",
        range=[-area * 0.03, area * 1.03],
        scaleanchor="x",
        scaleratio=1.0,
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Erro Angular Δθ (graus)", row=1, col=2)
    fig.update_yaxes(title_text="Atenuação por Desalinhamento (dB)", range=[-25, 1], row=1, col=2)
    return fig


def build_network_mobility_timeseries_figure(
    res: NetworkMobilityResult,
    view_mode: str = "aggregate",  # "aggregate" ou "per_ue"
    compare_predictive: bool = True,
) -> go.Figure:
    """
    Painel 3x1 da rede móvel completa:
      1. Ganho de Arranjo (dB) — Média da rede + banda min/max ou por UE;
      2. Perda por Desalinhamento (dB) — Média + Pior UE (Hold vs. Preditivo) ou por UE;
      3. Sum-SE Líquida Agregada de Todos os K UEs (bps/Hz).
    """
    mode_lbl = (
        f"Agregado dos {res.K} UEs" if view_mode == "aggregate" else f"Curvas Individuais ({res.K} UEs)"
    )
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            f"1. Ganho de Arranjo — {mode_lbl} (dB)",
            f"2. Perda por Desalinhamento — {mode_lbl} (dB)",
            f"3. Sum-SE Líquida Agregada da Rede ({res.K} UEs, descontando overhead {res.overhead_fraction*100:.1f}%)",
        ),
        vertical_spacing=0.10,
    )

    stride = max(1, len(res.t_s) // 1000)
    t = res.t_s[::stride]

    fig.add_trace(
        go.Scatter(
            x=t,
            y=np.full_like(t, res.gain_ideal_db),
            mode="lines",
            name=f"Ideal 10·log₁₀(L) = {res.gain_ideal_db:.1f} dB",
            line=dict(color="#94a3b8", width=1.5, dash="dash"),
        ),
        row=1,
        col=1,
    )

    if view_mode == "aggregate":
        g_hold_mean = np.mean(res.gain_hold_db[::stride], axis=1)
        g_hold_min = np.min(res.gain_hold_db[::stride], axis=1)
        l_hold_mean = np.mean(res.loss_hold_db[::stride], axis=1)
        l_hold_max = np.max(res.loss_hold_db[::stride], axis=1)

        fig.add_trace(
            go.Scatter(
                x=t,
                y=g_hold_mean,
                mode="lines",
                name=f"Média Hold ({res.K} UEs) — Perda média {res.mean_loss_hold_db:.2f} dB",
                legendgroup="hold",
                line=dict(color="#ef4444", width=2.0),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=g_hold_min,
                mode="lines",
                name="Pior UE (Hold)",
                legendgroup="hold_worst",
                line=dict(color="rgba(239,68,68,0.45)", width=1.2, dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=l_hold_mean,
                mode="lines",
                name="Perda Média Hold",
                legendgroup="hold",
                showlegend=False,
                line=dict(color="#ef4444", width=2.0),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=l_hold_max,
                mode="lines",
                name="Perda Máx (Pior UE Hold)",
                legendgroup="hold_worst",
                showlegend=False,
                line=dict(color="rgba(239,68,68,0.45)", width=1.2, dash="dot"),
            ),
            row=2,
            col=1,
        )

        if compare_predictive:
            g_pred_mean = np.mean(res.gain_pred_db[::stride], axis=1)
            l_pred_mean = np.mean(res.loss_pred_db[::stride], axis=1)
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=g_pred_mean,
                    mode="lines",
                    name=f"Média Tracking Preditivo ({res.K} UEs) — Perda média {res.mean_loss_pred_db:.2f} dB",
                    legendgroup="pred",
                    line=dict(color="#10b981", width=2.2),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=l_pred_mean,
                    mode="lines",
                    name="Perda Média Preditivo",
                    legendgroup="pred",
                    showlegend=False,
                    line=dict(color="#10b981", width=2.2),
                ),
                row=2,
                col=1,
            )
    else:
        for k in range(res.K):
            col = _UE_TRACE_PALETTE[k % len(_UE_TRACE_PALETTE)]
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=res.gain_hold_db[::stride, k],
                    mode="lines",
                    name=f"UE {k} (Hold)",
                    legendgroup=f"ue_{k}",
                    line=dict(color=col, width=1.6),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=res.loss_hold_db[::stride, k],
                    mode="lines",
                    name=f"Perda UE {k}",
                    legendgroup=f"ue_{k}",
                    showlegend=False,
                    line=dict(color=col, width=1.6),
                ),
                row=2,
                col=1,
            )

    # Row 3: Sum-SE líquida da rede completa
    fig.add_trace(
        go.Scatter(
            x=res.t_eval_s,
            y=res.sum_se_hold,
            mode="lines",
            name=f"Sum-SE Hold (média {res.mean_sum_se_hold:.2f} bps/Hz)",
            line=dict(color="#ef4444", width=2.0),
        ),
        row=3,
        col=1,
    )
    if compare_predictive:
        fig.add_trace(
            go.Scatter(
                x=res.t_eval_s,
                y=res.sum_se_pred,
                mode="lines",
                name=f"Sum-SE Preditivo (média {res.mean_sum_se_pred:.2f} bps/Hz)",
                line=dict(color="#10b981", width=2.2),
            ),
            row=3,
            col=1,
        )

    _base_layout(fig, height=700)
    fig.update_yaxes(title_text="Ganho [dB]", row=1, col=1)
    fig.update_yaxes(title_text="Perda [dB]", row=2, col=1)
    fig.update_yaxes(title_text="Sum-SE [bps/Hz]", row=3, col=1)
    fig.update_xaxes(title_text="Tempo de Simulação [s]", row=3, col=1)
    return fig


def build_network_sdma_figure(
    res: NetworkMobilityResult,
    combiner: str = "both",
) -> go.Figure:
    """SDMA na rede completa: SINR por UE, Interferência por UE e Sum-SE agregada (MRC vs. ZF)."""
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"SINR por UE ({'ZF' if combiner == 'zf' else 'MRC'}) [dB]",
            f"Interferência Co-Canal por UE ({'ZF' if combiner == 'zf' else 'MRC'}) [dBm]",
            f"Sum-SE Agregada ({res.K} UEs): MRC vs. Zero-Forcing",
        ),
        horizontal_spacing=0.07,
    )
    t = res.t_eval_s
    sinr_mat = res.sinr_zf_db if combiner == "zf" else res.sinr_mrc_db
    int_mat = res.interf_zf_dbm if combiner == "zf" else res.interf_mrc_dbm

    for k in range(res.K):
        col = _UE_TRACE_PALETTE[k % len(_UE_TRACE_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=t,
                y=sinr_mat[:, k],
                mode="lines",
                name=f"UE {k}",
                legendgroup=f"nsdma_ue_{k}",
                line=dict(color=col, width=1.7),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=int_mat[:, k],
                mode="lines",
                name=f"Interf UE {k}",
                legendgroup=f"nsdma_ue_{k}",
                showlegend=False,
                line=dict(color=col, width=1.5),
            ),
            row=1,
            col=2,
        )

    if combiner in ("both", "mrc"):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=res.sum_se_mrc,
                mode="lines",
                name=f"MRC (média {res.mean_sum_se_mrc:.2f} bps/Hz)",
                line=dict(color="#f97316", width=2.2),
            ),
            row=1,
            col=3,
        )
    if combiner in ("both", "zf"):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=res.sum_se_zf,
                mode="lines",
                name=f"Zero-Forcing (média {res.mean_sum_se_zf:.2f} bps/Hz)",
                line=dict(color="#10b981", width=2.2),
            ),
            row=1,
            col=3,
        )

    _base_layout(fig, height=400)
    for c_i in (1, 2, 3):
        fig.update_xaxes(title_text="Tempo [s]", row=1, col=c_i)
    fig.update_yaxes(title_text="SINR [dB]", row=1, col=1)
    fig.update_yaxes(title_text="Interferência [dBm]", row=1, col=2)
    fig.update_yaxes(title_text="Sum-SE [bps/Hz]", row=1, col=3)
    return fig


# ---------------------------------------------------------------------------
# 9. Handover na REDE COMPLETA (M APs × K UEs)
# ---------------------------------------------------------------------------


def build_network_handover_map_figure(
    res: NetworkHandoverResult,
    algo_view: str = "hard",
) -> go.Figure:
    """Mapa 2D com as trajetórias de todos os K UEs, marcando eventos de Handover (◆) e Outage (×)."""
    fig = go.Figure()
    area = res.area_m
    grid_dim = max(1, math.isqrt(res.M))
    cell_size = area / grid_dim

    gx: list[Optional[float]] = []
    gy: list[Optional[float]] = []
    for i in range(grid_dim + 1):
        pos = i * cell_size
        gx.extend([pos, pos, None, 0.0, area, None])
        gy.extend([0.0, area, None, pos, pos, None])
    fig.add_trace(
        go.Scatter(
            x=gx,
            y=gy,
            mode="lines",
            line=dict(color="rgba(148,163,184,0.30)", width=1, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    srv_mat = res.hard_serving_ap if algo_view == "hard" else res.soft_serving_ap
    out_mat = res.hard_outage_mask if algo_view == "hard" else res.soft_outage_mask
    snr_mat = res.hard_snr_db if algo_view == "hard" else res.soft_snr_db

    ho_x: list[float] = []
    ho_y: list[float] = []
    ho_txt: list[str] = []
    ox: list[float] = []
    oy: list[float] = []
    otxt: list[str] = []

    for k in range(res.K):
        col = _UE_TRACE_PALETTE[k % len(_UE_TRACE_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=res.ue_x_m[:, k],
                y=res.ue_y_m[:, k],
                mode="lines",
                line=dict(color=col, width=2.1),
                name=f"UE {k} ({res.ue_velocities_kmh[k]:.0f} km/h)",
                hoverinfo="skip",
            )
        )
        # Início
        fig.add_trace(
            go.Scatter(
                x=[res.ue_x_m[0, k]],
                y=[res.ue_y_m[0, k]],
                mode="markers+text",
                text=[f"UE {k}"],
                textposition="top center",
                textfont=dict(size=9),
                marker=dict(symbol="circle-open", size=9, color=col, line=dict(width=1.8)),
                showlegend=False,
                hovertemplate=f"<b>Início UE {k}</b><br>AP inicial: AP {int(srv_mat[0, k])}<extra></extra>",
            )
        )
        # Eventos de Handover
        ho_steps = np.flatnonzero(np.diff(srv_mat[:, k]) != 0) + 1
        for stp in ho_steps:
            ho_x.append(float(res.ue_x_m[stp, k]))
            ho_y.append(float(res.ue_y_m[stp, k]))
            ho_txt.append(
                f"<b>Handover UE {k}</b> (t={res.t_s[stp]:.1f} s)<br>"
                f"AP {int(srv_mat[stp - 1, k])} → AP {int(srv_mat[stp, k])}"
            )
        # Eventos de Outage
        out_steps = np.flatnonzero(out_mat[:, k])
        for stp in out_steps:
            ox.append(float(res.ue_x_m[stp, k]))
            oy.append(float(res.ue_y_m[stp, k]))
            otxt.append(
                f"<b>Outage UE {k}</b> (t={res.t_s[stp]:.1f} s)<br>"
                f"AP {int(srv_mat[stp, k])} | SNR: {snr_mat[stp, k]:+.1f} dB"
            )

    if ho_x:
        fig.add_trace(
            go.Scatter(
                x=ho_x,
                y=ho_y,
                mode="markers",
                marker=dict(symbol="diamond", size=9, color="#f59e0b", line=dict(color="#0f172a", width=1)),
                text=ho_txt,
                hovertemplate="%{text}<extra></extra>",
                name=f"Eventos de Handover ({len(ho_x)})",
            )
        )
    if ox:
        fig.add_trace(
            go.Scatter(
                x=ox,
                y=oy,
                mode="markers",
                marker=dict(symbol="x", size=9, color="#ef4444", line=dict(width=1.6)),
                text=otxt,
                hovertemplate="%{text}<extra></extra>",
                name=f"Eventos de Outage ({len(ox)})",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=res.ap_xy[:, 0],
            y=res.ap_xy[:, 1],
            mode="markers+text",
            text=[f"AP {m}" for m in range(res.M)],
            textposition="bottom center",
            marker=dict(
                symbol="triangle-up",
                size=15,
                color=[AP_PALETTE_HEX[m % len(AP_PALETTE_HEX)] for m in range(res.M)],
                line=dict(color="#0f172a", width=1.8),
            ),
            name="Access Points (APs)",
            hovertemplate="<b>%{text}</b><br>(%{x:.1f}, %{y:.1f}) m<extra></extra>",
        )
    )

    algo_lbl = "Hard Handover" if algo_view == "hard" else f"Soft Handover ({res.config.soft_combining})"
    _base_layout(
        fig,
        height=520,
        title=f"Mapa 2D da Rede Completa — {algo_lbl} ({res.M} APs × {res.K} UEs)",
    )
    fig.update_xaxes(title_text="X (m)", range=[-area * 0.03, area * 1.03])
    fig.update_yaxes(
        title_text="Y (m)",
        range=[-area * 0.03, area * 1.03],
        scaleanchor="x",
        scaleratio=1.0,
    )
    return fig


def build_network_handover_aggregate_figure(res: NetworkHandoverResult) -> go.Figure:
    """Painel 2x2 agregado de TODOS os K UEs: SNR média, Outage instantânea, Handovers/UE e Outage/UE."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"1. SNR Média de Todos os {res.K} UEs ao Longo do Tempo (dB)",
            f"2. Taxa de Outage Instantânea da Rede ({res.K} UEs) [%]",
            "3. Total de Handovers por UE (Hard vs. Soft)",
            "4. Taxa de Outage por UE (Hard vs. Soft) [%]",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.09,
    )

    t = res.t_s
    fig.add_trace(
        go.Scatter(
            x=t,
            y=np.mean(res.hard_snr_db, axis=1),
            mode="lines",
            name="Hard Handover",
            legendgroup="hard",
            line=dict(color="#a855f7", width=2.2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=np.mean(res.soft_snr_db, axis=1),
            mode="lines",
            name="Soft Handover",
            legendgroup="soft",
            line=dict(color="#10b981", width=2.2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=np.mean(res.hard_outage_mask, axis=1) * 100.0,
            mode="lines",
            name="Outage Hard",
            legendgroup="hard",
            showlegend=False,
            line=dict(color="#a855f7", width=2.0),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=np.mean(res.soft_outage_mask, axis=1) * 100.0,
            mode="lines",
            name="Outage Soft",
            legendgroup="soft",
            showlegend=False,
            line=dict(color="#10b981", width=2.0),
        ),
        row=1,
        col=2,
    )

    ue_labels = [f"UE {k}" for k in range(res.K)]
    fig.add_trace(
        go.Bar(
            x=ue_labels,
            y=res.hard_handovers_per_ue,
            marker_color="#a855f7",
            name="HO Hard",
            legendgroup="hard",
            showlegend=False,
            text=[str(int(v)) for v in res.hard_handovers_per_ue],
            textposition="auto",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=ue_labels,
            y=res.soft_handovers_per_ue,
            marker_color="#10b981",
            name="HO Soft",
            legendgroup="soft",
            showlegend=False,
            text=[str(int(v)) for v in res.soft_handovers_per_ue],
            textposition="auto",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=ue_labels,
            y=res.hard_outage_pct_per_ue,
            marker_color="#a855f7",
            name="Outage % Hard",
            legendgroup="hard",
            showlegend=False,
            text=[f"{v:.1f}%" for v in res.hard_outage_pct_per_ue],
            textposition="auto",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=ue_labels,
            y=res.soft_outage_pct_per_ue,
            marker_color="#10b981",
            name="Outage % Soft",
            legendgroup="soft",
            showlegend=False,
            text=[f"{v:.1f}%" for v in res.soft_outage_pct_per_ue],
            textposition="auto",
        ),
        row=2,
        col=2,
    )

    _base_layout(fig, height=620, title=f"Desempenho Agregado de Handover — Todos os {res.K} UEs")
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Tempo [s]", row=1, col=1)
    fig.update_xaxes(title_text="Tempo [s]", row=1, col=2)
    fig.update_yaxes(title_text="SNR Média [dB]", row=1, col=1)
    fig.update_yaxes(title_text="Outage [%]", row=1, col=2)
    fig.update_yaxes(title_text="Nº Handovers", row=2, col=1)
    fig.update_yaxes(title_text="Outage [%]", row=2, col=2)
    return fig


def build_network_handover_ue_detail_figure(
    res: NetworkHandoverResult,
    ue_id: int,
) -> tuple[go.Figure, go.Figure]:
    """Detalhe individual de um UE específico: AP conectado (Hard vs. Soft + Outage X) e RSRP/SNR."""
    k = max(0, min(int(ue_id), res.K - 1))
    t = res.t_s
    h_ap = res.hard_serving_ap[:, k]
    s_ap = res.soft_serving_ap[:, k]
    h_out = res.hard_outage_mask[:, k]
    s_out = res.soft_outage_mask[:, k]

    fig_dec = go.Figure()
    fig_dec.add_trace(
        go.Scatter(
            x=t,
            y=h_ap,
            mode="lines",
            line_shape="hv",
            line=dict(color="#a855f7", width=2.4),
            name=f"Hard Handover ({int(res.hard_handovers_per_ue[k])} trocas)",
        )
    )
    fig_dec.add_trace(
        go.Scatter(
            x=t,
            y=s_ap + 0.08,
            mode="lines",
            line_shape="hv",
            line=dict(color="#10b981", width=2.2),
            name=f"Soft Handover (+0.08 offset | {int(res.soft_handovers_per_ue[k])} trocas)",
        )
    )
    if np.any(h_out):
        fig_dec.add_trace(
            go.Scatter(
                x=t[h_out],
                y=h_ap[h_out],
                mode="markers",
                marker=dict(symbol="x", size=11, color="#ef4444", line=dict(width=1.6)),
                name=f"Outage Hard ({int(np.sum(h_out))})",
            )
        )
    if np.any(s_out):
        fig_dec.add_trace(
            go.Scatter(
                x=t[s_out],
                y=s_ap[s_out] + 0.08,
                mode="markers",
                marker=dict(symbol="x", size=11, color="#f59e0b", line=dict(width=1.6)),
                name=f"Outage Soft ({int(np.sum(s_out))})",
            )
        )

    _base_layout(fig_dec, height=340, title=f"Decisão de AP Servidor — UE {k} ({res.ue_velocities_kmh[k]:.0f} km/h)")
    fig_dec.update_xaxes(title_text="Tempo [s]")
    fig_dec.update_yaxes(
        title_text="AP Servidor",
        tickmode="array",
        tickvals=list(range(res.M)),
        ticktext=[f"AP {m}" for m in range(res.M)],
    )

    fig_sig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Potência Recebida por AP — UE {k} (dBm)",
            f"SNR Recebida — UE {k}: Hard vs. Soft (dB)",
        ),
        horizontal_spacing=0.08,
    )
    # Mostra os APs que foram relevantes (até 6 mais fortes) para não poluir com 25 APs
    peak_per_ap = np.max(res.rx_power_dbm[:, k, :], axis=0)
    top_aps = sorted(np.argsort(peak_per_ap)[-min(res.M, 6) :])
    for m in top_aps:
        col = AP_PALETTE_HEX[int(m) % len(AP_PALETTE_HEX)]
        fig_sig.add_trace(
            go.Scatter(
                x=t,
                y=res.rx_power_dbm[:, k, int(m)],
                mode="lines",
                line=dict(color=col, width=1.6),
                name=f"AP {int(m)}",
            ),
            row=1,
            col=1,
        )
    fig_sig.add_hline(
        y=res.p_min_dbm,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"P_min ({res.p_min_dbm:.0f} dBm)",
        row=1,
        col=1,
    )
    fig_sig.add_trace(
        go.Scatter(
            x=t,
            y=res.hard_snr_db[:, k],
            mode="lines",
            line=dict(color="#a855f7", width=2.0),
            name="SNR Hard",
        ),
        row=1,
        col=2,
    )
    fig_sig.add_trace(
        go.Scatter(
            x=t,
            y=res.soft_snr_db[:, k],
            mode="lines",
            line=dict(color="#10b981", width=2.0),
            name="SNR Soft",
        ),
        row=1,
        col=2,
    )
    _base_layout(fig_sig, height=370)
    fig_sig.update_xaxes(title_text="Tempo [s]", row=1, col=1)
    fig_sig.update_xaxes(title_text="Tempo [s]", row=1, col=2)
    fig_sig.update_yaxes(title_text="Potência Recebida [dBm]", row=1, col=1)
    fig_sig.update_yaxes(title_text="SNR [dB]", row=1, col=2)
    return fig_dec, fig_sig

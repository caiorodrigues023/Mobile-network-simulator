# ---------------------------------------------------------------------------
# src/simulator/gui/view3d.py — Gerador de Cena 3D Interativa com WebGL (Plotly)
# ---------------------------------------------------------------------------

import json
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo


class View3DGenerator:
    """Gera o código HTML interativo da visualização 3D do mapa de rede móvel."""

    AP_COLORS = [
        "#00d2ff", "#ff007f", "#00f0a8", "#ffaa00",
        "#9d4edd", "#3a86ff", "#ff595e", "#8ac926",
        "#ffca3a", "#1982c4", "#6a4c93", "#f72585",
        "#4cc9f0", "#7209b7", "#3f37c9", "#4895ef"
    ]

    @classmethod
    def generate_html(
        cls,
        snapshot_data: dict,
        show_heatmap: bool = True,
        show_beams: bool = True,
        show_links: bool = True,
        show_towers: bool = True,
        show_ues: bool = True,
        heatmap_opacity: float = 0.75,
        colormap: str = "Turbo",
        plotly_url: str = None,
    ) -> str:
        """Monta a figura Plotly 3D completa e exporta como HTML autocontido."""
        fig = go.Figure()

        aps = snapshot_data.get("aps", [])
        ues = snapshot_data.get("ues", [])
        heatmap = snapshot_data.get("heatmap", None)
        model = snapshot_data.get("model", "smb")
        L = snapshot_data.get("L", 8)
        J = snapshot_data.get("J", 8)

        # -------------------------------------------------------------------
        # 1. Mapa de Calor 3D (Heatmap de Cobertura)
        # -------------------------------------------------------------------
        if show_heatmap and heatmap is not None:
            gx = heatmap["x"]
            gy = heatmap["y"]
            z_vals = np.array(heatmap["z"])
            label = heatmap.get("label", "SINR (dB)")

            # Cria superfície plana ligeiramente acima do piso (z=0.3m)
            # para evitar z-fighting com o chão
            z_surface = np.full_like(z_vals, 0.3)

            fig.add_trace(
                go.Surface(
                    x=gx,
                    y=gy,
                    z=z_surface,
                    surfacecolor=z_vals,
                    colorscale=colormap,
                    opacity=heatmap_opacity,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text=label, font=dict(color="#f8fafc", size=13)),
                        tickfont=dict(color="#cbd5e1", size=11),
                        len=0.7,
                        thickness=18,
                        x=1.02,
                        bgcolor="rgba(15, 23, 42, 0.6)",
                        bordercolor="#334155",
                        borderwidth=1,
                    ),
                    name="Mapa de Cobertura",
                    hovertemplate=(
                        "<b>Ponto de Cobertura</b><br>"
                        "X: %{x:.0f} m<br>"
                        "Y: %{y:.0f} m<br>"
                        + f"{label}: " + "%{surfacecolor:.1f}<extra></extra>"
                    ),
                )
            )

        # -------------------------------------------------------------------
        # 2. Torres dos APs (Base Stations)
        # -------------------------------------------------------------------
        if show_towers and aps:
            tower_x, tower_y, tower_z = [], [], []
            ap_top_x, ap_top_y, ap_top_z, ap_text, ap_hover = [], [], [], [], []
            ap_base_x, ap_base_y, ap_base_z = [], [], []

            for ap in aps:
                c_idx = ap["id"] % len(cls.AP_COLORS)
                # Mastros verticais da torre
                tower_x.extend([ap["x"], ap["x"], None])
                tower_y.extend([ap["y"], ap["y"], None])
                tower_z.extend([0, ap["z"], None])

                # Base da torre
                ap_base_x.append(ap["x"])
                ap_base_y.append(ap["y"])
                ap_base_z.append(0)

                # Topo / Antena do AP
                ap_top_x.append(ap["x"])
                ap_top_y.append(ap["y"])
                ap_top_z.append(ap["z"])
                ap_text.append(f"📡 AP {ap['id']}")
                ap_hover.append(
                    f"<b>📡 Estação Rádio-Base AP {ap['id']}</b><br>"
                    f"Posição: ({ap['x']:.1f}, {ap['y']:.1f}, {ap['z']:.1f}) m<br>"
                    f"UEs Conectados: {ap['connected_ues']}<br>"
                    f"Vazão Agregada: {ap['sum_cap_mbps']:.2f} Mbps"
                )

            # Traço dos mastros verticais
            fig.add_trace(
                go.Scatter3d(
                    x=tower_x,
                    y=tower_y,
                    z=tower_z,
                    mode="lines",
                    line=dict(color="#94a3b8", width=7),
                    name="Mastros das Torres",
                    hoverinfo="skip",
                )
            )

            # Traço das bases das torres
            fig.add_trace(
                go.Scatter3d(
                    x=ap_base_x,
                    y=ap_base_y,
                    z=ap_base_z,
                    mode="markers",
                    marker=dict(symbol="square", size=8, color="#475569", line=dict(color="#1e293b", width=1)),
                    name="Base da Torre",
                    hoverinfo="skip",
                )
            )

            # Traço dos topos das antenas com etiquetas
            fig.add_trace(
                go.Scatter3d(
                    x=ap_top_x,
                    y=ap_top_y,
                    z=ap_top_z,
                    mode="markers+text",
                    marker=dict(
                        symbol="diamond",
                        size=12,
                        color=[cls.AP_COLORS[ap["id"] % len(cls.AP_COLORS)] for ap in aps],
                        line=dict(color="#ffffff", width=2),
                    ),
                    text=ap_text,
                    textposition="top center",
                    textfont=dict(color="#38bdf8", size=12, family="Arial Black, Segoe UI"),
                    hovertext=ap_hover,
                    hoverinfo="text",
                    name="Antenas dos APs",
                )
            )

        # -------------------------------------------------------------------
        # 3. Feixes das Antenas em 3D (Radiation Beams)
        # -------------------------------------------------------------------
        if show_beams and aps:
            beam_x, beam_y, beam_z = [], [], []
            beam_colors = []
            active_pairs = {(u["ap_id"], u["beam"]) for u in ues}

            r_beam = 90.0  # Raio visual do feixe no mapa

            for ap in aps:
                c_ap = cls.AP_COLORS[ap["id"] % len(cls.AP_COLORS)]

                if model == "omni":
                    # Padrão omnidirecional: anel de radiação 3D ao redor do topo da torre
                    angles = np.linspace(0, 2 * np.pi, 60)
                    circ_x = ap["x"] + (r_beam * 0.7) * np.cos(angles)
                    circ_y = ap["y"] + (r_beam * 0.7) * np.sin(angles)
                    circ_z = np.full_like(angles, ap["z"])
                    fig.add_trace(
                        go.Scatter3d(
                            x=circ_x,
                            y=circ_y,
                            z=circ_z,
                            mode="lines",
                            line=dict(color=c_ap, width=4),
                            name=f"Radiação Omni AP {ap['id']}",
                            hoverinfo="name",
                        )
                    )

                elif model == "sector":
                    # 3 setores de 120° (boresight: 0°, 120°, 240°)
                    boresights = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
                    for b_idx, phi_0 in enumerate(boresights):
                        is_active = (ap["id"], b_idx) in active_pairs
                        angles = np.linspace(phi_0 - np.pi / 3, phi_0 + np.pi / 3, 25)
                        pts_x = [ap["x"]]
                        pts_y = [ap["y"]]
                        pts_z = [ap["z"]]
                        for a in angles:
                            pts_x.append(ap["x"] + r_beam * np.cos(a))
                            pts_y.append(ap["y"] + r_beam * np.sin(a))
                            pts_z.append(ap["z"] - 6.0)  # ligeira inclinação (downtilt)
                        pts_x.append(ap["x"])
                        pts_y.append(ap["y"])
                        pts_z.append(ap["z"])

                        fig.add_trace(
                            go.Scatter3d(
                                x=pts_x,
                                y=pts_y,
                                z=pts_z,
                                mode="lines",
                                line=dict(
                                    color=c_ap if is_active else "rgba(148, 163, 184, 0.35)",
                                    width=5 if is_active else 2,
                                    dash="solid" if is_active else "dash",
                                ),
                                name=f"Setor {b_idx} AP {ap['id']}" + (" (Ativo)" if is_active else ""),
                                hoverinfo="name",
                            )
                        )

                elif model == "smb":
                    # J feixes direcionais UCA
                    theta_k = np.arange(J) * (2 * np.pi / J)
                    for k_idx, phi_k in enumerate(theta_k):
                        is_active = (ap["id"], k_idx) in active_pairs
                        # Lóbulos estreitos apontando na direção phi_k
                        d_phi = np.linspace(-np.pi / (J * 1.5), np.pi / (J * 1.5), 15)
                        pts_x = [ap["x"]]
                        pts_y = [ap["y"]]
                        pts_z = [ap["z"]]
                        for da in d_phi:
                            ang = phi_k + da
                            gain_factor = np.cos(da * (J / 2.0)) ** 2
                            cur_r = r_beam * gain_factor
                            pts_x.append(ap["x"] + cur_r * np.cos(ang))
                            pts_y.append(ap["y"] + cur_r * np.sin(ang))
                            pts_z.append(ap["z"] - 5.0 * gain_factor)
                        pts_x.append(ap["x"])
                        pts_y.append(ap["y"])
                        pts_z.append(ap["z"])

                        fig.add_trace(
                            go.Scatter3d(
                                x=pts_x,
                                y=pts_y,
                                z=pts_z,
                                mode="lines",
                                line=dict(
                                    color=c_ap if is_active else "rgba(100, 116, 139, 0.3)",
                                    width=5 if is_active else 1.5,
                                    dash="solid" if is_active else "dot",
                                ),
                                name=f"Feixe SMB {k_idx} AP {ap['id']}" + (" (Ativo)" if is_active else ""),
                                hoverinfo="name",
                            )
                        )

                elif model.startswith("adaptive"):
                    # Feixes adaptativos MRC sintetizados diretamente na direção dos UEs conectados
                    connected_ues = [u for u in ues if u["ap_id"] == ap["id"]]
                    for u in connected_ues:
                        dx = u["x"] - ap["x"]
                        dy = u["y"] - ap["y"]
                        phi_ue = np.arctan2(dy, dx)
                        d_phi = np.linspace(-0.15, 0.15, 15)
                        pts_x = [ap["x"]]
                        pts_y = [ap["y"]]
                        pts_z = [ap["z"]]
                        beam_reach = min(r_beam * 1.4, u["dist"])
                        for da in d_phi:
                            ang = phi_ue + da
                            gain_factor = np.cos(da / 0.15 * (np.pi / 2)) ** 2
                            cur_r = beam_reach * gain_factor
                            pts_x.append(ap["x"] + cur_r * np.cos(ang))
                            pts_y.append(ap["y"] + cur_r * np.sin(ang))
                            pts_z.append(ap["z"] - 8.0 * (cur_r / beam_reach))
                        pts_x.append(ap["x"])
                        pts_y.append(ap["y"])
                        pts_z.append(ap["z"])

                        fig.add_trace(
                            go.Scatter3d(
                                x=pts_x,
                                y=pts_y,
                                z=pts_z,
                                mode="lines",
                                line=dict(color="#f43f5e", width=5),
                                name=f"Feixe MRC Adaptativo → UE {u['id']}",
                                hoverinfo="name",
                            )
                        )

        # -------------------------------------------------------------------
        # 4. Linhas de Associação UE ↔ AP (Radio Links)
        # -------------------------------------------------------------------
        if show_links and ues:
            link_x, link_y, link_z = [], [], []
            for u in ues:
                link_x.extend([u["x"], u["ap_x"], None])
                link_y.extend([u["y"], u["ap_y"], None])
                link_z.extend([u["z"], u["ap_z"], None])

            fig.add_trace(
                go.Scatter3d(
                    x=link_x,
                    y=link_y,
                    z=link_z,
                    mode="lines",
                    line=dict(color="rgba(56, 189, 248, 0.65)", width=2, dash="dash"),
                    name="Enlaces de Rádio (UE ↔ AP)",
                    hoverinfo="skip",
                )
            )

        # -------------------------------------------------------------------
        # 5. Equipamentos de Usuário (UEs)
        # -------------------------------------------------------------------
        if show_ues and ues:
            ue_x = [u["x"] for u in ues]
            ue_y = [u["y"] for u in ues]
            ue_z = [u["z"] for u in ues]
            ue_colors = [cls.AP_COLORS[u["ap_id"] % len(cls.AP_COLORS)] for u in ues]
            ue_text = [f"UE {u['id']}" for u in ues]

            ue_hover = []
            for u in ues:
                ue_hover.append(
                    f"<b>📱 Equipamento de Usuário UE {u['id']}</b><br>"
                    f"Posição: ({u['x']:.1f}, {u['y']:.1f}, {u['z']:.1f}) m<br>"
                    f"AP Servidor: 📡 AP {u['ap_id']}<br>"
                    f"Feixe / Setor: {u['beam']}<br>"
                    f"Canal Alocado: {u['channel']}<br>"
                    f"Distância ao AP: {u['dist']:.1f} m (Azimute: {u['angle_deg']:.1f}°)<br>"
                    f"Potência Tx: {u['power_w'] * 1000:.1f} mW ({u['power_dbm']:.1f} dBm)<br>"
                    f"SINR: <b>{u['sinr_db']:+.1f} dB</b><br>"
                    f"Capacidade de Shannon: <b>{u['cap_mbps']:.2f} Mbps</b><br>"
                    f"Status: {u['status']}"
                )

            fig.add_trace(
                go.Scatter3d(
                    x=ue_x,
                    y=ue_y,
                    z=ue_z,
                    mode="markers+text",
                    marker=dict(
                        symbol="circle",
                        size=9,
                        color=ue_colors,
                        line=dict(color="#ffffff", width=2),
                    ),
                    text=ue_text,
                    textposition="bottom center",
                    textfont=dict(color="#e2e8f0", size=10, family="Segoe UI"),
                    hovertext=ue_hover,
                    hoverinfo="text",
                    name="Usuários (UEs)",
                )
            )

        # -------------------------------------------------------------------
        # 6. Limite do Mapa (Caixa 1000 x 1000 m)
        # -------------------------------------------------------------------
        box_x = [0, 1000, 1000, 0, 0]
        box_y = [0, 0, 1000, 1000, 0]
        box_z = [0, 0, 0, 0, 0]
        fig.add_trace(
            go.Scatter3d(
                x=box_x,
                y=box_y,
                z=box_z,
                mode="lines",
                line=dict(color="#334155", width=3),
                name="Perímetro da Área (1000×1000 m)",
                hoverinfo="skip",
            )
        )

        # -------------------------------------------------------------------
        # 7. Layout 3D, Câmera e Estilo
        # -------------------------------------------------------------------
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0e14",
            plot_bgcolor="#0b0e14",
            margin=dict(l=0, r=0, t=30, b=0),
            scene=dict(
                xaxis=dict(
                    title=dict(text="X (metros)", font=dict(color="#38bdf8", size=12)),
                    range=[0, 1000],
                    backgroundcolor="#111622",
                    gridcolor="#1e293b",
                    showbackground=True,
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94a3b8", size=10),
                ),
                yaxis=dict(
                    title=dict(text="Y (metros)", font=dict(color="#38bdf8", size=12)),
                    range=[0, 1000],
                    backgroundcolor="#111622",
                    gridcolor="#1e293b",
                    showbackground=True,
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94a3b8", size=10),
                ),
                zaxis=dict(
                    title=dict(text="Altura Z (m)", font=dict(color="#38bdf8", size=12)),
                    range=[0, 60],
                    backgroundcolor="#0f141d",
                    gridcolor="#1e293b",
                    showbackground=True,
                    zerolinecolor="#334155",
                    tickfont=dict(color="#94a3b8", size=10),
                ),
                aspectratio=dict(x=1.0, y=1.0, z=0.35),
                camera=dict(
                    eye=dict(x=1.35, y=-1.35, z=0.95),
                    center=dict(x=0, y=0, z=-0.1),
                    up=dict(x=0, y=0, z=1),
                ),
            ),
            legend=dict(
                bgcolor="rgba(17, 22, 34, 0.75)",
                bordercolor="#232c3d",
                borderwidth=1,
                font=dict(color="#cbd5e1", size=11),
                x=0.01,
                y=0.99,
                xanchor="left",
                yanchor="top",
            ),
        )

        # Gera o HTML com plotly.js embutido offline
        html = pyo.plot(
            fig,
            include_plotlyjs="cdn",
            output_type="div",
            config={
                "displayModeBar": True,
                "scrollZoom": True,
                "responsive": True,
                "modeBarButtonsToRemove": ["toImage"],
            },
        )

        import os
        if plotly_url is None:
            assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
            plotly_src = "file:///" + os.path.join(assets_dir, "plotly.min.js").replace("\\", "/")
        else:
            plotly_src = plotly_url

        full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="{plotly_src}"></script>
    <script>
        // Fallback para CDN caso o arquivo local não carregue
        if (typeof Plotly === 'undefined') {{
            document.write('<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"><\\/script>');
        }}
    </script>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            background-color: #0b0e14;
            font-family: 'Segoe UI', sans-serif;
        }}
        .plotly-graph-div {{
            width: 100% !important;
            height: 100% !important;
        }}
        .floating-badge {{
            position: absolute;
            top: 10px;
            right: 140px;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 8px 14px;
            color: #f8fafc;
            font-size: 11px;
            z-index: 100;
            pointer-events: none;
            backdrop-filter: blur(4px);
        }}
        .floating-badge span {{
            color: #38bdf8;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="floating-badge">
        Cenário: <span>M={len(aps)} APs | K={len(ues)} UEs | Modelo: {model.upper()}</span> |
        Vazão Total: <span>{snapshot_data.get('sum_capacity', 0):.1f} Mbps</span>
    </div>
    {html}
</body>
</html>"""

        return full_html

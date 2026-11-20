# ---------------------------------------------------------------------------
# app_web.py — Interface Web Ultraleve para o Simulador de Redes Móveis
# ---------------------------------------------------------------------------
"""
Servidor local ultraleve (consome apenas ~35 MB de RAM).
Executa no navegador nativo do usuário (Edge, Chrome, Firefox) sem a necessidade
do peso do QtWebEngine / Chromium embutido.
"""

import json
import os
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.simulator.gui.worker import SnapshotCalculator
from src.simulator.gui.view3d import View3DGenerator
from src.simulator.gui.kpi_view import KPIViewGenerator
from src.simulator.gui.mobility_view import generate_mobility_html
from src.simulator.runner import simular_experimento, simular_scichal4a

PORT = 8050
ASSETS_DIR = os.path.join(ROOT_DIR, "src", "simulator", "gui", "assets")


class SimulatorWebHandler(SimpleHTTPRequestHandler):
    """Manipulador HTTP concorrente para servir a aplicação e endpoints."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        # Suprime logs de requisições de arquivos para manter o console limpo
        pass

    def end_headers(self):
        # Cabeçalhos CORS e anti-cache para chamadas de API
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            html = self._get_main_html().encode("utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        elif path == "/assets/plotly.min.js":
            js_file = os.path.join(ASSETS_DIR, "plotly.min.js")
            if os.path.exists(js_file):
                file_size = os.path.getsize(js_file)
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript")
                self.send_header("Content-Length", str(file_size))
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                with open(js_file, "rb") as f:
                    # Envia em blocos para evitar pico de memória
                    while chunk := f.read(65536):
                        self.wfile.write(chunk)
                return
            else:
                self.send_error(404, "plotly.min.js não encontrado")
                return

        elif path == "/api/initial_data":
            params = {
                "M": 4, "K": 8, "N": 1, "model": "smb", "L": 8, "J": 8,
                "use_rayleigh": False, "power_mode": "full_power",
                "heatmap_res": 45, "heatmap_metric": "sinr",
                "h_ap": 25.0, "h_ue": 1.5,
            }
            snap = SnapshotCalculator.compute_snapshot(params)
            html_3d = View3DGenerator.generate_html(snap, plotly_url="/assets/plotly.min.js")
            self.send_json_response({"snapshot": snap, "html_3d": html_3d})
            return

        self.send_error(404, "Arquivo não encontrado")

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            payload = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            payload = {}

        try:
            if path == "/api/snapshot":
                snap = SnapshotCalculator.compute_snapshot(payload)
                opacity = float(payload.get("opacity", 75)) / 100.0
                show_heatmap = bool(payload.get("show_heatmap", True))
                show_beams = bool(payload.get("show_beams", True))
                show_links = bool(payload.get("show_links", True))
                show_towers = bool(payload.get("show_towers", True))
                show_ues = bool(payload.get("show_ues", True))

                html_3d = View3DGenerator.generate_html(
                    snapshot_data=snap,
                    show_heatmap=show_heatmap,
                    show_beams=show_beams,
                    show_links=show_links,
                    show_towers=show_towers,
                    show_ues=show_ues,
                    heatmap_opacity=opacity,
                    plotly_url="/assets/plotly.min.js",
                )
                self.send_json_response({"snapshot": snap, "html_3d": html_3d})
                return

            elif path == "/api/simulate":
                # Executa simulação Monte Carlo
                M = int(payload.get("M", 4))
                N = int(payload.get("N", 1))
                K = int(payload.get("K", 4))
                sim = int(payload.get("sim", 100))
                power_ctrl = str(payload.get("power_mode", "full_power"))

                if power_ctrl in ("dpc_fixed", "dual_loop"):
                    stats = simular_scichal4a(M=M, N=N, K=K, sim=sim, n_hist=min(sim, 10))
                    res = {"type": "scichal4a", "data": stats, "params": payload}
                else:
                    raw_res = simular_experimento(
                        M=M, N=N, K=K, sim=sim,
                        allocation="ortogonal",
                        antenna_models=("omni", "sector", "smb", "adaptive"),
                        use_rayleigh=bool(payload.get("use_rayleigh", False)),
                        L=int(payload.get("L", 8)),
                        J=int(payload.get("J", 8)),
                    )
                    res = {"type": "antenna_comparison", "data": raw_res, "params": payload}

                kpi_html = KPIViewGenerator.generate_kpi_html(res, plotly_url="/assets/plotly.min.js")
                self.send_json_response({"kpi_html": kpi_html, "status": "success"})
                return

            elif path == "/api/mobility":
                v_kmh = float(payload.get("v_kmh", 120.0))
                dt = float(payload.get("dt", 0.100))
                D = float(payload.get("D", 50.0))
                L = int(payload.get("L", 16))

                html = generate_mobility_html(v_kmh, dt, D, L, plotly_url="/assets/plotly.min.js")
                self.send_json_response({"mobility_html": html})
                return

            self.send_error(404, "Endpoint não encontrado")

        except Exception as e:
            err_trace = traceback.format_exc()
            print(f"[ERRO API] Exceção em {path}:\n{err_trace}")
            self.send_json_response(
                {"error": str(e), "traceback": err_trace},
                status_code=500
            )

    def send_json_response(self, data: dict, status_code: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _get_main_html(self) -> str:
        return """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="utf-8">
    <title>Mobile Network Simulator 3D (Modo Ultraleve)</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="/assets/plotly.min.js"></script>
    <style>
        :root {
            --bg-dark: #0b0e14;
            --bg-card: #141923;
            --bg-panel: #111622;
            --border-color: #232c3d;
            --accent: #38bdf8;
            --accent-green: #10b981;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg-dark);
            color: var(--text-main);
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }
        /* Barra Lateral */
        #sidebar {
            width: 360px;
            background: var(--bg-panel);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            flex-shrink: 0;
            padding: 14px;
            gap: 12px;
        }
        .group-box {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .group-title {
            font-size: 11px;
            font-weight: bold;
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid #1e2638;
            padding-bottom: 4px;
        }
        label {
            font-size: 11px;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
        }
        select, input[type="number"], input[type="range"] {
            background: #1a2232;
            color: var(--text-main);
            border: 1px solid #2e3a4e;
            border-radius: 5px;
            padding: 6px 8px;
            font-size: 12px;
            outline: none;
            width: 100%;
        }
        select:focus, input:focus { border-color: var(--accent); }
        .checkbox-row {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: var(--text-main);
            cursor: pointer;
        }
        .btn {
            background: #1e293b;
            color: #fff;
            border: 1px solid #334155;
            padding: 9px 12px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
            text-align: center;
        }
        .btn:hover { background: #2e3a4e; }
        .btn-primary {
            background: linear-gradient(135deg, #0284c7, #0ea5e9);
            border: 1px solid #38bdf8;
            font-size: 13px;
        }
        .btn-primary:hover { opacity: 0.9; }
        .btn-accent {
            background: #10b981;
            border-color: #34d399;
        }
        .btn-accent:hover { background: #059669; }

        /* Painel Central */
        #main-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        /* Barra de Abas */
        #tabs-bar {
            display: flex;
            background: var(--bg-panel);
            border-bottom: 1px solid var(--border-color);
            padding: 6px 12px 0 12px;
            gap: 4px;
        }
        .tab-btn {
            background: transparent;
            color: var(--text-muted);
            border: 1px solid transparent;
            border-bottom: none;
            padding: 8px 16px;
            font-size: 12px;
            font-weight: bold;
            border-radius: 6px 6px 0 0;
            cursor: pointer;
        }
        .tab-btn.active {
            background: var(--bg-dark);
            color: var(--accent);
            border-color: var(--border-color);
            border-top: 2px solid var(--accent);
        }
        /* Conteúdo das Abas */
        #tab-contents {
            flex: 1;
            position: relative;
            overflow: hidden;
        }
        .tab-pane {
            display: none;
            width: 100%;
            height: 100%;
            overflow: auto;
        }
        .tab-pane.active { display: block; }
        iframe {
            width: 100%;
            height: 100%;
            border: none;
        }
        /* Barra de Status */
        #statusbar {
            background: #080a0f;
            border-top: 1px solid var(--border-color);
            padding: 4px 12px;
            font-size: 11px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .spinner {
            display: inline-block;
            width: 12px;
            height: 12px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: var(--accent);
            animation: spin 0.8s ease-in-out infinite;
            margin-right: 6px;
            display: none;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Tabela de Usuários */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        th, td {
            padding: 8px 10px;
            text-align: center;
            border-bottom: 1px solid #1e2638;
        }
        th {
            background: #141923;
            color: var(--accent);
            position: sticky;
            top: 0;
        }
        tr:hover { background: #1a2232; }
    </style>
</head>
<body>
    <!-- Barra Lateral de Configuração -->
    <div id="sidebar">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h3 style="font-size:14px; color:#38bdf8;">📡 Simulador 3D</h3>
            <span style="font-size:10px; background:#10b981; color:#fff; padding:2px 6px; border-radius:4px; font-weight:bold;">MODO LEVE</span>
        </div>

        <!-- Topologia -->
        <div class="group-box">
            <div class="group-title">📐 Topologia da Rede</div>
            <div>
                <label>Número de APs (M):</label>
                <select id="param-M" onchange="updateSnapshot()">
                    <option value="4" selected>4 APs (Grade 2×2)</option>
                    <option value="9">9 APs (Grade 3×3)</option>
                    <option value="16">16 APs (Grade 4×4)</option>
                </select>
            </div>
            <div>
                <label>Número de UEs (K):</label>
                <input type="number" id="param-K" value="8" min="1" max="64" onchange="updateSnapshot()">
            </div>
            <div>
                <label>Canais Ortogonais (N):</label>
                <input type="number" id="param-N" value="1" min="1" max="8" onchange="updateSnapshot()">
            </div>
        </div>

        <!-- Antenas e Feixes -->
        <div class="group-box">
            <div class="group-title">📡 Antenas & Feixes</div>
            <div>
                <label>Modelo de Antena:</label>
                <select id="param-model" onchange="updateSnapshot()">
                    <option value="smb" selected>SMB (Switched Multi-Beam UCA)</option>
                    <option value="sector">Setorizado (3 setores 120°)</option>
                    <option value="omni">Omnidirecional</option>
                    <option value="adaptive">Adaptativo MRC (Ideal 0°)</option>
                    <option value="adaptive_1">Adaptativo MRC (Erro 1°)</option>
                    <option value="adaptive_5">Adaptativo MRC (Erro 5°)</option>
                </select>
            </div>
            <div>
                <label>Elementos do Arranjo (L):</label>
                <select id="param-L" onchange="updateSnapshot()">
                    <option value="4">L = 4 elementos</option>
                    <option value="8" selected>L = 8 elementos</option>
                    <option value="16">L = 16 elementos</option>
                    <option value="32">L = 32 elementos</option>
                </select>
            </div>
            <div>
                <label>Feixes SMB (J):</label>
                <select id="param-J" onchange="updateSnapshot()">
                    <option value="4">J = 4 feixes</option>
                    <option value="8" selected>J = 8 feixes</option>
                    <option value="12">J = 12 feixes</option>
                    <option value="16">J = 16 feixes</option>
                </select>
            </div>
        </div>

        <!-- Propagação e Potência -->
        <div class="group-box">
            <div class="group-title">🌊 Canal & Potência</div>
            <label class="checkbox-row">
                <input type="checkbox" id="param-rayleigh" onchange="updateSnapshot()">
                <span>Fading Rayleigh</span>
            </label>
            <div>
                <label>Controle de Potência:</label>
                <select id="param-power" onchange="updateSnapshot()">
                    <option value="full_power" selected>Full Power (1W / 30 dBm)</option>
                    <option value="dpc_fixed">DPC Fixo (Meta 10 dB)</option>
                    <option value="dual_loop">Dual-Loop DPC (Mitigação Outage)</option>
                </select>
            </div>
        </div>

        <!-- Camadas do Mapa 3D -->
        <div class="group-box">
            <div class="group-title">🗺️ Mapa 3D & Heatmap</div>
            <div>
                <label>Métrica do Mapa de Calor:</label>
                <select id="param-metric" onchange="updateSnapshot()">
                    <option value="sinr" selected>SINR (dB)</option>
                    <option value="rsrp">RSRP / Potência (dBm)</option>
                    <option value="cap">Capacidade (Mbps)</option>
                    <option value="server">AP Dominante</option>
                </select>
            </div>
            <div>
                <label>Opacidade do Mapa:</label>
                <input type="range" id="param-opacity" min="10" max="100" value="75" oninput="updateSnapshot()">
            </div>
            <label class="checkbox-row">
                <input type="checkbox" id="chk-heatmap" checked onchange="updateSnapshot()">
                <span>Exibir Mapa de Cobertura</span>
            </label>
            <label class="checkbox-row">
                <input type="checkbox" id="chk-beams" checked onchange="updateSnapshot()">
                <span>Exibir Feixes das Antenas</span>
            </label>
            <label class="checkbox-row">
                <input type="checkbox" id="chk-links" checked onchange="updateSnapshot()">
                <span>Exibir Enlaces UE ↔ AP</span>
            </label>
        </div>

        <!-- Ações -->
        <div style="display:flex; flex-direction:column; gap:8px;">
            <button class="btn btn-primary" id="btn-simulate" onclick="runSimulation()">🚀 Executar Simulação Monte Carlo</button>
            <button class="btn btn-accent" onclick="updateSnapshot()">🎲 Gerar Novo Cenário 3D</button>
        </div>
    </div>

    <!-- Painel Central com Abas -->
    <div id="main-area">
        <div id="tabs-bar">
            <button class="tab-btn active" onclick="switchTab(0)">🌐 Visão 3D Interativa</button>
            <button class="tab-btn" onclick="switchTab(1)">📊 KPIs & Gráficos</button>
            <button class="tab-btn" onclick="switchTab(2)">🚗 Mobilidade & Tracking</button>
            <button class="tab-btn" onclick="switchTab(3)">📋 Tabela de Usuários</button>
        </div>

        <div id="tab-contents">
            <!-- Aba 1: 3D View -->
            <div id="tab-3d" class="tab-pane active">
                <iframe id="iframe-3d"></iframe>
            </div>

            <!-- Aba 2: KPIs -->
            <div id="tab-kpi" class="tab-pane">
                <iframe id="iframe-kpi"></iframe>
            </div>

            <!-- Aba 3: Mobilidade -->
            <div id="tab-mobility" class="tab-pane">
                <iframe id="iframe-mobility"></iframe>
            </div>

            <!-- Aba 4: Tabela -->
            <div id="tab-users" class="tab-pane" style="padding:16px;">
                <div style="margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="color:#38bdf8;">Inspeção de Usuários Conectados</h3>
                    <button class="btn" onclick="exportTableCSV()">📥 Exportar CSV</button>
                </div>
                <div style="overflow-x:auto;">
                    <table id="users-table">
                        <thead>
                            <tr>
                                <th>UE ID</th><th>Posição (X, Y)</th><th>AP Servidor</th>
                                <th>Distância</th><th>Azimute</th><th>Feixe</th><th>Canal</th>
                                <th>Potência (W)</th><th>Potência (dBm)</th><th>SINR (dB)</th>
                                <th>Capacidade</th><th>Status</th>
                            </tr>
                        </thead>
                        <tbody id="users-tbody"></tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Barra de Status -->
        <div id="statusbar">
            <div>
                <span class="spinner" id="spinner"></span>
                <span id="status-text" style="color:#38bdf8; font-weight:bold;">Pronto</span>
            </div>
            <div id="status-info" style="color:#94a3b8;">Área: 1000×1000m | Modo Ultraleve (Navegador)</div>
        </div>
    </div>

    <script>
        let currentSnapshot = null;
        let isSimulating = false;

        function getParams() {
            return {
                M: parseInt(document.getElementById('param-M').value),
                K: parseInt(document.getElementById('param-K').value),
                N: parseInt(document.getElementById('param-N').value),
                model: document.getElementById('param-model').value,
                L: parseInt(document.getElementById('param-L').value),
                J: parseInt(document.getElementById('param-J').value),
                use_rayleigh: document.getElementById('param-rayleigh').checked,
                power_mode: document.getElementById('param-power').value,
                heatmap_metric: document.getElementById('param-metric').value,
                opacity: parseInt(document.getElementById('param-opacity').value),
                show_heatmap: document.getElementById('chk-heatmap').checked,
                show_beams: document.getElementById('chk-beams').checked,
                show_links: document.getElementById('chk-links').checked,
                show_towers: true,
                show_ues: true,
                heatmap_res: 45,
                h_ap: 25.0,
                h_ue: 1.5,
                sim: 100
            };
        }

        function setStatus(text, loading = false) {
            document.getElementById('status-text').innerText = text;
            document.getElementById('spinner').style.display = loading ? 'inline-block' : 'none';
        }

        async function updateSnapshot() {
            if (isSimulating) return;
            setStatus("Calculando novo cenário 3D...", true);
            const params = getParams();
            try {
                const res = await fetch('/api/snapshot', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(params)
                });
                if (!res.ok) {
                    const errData = await res.json().catch(() => ({}));
                    throw new Error(errData.error || `HTTP ${res.status}`);
                }
                const data = await res.json();
                currentSnapshot = data.snapshot;

                // Atualiza iframe 3D
                const iframe = document.getElementById('iframe-3d');
                iframe.srcdoc = data.html_3d;

                // Atualiza tabela
                renderUsersTable(currentSnapshot.ues);

                setStatus(`Cenário 3D atualizado! M=${params.M} APs | K=${params.K} UEs`);
                document.getElementById('status-info').innerText = 
                    `SINR Médio: ${currentSnapshot.avg_sinr.toFixed(1)} dB | Vazão Total: ${currentSnapshot.sum_capacity.toFixed(1)} Mbps`;
            } catch (err) {
                console.error(err);
                setStatus("Erro ao atualizar cenário: " + err.message);
            }
        }

        async function runSimulation() {
            if (isSimulating) return;
            isSimulating = true;
            const btn = document.getElementById('btn-simulate');
            btn.disabled = true;
            btn.innerText = "⏳ Simulando...";
            setStatus("Executando Simulação Monte Carlo (calculando estatísticas)...", true);

            const params = getParams();
            try {
                const res = await fetch('/api/simulate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(params)
                });
                if (!res.ok) {
                    const errData = await res.json().catch(() => ({}));
                    throw new Error(errData.error || `HTTP ${res.status}`);
                }
                const data = await res.json();
                document.getElementById('iframe-kpi').srcdoc = data.kpi_html;
                switchTab(1); // Vai para aba de KPIs
                setStatus("Simulação Monte Carlo concluída com sucesso!");
            } catch (err) {
                console.error("Erro na simulação:", err);
                setStatus("Erro na simulação: " + err.message);
            } finally {
                isSimulating = false;
                btn.disabled = false;
                btn.innerText = "🚀 Executar Simulação Monte Carlo";
            }
        }

        async function loadMobility() {
            try {
                const res = await fetch('/api/mobility', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({v_kmh: 120, dt: 0.100, D: 50, L: 16})
                });
                const data = await res.json();
                document.getElementById('iframe-mobility').srcdoc = data.mobility_html;
            } catch (err) {
                console.error(err);
            }
        }

        function renderUsersTable(ues) {
            const tbody = document.getElementById('users-tbody');
            tbody.innerHTML = '';
            if (!ues) return;
            ues.forEach(u => {
                const tr = document.createElement('tr');
                const sColor = u.sinr_db >= 10 ? '#34d399' : (u.sinr_db >= 0 ? '#fbbf24' : '#f87171');
                tr.innerHTML = `
                    <td>UE ${u.id}</td>
                    <td>(${u.x.toFixed(0)}, ${u.y.toFixed(0)}) m</td>
                    <td>AP ${u.ap_id}</td>
                    <td>${u.dist.toFixed(1)} m</td>
                    <td>${u.angle_deg.toFixed(1)}°</td>
                    <td>${u.beam}</td>
                    <td>${u.channel}</td>
                    <td>${u.power_w.toFixed(4)}</td>
                    <td>${u.power_dbm.toFixed(1)}</td>
                    <td style="color:${sColor}; font-weight:bold;">${u.sinr_db > 0 ? '+' : ''}${u.sinr_db.toFixed(1)} dB</td>
                    <td>${u.cap_mbps.toFixed(2)} Mbps</td>
                    <td>${u.status}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        function switchTab(idx) {
            const btns = document.querySelectorAll('.tab-btn');
            const panes = document.querySelectorAll('.tab-pane');
            btns.forEach((b, i) => b.classList.toggle('active', i === idx));
            panes.forEach((p, i) => p.classList.toggle('active', i === idx));
            if (idx === 2 && !document.getElementById('iframe-mobility').srcdoc) {
                loadMobility();
            }
        }

        function exportTableCSV() {
            if (!currentSnapshot || !currentSnapshot.ues) return;
            let csv = "UE_ID,X_m,Y_m,AP_ID,Distancia_m,Azimute_deg,Feixe,Canal,Potencia_W,Potencia_dBm,SINR_dB,Capacidade_Mbps,Status\\n";
            currentSnapshot.ues.forEach(u => {
                csv += `${u.id},${u.x},${u.y},${u.ap_id},${u.dist},${u.angle_deg},${u.beam},${u.channel},${u.power_w},${u.power_dbm},${u.sinr_db},${u.cap_mbps},${u.status}\\n`;
            });
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = "usuarios_simulacao.csv";
            a.click();
        }

        // Inicialização
        window.onload = () => {
            updateSnapshot();
        };
    </script>
</body>
</html>"""


def start_server():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), SimulatorWebHandler)
    print(f"\n========================================================")
    print(f" Servidor Web Ultraleve Iniciado com Sucesso!")
    print(f" URL: http://127.0.0.1:{PORT}")
    print(f" Consumo de RAM estimado: ~35 MB (super leve)")
    print(f" Pressione Ctrl+C no terminal para encerrar o servidor.")
    print(f"========================================================\n")

    # Abre o navegador padrão automaticamente após 0.8s
    def open_browser():
        time.sleep(0.8)
        webbrowser.open(f"http://127.0.0.1:{PORT}")

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor encerrado pelo usuário.")
        server.server_close()


if __name__ == "__main__":
    start_server()

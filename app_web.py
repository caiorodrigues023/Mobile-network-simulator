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
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.simulator.gui.kpi_view import KPIViewGenerator
from src.simulator.gui.mobility_view import generate_mobility_html
from src.simulator.gui.view3d import View3DGenerator
from src.simulator.gui.worker import (
    SnapshotCalculator,
    mc_round_seed,
    run_monte_carlo_suite,
)

PORT = 8050
ASSETS_DIR = os.path.join(ROOT_DIR, "src", "simulator", "gui", "assets")

_SIM_STATE_LOCK = threading.Lock()
_SIM_STATE = {
    "running": False,
    "pct": 0,
    "msg": "Pronto",
    "started_at": 0.0,
}


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class SimulatorWebHandler(SimpleHTTPRequestHandler):
    """Manipulador HTTP concorrente para servir a aplicação e endpoints."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _build_snapshot_bundle(self, payload: dict) -> tuple[dict, str, str]:
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
        snapshot_kpi_html = KPIViewGenerator.generate_snapshot_kpi_html(
            snap, plotly_url="/assets/plotly.min.js"
        )
        return snap, html_3d, snapshot_kpi_html

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            html = (
                self._get_main_html()
                .replace("</head>", '<link rel="stylesheet" href="/assets/ux.css"></head>')
                .replace("</body>", '<script src="/assets/ux.js"></script></body>')
            ).encode("utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        elif path == "/api/progress":
            with _SIM_STATE_LOCK:
                running = bool(_SIM_STATE["running"])
                elapsed = (
                    max(0.0, time.time() - _SIM_STATE["started_at"])
                    if running
                    else 0.0
                )
                state = {
                    "running": running,
                    "pct": int(_SIM_STATE["pct"]),
                    "msg": str(_SIM_STATE["msg"]),
                    "elapsed_s": round(elapsed, 2),
                }
            self.send_json_response(state)
            return

        elif path.startswith("/assets/"):
            name = os.path.basename(path)  # evita path traversal
            f = os.path.join(ASSETS_DIR, name)
            ctype = {".js": "application/javascript", ".css": "text/css"}.get(
                os.path.splitext(name)[1]
            )
            if ctype and os.path.isfile(f):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(os.path.getsize(f)))
                self.send_header(
                    "Cache-Control",
                    "public, max-age=86400" if name.startswith("plotly") else "no-cache",
                )
                self.end_headers()
                with open(f, "rb") as fh:
                    while chunk := fh.read(65536):
                        self.wfile.write(chunk)
                return
            self.send_error(404)
            return

        elif path == "/api/initial_data":
            t0 = time.perf_counter()
            _log("▶ Carregando cenário inicial 3D...")
            snap, html_3d, snapshot_kpi_html = self._build_snapshot_bundle({})
            _log(f"✔ Cenário inicial pronto em {time.perf_counter() - t0:.2f}s")
            self.send_json_response(
                {
                    "snapshot": snap,
                    "html_3d": html_3d,
                    "snapshot_kpi_html": snapshot_kpi_html,
                }
            )
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
                t0 = time.perf_counter()
                _log(
                    f"▶ Gerando cenário 3D (M={payload.get('M', 4)}, K={payload.get('K', 8)}, "
                    f"antena={payload.get('model', 'smb')}, potência={payload.get('power_mode', 'full_power')}, "
                    f"seed={payload.get('seed', -1)})..."
                )
                snap, html_3d, snapshot_kpi_html = self._build_snapshot_bundle(payload)
                _log(
                    f"✔ Cenário 3D pronto em {time.perf_counter() - t0:.2f}s "
                    f"(SINR médio: {snap['avg_sinr']:+.1f} dB)"
                )
                self.send_json_response(
                    {
                        "snapshot": snap,
                        "html_3d": html_3d,
                        "snapshot_kpi_html": snapshot_kpi_html,
                    }
                )
                return

            elif path == "/api/snapshot_kpis":
                t0 = time.perf_counter()
                _log("▶ Gerando KPIs instantâneos do cenário 3D...")
                snap = SnapshotCalculator.compute_snapshot(payload)
                kpi_html = KPIViewGenerator.generate_snapshot_kpi_html(
                    snap, plotly_url="/assets/plotly.min.js"
                )
                _log(f"✔ KPIs instantâneos prontos em {time.perf_counter() - t0:.2f}s")
                self.send_json_response({"snapshot": snap, "kpi_html": kpi_html})
                return

            elif path == "/api/simulate":
                t0 = time.perf_counter()
                scope = payload.get("mc_scope", "both")
                sim = payload.get("sim", 100)
                model = payload.get("model", "smb")
                pwr = payload.get("power_mode", "full_power")
                _log(
                    f"🚀 Iniciando Simulação Monte Carlo: escopo={scope}, iterações={sim}, "
                    f"M={payload.get('M', 4)}, K={payload.get('K', 8)}, "
                    f"antena={model}, potência={pwr}"
                )
                with _SIM_STATE_LOCK:
                    _SIM_STATE["running"] = True
                    _SIM_STATE["pct"] = 1
                    _SIM_STATE["msg"] = "Iniciando Monte Carlo..."
                    _SIM_STATE["started_at"] = time.time()

                def _on_progress(pct: int, msg: str) -> None:
                    with _SIM_STATE_LOCK:
                        _SIM_STATE["pct"] = int(pct)
                        _SIM_STATE["msg"] = str(msg)
                    _log(f"  ↳ [{int(pct):3d}%] {msg}")

                try:
                    res = run_monte_carlo_suite(payload, progress_cb=_on_progress)
                    kpi_html = KPIViewGenerator.generate_kpi_html(
                        res, plotly_url="/assets/plotly.min.js"
                    )
                    with _SIM_STATE_LOCK:
                        _SIM_STATE["pct"] = 100
                        _SIM_STATE["msg"] = "Simulação concluída!"
                    _log(f"✔ Simulação Monte Carlo concluída em {time.perf_counter() - t0:.2f}s!")
                    self.send_json_response(
                        {
                            "kpi_html": kpi_html,
                            "mc_rounds": res.get("mc_rounds") if res else None,
                            "status": "success",
                        }
                    )
                finally:
                    with _SIM_STATE_LOCK:
                        _SIM_STATE["running"] = False
                return

            elif path == "/api/mc_round":
                t0 = time.perf_counter()
                round_num = max(1, int(payload.get("round", 1)))
                base_seed = int(payload.get("base_seed", 0))
                r_seed = mc_round_seed(base_seed, round_num - 1)
                _log(f"🔎 Carregando rodada Monte Carlo #{round_num} (seed={r_seed})...")
                snap = SnapshotCalculator.compute_snapshot(
                    {**payload, "seed": r_seed}, include_heatmap=False
                )
                _log(f"✔ Rodada #{round_num} pronta em {time.perf_counter() - t0:.2f}s")
                self.send_json_response(
                    {
                        "round": round_num,
                        "seed": r_seed,
                        "ues": snap["ues"],
                        "aps": snap["aps"],
                        "avg_sinr": snap["avg_sinr"],
                        "sum_capacity": snap["sum_capacity"],
                    }
                )
                return

            elif path == "/api/mobility":
                t0 = time.perf_counter()
                v_kmh = float(payload.get("v_kmh", 120.0))
                dt = float(payload.get("dt", 0.100))
                D = float(payload.get("D", 50.0))
                L = int(payload.get("L", 16))
                _log(f"🚗 Calculando Mobilidade (v={v_kmh} km/h, ΔT={dt * 1000:.0f} ms, D={D} m, L={L})...")
                html = generate_mobility_html(
                    v_kmh, dt, D, L, plotly_url="/assets/plotly.min.js"
                )
                _log(f"✔ Mobilidade calculada em {time.perf_counter() - t0:.2f}s")
                self.send_json_response({"mobility_html": html})
                return

            self.send_error(404, "Endpoint não encontrado")

        except Exception as e:
            err_trace = traceback.format_exc()
            _log(f"[ERRO API] Exceção em {path}:\n{err_trace}")
            self.send_json_response(
                {"error": str(e), "traceback": err_trace}, status_code=500
            )

    def send_json_response(self, data: dict, status_code: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
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
            width: 390px;
            background: var(--bg-panel);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            flex-shrink: 0;
            padding: 12px;
            gap: 10px;
        }
        details.group-box {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 8px 10px;
        }
        details.group-box summary {
            font-size: 11px;
            font-weight: bold;
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            cursor: pointer;
            list-style: none;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        details.group-box[open] summary {
            border-bottom: 1px solid #1e2638;
            padding-bottom: 6px;
            margin-bottom: 8px;
        }
        details.group-box summary::after {
            content: '▸';
            font-size: 12px;
            color: var(--text-muted);
        }
        details.group-box[open] summary::after {
            content: '▾';
        }
        .group-body {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }
        .grid-3 {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 6px;
        }
        label {
            font-size: 10.5px;
            color: var(--text-muted);
            display: block;
            margin-bottom: 3px;
        }
        select, input[type="number"], input[type="text"], input[type="range"] {
            background: #1a2232;
            color: var(--text-main);
            border: 1px solid #2e3a4e;
            border-radius: 5px;
            padding: 5px 7px;
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
            margin: 0;
        }
        .checkbox-row input { width: auto; }
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
        #tab-mobility.active {
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .mobility-toolbar {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 12px;
            padding: 10px 14px;
            background: var(--bg-panel);
            border-bottom: 1px solid var(--border-color);
            flex-shrink: 0;
        }
        .mobility-field {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .mobility-field label { margin: 0; }
        .mobility-field select { width: auto; }
        iframe {
            width: 100%;
            height: 100%;
            border: none;
        }
        #iframe-mobility {
            flex: 1;
            height: auto;
            min-height: 0;
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
    <!-- Barra Lateral de Configuração Completa -->
    <div id="sidebar">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h3 style="font-size:14px; color:#38bdf8;">📡 Simulador 3D</h3>
            <span style="font-size:10px; background:#10b981; color:#fff; padding:2px 6px; border-radius:4px; font-weight:bold;">MODO LEVE</span>
        </div>

        <!-- Topologia & Alocação -->
        <details class="group-box" open>
            <summary>📐 Topologia & Alocação</summary>
            <div class="group-body">
                <div class="grid-2">
                    <div>
                        <label for="param-M">APs (M):</label>
                        <select id="param-M" onchange="updateSnapshot()">
                            <option value="4" selected>4 APs (2×2)</option>
                            <option value="9">9 APs (3×3)</option>
                            <option value="16">16 APs (4×4)</option>
                            <option value="25">25 APs (5×5)</option>
                            <option value="36">36 APs (6×6)</option>
                        </select>
                    </div>
                    <div>
                        <label for="param-K">Usuários (K):</label>
                        <input type="number" id="param-K" value="8" min="1" max="128" onchange="updateSnapshot()">
                    </div>
                </div>
                <div class="grid-2">
                    <div>
                        <label for="param-N">Canais (N):</label>
                        <input type="number" id="param-N" value="1" min="1" max="16" onchange="updateSnapshot()">
                    </div>
                    <div>
                        <label for="param-allocation">Alocação:</label>
                        <select id="param-allocation" onchange="updateSnapshot()">
                            <option value="ortogonal" selected>Ortogonal intra-AP</option>
                            <option value="random">Aleatória (Random)</option>
                        </select>
                    </div>
                </div>
                <div class="grid-3">
                    <div>
                        <label for="param-h-ap">Alt. AP (m):</label>
                        <input type="number" id="param-h-ap" value="25" min="5" max="100" step="1" onchange="updateSnapshot()">
                    </div>
                    <div>
                        <label for="param-h-ue">Alt. UE (m):</label>
                        <input type="number" id="param-h-ue" value="1.5" min="0.5" max="20" step="0.5" onchange="updateSnapshot()">
                    </div>
                    <div>
                        <label for="param-seed">Seed (-1=rnd):</label>
                        <input type="number" id="param-seed" value="-1" min="-1" max="2147483647" step="1" onchange="updateSnapshot()">
                    </div>
                </div>
            </div>
        </details>

        <!-- Antenas e Feixes -->
        <details class="group-box" open>
            <summary>📡 Antenas & Beamforming</summary>
            <div class="group-body">
                <div>
                    <label for="param-model">Modelo de Antena:</label>
                    <select id="param-model" onchange="updateSnapshot()">
                        <option value="smb" selected>SMB (Switched Multi-Beam UCA)</option>
                        <option value="sector">Setorizado (3 setores)</option>
                        <option value="omni">Omnidirecional</option>
                        <option value="adaptive">Adaptativo MRC (Ideal 0°)</option>
                        <option value="adaptive_1">Adaptativo MRC (Erro 1°)</option>
                        <option value="adaptive_5">Adaptativo MRC (Erro 5°)</option>
                        <option value="adaptive_10">Adaptativo MRC (Erro 10°)</option>
                    </select>
                </div>
                <div class="grid-2">
                    <div>
                        <label for="param-L">Elementos (L):</label>
                        <select id="param-L" onchange="updateSnapshot()">
                            <option value="2">L = 2</option>
                            <option value="4">L = 4</option>
                            <option value="8" selected>L = 8</option>
                            <option value="16">L = 16</option>
                            <option value="32">L = 32</option>
                            <option value="64">L = 64</option>
                        </select>
                    </div>
                    <div>
                        <label for="param-J">Feixes SMB (J):</label>
                        <select id="param-J" onchange="updateSnapshot()">
                            <option value="4">J = 4</option>
                            <option value="8" selected>J = 8</option>
                            <option value="12">J = 12</option>
                            <option value="16">J = 16</option>
                            <option value="24">J = 24</option>
                            <option value="32">J = 32</option>
                        </select>
                    </div>
                </div>
                <div class="grid-3">
                    <div>
                        <label for="param-aoa-err">Erro AoA (°):</label>
                        <input type="number" id="param-aoa-err" value="0" min="0" max="30" step="0.5" onchange="updateSnapshot()">
                    </div>
                    <div>
                        <label for="param-beamwidth">Abertura (°):</label>
                        <input type="number" id="param-beamwidth" value="120" min="30" max="360" step="10" onchange="updateSnapshot()">
                    </div>
                    <div>
                        <label for="param-g-min">Lób. g_min:</label>
                        <input type="number" id="param-g-min" value="0.01" min="0.001" max="0.5" step="0.001" onchange="updateSnapshot()">
                    </div>
                </div>
            </div>
        </details>

        <!-- Canal & Propagação -->
        <details class="group-box">
            <summary>🌊 Canal de Propagação & Ruído</summary>
            <div class="group-body">
                <div class="grid-2">
                    <div>
                        <label for="param-bw">Banda Total (MHz):</label>
                        <input type="number" id="param-bw" value="100" min="1" max="1000" step="1" onchange="updateSnapshot()">
                    </div>
                    <div>
                        <label for="param-pathloss">Expoente Pathloss (n):</label>
                        <input type="number" id="param-pathloss" value="4.0" min="2.0" max="6.0" step="0.1" onchange="updateSnapshot()">
                    </div>
                </div>
                <div class="grid-3">
                    <div>
                        <label for="param-k-const">Constante K:</label>
                        <input type="text" id="param-k-const" value="1e-4" onchange="updateSnapshot()">
                    </div>
                    <div>
                        <label for="param-d0">d₀ mín (m):</label>
                        <input type="number" id="param-d0" value="1.0" min="0.5" max="10" step="0.5" onchange="updateSnapshot()">
                    </div>
                    <div>
                        <label for="param-k0">Ruído K₀ (W/Hz):</label>
                        <input type="text" id="param-k0" value="1e-20" onchange="updateSnapshot()">
                    </div>
                </div>
                <div class="grid-2">
                    <div>
                        <label for="param-sigma-x">Shadowing σ_X:</label>
                        <input type="number" id="param-sigma-x" value="2.0" min="0" max="10" step="0.25" onchange="updateSnapshot()">
                    </div>
                    <div>
                        <label for="param-sigma-r">Rayleigh σ_R:</label>
                        <input type="number" id="param-sigma-r" value="0.7071" min="0.05" max="2.0" step="any" onchange="updateSnapshot()">
                    </div>
                </div>
                <label class="checkbox-row">
                    <input type="checkbox" id="param-rayleigh" onchange="updateSnapshot()">
                    <span>Habilitar Fast Fading (Rayleigh)</span>
                </label>
            </div>
        </details>

        <!-- Controle de Potência (DPC & Dual-Loop) -->
        <details class="group-box" open>
            <summary>⚡ Controle de Potência (DPC & Dual-Loop)</summary>
            <div class="group-body">
                <div>
                    <label for="param-power">Algoritmo de Potência:</label>
                    <select id="param-power" onchange="updateSnapshot()">
                        <option value="full_power" selected>Full Power (P_max — normal)</option>
                        <option value="dpc_fixed">DPC Fixo (Meta SINR — normal)</option>
                        <option value="dual_loop">Dual-Loop DPC (Desliga Outage)</option>
                    </select>
                </div>
                <div class="grid-2">
                    <div>
                        <label for="param-p-max">P_max (dBm):</label>
                        <input type="number" id="param-p-max" value="30" min="0" max="50" step="1" onchange="updateSnapshot()">
                    </div>
                    <div>
                        <label for="param-p-min">P_min (dBm):</label>
                        <input type="number" id="param-p-min" value="0" min="-40" max="20" step="1" onchange="updateSnapshot()">
                    </div>
                </div>
                <div class="grid-3">
                    <div>
                        <label for="param-gamma-tar">γ_tar (dB):</label>
                        <input type="number" id="param-gamma-tar" value="10" min="-5" max="30" step="1" onchange="updateSnapshot()">
                    </div>
                    <div>
                        <label for="param-gamma-min">γ_min (dB):</label>
                        <input type="number" id="param-gamma-min" value="-10" min="-30" max="5" step="1" onchange="updateSnapshot()">
                    </div>
                    <div>
                        <label for="param-gamma-max">γ_max (dB):</label>
                        <input type="number" id="param-gamma-max" value="20" min="5" max="40" step="1" onchange="updateSnapshot()">
                    </div>
                </div>
                <div class="grid-2">
                    <div>
                        <label for="param-mu-step">Passo μ (dB):</label>
                        <input type="number" id="param-mu-step" value="1.0" min="0.1" max="5.0" step="0.1" onchange="updateSnapshot()">
                    </div>
                    <div>
                        <label for="param-tol">Tol. Loop Ext (dB):</label>
                        <input type="number" id="param-tol" value="0.5" min="0.1" max="3.0" step="0.1" onchange="updateSnapshot()">
                    </div>
                </div>
                <div class="grid-2">
                    <div>
                        <label for="param-t-outer">Iter. Loop Externo:</label>
                        <input type="number" id="param-t-outer" value="50" min="1" max="500" step="1" onchange="updateSnapshot()">
                    </div>
                    <div>
                        <label for="param-t-inner">Iter. Loop Interno:</label>
                        <input type="number" id="param-t-inner" value="150" min="10" max="5000" step="10" onchange="updateSnapshot()">
                    </div>
                </div>
            </div>
        </details>

        <!-- Camadas do Mapa 3D -->
        <details class="group-box">
            <summary>🗺️ Camadas do Mapa 3D & Heatmap</summary>
            <div class="group-body">
                <div class="grid-2">
                    <div>
                        <label for="param-metric">Métrica Heatmap:</label>
                        <select id="param-metric" onchange="updateSnapshot()">
                            <option value="sinr" selected>SINR (dB)</option>
                            <option value="rsrp">RSRP (dBm)</option>
                            <option value="cap">Capacidade (Mbps)</option>
                            <option value="server">AP Dominante</option>
                        </select>
                    </div>
                    <div>
                        <label for="param-heatmap-res">Resolução Grade:</label>
                        <select id="param-heatmap-res" onchange="updateSnapshot()">
                            <option value="25">25×25 (Rápida)</option>
                            <option value="35">35×35</option>
                            <option value="45" selected>45×45 (Normal)</option>
                            <option value="55">55×55</option>
                            <option value="75">75×75 (Alta)</option>
                        </select>
                    </div>
                </div>
                <div>
                    <label for="param-opacity">Opacidade do Mapa:</label>
                    <input type="range" id="param-opacity" min="10" max="100" value="75" onchange="updateSnapshot()">
                </div>
                <label class="checkbox-row"><input type="checkbox" id="chk-heatmap" checked onchange="updateSnapshot()"><span>Exibir Mapa de Cobertura</span></label>
                <label class="checkbox-row"><input type="checkbox" id="chk-beams" checked onchange="updateSnapshot()"><span>Exibir Feixes das Antenas</span></label>
                <label class="checkbox-row"><input type="checkbox" id="chk-links" checked onchange="updateSnapshot()"><span>Exibir Enlaces UE ↔ AP</span></label>
                <label class="checkbox-row"><input type="checkbox" id="chk-towers" checked onchange="updateSnapshot()"><span>Exibir Torres dos APs</span></label>
                <label class="checkbox-row"><input type="checkbox" id="chk-ues" checked onchange="updateSnapshot()"><span>Exibir Usuários (UEs)</span></label>
            </div>
        </details>

        <!-- Experimento Monte Carlo -->
        <details class="group-box" open>
            <summary>🎲 Experimento Monte Carlo</summary>
            <div class="group-body">
                <div class="grid-2">
                    <div>
                        <label for="param-mc-scope">Escopo KPIs:</label>
                        <select id="param-mc-scope">
                            <option value="both" selected>Completo (Antenas + Potência)</option>
                            <option value="antennas">Apenas Antenas</option>
                            <option value="power">Apenas Controle de Potência</option>
                        </select>
                    </div>
                    <div>
                        <label for="param-sim">Iterações:</label>
                        <input type="number" id="param-sim" value="100" min="5" max="5000" step="1">
                    </div>
                </div>
            </div>
        </details>

        <!-- Ações -->
        <div style="display:flex; flex-direction:column; gap:7px; padding-bottom:8px;">
            <button class="btn btn-primary" id="btn-simulate" onclick="runSimulation()">🚀 Executar Simulação Monte Carlo</button>
            <button class="btn btn-accent" onclick="updateSnapshot()">🎲 Gerar Novo Cenário 3D</button>
            <button class="btn" onclick="showSnapshotKpis()">📊 Ver KPIs deste Cenário 3D</button>
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
                <div class="mobility-toolbar">
                    <div class="mobility-field">
                        <label for="mob-v">Velocidade:</label>
                        <select id="mob-v" onchange="loadMobility()">
                            <option value="3">Pedestrian (3 km/h)</option>
                            <option value="50">Urban (50 km/h)</option>
                            <option value="120" selected>Highway (120 km/h)</option>
                        </select>
                    </div>
                    <div class="mobility-field">
                        <label for="mob-dt">Atualização (ΔT):</label>
                        <select id="mob-dt" onchange="loadMobility()">
                            <option value="0.010">10 ms (Rápido / High)</option>
                            <option value="0.050">50 ms (Médio / Fair)</option>
                            <option value="0.100" selected>100 ms (Lento / Slow)</option>
                        </select>
                    </div>
                    <div class="mobility-field">
                        <label for="mob-D">Distância (D):</label>
                        <select id="mob-D" onchange="loadMobility()">
                            <option value="25">25 m</option>
                            <option value="50" selected>50 m</option>
                            <option value="75">75 m</option>
                            <option value="150">150 m</option>
                        </select>
                    </div>
                    <div class="mobility-field">
                        <label for="mob-L">Elementos ULA (L):</label>
                        <select id="mob-L" onchange="loadMobility()">
                            <option value="8">L = 8</option>
                            <option value="16" selected>L = 16</option>
                            <option value="32">L = 32</option>
                        </select>
                    </div>
                    <button class="btn btn-accent" onclick="loadMobility()">🔄 Atualizar</button>
                </div>
                <iframe id="iframe-mobility"></iframe>
            </div>

            <!-- Aba 4: Tabela -->
            <div id="tab-users" class="tab-pane" style="padding:16px;">
                <div style="margin-bottom:12px; display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap;">
                    <div>
                        <h3 style="color:#38bdf8;">Inspeção de Usuários Conectados</h3>
                        <span id="users-summary" style="font-size:11px; color:#94a3b8;">Aguardando cenário...</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                        <select id="mc-round-select" onchange="onMcRoundSelectChange()" style="width:auto; min-width:225px;">
                            <option value="snapshot">📍 Cenário 3D Atual (Snapshot)</option>
                        </select>
                        <div id="mc-round-nav" style="display:none; align-items:center; gap:4px;">
                            <button class="btn" style="padding:5px 9px;" onclick="stepMcRound(-1)" title="Rodada anterior">◀</button>
                            <input type="number" id="mc-round-num" min="1" value="1" onchange="loadSpecificMcRound(this.value)" style="width:74px; text-align:center;" title="Número da rodada">
                            <button class="btn" style="padding:5px 9px;" onclick="stepMcRound(1)" title="Próxima rodada">▶</button>
                            <button class="btn btn-accent" style="padding:5px 10px;" onclick="viewInspectedRoundIn3D()" title="Carregar esta rodada na Visão 3D">🌐 Ver esta Rodada no 3D</button>
                        </div>
                        <button class="btn" onclick="exportTableCSV()">📥 Exportar CSV</button>
                    </div>
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
        let currentSnapshotKpiHtml = "";
        let hasMonteCarloKpis = false;
        let isSimulating = false;
        let currentMcRounds = null;
        let currentMcParams = null;
        let currentInspectedSeed = null;
        let currentTableSourceLabel = "📍 Cenário 3D Atual";
        let displayedUsers = [];

        function numOr(id, fallback) {
            const v = parseFloat(document.getElementById(id).value);
            return Number.isFinite(v) ? v : fallback;
        }

        function intOr(id, fallback) {
            const v = parseInt(document.getElementById(id).value);
            return Number.isFinite(v) ? v : fallback;
        }

        function getParams() {
            return {
                M: intOr('param-M', 4),
                K: intOr('param-K', 8),
                N: intOr('param-N', 1),
                allocation: document.getElementById('param-allocation').value,
                h_ap: numOr('param-h-ap', 25.0),
                h_ue: numOr('param-h-ue', 1.5),
                seed: intOr('param-seed', -1),
                model: document.getElementById('param-model').value,
                L: intOr('param-L', 8),
                J: intOr('param-J', 8),
                aoa_err_deg: numOr('param-aoa-err', 0.0),
                beamwidth_deg: numOr('param-beamwidth', 120.0),
                g_min: numOr('param-g-min', 0.01),
                b_total_mhz: numOr('param-bw', 100.0),
                pathloss_exp: numOr('param-pathloss', 4.0),
                k_const: numOr('param-k-const', 1e-4),
                d_0: numOr('param-d0', 1.0),
                k_0: numOr('param-k0', 1e-20),
                sigma_x: numOr('param-sigma-x', 2.0),
                sigma_r: numOr('param-sigma-r', 0.7071),
                use_rayleigh: document.getElementById('param-rayleigh').checked,
                power_mode: document.getElementById('param-power').value,
                p_max_dbm: numOr('param-p-max', 30.0),
                p_min_dbm: numOr('param-p-min', 0.0),
                gamma_tar_db: numOr('param-gamma-tar', 10.0),
                gamma_min_db: numOr('param-gamma-min', -10.0),
                gamma_max_db: numOr('param-gamma-max', 20.0),
                mu_step_db: numOr('param-mu-step', 1.0),
                tol_db: numOr('param-tol', 0.5),
                t_outer: intOr('param-t-outer', 50),
                t_inner: intOr('param-t-inner', 150),
                heatmap_metric: document.getElementById('param-metric').value,
                heatmap_res: intOr('param-heatmap-res', 45),
                opacity: intOr('param-opacity', 75),
                show_heatmap: document.getElementById('chk-heatmap').checked,
                show_beams: document.getElementById('chk-beams').checked,
                show_links: document.getElementById('chk-links').checked,
                show_towers: document.getElementById('chk-towers').checked,
                show_ues: document.getElementById('chk-ues').checked,
                mc_scope: document.getElementById('param-mc-scope').value,
                sim: Math.max(5, Math.min(5000, intOr('param-sim', 100)))
            };
        }

        function formatFetchError(err) {
            const msg = (err && err.message) ? err.message : String(err || 'Erro desconhecido');
            if (/failed to fetch|networkerror|load failed/i.test(msg)) {
                return "Servidor local offline (http://127.0.0.1:8050). Verifique se o terminal do run_web.bat está em execução.";
            }
            return msg;
        }

        function setStatus(text, loading = false) {
            document.getElementById('status-text').innerText = text;
            document.getElementById('spinner').style.display = loading ? 'inline-block' : 'none';
        }

        async function updateSnapshot() {
            if (isSimulating) return;
            setStatus("Calculando novo cenário 3D...", true);
            try {
                const params = getParams();
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
                currentSnapshotKpiHtml = data.snapshot_kpi_html || "";

                document.getElementById('iframe-3d').srcdoc = data.html_3d;
                if (!hasMonteCarloKpis && currentSnapshotKpiHtml) {
                    document.getElementById('iframe-kpi').srcdoc = currentSnapshotKpiHtml;
                }

                const roundSel = document.getElementById('mc-round-select');
                if (roundSel) roundSel.value = 'snapshot';
                currentTableSourceLabel = "📍 Cenário 3D Atual";
                currentInspectedSeed = (currentSnapshot.params && currentSnapshot.params.seed >= 0) ? currentSnapshot.params.seed : null;
                renderUsersTable(currentSnapshot.ues);

                setStatus(`Cenário 3D atualizado! M=${params.M} APs | K=${params.K} UEs`);
                document.getElementById('status-info').innerText =
                    `SINR Médio: ${currentSnapshot.avg_sinr.toFixed(1)} dB | Vazão Total: ${currentSnapshot.sum_capacity.toFixed(1)} Mbps`;
            } catch (err) {
                console.error(err);
                setStatus("Erro ao atualizar cenário: " + formatFetchError(err));
            }
        }

        function showSnapshotKpis() {
            if (!currentSnapshotKpiHtml) return;
            hasMonteCarloKpis = false;
            document.getElementById('iframe-kpi').srcdoc = currentSnapshotKpiHtml;
            switchTab(1);
            setStatus("Exibindo KPIs instantâneos do cenário 3D atual.");
        }

        async function runSimulation() {
            if (isSimulating) return;
            isSimulating = true;
            const btn = document.getElementById('btn-simulate');
            const loadingSpan = document.querySelector('#ux-loading span');
            btn.disabled = true;
            btn.innerText = "⏳ Simulando...";
            setStatus("Executando Simulação Monte Carlo (calculando estatísticas)...", true);

            const pollTimer = setInterval(async () => {
                if (!isSimulating) return;
                try {
                    const pr = await fetch('/api/progress', { cache: 'no-store' });
                    if (!pr.ok || !isSimulating) return;
                    const p = await pr.json();
                    if (!isSimulating || !p || !p.running) return;
                    const secText = Number.isFinite(p.elapsed_s) ? `${p.elapsed_s.toFixed(1)}s` : '0.0s';
                    btn.innerText = `⏳ Simulando... ${p.pct}%`;
                    setStatus(`${p.msg} (${secText})`, true);
                    if (loadingSpan) {
                        loadingSpan.innerText = `⏳ ${p.msg} (${secText})`;
                    }
                } catch (_) {}
            }, 400);

            try {
                const params = getParams();
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
                isSimulating = false;
                clearInterval(pollTimer);

                hasMonteCarloKpis = true;
                document.getElementById('iframe-kpi').srcdoc = data.kpi_html;
                if (data.mc_rounds) {
                    currentMcRounds = data.mc_rounds;
                    currentMcParams = params;
                    populateMcRoundsUI(data.mc_rounds);
                }
                switchTab(1);
                setStatus("Simulação Monte Carlo concluída com sucesso!", false);
            } catch (err) {
                isSimulating = false;
                clearInterval(pollTimer);
                console.error("Erro na simulação:", err);
                setStatus("Erro na simulação: " + formatFetchError(err), false);
            } finally {
                clearInterval(pollTimer);
                isSimulating = false;
                btn.disabled = false;
                btn.innerText = "🚀 Executar Simulação Monte Carlo";
                if (loadingSpan) {
                    loadingSpan.innerText = "⏳ Processando…";
                }
            }
        }

        async function loadMobility() {
            setStatus("Calculando análise de mobilidade e beam tracking...", true);
            const payload = {
                v_kmh: parseFloat(document.getElementById('mob-v').value),
                dt: parseFloat(document.getElementById('mob-dt').value),
                D: parseFloat(document.getElementById('mob-D').value),
                L: parseInt(document.getElementById('mob-L').value)
            };
            try {
                const res = await fetch('/api/mobility', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                if (!res.ok) {
                    const errData = await res.json().catch(() => ({}));
                    throw new Error(errData.error || `HTTP ${res.status}`);
                }
                const data = await res.json();
                document.getElementById('iframe-mobility').srcdoc = data.mobility_html;
                setStatus(`Mobilidade atualizada: v=${payload.v_kmh} km/h | ΔT=${payload.dt * 1000} ms | D=${payload.D} m | L=${payload.L}`);
            } catch (err) {
                console.error(err);
                setStatus("Erro ao carregar mobilidade: " + formatFetchError(err));
            }
        }

        function populateMcRoundsUI(mc) {
            const sel = document.getElementById('mc-round-select');
            const nav = document.getElementById('mc-round-nav');
            const numInput = document.getElementById('mc-round-num');
            if (!sel || !mc || !mc.presets) return;

            const p = mc.presets;
            sel.innerHTML = `
                <option value="snapshot">📍 Cenário 3D Atual (Snapshot)</option>
                <option value="first">1ª Rodada (#${p.first.round}) — ${p.first.sum_capacity.toFixed(0)} Mbps</option>
                <option value="last">Última Rodada (#${p.last.round}) — ${p.last.sum_capacity.toFixed(0)} Mbps</option>
                <option value="best">🏆 Melhor Rodada (#${p.best.round}) — ${p.best.sum_capacity.toFixed(0)} Mbps</option>
                <option value="median">⚖️ Rodada Mediana (#${p.median.round}) — ${p.median.sum_capacity.toFixed(0)} Mbps</option>
                <option value="worst">⚠️ Pior Rodada (#${p.worst.round}) — ${p.worst.sum_capacity.toFixed(0)} Mbps</option>
                <option value="custom">🔎 Rodada específica (#1..#${mc.sim})</option>
            `;
            numInput.min = 1;
            numInput.max = mc.sim;
            nav.style.display = 'flex';
            sel.value = 'last';
            onMcRoundSelectChange();
        }

        function onMcRoundSelectChange() {
            const sel = document.getElementById('mc-round-select');
            if (!sel) return;
            const val = sel.value;

            if (val === 'snapshot') {
                if (currentSnapshot) {
                    currentTableSourceLabel = "📍 Cenário 3D Atual";
                    currentInspectedSeed = (currentSnapshot.params && currentSnapshot.params.seed >= 0) ? currentSnapshot.params.seed : null;
                    renderUsersTable(currentSnapshot.ues);
                }
                return;
            }
            if (val === 'custom') {
                loadSpecificMcRound(document.getElementById('mc-round-num').value);
                return;
            }
            if (currentMcRounds && currentMcRounds.presets && currentMcRounds.presets[val]) {
                const r = currentMcRounds.presets[val];
                const names = {
                    first: '1ª Rodada',
                    last: 'Última Rodada',
                    best: '🏆 Melhor Rodada',
                    median: '⚖️ Rodada Mediana',
                    worst: '⚠️ Pior Rodada'
                };
                currentTableSourceLabel = `${names[val] || val} (#${r.round}/${currentMcRounds.sim})`;
                currentInspectedSeed = r.seed;
                document.getElementById('mc-round-num').value = r.round;
                renderUsersTable(r.ues);
            }
        }

        function stepMcRound(delta) {
            if (!currentMcRounds) return;
            const cur = intOr('mc-round-num', 1);
            const next = Math.max(1, Math.min(currentMcRounds.sim, cur + delta));
            loadSpecificMcRound(next);
        }

        async function loadSpecificMcRound(roundVal) {
            if (!currentMcRounds) return;
            const roundNum = Math.max(1, Math.min(currentMcRounds.sim, parseInt(roundVal) || 1));
            document.getElementById('mc-round-num').value = roundNum;
            document.getElementById('mc-round-select').value = 'custom';
            setStatus(`Carregando rodada #${roundNum}/${currentMcRounds.sim}...`, true);
            try {
                const baseParams = currentMcParams || getParams();
                const res = await fetch('/api/mc_round', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ ...baseParams, round: roundNum, base_seed: currentMcRounds.base_seed })
                });
                if (!res.ok) {
                    const errData = await res.json().catch(() => ({}));
                    throw new Error(errData.error || `HTTP ${res.status}`);
                }
                const data = await res.json();
                currentTableSourceLabel = `🔎 Rodada #${data.round}/${currentMcRounds.sim}`;
                currentInspectedSeed = data.seed;
                renderUsersTable(data.ues);
                setStatus(`Rodada #${data.round} exibida na Tabela de Usuários.`);
            } catch (err) {
                console.error(err);
                setStatus("Erro ao carregar rodada: " + formatFetchError(err));
            }
        }

        function viewInspectedRoundIn3D() {
            if (currentInspectedSeed === null || !Number.isFinite(currentInspectedSeed)) return;
            const seedEl = document.getElementById('param-seed');
            if (seedEl) seedEl.value = currentInspectedSeed;
            updateSnapshot();
            switchTab(0);
        }

        function renderUsersTable(ues) {
            displayedUsers = ues || [];
            const tbody = document.getElementById('users-tbody');
            tbody.innerHTML = '';
            if (!ues) return;

            let activeCnt = 0;
            let offCnt = 0;
            let sumCap = 0;
            let activeSinrSum = 0;

            ues.forEach(u => {
                const tr = document.createElement('tr');
                const isOff = (u.active === false) || (u.power_w <= 0);
                if (isOff) {
                    offCnt++;
                } else {
                    activeCnt++;
                    sumCap += u.cap_mbps;
                    activeSinrSum += u.sinr_db;
                }
                const sColor = isOff ? '#ef4444' : (u.sinr_db >= 10 ? '#34d399' : (u.sinr_db >= 0 ? '#fbbf24' : '#f87171'));
                const pDbmText = isOff ? 'Desligado' : u.power_dbm.toFixed(1);
                const sinrText = isOff ? 'Desligado (&lt; limiar)' : `${u.sinr_db > 0 ? '+' : ''}${u.sinr_db.toFixed(1)} dB`;
                const capText = isOff ? '0.00 Mbps' : `${u.cap_mbps.toFixed(2)} Mbps`;
                const statusText = isOff ? 'Outage (Desligado)' : u.status;
                const statusStyle = (isOff || (u.status && u.status.startsWith('Outage'))) ? 'color:#ef4444; font-weight:bold;' : '';
                tr.innerHTML = `
                    <td>UE ${u.id}</td>
                    <td>(${u.x.toFixed(0)}, ${u.y.toFixed(0)}) m</td>
                    <td>AP ${u.ap_id}</td>
                    <td>${u.dist.toFixed(1)} m</td>
                    <td>${u.angle_deg.toFixed(1)}°</td>
                    <td>${u.beam}</td>
                    <td>${u.channel}</td>
                    <td>${u.power_w.toFixed(4)}</td>
                    <td>${pDbmText}</td>
                    <td style="color:${sColor}; font-weight:bold;">${sinrText}</td>
                    <td>${capText}</td>
                    <td style="${statusStyle}">${statusText}</td>
                `;
                tbody.appendChild(tr);
            });

            const avgSinr = activeCnt > 0 ? (activeSinrSum / activeCnt) : 0;
            document.getElementById('users-summary').innerText =
                `[${currentTableSourceLabel}] | Total: ${ues.length} UEs (${activeCnt} ativos, ${offCnt} desligados) | SINR Médio (ativos): ${avgSinr >= 0 ? '+' : ''}${avgSinr.toFixed(1)} dB | Vazão: ${sumCap.toFixed(1)} Mbps`;
        }

        function switchTab(idx) {
            const btns = document.querySelectorAll('.tab-btn');
            const panes = document.querySelectorAll('.tab-pane');
            btns.forEach((b, i) => b.classList.toggle('active', i === idx));
            panes.forEach((p, i) => p.classList.toggle('active', i === idx));
            if (idx === 2 && !document.getElementById('iframe-mobility').srcdoc) {
                loadMobility();
            }
            if (idx === 1) {
                const kpiIframe = document.getElementById('iframe-kpi');
                if (kpiIframe && kpiIframe.contentWindow) {
                    try { kpiIframe.contentWindow.dispatchEvent(new Event('resize')); } catch (e) {}
                }
            }
        }

        function exportTableCSV() {
            const rows = (displayedUsers && displayedUsers.length) ? displayedUsers : (currentSnapshot ? currentSnapshot.ues : null);
            if (!rows) return;
            let csv = "UE_ID,X_m,Y_m,AP_ID,Distancia_m,Azimute_deg,Feixe,Canal,Potencia_W,Potencia_dBm,SINR_dB,Capacidade_Mbps,Status\\n";
            rows.forEach(u => {
                csv += `${u.id},${u.x},${u.y},${u.ap_id},${u.dist},${u.angle_deg},${u.beam},${u.channel},${u.power_w},${u.power_dbm},${u.sinr_db},${u.cap_mbps},${u.status}\\n`;
            });
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = "usuarios_simulacao.csv";
            a.click();
        }

        window.onload = () => {
            updateSnapshot();
        };
    </script>
</body>
</html>"""


class _LocalWebServer(ThreadingHTTPServer):
    allow_reuse_address = False


def start_server():
    try:
        server = _LocalWebServer(("127.0.0.1", PORT), SimulatorWebHandler)
    except OSError as exc:
        print("\n========================================================")
        print(f" [AVISO] A porta {PORT} já está em uso por outro terminal!")
        print(f" Detalhe: {exc}")
        print(f" Abrindo http://127.0.0.1:{PORT} no navegador...")
        print(" Se você atualizou o código, feche o terminal antigo antes")
        print(" de abrir o run_web.bat novamente.")
        print("========================================================\n", flush=True)
        webbrowser.open(f"http://127.0.0.1:{PORT}")
        return

    print("\n========================================================")
    print(" Servidor Web Ultraleve Iniciado com Sucesso!")
    print(f" URL: http://127.0.0.1:{PORT}")
    print(" Consumo de RAM estimado: ~35 MB (super leve)")
    print(" Pressione Ctrl+C no terminal para encerrar o servidor.")
    print(" Acompanhe abaixo os logs em tempo real das simulações:")
    print("========================================================\n", flush=True)

    def open_browser():
        time.sleep(0.8)
        webbrowser.open(f"http://127.0.0.1:{PORT}")

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor encerrado pelo usuário.", flush=True)
        server.server_close()


if __name__ == "__main__":
    start_server()

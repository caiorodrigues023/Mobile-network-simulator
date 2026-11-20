"""Teste end-to-end de todos os endpoints do app_web."""
import sys, os, json, urllib.request, threading, time, traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from http.server import ThreadingHTTPServer
from app_web import SimulatorWebHandler

PORT = 18765
errors = []

def test(label, fn):
    print(f"  [{label}] ", end="", flush=True)
    try:
        fn()
        print("✅ OK")
    except Exception as e:
        print(f"❌ FALHOU: {e}")
        errors.append((label, traceback.format_exc()))

def post_json(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())

def get_text(path):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()

# Start server
server = ThreadingHTTPServer(("127.0.0.1", PORT), SimulatorWebHandler)
t = threading.Thread(target=server.serve_forever, daemon=True)
t.start()
time.sleep(0.5)

print("=== Teste End-to-End do app_web.py ===\n")

# 1. Página principal
test("GET /", lambda: (
    html := get_text("/"),
    assert "Simulador 3D" in html, f"HTML principal não contém 'Simulador 3D'"
))

# 2. Plotly.js
test("GET /assets/plotly.min.js", lambda: (
    js := get_text("/assets/plotly.min.js"),
    assert len(js) > 100000, f"plotly.min.js muito pequeno: {len(js)} bytes"
))

# 3. Snapshot 3D
test("POST /api/snapshot", lambda: (
    r := post_json("/api/snapshot", {"M": 4, "K": 4, "N": 1, "model": "smb", "L": 8, "J": 8}),
    assert "html_3d" in r, "Sem campo html_3d",
    assert "snapshot" in r, "Sem campo snapshot",
    assert len(r["html_3d"]) > 1000, f"HTML 3D muito curto: {len(r['html_3d'])} chars",
    assert "/assets/plotly.min.js" in r["html_3d"], "Plotly URL incorreta no HTML 3D",
))

# 4. Simulação Monte Carlo (antenna_comparison)
test("POST /api/simulate (antenna)", lambda: (
    r := post_json("/api/simulate", {"M": 4, "K": 4, "N": 1, "sim": 10, "power_mode": "full_power", "L": 8, "J": 8}),
    assert r.get("status") == "success", f"Status: {r.get('status')}, erro: {r.get('error')}",
    assert "kpi_html" in r, "Sem campo kpi_html",
    assert len(r["kpi_html"]) > 2000, f"KPI HTML muito curto: {len(r['kpi_html'])} chars",
    assert "/assets/plotly.min.js" in r["kpi_html"], "Plotly URL incorreta no KPI HTML",
))

# 5. Simulação Monte Carlo (scichal4a / DPC)
test("POST /api/simulate (dpc_fixed)", lambda: (
    r := post_json("/api/simulate", {"M": 4, "K": 4, "N": 1, "sim": 5, "power_mode": "dpc_fixed"}),
    assert r.get("status") == "success", f"Status: {r.get('status')}, erro: {r.get('error')}",
    assert "/assets/plotly.min.js" in r["kpi_html"], "Plotly URL incorreta no SciChal4A HTML",
))

# 6. Mobilidade
test("POST /api/mobility", lambda: (
    r := post_json("/api/mobility", {"v_kmh": 120, "dt": 0.100, "D": 50, "L": 16}),
    assert "mobility_html" in r, "Sem campo mobility_html",
    assert len(r["mobility_html"]) > 1000, f"Mobility HTML muito curto",
    assert "/assets/plotly.min.js" in r["mobility_html"], "Plotly URL incorreta no Mobility HTML",
))

server.shutdown()

print(f"\n{'='*50}")
if errors:
    print(f"❌ {len(errors)} teste(s) falharam:")
    for label, tb in errors:
        print(f"\n--- {label} ---\n{tb}")
    sys.exit(1)
else:
    print("✅ TODOS OS 6 TESTES PASSARAM!")
    sys.exit(0)

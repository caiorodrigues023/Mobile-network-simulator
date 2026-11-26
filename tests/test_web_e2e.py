"""Teste end-to-end de todos os endpoints do app_web."""
import json
import os
import sys
import threading
import time
import traceback
import urllib.request
from http.server import ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app_web import SimulatorWebHandler

PORT = 18765
errors = []


def run_test(label, fn):
    print(f"  [{label}] ", end="", flush=True)
    try:
        fn()
        print("OK")
    except Exception as e:
        print(f"FALHOU: {e}")
        errors.append((label, traceback.format_exc()))


def post_json(path, data):
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        assert r.status == 200, f"Status HTTP inesperado: {r.status}"
        return json.loads(r.read().decode("utf-8"))


def get_text(path):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}")
    with urllib.request.urlopen(req, timeout=30) as r:
        assert r.status == 200, f"Status HTTP inesperado: {r.status}"
        return r.read().decode("utf-8")


def test_get_index():
    html = get_text("/")
    assert "Simulador 3D" in html, "HTML principal nao contem 'Simulador 3D'"
    assert 'id="param-sim"' in html, "HTML principal nao contem controle 'param-sim'"
    assert 'id="mc-round-select"' in html, "HTML principal nao contem controle 'mc-round-select'"
    assert 'id="mc-round-num"' in html, "HTML principal nao contem controle 'mc-round-num'"
    assert 'id="mob-v"' in html, "HTML principal nao contem controle 'mob-v'"
    assert '<head>\n    <meta charset="utf-8">\n    <title>Mobile Network Simulator 3D (Modo Ultraleve)</title>\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <script src="/assets/plotly.min.js">' not in html


def test_get_plotly_js():
    js = get_text("/assets/plotly.min.js")
    assert len(js) > 100000, f"plotly.min.js muito pequeno: {len(js)} bytes"


def test_get_ux_assets():
    css = get_text("/assets/ux.css")
    js = get_text("/assets/ux.js")
    assert len(css) > 100, f"ux.css muito pequeno: {len(css)} bytes"
    assert len(js) > 100, f"ux.js muito pequeno: {len(js)} bytes"


def test_post_snapshot():
    r = post_json("/api/snapshot", {"M": 4, "K": 4, "N": 1, "model": "smb", "L": 8, "J": 8})
    assert "html_3d" in r, "Sem campo html_3d"
    assert "snapshot" in r, "Sem campo snapshot"
    assert len(r["html_3d"]) > 1000, f"HTML 3D muito curto: {len(r['html_3d'])} chars"
    assert "/assets/plotly.min.js" in r["html_3d"], "Plotly URL incorreta no HTML 3D"
    assert r["html_3d"].count("plotly-latest.min.js") == 0, "HTML 3D ainda injeta plotly CDN duplicado"


def test_post_simulate_antenna():
    r = post_json("/api/simulate", {"M": 4, "K": 4, "N": 1, "sim": 10, "power_mode": "full_power", "L": 8, "J": 8, "seed": 42})
    assert r.get("status") == "success", f"Status: {r.get('status')}, erro: {r.get('error')}"
    assert "kpi_html" in r, "Sem campo kpi_html"
    assert len(r["kpi_html"]) > 2000, f"KPI HTML muito curto: {len(r['kpi_html'])} chars"
    assert "/assets/plotly.min.js" in r["kpi_html"], "Plotly URL incorreta no KPI HTML"

    mc_rounds = r.get("mc_rounds")
    assert mc_rounds and "presets" in mc_rounds, "Sem mc_rounds.presets na resposta de /api/simulate"
    presets = mc_rounds["presets"]
    for key in ("first", "last", "best", "median", "worst"):
        assert key in presets, f"Preset '{key}' ausente em mc_rounds"
        assert len(presets[key]["ues"]) == 4, f"Preset '{key}' deveria ter 4 UEs"
    assert presets["first"]["round"] == 1
    assert presets["last"]["round"] == 10

    r_round3 = post_json("/api/mc_round", {
        "M": 4, "K": 4, "N": 1, "power_mode": "full_power", "L": 8, "J": 8,
        "round": 3, "base_seed": mc_rounds["base_seed"],
    })
    assert r_round3["round"] == 3
    assert len(r_round3["ues"]) == 4

    r_round1 = post_json("/api/mc_round", {
        "M": 4, "K": 4, "N": 1, "power_mode": "full_power", "L": 8, "J": 8,
        "round": 1, "base_seed": mc_rounds["base_seed"],
    })
    assert r_round1["ues"] == presets["first"]["ues"], "Rodada 1 deveria ser identica ao preset 'first'"


def test_get_progress():
    txt = get_text("/api/progress")
    data = json.loads(txt)
    assert "running" in data and "pct" in data and "msg" in data and "elapsed_s" in data, (
        f"Resposta /api/progress inesperada: {data}"
    )


def test_post_simulate_dpc():
    r = post_json("/api/simulate", {"M": 4, "K": 4, "N": 1, "sim": 5, "power_mode": "dpc_fixed"})
    assert r.get("status") == "success", f"Status: {r.get('status')}, erro: {r.get('error')}"
    assert "kpi_html" in r, "Sem campo kpi_html"
    assert "/assets/plotly.min.js" in r["kpi_html"], "Plotly URL incorreta no SciChal4A HTML"


def test_post_simulate_dual_loop():
    t0 = time.perf_counter()
    r = post_json(
        "/api/simulate",
        {"M": 4, "K": 6, "N": 1, "sim": 10, "power_mode": "dual_loop", "mc_scope": "both", "seed": 42},
    )
    dt = time.perf_counter() - t0
    assert r.get("status") == "success", f"Status: {r.get('status')}, erro: {r.get('error')}"
    assert "kpi_html" in r and len(r["kpi_html"]) > 2000
    assert r.get("mc_rounds") and r["mc_rounds"]["sim"] == 10
    assert dt < 15.0, f"Simulacao dual_loop muito lenta: {dt:.2f}s"


def test_post_mobility():
    r = post_json("/api/mobility", {"v_kmh": 120, "dt": 0.100, "D": 50, "L": 16})
    assert "mobility_html" in r, "Sem campo mobility_html"
    assert len(r["mobility_html"]) > 1000, "Mobility HTML muito curto"
    assert "/assets/plotly.min.js" in r["mobility_html"], "Plotly URL incorreta no Mobility HTML"


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), SimulatorWebHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)

    try:
        print("=== Teste End-to-End do app_web.py ===\n")
        run_test("GET /", test_get_index)
        run_test("GET /api/progress", test_get_progress)
        run_test("GET /assets/plotly.min.js", test_get_plotly_js)
        run_test("GET /assets/ux.css & ux.js", test_get_ux_assets)
        run_test("POST /api/snapshot", test_post_snapshot)
        run_test("POST /api/simulate (antenna)", test_post_simulate_antenna)
        run_test("POST /api/simulate (dpc_fixed)", test_post_simulate_dpc)
        run_test("POST /api/simulate (dual_loop)", test_post_simulate_dual_loop)
        run_test("POST /api/mobility", test_post_mobility)
    finally:
        server.shutdown()
        server.server_close()

    print("\n" + "=" * 50)
    if errors:
        print(f"FALHA: {len(errors)} teste(s) falharam:")
        for label, tb in errors:
            print(f"\n--- {label} ---\n{tb}")
        sys.exit(1)
    else:
        print("TODOS OS TESTES PASSARAM!")
        sys.exit(0)


if __name__ == "__main__":
    main()

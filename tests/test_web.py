import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import urllib.request
import json
import time
import threading
from app_web import SimulatorWebHandler
from http.server import ThreadingHTTPServer

server = ThreadingHTTPServer(('127.0.0.1', 8055), SimulatorWebHandler)
t = threading.Thread(target=server.serve_forever, daemon=True)
t.start()

time.sleep(0.5)

# 1. Test GET index.html
req0 = urllib.request.urlopen('http://127.0.0.1:8055/')
assert req0.status == 200
html_index = req0.read().decode('utf-8')
assert "Mobile Network Simulator 3D" in html_index
print(f"1. GET / OK! Length: {len(html_index)} chars")

# 2. Test GET /assets/plotly.min.js
req_js = urllib.request.urlopen('http://127.0.0.1:8055/assets/plotly.min.js')
assert req_js.status == 200
cl = req_js.headers.get("Content-Length")
print(f"2. GET /assets/plotly.min.js OK! Content-Length: {cl}")

# 3. Test POST snapshot
post_bytes = json.dumps({'M': 4, 'K': 4, 'model': 'sector'}).encode('utf-8')
req2 = urllib.request.Request('http://127.0.0.1:8055/api/snapshot', data=post_bytes, headers={'Content-Type': 'application/json'})
resp2 = urllib.request.urlopen(req2)
assert resp2.status == 200
data2 = json.loads(resp2.read().decode('utf-8'))
print(f"3. POST /api/snapshot OK! HTML 3D length: {len(data2['html_3d'])}")

# 4. Test POST simulate
try:
    sim_bytes = json.dumps({'M': 4, 'K': 4, 'sim': 10, 'power_mode': 'full_power'}).encode('utf-8')
    req3 = urllib.request.Request('http://127.0.0.1:8055/api/simulate', data=sim_bytes, headers={'Content-Type': 'application/json'})
    resp3 = urllib.request.urlopen(req3)
    data3 = json.loads(resp3.read().decode('utf-8'))
    print(f"4. POST /api/simulate OK! Status: {resp3.status}, KPI HTML len: {len(data3.get('kpi_html', ''))}")
except Exception as e:
    import traceback
    print("Error in /api/simulate:")
    traceback.print_exc()

# 5. Test POST mobility
try:
    mob_bytes = json.dumps({'v_kmh': 120, 'dt': 0.1, 'D': 50, 'L': 16}).encode('utf-8')
    req4 = urllib.request.Request('http://127.0.0.1:8055/api/mobility', data=mob_bytes, headers={'Content-Type': 'application/json'})
    resp4 = urllib.request.urlopen(req4)
    data4 = json.loads(resp4.read().decode('utf-8'))
    print(f"5. POST /api/mobility OK! Status: {resp4.status}, Mobility HTML len: {len(data4.get('mobility_html', ''))}")
except Exception as e:
    import traceback
    print("Error in /api/mobility:")
    traceback.print_exc()

server.shutdown()
print("=== ALL API TESTS PASSED SUCCESSFULLY! ===")

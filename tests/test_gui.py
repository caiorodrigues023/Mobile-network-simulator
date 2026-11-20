import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PyQt6.QtWidgets import QApplication
from src.simulator.gui.worker import SnapshotCalculator
from src.simulator.gui.view3d import View3DGenerator
from src.simulator.gui.kpi_view import KPIViewGenerator
from src.simulator.gui.main_window import SimulatorMainWindow

print("1. Testing SnapshotCalculator...")
params = {
    "M": 4, "K": 4, "N": 1, "model": "smb", "L": 8, "J": 8,
    "use_rayleigh": False, "power_mode": "full_power",
    "heatmap_res": 30, "heatmap_metric": "sinr",
    "h_ap": 25.0, "h_ue": 1.5, "seed": 42
}
snap = SnapshotCalculator.compute_snapshot(params)
print(f"   Snapshot OK! APs: {len(snap['aps'])}, UEs: {len(snap['ues'])}, Avg SINR: {snap['avg_sinr']:.1f} dB")

print("2. Testing View3DGenerator...")
html_3d = View3DGenerator.generate_html(snap)
print(f"   3D HTML OK! Length: {len(html_3d)} chars")

print("3. Testing KPIViewGenerator...")
mock_mc = {
    "type": "antenna_comparison",
    "data": {
        "omni": {"sinr": [1.0, 2.0, 3.0], "cap": [10.0, 15.0, 20.0], "sum_cap": 45.0},
        "smb": {"sinr": [5.0, 8.0, 12.0], "cap": [25.0, 30.0, 40.0], "sum_cap": 95.0}
    },
    "params": params
}
html_kpi = KPIViewGenerator.generate_kpi_html(mock_mc)
print(f"   KPI HTML OK! Length: {len(html_kpi)} chars")

print("4. Testing SimulatorMainWindow instantiation...")
app = QApplication(sys.argv)
win = SimulatorMainWindow()
print("   SimulatorMainWindow instantiated successfully!")

print("5. Testing MonteCarloWorker...")
from src.simulator.gui.worker import MonteCarloWorker
mc_worker = MonteCarloWorker({"M": 4, "N": 1, "K": 2, "sim": 5, "models": ("omni", "sector", "smb")})
done = False
res_data = None
def on_finished(res):
    global done, res_data
    done = True
    res_data = res

mc_worker.simulation_finished.connect(on_finished)
mc_worker.start()
import time
t_start = time.time()
while not done and time.time() - t_start < 10:
    app.processEvents()
    time.sleep(0.02)
mc_worker.wait()
assert done is True
print(f"   MonteCarloWorker finished successfully! Models: {list(res_data['data'].keys())}")

print("=== ALL TESTS PASSED! ===")

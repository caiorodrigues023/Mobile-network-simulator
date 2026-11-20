# ---------------------------------------------------------------------------
# src/simulator/gui/worker.py — Thread em Segundo Plano para Cálculos Pesados
# ---------------------------------------------------------------------------

import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from ..allocation import attach_AP_UE, alocar_canais_ortogonal, DPC_Dual_Loop
from ..antennas import antenna_gain
from ..channel import adaptive_gains, angle_matrix, gain_matrix, gerar_shadowing
from ..config import B_TOTAL, K_0, PATHLOSS_EXP, K_CONST, SIGMA_X, P_MAX_DBM, P_MIN_DBM, MU_STEP, GAMMA_TAR_INIT
from ..entities import AP, UE
from ..metrics import SINR, SINR_adaptive, channel_capacity, evaluate_physical_kpis
from ..runner import distribuir_AP, simular_experimento, simular_scichal4a


class MonteCarloWorker(QThread):
    """Executa o experimento de Monte Carlo em segundo plano sem congelar a UI."""

    progress_changed = pyqtSignal(int, str)
    simulation_finished = pyqtSignal(dict)
    simulation_error = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            M = self.params.get("M", 4)
            N = self.params.get("N", 1)
            K = self.params.get("K", 4)
            sim = self.params.get("sim", 100)
            models = self.params.get("models", ("omni", "sector", "smb", "adaptive"))
            allocation = self.params.get("allocation", "ortogonal")
            use_rayleigh = self.params.get("use_rayleigh", False)
            L = self.params.get("L", 8)
            J = self.params.get("J", 8)
            power_control = self.params.get("power_control", "full_power")

            self.progress_changed.emit(5, f"Iniciando Monte Carlo ({sim} iterações, M={M}, K={K})...")

            if power_control in ("dpc_fixed", "dual_loop"):
                # Executa simulação avançada de controle de potência SciChal4A
                self.progress_changed.emit(20, "Executando simulação SciChal4A com controle de potência...")
                stats = simular_scichal4a(M=M, N=N, K=K, sim=sim, n_hist=min(sim, 10))
                self.progress_changed.emit(100, "Simulação concluída com sucesso!")
                self.simulation_finished.emit({"type": "scichal4a", "data": stats, "params": self.params})
            else:
                # Executa simulação comparativa de modelos de antena
                aps = distribuir_AP(M)
                resultados = {
                    model: {"power": [], "sinr": [], "cap": [], "sum_cap": []}
                    for model in models
                }

                step_size = max(1, sim // 20)
                for it in range(sim):
                    if self._is_cancelled:
                        self.progress_changed.emit(0, "Simulação cancelada pelo usuário.")
                        return

                    UE.id_counter = 0
                    ues = [UE(aps) for _ in range(K)]
                    shadowing = gerar_shadowing(K, M)

                    for model in models:
                        for ue in ues:
                            ue.ap, ue.beam, ue.channel = None, 0, 0
                        for ap in aps:
                            ap.ues = []

                        G = antenna_gain(ues, aps, model=model, L=L, J=J)
                        ue_xy = np.array([[ue.x, ue.y] for ue in ues], dtype=float)
                        ap_xy = np.array([[ap.x, ap.y] for ap in aps], dtype=float)

                        gains = gain_matrix(
                            ue_xy, ap_xy, G, shadowing=shadowing, use_rayleigh=use_rayleigh
                        )

                        serving_ap, serving_beam = attach_AP_UE(gains)
                        for i, ue in enumerate(ues):
                            ue.ap = aps[serving_ap[i]]
                            ue.beam = int(serving_beam[i])
                            ue.gain = gains[i, serving_ap[i], serving_beam[i]]
                            ue.ap.ues.append(ue)

                        alocar_canais_ortogonal(aps, ues, N, allocation)

                        if model.startswith("adaptive"):
                            err_deg = 0.0
                            if "_" in model:
                                try:
                                    err_deg = float(model.split("_")[1])
                                except ValueError:
                                    pass
                            Gm = adaptive_gains(ues, aps, L=L, aoa_err_deg=err_deg)
                            s = SINR_adaptive(ues, N, gains, Gm)
                            power = [ue.gain * ue.power * Gm[k, k] for k, ue in enumerate(ues)]
                        else:
                            s = SINR(ues, N, gains, G)
                            power = [ue.gain * ue.power for ue in ues]

                        cap = channel_capacity(s, N)
                        sum_cap = float(np.sum(cap))

                        resultados[model]["power"].extend(power)
                        resultados[model]["sinr"].extend(s)
                        resultados[model]["cap"].extend(cap)
                        resultados[model]["sum_cap"].append(sum_cap)

                    if (it + 1) % step_size == 0 or it == sim - 1:
                        pct = int(((it + 1) / sim) * 95)
                        self.progress_changed.emit(pct, f"Progresso: {it + 1}/{sim} iterações ({pct}%)...")

                for model in models:
                    resultados[model]["sum_cap"] = float(np.mean(resultados[model]["sum_cap"]))

                self.progress_changed.emit(100, "Experimento finalizado com sucesso!")
                self.simulation_finished.emit({"type": "antenna_comparison", "data": resultados, "params": self.params})

        except Exception as e:
            import traceback
            err = traceback.format_exc()
            self.simulation_error.emit(f"Erro na simulação: {str(e)}\n\n{err}")


class SnapshotCalculator:
    """Calcula um snapshot representativo da rede com mapa de calor 3D."""

    @staticmethod
    def compute_snapshot(params: dict) -> dict:
        """
        Calcula as posições dos APs, UEs, associação, potências, SINR,
        capacidade e a matriz do mapa de calor.
        """
        M = params.get("M", 4)
        N = params.get("N", 1)
        K = params.get("K", 4)
        model = params.get("model", "smb")
        L = params.get("L", 8)
        J = params.get("J", 8)
        aoa_err_deg = params.get("aoa_err_deg", 0.0)
        use_rayleigh = params.get("use_rayleigh", False)
        seed = params.get("seed", None)
        power_mode = params.get("power_mode", "full_power")
        h_ap = params.get("h_ap", 25.0)
        h_ue = params.get("h_ue", 1.5)

        if seed is not None and seed >= 0:
            np.random.seed(seed)

        aps = distribuir_AP(M)
        UE.id_counter = 0
        ues = [UE(aps) for _ in range(K)]
        shadowing = gerar_shadowing(K, M)

        G = antenna_gain(ues, aps, model=model, L=L, J=J)
        ue_xy = np.array([[ue.x, ue.y] for ue in ues], dtype=float)
        ap_xy = np.array([[ap.x, ap.y] for ap in aps], dtype=float)

        gains = gain_matrix(ue_xy, ap_xy, G, shadowing=shadowing, use_rayleigh=use_rayleigh)

        serving_ap, serving_beam = attach_AP_UE(gains)
        for i, ue in enumerate(ues):
            ue.ap = aps[serving_ap[i]]
            ue.beam = int(serving_beam[i])
            ue.gain = gains[i, serving_ap[i], serving_beam[i]]
            ue.ap.ues.append(ue)

        alocar_canais_ortogonal(aps, ues, N, "ortogonal")

        # Ajuste de potência conforme modo
        if power_mode == "full_power":
            for ue in ues:
                ue.power = 1.0  # 1 Watt (30 dBm)
        elif power_mode == "dpc_fixed":
            # DPC simples convergindo para meta 10 dB
            P_dbm = np.full(K, P_MAX_DBM)
            gamma_tar_lin = 10.0 ** (GAMMA_TAR_INIT / 10.0)
            B_c = B_TOTAL / N
            Pn = K_0 * B_c
            G_diag = np.array([ue.gain for ue in ues])
            for _ in range(60):
                p_w = 10 ** (P_dbm / 10.0) / 1000.0
                rx_sig = G_diag * p_w
                # Interfência aproximada
                s_est = rx_sig / (Pn + 1e-15)
                P_dbm = np.clip(P_dbm + MU_STEP * np.sign(gamma_tar_lin - s_est), P_MIN_DBM, P_MAX_DBM)
            for i, ue in enumerate(ues):
                ue.power = float(10 ** (P_dbm[i] / 10.0) / 1000.0)

        # Cálculo de SINR e capacidade
        if model.startswith("adaptive"):
            Gm = adaptive_gains(ues, aps, L=L, aoa_err_deg=aoa_err_deg)
            sinr_lin = np.asarray(SINR_adaptive(ues, N, gains, Gm)).flatten()
        else:
            sinr_lin = np.asarray(SINR(ues, N, gains, G)).flatten()

        sinr_db = 10.0 * np.log10(np.maximum(sinr_lin, 1e-12))
        cap_mbps = np.asarray(channel_capacity(sinr_lin, N)).flatten()

        # Calcula dados detalhados para cada UE
        ue_records = []
        for i, ue in enumerate(ues):
            dx = ue.x - ue.ap.x
            dy = ue.y - ue.ap.y
            dist = float(np.sqrt(dx**2 + dy**2))
            angle_deg = float(np.degrees(np.mod(np.arctan2(dy, dx), 2 * np.pi)))
            p_w = float(ue.power)
            p_dbm = float(10.0 * np.log10(max(p_w, 1e-12) * 1000.0))
            s_db = float(sinr_db[i])
            c_mbps = float(cap_mbps[i])

            ue_records.append({
                "id": ue.id,
                "x": float(ue.x),
                "y": float(ue.y),
                "z": h_ue,
                "ap_id": ue.ap.id,
                "ap_x": float(ue.ap.x),
                "ap_y": float(ue.ap.y),
                "ap_z": h_ap,
                "beam": int(ue.beam),
                "channel": int(ue.channel) if ue.channel is not None else 0,
                "dist": dist,
                "angle_deg": angle_deg,
                "power_w": p_w,
                "power_dbm": p_dbm,
                "sinr_lin": float(sinr_lin[i]),
                "sinr_db": s_db,
                "cap_mbps": c_mbps,
                "status": "Normal" if s_db >= 0 else "Alerta (Borda)",
            })

        # Resumo dos APs
        ap_records = []
        for ap in aps:
            connected = [u for u in ue_records if u["ap_id"] == ap.id]
            sum_cap = sum(u["cap_mbps"] for u in connected)
            ap_records.append({
                "id": ap.id,
                "x": float(ap.x),
                "y": float(ap.y),
                "z": h_ap,
                "connected_ues": len(connected),
                "sum_cap_mbps": sum_cap,
            })

        # Cálculo do Mapa de Calor (Heatmap)
        grid_res = params.get("heatmap_res", 45)
        heatmap_metric = params.get("heatmap_metric", "sinr")  # sinr, rsrp, cap, server

        heatmap_data = SnapshotCalculator._compute_heatmap(
            ap_xy=ap_xy,
            res=grid_res,
            model=model,
            L=L,
            J=J,
            N=N,
            metric=heatmap_metric,
        )

        return {
            "aps": ap_records,
            "ues": ue_records,
            "heatmap": heatmap_data,
            "params": params,
            "L": L,
            "J": J,
            "model": model,
            "avg_sinr": float(np.mean(sinr_db)),
            "sum_capacity": float(np.sum(cap_mbps)),
            "min_sinr": float(np.min(sinr_db)),
        }

    @staticmethod
    def _compute_heatmap(ap_xy: np.ndarray, res: int, model: str, L: int, J: int, N: int, metric: str) -> dict:
        """Gera uma grade 2D/3D vetorial de cobertura de sinal."""
        gx = np.linspace(0, 1000, res)
        gy = np.linspace(0, 1000, res)
        X, Y = np.meshgrid(gx, gy)
        pts = np.column_stack([X.ravel(), Y.ravel()])

        dx = pts[:, None, 0] - ap_xy[None, :, 0]
        dy = pts[:, None, 1] - ap_xy[None, :, 1]
        dist = np.maximum(np.sqrt(dx**2 + dy**2), 1.0)
        phi = np.mod(np.arctan2(dy, dx), 2 * np.pi)

        # Ganho de antena
        if model == "omni":
            G = np.ones((len(pts), len(ap_xy)))
        elif model == "sector":
            # Máximo ganho de setor em direção ao ponto
            theta_boresight = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
            delta = np.abs(phi[:, :, None] - theta_boresight[None, None, :]) % (2 * np.pi)
            delta = np.minimum(delta, 2 * np.pi - delta)
            sec_gains = np.where(delta <= (np.pi / 3), 1.0, 0.01)
            G = np.max(sec_gains, axis=-1)
        elif model == "smb":
            # Ganho SMB avaliado na direção do ponto
            from ..antennas import smb_gains
            # Avalia ganho máximo dos feixes para cada ponto
            smb_all = smb_gains(phi.ravel(), L=L, J=J)
            G = np.max(smb_all, axis=-1).reshape(len(pts), len(ap_xy)) / L
        else:
            # Adaptativo (potencial de ganho MRC ideal)
            G = np.ones((len(pts), len(ap_xy))) * (L / 2.0)

        # Perda de percurso
        pl = K_CONST / (dist ** PATHLOSS_EXP)
        P_rx = 1.0 * pl * G  # 1W por AP

        best_ap = np.argmax(P_rx, axis=1)
        S = np.max(P_rx, axis=1)
        I = np.sum(P_rx, axis=1) - S
        Pn = K_0 * (B_TOTAL / N)
        sinr = S / (I + Pn)
        sinr_db = 10.0 * np.log10(np.maximum(sinr, 1e-12))
        rsrp_dbm = 10.0 * np.log10(np.maximum(S, 1e-15) * 1000.0)
        cap = ((B_TOTAL / N) * np.log2(1.0 + sinr)) / 1e6

        if metric == "rsrp":
            values = rsrp_dbm.reshape(res, res)
            label = "RSRP (dBm)"
        elif metric == "cap":
            values = cap.reshape(res, res)
            label = "Capacidade (Mbps)"
        elif metric == "server":
            values = best_ap.reshape(res, res).astype(float)
            label = "AP Servidor"
        else:
            values = sinr_db.reshape(res, res)
            label = "SINR (dB)"

        return {
            "x": gx.tolist(),
            "y": gy.tolist(),
            "z": values.tolist(),
            "metric": metric,
            "label": label,
            "min_val": float(np.min(values)),
            "max_val": float(np.max(values)),
        }

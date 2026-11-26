# ---------------------------------------------------------------------------
# src/simulator/gui/worker.py — Snapshot 3D e Thread Monte Carlo
# ---------------------------------------------------------------------------

import numpy as np

from ..allocation import alocar_canais_ortogonal, attach_AP_UE
from ..antennas import antenna_gain
from ..channel import adaptive_gains, gain_matrix, gerar_shadowing
from ..config import (
    B_TOTAL,
    D_0,
    GAMMA_MAX_DB,
    GAMMA_MIN_DB,
    GAMMA_TAR_INIT,
    K_0,
    K_CONST,
    MU_STEP,
    PATHLOSS_EXP,
    P_MAX_DBM,
    P_MIN_DBM,
    SIGMA_R,
    SIGMA_X,
)
from ..entities import UE
from ..metrics import channel_capacity
from ..runner import (
    aplicar_controle_potencia,
    distribuir_AP,
    simular_experimento,
    simular_scichal4a,
)


MC_SEED_STRIDE = 104729


def mc_round_seed(base_seed: int, round_idx: int) -> int:
    """Retorna seed determinística para a iteração `round_idx` (base 0)."""
    return int((base_seed + round_idx * MC_SEED_STRIDE) % 2147483647)


def extract_sim_params(params: dict) -> dict:
    """Normaliza e aplica valores padrão para todos os parâmetros do simulador."""
    p = params or {}
    seed_raw = p.get("seed", -1)
    try:
        seed_val = int(seed_raw) if seed_raw is not None else -1
    except (TypeError, ValueError):
        seed_val = -1

    b_mhz = float(p.get("b_total_mhz", B_TOTAL / 1e6))
    return {
        "M": int(p.get("M", 4)),
        "K": int(p.get("K", 8)),
        "N": max(1, int(p.get("N", 1))),
        "allocation": str(p.get("allocation", "ortogonal")),
        "h_ap": float(p.get("h_ap", 25.0)),
        "h_ue": float(p.get("h_ue", 1.5)),
        "seed": seed_val,
        "model": str(p.get("model", "smb")),
        "L": int(p.get("L", 8)),
        "J": int(p.get("J", 8)),
        "aoa_err_deg": float(p.get("aoa_err_deg", 0.0)),
        "beamwidth_deg": float(p.get("beamwidth_deg", 120.0)),
        "g_min": float(p.get("g_min", 0.01)),
        "b_total_mhz": b_mhz,
        "b_total": b_mhz * 1e6,
        "k_0": float(p.get("k_0", K_0)),
        "pathloss_exp": float(p.get("pathloss_exp", PATHLOSS_EXP)),
        "k_const": float(p.get("k_const", K_CONST)),
        "d_0": float(p.get("d_0", D_0)),
        "sigma_x": float(p.get("sigma_x", SIGMA_X)),
        "use_rayleigh": bool(p.get("use_rayleigh", False)),
        "sigma_r": float(p.get("sigma_r", SIGMA_R)),
        "power_mode": str(p.get("power_mode", p.get("power_control", "full_power"))),
        "p_max_dbm": float(p.get("p_max_dbm", P_MAX_DBM)),
        "p_min_dbm": float(p.get("p_min_dbm", P_MIN_DBM)),
        "mu_step_db": float(p.get("mu_step_db", MU_STEP)),
        "gamma_tar_db": float(p.get("gamma_tar_db", GAMMA_TAR_INIT)),
        "gamma_max_db": float(p.get("gamma_max_db", GAMMA_MAX_DB)),
        "gamma_min_db": float(p.get("gamma_min_db", GAMMA_MIN_DB)),
        "t_outer": int(p.get("t_outer", 50)),
        "t_inner": int(p.get("t_inner", 150)),
        "tol_db": float(p.get("tol_db", 0.5)),
        "heatmap_res": int(p.get("heatmap_res", 45)),
        "heatmap_metric": str(p.get("heatmap_metric", "sinr")),
        "sim": int(p.get("sim", 100)),
        "mc_scope": p.get("mc_scope", None),
    }


def _apply_power_control(
    ues: list,
    gains: np.ndarray,
    G: np.ndarray,
    Gm: np.ndarray,
    is_adaptive: bool,
    cfg: dict,
    return_history: bool = False,
) -> np.ndarray | tuple:
    """Delega para `aplicar_controle_potencia` com os parâmetros de `cfg`."""
    return aplicar_controle_potencia(
        ues=ues,
        N=cfg["N"],
        gains=gains,
        G=G,
        Gm=Gm,
        is_adaptive=is_adaptive,
        power_mode=cfg["power_mode"],
        b_total=cfg["b_total"],
        k_0=cfg["k_0"],
        p_max_dbm=cfg["p_max_dbm"],
        p_min_dbm=cfg["p_min_dbm"],
        mu_step=cfg["mu_step_db"],
        gamma_tar_db=cfg["gamma_tar_db"],
        gamma_max_db=cfg["gamma_max_db"],
        gamma_min_db=cfg["gamma_min_db"],
        t_outer=cfg["t_outer"],
        t_inner=cfg["t_inner"],
        tol_db=cfg["tol_db"],
        return_history=return_history,
    )


def _serialize_dl_hist(hist: dict | None, K: int, snapshot_idx: int, model: str) -> dict | None:
    """Converte o histórico do Dual-Loop em dicionário compatível com JSON."""
    if hist is None:
        return None
    return {
        "K": int(K),
        "snapshot": int(snapshot_idx),
        "model": str(model),
        "power_w": np.asarray(hist["power_w"], dtype=float).tolist(),
        "sinr_lin": np.asarray(hist["sinr_lin"], dtype=float).tolist(),
        "target_db": np.asarray(hist["target_db"], dtype=float).tolist(),
        "active": np.asarray(hist["active"], dtype=bool).tolist(),
        "outer_marks": [int(x) for x in hist.get("outer_marks", [])],
        "p_max_dbm": float(hist.get("p_max_dbm", P_MAX_DBM)),
        "p_min_dbm": float(hist.get("p_min_dbm", P_MIN_DBM)),
        "mu_step": float(hist.get("mu_step", MU_STEP)),
        "gamma_min_db": float(hist.get("gamma_min_db", GAMMA_MIN_DB)),
        "gamma_max_db": float(hist.get("gamma_max_db", GAMMA_MAX_DB)),
        "tol_db": float(hist.get("tol_db", 0.5)),
    }


class SnapshotCalculator:
    """Calcula um snapshot representativo da rede com mapa de calor 3D."""

    @staticmethod
    def compute_snapshot(params: dict, include_heatmap: bool = True) -> dict:
        """
        Calcula as posições dos APs, UEs, associação, potências, SINR,
        capacidade e, opcionalmente, a matriz do mapa de calor.
        Apenas `power_mode == "dual_loop"` desliga UEs em outage.
        """
        cfg = extract_sim_params(params)
        M = cfg["M"]
        N = cfg["N"]
        K = cfg["K"]
        model = cfg["model"]
        L = cfg["L"]
        J = cfg["J"]
        aoa_err_deg = cfg["aoa_err_deg"]
        power_mode = cfg["power_mode"]
        h_ap = cfg["h_ap"]
        h_ue = cfg["h_ue"]
        b_total = cfg["b_total"]
        k_0 = cfg["k_0"]

        if cfg["seed"] >= 0:
            np.random.seed(cfg["seed"])

        aps = distribuir_AP(M)
        UE.id_counter = 0
        ues = [UE(aps) for _ in range(K)]
        shadowing = gerar_shadowing(K, M, sigma_x=cfg["sigma_x"])

        G = antenna_gain(
            ues,
            aps,
            model=model,
            beamwidth=np.deg2rad(cfg["beamwidth_deg"]),
            g_min=cfg["g_min"],
            L=L,
            J=J,
        )
        ue_xy = np.array([[ue.x, ue.y] for ue in ues], dtype=float)
        ap_xy = np.array([[ap.x, ap.y] for ap in aps], dtype=float)

        gains = gain_matrix(
            ue_xy,
            ap_xy,
            G,
            shadowing=shadowing,
            use_rayleigh=cfg["use_rayleigh"],
            pathloss_exp=cfg["pathloss_exp"],
            k_const=cfg["k_const"],
            d_0=cfg["d_0"],
            sigma_x=cfg["sigma_x"],
            sigma_r=cfg["sigma_r"],
        )

        serving_ap, serving_beam = attach_AP_UE(gains)
        for i, ue in enumerate(ues):
            ue.ap = aps[serving_ap[i]]
            ue.beam = int(serving_beam[i])
            ue.gain = gains[i, serving_ap[i], serving_beam[i]]
            ue.ap.ues.append(ue)

        alocar_canais_ortogonal(aps, ues, N, cfg["allocation"])

        is_adaptive = model.startswith("adaptive")
        if is_adaptive:
            if aoa_err_deg == 0.0 and "_" in model:
                try:
                    aoa_err_deg = float(model.split("_")[1])
                except ValueError:
                    pass
            Gm = adaptive_gains(ues, aps, L=L, aoa_err_deg=aoa_err_deg)
        else:
            Gm = None

        p_max_w = 10.0 ** (cfg["p_max_dbm"] / 10.0) / 1000.0

        # Ajuste de potência conforme algoritmo selecionado
        want_hist = bool(include_heatmap and power_mode == "dual_loop")
        if want_hist:
            sinr_lin, raw_dl_hist = _apply_power_control(
                ues, gains, G, Gm, is_adaptive, cfg, return_history=True
            )
            dl_hist = _serialize_dl_hist(raw_dl_hist, K=K, snapshot_idx=0, model=model)
        else:
            sinr_lin = _apply_power_control(ues, gains, G, Gm, is_adaptive, cfg)
            dl_hist = None

        cap_mbps = np.asarray(channel_capacity(sinr_lin, N, b_total=b_total)).flatten()
        if power_mode == "dual_loop":
            cap_mbps = np.where(np.array([ue.power > 0.0 for ue in ues]), cap_mbps, 0.0)

        # Calcula dados detalhados para cada UE
        gamma_min_db = cfg["gamma_min_db"]
        ue_records = []
        for i, ue in enumerate(ues):
            dx = ue.x - ue.ap.x
            dy = ue.y - ue.ap.y
            dist = float(np.sqrt(dx**2 + dy**2))
            angle_deg = float(np.degrees(np.mod(np.arctan2(dy, dx), 2 * np.pi)))

            if power_mode == "dual_loop" and ue.power <= 0.0:
                is_active = False
                p_w = 0.0
                p_dbm = -999.0
                s_lin = 0.0
                s_db = -999.0
                c_mbps = 0.0
                status = "Outage (Desligado)"
            else:
                is_active = True
                p_w = float(ue.power)
                p_dbm = float(10.0 * np.log10(max(p_w, 1e-12) * 1000.0))
                s_lin = float(sinr_lin[i])
                s_db = float(10.0 * np.log10(np.maximum(s_lin, 1e-12)))
                c_mbps = float(cap_mbps[i])
                if s_db < gamma_min_db:
                    status = f"Outage (< {gamma_min_db:g} dB)"
                elif s_db < 0.0:
                    status = "Alerta (Borda)"
                else:
                    status = "Normal"

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
                "active": is_active,
                "power_w": p_w,
                "power_dbm": p_dbm,
                "sinr_lin": s_lin,
                "sinr_db": s_db,
                "cap_mbps": c_mbps,
                "status": status,
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

        heatmap_data = None
        if include_heatmap:
            heatmap_data = SnapshotCalculator._compute_heatmap(
                ap_xy=ap_xy,
                res=cfg["heatmap_res"],
                model=model,
                L=L,
                J=J,
                N=N,
                metric=cfg["heatmap_metric"],
                p_max_w=p_max_w,
                beamwidth_deg=cfg["beamwidth_deg"],
                g_min=cfg["g_min"],
                b_total=b_total,
                k_0=k_0,
                pathloss_exp=cfg["pathloss_exp"],
                k_const=cfg["k_const"],
                d_0=cfg["d_0"],
            )

        active_sinr_db = [u["sinr_db"] for u in ue_records if u["active"]]
        avg_sinr = float(np.mean(active_sinr_db)) if active_sinr_db else 0.0
        min_sinr = float(np.min(active_sinr_db)) if active_sinr_db else 0.0

        result_snap = {
            "aps": ap_records,
            "ues": ue_records,
            "heatmap": heatmap_data,
            "params": cfg,
            "L": L,
            "J": J,
            "model": model,
            "avg_sinr": avg_sinr,
            "sum_capacity": float(np.sum(cap_mbps)),
            "min_sinr": min_sinr,
        }
        if dl_hist is not None:
            result_snap["dual_loop_hist"] = [dl_hist]
        return result_snap

    @staticmethod
    def _compute_heatmap(
        ap_xy: np.ndarray,
        res: int,
        model: str,
        L: int,
        J: int,
        N: int,
        metric: str,
        p_max_w: float = 1.0,
        beamwidth_deg: float = 120.0,
        g_min: float = 0.01,
        b_total: float = B_TOTAL,
        k_0: float = K_0,
        pathloss_exp: float = PATHLOSS_EXP,
        k_const: float = K_CONST,
        d_0: float = D_0,
    ) -> dict:
        """Gera uma grade 2D/3D vetorial de cobertura de sinal."""
        gx = np.linspace(0, 1000, res)
        gy = np.linspace(0, 1000, res)
        X, Y = np.meshgrid(gx, gy)
        pts = np.column_stack([X.ravel(), Y.ravel()])

        dx = pts[:, None, 0] - ap_xy[None, :, 0]
        dy = pts[:, None, 1] - ap_xy[None, :, 1]
        dist = np.maximum(np.sqrt(dx**2 + dy**2), d_0)
        phi = np.mod(np.arctan2(dy, dx), 2 * np.pi)

        if model == "omni":
            G = np.ones((len(pts), len(ap_xy)))
        elif model == "sector":
            theta_boresight = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
            delta = np.abs(phi[:, :, None] - theta_boresight[None, None, :]) % (2 * np.pi)
            delta = np.minimum(delta, 2 * np.pi - delta)
            half_bw = np.deg2rad(beamwidth_deg) / 2.0
            sec_gains = np.where(delta <= half_bw, 1.0, g_min)
            G = np.max(sec_gains, axis=-1)
        elif model == "smb":
            from ..antennas import smb_gains

            smb_all = smb_gains(phi.ravel(), L=L, J=J)
            G = np.max(smb_all, axis=-1).reshape(len(pts), len(ap_xy)) / L
        else:
            G = np.ones((len(pts), len(ap_xy))) * (L / 2.0)

        pl = k_const / (dist ** pathloss_exp)
        P_rx = p_max_w * pl * G

        best_ap = np.argmax(P_rx, axis=1)
        S = np.max(P_rx, axis=1)
        I = np.sum(P_rx, axis=1) - S
        Pn = k_0 * (b_total / N)
        sinr = S / (I + Pn)
        sinr_db = 10.0 * np.log10(np.maximum(sinr, 1e-12))
        rsrp_dbm = 10.0 * np.log10(np.maximum(S, 1e-15) * 1000.0)
        cap = ((b_total / N) * np.log2(1.0 + sinr)) / 1e6

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


def _pack_mc_round(it: int, r_seed: int, snap: dict, gamma_min_db: float) -> dict:
    """Formata o dicionário resumido de uma rodada Monte Carlo."""
    outage_cnt = sum(
        1
        for u in snap["ues"]
        if (not u["active"]) or (u["power_w"] <= 0.0) or (u["sinr_db"] < gamma_min_db)
    )
    return {
        "round": it + 1,
        "seed": r_seed,
        "ues": snap["ues"],
        "aps": snap["aps"],
        "avg_sinr": snap["avg_sinr"],
        "sum_capacity": snap["sum_capacity"],
        "outage_count": outage_cnt,
    }


def _collect_mc_rounds(cfg: dict, base_seed: int, round_metrics: list = None) -> dict:
    """Coleta snapshots rápidos das rodadas-chave do experimento Monte Carlo."""
    sim = max(1, int(cfg["sim"]))
    gamma_min_db = float(cfg["gamma_min_db"])

    if round_metrics:
        first_m = round_metrics[0]
        last_m = round_metrics[-1]
        best_m = max(round_metrics, key=lambda r: (r["sum_capacity"], r["min_sinr"]))
        worst_m = min(round_metrics, key=lambda r: (r["sum_capacity"], r["min_sinr"]))
        ordered = sorted(round_metrics, key=lambda r: (r["sum_capacity"], r["min_sinr"]))
        median_m = ordered[len(ordered) // 2]

        role_to_it = {
            "first": int(first_m["it"]),
            "last": int(last_m["it"]),
            "best": int(best_m["it"]),
            "median": int(median_m["it"]),
            "worst": int(worst_m["it"]),
        }
        cache = {}
        for it in sorted(set(role_to_it.values())):
            r_seed = mc_round_seed(base_seed, it)
            snap = SnapshotCalculator.compute_snapshot({**cfg, "seed": r_seed}, include_heatmap=False)
            cache[it] = _pack_mc_round(it, r_seed, snap, gamma_min_db)

        presets = {role: cache[it] for role, it in role_to_it.items()}
    else:
        if sim <= 50:
            indices = list(range(sim))
        else:
            indices = sorted({int(x) for x in np.linspace(0, sim - 1, 50)} | {0, sim - 1})

        records = []
        for it in indices:
            r_seed = mc_round_seed(base_seed, it)
            snap = SnapshotCalculator.compute_snapshot({**cfg, "seed": r_seed}, include_heatmap=False)
            records.append({
                "it": it,
                "seed": r_seed,
                "snap": snap,
                "sum_capacity": snap["sum_capacity"],
                "min_sinr": snap["min_sinr"],
            })

        first_rec = next((r for r in records if r["it"] == 0), records[0])
        last_rec = next((r for r in records if r["it"] == sim - 1), records[-1])
        best_rec = max(records, key=lambda r: (r["sum_capacity"], r["min_sinr"]))
        worst_rec = min(records, key=lambda r: (r["sum_capacity"], r["min_sinr"]))
        ordered = sorted(records, key=lambda r: (r["sum_capacity"], r["min_sinr"]))
        median_rec = ordered[len(ordered) // 2]

        presets = {
            role: _pack_mc_round(rec["it"], rec["seed"], rec["snap"], gamma_min_db)
            for role, rec in (
                ("first", first_rec),
                ("last", last_rec),
                ("best", best_rec),
                ("median", median_rec),
                ("worst", worst_rec),
            )
        }

    return {
        "sim": sim,
        "base_seed": int(base_seed),
        "stride": MC_SEED_STRIDE,
        "model": cfg["model"],
        "power_mode": cfg["power_mode"],
        "presets": presets,
    }


def run_monte_carlo_suite(params: dict, progress_cb=None, cancel_check=None) -> dict:
    """
    Executa experimentos Monte Carlo de acordo com `mc_scope`:
      - 'both': executa comparativo de antenas + controle de potência SciChal4A.
      - 'power': executa apenas controle de potência SciChal4A.
      - 'antennas' (ou fallback legado): executa apenas comparativo de antenas.
    Apenas o algoritmo 'dual_loop' desliga usuários em outage.
    """
    cfg = extract_sim_params(params)
    M = cfg["M"]
    N = cfg["N"]
    K = cfg["K"]
    sim = cfg["sim"]
    models = tuple(params.get("models", ("omni", "sector", "smb", "adaptive", "adaptive_1", "adaptive_5", "adaptive_10")))
    allocation = cfg["allocation"]
    use_rayleigh = cfg["use_rayleigh"]
    L = cfg["L"]
    J = cfg["J"]
    power_mode = cfg["power_mode"]
    scope = cfg["mc_scope"]
    if not scope:
        scope = "power" if power_mode in ("dpc_fixed", "dual_loop") else "antennas"

    base_seed = cfg["seed"] if cfg["seed"] >= 0 else int(np.random.randint(0, 1_000_000))

    def emit(pct: int, msg: str):
        if progress_cb is not None:
            progress_cb(pct, msg)

    def _run_antennas(pct_start: int, pct_span: int) -> tuple | None:
        aps = distribuir_AP(M)
        p_max_w = 10.0 ** (cfg["p_max_dbm"] / 10.0) / 1000.0
        resultados = {
            model: {"power": [], "tx_power": [], "sinr": [], "cap": [], "sum_cap": []}
            for model in models
        }
        round_metrics = []
        ant_dl_hist = []
        target_model = cfg["model"]
        step_size = max(1, sim // 15)
        for it in range(sim):
            if cancel_check is not None and cancel_check():
                return None

            np.random.seed(mc_round_seed(base_seed, it))
            UE.id_counter = 0
            ues = [UE(aps) for _ in range(K)]
            shadowing = gerar_shadowing(K, M, sigma_x=cfg["sigma_x"])

            for model in models:
                for ue in ues:
                    ue.ap, ue.beam, ue.channel, ue.power = None, 0, 0, p_max_w
                for ap in aps:
                    ap.ues = []

                G = antenna_gain(
                    ues,
                    aps,
                    model=model,
                    beamwidth=np.deg2rad(cfg["beamwidth_deg"]),
                    g_min=cfg["g_min"],
                    L=L,
                    J=J,
                )
                ue_xy = np.array([[ue.x, ue.y] for ue in ues], dtype=float)
                ap_xy = np.array([[ap.x, ap.y] for ap in aps], dtype=float)

                gains = gain_matrix(
                    ue_xy,
                    ap_xy,
                    G,
                    shadowing=shadowing,
                    use_rayleigh=use_rayleigh,
                    pathloss_exp=cfg["pathloss_exp"],
                    k_const=cfg["k_const"],
                    d_0=cfg["d_0"],
                    sigma_x=cfg["sigma_x"],
                    sigma_r=cfg["sigma_r"],
                )

                serving_ap, serving_beam = attach_AP_UE(gains)
                for i, ue in enumerate(ues):
                    ue.ap = aps[serving_ap[i]]
                    ue.beam = int(serving_beam[i])
                    ue.gain = gains[i, serving_ap[i], serving_beam[i]]
                    ue.ap.ues.append(ue)

                alocar_canais_ortogonal(aps, ues, N, allocation)

                is_adaptive = model.startswith("adaptive")
                if is_adaptive:
                    err_deg = cfg["aoa_err_deg"]
                    if "_" in model:
                        try:
                            err_deg = float(model.split("_")[1])
                        except ValueError:
                            pass
                    Gm = adaptive_gains(ues, aps, L=L, aoa_err_deg=err_deg)
                else:
                    Gm = None

                capture_ant_hist = (
                    power_mode == "dual_loop" and model == target_model and it < min(sim, 10)
                )
                if capture_ant_hist:
                    sinr_lin, dl_h = _apply_power_control(
                        ues, gains, G, Gm, is_adaptive, cfg, return_history=True
                    )
                    if dl_h is not None:
                        dl_h["K"] = K
                        dl_h["snapshot"] = it
                        dl_h["model"] = model
                        ant_dl_hist.append(dl_h)
                else:
                    sinr_lin = _apply_power_control(ues, gains, G, Gm, is_adaptive, cfg)
                if is_adaptive:
                    power = [
                        ue.gain * ue.power * Gm[k, k] if ue.power > 0.0 else 0.0
                        for k, ue in enumerate(ues)
                    ]
                else:
                    power = [
                        ue.gain * ue.power if ue.power > 0.0 else 0.0
                        for ue in ues
                    ]

                tx_power_tot = float(sum(ue.power for ue in ues))
                cap = np.asarray(channel_capacity(sinr_lin, N, b_total=cfg["b_total"])).flatten()
                if power_mode == "dual_loop":
                    cap = np.where(np.array([ue.power > 0.0 for ue in ues]), cap, 0.0)
                sum_cap = float(np.sum(cap))

                if model == target_model:
                    if power_mode == "dual_loop":
                        active_mask = np.array([ue.power > 0.0 for ue in ues], dtype=bool)
                    else:
                        active_mask = np.ones(K, dtype=bool)
                    if np.any(active_mask):
                        min_sinr_db = float(np.min(10.0 * np.log10(np.maximum(sinr_lin[active_mask], 1e-12))))
                    else:
                        min_sinr_db = 0.0
                    round_metrics.append({
                        "it": it,
                        "sum_capacity": sum_cap,
                        "min_sinr": min_sinr_db,
                    })

                resultados[model]["power"].extend(power)
                resultados[model]["tx_power"].append(tx_power_tot)
                resultados[model]["sinr"].extend(sinr_lin.tolist())
                resultados[model]["cap"].extend(cap.tolist())
                resultados[model]["sum_cap"].append(sum_cap)

            if (it + 1) % step_size == 0 or it == sim - 1:
                pct = pct_start + int(((it + 1) / sim) * pct_span)
                emit(pct, f"Antenas: {it + 1}/{sim} iterações ({pct}%)...")

        for model in models:
            resultados[model]["sum_cap"] = float(np.mean(resultados[model]["sum_cap"]))
            resultados[model]["tx_power"] = float(np.mean(resultados[model]["tx_power"]))
        return resultados, round_metrics, ant_dl_hist

    def _run_scichal4a(pct_start: int, pct_span: int) -> dict:
        def _sci_cb(done: int, total: int):
            pct = pct_start + int((done / max(1, total)) * pct_span)
            emit(pct, f"SciChal4A (Potência): {done}/{total} iterações ({pct}%)...")

        return simular_scichal4a(
            M=M,
            N=N,
            K=K,
            sim=sim,
            n_hist=min(sim, 10),
            L=L,
            use_rayleigh=bool(params.get("scichal_rayleigh", True))
            if "scichal_rayleigh" in params
            else use_rayleigh,
            b_total=cfg["b_total"],
            k_0=cfg["k_0"],
            pathloss_exp=cfg["pathloss_exp"],
            k_const=cfg["k_const"],
            d_0=cfg["d_0"],
            sigma_x=cfg["sigma_x"],
            sigma_r=cfg["sigma_r"],
            p_max_dbm=cfg["p_max_dbm"],
            p_min_dbm=cfg["p_min_dbm"],
            mu_step=cfg["mu_step_db"],
            gamma_tar_init=cfg["gamma_tar_db"],
            gamma_max_db=cfg["gamma_max_db"],
            gamma_min_db=cfg["gamma_min_db"],
            t_outer=cfg["t_outer"],
            t_inner=cfg["t_inner"],
            tol_db=cfg["tol_db"],
            aoa_err_deg=cfg["aoa_err_deg"],
            seed=cfg["seed"],
            progress_cb=_sci_cb,
        )

    emit(5, f"Iniciando Monte Carlo ({sim} iterações, M={M}, K={K})...")

    if scope == "both":
        ant_out = _run_antennas(5, 50)
        if ant_out is None:
            return None
        res_ant, round_metrics, _ = ant_out
        emit(56, "Executando comparativo de Controle de Potência (SciChal4A)...")
        res_sci = _run_scichal4a(56, 39)
        for h in res_sci.get("dual_loop_hist", []):
            if isinstance(h, dict):
                h.setdefault("model", "adaptive")
        emit(96, "Coletando rodadas-chave do Monte Carlo...")
        mc_rounds = _collect_mc_rounds(cfg, base_seed, round_metrics=round_metrics)
        emit(100, "Simulação combinada concluída com sucesso!")
        return {
            "type": "combined",
            "data": {"antennas": res_ant, "scichal4a": res_sci},
            "params": cfg,
            "mc_rounds": mc_rounds,
        }
    elif scope == "power":
        emit(10, "Executando simulação SciChal4A (Controle de Potência)...")
        stats = _run_scichal4a(10, 80)
        for h in stats.get("dual_loop_hist", []):
            if isinstance(h, dict):
                h.setdefault("model", "adaptive")
        emit(92, "Coletando rodadas-chave do Monte Carlo...")
        mc_rounds = _collect_mc_rounds(cfg, base_seed)
        emit(100, "Simulação concluída com sucesso!")
        return {"type": "scichal4a", "data": stats, "params": cfg, "mc_rounds": mc_rounds}
    else:
        ant_out = _run_antennas(5, 88)
        if ant_out is None:
            return None
        res_ant, round_metrics, ant_dl_hist = ant_out
        emit(95, "Coletando rodadas-chave do Monte Carlo...")
        mc_rounds = _collect_mc_rounds(cfg, base_seed, round_metrics=round_metrics)
        emit(100, "Experimento finalizado com sucesso!")
        out = {
            "type": "antenna_comparison",
            "data": res_ant,
            "params": cfg,
            "mc_rounds": mc_rounds,
        }
        if ant_dl_hist:
            out["dual_loop_hist"] = ant_dl_hist
        return out


def _build_monte_carlo_worker_class():
    from PyQt6.QtCore import QThread, pyqtSignal

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
                res = run_monte_carlo_suite(
                    self.params,
                    progress_cb=self.progress_changed.emit,
                    cancel_check=lambda: self._is_cancelled,
                )
                if res is None:
                    self.progress_changed.emit(0, "Simulação cancelada pelo usuário.")
                    return
                self.simulation_finished.emit(res)
            except Exception as e:
                import traceback

                err = traceback.format_exc()
                self.simulation_error.emit(f"Erro na simulação: {str(e)}\n\n{err}")

    return MonteCarloWorker


def __getattr__(name: str):
    if name == "MonteCarloWorker":
        cls = _build_monte_carlo_worker_class()
        globals()["MonteCarloWorker"] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

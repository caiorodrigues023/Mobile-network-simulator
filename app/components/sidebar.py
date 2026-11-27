# ---------------------------------------------------------------------------
# app/components/sidebar.py — Barra Lateral de Configuração Completa
# ---------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import streamlit as st

from simulator.config import (
    ALLOCATION_LABELS,
    MOBILITY_PROFILES_KMH,
    MODEL_LABELS,
)
from simulator.schemas import (
    AntennaConfig,
    ChannelConfig,
    HandoverConfig,
    MobilityConfig,
    PowerControlConfig,
    SimulationConfig,
)

ANTENNA_OPTIONS: list[str] = [
    "omni",
    "sector",
    "smb",
    "mrc",
    "ula",
    "adaptive_1",
    "adaptive_5",
    "adaptive_10",
]

POWER_OPTIONS: list[str] = [
    "full_power",
    "dpc_fixed",
    "dual_loop",
    "cipc",
    "optimal",
    "max_sum",
    "max_prod",
]

EXERCISE_PRESETS: dict[str, dict[str, Any]] = {
    "Padrão SciChal4A — Dual-Loop (M=9, K=4)": {
        "M": 9,
        "K": 4,
        "k_scenarios": [1, 4, 8],
        "N": 1,
        "allocation_policy": "full_reuse",
        "comparison_mode": "power",
        "snapshot_antenna_model": "mrc",
        "snapshot_power_mode": "dual_loop",
        "channel": {"use_shadowing": True, "sigma_x": 2.0, "use_rayleigh": True},
        "antenna": {"models": ["omni", "sector", "smb", "mrc", "ula"], "L": 8, "J": 8, "aoa_err_deg": 0.0},
        "power_control": {
            "algorithms": ["full_power", "dpc_fixed", "dual_loop"],
            "apply_physical_limits": True,
        },
    },
    "Parte 4 Ex. 1–3 — Antenas (M=16, N=1)": {
        "M": 16,
        "K": 4,
        "k_scenarios": [1, 4, 8],
        "N": 1,
        "allocation_policy": "full_reuse",
        "comparison_mode": "antennas",
        "snapshot_antenna_model": "smb",
        "snapshot_power_mode": "full_power",
        "channel": {"use_shadowing": True, "sigma_x": 2.0, "use_rayleigh": False},
        "antenna": {"models": ["omni", "sector", "smb", "mrc", "ula"], "L": 8, "J": 8, "aoa_err_deg": 0.0},
        "power_control": {
            "algorithms": ["full_power"],
            "apply_physical_limits": False,
        },
    },
    "Parte 4 Ex. 4 — Sensibilidade a Erro AoA (M=16)": {
        "M": 16,
        "K": 4,
        "k_scenarios": [1, 4, 8],
        "N": 1,
        "allocation_policy": "full_reuse",
        "comparison_mode": "antennas",
        "snapshot_antenna_model": "adaptive_5",
        "snapshot_power_mode": "full_power",
        "channel": {"use_shadowing": True, "sigma_x": 2.0, "use_rayleigh": False},
        "antenna": {
            "models": ["omni", "mrc", "adaptive_1", "adaptive_5", "adaptive_10"],
            "L": 8,
            "J": 8,
            "aoa_err_deg": 5.0,
        },
        "power_control": {"algorithms": ["full_power"], "apply_physical_limits": False},
    },
    "Parte 2 Ex. 6–7 — PAPOA + Carrier Aggregation (M=16, N=2)": {
        "M": 16,
        "K": 16,
        "k_scenarios": [8, 16, 32],
        "N": 2,
        "allocation_policy": "papoa_ca",
        "comparison_mode": "antennas",
        "snapshot_antenna_model": "omni",
        "snapshot_power_mode": "full_power",
        "channel": {"use_shadowing": True, "sigma_x": 2.0, "use_rayleigh": False},
        "antenna": {"models": ["omni", "sector", "smb"], "L": 8, "J": 8, "aoa_err_deg": 0.0},
        "power_control": {"algorithms": ["full_power"], "apply_physical_limits": False},
    },
    "Parte 2 Ex. 9–11 — CIPC vs. Controle Ótimo (M=1, K=2)": {
        "M": 1,
        "K": 2,
        "k_scenarios": [2, 4],
        "N": 1,
        "area_m": 100.0,
        "allocation_policy": "full_reuse",
        "comparison_mode": "power",
        "snapshot_antenna_model": "omni",
        "snapshot_power_mode": "optimal",
        "channel": {"use_shadowing": True, "sigma_x": 2.0, "use_rayleigh": False},
        "antenna": {"models": ["omni"], "L": 8, "J": 8, "aoa_err_deg": 0.0},
        "power_control": {
            "algorithms": ["full_power", "cipc", "optimal", "dpc_fixed", "dual_loop"],
            "gamma_tar_init_db": -11.44,
            "apply_physical_limits": False,
        },
    },
    "Parte 3 — RRM: DPC vs. MaxSum vs. MaxProd (M=4, K=4)": {
        "M": 4,
        "K": 4,
        "k_scenarios": [4, 8],
        "N": 1,
        "allocation_policy": "full_reuse",
        "comparison_mode": "power",
        "snapshot_antenna_model": "omni",
        "snapshot_power_mode": "max_sum",
        "channel": {"use_shadowing": True, "sigma_x": 2.0, "use_rayleigh": True},
        "antenna": {"models": ["omni", "mrc"], "L": 8, "J": 8, "aoa_err_deg": 0.0},
        "power_control": {
            "algorithms": ["full_power", "dpc_fixed", "max_sum", "max_prod", "dual_loop"],
            "apply_physical_limits": False,
        },
    },
}


def _parse_k_scenarios(raw_text: str, fallback_k: int) -> tuple[int, ...]:
    vals: list[int] = []
    for token in raw_text.replace(";", ",").split(","):
        tok = token.strip()
        if not tok:
            continue
        try:
            v = int(tok)
            if v >= 1:
                vals.append(v)
        except ValueError:
            pass
    if not vals:
        return (fallback_k,)
    return tuple(sorted(set(vals)))


def _apply_config_to_widget_state(cfg: SimulationConfig) -> None:
    """Popula as chaves dos widgets da sidebar a partir de um `SimulationConfig`."""
    st.session_state["sb_mc_runs"] = int(cfg.monte_carlo_runs)
    st.session_state["sb_seed"] = int(cfg.seed)
    st.session_state["sb_ue_mode"] = (
        "Em Movimento (Mobilidade / Handover / SDMA)" if cfg.ue_mode == "mobile" else "Estático"
    )
    st.session_state["sb_comp_mode"] = cfg.comparison_mode
    st.session_state["sb_M"] = int(cfg.M)
    st.session_state["sb_area"] = float(cfg.area_m)
    st.session_state["sb_K"] = int(cfg.K)
    st.session_state["sb_k_scenarios"] = ", ".join(str(k) for k in cfg.k_scenarios)
    st.session_state["sb_N"] = int(cfg.N)
    st.session_state["sb_bt_mhz"] = float(cfg.channel.b_total_hz / 1e6)
    st.session_state["sb_alloc"] = cfg.allocation_policy

    st.session_state["sb_use_shad"] = bool(cfg.channel.use_shadowing)
    st.session_state["sb_sigma_x"] = float(cfg.channel.sigma_x)
    st.session_state["sb_use_ray"] = bool(cfg.channel.use_rayleigh)
    st.session_state["sb_sigma_r"] = float(cfg.channel.sigma_r)
    st.session_state["sb_n_exp"] = float(cfg.channel.pathloss_exp)
    st.session_state["sb_k_const"] = f"{cfg.channel.k_const:.1e}"
    st.session_state["sb_d0"] = float(cfg.channel.d_0)

    st.session_state["sb_ant_models"] = list(cfg.antenna.models)
    st.session_state["sb_L"] = int(cfg.antenna.L)
    st.session_state["sb_J"] = int(cfg.antenna.J)
    st.session_state["sb_aoa_err"] = float(cfg.antenna.aoa_err_deg)
    st.session_state["sb_beamwidth"] = float(cfg.antenna.beamwidth_deg)
    st.session_state["sb_g_min"] = float(cfg.antenna.g_min)

    st.session_state["sb_pwr_algos"] = list(cfg.power_control.algorithms)
    st.session_state["sb_p_max"] = float(cfg.power_control.p_max_dbm)
    st.session_state["sb_p_min"] = float(cfg.power_control.p_min_dbm)
    st.session_state["sb_mu"] = float(cfg.power_control.mu_step_db)
    st.session_state["sb_gamma_tar"] = float(cfg.power_control.gamma_tar_init_db)
    st.session_state["sb_gamma_max"] = float(cfg.power_control.gamma_max_db)
    st.session_state["sb_gamma_min"] = float(cfg.power_control.gamma_min_db)
    st.session_state["sb_t_outer"] = int(cfg.power_control.t_outer)
    st.session_state["sb_t_inner"] = int(cfg.power_control.t_inner)
    st.session_state["sb_tol_db"] = float(cfg.power_control.tol_db)
    st.session_state["sb_phys_limits"] = bool(cfg.power_control.apply_physical_limits)

    st.session_state["tab1_ant_select"] = cfg.snapshot_antenna_model
    st.session_state["tab1_pwr_select"] = cfg.snapshot_power_mode


def render_sidebar() -> tuple[SimulationConfig, bool, bool]:
    """
    Renderiza todos os controles editáveis na barra lateral.
    Retorna: (config_validada, clicked_run_mc, clicked_refresh_snapshot).
    """
    if "sb_initialized" not in st.session_state:
        _apply_config_to_widget_state(SimulationConfig())
        st.session_state["sb_initialized"] = True

    pending_cfg = st.session_state.pop("_pending_config_to_apply", None)
    if pending_cfg is not None:
        _apply_config_to_widget_state(pending_cfg)

    with st.sidebar:
        st.title("📡 Mobile Network Simulator")
        st.caption("Arquitetura Híbrida (NumPy Vetorizado + Plotly)")

        # ------------------------------------------------------------------
        # 1. Execução & Monte Carlo (topo da sidebar)
        # ------------------------------------------------------------------
        with st.expander("🚀 Execução & Monte Carlo", expanded=True):
            c1, c2 = st.columns(2)
            mc_runs = c1.number_input(
                "Monte Carlo Runs",
                min_value=10,
                max_value=20000,
                step=100,
                key="sb_mc_runs",
            )
            seed = c2.number_input(
                "Seed",
                min_value=0,
                max_value=2_000_000_000,
                step=1,
                key="sb_seed",
            )

            ue_mode_label = st.radio(
                "Modo do UE",
                options=["Estático", "Em Movimento (Mobilidade / Handover / SDMA)"],
                key="sb_ue_mode",
            )
            ue_mode = "mobile" if "Movimento" in ue_mode_label else "static"

            comp_labels = {
                "power": "Algoritmos de Potência",
                "antennas": "Modelos de Antena",
                "both": "Ambos (Antenas + Potência)",
            }
            comparison_mode = st.selectbox(
                "Comparativo no Monte Carlo",
                options=["power", "antennas", "both"],
                format_func=lambda k: comp_labels[k],
                key="sb_comp_mode",
                help="Evita explosão combinatória: compara Antenas com a potência do snapshot e/ou Potência com a antena do snapshot.",
            )

            col_b1, col_b2 = st.columns(2)
            run_mc_clicked = col_b1.button(
                "▶️ Executar Monte Carlo",
                type="primary",
                width="stretch",
            )
            refresh_snap_clicked = col_b2.button(
                "🔄 Atualizar Snapshot",
                width="stretch",
            )

            # Presets rápidos dos roteiros
            preset_name = st.selectbox(
                "Presets dos Roteiros",
                options=["— Selecionar preset —", *EXERCISE_PRESETS.keys()],
                index=0,
                key="sb_preset_sel",
            )
            if preset_name != "— Selecionar preset —":
                if st.button("Aplicar Preset", width="stretch"):
                    base = SimulationConfig().to_dict()
                    patch = EXERCISE_PRESETS[preset_name]
                    for k, v in patch.items():
                        if isinstance(v, dict) and isinstance(base.get(k), dict):
                            base[k].update(v)
                        else:
                            base[k] = v
                    st.session_state["_pending_config_to_apply"] = SimulationConfig.from_dict(base)
                    st.rerun()

        # ------------------------------------------------------------------
        # 2. Topologia & Recursos
        # ------------------------------------------------------------------
        with st.expander("🗺️ Topologia & Recursos", expanded=True):
            t1, t2 = st.columns(2)
            M = t1.selectbox("APs (M)", options=[1, 4, 9, 16, 25], key="sb_M")
            area_m = t2.number_input(
                "Área (m)", min_value=50.0, max_value=5000.0, step=100.0, key="sb_area"
            )
            t3, t4 = st.columns(2)
            K_snap = t3.number_input(
                "UEs no Snapshot (K)", min_value=1, max_value=128, step=1, key="sb_K"
            )
            N_ch = t4.number_input("Canais (N)", min_value=1, max_value=32, step=1, key="sb_N")

            k_scen_str = st.text_input(
                "Cargas K para Monte Carlo (vírgula)",
                key="sb_k_scenarios",
                help="Ex.: 1, 4, 8",
            )
            k_scenarios = _parse_k_scenarios(k_scen_str, int(K_snap))

            bt_mhz = st.number_input(
                "Banda Total B_T (MHz)",
                min_value=1.0,
                max_value=1000.0,
                step=10.0,
                key="sb_bt_mhz",
            )
            alloc_policy = st.selectbox(
                "Política de Alocação de Canais",
                options=["full_reuse", "papoa", "papoa_ca", "random"],
                format_func=lambda k: ALLOCATION_LABELS.get(k, k),
                key="sb_alloc",
            )

        # ------------------------------------------------------------------
        # 3. Modelagem de Canal
        # ------------------------------------------------------------------
        with st.expander("🌊 Modelagem de Canal", expanded=False):
            use_shadowing = st.checkbox("Ativar Shadowing (Lognormal)", key="sb_use_shad")
            sigma_x = st.number_input(
                "Desvio-padrão Shadowing (σ_X)",
                min_value=0.0,
                max_value=10.0,
                step=0.25,
                disabled=not use_shadowing,
                key="sb_sigma_x",
            )
            use_rayleigh = st.checkbox("Ativar Fast Fading (Rayleigh)", key="sb_use_ray")
            sigma_r = st.number_input(
                "Escala Rayleigh (σ_r)",
                min_value=0.05,
                max_value=3.0,
                step=0.05,
                format="%.4f",
                disabled=not use_rayleigh,
                key="sb_sigma_r",
            )
            c_ch1, c_ch2, c_ch3 = st.columns(3)
            n_exp = c_ch1.number_input(
                "Expoente n", min_value=1.5, max_value=6.0, step=0.1, key="sb_n_exp"
            )
            k_const_str = c_ch2.text_input("Const. k", key="sb_k_const")
            try:
                k_const = max(1e-12, float(k_const_str))
            except ValueError:
                k_const = 1e-4
            d_0 = c_ch3.number_input("d₀ (m)", min_value=0.1, max_value=20.0, step=0.5, key="sb_d0")

        # ------------------------------------------------------------------
        # 4. Antenas & MIMO
        # ------------------------------------------------------------------
        with st.expander("📡 Antenas & MIMO", expanded=False):
            ant_models = st.multiselect(
                "Modelos de Antena",
                options=ANTENNA_OPTIONS,
                format_func=lambda k: MODEL_LABELS.get(k, k),
                key="sb_ant_models",
            )
            if not ant_models:
                ant_models = ["omni"]
                st.warning("Mantendo 'Omni' (selecione ao menos 1 modelo).")

            a1, a2 = st.columns(2)
            L_ant = a1.selectbox("Antenas MIMO (L)", options=[2, 4, 8, 16, 32, 64], key="sb_L")
            J_smb = a2.selectbox("Feixes SMB (J)", options=[4, 8, 12, 16, 24, 32], key="sb_J")

            a3, a4 = st.columns(2)
            aoa_err = a3.number_input(
                "Erro AoA σ_θ (°)", min_value=0.0, max_value=30.0, step=0.5, key="sb_aoa_err"
            )
            bw_deg = a4.number_input(
                "Abertura Setor (°)", min_value=30.0, max_value=360.0, step=10.0, key="sb_beamwidth"
            )
            g_min = st.number_input(
                "Ganho fora do setor (g_min)",
                min_value=0.001,
                max_value=0.5,
                step=0.005,
                format="%.3f",
                key="sb_g_min",
            )

        # ------------------------------------------------------------------
        # 5. Controle de Potência & Limites Físicos
        # ------------------------------------------------------------------
        with st.expander("⚡ Controle de Potência & Limites Físicos", expanded=False):
            pwr_algos = st.multiselect(
                "Algoritmos de Controle de Potência",
                options=POWER_OPTIONS,
                format_func=lambda k: MODEL_LABELS.get(k, k),
                key="sb_pwr_algos",
            )
            if not pwr_algos:
                pwr_algos = ["full_power"]
                st.warning("Mantendo 'Full Power' (selecione ao menos 1 algoritmo).")

            apply_phys = st.checkbox(
                "Aplicar Limites Físicos (Saturação QAM & Outage)",
                key="sb_phys_limits",
            )
            p1, p2 = st.columns(2)
            p_max_dbm = p1.number_input(
                "P_max (dBm)", min_value=0.0, max_value=50.0, step=1.0, key="sb_p_max"
            )
            p_min_dbm = p2.number_input(
                "P_min (dBm)", min_value=-40.0, max_value=30.0, step=1.0, key="sb_p_min"
            )
            p3, p4 = st.columns(2)
            mu_step = p3.number_input(
                "Passo μ (dB)", min_value=0.1, max_value=5.0, step=0.5, key="sb_mu"
            )
            gamma_tar = p4.number_input(
                "Meta γ_tar (dB)", min_value=-20.0, max_value=30.0, step=1.0, key="sb_gamma_tar"
            )
            g1, g2 = st.columns(2)
            gamma_max = g1.number_input(
                "Teto QAM (dB)", min_value=0.0, max_value=40.0, step=1.0, key="sb_gamma_max"
            )
            gamma_min = g2.number_input(
                "Outage γ_min (dB)", min_value=-30.0, max_value=10.0, step=1.0, key="sb_gamma_min"
            )
            l1, l2, l3 = st.columns(3)
            t_outer = l1.number_input("T_outer", min_value=1, max_value=200, step=5, key="sb_t_outer")
            t_inner = l2.number_input("T_inner", min_value=5, max_value=1000, step=5, key="sb_t_inner")
            tol_db = l3.number_input("tol (dB)", min_value=0.0, max_value=3.0, step=0.1, key="sb_tol_db")

        # ------------------------------------------------------------------
        # 6. Mobilidade, Tracking Preditivo, SDMA & Handover
        # ------------------------------------------------------------------
        with st.expander(
            "🚗 Mobilidade, Tracking Preditivo, SDMA & Handover",
            expanded=(ue_mode == "mobile"),
        ):
            vel_preset = st.selectbox(
                "Perfil de Velocidade",
                options=["Highway (120 km/h)", "Urban (50 km/h)", "Pedestrian (3 km/h)", "Customizado"],
                index=0,
                key="sb_mob_vel_preset",
            )
            if vel_preset == "Customizado":
                sel_v = st.number_input(
                    "Velocidade (km/h)", min_value=1.0, max_value=400.0, value=90.0, step=5.0
                )
            else:
                sel_v = MOBILITY_PROFILES_KMH[vel_preset.split(" ")[0]]

            m1, m2 = st.columns(2)
            sel_dt_ms = m1.selectbox("ΔT Principal (ms)", options=[100, 50, 10], index=0)
            sel_D = m2.selectbox("Distância D (m)", options=[25.0, 50.0, 75.0, 150.0], index=1)

            m3, m4 = st.columns(2)
            sel_mob_L = m3.selectbox("ULA Antenas (L)", options=[8, 16, 32], index=2)
            mob_sigma = m4.number_input(
                "Erro AoA σ_θ (°)", min_value=0.0, max_value=10.0, value=1.0, step=0.5
            )

            m5, m6, m7 = st.columns(3)
            mob_dt_ms = m5.number_input("dt (ms)", min_value=0.5, max_value=10.0, value=1.0, step=0.5)
            mob_t_sim = m6.number_input("T_sim (s)", min_value=0.5, max_value=10.0, value=4.0, step=0.5)
            mob_t_ovh_ms = m7.number_input(
                "T_ovh (ms)", min_value=0.0, max_value=10.0, value=1.5, step=0.5
            )

            compare_pred = st.checkbox(
                "Comparar Rastreamento Preditivo vs. Padrão (Hold)", value=True
            )
            enable_sdma = st.checkbox("Habilitar SDMA Multi-usuário Móvel", value=True)
            s1, s2 = st.columns(2)
            sdma_k = s1.number_input(
                "K_móvel (SDMA)", min_value=1, max_value=8, value=4, disabled=not enable_sdma
            )
            sdma_comb = s2.selectbox(
                "Combinador SDMA",
                options=["both", "mrc", "zf"],
                format_func=lambda k: {"both": "MRC vs. ZF", "mrc": "MRC", "zf": "Zero-Forcing"}[k],
                disabled=not enable_sdma,
            )

            st.markdown("---")
            st.caption("**Parâmetros de Handover (Hard vs. Soft)**")
            compare_ho = st.checkbox("Comparar Hard vs. Soft Handover", value=True)
            h1, h2 = st.columns(2)
            ho_aps = h1.number_input("APs na Rodovia", min_value=2, max_value=6, value=2, step=1)
            ho_dist = h2.number_input(
                "Distância Total (m)", min_value=500.0, max_value=10000.0, value=3100.0, step=100.0
            )
            h3, h4 = st.columns(2)
            ho_hyst = h3.number_input(
                "Histerese (dB)",
                min_value=0.0,
                max_value=15.0,
                value=0.0,
                step=1.0,
                help="0 dB = modo reativo fiel ao Exercício 13.",
            )
            ho_ttt = h4.number_input("TTT (ms)", min_value=0.0, max_value=5000.0, value=0.0, step=500.0)
            h5, h6 = st.columns(2)
            ho_as_margin = h5.number_input(
                "Margem Conj. Ativo (dB)",
                min_value=0.0,
                max_value=15.0,
                value=0.0,
                step=1.0,
                help="0 dB = seleção pura (Ex. 13); >0 dB habilita macro-diversidade.",
            )
            ho_runs = h6.number_input("Rodadas MC (HO)", min_value=50, max_value=5000, value=500, step=100)
            ho_soft_mode = "macro_diversity" if ho_as_margin > 0.0 else "selection"

        # ------------------------------------------------------------------
        # Montagem e validação do contrato `SimulationConfig`
        # ------------------------------------------------------------------
        if p_min_dbm > p_max_dbm:
            st.error("P_min deve ser ≤ P_max. Ajustando P_min = P_max.")
            p_min_dbm = p_max_dbm
        if gamma_min >= gamma_max:
            st.error("γ_min deve ser < γ_max. Ajustando γ_min = γ_max - 5.")
            gamma_min = gamma_max - 5.0

        snap_ant = st.session_state.get("tab1_ant_select", ant_models[0])
        snap_pwr = st.session_state.get("tab1_pwr_select", pwr_algos[-1])

        config = SimulationConfig(
            M=int(M),
            K=int(K_snap),
            k_scenarios=k_scenarios,
            N=int(N_ch),
            area_m=float(area_m),
            allocation_policy=str(alloc_policy),
            monte_carlo_runs=int(mc_runs),
            seed=int(seed),
            ue_mode=ue_mode,
            comparison_mode=str(comparison_mode),
            snapshot_antenna_model=str(snap_ant),
            snapshot_power_mode=str(snap_pwr),
            channel=ChannelConfig(
                use_shadowing=bool(use_shadowing),
                sigma_x=float(sigma_x),
                use_rayleigh=bool(use_rayleigh),
                sigma_r=float(sigma_r),
                pathloss_exp=float(n_exp),
                k_const=float(k_const),
                d_0=float(d_0),
                b_total_hz=float(bt_mhz) * 1e6,
            ),
            antenna=AntennaConfig(
                models=tuple(ant_models),
                L=int(L_ant),
                J=int(J_smb),
                aoa_err_deg=float(aoa_err),
                beamwidth_deg=float(bw_deg),
                g_min=float(g_min),
            ),
            power_control=PowerControlConfig(
                algorithms=tuple(pwr_algos),
                p_max_dbm=float(p_max_dbm),
                p_min_dbm=float(p_min_dbm),
                mu_step_db=float(mu_step),
                gamma_tar_init_db=float(gamma_tar),
                gamma_max_db=float(gamma_max),
                gamma_min_db=float(gamma_min),
                t_outer=int(t_outer),
                t_inner=int(t_inner),
                tol_db=float(tol_db),
                apply_physical_limits=bool(apply_phys),
            ),
            mobility=MobilityConfig(
                selected_velocity_kmh=float(sel_v),
                selected_delta_t_s=float(sel_dt_ms) / 1000.0,
                selected_distance_m=float(sel_D),
                selected_L=int(sel_mob_L),
                dt_s=float(mob_dt_ms) / 1000.0,
                t_sim_s=float(mob_t_sim),
                t_overhead_s=float(mob_t_ovh_ms) / 1000.0,
                sigma_theta_deg=float(mob_sigma),
                compare_predictive=bool(compare_pred),
                enable_sdma=bool(enable_sdma),
                sdma_k_ues=int(sdma_k),
                sdma_combiner=str(sdma_comb),
                seed=int(seed),
            ),
            handover=HandoverConfig(
                num_aps=int(ho_aps),
                dist_total_m=float(ho_dist),
                hysteresis_db=float(ho_hyst),
                active_set_margin_db=float(ho_as_margin),
                ttt_ms=float(ho_ttt),
                soft_combining=ho_soft_mode,
                compare_hard_soft=bool(compare_ho),
                mc_runs=int(ho_runs),
                seed=int(seed),
            ),
        )

        # ------------------------------------------------------------------
        # 7. Exportar / Importar Configuração JSON
        # ------------------------------------------------------------------
        with st.expander("💾 Configuração (JSON)", expanded=False):
            st.download_button(
                "📥 Baixar config.json",
                data=config.to_json(),
                file_name="simulator_config.json",
                mime="application/json",
                width="stretch",
            )
            uploaded = st.file_uploader("📂 Importar config.json", type=["json"])
            if uploaded is not None:
                raw_bytes = uploaded.getvalue()
                sig = (uploaded.name, len(raw_bytes), hash(raw_bytes))
                if st.session_state.get("_last_uploaded_json_sig") != sig:
                    try:
                        loaded = SimulationConfig.from_json(raw_bytes.decode("utf-8"))
                        st.session_state["_last_uploaded_json_sig"] = sig
                        st.session_state["_pending_config_to_apply"] = loaded
                        st.rerun()
                    except Exception as exc:
                        st.error(f"JSON inválido: {exc}")

    return config, run_mc_clicked, refresh_snap_clicked

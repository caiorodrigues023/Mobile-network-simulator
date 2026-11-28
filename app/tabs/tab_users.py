# ---------------------------------------------------------------------------
# app/tabs/tab_users.py — Aba 4: Tabela de UEs, Busca/Filtros e Inspetor de Rodadas
# ---------------------------------------------------------------------------

from __future__ import annotations

import pandas as pd
import streamlit as st

from simulator.config import MODEL_LABELS
from simulator.runner import run_single_snapshot
from simulator.schemas import MonteCarloResult, SimulationConfig, SnapshotResult


def _step_mc_round(delta: int, max_runs: int) -> None:
    cur = int(st.session_state.get("tab_users_round_num", 1))
    st.session_state["tab_users_round_num"] = max(1, min(max_runs, cur + delta))
    st.session_state["tab_users_origin"] = "custom"


def render_tab_users(
    config: SimulationConfig,
    snapshot: SnapshotResult,
    mc_result: MonteCarloResult | None,
) -> None:
    """
    Aba dedicada à inspeção detalhada de UEs e APs (trazida da branch `interface`),
    com busca textual, filtro por AP, filtro por status, seletor de rodadas do Monte Carlo
    (1ª, Última, Melhor, Mediana, Pior ou rodada #k) e exportação CSV.
    """
    st.subheader("📋 Tabela Detalhada de Usuários (UEs), APs & Inspetor de Rodadas")

    inspected_snap = snapshot
    source_label = f"Snapshot Atual (Seed={snapshot.seed})"

    # ------------------------------------------------------------------
    # Inspetor de Rodadas do Monte Carlo (quando disponível)
    # ------------------------------------------------------------------
    if mc_result is not None and mc_result.round_inspector:
        ri = mc_result.round_inspector
        presets = ri["presets"]
        n_runs = int(ri["n_runs"])
        ri_k = int(ri["K"])

        with st.container(border=True):
            st.caption(f"🔎 **Origem dos Dados** — Monte Carlo com {n_runs} rodadas (K = {ri_k} UEs)")
            r1, r2, r3, r4, r5 = st.columns([2.6, 0.5, 1.0, 0.5, 1.6])

            preset_opts = {
                "snapshot": f"📍 Snapshot Atual (Seed={snapshot.seed})",
                "first": f"1ª Rodada (#{presets['first']['round']}) — {presets['first']['sum_cap_mbps']:.1f} Mbps",
                "last": f"Última Rodada (#{presets['last']['round']}) — {presets['last']['sum_cap_mbps']:.1f} Mbps",
                "best": f"🏆 Melhor Rodada (#{presets['best']['round']}) — {presets['best']['sum_cap_mbps']:.1f} Mbps",
                "median": f"⚖️ Rodada Mediana (#{presets['median']['round']}) — {presets['median']['sum_cap_mbps']:.1f} Mbps",
                "worst": f"⚠️ Pior Rodada (#{presets['worst']['round']}) — {presets['worst']['sum_cap_mbps']:.1f} Mbps",
                "custom": "🔢 Rodada específica (#k)",
            }
            origin = r1.selectbox(
                "Origem",
                options=list(preset_opts.keys()),
                format_func=lambda k: preset_opts[k],
                label_visibility="collapsed",
                key="tab_users_origin",
            )

            if "tab_users_round_num" not in st.session_state:
                st.session_state["tab_users_round_num"] = 1
            elif st.session_state["tab_users_round_num"] > n_runs:
                st.session_state["tab_users_round_num"] = n_runs

            r2.button(
                "◀",
                help="Rodada anterior",
                width="stretch",
                key="tab_users_prev",
                on_click=_step_mc_round,
                args=(-1, n_runs),
            )
            round_k = r3.number_input(
                "Rodada #",
                min_value=1,
                max_value=n_runs,
                step=1,
                label_visibility="collapsed",
                key="tab_users_round_num",
            )
            r4.button(
                "▶",
                help="Próxima rodada",
                width="stretch",
                key="tab_users_next",
                on_click=_step_mc_round,
                args=(1, n_runs),
            )

            selected_seed: int | None = None
            if origin in presets:
                selected_seed = int(presets[origin]["seed"])
                source_label = f"MC {preset_opts[origin]}"
            elif origin == "custom":
                selected_seed = int(ri["all_seeds"][int(round_k) - 1])
                source_label = f"MC Rodada #{int(round_k)}/{n_runs} (Seed={selected_seed})"

            if selected_seed is not None:
                inspected_snap = run_single_snapshot(
                    config=config,
                    antenna_model=snapshot.antenna_model,
                    power_mode=snapshot.power_mode,
                    seed=selected_seed,
                    K_override=ri_k,
                )
                if r5.button("🗺️ Carregar no Mapa", type="primary", width="stretch"):
                    st.session_state["snap_seed_active"] = selected_seed
                    st.session_state["snap_k_override"] = ri_k
                    st.rerun()
            else:
                r5.button("🗺️ Já está no Mapa", disabled=True, width="stretch")

    # ------------------------------------------------------------------
    # Barra de busca e filtros
    # ------------------------------------------------------------------
    f1, f2, f3 = st.columns([2.4, 1.3, 1.3])
    search_q = f1.text_input(
        "🔍 Buscar",
        placeholder="Filtrar por UE ID, AP, Canal ou Status...",
        key="tab_users_search",
    ).strip().lower()

    ap_options = ["Todos os APs"] + [f"AP {ap.ap_id}" for ap in inspected_snap.aps]
    ap_filter = f2.selectbox("Filtrar por AP", options=ap_options, key="tab_users_ap_filter")
    status_filter = f3.selectbox(
        "Status",
        options=["Todos", "Apenas Ativos", "Apenas Outage"],
        key="tab_users_status_filter",
    )

    rows = []
    for u in inspected_snap.ues:
        ap_lbl = f"AP {u.serving_ap}"
        if ap_filter != "Todos os APs" and ap_lbl != ap_filter:
            continue
        if status_filter == "Apenas Ativos" and u.is_outage:
            continue
        if status_filter == "Apenas Outage" and not u.is_outage:
            continue

        row = {
            "UE": f"UE {u.ue_id}",
            "X (m)": round(u.x, 1),
            "Y (m)": round(u.y, 1),
            "AP Servidor": ap_lbl,
            "Distância (m)": round(u.distance_m, 1),
            "Azimute (°)": round(u.aoa_deg, 1),
            "Feixe": u.serving_beam,
            "Canal": "CA (Todos)" if u.is_aggregated else str(u.channel),
            "Potência (W)": round(u.tx_power_w, 4),
            "Potência (dBm)": None if u.is_blocked else round(u.tx_power_dbm, 1),
            "Ganho (dB)": round(u.rx_gain_db, 1),
            "SINR (dB)": None if u.is_blocked else round(u.sinr_db, 2),
            "SE (bps/Hz)": round(u.se_bps_hz, 2),
            "Capacidade (Mbps)": round(u.capacity_mbps, 2),
            "γ_tar (dB)": None if u.gamma_tar_db is None else round(u.gamma_tar_db, 1),
            "Status": u.status,
        }
        if search_q:
            blob = " ".join(str(v).lower() for v in row.values())
            if search_q not in blob:
                continue
        rows.append(row)

    df_ues = pd.DataFrame(rows)

    active_cnt = sum(1 for u in inspected_snap.ues if not u.is_outage)
    outage_cnt = len(inspected_snap.ues) - active_cnt
    st.caption(
        f"**[{source_label}]** | Modelo: **{MODEL_LABELS.get(inspected_snap.antenna_model, inspected_snap.antenna_model)}** "
        f"| Potência: **{MODEL_LABELS.get(inspected_snap.power_mode, inspected_snap.power_mode)}** | "
        f"Exibindo **{len(df_ues)}/{len(inspected_snap.ues)}** UEs ({active_cnt} ativos, {outage_cnt} outage) | "
        f"SINR Médio: **{inspected_snap.avg_sinr_db:+.1f} dB** | Sum-Capacity: **{inspected_snap.sum_capacity_mbps:.1f} Mbps**"
    )

    t_ue, t_ap = st.tabs(["📱 Tabela de Usuários (UEs)", "📡 Carga por Access Point (APs)"])
    with t_ue:
        st.dataframe(df_ues, width="stretch", hide_index=True)
        st.download_button(
            "📥 Exportar Tabela de UEs (CSV)",
            data=df_ues.to_csv(index=False).encode("utf-8"),
            file_name=f"ues_seed{inspected_snap.seed}.csv",
            mime="text/csv",
            width="stretch",
        )

    with t_ap:
        df_aps = pd.DataFrame(
            [
                {
                    "AP": f"AP {ap.ap_id}",
                    "X (m)": round(ap.x, 1),
                    "Y (m)": round(ap.y, 1),
                    "Qtd. UEs": ap.num_ues,
                    "Lista de UEs Conectados": (
                        ", ".join(f"UE {uid}" for uid in ap.connected_ue_ids)
                        if ap.connected_ue_ids
                        else "Nenhum"
                    ),
                    "Sum-Capacity (Mbps)": round(ap.sum_capacity_mbps, 2),
                    "UEs em Outage": ap.num_outage,
                }
                for ap in inspected_snap.aps
            ]
        )
        st.dataframe(df_aps, width="stretch", hide_index=True)
        st.download_button(
            "📥 Exportar Carga por AP (CSV)",
            data=df_aps.to_csv(index=False).encode("utf-8"),
            file_name=f"aps_seed{inspected_snap.seed}.csv",
            mime="text/csv",
            width="stretch",
        )

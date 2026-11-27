# 📡 Mobile Network Simulator

Simulador de redes móveis celulares desenvolvido no **GTEL – Grupo de Telecomunicações (UFC)**, estruturado em **arquitetura híbrida de alto desempenho** (contratos `@dataclass` + estratégias intercambiáveis + núcleo NumPy vetorizado) com **interface gráfica web interativa em Streamlit + Plotly** e suporte completo a execução via linha de comando (CLI).

---

## Recursos Principais

- 🗺️ **Tela 1 — Snapshot Interativo + Monte Carlo Integrado**:
  - Mapa interativo **2D (padrão)** e **3D WebGL (bônus opcional)** com lóbulos reais de antena (Omni, Setorial, SMB, MRC/UCA e ULA), enlaces UE↔AP, interferência co-canal e *heatmap* de cobertura (SINR, RSRP, Capacidade, AP dominante).
  - **Hover rico** em cada UE (ID, posição, AP servidor, distância, AoA, canal/CA, potência Tx, ganho, SINR, SE, capacidade, meta $\gamma_{\text{tar}}$ e status) e em cada AP (ID, posição, quantidade e lista exata de UEs conectados, Sum-Capacity da célula e outages), com `hoverinfo="skip"` nas linhas e lóbulos.
  - Execução de baterias de **Monte Carlo** exibindo o painel 2×2 de KPIs e CDFs diretamente abaixo do Snapshot, além de um **Inspetor de Rodadas** (melhor, mediana, pior ou rodada `#k`).
- 📈 **Aba 2 — CDFs Empíricas Comparativas**: Potência (Tx/Rx), SINR (dB) e Eficiência Espectral / Capacidade com linhas de referência (10º/50º percentis, $\gamma_{\min}$, $\gamma_{\text{tar}}$, $\gamma_{\max}$).
- 🏆 **Aba 3 — Painel Executivo de KPIs & Benchmark**: Painel 2×2 (Border SE 10º pct, Sum-SE, Eficiência Energética em escala log e Taxa de Outage), tabela completa (incluindo Índice de Jain e ASE) e exportação CSV/JSON.
- 🚗 **Aba 4 — Mobilidade, Beam Tracking Preditivo & SDMA Móvel**: Motor *time-driven* (`dt = 1 ms`) comparando **Zero-Order Hold ($\Delta T$)** vs. **Rastreamento Preditivo ($\alpha\text{-}\beta$)**, trade-off de Eficiência Espectral Líquida com *overhead* de treinamento e **SDMA Multi-usuário Móvel** (MRC vs. Zero-Forcing).
- 🔄 **Aba 5 — Handover (Hard vs. Soft)**: Travessia rodoviária entre APs com histerese (dB), *Time-to-Trigger* (ms), modo seleção ou macro-diversidade (*Active Set*), marcadores `"X"` de outage e contagem de *ping-pong*.
- 🔬 **Aba 6 — Convergência Passo a Passo**: Evolução por iteração de $P_k$ (dBm / mW), SINR real e meta negociada $\gamma_{\text{tar}}$ para *Dual-Loop PC*, *DPC Fixo*, *MaxSum* e *MaxProd*, destacando eventos de bloqueio (Outage).
- 💾 **Presets dos Roteiros & JSON**: Presets prontos para `SciChal4A` e Exercícios das Partes 2–4, além de exportar/importar configurações completas em JSON.

---

## Instalação

```bash
# Crie/ative um ambiente virtual (opcional, recomendado)
python -m venv venv
source venv/bin/activate  # Linux/macOS (ou venv\Scripts\activate no Windows)

# Instale as dependências e o pacote em modo editável
pip install -r requirements.txt
pip install -e .
```

---

## Como Executar

### 1. Interface Gráfica Web (Streamlit + Plotly)

```bash
streamlit run app/main_app.py
```

Acesse no navegador o endereço exibido no terminal (padrão `http://localhost:8501`).

### 2. Linha de Comando (CLI — Retrocompatibilidade Preservada)

```bash
python main.py
```

### 3. Uso Programático (API Híbrida)

```python
from simulator import SimulationConfig, run_single_snapshot, run_monte_carlo

cfg = SimulationConfig(
    M=9,
    K=4,
    k_scenarios=(1, 4, 8),
    N=1,
    monte_carlo_runs=500,
    comparison_mode="power",
    snapshot_antenna_model="mrc",
    snapshot_power_mode="dual_loop",
)

# Snapshot único instantâneo
snap = run_single_snapshot(cfg)
print(f"Sum-Capacity: {snap.sum_capacity_mbps:.2f} Mbps | Outage: {snap.outage_rate_pct:.1f}%")

# Bateria Monte Carlo vetorizada
mc = run_monte_carlo(cfg)
for algo, by_k in mc.power_metrics.items():
    sm = by_k[8]
    print(f"{algo:12s} | Border SE={sm.border_se_p10:.2f} | EE={sm.energy_efficiency_mbit_j:.1f} Mb/J | Outage={sm.outage_rate_pct:.1f}%")
```

---

## Estrutura do Projeto

Consulte [`ARCHITECTURE.md`](./ARCHITECTURE.md) para o diagrama detalhado em 4 camadas, contratos de dados (`schemas.py`), registros de estratégias e responsabilidades de cada módulo em `src/simulator/` e `app/`.

---

## Testes Automatizados e Qualidade

```bash
# Testes de sanidade física + testes E2E da interface Streamlit/Plotly
PYTHONPATH=src:. pytest -v

# Verificação de lint (Ruff)
ruff check src app tests
```

---

## Licença

Uso acadêmico — GTEL, Universidade Federal do Ceará.

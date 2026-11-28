# 📡 Mobile Network Simulator

Simulador de redes móveis celulares desenvolvido no **GTEL – Grupo de Telecomunicações (UFC)**, estruturado em **arquitetura híbrida de alto desempenho** (contratos `@dataclass` + estratégias intercambiáveis + núcleo NumPy vetorizado) com **interface gráfica web interativa em Streamlit + Plotly** e suporte completo a execução via linha de comando (CLI).

---

## Recursos Principais

- 🗺️ **Aba 1 — Snapshot Interativo & Visão Geral**:
  - Mapa interativo **2D (padrão)** e **3D WebGL (bônus opcional)** com lóbulos reais de antena (Omni, Setorial, SMB, MRC/UCA e ULA), enlaces UE↔AP, interferência co-canal e *heatmap* de cobertura.
  - Blocos **Desempenho Agregado** e **Qualidade de Sinal & Energia**, **hover rico** em cada UE e AP (`hoverinfo="skip"` em linhas/lóbulos) e **comparativo instantâneo no Snapshot** quando mais de uma antena ou algoritmo de potência estiver selecionado.
  - Execução automática de **Monte Carlo** exibindo os comparativos de todas as **Antenas** e/ou todos os **Algoritmos de Controle de Potência** selecionados diretamente abaixo do Snapshot.
- 📊 **Aba 2 — KPIs & Gráficos**: Grid 2×2 instantâneo do Snapshot (**SINR por UE**, **Capacidade por UE**, **Potência Tx por UE**, **Carga e Vazão por AP**) disponível antes mesmo do Monte Carlo + Benchmark Monte Carlo (Painel 2×2, **Ganho vs. Baseline em dB**, **Razão de Capacidade**, tabelas e exportação CSV/JSON).
- 📈 **Aba 3 — CDFs Comparativas**: CDFs empíricas lado a lado para cada modelo de antena e/ou algoritmo de potência selecionado por carga $K$, com linhas de referência (10º/50º pct, $\gamma_{\min}$, $\gamma_{\text{tar}}$, $\gamma_{\max}$).
- 📋 **Aba 4 — Tabela de UEs & Inspetor de Rodadas**: Busca textual, filtro por AP e por status, navegação pelas rodadas do Monte Carlo (`1ª`, `Última`, `Melhor`, `Mediana`, `Pior`, `#k` com `◀`/`▶`), carregamento direto no mapa e exportação CSV.
- 🔬 **Aba 5 — Convergência dos Algoritmos**: Evolução passo a passo de $P_k$ (dBm / mW), SINR por UE e meta negociada $\gamma_{\text{tar}}$ (*Dual-Loop PC*, *DPC Fixo*, *MaxSum* e *MaxProd*) destacando bloqueios (Outage).
- 🚗 **Aba 6 — Mobilidade, Beam Tracking & SDMA *(visível somente no modo "Em Movimento")***: Todos os $K$ UEs em movimento sobre a grade completa de $M$ APs, com mapa 2D de trajetórias, **curva de perda por desalinhamento angular $\Delta\theta$ ($L = 8, 16, 32$)**, evolução temporal **Hold vs. Tracking Preditivo** (agregado da rede ou por UE), *trade-off* de Sum-SE líquida e **SDMA (MRC vs. Zero-Forcing)**.
- 🔄 **Aba 7 — Handover Hard vs. Soft *(visível somente no modo "Em Movimento")***: Todos os $K$ UEs atravessando a grade de $M$ APs com mapa 2D de eventos de Handover (`◆`) e Outage (`×`), KPIs agregados da rede e inspeção detalhada de qualquer UE individual.
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

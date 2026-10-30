# 📡 Mobile Network Simulator

Simulador de redes móveis celulares desenvolvido no **GTEL – Grupo de Telecomunicações (UFC)**.
Compara o desempenho de quatro tecnologias de antena — **Omnidirecional, Setorial, SMB** (*Switched Multi-Beam*) e **MRC Adaptativo** — por meio de simulações de Monte Carlo.

---

## ✨ Funcionalidades

| Módulo | O que faz |
|---|---|
| **Omni** | Antena omnidirecional de referência (baseline) |
| **Setorial** | 3 setores de 120°, ganho fixo por setor |
| **SMB** | Arranjo circular uniforme (UCA) com `J` feixes pré-definidos |
| **MRC Adaptativo** | Beamforming adaptativo em tempo real; suporta erro de estimativa de AoA |

**KPIs calculados:** SINR por UE, Capacidade de Shannon (Mbps), Soma-capacidade por célula.

**Visualizações:** mapas 2D de cobertura, CDFs, gráficos de barras comparativos e padrões polares SMB.

---

## 📂 Estrutura do Projeto

```text
mobile-network-simulator/
├── src/
│   └── simulator/
│       ├── __init__.py        # Pacote Python — re-exporta todos os símbolos
│       ├── config.py          # Constantes físicas e paletas de cor
│       ├── entities.py        # Classes AP e UE
│       ├── antennas.py        # Modelos de antena (Omni, Setorial, SMB, MRC)
│       ├── channel.py         # Path loss, shadowing log-normal e geometria
│       ├── allocation.py      # Attach UE↔AP, alocação de canais e DPC
│       ├── metrics.py         # SINR, SINR_adaptive, capacidade de Shannon
│       ├── runner.py          # Motor Monte Carlo e posicionamento de APs
│       └── visualization.py  # Mapas 2D, CDFs e gráficos de KPI
├── tests/                     # Testes unitários (Pytest)
├── pyproject.toml             # Configuração do pacote (pip install -e .)
├── main.py                    # Ponto de entrada do simulador
├── scichal.py                 # Código monolítico original (referência)
└── ARCHITECTURE.md            # Guia de arquitetura detalhado
```

---

## 🚀 Instalação

### Pré-requisitos

- Python ≥ 3.9
- [`uv`](https://docs.astral.sh/uv/) (gerenciador de ambientes, recomendado) **ou** `pip`

### Com `uv` (recomendado)

```bash
# 1. Clonar o repositório
git clone https://github.com/SEU-USUARIO/mobile-network-simulator.git
cd mobile-network-simulator

# 2. Criar e ativar o ambiente virtual
uv venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 3. Instalar o pacote em modo editável (alterações no código refletem imediatamente)
uv pip install -e .
```

### Com `pip`

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

---

## ▶️ Uso

### Executar o simulador

```bash
python main.py
```

Saída esperada (exemplo com `M=4` APs, `N=4` canais, `sim=5` iterações):

```
=== Simulador de Redes Moveis ===

[K=1] Simulando 5 iteracoes com M=4 APs, N=4 canais ...
[K=4] Simulando 5 iteracoes com M=4 APs, N=4 canais ...

Simulacao concluida.

--- K=1 ---
  omni          SINR 10th=-15.0 dB  Cap 10th= 1.13 Mbps  SumCap= 17.88 Mbps
  sector        SINR 10th=-15.0 dB  Cap 10th= 1.13 Mbps  SumCap= 17.88 Mbps
  smb           SINR 10th= -6.7 dB  Cap 10th= 6.64 Mbps  SumCap= 43.48 Mbps

--- K=4 ---
  omni          SINR 10th=-19.7 dB  Cap 10th= 0.38 Mbps  SumCap= 69.09 Mbps
  sector        SINR 10th=-19.5 dB  Cap 10th= 0.40 Mbps  SumCap= 79.31 Mbps
  smb           SINR 10th= -9.5 dB  Cap 10th= 3.85 Mbps  SumCap=158.32 Mbps
```

### Habilitar gráficos interativos

Edite `main.py` e:

1. **Remova** a linha `matplotlib.use('Agg')` no início do arquivo.
2. **Descomente** os blocos de plotagem no final do arquivo.

### Usar como biblioteca

```python
from simulator import simular_experimento, plotar_mapa, cenario_aleatorio

# Experimento Monte Carlo
resultados = simular_experimento(
    M=4,            # APs (quadrado perfeito)
    N=4,            # canais ortogonais
    K=8,            # UEs por iteração
    sim=100,        # iterações Monte Carlo
    allocation='ortogonal',
    antenna_models=('omni', 'sector', 'smb'),
)

# Mapa de um cenário representativo
aps, ues, shadowing = cenario_aleatorio(M=4, K=8, seed=42)
plotar_mapa(aps, ues, shadowing=shadowing, models=['omni', 'sector', 'smb'])
```

---

## 🔬 Modelos de Antena

### Omnidirecional (`omni`)
Ganho unitário em todas as direções. Usado como *baseline* de comparação.

### Setorial (`sector`)
Três setores de 120° (boresights em 0°, 120° e 240°). Ganho máximo dentro do setor, `g_min` fora.

### SMB — *Switched Multi-Beam* (`smb`)
Arranjo circular uniforme (UCA) de `L` elementos com `J` feixes fixos.
O UE é associado ao feixe de maior ganho no instante do *initial access*.

$$G_k(\varphi) = \left| \mathbf{w}_k^T \mathbf{a}(\varphi) \right|^2, \quad \mathbf{w}_k = \frac{\mathbf{a}^*(\theta_k)}{\|\mathbf{a}(\theta_k)\|}$$

### MRC Adaptativo (`adaptive`, `adaptive_1`, `adaptive_5`, `adaptive_10`)
Beamforming em tempo real: o feixe é formado exatamente na direção do UE.
O sufixo indica o desvio-padrão do erro de estimativa de AoA em graus
(ex.: `adaptive_5` → erro gaussiano $\sigma = 5°$).

---

## ⚙️ Parâmetros Principais

| Parâmetro | Descrição | Padrão em `main.py` |
|---|---|---|
| `M` | Número de APs (deve ser quadrado perfeito) | `4` |
| `N` | Canais ortogonais disponíveis | `4` |
| `K` | UEs por iteração de Monte Carlo | `1` ou `4` |
| `sim` | Número de iterações Monte Carlo | `5` |
| `L` | Elementos do UCA (SMB/MRC) | `8` |
| `J` | Feixes do SMB | `8` |
| `allocation` | `'random'` ou `'ortogonal'` | `'ortogonal'` |

---

## 🧪 Testes

```bash
pytest
```

---

## 📖 Referências

- R. Agarwal *et al.*, **"Switched Multi-Beam Antenna Systems"**, IEEE.
- T. Rappaport, *Wireless Communications: Principles and Practice*, Prentice Hall.
- Roteiro de laboratório — GTEL/UFC.

---

## 🤝 Contribuindo

1. Nunca faça `commit` direto em `main`.
2. Crie uma branch temática: `feature/`, `fix/`, `docs/`.
3. Rode `pytest` antes de abrir um *Pull Request*.

---

## 📄 Licença

Uso acadêmico — GTEL, Universidade Federal do Ceará.
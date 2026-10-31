# 📡 Mobile Network Simulator

Simulador de redes móveis celulares desenvolvido no **GTEL – Grupo de Telecomunicações (UFC)**.
Compara o desempenho de quatro tecnologias de antena — **Omnidirecional, Setorial, SMB** (*Switched Multi-Beam*) e **MRC Adaptativo** — por meio de simulações de Monte Carlo, e avalia o impacto da **mobilidade** do UE no beamforming adaptativo.

---

## Estrutura do Projeto

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
│       ├── mobility.py        # Simulação de mobilidade (beam tracking)
│       └── visualization.py   # Mapas 2D, CDFs, KPIs e gráficos de mobilidade
├── figures/                   # Imagens geradas (ignoradas pelo Git)
│   └── mobilidade/            # Gráficos do estudo de mobilidade
├── pyproject.toml             # Configuração do pacote (pip install -e .)
├── main.py                    # Ponto de entrada do simulador
├── scichal.py                 # Código original (referência)
└── ARCHITECTURE.md            # Guia de arquitetura detalhado
```

> Para entender como os módulos dependem uns dos outros, consulte o diagrama em [`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## Como Usar o Simulador

### 1. Executar o script principal

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

### 2. Ajustar os parâmetros da simulação

Abra `main.py` e edite o bloco de parâmetros no topo:

```python
M   = 4       # APs (deve ser quadrado perfeito: 4, 9, 16, 25 …)
N   = 4       # canais ortogonais
SIM = 100     # iterações Monte Carlo (aumente para resultados estatísticos)

ANTENNA_MODELS = ('omni', 'sector', 'smb')
LOAD_CONFIGS   = {'K=1': 1, 'K=4': 4, 'K=8': 8}  # label → nº de UEs
```

| Parâmetro | Descrição | Observação |
|---|---|---|
| `M` | Número de APs | Deve ser quadrado perfeito |
| `N` | Canais ortogonais disponíveis | Aumentar reduz interferência |
| `K` | UEs por iteração | Aumentar aumenta a carga |
| `SIM` | Iterações Monte Carlo | ~ 10000 |
| `L` | Elementos do UCA (SMB/MRC) | Passado em `simular_experimento()` |
| `J` | Feixes pré-definidos do SMB | Passado em `simular_experimento()` |
| `allocation` | `'random'` ou `'ortogonal'` | Ortogonal reduz interferência intra-célula |

### 3. Habilitar gráficos interativos

No `main.py`:

1. **Remova** a linha `matplotlib.use('Agg')` no início do arquivo (ela desativa a janela gráfica).
2. **Descomente** os blocos de plotagem no final do arquivo:

```python
# Descomente conforme necessário:
plot_all_cdfs(resultados_por_carga, list(LOAD_CONFIGS.keys()), models=list(ANTENNA_MODELS))
compare_kpis(resultados_por_carga, list(LOAD_CONFIGS.keys()), models=list(ANTENNA_MODELS))

aps, ues, shadowing = cenario_aleatorio(M=M, K=4, seed=42)
plotar_mapa(aps, ues, shadowing=shadowing, models=list(ANTENNA_MODELS), show=True)
```

### Onde ficam as figuras

Todo `savepath` relativo passado às funções `plot_*` / `plotar_mapa` é salvo dentro de `figures/` (pasta definida por `FIGURES_DIR` em `config.py`, criada automaticamente, subpastas inclusive). Ex.: `plotar_mapa(..., savepath='mapas/cenario1.png')` → `figures/mapas/cenario1.png`. Caminhos absolutos são respeitados.

### 4. Simulação de mobilidade (beam tracking)

Um UE cruza a célula tangencialmente ao broadside do arranjo; o AP atualiza os pesos MRC a cada `T` ms e o UE se move entre atualizações, gerando o erro total
$\varphi_{total} = \varepsilon_{est} + \varphi_{lag}(v, T)$.

Pelo `main.py`, com `RUN_MOBILITY = True` (salva em `figures/mobilidade/`: `ganho_vs_tempo.png`, `perda_desalinhamento.png`, `perda_por_arranjo.png` e `tradeoff_se.png`). Uso como biblioteca:

```python
from simulator import (
    simular_trajetoria, estudo_mobilidade, tabela_kpis,
    plot_ganho_vs_tempo, plot_perda_desalinhamento,
    plot_perda_por_arranjo, plot_tradeoff_se,
)

# Um cenário: 120 km/h, a 25 m do AP, T=50 ms, L=32
r = simular_trajetoria(v_kmh=120, d_perp=25, T_ms=50, L=32, seed=42)
print(r['loss_medio_db'], r['se_net_medio'])

# Grid completo (velocidade × T × L × distância)
res = estudo_mobilidade(seed=42)
tabela_kpis(res)
plot_ganho_vs_tempo(res, L=16, d_perp=100, janela_ms=600)
plot_perda_desalinhamento(res, L=16, d_perp=100, janela_ms=600)
plot_tradeoff_se(res, d_perp=100)
```

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `v_kmh` | 3 / 50 / 120 | Velocidade do UE (pedestre / urbano / rodovia) |
| `T_ms` | 100 / 50 / 10 | Período de atualização dos pesos MRC |
| `L` | 8 / 16 / 32 | Elementos do arranjo |
| `d_perp` | 25 / 100 / 250 m | Distância perpendicular da trajetória ao AP |
| `dt_ms` | 1 | Passo de simulação |
| `sigma_est_deg` | 1 | Desvio-padrão do erro de estimação de AoA |
| `t_pilot_ms` | 0,5 | Tempo de treino por atualização (overhead = `t_pilot/T`) |
| `seed` | 0 | Semente reiniciada a cada cenário (mesmas realizações) |

### 5. Usar como biblioteca em um notebook ou script próprio

O pacote pode ser importado diretamente após o `pip install -e .`:

```python
import numpy as np
from simulator import (
    simular_experimento,
    cenario_aleatorio,
    plotar_mapa,
    plot_all_cdfs,
    compare_kpis,
    plot_padroes_smb,
)

# --- Experimento Monte Carlo ---
resultados = simular_experimento(
    M=9,                    # grade 3×3 de APs
    N=4,
    K=8,
    sim=200,
    allocation='ortogonal',
    antenna_models=('omni', 'sector', 'smb', 'adaptive', 'adaptive_5'),
    L=8, J=8,
)

# --- KPI do modelo SMB ---
sinr_p10 = np.percentile(resultados['smb']['sinr'], 10)
print(f"SMB SINR 10th pct: {10*np.log10(sinr_p10):.1f} dB")

# --- Mapa de cobertura reproduzível ---
aps, ues, shadowing = cenario_aleatorio(M=9, K=8, seed=7)
plotar_mapa(aps, ues, shadowing=shadowing,
            models=['omni', 'sector', 'smb'], N=4, show=True)

# --- CDFs de todos os KPIs ---
resultados_por_carga = {'omni': {'K=8': resultados['omni']},
                        'smb':  {'K=8': resultados['smb']}}
plot_all_cdfs(resultados_por_carga, ['K=8'])

# --- Padrões polares SMB ---
plot_padroes_smb(configs=[(8, 8), (8, 16), (16, 8)])
```

### 6. Guia rápido de funções públicas

| Função | Módulo | Descrição |
|---|---|---|
| `simular_experimento()` | `runner` | Executa N rodadas de Monte Carlo |
| `cenario_aleatorio()` | `runner` | Gera APs, UEs e shadowing reproduzíveis |
| `distribuir_AP()` | `runner` | Posiciona APs em grade uniforme |
| `antenna_gain()` | `antennas` | Ganho G[i, j, b] para qualquer modelo |
| `gain_matrix()` | `channel` | Path loss + shadowing + ganho de antena |
| `attach_AP_UE()` | `allocation` | Conecta cada UE ao melhor (AP, feixe) |
| `alocar_canais_ortogonal()` | `allocation` | Distribui canais por célula |
| `SINR()` | `metrics` | SINR para antenas convencionais |
| `SINR_adaptive()` | `metrics` | SINR com beamforming MRC |
| `channel_capacity()` | `metrics` | Capacidade de Shannon em Mbps |
| `simular_trajetoria()` | `mobility` | Um UE cruzando a célula: ganho, perda e SE no tempo |
| `estudo_mobilidade()` | `mobility` | Grid velocidade × T × L × distância |
| `tabela_kpis()` | `metrics` | Tabela de KPIs médios por cenário |
| `plot_ganho_vs_tempo()` | `visualization` | Array Gain × tempo |
| `plot_perda_desalinhamento()` | `visualization` | Beam Misalignment Loss × tempo |
| `plot_perda_por_arranjo()` | `visualization` | Perda média × T para cada L |
| `plot_tradeoff_se()` | `visualization` | SE líquida × período de atualização |
| `plotar_mapa()` | `visualization` | Mapa 2D com UEs, APs e feixes |
| `plot_all_cdfs()` | `visualization` | CDFs de potência, SINR e capacidade |
| `compare_kpis()` | `visualization` | Barras comparativas entre modelos |
| `plot_padroes_smb()` | `visualization` | Padrões polares do arranjo SMB |

---

## Modelos de Antena

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
O sufixo numérico indica o desvio-padrão do erro de estimativa de AoA em graus
(ex.: `adaptive_5` → erro gaussiano $\sigma = 5°$).
Na simulação de mobilidade, o feixe é atualizado apenas a cada `T` ms (ver seção 4).

---

## Guia de uso

### Fluxo de trabalho Git

```
main  ←── pull request ←── feature/minha-funcionalidade
  │
  └── nunca recebe commits diretos
```

**Passo a passo:**

```bash
# 1. Sempre parta do main atualizado
git checkout main
git pull origin main

# 2. Crie uma branch com nome descritivo
git checkout -b feature/modelo-canal-3gpp
#             fix/correcao-calculo-sinr
#             docs/atualiza-readme
#             refactor/renomeia-funcoes

# 3. Faça suas alterações e commits atômicos
git add src/simulator/channel.py
git commit -m "feat(channel): adiciona modelo de path loss 3GPP UMa"

# 4. Envie a branch e abra o Pull Request
git push origin feature/modelo-canal-3gpp
```

### Onde implementar cada tipo de mudança

| Tipo de alteração | Arquivo a editar |
|---|---|
| Nova constante física ou cor de gráfico | `src/simulator/config.py` |
| Novo campo na classe AP ou UE | `src/simulator/entities.py` |
| Novo modelo de antena ou padrão de feixe | `src/simulator/antennas.py` |
| Novo modelo de canal (path loss, shadowing) | `src/simulator/channel.py` |
| Nova política de alocação de canal ou potência | `src/simulator/allocation.py` |
| Nova métrica de desempenho | `src/simulator/metrics.py` |
| Novo tipo de experimento ou topologia de rede | `src/simulator/runner.py` |
| Novo cenário de mobilidade, perfil de velocidade ou política de tracking | `src/simulator/mobility.py` |
| Novo tipo de gráfico ou mapa (inclusive de mobilidade) | `src/simulator/visualization.py` |

### Adicionando um novo modelo de antena

1. **Implemente a função de ganho** em `src/simulator/antennas.py`:

```python
def meu_modelo_gains(phi, **kwargs) -> np.ndarray:
    """Retorna ganho shape (n_ues, n_aps, n_beams)."""
    ...
```

2. **Registre-o no dispatcher** `antenna_gain()` no mesmo arquivo:

```python
if model == 'meu_modelo':
    return meu_modelo_gains(phi, **kwargs)
```

3. **Adicione o rótulo e a cor** em `src/simulator/config.py`:

```python
MODEL_LABELS['meu_modelo'] = 'Meu Modelo'
MODEL_COLORS['meu_modelo'] = 'tab:cyan'
```

4. **Use-o na simulação** passando a string no `antenna_models`:

```python
simular_experimento(..., antenna_models=('omni', 'smb', 'meu_modelo'))
```

### Adicionando uma nova métrica

1. **Implemente em `src/simulator/metrics.py`:**

```python
def minha_metrica(ues: list, resultados: dict, **kwargs):
    """Calcula e retorna a nova métrica."""
    ...
```

2. **Colete os dados em `runner.py`** dentro do loop de modelos:

```python
resultados[model]['minha_metrica'].append(minha_metrica(ues, ...))
```

- Ao comparar modelos, **use o mesmo shadowing** pré-sorteado para garantir condições justas.
- Ao comparar cenários de mobilidade, **reinicie a semente** antes de cada cenário (já feito em `simular_trajetoria()`).

## Licença

Uso acadêmico — GTEL, Universidade Federal do Ceará.

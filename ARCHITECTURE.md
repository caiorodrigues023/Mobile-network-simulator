# Guia de Arquitetura do Simulador de Redes Móveis

Este documento descreve a estrutura modular do projeto, as responsabilidades de cada arquivo, o fluxo de dados durante uma simulação e as instruções para que novos colaboradores possam configurar e trabalhar no projeto sem conflitos.

---

## Estrutura de Diretórios

```text
mobile-network-simulator/
├── src/
│   └── simulator/
│       ├── __init__.py           # Transforma a pasta em um pacote Python exportável
│       ├── config.py             # Constantes físicas, globais e paletas visuais
│       ├── entities.py           # Abstração de hardware: AP (Access Point) e UE (User Equipment)
│       ├── antennas.py           # Modelos e diagramas de irradiação (Omni, Setorial, SMB, MRC)
│       ├── channel.py            # Modelagem de propagação: Path Loss, Shadowing e Geometria
│       ├── allocation.py         # Acesso inicial (Attach), Alocação de Canais e DPC
│       ├── metrics.py            # SINR (estático e adaptativo), Capacidade de Shannon e tabela de KPIs
│       ├── runner.py             # Motor do experimento: orquestra rodadas e distribui APs
│       ├── mobility.py           # Mobilidade: simulação de trajetória, lag de beam tracking e KPIs
│       └── visualization.py      # Geração de mapas 2D, gráficos CDF, comparações de KPI e gráficos de mobilidade
├── figures/                      # Imagens geradas pelo simulador (conteúdo ignorado pelo Git)
├── pyproject.toml                # Configuração de empacotamento e dependências (pip install -e .)
├── requirements.txt              # Bibliotecas externas necessárias
├── .gitignore                    # Arquivos ignorados pelo Git
├── main.py                       # Ponto de entrada (Entrypoint) do simulador
└── ARCHITECTURE.md               # Este guia
```

---

## Responsabilidade de Cada Arquivo

### 1. `src/simulator/config.py`
* **Função**: Centralizar constantes físicas e configurações de interface gráfica.
* **Conteúdo**:
  * Comprimento de onda ($\lambda = 1.0$).
  * `FIGURES_DIR`: pasta de saída das figuras (`<raiz>/figures`); `visualization._salvar()` resolve nela os `savepath` relativos.
  * Dicionários `MODEL_LABELS` e `MODEL_COLORS` para padronizar rótulos e cores nos gráficos.

### 2. `src/simulator/entities.py`
* **Função**: Definir os objetos da rede.
* **Conteúdo**:
  * `AP`: Armazena posição $(x, y)$, ID, canal alocado e lista de UEs associados.
  * `UE`: Armazena posição $(x, y)$, ID único, AP servidor, feixe ativo, canal, potência de transmissão ($P_t$) e histórico de interferência.

### 3. `src/simulator/antennas.py`
* **Função**: Calcular os ganhos de irradiação para diferentes tecnologias de antena.
* **Conteúdo**:
  * `smb_gains()`: Switched Multi-Beam com arranjo circular uniforme (UCA).
  * `mrc_gain()`: Beamforming adaptativo com algoritmo Maximum Ratio Combining.
  * `antenna_gain()`: Matriz tridimensional $G[i, j, b]$ indexada por UE $i$, AP $j$ e setor/feixe $b$.

### 4. `src/simulator/channel.py`
* **Função**: Simular a propagação de ondas eletromagnéticas e geometria espacial.
* **Conteúdo**:
  * `angle_matrix()`: Calcula matriz de ângulos de azimute entre todos os UEs e APs.
  * `gerar_shadowing()`: Sorteia variações do canal log-normal.
  * `gain_matrix()`: Combina perda de percurso (Path Loss $d^{-4}$), shadowing e ganho de antena.

### 5. `src/simulator/allocation.py`
* **Função**: Gerenciar procedimentos de controle da camada de enlace e acesso ao meio.
* **Conteúdo**:
  * `attach_AP_UE()`: Associação do UE ao par (AP, Feixe) que oferece o maior ganho.
  * `alocar_canais_ortogonal()`: Distribuição de canais intra-célula (aleatória ou ortogonal).
  * `DPC()`: Algoritmo de Controle Distribuído de Potência.

### 6. `src/simulator/metrics.py`
* **Função**: Avaliação do desempenho do sistema sem fio.
* **Conteúdo**:
  * `SINR()`: Calcula Razão Sinal-Interferência-mais-Ruído para antenas convencionais/SMB.
  * `SINR_adaptive()`: Calcula SINR para feixes adaptativos MRC com filtro de interferência espacial.
  * `channel_capacity()`: Aplica a equação de Shannon-Hartley: $C = B \log_2(1 + \text{SINR})$.
  * `tabela_kpis()`: Imprime os KPIs médios (perda de desalinhamento e eficiência espectral) de cada cenário de `mobility.estudo_mobilidade()`.

### 7. `src/simulator/runner.py`
* **Função**: Gerenciar o ciclo de vida da simulação de Monte Carlo.
* **Conteúdo**:
  * `distribuir_AP()`: Posiciona APs em grade uniforme na área de simulação.
  * `simular_experimento()`: Executa $N$ iterações, variando posições de UEs e shadowing, consolidando os KPIs em dicionários.

### 8. `src/simulator/mobility.py`
* **Função**: Avaliar o impacto da mobilidade do UE no beamforming adaptativo (simulação dinâmica com passo de tempo $dt$, ao contrário dos *snapshots* estáticos do `runner.py`).
* **Modelo**: o AP atualiza os pesos MRC a cada $T$ ms; entre atualizações o UE se move, gerando o erro angular $\varphi_{total} = \varepsilon_{est} + \varphi_{lag}(v, T)$, com $\varepsilon_{est} \sim \mathcal{N}(0, 1°)$.
* **Conteúdo**:
  * `simular_trajetoria()`: Um UE cruzando a célula tangencialmente ao broadside, com velocidade $v$, distância perpendicular $d_\perp$, período $T$ e arranjo de $L$ elementos. Devolve ganho de arranjo, perda de desalinhamento (dB), SNR e eficiência espectral (bruta e líquida de overhead $t_{pilot}/T$). Reinicia a semente a cada cenário.
  * `estudo_mobilidade()`: Varre velocidades (3, 50, 120 km/h) × $T$ (100, 50, 10 ms) × $L$ (8, 16, 32) × distâncias.
* **KPIs e gráficos**: a tabela de KPIs (`tabela_kpis()`) fica em `metrics.py` (seção 6) e os plots em `visualization.py` (seção 9); `mobility.py` não depende de matplotlib.

### 9. `src/simulator/visualization.py`
* **Função**: Plotar e exportar resultados.
* **Conteúdo**:
  * `plotar_mapa()`: Desenha o cenário 2D com as linhas de conexão, APs, UEs e formatos dos feixes ativos.
  * `plot_all_cdfs()`: Desenha as Distribuições Acumuladas de Probabilidade para Potência, SINR e Capacidade.
  * `compare_kpis()`: Gera gráficos de barras comparando ganhos em dB e razões de capacidade em relação à antena Omnidirecional.
  * `plot_ganho_vs_tempo()`, `plot_perda_desalinhamento()`, `plot_perda_por_arranjo()`, `plot_tradeoff_se()`: Gráficos de mobilidade (Array Gain × tempo, Beam Misalignment Loss, perda média × T e trade-off de eficiência espectral), alimentados pelo resultado de `mobility.estudo_mobilidade()`.

---

### Fluxo de Dados Durante Uma Rodada de Simulação

1. **Grade & Entidades**: `runner.py` chama `distribuir_AP()` (`entities.py`) para criar as estações rádio-base.
2. **Posicionamento de UEs**: `runner.py` instancia os `UE`s (`entities.py`) em posições aleatórias válidas.
3. **Cálculo de Antena**: `runner.py` chama `antenna_gain()` (`antennas.py`) para computar o ganho angular.
4. **Matriz de Canal**: `channel.py` combina perda por distância, shadowing e ganhos para gerar a matriz de ganho final.
5. **Associação (Attach)**: `allocation.py` conecta cada UE ao AP com melhor ganho e seleciona o feixe ativo.
6. **Alocação de Canais**: `allocation.py` distribui os canais de radiofrequência entre os UEs.
7. **Métricas (SINR/Capacidade)**: `metrics.py` lê as posições, canais e potências para calcular o $SINR$ e a capacidade de Shannon em Mbps.
8. **Visualização**: `runner.py` devolve os dicionários de dados consolidados para o `main.py`, que invoca as rotinas de gráfico do `visualization.py`.

### Fluxo de Dados da Simulação de Mobilidade

Fluxo independente do Monte Carlo acima (um único UE e um único AP, sem attach nem alocação de canais):

1. **Trajetória**: `mobility.simular_trajetoria()` gera a posição $y(t)$ do UE e o AoA verdadeiro $\varphi(t)$ a cada $dt$.
2. **Tracking**: a cada $T$ ms o feixe é reapontado para $\varphi(t_{upd}) + \varepsilon_{est}$.
3. **Ganho**: `antennas.mrc_gain()` avalia o ganho real do feixe desatualizado na direção verdadeira; a perda é a diferença para o ganho ideal $L$.
4. **Enlace**: path loss $d^{-4}$, ruído térmico e teto de SINR de 20 dB resultam na eficiência espectral bruta; o overhead de atualização gera a líquida.
5. **Visualização**: `estudo_mobilidade()` consolida os cenários e o `main.py` chama as funções `plot_*` de mobilidade em `visualization.py`.

---

# Guia de Arquitetura do Simulador de Redes Móveis

Este documento descreve a estrutura modular do projeto, as responsabilidades de cada arquivo, o fluxo de dados durante uma simulação e as instruções para que novos colaboradores possam configurar e trabalhar no projeto sem conflitos.

---

## 📂 Estrutura de Diretórios

```text
mobile-network-simulator/
├── src/
│   └── simulator/
│       ├── __init__.py           # Transforma a pasta em um pacote Python exportável
│       ├── config.py             # Constantes físicas, globais e paletas visuais
│       ├── entities.py           # Abstração de hardware: AP (Access Point) e UE (User Equipment)
│       ├── antennas.py           # Modelos e diagramas de irradiação (Omni, Setorial, SMB, MRC)
│       ├── channel.py            # Modelagem de propagação: Path Loss, Shadowing e Geometria
│       ├── allocation.py       # Acesso inicial (Attach), Alocação de Canais e DPC
│       ├── metrics.py            # Cálculo de SINR (estático e adaptativo) e Capacidade de Shannon
│       ├── runner.py             # Motor do experimento: orquestra rodadas e distribui APs
│       └── visualization.py    # Geração de mapas 2D, gráficos CDF e comparações de KPI
├── tests/                        # Testes unitários automatizados com Pytest
│   ├── test_antennas.py
│   └── test_metrics.py
├── pyproject.toml                # Configuração de empacotamento e dependências (pip install -e .)
├── requirements.txt              # Bibliotecas externas necessárias
├── .gitignore                    # Arquivos ignorados pelo Git
├── main.py                       # Ponto de entrada (Entrypoint) do simulador
└── ARCHITECTURE.md               # Este guia
```

---

## 📄 Responsabilidade de Cada Arquivo

### 1. `src/simulator/config.py`
* **Função**: Centralizar constantes físicas e configurações de interface gráfica.
* **Conteúdo**:
  * Comprimento de onda ($\lambda = 1.0$).
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

### 7. `src/simulator/runner.py`
* **Função**: Gerenciar o ciclo de vida da simulação de Monte Carlo.
* **Conteúdo**:
  * `distribuir_AP()`: Posiciona APs em grade uniforme na área de simulação.
  * `simular_experimento()`: Executa $N$ iterações, variando posições de UEs e shadowing, consolidando os KPIs em dicionários.

### 8. `src/simulator/visualization.py`
* **Função**: Plotar e exportar resultados.
* **Conteúdo**:
  * `plotar_mapa()`: Desenha o cenário 2D com as linhas de conexão, APs, UEs e formatos dos feixes ativos.
  * `plot_all_cdfs()`: Desenha as Distribuições Acumuladas de Probabilidade para Potência, SINR e Capacidade.
  * `compare_kpis()`: Gera gráficos de barras comparando ganhos em dB e razões de capacidade em relação à antena Omnidirecional.

---

## 🔗 Como os Arquivos se Conectam (Mapa de Dependências)

Abaixo está o diagrama mostrando como os módulos importam recursos uns dos outros:

```text
               +-------------------+
               |     config.py     |
               +---------+---------+
                         |
      +------------------+------------------+
      |                  |                  |
      v                  v                  v
+-----------+     +-------------+    +---------------+
|entities.py|     | antennas.py |    |visualization.py
+-----+-----+     +------+------+    +-------+-------+
      |                  |                   |
      +--------+---------+                   |
               |                             |
               v                             |
       +---------------+                     |
       |  channel.py   |                     |
       +-------+-------+                     |
               |                             |
               v                             |
      +-----------------+                    |
      |  allocation.py  |                    |
      +--------+--------+                    |
               |                             |
               v                             |
      +-----------------+                    |
      |   metrics.py    |                    |
      +--------+--------+                    |
               |                             |
               +--------------+--------------+
                              |
                              v
                      +---------------+
                      |   runner.py   |
                      +-------+-------+
                              |
                              v
                         +---------+
                         | main.py |
                         +---------+
```

### Fluxo de Dados Durante Uma Rodada de Simulação

1. **Grade & Entidades**: `runner.py` chama `distribuir_AP()` (`entities.py`) para criar as estações rádio-base.
2. **Posicionamento de UEs**: `runner.py` instancia os `UE`s (`entities.py`) em posições aleatórias válidas.
3. **Cálculo de Antena**: `runner.py` chama `antenna_gain()` (`antennas.py`) para computar o ganho angular.
4. **Matriz de Canal**: `channel.py` combina perda por distância, shadowing e ganhos para gerar a matriz de ganho final.
5. **Associação (Attach)**: `allocation.py` conecta cada UE ao AP com melhor ganho e seleciona o feixe ativo.
6. **Alocação de Canais**: `allocation.py` distribui os canais de radiofrequência entre os UEs.
7. **Métricas (SINR/Capacidade)**: `metrics.py` lê as posições, canais e potências para calcular o $SINR$ e a capacidade de Shannon em Mbps.
8. **Visualização**: `runner.py` devolve os dicionários de dados consolidados para o `main.py`, que invoca as rotinas de gráfico do `visualization.py`.

---

## 🚀 Como Configurar e Rodar o Projeto na Sua Máquina

### 1. Clonar e Criar o Ambiente Virtual

```bash
git clone https://github.com/SEU-USUARIO/mobile-network-simulator.git
cd mobile-network-simulator

# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente virtual
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
```

### 2. Instalar o Pacote em Modo Editável (`-e`)

A presença do arquivo `pyproject.toml` permite que o Python trate a pasta `src/simulator` como um pacote instalado no ambiente virtual. Qualquer alteração no código fonte é aplicada **instantaneamente**, sem necessidade de reinstalação.

```bash
pip install -e .
```

### 3. Rodar os Testes Unitários

Para garantir que a matemática do sistema não foi violada por alterações recentes:

```bash
pytest
```

### 4. Executar o Simulador Principal

```bash
python main.py
```

---

## 🛠️ Boas Práticas para Desenvolvimento em Equipe

1. **Nunca faça `commit` direto na branch `main`**.
2. Criar branches temáticas:
   * `feature/modelo-canal-3gpp`
   * `fix/correcao-calculo-sinr`
   * `docs/atualiza-readme`
3. **Sempre rode os testes (`pytest`)** antes de abrir um *Pull Request* (PR).
4. Evite variáveis globais com estado mutável.
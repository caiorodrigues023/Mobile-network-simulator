import random
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.lines import Line2D
from tqdm import tqdm


# Rótulos e cores usados nos gráficos para cada modelo de antena
MODEL_LABELS = {
    'omni': 'Omni', 
    'sector': 'Setorizado', 
    'smb': 'SMB', 
    'adaptive': 'Adaptativo (Ideal 0°)',
    'adaptive_1': 'Adaptativo (Erro 1°)',
    'adaptive_5': 'Adaptativo (Erro 5°)',
    'adaptive_10': 'Adaptativo (Erro 10°)'
}
MODEL_COLORS = {
    'omni': 'tab:blue', 
    'sector': 'tab:orange', 
    'smb': 'tab:green', 
    'adaptive': 'tab:red',
    'adaptive_1': 'tab:purple',
    'adaptive_5': 'tab:brown',
    'adaptive_10': 'tab:pink'
}


class AP:
    def __init__(self, x: float, y: float, id: int):
        self.x = x
        self.y = y
        self.id = id
        self.channel = None
        self.ues = []

    def __str__(self):
        return f'AP[{self.id}]({self.x}, {self.y})'

class UE:
    id_counter = 0
    def __init__(self, aps: list):
        self.id = UE.id_counter
        UE.id_counter += 1
        ap_coords = np.array([[ap.x, ap.y] for ap in aps])
        self.x = np.random.randint(0, 1001)
        self.y = np.random.randint(0, 1001)
        while True:
                    ue_coord = np.array([self.x, self.y])
                    d_all = np.linalg.norm(ap_coords - ue_coord, axis=1)
                    if np.all(d_all >= 1):
                        break
                    self.x = np.random.randint(0, 1001)
                    self.y = np.random.randint(0, 1001)
        self.ap = None
        self.beam = 0   # índice do setor/feixe ativo no AP servidor
        self.channel = 0
        self.dist = 0
        self.angle = 0
        self.gain = 0
        self.power = 1  # Transmit power

    def __str__(self):
        return f'UE({self.x}, {self.y})'

    def __eq__(self, other):
        if isinstance(other, UE):
            return self.id == other.id
        return False

def distribuir_AP(M: int) -> list:
    APs = []
    dx = 1000/(2*np.sqrt(M))
    a = np.arange(dx, 1001-dx, 2*dx)
    x, y = np.meshgrid(a, a)
    id = 0
    for xi, yi in zip(x.ravel(), y.ravel()):
        APs.append(AP(xi, yi, id))
        id += 1
    return APs

def angle(ue: UE, ap: AP) -> float:
    delta_x = ue.x - ap.x
    delta_y = ue.y - ap.y
    theta = np.arctan2(delta_y, delta_x)
    if theta < 0:
        theta += 2*np.pi
    return theta

def angle_matrix(ues: list, aps: list) -> np.ndarray:
    """Ângulo (0 a 2*pi) de cada UE i em relação a cada AP j -> matriz (n_ues, n_aps)."""
    ue_xy = np.array([[ue.x, ue.y] for ue in ues], dtype=float)
    ap_xy = np.array([[ap.x, ap.y] for ap in aps], dtype=float)
    dx = ue_xy[:, None, 0] - ap_xy[None, :, 0]
    dy = ue_xy[:, None, 1] - ap_xy[None, :, 1]
    return np.mod(np.arctan2(dy, dx), 2*np.pi)

# ---------------------------------------------------------------------------
# Switched Multi-Beam (SMB) com Uniform Circular Array (Apêndice A/B)
# ---------------------------------------------------------------------------
_LAMBDA = 1.0  # comprimento de onda (o resultado não depende dele: R é proporcional a lambda)

@lru_cache(maxsize=None)
def _smb_weights(L: int, J: int):
    """
    Pré-computa a matriz de pesos W (L x J) do SMB: cada coluna é o vetor
    de pesos MRC (conjugado do steering vector) apontado para theta_k = k*2*pi/J,
    normalizado por norm(w), exatamente como no Apêndice B. Calculada uma só vez.
    """
    R = L * _LAMBDA / (4*np.pi)
    phi_el = np.arange(L) * 2*np.pi / L             # posição angular dos elementos
    theta_k = np.arange(J) * 2*np.pi / J            # direção de apontamento de cada feixe
    a_tgt = np.exp(1j*2*np.pi/_LAMBDA*R*np.cos(theta_k[None, :] - phi_el[:, None]))  # (L, J)
    W = np.conj(a_tgt) / np.linalg.norm(a_tgt, axis=0, keepdims=True)
    return W, phi_el, R

def smb_gains(phi, L: int = 8, J: int = 8) -> np.ndarray:
    """
    Função de ganho SMB (Exercício 2, item 2).

    Entrada : phi, azimute(s) UE->AP em radianos (escalar ou array de qualquer shape).
    Saída   : ganho linear |w_k^T a(phi)|^2 de TODOS os J feixes, shape = phi.shape + (J,).
    O pico de cada feixe vale L (ex.: 8) e o cruzamento entre feixes vizinhos ~1,82 (L=J=8).
    """
    phi = np.asarray(phi, dtype=float)
    W, phi_el, R = _smb_weights(L, J)
    flat = phi.ravel()
    A = np.exp(1j*2*np.pi/_LAMBDA*R*np.cos(flat[None, :] - phi_el[:, None]))  # (L, N)
    y = W.T @ A                                                               # (J, N)
    return (np.abs(y)**2).T.reshape(phi.shape + (J,))

# ---------------------------------------------------------------------------
# Beamforming adaptativo com MRC (Exercício 3) — mesmo UCA do SMB, pesos em tempo real
# ---------------------------------------------------------------------------
def _uca_steering(phi, L: int) -> np.ndarray:
    """Steering vector a(phi) do UCA de L elementos (Apêndice A). Retorna shape phi.shape + (L,)."""
    phi = np.asarray(phi, dtype=float)
    R = L * _LAMBDA / (4*np.pi)
    phi_el = np.arange(L) * 2*np.pi / L
    return np.exp(1j*2*np.pi/_LAMBDA*R*np.cos(phi[..., None] - phi_el))

def mrc_gain(phi_beam, phi_user, L: int = 8) -> np.ndarray:
    """
    Ganho do feixe MRC apontado para phi_beam, na direção phi_user (broadcast entre os dois):
        w = conj(a(phi_beam)) / ||a||,   G = |w^T a(phi_user)|^2
    Com phi_user == phi_beam o ganho é exatamente L (ganho de arranjo máximo, eq. 13).
    Com phi_beam = k*2*pi/J reproduz o feixe k do SMB (a diferença é só o ponto de apontamento).
    """
    a_b = _uca_steering(phi_beam, L)
    a_u = _uca_steering(phi_user, L)
    y = np.sum(np.conj(a_b) / np.sqrt(L) * a_u, axis=-1)
    return np.abs(y)**2

def antenna_gain(ues: list, aps: list, model: str = 'sector', beamwidth: float = 2*np.pi/3,
                  g_min: float = 0.01, L: int = 8, J: int = 8) -> np.ndarray:
    """
    Calcula a matriz de ganho de antena G[i, j, b] para cada UE i, AP j e
    setor/feixe b.

    model:
      - 'omni'  : antena omnidirecional (ganho unitário, 1 "feixe" por AP)
      - 'sector': antena setorial ideal (3 setores de 'beamwidth' cada)
      - 'adaptive': beamforming adaptativo MRC. Aqui devolve ganho unitário (omni), pois o
                  initial access usa uma antena isolada do arranjo; o ganho MRC é aplicado
                  depois, por UE, via adaptive_gains()
      - 'smb'   : Switched Multi-Beam com UCA de L elementos e J feixes
                  (G[i, j, :] = ganho dos J feixes do AP j na direção do UE i)

    Cada UE-AP tem seu próprio ângulo relativo, e cada setor/feixe é avaliado
    individualmente para esse ângulo.
    """
    
    phi = angle_matrix(ues, aps)   # (n_ues, n_aps)

    # Verifica se é omni ou qualquer variação do adaptativo (ex: adaptive_5)
    if model == 'omni' or model.startswith('adaptive'):
        return np.ones((len(ues), len(aps), 1))

    if model == 'sector':
        theta_boresight = np.array([0, 2*np.pi/3, 4*np.pi/3])  # 3 setores de 120°
        delta = np.abs(phi[:, :, None] - theta_boresight[None, None, :]) % (2*np.pi)
        delta = np.minimum(delta, 2*np.pi - delta)
        return np.where(delta <= beamwidth/2, 1.0, g_min)

    if model == 'smb':
        return smb_gains(phi, L=L, J=J)

    raise ValueError(f"Modelo de antena desconhecido: {model!r}")

def gerar_shadowing(ues: list, aps: list) -> np.ndarray:
    """
    Sorteia o shadowing log-normal de forma independente para cada link (UE, AP).
    Comportamento original: variável aleatória log-normal(0, 2).
    """
    return np.random.lognormal(0, 2, size=(len(ues), len(aps)))

def gain_matrix(ues: list, aps: list, G: np.ndarray, shadowing: np.ndarray = None) -> np.ndarray:
    '''
    Calcula a matriz de ganho de canal (path loss + shadowing + ganho de antena).

    shadowing: matriz (n_ues, n_aps) pré-sorteada (broadcast sobre setores/feixes).
    Se None, sorteia um novo shadowing (comportamento legado, não recomendado
    quando se quer comparar modelos de antena sob as mesmas condições).
    '''
    if shadowing is None:
        shadowing = gerar_shadowing(ues, aps)
    shadowing = np.asarray(shadowing)
    if shadowing.ndim == 2:
        shadowing = shadowing[:, :, None]

    ue_xy = np.array([[ue.x, ue.y] for ue in ues], dtype=float)
    ap_xy = np.array([[ap.x, ap.y] for ap in aps], dtype=float)
    d = np.linalg.norm(ue_xy[:, None, :] - ap_xy[None, :, :], axis=2)  # (n_ues, n_aps)
    d = np.maximum(d, 1.0)

    return shadowing * 1e-4 / (d[:, :, None]**4) * G

def alocar_canais_ortogonal(access_points, ues, number_channels, allocation):
    """
    Aloca canais aos UEs de forma ortogonal dentro de cada célula (AP).

    Argumentos:
    access_points (list): A lista de objetos AccessPoint.
    user_equipaments (list): A lista de objetos UserEquipament.
    number_channels (int): O número total de canais ortogonais (N).

    Modifica:
    O atributo 'channel' de cada objeto na lista user_equipaments é
    atualizado "in-place" (no próprio objeto).
    """

    # 1. Zera os canais de todos os UEs
    for ue in ues:
        ue.channel = None

    if allocation == 'random':
        for ue in ues:
            ue.channel = np.random.randint(1, number_channels + 1)
        return None

    # 2. Inicializa o contador global de uso de canal
    channels = {i: 0 for i in range(1, number_channels + 1)}

    # 3. Itera por cada AP para alocação intra-célula
    for ap in access_points:
        available_channels = list(channels.keys())
        random.shuffle(available_channels)

        # 4. Itera pelos UEs (por índice) conectados a este AP
        for ue0 in ap.ues:
            if not available_channels:
                # Para se o AP tiver mais UEs do que canais
                break

            # 5. Atribui um canal disponível e o remove da lista do AP
            channel = available_channels.pop()

            # 6. Atribui o canal ao UE e incrementa o contador global
            index_ue = ue0.id
            ues[index_ue].channel = channel
            channels[channel] += 1

    # 7. Etapa de Preenchimento (Cleanup)
    for ue in ues:
        if ue.channel is None:
            # Se o UE não recebeu um canal (passo 4),
            # atribui o canal menos usado globalmente.
            least_used_channel = min(channels, key=channels.get)
            ue.channel = least_used_channel
            channels[least_used_channel] += 1

def attach_AP_UE(ues: list, aps: list, gains: np.ndarray) -> None:
    """
    Initial access: cada UE se associa ao par (AP, setor/feixe) com o MAIOR
    ganho de canal, entre as M*B combinações (B = nº de setores/feixes por AP:
    1 no omni, 3 no setorizado, J no SMB).

    Guarda em cada UE: ue.ap (AP servidor), ue.beam (setor/feixe ativo) e ue.gain.
    """
    for i, ue in enumerate(ues):
        j, b = np.unravel_index(np.argmax(gains[i]), gains[i].shape)
        ue.ap = aps[j]
        ue.beam = int(b)
        ue.gain = gains[i, j, b]
        ue.dist = np.linalg.norm(np.array([ue.x, ue.y]) - np.array([ue.ap.x, ue.ap.y]))
        ue.angle = angle(ue, ue.ap)
        aps[j].ues.append(ue)

def DPC(ues: list, N: int, t_max: int, G: np.ndarray, R: np.ndarray, y_tar: float, p_min: float, p_max: float, p_init: float) -> np.ndarray:
    # vetor de potencias
    p = np.ones((len(ues), t_max)) * p_init         # inicializa as potências com p_init para cada UE
    y = np.zeros((len(ues), t_max))                 # vetor para armazenar os SINRs ao longo das iterações
    pn = 1e8*1e-20/N
    iteracoes = 0

    for t in range(t_max):
        for i, ue in enumerate(ues):
            # Calcula a interferência total para este UE
            interference = sum([p[k][t] * G[k][ue.ap.id] * R[k][ue.ap.id] for k in range(len(ues)) if k != i and ues[k].channel == ue.channel])
            ue.interference.append(interference)

            # Calcula o SINR para este UE
            sinr = (p[i][t] * ue.gain * R[i][ue.ap.id]) / (interference + pn)
            y[i][t] = sinr

            # Atualiza a potência usando o algoritmo de controle de potência
            pt = min(max(p_min, p[i][t] * y_tar / sinr), p_max)

            if t < t_max - 1:  # Evita atualizar a potência na última iteração
                p[i][t+1] = pt

            #print(f'DPC: UE{i} - Iteração {t} - Potência: {p[i][t]:.4f} W - Ganho: {ue.gain} - SINR: {sinr:.4f} - Interferência: {interference:.4e}')

        iteracoes += 1
        if iteracoes > 5 and t < t_max - 1:
            soma_atual = np.sum(p[:, t+1])
            soma_anterior = np.sum(p[:, t])
            if abs(soma_atual - soma_anterior) < 1e-3:
                break
    return p[:, 0:iteracoes]

def SINR(ues: list, N: int, gains: np.ndarray, G: np.ndarray = None) -> list:
    pt=1        # transmited power
    bt=1e8      # avaiable bandwidth
    k0=1e-20    # constant for the noise power
    pn = k0*bt/N

    num_ues = len(ues)

    # Vetor de Potência (P)
    P = np.array([ue.power for ue in ues])

    # Vetor de Associação de AP (A): A[k] = índice do AP que serve o UE k
    A = np.array([ue.ap.id for ue in ues])

    # Vetor de Setor/Feixe ativo (B): B[k] = setor/feixe que serve o UE k
    B = np.array([ue.beam for ue in ues])

    # Vetor de Alocação de Canal (C): C[k] = canal usado pelo UE k
    C = np.array([ue.channel for ue in ues])

    sinr_list = []

    # Itera por cada UE 'k' para calcular seu SINR
    for k, ue in enumerate(ues):
        m = A[k]   # AP servidor
        b = B[k]   # setor/feixe ativo do UE k no AP m

        # Sinal do UE k no AP m, pelo setor/feixe ativo
        S = gains[k, m, b] * P[k]

        # Interferência: todos os *outros* UEs no mesmo canal, recebidos no AP m
        # ATRAVÉS DO FEIXE ATIVO b do UE desejado (filtro espacial)
        I = 0.0
        for i in range(num_ues):
            if i != k and C[i] == C[k]:
                I += gains[i, m, b] * P[i]

        sinr_list.append(S / (I + pn))

    return sinr_list

def adaptive_gains(ues: list, aps: list, L: int = 8, aoa_err_deg: float = 0.0) -> np.ndarray:
    """
    Beamforming adaptativo MRC (Exercício 3; com aoa_err_deg > 0, Exercício 4).
    Requer o attachment já feito (ue.ap). Para cada UE k, servido pelo AP m_k, forma um feixe MRC
    apontado para o AoA (estimado) do UE k:  phi_hat_k = phi[k, m_k] + eps,  eps ~ N(0, aoa_err^2).

    Retorna Gm (n_ues, n_ues) com Gm[k, i] = ganho do feixe do UE k, visto no AP m_k, na direção
    REAL do UE i. Logo: Gm[k, k] = ganho no UE desejado (= L se aoa_err_deg = 0) e Gm[k, i != k]
    = ganho aplicado à interferência do UE i sobre o UE k.
    """
    n = len(ues)
    phi = angle_matrix(ues, aps)                       # (n_ues, n_aps)
    m = np.array([ue.ap.id for ue in ues])             # AP servidor de cada UE
    phi_hat = phi[np.arange(n), m] + np.deg2rad(aoa_err_deg) * np.random.randn(n)
    phi_dir = phi[:, m].T                              # phi_dir[k, i] = azimute do UE i no AP m_k
    return mrc_gain(phi_hat[:, None], phi_dir, L)

def SINR_adaptive(ues: list, N: int, gains: np.ndarray, Gm: np.ndarray) -> list:
    """
    SINR com beamforming adaptativo. gains: (n_ues, n_aps, 1) SEM ganho de antena (path loss +
    shadowing); Gm: saída de adaptive_gains(). Mesma convenção de ruído/interferência de SINR().
    """
    bt = 1e8
    k0 = 1e-20
    pn = k0*bt/N
    P = np.array([ue.power for ue in ues])
    A = np.array([ue.ap.id for ue in ues])
    C = np.array([ue.channel for ue in ues])
    sinr_list = []
    for k in range(len(ues)):
        m = A[k]
        S = gains[k, m, 0] * P[k] * Gm[k, k]
        mask = (C == C[k])
        mask[k] = False
        I = np.sum(gains[mask, m, 0] * P[mask] * Gm[k, mask])
        sinr_list.append(S / (I + pn))
    return sinr_list

def channel_capacity(sinr_valores: list, N: int) -> list:
    # 1. Calcula a largura de banda por canal (B_canal)
    B_per_channel = 1e8 / N

    # 2. Converte a(s) entrada(s) SINR para um array numpy para permitir o cálculo elemento a elemento (vetorizado).
    sinr_array = np.asarray(sinr_valores)

    # 3. Aplica a fórmula de Shannon-Hartley
    # C = B * log2(1 + SINR)
    capacity_mbps = B_per_channel * np.log2(1 + sinr_array) / 1e6  # Converte para Mbps

    return capacity_mbps

def simular_experimento(M: int, N: int, K: int, sim: int, allocation: str = '',
                        antenna_models: list = ('omni', 'adaptive', 'adaptive_1', 'adaptive_5', 'adaptive_10'),
                        shadowing_params: dict = None, L: int = 8, J: int = 8) -> dict:
    
    aps = distribuir_AP(M)

    resultados = {model: {'power': [], 'sinr': [], 'cap': [], 'sum_cap': []} for model in antenna_models}

    for sim_idx in tqdm(range(sim)):
        UE.id_counter = 0
        ues = [UE(aps) for i in range(K)]

        shadowing = gerar_shadowing(ues, aps)

        for model in antenna_models:
            for ue in ues:
                ue.ap, ue.beam, ue.channel = None, 0, 0
            for ap in aps:
                ap.ues = []

            G = antenna_gain(ues, aps, model=model, L=L, J=J)
            gains = gain_matrix(ues, aps, G, shadowing=shadowing)
            attach_AP_UE(ues, aps, gains)   
            
            # --- Lógica de extração de erro para o Exercício 4 ---
            if model.startswith('adaptive'):
                err_deg = 0.0
                if '_' in model: # Se tiver '_', extrai o número que vem depois (ex: 'adaptive_5' -> 5.0)
                    try:
                        err_deg = float(model.split('_')[1])
                    except ValueError:
                        pass
                
                Gm = adaptive_gains(ues, aps, L=L, aoa_err_deg=err_deg)
                s = SINR_adaptive(ues, N, gains, Gm)
                power = [ue.gain * ue.power * Gm[k, k] for k, ue in enumerate(ues)]
            else:
                s = SINR(ues, N, gains, G)
                power = [ue.gain * ue.power for ue in ues]
                
            cap = channel_capacity(s, N)
            sum_cap = np.sum(cap)

            resultados[model]['power'].extend(power)
            resultados[model]['sinr'].extend(s)
            resultados[model]['cap'].extend(cap)
            resultados[model]['sum_cap'].append(sum_cap)

            for ap in aps:
                ap.ues = []

    for model in antenna_models:
        resultados[model]['sum_cap'] = np.mean(resultados[model]['sum_cap'])

    return resultados

def plot_cdfs(cdf_sinr: list, cdf_capacity: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot SINR CDF
    for i in range(len(cdf_sinr)):
        cdf_sinr[i] = [10*np.log10(a) for a in cdf_sinr[i]]
        cdf_sinr[i].sort()
        percentis = np.linspace(0, 1, len(cdf_sinr[i]))
        axes[0].plot(cdf_sinr[i], percentis, label=f'SINR {"Per-AP" if i==1 else "Random"} channel allocation')

    axes[0].axhline(y=0.10, color='r', linewidth=0.7, linestyle='--', label = f'10th Percentil')
    axes[0].axhline(y=0.50, color='b', linewidth=0.7, linestyle='--', label = f'50th Percentil')
    axes[0].set_title('CDF do SINR por UE')
    axes[0].set_xlabel('SINR (dB)')
    axes[0].set_ylabel('Percentil')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Capacity CDF
    for i in range(len(cdf_capacity)):
        cdf_capacity[i].sort()
        percentis = np.linspace(0, 1, len(cdf_capacity[i]))
        axes[1].plot(cdf_capacity[i], percentis, label=f'Cap. do Canal {"Per-AP" if i==1 else "Random"} channel allocation')

    axes[1].axhline(y=0.10, color='r', linewidth=0.7, linestyle='--', label = f'10th Percentil')
    axes[1].axhline(y=0.50, color='b', linewidth=0.7, linestyle='--', label = f'50th Percentil')
    axes[1].set_title('CDF da Capacidade do Canal por UE')
    axes[1].set_xlabel('Capacidade do Canal (Mbps)')
    axes[1].set_ylabel('Percentil')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def compare_kpis(resultados: dict, load_labels: list, models: list = None) -> None:
    """
    Implementa o procedimento do Remark 2: para cada cenário de carga,
    compara os KPIs do caso omni (baseline) com os demais modelos.
    
    - SINR (10th percentil): mostrado como GANHO em dB em relação ao modelo omni
      (omni fica nivelado em 0 dB).
    - Capacidade (10th percentil) e Soma-capacidade média: normalizadas
      linearmente em relação ao omni (omni = 1, demais = razão).
    """
    if models is None:
        models = list(resultados.keys())
    labels = [MODEL_LABELS.get(m, m) + (' (baseline)' if m == 'omni' else '') for m in models]
    colors = [MODEL_COLORS.get(m, None) for m in models]

    fig, axes = plt.subplots(2, len(load_labels), figsize=(6*len(load_labels), 9))
    if len(load_labels) == 1:
        axes = axes.reshape(2, 1)

    for col, load in enumerate(load_labels):
        # --- SINR (10th percentil, diferença em dB em relação ao omni) ---
        base_sinr = np.percentile(np.asarray(resultados['omni'][load]['sinr']).flatten(), 10)
        base_sinr_db = 10 * np.log10(base_sinr)

        # Subtrai o base_sinr_db de todos os modelos (omni será 0 dB)
        sinr_db = [10*np.log10(np.percentile(np.asarray(resultados[m][load]['sinr']).flatten(), 10)) - base_sinr_db
                   for m in models]

        ax_sinr = axes[0, col]
        ax_sinr.bar(labels, sinr_db, color=colors)
        ax_sinr.set_title(f'Ganho de SINR 10th pct ({load})')
        ax_sinr.set_ylabel('Ganho vs Omni (dB)')
        ax_sinr.axhline(y=0, color='k', linewidth=0.8, linestyle='--')
        ax_sinr.grid(True, axis='y')
        ax_sinr.tick_params(axis='x', rotation=15)

        # --- Capacidade (10th percentil) e Soma-capacidade (linear, normalizado) ---
        base = resultados['omni'][load]
        cap_base = np.percentile(np.asarray(base['cap']).flatten(), 10)
        sumcap_base = base['sum_cap']

        metrics = ['Capacidade (10th pct)', 'Soma-capacidade média']
        x = np.arange(len(metrics))
        width = 0.8 / len(models)

        ax = axes[1, col]
        for idx, m in enumerate(models):
            res = resultados[m][load]
            vals = [np.percentile(np.asarray(res['cap']).flatten(), 10) / cap_base,
                    res['sum_cap'] / sumcap_base]
            offset = (idx - (len(models) - 1)/2) * width
            ax.bar(x + offset, vals, width, label=labels[idx], color=colors[idx])
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=15)
        ax.set_title(f'Capacidade normalizada ({load})')
        ax.axhline(y=1.0, color='k', linewidth=0.8, linestyle='--') # Linha base em 1.0 para métricas de razão
        ax.legend()
        ax.grid(True, axis='y')

    plt.tight_layout()
    plt.show()

def plot_all_cdfs(resultados_por_carga: dict, load_labels: list, models: list = None) -> None:
    """
    Plota as CDFs de potência recebida, SINR e capacidade do canal por UE,
    comparando as tecnologias de antena (omni, setorizado, SMB) para cada
    cenário de carga, com marcação dos percentis 10 e 50.

    resultados_por_carga: dicionário no formato
        {
          'omni':   {K_label: {'power': [...], 'sinr': [...], 'cap': [...], ...}, ...},
          'sector': {K_label: {...}, ...},
          'smb':    {K_label: {...}, ...},
        }
    load_labels: lista de labels de carga (ex: ['K=1', 'K=4', 'K=8'])
    models: modelos a plotar. Se None, usa todas as chaves do dicionário.
    """
    if models is None:
        models = list(resultados_por_carga.keys())

    n_loads = len(load_labels)
    fig, axes = plt.subplots(n_loads, 3, figsize=(18, 5*n_loads))
    if n_loads == 1:
        axes = axes.reshape(1, 3)

    kpi_specs = [
        ('power', 'Potência Recebida', 'Potência Recebida (dBW)', True),
        ('sinr',  'SINR',              'SINR (dB)',               True),
        ('cap',   'Capacidade do Canal', 'Capacidade do Canal (Mbps)', False),
    ]

    for row, load in enumerate(load_labels):
        for col, (key, title, xlabel, to_db) in enumerate(kpi_specs):
            ax = axes[row, col]
            for model in models:
                data = np.asarray(resultados_por_carga[model][load][key]).flatten()
                if to_db:
                    data = 10*np.log10(data)
                data = np.sort(data)
                percentis = np.linspace(0, 1, len(data))
                ax.plot(data, percentis,
                        label=MODEL_LABELS.get(model, model),
                        color=MODEL_COLORS.get(model))

            ax.axhline(y=0.10, color='r', linewidth=0.7, linestyle='--', label='10th Percentil')
            ax.axhline(y=0.50, color='k', linewidth=0.7, linestyle='--', label='50th Percentil')
            ax.set_title(f'CDF {title} ({load})')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Percentil')
            ax.legend(fontsize=8)
            ax.grid(True)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Visualização espacial: mapa de APs, UEs e setores/feixes
# ---------------------------------------------------------------------------
def cenario_aleatorio(M: int, K: int, seed: int = None, shadowing_params: dict = None):
    """Gera um cenário (aps, ues, shadowing) para visualização. Use 'seed' para reproduzir."""
    if seed is not None:
        np.random.seed(seed)
    aps = distribuir_AP(M)
    UE.id_counter = 0
    ues = [UE(aps) for _ in range(K)]
    shadowing = gerar_shadowing(ues, aps)
    return aps, ues, shadowing

def plotar_mapa(aps: list, ues: list, shadowing: np.ndarray = None,
                models: list = ('omni', 'sector', 'smb', 'adaptive'), r_ap: float = 110.0,
                L: int = 8, J: int = 8, show_sinr: bool = True, N: int = 1,
                savepath: str = None, show: bool = True):
    """
    Mapa 2D da área de cobertura, um painel por modelo de antena, todos com
    OS MESMOS APs, UEs e shadowing (comparação justa).
    """
    if shadowing is None:
        shadowing = gerar_shadowing(ues, aps)

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(7.2*n, 7.6), squeeze=False)
    axes = axes[0]
    cmap = plt.get_cmap('tab20')
    sector_colors = ['tab:blue', 'tab:red', 'tab:green']
    theta_b = [0, 120, 240]  # graus, 3 setores
    th = np.linspace(0, 2*np.pi, 721)
    curves = smb_gains(th, L=L, J=J) / L if 'smb' in models else None  # normalizado pelo pico

    for ax, model in zip(axes, models):
        # Reseta e refaz o attachment para este modelo
        for ue in ues:
            ue.ap, ue.beam, ue.channel = None, 0, 0
        for ap in aps:
            ap.ues = []
        G = antenna_gain(ues, aps, model=model, L=L, J=J)
        gains = gain_matrix(ues, aps, G, shadowing=shadowing)
        attach_AP_UE(ues, aps, gains)
        
        if model.startswith('adaptive'):
            err_deg = 0.0
            if '_' in model:
                try: err_deg = float(model.split('_')[1])
                except ValueError: pass
            sinr = np.asarray(SINR_adaptive(ues, N, gains, adaptive_gains(ues, aps, L=L, aoa_err_deg=err_deg))).flatten()
        else:
            sinr = np.asarray(SINR(ues, N, gains, G)).flatten()
        sinr_db = 10*np.log10(sinr)
        ativos = {(ue.ap.id, ue.beam) for ue in ues}

        # --- padrões de antena ---
        for ap in aps:
            if model == 'omni':
                ax.add_patch(Circle((ap.x, ap.y), r_ap*0.7, fc='tab:blue', alpha=0.10,
                                    ec='tab:blue', lw=0.8, zorder=1))
            elif model == 'sector':
                for b in range(3):
                    ativo = (ap.id, b) in ativos
                    ax.add_patch(Wedge((ap.x, ap.y), r_ap, theta_b[b]-60, theta_b[b]+60,
                                       fc=sector_colors[b], alpha=0.50 if ativo else 0.12,
                                       ec=sector_colors[b], lw=1.0, zorder=1))
            elif model == 'smb':
                for k in range(J):
                    r = r_ap * curves[:, k]
                    x, y = ap.x + r*np.cos(th), ap.y + r*np.sin(th)
                    if (ap.id, k) in ativos:
                        c = plt.cm.hsv(k / J)
                        ax.fill(x, y, color=c, alpha=0.45, zorder=2)
                        ax.plot(x, y, color=c, lw=1.6, zorder=2)
                    else:
                        ax.plot(x, y, color='gray', lw=0.6, ls='--', alpha=0.7, zorder=1)

        # --- feixes MRC adaptativos: um por UE, apontado exatamente para ele ---
        if model.startswith('adaptive'):
            for ue in ues:
                r = r_ap * mrc_gain(ue.angle, th, L) / L
                x, y = ue.ap.x + r*np.cos(th), ue.ap.y + r*np.sin(th)
                c = cmap(ue.ap.id % 20)
                ax.fill(x, y, color=c, alpha=0.45, zorder=2)
                ax.plot(x, y, color=c, lw=1.6, zorder=2)

        # --- ligações e UEs ---
        for ue, sdb in zip(ues, sinr_db):
            c = cmap(ue.ap.id % 20)
            ax.plot([ue.x, ue.ap.x], [ue.y, ue.ap.y], color=c, lw=1.2, alpha=0.9, zorder=3)
            ax.scatter(ue.x, ue.y, s=55, color=c, edgecolor='k', linewidth=0.8, zorder=6)
            if show_sinr:
                ax.annotate(f'{sdb:.1f} dB', (ue.x, ue.y), xytext=(5, 5),
                            textcoords='offset points', fontsize=7, zorder=7)

        # --- APs ---
        for ap in aps:
            ax.scatter(ap.x, ap.y, marker='^', s=70, color='k', zorder=8)
            ax.annotate(str(ap.id), (ap.x, ap.y), xytext=(4, -11),
                        textcoords='offset points', fontsize=7, color='k', zorder=8)

        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.set_aspect('equal')
        ax.set_facecolor('#f1f1f1')
        ax.grid(True, linestyle=':', color='gray', alpha=0.5)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f'{MODEL_LABELS.get(model, model)}  |  SINR médio: {np.mean(sinr_db):.1f} dB')

    # --- Tratamento da Legenda ---
    handles = [Line2D([], [], marker='^', color='k', ls='', label='AP'),
               Line2D([], [], marker='o', color='gray', mfc='gray', mec='k', ls='', label='UE (cor = AP servidor)')]
    if 'sector' in models:
        handles.append(Line2D([], [], color='tab:red', lw=6, alpha=0.5, label='Setor ativo'))
    if 'smb' in models:
        handles.append(Line2D([], [], color='tab:green', lw=2, alpha=0.5, label='Feixe SMB ativo'))
    if model.startswith('adaptive'):
        handles.append(Line2D([], [], color='tab:purple', lw=4, alpha=0.5, label='Feixe MRC (Adaptativo)'))

    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    
    if savepath:
        fig.savefig(savepath, dpi=150)
    if show:
        plt.show()
    # return fig

def plot_padroes_smb(configs: list = ((8, 4), (8, 16), (4, 8), (16, 8)),
                     savepath: str = None, show: bool = True):
    """
    Padrões de antena SMB em coordenadas polares para várias combinações (L, J),
    como na Fig. 9 do roteiro. configs: lista de pares (L, J).
    O círculo externo corresponde ao ganho de pico G = L.
    """
    n = len(configs)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, subplot_kw={'projection': 'polar'},
                             figsize=(5*cols, 5*rows), squeeze=False)
    th = np.linspace(0, 2*np.pi, 1000)
    for ax, (L_, J_) in zip(axes.ravel(), configs):
        g = smb_gains(th, L=L_, J=J_)
        for k in range(J_):
            ax.plot(th, g[:, k], color=plt.cm.hsv(k / J_), lw=1.6)
        ax.set_ylim(0, L_)
        ax.set_yticks([L_/2, L_])
        ax.set_yticklabels([f'{L_/2:g}', f'{L_}'], fontsize=7)
        ax.set_title(f'L={L_}, J={J_}  (G = {L_})', fontsize=10)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    if show:
        plt.show()
    return fig
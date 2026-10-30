# ---------------------------------------------------------------------------
# entities.py — Abstração de hardware: AP (Access Point) e UE (User Equipment)
# ---------------------------------------------------------------------------

import numpy as np


class AP:
    """Estação rádio-base (Access Point)."""

    def __init__(self, x: float, y: float, id: int) -> None:
        self.x = x
        self.y = y
        self.id = id
        self.channel = None
        self.ues: list = []

    def __str__(self) -> str:
        return f'AP[{self.id}]({self.x}, {self.y})'

    def __repr__(self) -> str:
        return self.__str__()


class UE:
    """Equipamento de usuário (User Equipment)."""

    id_counter: int = 0

    def __init__(self, aps: list) -> None:
        self.id = UE.id_counter
        UE.id_counter += 1

        ap_coords = np.array([[ap.x, ap.y] for ap in aps])

        # Posiciona o UE aleatoriamente, garantindo distância ≥ 1 m de qualquer AP
        self.x = np.random.randint(0, 1001)
        self.y = np.random.randint(0, 1001)
        while True:
            ue_coord = np.array([self.x, self.y])
            d_all = np.linalg.norm(ap_coords - ue_coord, axis=1)
            if np.all(d_all >= 1):
                break
            self.x = np.random.randint(0, 1001)
            self.y = np.random.randint(0, 1001)

        # Estado de associação (preenchido por allocation.attach_AP_UE)
        self.ap = None
        self.beam: int = 0          # índice do setor/feixe ativo no AP servidor
        self.channel: int = 0
        self.dist: float = 0.0
        self.angle: float = 0.0
        self.gain: float = 0.0
        self.power: float = 1.0     # Potência de transmissão (W)
        self.interference: list = []  # histórico de interferência para DPC

    def __str__(self) -> str:
        return f'UE({self.x}, {self.y})'

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, UE):
            return self.id == other.id
        return False

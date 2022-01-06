from estado import *
from acao import *


class ModeloTR:
    def __init__(self):
        self.T = {}
        self.R = {}

    def atualizar(self, s: Estado, a: Acao, r: float, sn: Estado):
        self.T[(s, a)] = sn  # Modelo determinista
        self.R[(s, a)] = r

    def amostrar(self):
        s, a = choice(self.T.keys())
        sn = self.T[(s, a)]
        r = self.R[(s, a)]
        return s, a, r, sn

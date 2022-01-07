import random as rnd
from estado_acao import *


class ModeloTR:
    def __init__(self):
        self.T = {}
        self.R = {}

    def atualizar(self, s: Estado, a: Acao, r: float, sn: Estado):
        self.T[(s, a)] = sn  # Modelo determinista
        self.R[(s, a)] = r

    def amostrar(self):
        s, a = self.T.keys()[rnd.randint(0, len(self.T.keys()) - 1)]
        sn = self.T[(s, a)]
        r = self.R[(s, a)]
        return s, a, r, sn

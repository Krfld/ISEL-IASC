import random as rnd
from memoria_esparsa import *


class SelAcao:
    def selecionar_acao(self, s: Estado) -> Acao:
        raise NotImplementedError


class EGreedy(SelAcao):
    def __init__(self, mem_aprend: MemoriaAprend, acoes: list, epsilon: float):
        self.mem_aprend = mem_aprend
        self.acoes = acoes
        self.epsilon = epsilon

    def max_acao(self, s: Estado) -> Acao:
        rnd.shuffle(self.acoes)
        return max(self.acoes, key=lambda a: self.mem_aprend.Q(s, a))

    def aproveitar(self, s: Estado) -> Acao:
        return self.max_acao(s)

    def explorar(self) -> Acao:
        return self.acoes[rnd.randint(0, len(self.acoes) - 1)]

    def selecionar_acao(self, s: Estado) -> Acao:
        if rnd.random() > self.epsilon:
            return self.aproveitar(s)
        else:
            return self.explorar()

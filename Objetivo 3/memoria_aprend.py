from estado import *
from acao import *


class MemoriaAprend:
    def atualizar(s: Estado, a: Acao, q: float):
        raise NotImplementedError

    def Q(s: Estado, a: Acao) -> float:
        raise NotImplementedError


class MemoriaEsparsa(MemoriaAprend):
    def __init__(self, valor_omissao: float = 0):
        self.valor_omissao = valor_omissao
        self.memoria = {}

    def Q(self, s: Estado, a: Acao) -> float:
        return self.memoria.get((s, a), self.valor_omissao)

    def atualizar(self, s: Estado, a: Acao, q: float):
        self.memoria[(s, a)] = q

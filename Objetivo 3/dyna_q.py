from e_greedy import *
from modelo_tr import *


class AprendRef:
    def __init__(self, mem_aprend: MemoriaAprend, sel_acao: SelAcao, alfa: float, gama: float):
        self.mem_aprend = mem_aprend
        self.sel_acao = sel_acao
        self.alfa = alfa
        self.gama = gama

    def aprender(s: Estado, a: Acao, r: float, sn: Estado, an: Acao = None):
        raise NotImplementedError


class QLearning(AprendRef):
    def aprender(self, s: Estado, a: Acao, r: float, sn: Estado, an: Acao = None):
        an = self.sel_acao.max_acao(sn)
        qsa = self.mem_aprend.Q(s, a)
        qsnan = self.mem_aprend.Q(sn, an)
        q = qsa + self.alfa * (r + self.gama * qsnan - qsa)
        self.mem_aprend.atualizar(s, a, q)


class DynaQ(QLearning):
    def __init__(self, mem_aprend: MemoriaAprend, sel_acao: SelAcao, alfa: float, gama: float, num_sim: int):
        super().__init__(mem_aprend, sel_acao, alfa, gama)
        self.num_sim = num_sim
        self.modelo = ModeloTR()

    def simular(self):  # f)
        for i in range(self.num_sim):
            s, a, r, sn = self.modelo.amostrar()
            super.aprender(s, a, r, sn)

    def aprender(self, s: Estado, a: Acao, r: float, sn: Estado):
        super.aprender(s, a, r, sn)  # d)
        self.modelo.atualizar(s, a, r, sn)  # e)
        self.simular()  # f)

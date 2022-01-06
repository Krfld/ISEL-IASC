from aprend_ref import *
from modelo_tr import *


class DynaQ(QLearning):
    def __init__(self, mem_aprend: MemoriaAprend, sel_acao: SelAcao, alfa: float, gama: float, num_sim: int):
        super().__init__(mem_aprend, sel_acao, alfa, gama)
        self.num_sim = num_sim
        self.modelo = ModeloTR()

    def aprender(self, s: Estado, a: Acao, r: float, sn: Estado):
        super.aprender(s, a, r, sn)
        self.modelo.atualizar(s, a, r, sn)
        self.simular()

    def simular(self):
        for i in range(self.num_sim):
            s, a, r, sn = self.modelo.amostrar()
            super.aprender(s, a, r, sn)

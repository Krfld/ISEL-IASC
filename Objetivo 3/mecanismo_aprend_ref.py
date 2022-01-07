from dyna_q import *


class MecanismoAprendRef:
    def __init__(self, acoes: list):
        self.acoes = acoes
        self.mem_aprend = MemoriaEsparsa()
        self.sel_acao = EGreedy(self.mem_aprend, self.acoes, 0.1)
        self.aprend_ref = DynaQ(self.mem_aprend, self.sel_acao, 0.1, 0.1, 10)

    # ?? Falta a) b) c)
    # def aprender(self, s: Estado, a: Acao, r: float, sn: Estado, an: Acao = None):
    #     self.aprend_ref.aprender(s, a, r, sn, an)

    # def selecionar_acao(self, s: Estado) -> Acao:
    #     return self.sel_acao.selecionar_acao(s)

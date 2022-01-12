from dyna_q import *


class MecanismoAprendRef:
    def __init__(self, acoes: list[Acao]):
        self.acoes = acoes
        self.mem_aprend = MemoriaEsparsa()
        self.sel_acao = EGreedy(self.mem_aprend, self.acoes, 0.2)
        self.aprend_ref = DynaQ(self.mem_aprend, self.sel_acao, 0.7, 0.9, 500)  # confirm alpha & gama

    def aprender(self, s: Estado, a: Acao, r: float, sn: Estado):
        self.aprend_ref.aprender(s, a, r, sn)

    def selecionar_acao(self, s: Estado) -> Acao:
        return self.sel_acao.selecionar_acao(s)

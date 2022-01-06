from aprend_ref import *


class QLearning(AprendRef):
    def aprender(self, s: Estado, a: Acao, r: float, sn: Estado, an: Acao = None):
        an = self.sel_acao.max_acao(sn)
        qsa = self.mem_aprend.Q(s, a)

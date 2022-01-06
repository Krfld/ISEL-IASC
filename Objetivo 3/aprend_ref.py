from abc import ABC, abstractmethod
from mecanismo_aprend_ref import *


class AprendRef(ABC):
    def __init__(self, mem_aprend: MemoriaAprend, sel_acao: SelAcao, alfa: float, gama: float):
        self.mem_aprend = mem_aprend
        self.sel_acao = sel_acao
        self.alfa = alfa
        self.gama = gama

    @abstractmethod
    def aprender(s: Estado, a: Acao, r: float, sn: Estado, an: Acao = None):
        raise NotImplementedError

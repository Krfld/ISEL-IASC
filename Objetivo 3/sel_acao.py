from abc import ABC, abstractmethod
from estado import *
from acao import *


class SelAccao:
    @abstractmethod
    def selecionar_acao(self, s: Estado) -> Acao:
        raise NotImplementedError

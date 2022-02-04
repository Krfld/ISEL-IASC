import numpy as np
import matplotlib.pyplot as plt
from estado_acao import *


class Mundo:
    def __init__(self, nomeArquivo: str):
        self.mundo = carregarMundo(nomeArquivo)

    def carregarMundo(self, nomeArquivo: str) -> list[list[int]]:
        with open(nomeArquivo, "r") as arquivo:
            lines = arquivo.readlines()
            mundo: list[list[int]] = np.zeros((len(lines), len(lines[0].removesuffix('\n'))), dtype=int)

            estadoInicial = Estado(0, 0)
            alvo = Estado(0, 0)

            m = 0
            for x in lines:
                n = 0
                for y in x[:-1]:
                    if y.__eq__('O'):
                        mundo[m][n] = -1
                    elif y.__eq__('A'):
                        mundo[m][n] = 2
                        alvo = Estado(n, m)
                    elif y.__eq__('>'):
                        # mundo[m][n] = 1
                        estadoInicial = Estado(n, m)
                    n += 1
                m += 1

            print(mundo)
            print(estadoInicial, 'Start')
            print(alvo, 'End')
            return mundo, estadoInicial, alvo

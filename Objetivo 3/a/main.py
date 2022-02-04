import numpy as np
import matplotlib.pyplot as plt
from estado import *


class Mundo:
    def __init__(self, nomeArquivo: str, mostrarGrafico: bool = True):
        self.mundo, self.start, self.alvo = self.carregarMundo(nomeArquivo)
        self.s = self.start
        self.mostrarGrafico = mostrarGrafico
        self.valorMundo = self.propagarValor([self.alvo])

        mundo = [[x for x in y] for y in self.mundo]
        for v in self.valorMundo:
            mundo[v.y][v.x] = self.valorMundo[v]

        path = self.getPath(self.s)

        # for s in path:
        #     mundo[s.y][s.x] = 0

        plt.imshow(mundo)
        plt.show()

    def carregarMundo(self, nomeArquivo: str):
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

    def mostrar(self):
        print(self.s)

        posicao = [[x for x in y] for y in self.mundo]
        posicao[self.s.y][self.s.x] = 1  # Colocar o agente na posicao atual

        if self.mostrarGrafico:
            # plt.title("Movimentos: " + str(self.movimentos))
            plt.imshow(posicao)
            plt.show()

    def propagarValor(self, objetivos: list[Estado], gain: int = 10):
        V = {}
        frenteDeOnda = []
        gama = len(self.mundo)/(len(self.mundo)+1)  # Gama proporcional oa mundo

        for o in objetivos:
            V[o] = gain
            frenteDeOnda.append(o)

        while len(frenteDeOnda) > 0:  # Enquanto houver estados na frente de onda
            s = frenteDeOnda.pop(0)
            for a in self.adjacentes(s):
                v = V[s] * gama  # Atenua o valor
                if v > V.get(a, -1):  # Caso haja um valor pior no estado adjacente, substitui e adiciona à frente de onda
                    V[a] = v
                    frenteDeOnda.append(a)
        return V

    def adjacentes(self, s: Estado) -> list[Estado]:
        # Estados adjacentes na vertical e horizental apenas, verificando obstáculos
        adjacentes: list[Estado] = []
        if s.x > 0:
            e = Estado(s.x - 1, s.y)
            if self.mundo[e.y][e.x] != -1:
                adjacentes.append(e)
        if s.x < len(self.mundo[0]) - 1:
            e = Estado(s.x + 1, s.y)
            if self.mundo[e.y][e.x] != -1:
                adjacentes.append(e)
        if s.y > 0:
            e = Estado(s.x, s.y - 1)
            if self.mundo[e.y][e.x] != -1:
                adjacentes.append(e)
        if s.y < len(self.mundo) - 1:
            e = Estado(s.x, s.y + 1)
            if self.mundo[e.y][e.x] != -1:
                adjacentes.append(e)
        return adjacentes

    def getPath(self, s: Estado):
        path = []
        while s != self.alvo:
            path.append(s)
            sn = max(self.adjacentes(s), key=lambda s: self.valorMundo[s])
            s = sn
        return path


if __name__ == '__main__':
    m = Mundo("Objetivo 3/proj-obj3-amb/amb1.txt")  # Carrega o mundo, colocando o agente de volta ao início

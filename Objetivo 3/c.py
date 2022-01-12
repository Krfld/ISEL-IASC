import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from mecanismo_aprend_ref import *


mundo: list[list[int]] = [[0, 0, 0, 0, 0, 0, 0, -1, 1],
                          [0, 0, -1, 0, 0, 0, 0, -1, 0],
                          [0, 0, -1, 0, 0, 0, 0, -1, 0],
                          [0, 0, -1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, -1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0]]


class Mundo:
    def __init__(self, mundo: list[list[int]], estadoInicial: Estado = Estado(0, 0), rMultiplicador: float = 10, custo: float = 0.01, mostrarGrafico: bool = True):
        self.mundo = mundo
        self.s = estadoInicial
        self.rMultiplicador = rMultiplicador
        self.custo = custo
        self.mostrarGrafico = mostrarGrafico
        self.movimentos = 0

        if self.mostrarGrafico:
            plt.ion()
            plt.figure()

    def estadoAtual(self) -> Estado:
        return self.s

    def atualizarEstado(self, sn: Estado):
        self.s = sn

    def mover(self, a: Acao):
        sn = Estado(self.s.x + a.dx, self.s.y + a.dy)

        if sn.x < 0 or sn.x >= len(self.mundo[0]) or sn.y < 0 or sn.y >= len(self.mundo):
            return self.s, -self.rMultiplicador*2

        self.movimentos += 1
        r = self.mundo[sn.y][sn.x]
        return sn if not r < 0 else self.s, r*self.rMultiplicador - self.custo

    def mostrar(self):
        # print(self.s, self.movimentos)

        posicao = [[x for x in y] for y in self.mundo]
        posicao[self.s.y][self.s.x] = 11

        sys.stdout.write(str(np.array(posicao))+'\n')
        sys.stdout.flush()

        time.sleep(0.05)
        if self.mostrarGrafico:
            plt.title("Movimentos: " + str(self.movimentos))
            plt.imshow(posicao)
            plt.pause(0.1)
            plt.show()


if __name__ == '__main__':
    #                         Right     | Left       | Up         | Down
    mar = MecanismoAprendRef([Acao(1, 0), Acao(-1, 0), Acao(0, -1), Acao(0, 1)])
    estadoInicial = Estado(0, 2)

    while True:
        m = Mundo(mundo, estadoInicial, mostrarGrafico=False)
        while True:
            a = mar.selecionar_acao(m.estadoAtual())
            sn, r = m.mover(a)
            mar.aprender(m.estadoAtual(), a, r, sn)

            m.atualizarEstado(sn)
            m.mostrar()

            if (sn == Estado(8, 0)):
                break
        print(m.movimentos)

import matplotlib.pyplot as plt
from mecanismo_aprend_ref import *


mundo = [[0, 0, 0, 0, 0, 0, 0, -1, 1],
         [0, 0, -1, 0, 0, 0, 0, -1, 0],
         [0, 0, -1, 0, 0, 0, 0, -1, 0],
         [0, 0, -1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, -1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]]


class Mundo:
    def __init__(self, mundo: list[list[int]], estadoInicial: Estado = Estado(0, 0), rMultiplicador: float = 1, custo: float = 0.01):
        self.mundo = mundo
        self.s = estadoInicial
        self.rMultiplicador = rMultiplicador
        self.custo = custo

        self.movimentos = 0

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

    def mostrar(self, mostrarGrafico: bool = True):
        print(self.s, self.movimentos)

        if not mostrarGrafico:
            return

        posicao = [[x for x in y] for y in self.mundo]
        posicao[self.s.y][self.s.x] = 2
        plt.title("Movimentos: " + str(self.movimentos))
        plt.imshow(posicao)
        plt.pause(0.1)
        plt.show()


if __name__ == '__main__':
    #                       Right     | Left       | Up         | Down
    mar = MecanismoAprendRef([Acao(1, 0), Acao(-1, 0), Acao(0, -1), Acao(0, 1)])
    estadoInicial = Estado(0, 2)

    plt.ion()
    while True:
        m = Mundo(mundo, estadoInicial)
        plt.figure()
        while True:
            a = mar.selecionar_acao(m.estadoAtual())
            sn, r = m.mover(a)
            mar.aprender(m.estadoAtual(), a, r, sn)

            m.atualizarEstado(sn)
            m.mostrar()

            if (sn == Estado(8, 0)):
                break

import numpy as np
import matplotlib.pyplot as plt
from mecanismo_aprend_ref import *


class Mundo:
    def __init__(self, nomeArquivo: str, multiplicadorReforço: float = 10, custoMover: float = 0.01, mostrarGrafico: bool = True):
        self.mundo, self.s, self.alvo = self.carregarMundo(nomeArquivo)  # Obtem o mundo, o estado inicial e o alvo

        self.multiplicadorReforço = multiplicadorReforço
        self.custoMover = custoMover
        self.mostrarGrafico = mostrarGrafico
        self.movimentos = 0

        if self.mostrarGrafico:
            plt.ion()

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

    def estadoAtual(self) -> Estado:
        return self.s

    def atualizarEstado(self, sn: Estado):
        self.s = sn

    def mover(self, a: Acao):
        sn = Estado(self.s.x + a.dx, self.s.y + a.dy)

        # Incrementa o numeros de movimentos
        self.movimentos += 1

        # Verifica limites do mundo
        # if sn.x < 0 or sn.x >= len(self.mundo[0]) or sn.y < 0 or sn.y >= len(self.mundo):
        #     return self.s, -self.multiplicadorReforço*2

        # Obtem a informacao da posicao seguinte
        r = self.mundo[sn.y][sn.x]

        # retorna o estado seguinte (verificando colisao) e o reforço de acordo com a posicao e o custo do movimento
        return sn if r >= 0 else self.s, r * self.multiplicadorReforço - self.custoMover

    def mostrar(self):
        print(self.s, self.movimentos)

        posicao = [[x for x in y] for y in self.mundo]
        posicao[self.s.y][self.s.x] = 1  # Colocar o agente na posicao atual

        if self.mostrarGrafico:
            plt.title("Movimentos: " + str(self.movimentos))
            plt.imshow(posicao)
            plt.pause(0.1)
            plt.clf()
            plt.show()


if __name__ == '__main__':
    #                         Right     | Left       | Up         | Down
    mar = MecanismoAprendRef([Acao(1, 0), Acao(-1, 0), Acao(0, -1), Acao(0, 1)])

    while True:
        m = Mundo("Objetivo 3/proj-obj3-amb/amb1.txt")  # Carrega o mundo, colocando o agente de volta ao início
        while True:
            a = mar.selecionar_acao(m.estadoAtual())  # Seleciona a acao
            sn, r = m.mover(a)  # Move o agente
            mar.aprender(m.estadoAtual(), a, r, sn)  # Aprende

            m.atualizarEstado(sn)  # Atualiza o estado
            m.mostrar()

            if (sn == m.alvo):
                break
        print(m.movimentos)

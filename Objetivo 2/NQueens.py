import random as rnd
from tkinter.tix import INTEGER
import matplotlib.pyplot as plt
from numpy import integer


class NQueens:
    def __init__(self, N: int, printState: bool = True):
        self.N = N
        # self.state = [rnd.randint(1, N) for i in range(N)]
        # if printState:
        #     print(self.state)
        #     self.printState(self.state, 'Initial board')

    def printState(self, state: list, msg: str = ''):
        # print(msg, state, '|', self.stateValue(state))

        # Build matrix to plot
        board = [[1 if j+1 == state[i] else 0 for j in range(len(state))] for i in range(len(state))]

        plt.figure()
        plt.title(msg + '\nAttacks: ' + str(self.stateValue(state) * -1))
        plt.imshow(board)
        # plt.axis(False)

        plt.show()

    def initialState(self):
        # Generate new random state
        return [rnd.randint(1, self.N) for i in range(self.N)]

    def randomNeighbor(self, state: list):
        randomState = state.copy()

        # Choose random index
        index = rnd.randint(0, len(state)-1)
        pos = randomState[index]

        # Move queen to a random position
        newPos = rnd.randint(1, len(state))
        while newPos == pos:
            newPos = rnd.randint(1, len(state))

        randomState[index] = newPos
        return randomState

    def bestNeighbor(self, state: list):
        bestState = state.copy()

        # Guarantee that returns a different state
        bestValue = -len(state)**2

        # Randomize order to find the best neighbor
        indexes = [i for i in range(len(state))]
        rnd.shuffle(indexes)

        for index in indexes:
            newState = state.copy()
            for pos in range(1, len(state)+1):
                if pos != state[index]:
                    # Move queen
                    newState[index] = pos

                    # If the new state is better than the current best
                    newValue = self.stateValue(newState)
                    if newValue > bestValue:
                        bestState = newState.copy()
                        bestValue = newValue

        return bestState

    def stateValue(self, state: list):
        # number of collisions
        value = 0
        for i in range(len(state) - 1):
            for j in range(i+1, len(state)):
                if state[j] == state[i] or abs(state[j] - state[i]) == j - i:
                    value += 1

        value *= -1
        return value  # returns negative value so that the best is closer to 0

    ### Genetic algorithm ###

    def reproduce(self, x: list, y: list):
        c = rnd.randint(1, len(x)-1)
        new = x[:c]
        for i in y[c:]:
            new.append(i)
        return new

    def population(self, populationSize: int = 100):
        population = [self.initialState() for i in range(populationSize)]
        return population

    def fitnessFunction(self, element: list) -> float:
        value = self.stateValue(element)*-1
        return 1/(value+1)

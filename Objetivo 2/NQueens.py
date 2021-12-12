import numpy as np
import random as rnd
import matplotlib.pyplot as plt


class NQueens:
    def __init__(self, N, printState=True):
        self.N = N
        # self.state = np.array([i for i in range(1, N+1)])
        # np.random.shuffle(self.state)
        self.state = np.array([rnd.randint(1, N) for i in range(N)])
        if printState:
            self.printState(self.state, 'Initial board')

    def printState(self, state, msg=''):
        # print(msg, state, '|', self.stateValue(state))

        # Build matrix to plot
        board = np.array([[1 if j+1 == state[i] else 0 for j in range(len(state))] for i in range(len(state))])

        plt.figure()
        plt.title(msg + '\nAttacks: ' + str(self.stateValue(state) * -1))
        plt.imshow(board)
        # plt.axis(False)

        plt.show()

    def initialState(self):
        # Always return the same start state
        return self.state.copy()

    def randomNeighbor(self, state):
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

    def bestNeighbor(self, state):
        bestState = state.copy()

        # Change the first queen to guarantee that the bestState will always be different than the original state
        bestState[0] = len(state)+1 - bestState[0]

        bestValue = self.stateValue(bestState)

        # Randomize order to find the best neighbor
        indexes = np.array([i for i in range(len(state))])
        np.random.shuffle(indexes)

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

    def stateValue(self, state):
        value = 0
        for i in range(len(state) - 1):
            for j in range(i+1, len(state)):
                if state[j] == state[i] or abs(state[j] - state[i]) == j - i:
                    value += 1

        value *= -1
        return value

    ### Genetic algorithm ###

    def reproduce(self, x, y):
        c = rnd.randint(1, len(x)-1)  # len(x) = self.N
        return np.append(x[:c], y[c:])

    def population(self, populationSize=50):
        population = np.array([NQueens(self.N, printState=False).initialState() for i in range(populationSize)])
        population = np.reshape(population, (populationSize, self.N))
        return population

    def fitnessFunction(self, element):
        value = self.stateValue(element)*-1
        return int(1000/(value+1))

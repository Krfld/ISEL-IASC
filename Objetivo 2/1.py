import math
import numpy as np
import random as rnd


class SearchAlgorithms:
    def stochasticHillClimbing(problem):
        current = problem.initialState()

        while True:
            neighbor = problem.bestNeighbor(current)

            if problem.stateValue(neighbor) <= problem.stateValue(current):
                return current

            current = neighbor.copy()

    def simulatedAnnealing(problem, schedule):
        current = problem.initialState()
        t = 0

        while True:
            t += 1
            T = schedule(t)

            if T == 0:
                return current

            neighbor = problem.randomNeighbor(current)
            deltaE = problem.stateValue(neighbor) - problem.stateValue(current)

            if deltaE > 0 or rnd.random() <= math.exp(deltaE/T):
                current = neighbor


class NQueens:
    def __init__(self, N):
        self.N = N
        self.state = self.initialState()

    def initialState(self):
        # state = np.array([i for i in range(1, self.N+1)])
        # np.random.shuffle(state)
        return np.array([rnd.randint(1, self.N) for i in range(self.N)])

    def randomNeighbor(self, state):
        col = rnd.randint(0, self.N-1)
        row = state[col]
        newRow = rnd.randint(1, self.N)

        while newRow == row:
            newRow = rnd.randint(1, self.N)

        state[col] = newRow
        return state

    def bestNeighbor(self, state):
        bestState = state.copy()
        bestState[0] = 5 - bestState[0]
        bestValue = self.stateValue(bestState)

        cols = np.array([i for i in range(self.N)])
        np.random.shuffle(cols)

        for col in cols:
            newState = state.copy()
            for row in range(1, self.N+1):
                if row != state[col]:
                    newState[col] = row
                    newValue = self.stateValue(newState)
                    if newValue >= bestValue:
                        bestState = newState.copy()
                        bestValue = newValue

        return bestState

    def stateValue(self, state):
        value = 0

        for i in range(self.N - 1):
            for j in range(i+1, self.N):
                if state[j] == state[i] or abs(state[j] - state[i]) == j - i:
                    value += 1

        value *= -1
        return value


print(SearchAlgorithms.stochasticHillClimbing(NQueens(8)))

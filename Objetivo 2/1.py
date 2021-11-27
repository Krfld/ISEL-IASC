# %%
import math
import numpy as np
import random as rnd
import matplotlib.pyplot as plt


# %%
class SearchAlgorithms:
    def stochasticHillClimbing(problem, stuckIterations=10):
        current = problem.initialState()
        oldNeighbor = current.copy()
        stuck = 0
        while True:
            # Obtain best neighbor state from current one
            neighbor = problem.bestNeighbor(current)
            # print(neighbor, current, '|', problem.stateValue(neighbor), problem.stateValue(current))

            # Check if it's stuck in an infinite loop
            if np.array_equal(neighbor, oldNeighbor):
                stuck += 1
            else:
                stuck = 0

            # Return state if it's stuck in a maximum (local or global) or in a loop
            if problem.stateValue(neighbor) < problem.stateValue(current) or stuck > stuckIterations:
                # problem.printState(current)
                return current

            oldNeighbor = current.copy()
            current = neighbor.copy()

    def hillClimbingWithRandomRestart(problem, iterations=25):
        bestSolution = SearchAlgorithms.stochasticHillClimbing(problem)
        bestValue = problem.stateValue(bestSolution)
        for i in range(iterations):
            # Obtain solution from stochastic hill climbing
            solution = SearchAlgorithms.stochasticHillClimbing(problem)

            # If solution is better than the best one
            solutionValue = problem.stateValue(solution)
            if solutionValue > bestValue:
                bestSolution = solution.copy()
                bestValue = solutionValue

        # Return best solution state after iterations
        problem.printState(bestSolution, 'Solution state:')
        return bestSolution

    def simulatedAnnealing(problem, schedule):
        current = problem.initialState()
        t = 0
        while True:
            t += 1
            T = schedule(t)

            if T == 0:
                problem.printState(current, 'Solution state:')
                return current

            neighbor = problem.randomNeighbor(current)
            deltaE = problem.stateValue(neighbor) - problem.stateValue(current)

            # print(deltaE, math.exp(deltaE/T), T)
            if deltaE > 0 or rnd.random() <= math.exp(deltaE/T):
                current = neighbor.copy()

    def schedule(time):
        return 100 * 0.9**time


# %%
class NQueens:
    def __init__(self, N):
        self.maxValue = 0
        # self.state = np.array([i for i in range(1, N+1)])
        # np.random.shuffle(self.state)
        self.state = np.array([rnd.randint(1, N) for i in range(N)])
        self.printState(self.state, 'Initial state:')

    def printState(self, state, msg=''):
        # print(msg, state, '|', self.stateValue(state))

        # Build matrix to plot
        board = np.array([[1 if j+1 == state[i] else 0 for j in range(len(state))] for i in range(len(state))])
        plt.figure()
        plt.title(msg)
        plt.imshow(board)
        # plt.axis(False)

        plt.show()

    def initialState(self):
        # Always return the same start state
        return self.state.copy()

    def randomNeighbor(self, state):
        col = rnd.randint(0, len(state)-1)
        row = state[col]

        newRow = rnd.randint(1, len(state))
        while newRow == row:
            newRow = rnd.randint(1, len(state))

        state[col] = newRow
        return state.copy()

    def bestNeighbor(self, state):
        bestState = state.copy()

        # Change the first queen to guarantee that the bestState will always be different than the original state
        bestState[0] = len(state)+1 - bestState[0]

        bestValue = self.stateValue(bestState)

        # Randomize order to find the best neighbor
        cols = np.array([i for i in range(len(state))])
        np.random.shuffle(cols)

        for col in cols:
            newState = state.copy()
            for row in range(1, len(state)+1):
                if row != state[col]:
                    # Move queen
                    newState[col] = row

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


# %%
SearchAlgorithms.hillClimbingWithRandomRestart(NQueens(10))


# %%
class TravellingSalesman:
    def __init__(self, N, size=100):
        self.state = np.array([], dtype=int)

        for i in range(N):
            # Check that every city is different
            city = [rnd.randint(0, size), rnd.randint(0, size)]
            while city in self.state:
                city = [rnd.randint(0, size), rnd.randint(0, size)]

            self.state = np.append(self.state, city)
            self.state = np.reshape(self.state, (i+1, 2))

        self.printState(self.state, 'Cities:', False)

    def printState(self, state, msg='', showPath=True):
        # print(msg, state, '|', self.stateValue(state))

        plt.figure()
        plt.title(msg)

        if showPath:
            # Append first city at the end
            length = len(state)
            state = np.append(state, state[0])
            state = np.reshape(state, (length+1, 2))

            plt.plot(*zip(*state), 'o-')
            for i in range(len(state)-1):
                plt.annotate(f'  {i+1}', (state[i][0], state[i][1]))
        else:
            plt.plot(*zip(*state), 'o')

        plt.show()

    def initialState(self):
        # Always return the same cities with different order
        np.random.shuffle(self.state)
        return self.state.copy()

    def distanceBetweenCities(self, city1, city2):
        # Calculate distance between two cities
        return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

    def randomNeighbor(self, state):
        randomState = state.copy()

        # Choose two random cities
        choices = np.array([i for i in range(len(state))])
        np.random.shuffle(choices)

        index = rnd.randint(0, len(state) - 1)
        city1Index = choices[index]
        city2Index = choices[len(state)-1 - index]

        # Swap cities
        tempCity = randomState[city1Index].copy()
        randomState[city1Index] = randomState[city2Index].copy()
        randomState[city2Index] = tempCity

        return randomState

    def bestNeighbor(self, state):
        bestState = state.copy()

        # Swap second two cities to guarantee that the bestState will always be different than the original state
        tempCity = bestState[1].copy()
        bestState[1] = bestState[2].copy()
        bestState[2] = tempCity

        bestValue = self.stateValue(bestState)

        # Randomize order to find the best neighbor
        indexes = np.array([i for i in range(len(state) - 1)])
        np.random.shuffle(indexes)

        for i in indexes:
            for j in range(i+1, len(state)):
                newState = state.copy()

                # Swap cities
                tempCity = newState[i].copy()
                newState[i] = newState[j].copy()
                newState[j] = tempCity

                # If new state is better than the current best
                newValue = self.stateValue(newState)
                if newValue > bestValue:
                    bestState = newState.copy()
                    bestValue = newValue

        return bestState

    def stateValue(self, state):
        # Append first city at the end
        length = len(state)
        state = np.append(state, state[0])
        state = np.reshape(state, (length+1, 2))

        totalDistance = 0
        for i in range(len(state)-1):
            totalDistance += self.distanceBetweenCities(state[i], state[i+1])

        totalDistance *= -1
        return totalDistance


# %%
SearchAlgorithms.hillClimbingWithRandomRestart(TravellingSalesman(20))


# %%
# SearchAlgorithms.simulatedAnnealing(TravellingSalesman(10), SearchAlgorithms.schedule)

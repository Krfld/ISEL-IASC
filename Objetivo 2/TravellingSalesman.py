import math
import numpy as np
import random as rnd
import matplotlib.pyplot as plt


class TravellingSalesman:
    def __init__(self, N, size=100, printState=True):
        self.N = N
        self.state = np.array([], dtype=int)

        for i in range(N):
            # Check that every city is different
            city = [rnd.randint(0, size), rnd.randint(0, size)]
            while city in self.state:
                city = [rnd.randint(0, size), rnd.randint(0, size)]

            self.state = np.append(self.state, city)
            self.state = np.reshape(self.state, (i+1, 2))

        if printState:
            self.printState(self.state, 'Cities', False)

    def printState(self, state, msg='', showPath=True):
        # print(msg, state, '|', self.stateValue(state))

        plt.figure()

        if showPath:
            plt.title(msg + '\nDistance: ' + str(int(self.stateValue(state) * -1)))

            # Append first city at the end
            length = len(state)
            state = np.append(state, state[0])
            state = np.reshape(state, (length+1, 2))

            plt.plot(*zip(*state), 'o-')
            for i in range(len(state)-1):
                plt.annotate(f'  {i+1}', (state[i][0], state[i][1]))
        else:
            plt.title(msg)
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
        randomState[city1Index], randomState[city2Index] = randomState[city2Index].copy(), randomState[city1Index].copy()

        return randomState

    def bestNeighbor(self, state):
        bestState = state.copy()

        # Swap second two cities to guarantee that the bestState will always be different than the original state
        bestState[1], bestState[2] = bestState[2].copy(), bestState[1].copy()

        bestValue = self.stateValue(bestState)

        # Randomize order to find the best neighbor
        indexes = np.array([i for i in range(len(state) - 1)])
        np.random.shuffle(indexes)

        for i in indexes:
            for j in range(i+1, len(state)):
                newState = state.copy()

                # Swap cities
                newState[i], newState[j] = newState[j].copy(), newState[i].copy()

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

    ### Genetic algorithm ###

    def reproduce(self, x, y):
        c = rnd.randint(1, len(x)-1)

        # No repeated elements
        child = np.array([i for i in x if i not in y[c:]])
        child = np.append(child, y[c:])
        child = np.reshape(child, (len(x), 2))
        return child

    def population(self, populationSize=2):
        population = np.array([TravellingSalesman(self.N, printState=False).initialState() for i in range(populationSize)])
        population = np.reshape(population, (populationSize, self.N*2))
        return population

    def fitnessFunction(self, element):
        value = self.stateValue(element)*-1
        return int(1000/(value+1))

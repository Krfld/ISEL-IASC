import math
import random as rnd
import matplotlib.pyplot as plt


class TravellingSalesman:
    def __init__(self, N: int, worldSize: int = 100, printState: bool = True):
        self.state = []

        for i in range(N):
            # Check that every city is different
            city = (rnd.randint(0, worldSize), rnd.randint(0, worldSize))
            while city in self.state:
                city = (rnd.randint(0, worldSize), rnd.randint(0, worldSize))

            self.state.append(city)
            # self.state = np.reshape(self.state, (i+1, 2))

        if printState:
            print(self.state)
            self.printState(self.state, 'Cities', False)

    def printState(self, state: list[tuple], msg: str = '', showPath: bool = True):
        # print(msg, state, '|', self.stateValue(state))

        plt.figure()

        if showPath:
            plt.title(msg + '\nDistance: ' + str(int(self.stateValue(state) * -1)))

            # Append first city at the end
            # length = len(state)
            state.append(state[0])
            # state = np.reshape(state, (length+1, 2))

            plt.plot(*zip(*state), 'o-')
            for i in range(len(state)-1):
                plt.annotate(f'  {i+1}', (state[i][0], state[i][1]))
        else:
            plt.title(msg)
            plt.plot(*zip(*state), 'o')

        plt.show()

    def initialState(self) -> list[tuple]:
        # Always return the same cities with different order
        rnd.shuffle(self.state)
        return self.state.copy()

    def randomNeighbor(self, state: list) -> list[tuple]:
        randomState = state.copy()

        # Choose two random cities
        choices = [i for i in range(len(state))]
        rnd.shuffle(choices)

        index = rnd.randint(0, len(state) - 1)
        city1Index = choices[index]
        city2Index = choices[len(state)-1 - index]

        # Swap cities
        randomState[city1Index], randomState[city2Index] = randomState[city2Index], randomState[city1Index]

        return randomState

    def bestNeighbor(self, state: list) -> list[tuple]:
        bestState = state.copy()

        # Guarantee that returns a different state
        bestValue = -len(state)**4

        # Randomize order to find the best neighbor
        indexes = [i for i in range(len(state) - 1)]
        rnd.shuffle(indexes)

        for i in indexes:
            for j in range(i+1, len(state)):
                newState = state.copy()

                # Swap cities
                newState[i], newState[j] = newState[j], newState[i]

                # If new state is better than the current best
                newValue = self.stateValue(newState)
                if newValue > bestValue:
                    bestState = newState.copy()
                    bestValue = newValue

        return bestState

    def distanceBetweenCities(self, city1: tuple, city2: tuple) -> float:
        # Calculate distance between two cities
        return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

    def stateValue(self, state: list) -> float:  # distance between cities (returning to the first)
        # Append first city at the end
        # length = len(state)
        newState = state.copy()
        newState.append(state[0])
        # state = np.reshape(state, (length+1, 2))

        totalDistance = 0
        for i in range(len(newState)-1):
            totalDistance += self.distanceBetweenCities(newState[i], newState[i+1])

        totalDistance *= -1
        return totalDistance  # returns negative value so that the best is closer to 0

    ### Genetic algorithm ###

    def reproduce(self, x: list[tuple], y: list[tuple]) -> list[tuple]:
        c = rnd.randint(1, len(x)-1)

        # No repeated elements
        child = [i for i in x if i not in y[c:]]
        for i in y[c:]:
            child.append(i)
        # print(child)
        # child = np.reshape(child, (len(x), 2))
        return child

    def population(self, populationworldSize: int = 100) -> list[list[tuple]]:
        population = [self.initialState() for i in range(populationworldSize)]
        # population = np.reshape(population, (populationworldSize, self.N*2))
        return population

    def fitnessFunction(self, element: list[tuple]) -> float:
        value = self.stateValue(element)*-1
        return 1/(value+1)  # int(1000000/(value+1))

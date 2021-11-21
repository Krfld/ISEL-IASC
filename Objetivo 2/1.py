import math
import random as rnd


class SearchAlgorithms:
    def stochasticHillClimbing(problem):
        current = problem.initialState()

        while True:
            neighbor = problem.bestNeighbor(current)

            if neighbor.value() <= current.value():
                return current

            current = neighbor

    def simulatedAnnealing(problem, schedule):
        current = problem.initialState()
        t = 0

        while True:
            t += 1
            T = schedule(t)

            if T == 0:
                return current

            next = problem.randomNeighbor(current)
            deltaE = next.value() - current.value()

            if deltaE > 0 or rnd.random() <= math.exp(deltaE/T):
                current = next

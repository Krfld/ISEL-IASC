import numpy as np
import random as rnd

SAMPLES = 2
MATRIX_SIZE = 4

data = np.array([], dtype=int)
numbers = np.array([], dtype=int)

while len(numbers) < SAMPLES:
    n = rnd.randint(0, 2**(MATRIX_SIZE**2) - 1)

    if(n in numbers):
        if len(numbers) == SAMPLES:
            break
        continue

    numbers = np.append(numbers, n)

    matrix = bin(n).removeprefix('0b').rjust(MATRIX_SIZE**2, '0')
    matrix = np.array([int(i) for i in matrix])
    data = np.append(data, matrix)

data = np.reshape(data, (SAMPLES, MATRIX_SIZE**2))

print(data)

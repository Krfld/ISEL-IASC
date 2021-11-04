import numpy as np
import random as rnd


MATRIX_SIZE = 4

data = np.array([])
numbers = np.array([], dtype=int)

while len(data) < 2:
    n = rnd.randint(0, 2**(MATRIX_SIZE**2)-1)

    if(n in numbers):
        continue

    numbers = np.append(numbers, n)

    data = np.append(data, [bin(n).removeprefix(
        '0b').rjust(MATRIX_SIZE**2, '0')[:]])


print(numbers)
print(data)


print(np.random.randint(2, size=(4, 4)))

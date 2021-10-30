import numpy as np
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers.core import Dense, Reshape
from keras import activations
from numpy.core.fromnumeric import ndim

SAMPLES = 1000
EPOCHS = 100

MATRIX_SIZE = 5

dummy = np.array([[1, 1, 0, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])


def get_square_matrix(size: int):
    m = np.random.randint(2, size=(size, size))
    print('\n', m, '\n')
    return m


def get_matrix_outputs(matrix):
    size = np.shape(matrix)[0]

    outputs = np.zeros((2, size), dtype=np.int)
    for row in range(size):
        index = 1
        for n in matrix[row, :]:
            print(n)


def get_model():
    model = Sequential()
    model.add(Dense(16, input_dim=MATRIX_SIZE*2, activation=activations.relu))
    # model.add(Dense(16, activation=activations.tanh))
    model.add(Dense(MATRIX_SIZE*MATRIX_SIZE, activation=activations.sigmoid))

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=["accuracy"])

    model.summary()


def __main__():
    # model = get_model()
    get_matrix_outputs(dummy)

    # model.fit(, , epochs=EPOCHS, verbose='auto')


__main__()

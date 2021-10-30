import numpy as np
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers.core import Dense, Reshape
from keras import activations

SAMPLES = 1000
EPOCHS = 100

MATRIX_SIZE = 5

dummy = [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]


def get_matrix(size: int):
    return np.random.randint(2, size=(size, size))


def get_results(matrix):
    return


def get_model():
    model = Sequential()
    model.add(Dense(16, input_shape=(MATRIX_SIZE,), activation=activations.tanh))
    model.add(Dense(16, activation=activations.tanh))
    model.add(Dense(MATRIX_SIZE*MATRIX_SIZE, activation=activations.sigmoid))
    model.add(Reshape((MATRIX_SIZE, MATRIX_SIZE)))

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=["accuracy"])

    model.summary()


def __main__():
    model = get_model()

    # model.fit(, , epochs=EPOCHS, verbose='auto')


__main__()

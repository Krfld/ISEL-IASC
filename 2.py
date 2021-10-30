import math
import numpy as np
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers.core import Dense
from keras import activations
from keras import optimizers

SAMPLES = 10000
EPOCHS = 10000

MATRIX_SIZE = 5

dummy = np.array([[1, 1, 0, 1, 0],
                  [1, 1, 1, 0, 1],
                  [1, 0, 1, 0, 1],
                  [1, 0, 1, 0, 0],
                  [0, 1, 0, 1, 0]])

#
#
#


def get_square_matrix(size: int):
    return np.random.randint(2, size=(size, size))


def get_matrix_outputs(matrix):
    size = int(math.sqrt(np.size(matrix)))

    matrix = np.reshape(matrix, (size, size))

    outputs = np.zeros((2, size), dtype=np.int)

    for row in range(size):
        space = True
        for n in matrix[row, :]:
            if n == 0:
                space = True
            elif space == True:
                outputs[0][row] *= 10
                space = False
            outputs[0][row] += n

    for col in range(size):
        space = True
        for n in matrix[:, col]:
            if n == 0:
                space = True
            elif space == True:
                outputs[1][col] *= 10
                space = False
            outputs[1][col] += n

    return np.reshape(outputs, size*2)

#
#
#


def get_model():
    model = Sequential()
    model.add(Dense(16, input_dim=MATRIX_SIZE*2, activation=activations.relu))
    model.add(Dense(16, activation=activations.relu))
    model.add(Dense(16, activation=activations.relu))
    model.add(Dense(MATRIX_SIZE**2, activation=activations.sigmoid))

    model.compile(optimizer=optimizers.adam_v2.Adam(),
                  loss='mean_squared_error',
                  metrics=["accuracy"])

    model.summary()

    return model


def __main__():
    target_data = np.array(
        [np.reshape(get_square_matrix(MATRIX_SIZE), MATRIX_SIZE**2) for i in range(SAMPLES)])
    #print('Target Data\n', target_data)

    train_data = np.array([get_matrix_outputs(target_data[i])
                          for i in range(target_data.shape[0])], dtype=np.int)
    #print('Train Data\n', train_data)

    model = get_model()
    history = model.fit(train_data, target_data,
                        epochs=EPOCHS, verbose='auto', validation_split=1/4)

    # model.predict(test_data).round()

    model.save('models/2', save_format='tf')

    pyplot.plot(history.history['loss'])
    pyplot.show()


__main__()

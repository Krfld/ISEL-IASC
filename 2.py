import math
import numpy as np
from matplotlib import pyplot
from keras.layers.core import Dense
from keras import activations
from keras import optimizers
from keras import models

SAMPLES = 1000000
EPOCHS = 10

MATRIX_SIZE = 5

LOAD_MODEL = False

sample = np.array([[1, 1, 0, 1, 0],
                   [1, 1, 1, 1, 0],
                   [1, 0, 1, 0, 1],
                   [0, 1, 0, 1, 1],
                   [1, 0, 0, 1, 0]])

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
    model = models.Sequential()
    model.add(Dense(64, input_dim=MATRIX_SIZE*2, activation=activations.relu))
    model.add(Dense(64, activation=activations.relu))
    model.add(Dense(MATRIX_SIZE**2, activation=activations.sigmoid))

    model.compile(optimizer=optimizers.adam_v2.Adam(),
                  loss='mean_squared_error',
                  metrics=["accuracy"])

    model.summary()

    return model

#
#
#


def __main__():
    print('Generating target data...')
    target_data = np.array(
        [np.reshape(get_square_matrix(MATRIX_SIZE), MATRIX_SIZE**2) for i in range(SAMPLES)])
    #print('Target Data\n', target_data)

    print('Generating train data...')
    train_data = np.array([get_matrix_outputs(target_data[i])
                          for i in range(target_data.shape[0])], dtype=np.int)
    #print('Train Data\n', train_data)

    if LOAD_MODEL:
        print('Loding model...')
        model = models.load_model('models/2/10k_samples_1k_epochs_64x64')
    else:
        model = get_model()

        print('Training model...')
        history = model.fit(train_data, target_data,
                            epochs=EPOCHS, verbose='auto', validation_split=0.2)

        print(f'Samples: {SAMPLES} | Epochs: {EPOCHS}')

        model.save('models/2/last', save_format='tf')

        pyplot.plot(history.history['loss'])
        pyplot.show()

    result = np.reshape(model.predict(np.array([get_matrix_outputs(
        np.reshape(sample, MATRIX_SIZE**2))])).round(), (MATRIX_SIZE, MATRIX_SIZE))
    print(result)
    print(np.array_equal(result, sample))


__main__()

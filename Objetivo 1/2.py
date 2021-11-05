import math
import time
import numpy as np
import random as rnd
from matplotlib import pyplot
from keras.layers.core import Dense
from keras import activations
from keras import optimizers
from keras import models

MATRIX_SIZE = 4

SAMPLES = 10000  # 2**(MATRIX_SIZE**2-1)
EPOCHS = 1000

TRAIN_TEST_RATIO = 0.8

LOAD_MODEL = False
FOLDER = 'last'

sample = np.array([[1, 1, 0, 1],
                   [1, 0, 1, 1],
                   [1, 0, 1, 0],
                   [0, 1, 0, 1]])

#
#
#


def get_square_matrix():
    return np.random.randint(2, size=(MATRIX_SIZE, MATRIX_SIZE))


def get_data():
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

    return data


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
    model.add(Dense(2**10, input_dim=MATRIX_SIZE * 2, activation=activations.relu))
    model.add(Dense(2**10, activation=activations.relu))
    model.add(Dense(2**10, activation=activations.relu))
    model.add(Dense(MATRIX_SIZE**2, activation=activations.sigmoid))

    model.compile(optimizer=optimizers.adam_v2.Adam(), loss='mean_squared_error', metrics=["accuracy"])

    model.summary()

    return model

#
#
#


def __main__():
    if LOAD_MODEL:
        print('[DEBUG] Loding model...')
        model = models.load_model(f'models/2/{FOLDER}')

    else:
        data = get_data()

        print('[DEBUG] Generating target data...')
        target_data = data[:int(SAMPLES*TRAIN_TEST_RATIO)]
        #print('Target Data\n', target_data)

        print('[DEBUG] Generating train data...')
        train_data = np.array([get_matrix_outputs(m) for m in target_data], dtype=int)
        #print('Train Data\n', train_data)

        model = get_model()

        print('[DEBUG] Training model...')
        start = time.time()
        history = model.fit(train_data, target_data,
                            epochs=EPOCHS,
                            verbose='auto',
                            validation_split=0.2,
                            validation_freq=1,
                            use_multiprocessing=True)

        model.summary()

        print(f'[DEBUG] Took {time.time()-start}s | Samples: {SAMPLES} | Epochs: {EPOCHS}')

        model.save('models/2/last', save_format='tf')

        pyplot.plot(history.history['loss'], label='loss')
        pyplot.plot(history.history['accuracy'], label='accuracy')
        pyplot.plot(history.history['val_loss'], label='val_loss')
        pyplot.plot(history.history['val_accuracy'], label='val_accuracy')
        pyplot.legend()
        pyplot.show()

        test_target_data = data[int(SAMPLES*TRAIN_TEST_RATIO):]
        test_data = np.array([get_matrix_outputs(m) for m in test_target_data], dtype=int)

        print('[DEBUG] Evaluating model...')
        print(model.evaluate(test_data, test_target_data))

    result = np.reshape(
        model.predict(np.array([get_matrix_outputs(np.reshape(sample, MATRIX_SIZE**2))])).round(),
        (MATRIX_SIZE, MATRIX_SIZE)
    )

    print(result)
    print(np.array_equal(result, sample))


__main__()

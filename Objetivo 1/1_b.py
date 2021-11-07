import numpy as np
import random as rnd
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers.core import Dense
from keras import activations
from keras import optimizers
from keras import callbacks
from keras import metrics
from keras.saving import save

#! Use numpy

SAMPLES = 1000
EPOCHS = 100


def __main__():
    #
    # Train data
    #

    a = [1, 1, 1, 1,
         1, 0, 0, 1,
         1, 0, 0, 1,
         1, 1, 1, 1]

    b = [1, 0, 0, 1,
         0, 1, 1, 0,
         0, 1, 1, 0,
         1, 0, 0, 1]

    c = [1, 0, 1, 0,
         0, 1, 0, 1,
         1, 0, 1, 0,
         0, 1, 0, 1]

    d = [1, 0, 1, 0,
         1, 0, 1, 0,
         1, 0, 1, 0,
         1, 0, 1, 0]

    train_data = [a, b, c, d] * 50

    for i in range(SAMPLES - len(train_data)):
        train_data.append([rnd.randint(0, 1) for i in range(16)])

    np.random.shuffle(train_data)

    #print('Training', train_data, len(train_data))

    #
    # Target data
    #

    target_data = []
    for data in train_data:
        if np.array_equal(data, a):
            target_data.append([1, 0, 0, 0])
        elif np.array_equal(data, b):
            target_data.append([0, 1, 0, 0])
        elif np.array_equal(data, c):
            target_data.append([0, 0, 1, 0])
        elif np.array_equal(data, d):
            target_data.append([0, 0, 0, 1])
        else:
            target_data.append([0, 0, 0, 0])

    #print('Target', target_data, len(target_data))

    #
    # Create model
    #

    model = Sequential()
    model.add(Dense(64, input_dim=16, activation=activations.relu))
    model.add(Dense(4, activation=activations.sigmoid))

    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.adam_v2.Adam(learning_rate=0.5),
                  metrics=['accuracy'])

    #
    # Train model
    #

    print('\nTraining model...\n')
    history = model.fit(train_data,
                        target_data,
                        epochs=EPOCHS,
                        verbose='auto',
                        validation_split=0.2,
                        callbacks=[callbacks.EarlyStopping(
                            monitor='val_loss',
                            mode='auto',
                            baseline=0.1)])

    #print('\nEvaluate:', model.evaluate(x=train_data, y=target_data)[0], '\n')

    #
    # Test data
    #

    test_data = train_data.copy()
    np.random.shuffle(test_data)

    #
    # Test model
    #

    predictions = model.predict(test_data).round()

    #
    # Save model
    #

    model.save('models/1_b', save_format='tf')

    #np.savetxt('test_data.txt', np.array(test_data).round(), fmt='%d', delimiter=',')
    #np.savetxt('model_predict.txt', predictions, fmt='%d', delimiter=',')

    print('\nSamples: ', SAMPLES, ' | Epochs: ', EPOCHS, '\n')

    pyplot.plot(history.history['loss'], label='loss')
    pyplot.plot(history.history['accuracy'], label='accuracy')
    pyplot.plot(history.history['val_loss'], label='val_loss')
    pyplot.plot(history.history['val_accuracy'], label='val_accuracy')
    pyplot.legend()
    pyplot.show()


__main__()

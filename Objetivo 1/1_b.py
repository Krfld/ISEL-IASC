
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras import activations
from keras import optimizers

SAMPLES = 10000
EPOCHS = 100


def __main__():
    #
    # Train data
    #

    a = np.array([1, 1, 1, 1,
                  1, 0, 0, 1,
                  1, 0, 0, 1,
                  1, 1, 1, 1])

    b = np.array([1, 0, 0, 1,
                  0, 1, 1, 0,
                  0, 1, 1, 0,
                  1, 0, 0, 1])

    c = np.array([1, 0, 1, 0,
                  0, 1, 0, 1,
                  1, 0, 1, 0,
                  0, 1, 0, 1])

    d = np.array([1, 0, 1, 0,
                  1, 0, 1, 0,
                  1, 0, 1, 0,
                  1, 0, 1, 0])

    train_data = np.concatenate(([a], [b], [c], [d]))

    train_data = np.concatenate(([train_data]*100))

    for i in range(SAMPLES - len(train_data)):
        train_data = np.append(train_data, [[rnd.randint(0, 1) for i in range(16)]], axis=0)

    np.random.shuffle(train_data)

    #print('Training', train_data, len(train_data))

    #
    # Target data
    #

    target_data = np.array([], dtype=int)
    for data in train_data:
        if np.array_equal(data, a):
            target_data = np.append(target_data, [1, 0, 0, 0])
        elif np.array_equal(data, b):
            target_data = np.append(target_data, [0, 1, 0, 0])
        elif np.array_equal(data, c):
            target_data = np.append(target_data, [0, 0, 1, 0])
        elif np.array_equal(data, d):
            target_data = np.append(target_data, [0, 0, 0, 1])
        else:
            target_data = np.append(target_data, [0, 0, 0, 0])
    target_data = np.reshape(target_data, (SAMPLES, 4))

    #print('Target', target_data, len(target_data))

    #
    # Create model
    #

    model = Sequential()
    model.add(Dense(128, input_dim=16, activation=activations.relu))
    model.add(Dense(4, activation=activations.sigmoid))

    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.adam_v2.Adam(),
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
                        use_multiprocessing=True)

    #print('\nEvaluate:', model.evaluate(x=train_data, y=target_data)[0], '\n')

    #
    # Test data
    #

    test_data = np.array(train_data)
    np.random.shuffle(test_data)

    #
    # Test model
    #

    predictions = model.predict(test_data).round()

    #
    # Save model
    #

    model.save('models/1_b', save_format='tf')

    np.savetxt('models/1_b/test_data.txt', np.array(test_data).round(), fmt='%d', delimiter=',')
    np.savetxt('models/1_b/model_predict.txt', predictions, fmt='%d', delimiter=',')

    print('\nSamples: ', SAMPLES, ' | Epochs: ', EPOCHS, '\n')

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.savefig('models/1_b/last.png')
    plt.show()


__main__()

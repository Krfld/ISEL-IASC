import numpy as np
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers.core import Dense
from keras import activations
from keras import optimizers
from keras import callbacks

# the four different states of the XOR gate
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# the four expected results in the same order
target_data = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(2, input_dim=2, activation=activations.tanh))
model.add(Dense(1, activation=activations.tanh))

model.compile(loss='mean_squared_error',
              optimizer=optimizers.gradient_descent_v2.SGD(
                  learning_rate=0.5,
                  momentum=0),
              metrics=None)

history = model.fit(training_data,
                    target_data,
                    shuffle=False,
                    epochs=100,
                    verbose='auto',)
'''callbacks=[callbacks.EarlyStopping(
        monitor='loss',
        verbose='auto',
        mode='max',
        baseline=0.1)])'''

print(model.predict(training_data).round())

# pyplot.plot(history.history['loss'])
# pyplot.show()

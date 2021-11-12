import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras import activations
from keras import optimizers
from keras import callbacks

EPOCHS = 300
HL_NEURONS = 2
LR = 0.1
MOMENTUUM = 0
SHUFFLE = True

LOSS_TARGET = 0.1

# the four different states of the XOR gate
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# the four expected results in the same order
target_data = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(HL_NEURONS, input_dim=2, activation=activations.tanh))
model.add(Dense(1, activation=activations.tanh))

model.compile(loss='mean_squared_error',
              optimizer=optimizers.gradient_descent_v2.SGD(
                  learning_rate=LR,
                  momentum=MOMENTUUM),
              metrics=None)

history = model.fit(training_data,
                    target_data,
                    shuffle=SHUFFLE,
                    epochs=EPOCHS,
                    verbose='auto')

print(model.predict(training_data).round())

plt.plot(history.history['loss'], 'o')
plt.plot([LOSS_TARGET for i in range(EPOCHS)], 'o')
plt.savefig('last.png')
plt.show()

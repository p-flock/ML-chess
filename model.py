from keras.models import Sequential
from keras.layers import Dense, Activation

# neural net structure
model = Sequential()
model.add(Dense(32, input_shape=(71,)))
model.add(Activation('tanh'))
model.add(Dense(1, activation='tanh'))

model.compile(optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 71))
labels = np.random.randint(2, size=(1000, 1))

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)


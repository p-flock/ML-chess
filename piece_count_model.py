from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras import optimizers
from keras import regularizers
from numpy import array
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pickle

model = Sequential()
model.add(Dense(12, input_shape=(12,)))
model.add(Activation('tanh'))
model.add(Dense(24, activation='tanh', ))
model.add(Dense(48, activation='tanh', ))
model.add(Dense(96, activation='tanh', ))
model.add(Dropout(0.2))
model.add(Dense(48, activation='tanh', ))
model.add(Dense(24, activation='tanh', ))
model.add(Dense(12, activation='tanh', ))
model.add(Dense(1, activation='tanh', ))

sgd = optimizers.SGD(lr=0.01, decay=0.002)
model.compile(optimizer=sgd,
        loss='mse',
        metrics=['mse', 'mae'])

vectors = pickle.load(open("piece_count_data.p", "rb"))
data = vectors[0]
y_values = vectors[1]

#scale output data in range (-1,1)
y_min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
labels = y_min_max_scaler.fit_transform(y_values)

history = model.fit(data, labels, epochs=100, batch_size=5000, validation_split=0.01)
model.save("piece_count.h5")

#plot loss and two error functions
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('piece count model mean squared error')
plt.ylabel('mean squared error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('piece count model mean absolute error')
plt.ylabel('mean absolute error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


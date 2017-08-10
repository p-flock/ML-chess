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


# neural net structure
model = Sequential()
model.add(Dense(71, input_shape=(71,)))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(71, activation='tanh', ))
model.add(Dropout(0.15))
model.add(Dense(32, activation='tanh', ))
model.add(Dense(16, activation='tanh', ))
model.add(Dense(8, activation='tanh', ))
# model.add(Dropout(0.05))
model.add(Dense(1, activation='tanh', ))

# rms = RMSprop(lr=0.1,decay=0.002)
sgd = optimizers.SGD(lr=0.01, decay=0.002)
model.compile(optimizer=sgd,
        loss='mse',
        metrics=['mae', 'mse' ])

# load data from pickle file
vectors = pickle.load(open("full_data_set.p", "rb"))

# x_values = array(vectors[0])
# data = np.random.random((1000, 71))
# labels = np.random.randint(2, size=(1000, 1))

# labels = array(vectors[1])
data = (vectors[0])
y_values = array(vectors[1])
# print(data.shape)
# print(labels.shape)
# print(data[1].shape)

#scale y values to -1, 1 range
# x_min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
# data = x_min_max_scaler.fit_transform(x_values)
y_min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
labels = y_min_max_scaler.fit_transform(y_values)

# Train the model, iterating on the data in batches of 32 samples
history = model.fit(data, labels, epochs=100, batch_size=5000, validation_split=0.01)
model.summary()

# save model as json and save weights as h5 file
model.save("chess_model.h5")

#weights
model.save_weights("model_weights.h5")
print("saved model and weights to disk")

# summarize history for accuracy
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('original model mean squared error')
plt.ylabel('mean squared error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('original model mean absolute error')
plt.ylabel('mean absolute error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


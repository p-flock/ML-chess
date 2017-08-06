from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras import regularizers
from numpy import array
from sklearn import preprocessing

# neural net structure
model = Sequential()
model.add(Dense(71, input_shape=(71,)))
model.add(Activation('linear'))
model.add(Dropout(0.2))
model.add(Dense(71, activation='tanh', ))
model.add(Dropout(0.15))
model.add(Dense(32, activation='tanh', ))
model.add(Dense(16, activation='tanh', ))
model.add(Dense(8, activation='tanh', ))
# model.add(Dropout(0.05))
model.add(Dense(1, activation='linear', ))

rms = RMSprop(lr=0.1,decay=0.002)
model.compile(optimizer=rms,
        loss='mae',
        metrics=['accuracy', 'mae'])

# load data from pickle file
import pickle
vectors = pickle.load(open("full_data_set.p", "rb"))

# x_values = array(vectors[0])
# y_values = array(vectors[1])
import numpy as np

# data = np.random.random((1000, 71))
# labels = np.random.randint(2, size=(1000, 1))

labels = array(vectors[1])
data = (vectors[0])
# print(data.shape)
# print(labels.shape)
# print(data[1].shape)

#scale y values to -1, 1 range
# x_min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
# data = x_min_max_scaler.fit_transform(x_values)
# y_min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
# labels = y_min_max_scaler.fit_transform(y_values)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=200, batch_size=5000, validation_split=0.05)
model.summary()

# save model as json and save weights as h5 file
model.save("chess_model.h5")

#weights
model.save_weights("model_weights.h5")
print("saved model and weights to disk")

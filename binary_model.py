from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras import optimizers
from keras import regularizers
from numpy import array
from sklearn import preprocessing
import matplotlib.pyplot as plt


# neural net structure
model = Sequential()
model.add(Dense(71, input_shape=(71,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(71, activation='sigmoid', ))
model.add(Dropout(0.15))
model.add(Dense(32, activation='sigmoid', ))
model.add(Dense(16, activation='sigmoid', ))
model.add(Dense(8, activation='sigmoid', ))
# model.add(Dropout(0.05))
model.add(Dense(1, activation='sigmoid', ))

# rms = RMSprop(lr=0.1,decay=0.002)
# sgd = optimizers.SGD(lr=0.01, decay=0.02)
model.compile(optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'],)

# load data from pickle file
import pickle
vectors = pickle.load(open("full_data_set.p", "rb"))

# x_values = array(vectors[0])
import numpy as np

# data = np.random.random((1000, 71))
# labels = np.random.randint(2, size=(1000, 1))

# labels = array(vectors[1])
data = (vectors[0])
labels = array(vectors[1])
# print(data.shape)
# print(labels.shape)
# print(data[1].shape)
for x in range(len(labels)):
    if labels[x] <= 0:
        labels[x] = 0
    else:
        labels[x] = 1

#scale y values to -1, 1 range
# x_min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
# data = x_min_max_scaler.fit_transform(x_values)
# y_min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# labels = y_min_max_scaler.fit_transform(y_values)

# Train the model, iterating on the data in batches of 32 samples
history = model.fit(data, labels, epochs=100, batch_size=5000, validation_split=0.01)
model.summary()

# save model as json and save weights as h5 file
model.save("chess_model.h5")

#weights
model.save_weights("model_weights.h5")
print("saved model and weights to disk")

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

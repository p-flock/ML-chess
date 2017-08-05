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
import pickle
vectors = pickle.load(open("training_set_3.p", "rb"))
data = vectors[0]
labels = vectors[1]


# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=200, batch_size=32)
model.summary()

# save model as json and save weights as h5 file
model.save("chess_model.h5")

#weights
model.save_weights("model_weights.h5")
print("saved model and weights to disk")

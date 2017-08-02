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
vectors = pickle.load(open("first_training_set.p"))
data = vectors[0]
labels = vectors[1]


# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)
model.summary()

# save model as json and save weights as h5 file
# model_as_json = model.to_json()
# with open("chess_model.json", "r") as json_file:
    # json_file.write(model_as_json)
model.save("chess_model.h5")

#weights
model.save_weights("model_weights.h5")
print("saved model and weights to disk")

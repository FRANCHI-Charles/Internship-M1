import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


print("Loading TensorFlow...")
import tensorflow as tf
import keras

from tfnameutils import N_LETTERS, load_data, train_test_split, preprocessing

### Hyperparameters
HIDDEN_SIZE = 20

LR = 0.005
N_EPOCH = 15
BATCH_SIZE = 150
TESTSIZE = 0.25

### Load the data

print("Loading data...")
names, labels, categories = load_data()
names = preprocessing(names)
print(len(names), len(labels))
maxsize = names.shape[1]

trainX, trainy, testX, testy = train_test_split(names, labels, testprop=0.3)




### model

print("Loading model...")
model = keras.Sequential()
model.add(keras.layers.Input((maxsize, N_LETTERS)))
model.add(keras.layers.Masking(mask_value=2))
model.add(keras.layers.SimpleRNN(HIDDEN_SIZE, activation="tanh", return_sequences=True))
model.add(keras.layers.SimpleRNN(HIDDEN_SIZE, activation="tanh"))
model.add(keras.layers.Dense(len(categories)))
model.add(keras.layers.Softmax())

loss = keras.losses.SparseCategoricalCrossentropy()
optim = keras.optimizers.Adam(learning_rate=LR)

model.compile(optim, loss, metrics=["accuracy"])

model.summary()


### training

print("Done. Start training...")
model.fit(trainX, trainy, batch_size=BATCH_SIZE, epochs = N_EPOCH, verbose = 1)

model.evaluate(testX, testy, batch_size = BATCH_SIZE, verbose=2)
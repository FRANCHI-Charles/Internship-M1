import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import keras

from tfnameutils import N_LETTERS, line_to_tensor_size, load_transform

### Hyperparameters
MAXSIZE = 19 # maximum length of a name in the dataset (print(findmax(category_lines)) answer is 19)

HIDDEN_SIZE = 20

LR = 0.01
N_EPOCH = 15
BATCH_SIZE = 150
TESTSIZE = 0.25

### Load the data

names, labels, categories = load_transform(MAXSIZE, 500)
print(len(names), len(labels))

#train test split

### model

model = keras.Sequential()
model.add(keras.layers.Input((MAXSIZE, N_LETTERS), sparse=True))
model.add(keras.layers.SimpleRNN(HIDDEN_SIZE, activation="sigmoid"))
model.add(keras.layers.Dense(len(categories)))

loss = keras.losses.SparseCategoricalCrossentropy()
optim = keras.optimizers.Adam(learning_rate=LR)

model.compile(optim, loss, metrics=["accuracy"])

model.summary()


### training

model.fit("something...")
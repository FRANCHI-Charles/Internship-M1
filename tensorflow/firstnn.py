import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import keras

mnist = keras.datasets.mnist

(trainx, trainy), (testx, testy) = mnist.load_data()

trainx = trainx / 255 # normalization
testx = testx / 255

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation = "relu"),
    keras.layers.Dense(10, activation = "relu")
])

model.summary()

loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True)
optim = keras.optimizers.Adam(learning_rate=0.005)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer = optim, metrics = metrics)

# trainings

model.fit(trainx, trainy, batch_size = 64, epochs = 3, shuffle=True, verbose = 2)

model.evaluate(testx, testy, batch_size = 64, verbose = 2)

predictions = tf.math.argmax(model(testx), axis=1) #tf.nn.softmax(model(testx), axis=1)

print(predictions, testy)



#saving 
model.save("firstnn.keras", overwrite = True)

loaded_model = keras.saving.load_model("firstnn.keras")

#or
model.save_weights("firstnnweights.weights.h5")
model.load_weights("firstnnweights.weights.h5")

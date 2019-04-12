import tensorflow as tf

mnist = tf.keras.datasets.mnist #28x28 images of handwritten digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))
model.compile (
    optimizer = "adam" ,
    loss = "sparse_categorical_crossentropy",
    metrics = ['acuracy']
)
model.fit(x_train, y_train, epochs = 3)



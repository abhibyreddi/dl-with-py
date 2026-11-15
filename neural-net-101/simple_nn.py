"""Super simple neural net showing the building blocks involved."""
from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical

# 0. Load the train and test sets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 1. Construct the nueural net with 2 layers
neural_net = models.Sequential()
# First layer is a dense layer with 512 neurons which means that
# each neuron in this layer connects to all neurons in the next layer
neural_net.add(layers.Dense(512, activation="relu", input_shape=(784, )))
# Second layer is a 10 way 'softmax' layer which means that
# it will return an array of 10 probability scores that add up to 1.
# Each neuron corresponds to 1 digit in [0, 10]
neural_net.add(layers.Dense(10, activation="softmax"))

# 2. Compile the neural net - configure its loss function, optimizer, and metrics
neural_net.compile(optimizer="rmsprop",
                   metrics=["accuracy"],
                   loss="categorical_crossentropy")

# 3.1 Reshape the train and test image sets
# convert them from 28 X 28 to 784 X 1.
train_images = train_images.reshape((60000, 28 * 28))
test_images = test_images.reshape((10000, 28 * 28))

# 3.2 Scale the numbers inside the images so they work well
# with the activation and loss functions.
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# 4. Categorically encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# TRAIN!!!
neural_net.fit(train_images, train_labels, epochs=10, batch_size=128)

# Evaluatre
loss, accuracy = neural_net.evaluate(test_images, test_labels)
print(f'loss: {loss}\taccuracy: {accuracy}')

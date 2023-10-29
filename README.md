# To implement multilayer perceptron 

``` python
import numpy as np
from sklearn.neural_network import MLPClassifier

# Generate XOR training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create a multilayer perceptron classi3ier
clf = MLPClassifier(hidden_layer_sizes=(3), activation='relu')

# Train the classifier
clf.fit(X, y)

# Make predictions on the training data
y_pred = clf.predict(X)

# Print the predictions
print(y_pred)

```
# implement gradient descent
```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate random data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)

# Define a linear regression model
model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
model.compile(optimizer='sgd',loss='mean_squared_error')
his = model.fit(X,y,epochs=10,batch_size = 10, verbose=0).history['loss']
plt.plot(his)

# Get the learned parameters
theta = model.get_weights()[0]
print("Final theta:", theta)

```

# implement Backpropagation
```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate random data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)

# Define a linear regression model
model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Training parameters
epochs = 10

# Lists to store loss and parameter values
loss_history = []
theta_history = []

# Training loop
for epoch in range(epochs):
    # Forward pass
    predictions = model.predict(X)
    loss = model.evaluate(X, y)

    # Backpropagation
    model.fit(X, y, epochs=1)

    # Store loss and parameter values
    loss_history.append(loss)
    theta_history.append(model.get_weights()[0][0][0])

# Print the final parameters
theta = model.get_weights()[0]
print("Final theta:", theta)

# Plot the loss and parameter updates
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Over Iterations')

plt.subplot(1, 2, 2)
plt.plot(theta_history)
plt.xlabel('Iterations')
plt.ylabel('Theta Value')
plt.title('Theta Value Over Iterations')

plt.tight_layout()
plt.show()

```

# autoencoder
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load the MNIST dataset and add random noise to the images
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train_noisy = x_train + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Define the denoising autoencoder architecture
input_img = Input(shape=(28, 28, 1))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the denoising autoencoder

autoencoder.fit(x_train_noisy, x_train, epochs=5, batch_size=128, shuffle=True, validation_data=(x_test_noisy, x_test))

# Test the denoising autoencoder by removing noise from test images
decoded_images = autoencoder.predict(x_test_noisy)

# Display original, noisy, and denoised images
n = 10  # Number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Noisy images
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Denoised images
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(decoded_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

```

# to implement cnn
```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)  # Number of classes in CIFAR-10
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Plot the training accuracy and loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

```
# to implement lstm
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate some sample data for a sequence prediction task
np.random.seed(0)
sequence_length = 100
X = np.arange(sequence_length)
y = X + np.sin(X) + np.random.normal(0, 0.5, sequence_length)

# Split the data into input sequences and corresponding output values
sequence_length = 10  # Length of input sequences
X_data, y_data = [], []
for i in range(len(X) - sequence_length):
    X_data.append(X[i:i + sequence_length])
    y_data.append(y[i + sequence_length])

X_data = np.array(X_data)
y_data = np.array(y_data)

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(len(X_data) * split_ratio)
X_train, X_test = X_data[:split_index], X_data[split_index:]
y_train, y_test = y_data[:split_index], y_data[split_index:]

# Reshape the data for the LSTM model
X_train = X_train.reshape(-1, sequence_length, 1)
X_test = X_test.reshape(-1, sequence_length, 1)

# Define the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, verbose=0)

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss:.4f}")

# Make predictions using the trained model
y_pred = model.predict(X_test)

# Plot the true values and predictions
plt.figure(figsize=(10, 5))
plt.plot(X[split_index + sequence_length:], y_test, label='True Values')
plt.plot(X[split_index + sequence_length:], y_pred, label='Predictions')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('LSTM Predictions')
plt.show()

```

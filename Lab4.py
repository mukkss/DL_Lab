import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Image Dataset (MNIST)
# -----------------------------
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape
x_train = x_train.astype("float32") / 255.
x_test  = x_test.astype("float32") / 255.

# Add channel dimension (28,28,1)
x_train = np.expand_dims(x_train, axis=-1)
x_test  = np.expand_dims(x_test, axis=-1)

# -----------------------------
# 2. Build Autoencoder Network
# -----------------------------

# Encoder
encoder_input = layers.Input(shape=(28, 28, 1))
x = layers.Flatten()(encoder_input)
x = layers.Dense(128, activation='relu')(x)
compressed = layers.Dense(32, activation='relu')(x)     # compressed representation

# Decoder
x = layers.Dense(128, activation='relu')(compressed)
x = layers.Dense(28 * 28, activation='sigmoid')(x)
decoder_output = layers.Reshape((28, 28, 1))(x)

# Combine encoder + decoder
autoencoder = models.Model(encoder_input, decoder_output)

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()

# -----------------------------
# 3. Train Autoencoder
# -----------------------------
history = autoencoder.fit(
    x_train, x_train,        # input = output (self reconstruction)
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

# -----------------------------
# 4. Compress & Reconstruct
# -----------------------------
encoded_imgs = autoencoder.layers[3].output
decoded_imgs = autoencoder.predict(x_test)

# -----------------------------
# 5. Display Original vs Reconstructed
# -----------------------------
n = 10  # show 10 images
plt.figure(figsize=(18, 4))

for i in range(n):
    # Original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis("off")

    # Reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis("off")

plt.show()

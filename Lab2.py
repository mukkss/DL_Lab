import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import confusion_matrix, f1_score

# ---------------------------
# 1. Load and Preprocess Data
# ---------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize & Flatten images
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# ---------------------------
# 2. Build Deep Neural Network
# ---------------------------
model = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------
# 3. Train the Model
# ---------------------------
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

# ---------------------------
# 4. Plot Training Curves
# ---------------------------
history_df = pd.DataFrame(history.history)

plt.figure(figsize=(10, 4))
history_df[['loss', 'val_loss']].plot()
plt.title("Loss vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.figure(figsize=(10, 4))
history_df[['accuracy', 'val_accuracy']].plot()
plt.title("Accuracy vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# ---------------------------
# 5. Model Predictions
# ---------------------------
y_predicted = model.predict(x_test)
y_predicted_classes = np.argmax(y_predicted, axis=1)

print("y_test shape:", y_test.shape)
print("y_predicted_classes shape:", y_predicted_classes.shape)

# ---------------------------
# 6. Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_test, y_predicted_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ---------------------------
# 7. F1 Score
# ---------------------------
f1score = f1_score(y_test, y_predicted_classes, average='weighted')
print(f"\nWeighted F1 Score: {f1score:.4f}")

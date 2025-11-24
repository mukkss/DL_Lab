# ============================================
# 1. IMPORT LIBRARIES
# ============================================
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import kagglehub

# ============================================
# 2. DOWNLOAD CATS & DOGS DATASET (KAGGLEHUB)
# ============================================
path = kagglehub.dataset_download("tongpython/cat-and-dog")
print("Downloaded to:", path)

train_dir = os.path.join(path, "training_set")
test_dir  = os.path.join(path, "test_set")

# ============================================
# 3. DATA GENERATORS
# ============================================
img_size = (150, 150)
batch_size = 32

train_gen = ImageDataGenerator(rescale=1/255)
test_gen  = ImageDataGenerator(rescale=1/255)

train_data = train_gen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode="binary"
)

test_data = test_gen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode="binary"
)

# ============================================
# 4. BUILD THE CNN MODEL
# ============================================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()

# ============================================
# 5. TRAIN THE MODEL
# ============================================
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)

# ============================================
# 6. PLOT TRAINING CURVES
# ============================================
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label="Train")
plt.plot(history.history['val_accuracy'], label="Val")
plt.title("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label="Train")
plt.plot(history.history['val_loss'], label="Val")
plt.title("Loss")
plt.legend()

plt.show()

# ============================================
# 7. CONFUSION MATRIX + REPORT
# ============================================
test_labels = test_data.classes
pred_probs = model.predict(test_data)
pred_classes = (pred_probs > 0.5).astype("int")

print("\nClassification Report:")
print(classification_report(test_labels, pred_classes))

cm = confusion_matrix(test_labels, pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ============================================
# 8. PREDICT A SINGLE IMAGE
# ============================================
from tensorflow.keras.utils import load_img, img_to_array

sample_image = os.path.join(test_dir, "cats", "cat.1001.jpg")

img = load_img(sample_image, target_size=img_size)
img_arr = img_to_array(img) / 255.0
img_arr = np.expand_dims(img_arr, axis=0)

pred = model.predict(img_arr)[0][0]

if pred > 0.5:
    print("Prediction: 🐶 Dog")
else:
    print("Prediction: 🐱 Cat")

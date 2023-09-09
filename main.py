import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pyheif
from PIL import Image


# 1. Loading and Resizing Images
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if filepath.lower().endswith('.heic'):
            heif_file = pyheif.read(filepath)
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(filepath)

        if img is not None:
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label)
    return images, labels

camera_present_images, camera_present_labels = load_images_from_folder("datasets/camera_present", 1)
no_camera_images, no_camera_labels = load_images_from_folder("datasets/no_camera_present", 0)

# 2. Data Normalization
X_data = np.array(camera_present_images + no_camera_images, dtype=np.float32) / 255.0
y_data = np.array(camera_present_labels + no_camera_labels)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# 4. Data Augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# 5. Model Architecture
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(2, activation='softmax')
# ])

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
# model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# 6. Early Stopping and Checkpoints
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)

# Manually split the training data into training and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create data generators for training and validation
train_gen = datagen.flow(X_train, y_train)
val_gen = datagen.flow(X_val, y_val)

# Model Training
history = model.fit(train_gen,
                    epochs=20,
                    validation_data=val_gen,
                    callbacks=[early_stopping, model_checkpoint])

# 8. Model Summary
model.summary()

# 9. Model Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy * 100}%')

# 10. Model Conversion to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 11. Save the model
with open('hidey_camguard.tflite', 'wb') as f:
    f.write(tflite_model)

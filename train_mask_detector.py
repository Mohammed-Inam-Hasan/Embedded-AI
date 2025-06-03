from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Constants
INIT_LR = 1e-4
EPOCHS = 10
BS = 32
DIRECTORY = "dataset"
CATEGORIES = ["with_mask", "without_mask"]

# Load images and labels
print("[INFO] loading images...")
data, labels = [], []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))
        data.append(image)
        labels.append(category)

# Normalize and encode
data = np.array(data) / 255.0
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2)

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# Build model
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

print("[INFO] compiling model...")
model.compile(optimizer=Adam(learning_rate=INIT_LR),
              loss="binary_crossentropy", metrics=["accuracy"])

print("[INFO] training model...")
model.fit(aug.flow(trainX, trainY, batch_size=BS),
          steps_per_epoch=len(trainX) // BS,
          validation_data=(testX, testY),
          validation_steps=len(testX) // BS,
          epochs=EPOCHS)

print("[INFO] saving mask detector model...")
model.save("mask_detector.h5")

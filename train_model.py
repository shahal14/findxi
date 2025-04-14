import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from tensorflow.keras.applications import Xception
import matplotlib.pyplot as plt

def load_images_from_folder(folder, label, size=(224, 224)):
    images, labels = [], []
    folder_path = Path(folder)  # Convert to Path object for consistency
    if not folder_path.exists():
        print(f"Error: Folder '{folder}' does not exist.")
        return np.array([]), np.array([])

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, size) / 255.0
            images.append(img)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Paths to your data
real_images_path = 'C:/Users/shibi/OneDrive/Documents/my mini project/real image'
ai_images_path = 'C:/Users/shibi/OneDrive/Documents/my mini project/ai image'

# Load real and AI-generated images
real_images, real_labels = load_images_from_folder(real_images_path, label=0)  # Label 0 for real
ai_images, ai_labels = load_images_from_folder(ai_images_path, label=1)  # Label 1 for AI-generated

# Check if images are loaded correctly
if real_images.size == 0 or ai_images.size == 0:
    raise ValueError("Error: One or both image datasets are empty. Check your folder paths.")

# Combine data
X = np.concatenate((real_images, ai_images), axis=0)
y = np.concatenate((real_labels, ai_labels), axis=0)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build and train the model
model = build_model()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

# Save the model
model.save('ai_image_detector.h5')
print("Model saved successfully.")

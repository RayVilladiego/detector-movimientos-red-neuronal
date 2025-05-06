import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from google.colab import files  # Solo se ejecuta en Colab

# Rutas
DATA_DIR = "data"
CATEGORIES = os.listdir(DATA_DIR)
IMG_SIZE = 64

# Cargar datos
X = []
y = []

for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category)
    label = CATEGORIES.index(category)
    for img_name in os.listdir(path):
        try:
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)
        except Exception as e:
            print(f"Error con {img_path}: {e}")

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = to_categorical(y, num_classes=len(CATEGORIES))

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(CATEGORIES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Guardar modelo
os.makedirs("model", exist_ok=True)
model.save("model/gesture_model.h5")

# Descargar (solo en Colab)
files.download("model/gesture_model.h5")

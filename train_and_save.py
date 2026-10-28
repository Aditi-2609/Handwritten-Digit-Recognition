"""Run once (locally or in Colab): trains the ANN and saves what the web app needs.

Outputs:
  models/ann.keras        - trained ANN
  models/knn_data.npz     - the 10,000 MNIST samples used by KNN (uint8, ~8 MB)
"""
import os
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

os.makedirs("models", exist_ok=True)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_flat = (X_train / 255.0).reshape(-1, 784).astype("float32")

ann = Sequential([
    Input(shape=(784,)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax"),
])
ann.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
ann.fit(X_flat, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

test_acc = ann.evaluate((X_test / 255.0).reshape(-1, 784), y_test, verbose=0)[1]
print(f"ANN test accuracy: {test_acc:.4f}")
ann.save("models/ann.keras")

# Same 10,000-sample subset as the notebook. Stored as uint8 to keep the file small.
np.savez_compressed("models/knn_data.npz", X=X_train[:10000].reshape(-1, 784), y=y_train[:10000])
print("Saved models/ann.keras and models/knn_data.npz")

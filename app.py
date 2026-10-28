import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import keras

THRESHOLD = 0.8  # same confidence cutoff as the notebook
BASE_DIR = Path(__file__).resolve().parent

ann = keras.models.load_model(BASE_DIR / "models" / "ann.keras")

data = np.load(BASE_DIR / "models" / "knn_data.npz")
X_knn_u8, y_knn = data["X"], data["y"]
knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean", n_jobs=1)
knn.fit(X_knn_u8 / 255.0, y_knn)

app = FastAPI(title="Hybrid digit recognizer")


class Drawing(BaseModel):
    pixels: list[float]  # 784 values in [0, 1], MNIST-style (white digit on black)


@app.post("/api/predict")
def predict(d: Drawing):
    if len(d.pixels) != 784:
        raise HTTPException(400, "Expected 784 pixel values")
    x = np.asarray(d.pixels, dtype="float32").reshape(1, 784)

    probs = ann(x, training=False).numpy()[0]
    ann_pred, confidence = int(probs.argmax()), float(probs.max())

    dist, idx = knn.kneighbors(x, n_neighbors=3)
    knn_pred = int(knn.predict(x)[0])

    used_ann = confidence > THRESHOLD
    return {
        "digit": ann_pred if used_ann else knn_pred,
        "decided_by": "ANN" if used_ann else "KNN",
        "confidence": confidence,
        "ann_pred": ann_pred,
        "knn_pred": knn_pred,
        "probs": probs.tolist(),
        "neighbors": [
            {"label": int(y_knn[i]), "dist": float(dd), "pixels": X_knn_u8[i].tolist()}
            for i, dd in zip(idx[0], dist[0])
        ],
    }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "index.html")

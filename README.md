# Handwritten Digit Recognizer

A small FastAPI app that recognizes hand-drawn digits with a neural network and falls back to a nearest-neighbor vote when the network is uncertain. The interface also shows class probabilities and the closest MNIST examples.

## Requirements

- Python 3.11 or 3.13
- Dependencies from `requirements.txt`

The trained model and KNN sample data are included in `models/`, so the app is ready to run after installing dependencies. To regenerate them, run `python train_and_save.py`; this downloads MNIST and trains the network.

## Run locally

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 7860
```

Open <http://127.0.0.1:7860>.

## Run with Docker

```powershell
docker build -t handwritten-digit-recognizer .
docker run --rm -p 7860:7860 handwritten-digit-recognizer
```

Open <http://127.0.0.1:7860>.
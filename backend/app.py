import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFile, UnidentifiedImageError
from transformers import AutoImageProcessor, BeitModel

ImageFile.LOAD_TRUNCATED_IMAGES = False

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "final_random_forest.pkl"
FEATURES_PATH = MODEL_DIR / "final_selected_features.npy"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

MODEL_ID = os.getenv("BEIT_MODEL_ID", "microsoft/beit-base-patch16-224")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# The model was trained with scikit-learn 1.6.1. Keeping that version in
# requirements.txt avoids pickle compatibility problems during deployment.

if not MODEL_PATH.exists() or not FEATURES_PATH.exists() or not METADATA_PATH.exists():
    raise RuntimeError("Required model files are missing from /model")

with METADATA_PATH.open("r", encoding="utf-8") as f:
    METADATA = json.load(f)

# The saved .npy contains names such as "feature_41", not raw integer indices.
# Convert those names back to their original BEiT feature positions.
raw_selected = np.load(FEATURES_PATH, allow_pickle=False)
SELECTED_INDICES = np.array(
    [int(re.fullmatch(r"feature_(\d+)", str(name)).group(1)) for name in raw_selected],
    dtype=np.int64,
)

if SELECTED_INDICES.shape != (200,):
    raise RuntimeError(f"Expected 200 selected features, found {SELECTED_INDICES.shape}")
if SELECTED_INDICES.min() < 0 or SELECTED_INDICES.max() >= 768:
    raise RuntimeError("Selected feature indices are outside the 768-dimensional BEiT feature space")

rf_model = joblib.load(MODEL_PATH)
if getattr(rf_model, "n_features_in_", None) != 200:
    raise RuntimeError(f"Random Forest expects {getattr(rf_model, 'n_features_in_', 'unknown')} features, not 200")

# Load the same pretrained BEiT representation used during feature extraction.
image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
beit_model = BeitModel.from_pretrained(MODEL_ID)
beit_model.eval()
DEVICE = torch.device("cpu")
beit_model.to(DEVICE)

app = FastAPI(
    title="OralScan API",
    description="Research/educational Normal vs OSCC histopathology classifier using BEiT + SHAP-selected features + Random Forest.",
    version="1.0.0",
)

allowed_origins_env = os.getenv(
    "ALLOWED_ORIGINS",
    "https://shubham132328.github.io,http://localhost:5500,http://127.0.0.1:5500,http://localhost:8000,http://127.0.0.1:8000",
)
ALLOWED_ORIGINS = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def extract_beit_features(image: Image.Image) -> np.ndarray:
    """Return one 768-D BEiT CLS-token representation as float32."""
    image = image.convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.inference_mode():
        outputs = beit_model(pixel_values=pixel_values)
        features = outputs.last_hidden_state[:, 0, :]

    return features.detach().cpu().numpy().astype(np.float32)


def predict_image(image: Image.Image) -> tuple[str, float]:
    full_features = extract_beit_features(image)  # shape: (1, 768)
    selected_features = full_features[:, SELECTED_INDICES]  # shape: (1, 200)

    probabilities = rf_model.predict_proba(selected_features)[0]
    predicted_class = int(rf_model.classes_[int(np.argmax(probabilities))])
    confidence = float(np.max(probabilities))

    label = "OSCC" if predicted_class == 1 else "Normal"
    return label, confidence


@app.get("/")
def root():
    return {
        "service": "OralScan API",
        "status": "online",
        "task": "Normal vs OSCC histopathology classification",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "BEiT + SHAP-selected 200 features + Random Forest",
        "beit_model": MODEL_ID,
        "selected_features": int(len(SELECTED_INDICES)),
        "rf_trees": int(rf_model.n_estimators),
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are accepted.")

    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image must be {MAX_UPLOAD_MB} MB or smaller.")

    try:
        from io import BytesIO
        pil_image = Image.open(BytesIO(data))
        pil_image.verify()
        pil_image = Image.open(BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid readable image.")

    try:
        label, confidence = predict_image(pil_image)
    except Exception as exc:
        # Do not expose internal model paths/tracebacks to the browser.
        print(f"Prediction error: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Model inference failed on the server.")

    return {
        "result": label,
        "confidence": round(confidence, 4),
        "model": "BEiT + SHAP-selected 200 features + Random Forest",
    }

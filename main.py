import io
import json
import os
from typing import Optional
import httpx
import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ResNet50 AI Engine (ONNX Lightweight)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TARGET_SERVER_URL = os.getenv("TARGET_SERVER_URL", "https://res-q-omega-two.vercel.app/api/sos/submit")

# Load lightweight ONNX model into memory (<150MB RAM)
ort_session = ort.InferenceSession("model.onnx")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocesses image to match ResNet50 input normalization without PyTorch."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    img_data = np.array(img).astype(np.float32) / 255.0
    
    # Normalize with ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    
    # Rearrange dimensions from (H, W, C) to (C, H, W) and add batch dimension (1, C, H, W)
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    return img_data


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def safe_float(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    lat: Optional[str] = Form(None),
    lon: Optional[str] = Form(None),
    lng: Optional[str] = Form(None)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    parsed_lat = None
    parsed_lon = None

    if os.path.exists("report.json"):
        try:
            with open("report.json", "r") as f:
                report_data = json.load(f)
                loc = report_data.get("location") or {}
                parsed_lat = safe_float(loc.get("lat"))
                parsed_lon = safe_float(loc.get("lng") if loc.get("lng") is not None else loc.get("lon"))
        except Exception as e:
            print(f"⚠️ Error reading report.json: {e}")

    if parsed_lat is None:
        parsed_lat = safe_float(lat)

    if parsed_lon is None:
        longitude = lon if lon is not None else lng
        parsed_lon = safe_float(longitude)

    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)

    # ONNX Inference
    outputs = ort_session.run(None, {"input": input_tensor})[0]
    probs = softmax(outputs)
    predicted_class = int(np.argmax(probs, axis=1)[0])
    confidence = float(probs[0][predicted_class]) * 100.0

    is_verified_real = (predicted_class == 1) and (confidence >= 75.0)

    dispatch_status = "NOT_DISPATCHED"
    target_response = None

    if is_verified_real:
        dispatch_payload = {
            "verdict": "REAL",
            "confidence": round(confidence, 2),
            "lat": parsed_lat,
            "lon": parsed_lon
        }
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                res = await client.post(TARGET_SERVER_URL, json=dispatch_payload)
                if res.status_code == 200:
                    dispatch_status = "DISPATCHED"
                    try:
                        target_response = res.json()
                    except Exception:
                        target_response = res.text
                else:
                    dispatch_status = f"FAILED_HTTP_{res.status_code}"
                    target_response = res.text
        except Exception as forward_err:
            dispatch_status = f"FORWARD_ERROR: {str(forward_err)}"

    return {
        "verdict": "REAL" if is_verified_real else "FAKE",
        "confidence": round(confidence, 2),
        "raw_class": "REAL" if predicted_class == 1 else "FAKE",
        "lat": parsed_lat,
        "lon": parsed_lon,
        "dispatch_status": dispatch_status,
        "target_response": target_response
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
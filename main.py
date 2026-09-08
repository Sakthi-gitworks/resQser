import io
import json
import os
from typing import Optional
import httpx
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ResNet50 AI Verification Engine")

# CORS setup to allow request origins from GitHub Pages or any browser frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TARGET_SERVER_URL = os.getenv("TARGET_SERVER_URL", "https://res-q-omega-two.vercel.app/api/sos/submit")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

try:
    model.load_state_dict(torch.load("custom_accident_model.pth", map_location=device))
    print(f"✅ Loaded model weights successfully on {device}")
except Exception as e:
    print(f"⚠️ Warning loading model: {e}")

model.to(device)
model.eval()

ai_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def safe_float(val: Optional[str]) -> Optional[float]:
    """Safely converts string inputs to float without throwing ValueError on invalid text."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


@app.get("/")
async def root():
    return {
        "status": "online",
        "device": str(device),
        "target_url": TARGET_SERVER_URL
    }


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

    # 1. Parse location from report.json
    if os.path.exists("report.json"):
        try:
            with open("report.json", "r") as f:
                report_data = json.load(f)
                loc = report_data.get("location") or {}
                parsed_lat = safe_float(loc.get("lat"))
                parsed_lon = safe_float(loc.get("lng") if loc.get("lng") is not None else loc.get("lon"))
        except Exception as e:
            print(f"⚠️ Error reading report.json: {e}")

    # 2. Fallback: Use form data parameters if report.json missing/unreadable
    if parsed_lat is None:
        parsed_lat = safe_float(lat)

    if parsed_lon is None:
        longitude = lon if lon is not None else lng
        parsed_lon = safe_float(longitude)

    print("\n" + "=" * 45)
    print(f"📍 LOCATION CAPTURED -> Lat: {parsed_lat}, Lon: {parsed_lon}")
    print("=" * 45 + "\n")

    # Read image content
    image_bytes = await file.read()
    raw_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor_img = ai_transform(raw_img).unsqueeze(0).to(device)

    # 3. Run Inference
    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100.0

    is_verified_real = (predicted_class == 1) and (confidence >= 75.0)

    dispatch_status = "NOT_DISPATCHED"
    target_response = None

    # 4. Forward verified alert payload to target server
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
            print(f"⚠️ Could not forward payload to {TARGET_SERVER_URL}: {forward_err}")
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
    # Automatically uses environment PORT (for cloud services) or defaults to 8000
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
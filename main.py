"""
ResQ / Golden Hour — unified AI verification + dispatch server.

This replaces two half-systems with one:

  * the ONNX accident-verification backend your web page already talks to, and
  * the dispatch hub the ambulance / hospital / police apps read from.

Previously the verification backend ran the model, forwarded the result to a
separate service, and then deleted the session — so nothing was left for any
responder to read. Here a verified incident is stored, served to the responder
apps, and kept until it is cleared.

Endpoints
---------
Web page (unchanged, your existing app.js keeps working):
    POST /api/session/start
    POST /api/location
    POST /api/upload/{category}
    POST /api/finalize

One-shot submit (multipart: lat, lon, photo):
    POST /api/sos/submit

Responder apps:
    GET  /api/alerts/ambulance | hospital | police
    GET  /api/alerts
    GET  /api/alerts/{id}/photo
    POST /api/alerts/{id}/status
    DELETE /api/alerts/{id}

Testing:
    POST /api/demo/seed        create a sample incident, no photo needed
    GET  /healthz
"""

import io
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

app = FastAPI(title="ResQ AI Dispatch Hub", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Optional: also forward verified incidents somewhere else. Left unset by
# default because the old Vercel target returns 500 on every route, which
# silently swallowed every dispatch.
TARGET_SERVER_URL = os.getenv("TARGET_SERVER_URL", "").strip()

MODEL_PATH = os.getenv("MODEL_PATH", "model.onnx")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "75.0"))
ACCIDENT_CLASS_INDEX = int(os.getenv("ACCIDENT_CLASS_INDEX", "1"))

# How pixels are scaled before inference. This MUST match how the model was
# trained - a mismatch makes the model emit confident nonsense (a photo of a
# desk scoring 99.9% "accident" is the classic symptom).
#   imagenet : x/255 then (x-mean)/std   (torchvision / timm default)
#   unit     : x/255                     (keras Rescaling(1./255))
#   signed   : x/127.5 - 1               (mobilenet, inception)
#   raw      : 0-255 untouched
NORMALIZE = os.getenv("NORMALIZE", "imagenet").strip().lower()

# Normally read off the model itself. Set these only if the ONNX export left
# the input shape dynamic.
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "0"))                    # 0 = auto
INPUT_LAYOUT = os.getenv("INPUT_LAYOUT", "auto").strip().lower()  # auto|nchw|nhwc

# Many exports already end in softmax/sigmoid. Applying softmax a second time
# flattens the distribution so nothing clears the threshold.
APPLY_SOFTMAX = os.getenv("APPLY_SOFTMAX", "auto").strip().lower()  # auto|yes|no

# Render's free tier caps memory, so keep only the most recent incidents and
# photos in RAM.
MAX_INCIDENTS = int(os.getenv("MAX_INCIDENTS", "50"))
MAX_PHOTO_BYTES = int(os.getenv("MAX_PHOTO_BYTES", str(6 * 1024 * 1024)))

ROLES = ("ambulance", "hospital", "police")

# --------------------------------------------------------------------------
# Model loading - optional, so the service still boots without model.onnx
# --------------------------------------------------------------------------

ort_session = None
MODEL_ERROR: Optional[str] = None
_model_input_name = "input"
_model_input_shape = None
_model_output_shape = None
INPUT_H, INPUT_W, LAYOUT = 224, 224, "nchw"


def _shape_of(node):
    try:
        return [d if isinstance(d, int) else str(d) for d in node.shape]
    except Exception:
        return None


def _detect_input_geometry(shape):
    """Derive (height, width, layout) from the model's declared input shape.

    A 4-D input is either (N,3,H,W) or (N,H,W,3); whichever axis is 3 tells us
    the layout, and the other two give the size.
    """
    if not shape or len(shape) != 4:
        return 224, 224, "nchw"
    dims = [d if isinstance(d, int) else None for d in shape]
    _, a, b, c = dims
    if a == 3:
        return (b or 224), (c or 224), "nchw"
    if c == 3:
        return (a or 224), (b or 224), "nhwc"
    return 224, 224, "nchw"


try:
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    if os.path.exists(MODEL_PATH):
        ort_session = ort.InferenceSession(MODEL_PATH,
                                           providers=["CPUExecutionProvider"])
        _inp = ort_session.get_inputs()[0]
        _model_input_name = _inp.name
        _model_input_shape = _shape_of(_inp)
        try:
            _model_output_shape = _shape_of(ort_session.get_outputs()[0])
        except Exception:
            _model_output_shape = None

        INPUT_H, INPUT_W, LAYOUT = _detect_input_geometry(_inp.shape)
        if INPUT_SIZE > 0:                      # explicit override wins
            INPUT_H = INPUT_W = INPUT_SIZE
        if INPUT_LAYOUT in ("nchw", "nhwc"):
            LAYOUT = INPUT_LAYOUT

        print("[resq] model %s loaded | input=%s shape=%s -> %dx%d %s | "
              "normalize=%s softmax=%s accident_class=%d threshold=%.1f"
              % (MODEL_PATH, _model_input_name, _model_input_shape,
                 INPUT_W, INPUT_H, LAYOUT.upper(), NORMALIZE, APPLY_SOFTMAX,
                 ACCIDENT_CLASS_INDEX, CONFIDENCE_THRESHOLD))
    else:
        MODEL_ERROR = "model file not found at %r" % MODEL_PATH
        print("[resq] WARNING: %s - running without AI verification" % MODEL_ERROR)
except Exception as exc:                                  # pragma: no cover
    MODEL_ERROR = "%s: %s" % (type(exc).__name__, exc)
    print("[resq] WARNING: could not load model (%s)" % MODEL_ERROR)


def preprocess_image(image_bytes: bytes):
    """Bytes -> model-ready tensor, using whatever the model actually wants."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((INPUT_W, INPUT_H))
    data = np.array(img).astype(np.float32)

    if NORMALIZE == "imagenet":
        data /= 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        data = (data - mean) / std
    elif NORMALIZE == "unit":
        data /= 255.0
    elif NORMALIZE == "signed":
        data = data / 127.5 - 1.0
    # "raw" leaves 0-255 alone

    if LAYOUT == "nchw":
        data = np.transpose(data, (2, 0, 1))
    return np.expand_dims(data, axis=0)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def to_probabilities(outputs):
    """Raw model output -> probability row, whatever the export looks like."""
    arr = np.array(outputs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    if arr.shape[1] == 1:                       # single sigmoid output
        p = float(arr[0][0])
        if not 0.0 <= p <= 1.0:
            p = 1.0 / (1.0 + float(np.exp(-p)))
        return np.array([[1.0 - p, p]], dtype=np.float32)

    row = arr[0]
    looks_like_probs = bool(np.all(row >= -1e-6) and
                            abs(float(row.sum()) - 1.0) < 1e-3)
    if APPLY_SOFTMAX == "no":
        return arr
    if APPLY_SOFTMAX == "auto" and looks_like_probs:
        return arr                              # already normalised - leave it
    return softmax(arr)


def classify_one(raw: bytes) -> dict:
    """Score a single image and show the working, for debugging."""
    outputs = ort_session.run(None, {_model_input_name: preprocess_image(raw)})[0]
    probs = to_probabilities(outputs)
    row = [float(v) for v in probs[0]]
    idx = int(np.argmax(probs, axis=1)[0])
    n = len(row)
    accident_p = row[ACCIDENT_CLASS_INDEX] if 0 <= ACCIDENT_CLASS_INDEX < n else 0.0
    return {
        "raw_output": [float(v) for v in np.array(outputs).reshape(-1)[:16]],
        "probabilities": row,
        "argmax_index": idx,
        "accident_class_index": ACCIDENT_CLASS_INDEX,
        "accident_probability": round(accident_p * 100.0, 2),
        "verdict": ("VERIFIED"
                    if accident_p * 100.0 >= CONFIDENCE_THRESHOLD else "REJECTED"),
    }


def classify(images: List[bytes]):
    """Return (is_accident, confidence_percent, note).

    The score reported is always the probability of the accident class, so the
    number the responder apps show means one consistent thing whichever way
    the verdict went. With no model the incident passes through for review
    rather than being dropped - a real emergency must not be discarded because
    a file is missing on the server.
    """
    if ort_session is None:
        return True, 0.0, "AI model unavailable on the server - needs review"

    best = 0.0
    note = ""
    for raw in images:
        try:
            best = max(best, classify_one(raw)["accident_probability"])
        except Exception as exc:
            note = "inference failed: %s" % str(exc)[:120]
            print("[resq] inference failed on one image: %s" % exc)
    return best >= CONFIDENCE_THRESHOLD, round(best, 2), note


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

_lock = threading.Lock()
sessions: Dict[str, dict] = {}          # in-flight web-page reports
incidents: Dict[str, dict] = {}         # dispatched incidents, newest last
photos: Dict[str, bytes] = {}           # incident id -> jpeg/png bytes


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().replace(
        microsecond=0).isoformat()


def _next_id() -> str:
    return "GH-" + uuid.uuid4().hex[:6].upper()


def _trim() -> None:
    """Keep memory bounded on small hosts."""
    while len(incidents) > MAX_INCIDENTS:
        oldest = next(iter(incidents))
        incidents.pop(oldest, None)
        photos.pop(oldest, None)


def make_incident(lat: Optional[float], lon: Optional[float],
                  confidence_pct: float, verdict_real: bool,
                  photo: Optional[bytes] = None,
                  notes: str = "", address: str = "",
                  severity: str = "CRITICAL") -> dict:
    """Build the record shape the responder apps parse."""
    iid = _next_id()
    inc = {
        "id": iid,
        "timestamp": _now_iso(),
        "location": {
            "lat": lat,
            "lon": lon,
            # kept for any client that reads 'lng'
            "lng": lon,
            "address": address,
        },
        "ai_verification": {
            "confidence_score": round(confidence_pct, 2),   # 0-100
            "verdict": "VERIFIED" if verdict_real else "REJECTED",
            "model": os.path.basename(MODEL_PATH) if ort_session else None,
        },
        "severity": severity,
        "status": "NEW",
        "notes": notes,
        "image_url": "/api/alerts/%s/photo" % iid if photo else None,
        "history": [],
    }
    with _lock:
        incidents[iid] = inc
        if photo:
            photos[iid] = photo[:MAX_PHOTO_BYTES]
        _trim()
    print("[resq] incident %s created verdict=%s conf=%.1f loc=%s,%s"
          % (iid, inc["ai_verification"]["verdict"], confidence_pct, lat, lon))
    return inc


async def forward_if_configured(inc: dict) -> None:
    if not TARGET_SERVER_URL:
        return
    try:
        import httpx
        payload = {
            "id": inc["id"],
            "verdict": inc["ai_verification"]["verdict"],
            "confidence": inc["ai_verification"]["confidence_score"],
            "lat": inc["location"]["lat"],
            "lon": inc["location"]["lon"],
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(TARGET_SERVER_URL, json=payload)
    except Exception as exc:
        print("[resq] forward to %s failed: %s" % (TARGET_SERVER_URL, exc))


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class LocationPayload(BaseModel):
    report_id: str
    lat: float
    lng: float
    accuracy: Optional[float] = None


class FinalizePayload(BaseModel):
    report_id: str


class StatusPayload(BaseModel):
    status: str
    role: Optional[str] = None
    id: Optional[str] = None
    at: Optional[str] = None
    note: Optional[str] = None


# --------------------------------------------------------------------------
# Service info
# --------------------------------------------------------------------------

@app.get("/")
def root(request: Request):
    return {
        "status": "online",
        "service": "ResQ AI Dispatch Hub",
        "version": app.version,
        # derived from the actual request, never hard-coded: a stale
        # self-reported host makes every photo URL point somewhere dead
        "base_url": str(request.base_url).rstrip("/"),
        "model_loaded": ort_session is not None,
        "model_error": MODEL_ERROR,
        "active_incidents": len(incidents),
        "endpoints": {
            "responders": ["/api/alerts/%s" % r for r in ROLES],
            "submit": "/api/sos/submit",
            "web_flow": ["/api/session/start", "/api/location",
                         "/api/upload/{category}", "/api/finalize"],
        },
    }


@app.get("/healthz")
def healthz():
    return {"ok": True, "incidents": len(incidents),
            "model_loaded": ort_session is not None}


# --------------------------------------------------------------------------
# Web page flow — unchanged contract
# --------------------------------------------------------------------------

@app.post("/api/session/start")
def start_session():
    report_id = str(uuid.uuid4())[:8]
    with _lock:
        sessions[report_id] = {"report_id": report_id, "location": None,
                               "images": [], "started": time.time()}
    return {"report_id": report_id}


@app.post("/api/location")
def receive_location(payload: LocationPayload):
    with _lock:
        sess = sessions.get(payload.report_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    sess["location"] = {"lat": payload.lat, "lng": payload.lng,
                        "accuracy": payload.accuracy}
    return {"status": "location_saved"}


@app.post("/api/upload/{category}")
async def upload_image(category: str, report_id: str = Form(...),
                       image: UploadFile = File(...)):
    with _lock:
        sess = sessions.get(report_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    contents = await image.read()
    if len(contents) > MAX_PHOTO_BYTES:
        contents = contents[:MAX_PHOTO_BYTES]
    sess["images"].append(contents)
    return {"status": "image_uploaded", "category": category,
            "images": len(sess["images"])}


@app.post("/api/finalize")
async def finalize_report(payload: FinalizePayload):
    with _lock:
        sess = sessions.get(payload.report_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if not sess["images"]:
        raise HTTPException(status_code=400,
                            detail="No images uploaded for this report")

    is_accident, conf, note = classify(sess["images"])
    loc = sess.get("location") or {}

    # The incident is stored here rather than forwarded and forgotten, so the
    # responder apps have something to show.
    inc = make_incident(
        lat=loc.get("lat"), lon=loc.get("lng"),
        confidence_pct=conf, verdict_real=is_accident,
        photo=sess["images"][0] if sess["images"] else None,
        notes=note or "Citizen SOS submitted from the web reporter.",
    )
    await forward_if_configured(inc)

    with _lock:
        sessions.pop(payload.report_id, None)

    return {
        "report_id": payload.report_id,
        "incident_id": inc["id"],
        "verdict": "REAL" if is_accident else "FAKE",
        "max_confidence": conf,
        "dispatched": is_accident,
    }


# --------------------------------------------------------------------------
# One-shot submit
# --------------------------------------------------------------------------

@app.post("/api/sos/submit")
async def submit_sos(lat: float = Form(...), lon: float = Form(...),
                     photo: UploadFile = File(None),
                     notes: str = Form(""), address: str = Form("")):
    raw = await photo.read() if photo is not None else None
    if raw:
        is_accident, conf, note = classify([raw])
    else:
        is_accident, conf, note = True, 0.0, "No photo supplied - needs review"

    inc = make_incident(lat=lat, lon=lon, confidence_pct=conf,
                        verdict_real=is_accident, photo=raw,
                        notes=notes or note, address=address)
    await forward_if_configured(inc)
    return {"incident_id": inc["id"],
            "verdict": inc["ai_verification"]["verdict"],
            "confidence": conf, "dispatched": is_accident}


# --------------------------------------------------------------------------
# Responder apps
# --------------------------------------------------------------------------

def _visible(role: str) -> List[dict]:
    """Rejected reports never reach a responder — that is the spam filter."""
    with _lock:
        rows = [dict(i) for i in incidents.values()]
    return [r for r in rows
            if r["ai_verification"]["verdict"] != "REJECTED"
            and r["status"] not in ("CLEARED", "COMPLETED")]


@app.get("/api/alerts")
def all_alerts():
    with _lock:
        return {"alerts": [dict(i) for i in incidents.values()]}


@app.get("/api/alerts/{role}")
def role_alerts(role: str):
    role = role.lower()
    if role not in ROLES:
        raise HTTPException(status_code=404, detail="Unknown responder role")
    return {"alerts": _visible(role)}


@app.get("/api/alerts/{incident_id}/photo")
def alert_photo(incident_id: str):
    with _lock:
        raw = photos.get(incident_id)
    if raw is None:
        raise HTTPException(status_code=404, detail="No photo for this incident")
    kind = "image/png" if raw[:8].startswith(b"\x89PNG") else "image/jpeg"
    return Response(content=raw, media_type=kind,
                    headers={"Cache-Control": "public, max-age=3600"})


@app.post("/api/alerts/{incident_id}/status")
def set_status(incident_id: str, payload: StatusPayload):
    """The endpoint the responder apps need to report progress.

    Ambulance: ACCEPTED / EN ROUTE / ON SCENE / TRANSPORTING / COMPLETED
    Hospital:  CAN ADMIT / CANNOT ADMIT / BED READY / TRAUMA TEAM ALERTED /
               PATIENT ADMITTED
    Police:    ACKNOWLEDGED / UNIT DISPATCHED / ON SCENE / CLEARED
    """
    with _lock:
        inc = incidents.get(incident_id)
        if inc is None:
            raise HTTPException(status_code=404, detail="Unknown incident")
        inc["status"] = payload.status.upper()
        inc["history"].append({
            "status": inc["status"],
            "role": (payload.role or "unknown").lower(),
            "at": payload.at or _now_iso(),
            "note": payload.note or "",
        })
    print("[resq] %s -> %s by %s" % (incident_id, inc["status"], payload.role))
    return {"ok": True, "id": incident_id, "status": inc["status"],
            "history": inc["history"]}


@app.delete("/api/alerts/{incident_id}")
def clear_alert(incident_id: str):
    with _lock:
        existed = incidents.pop(incident_id, None) is not None
        photos.pop(incident_id, None)
    if not existed:
        raise HTTPException(status_code=404, detail="Unknown incident")
    return {"ok": True, "cleared": incident_id}


# --------------------------------------------------------------------------
# Testing helpers
# --------------------------------------------------------------------------

@app.post("/api/demo/seed")
def seed_demo(count: int = 1):
    """Create sample incidents so the three apps can be demonstrated without
    taking a photo. Hit this from a browser or curl."""
    samples = [
        (12.971599, 77.594566, "Anil Kumble Circle, MG Road, Bengaluru", 94.0,
         True, "CRITICAL", "Two-wheeler vs car. One rider down, not moving."),
        (12.934533, 77.626579, "Sony World Junction, Koramangala", 78.5,
         True, "MODERATE", "Auto-rickshaw overturned. Minor injuries."),
        (13.035000, 77.597000, "Hebbal Flyover, Outer Ring Road", 41.0,
         False, "LOW", "AI could not confirm an incident in the photo."),
    ]
    made = []
    for i in range(max(1, min(count, len(samples)))):
        lat, lon, addr, conf, real, sev, note = samples[i]
        made.append(make_incident(lat, lon, conf, real, None, note, addr, sev)["id"])
    return {"created": made}


@app.delete("/api/demo/clear")
def clear_all():
    with _lock:
        n = len(incidents)
        incidents.clear()
        photos.clear()
    return {"ok": True, "cleared": n}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)


# --------------------------------------------------------------------------
# Model debugging - how to work out the right settings for YOUR model
# --------------------------------------------------------------------------

@app.get("/api/debug/model")
def debug_model():
    """What the loaded model expects, and how we are currently feeding it."""
    return {
        "loaded": ort_session is not None,
        "error": MODEL_ERROR,
        "path": MODEL_PATH,
        "input_name": _model_input_name,
        "input_shape_declared": _model_input_shape,
        "output_shape_declared": _model_output_shape,
        "using": {
            "width": INPUT_W,
            "height": INPUT_H,
            "layout": LAYOUT,
            "normalize": NORMALIZE,
            "apply_softmax": APPLY_SOFTMAX,
            "accident_class_index": ACCIDENT_CLASS_INDEX,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        },
        "tune_with_env": ["NORMALIZE", "INPUT_SIZE", "INPUT_LAYOUT",
                          "APPLY_SOFTMAX", "ACCIDENT_CLASS_INDEX",
                          "CONFIDENCE_THRESHOLD"],
    }


@app.post("/api/debug/classify")
async def debug_classify(photo: UploadFile = File(...)):
    """Score one image and show every intermediate value.

    Submit a known accident photo and a known non-accident photo. Whichever
    class index separates them is the correct ACCIDENT_CLASS_INDEX; if neither
    separates them, the preprocessing (NORMALIZE / INPUT_SIZE / INPUT_LAYOUT)
    does not match how the model was trained.
    """
    if ort_session is None:
        raise HTTPException(status_code=503, detail=MODEL_ERROR or "no model")
    raw = await photo.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    try:
        detail = classify_one(raw)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)[:300])
    detail["filename"] = photo.filename
    detail["bytes"] = len(raw)
    detail["preprocessing"] = {
        "resized_to": [INPUT_W, INPUT_H],
        "layout": LAYOUT,
        "normalize": NORMALIZE,
        "apply_softmax": APPLY_SOFTMAX,
    }
    detail["per_class_percent"] = {
        str(i): round(p * 100.0, 2) for i, p in enumerate(detail["probabilities"])
    }
    return detail

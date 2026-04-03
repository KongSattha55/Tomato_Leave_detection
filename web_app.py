#!/usr/bin/env python
"""FastAPI backend that serves the HTML demo UI.

Usage
-----
    python web_app.py
    python web_app.py --model efficientnetb0 --port 8000
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.data.transforms import get_val_transforms
from src.models import build_model, list_models

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_PATH = Path("configs/config.yaml")
SAVED_MODELS_DIR = Path("saved_models")

DISEASE_INFO: dict[str, dict] = {
    "Tomato___Bacterial_spot": {
        "icon": "🦠", "severity": "Moderate", "color": "#f39c12",
        "description": "Caused by Xanthomonas bacteria. Small water-soaked lesions that turn brown with yellow halos on leaves and fruit.",
        "treatment": "Apply copper-based bactericides. Remove infected plant debris. Use certified disease-free seed.",
        "prevention": "Avoid overhead irrigation. Rotate crops annually. Space plants for good air circulation.",
    },
    "Tomato___Early_blight": {
        "icon": "🍂", "severity": "Moderate", "color": "#e67e22",
        "description": "Caused by Alternaria solani fungus. Dark lesions with concentric rings (target-board pattern) starting on older leaves.",
        "treatment": "Apply fungicides (chlorothalonil or mancozeb). Remove and destroy infected leaves promptly.",
        "prevention": "Mulch around plants. Avoid wetting foliage. Rotate crops every 2–3 years.",
    },
    "Tomato___Late_blight": {
        "icon": "⚠️", "severity": "Severe", "color": "#e74c3c",
        "description": "Caused by Phytophthora infestans — same pathogen as the Irish Potato Famine. Spreads rapidly in cool, wet weather.",
        "treatment": "Apply systemic fungicides immediately. Remove and destroy all infected plant material. Do not compost.",
        "prevention": "Use resistant varieties. Avoid overhead watering. Improve field drainage.",
    },
    "Tomato___Leaf_Mold": {
        "icon": "🌫️", "severity": "Moderate", "color": "#8e44ad",
        "description": "Caused by Passalora fulva. Pale green/yellow patches on upper surface; olive-green velvety mold on underside.",
        "treatment": "Improve greenhouse ventilation. Reduce humidity below 85%. Apply mancozeb or chlorothalonil.",
        "prevention": "Use resistant cultivars. Maintain low humidity. Avoid dense planting.",
    },
    "Tomato___Septoria_leaf_spot": {
        "icon": "⭕", "severity": "Moderate", "color": "#e67e22",
        "description": "Caused by Septoria lycopersici. Small circular spots with dark borders and light grey centres containing dark specks.",
        "treatment": "Remove infected lower leaves. Apply fungicides. Avoid working with wet plants.",
        "prevention": "Mulch soil surface. Stake plants for air flow. Rotate out of solanaceous crops.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "icon": "🕷️", "severity": "Low–Moderate", "color": "#16a085",
        "description": "Tiny arachnids that suck plant sap causing stippling, bronzing of leaves, and fine webbing on undersides.",
        "treatment": "Use miticides or insecticidal soap. Introduce predatory mites (Phytoseiidae). Keep plants well-watered.",
        "prevention": "Monitor regularly. Avoid water stress. Reduce dust on plants.",
    },
    "Tomato___Target_Spot": {
        "icon": "🎯", "severity": "Moderate", "color": "#d35400",
        "description": "Caused by Corynespora cassiicola. Brown necrotic lesions with concentric rings resembling a bullseye target.",
        "treatment": "Apply fungicides (azoxystrobin or difenoconazole). Remove infected leaves and fruit.",
        "prevention": "Improve air circulation. Avoid leaf wetness. Use clean transplants.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "icon": "🌀", "severity": "Severe", "color": "#c0392b",
        "description": "Viral disease spread by whiteflies (Bemisia tabaci). Leaves curl upward, turn yellow; plant is severely stunted.",
        "treatment": "No cure — remove and destroy infected plants immediately. Control whitefly with insecticides.",
        "prevention": "Use reflective mulch. Install insect-proof screens. Plant resistant varieties.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "icon": "🦋", "severity": "Moderate–Severe", "color": "#8e44ad",
        "description": "Highly contagious viral disease spread by contact and tools. Mottled light/dark green mosaic pattern on leaves.",
        "treatment": "No cure. Remove infected plants. Sanitise tools with 10% bleach solution between plants.",
        "prevention": "Wash hands frequently. Use virus-free seed. Do not smoke near plants (tobacco mosaic virus cross-infection).",
    },
    "Tomato___healthy": {
        "icon": "✅", "severity": "Healthy", "color": "#27ae60",
        "description": "The plant shows no signs of disease or pest damage. Leaves are firm, uniformly green, and fully developed.",
        "treatment": "No treatment needed. Continue regular care and monitoring.",
        "prevention": "Maintain proper watering, fertilisation, and spacing. Scout weekly for early detection.",
    },
}

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _auto_model() -> tuple[str | None, Path | None]:
    for p in sorted(SAVED_MODELS_DIR.glob("*_best.pth")):
        return p.stem.replace("_best", ""), p
    return None, None


def create_app(model_name_arg: str | None = None) -> FastAPI:
    cfg = _load_config()
    device = _pick_device()

    # Load model
    if model_name_arg:
        model_name = model_name_arg
        ckpt_path = SAVED_MODELS_DIR / f"{model_name}_best.pth"
    else:
        model_name, ckpt_path = _auto_model()

    model: Optional[torch.nn.Module] = None
    model_loaded_name: str = "none"

    if model_name and ckpt_path and ckpt_path.exists():
        m = build_model(
            name=model_name,
            num_classes=cfg["model"]["num_classes"],
            pretrained=False,
            dropout=cfg["model"].get("dropout", 0.3),
        )
        m.load_state_dict(torch.load(ckpt_path, map_location=device))
        m = m.to(device)
        m.eval()
        model = m
        model_loaded_name = model_name
        print(f"Loaded model: {model_name} | device: {device}")
    else:
        print("WARNING: No trained model found. Train first: python scripts/train.py")

    classes = sorted(DISEASE_INFO.keys())
    transform = get_val_transforms(cfg["data"]["image_size"])

    # ------------------------------------------------------------------
    api = FastAPI(title="Tomato Leaf Disease Detector API")

    @api.get("/api/health")
    def health():
        return {
            "status": "ok",
            "model": model_loaded_name,
            "device": str(device),
            "classes": len(classes),
        }

    @api.get("/api/models")
    def models_endpoint():
        available = [
            p.stem.replace("_best", "")
            for p in sorted(SAVED_MODELS_DIR.glob("*_best.pth"))
        ]
        return {"available": available, "all": list_models(), "loaded": model_loaded_name}

    @api.post("/api/predict")
    async def predict(file: UploadFile = File(...)):
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="No trained model loaded. Run: python scripts/train.py --model efficientnetb0",
            )

        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

        indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        top5 = [
            {
                "class": classes[i],
                "label": classes[i].replace("Tomato___", "").replace("_", " "),
                "confidence": round(p * 100, 2),
            }
            for i, p in indexed[:5]
        ]

        best_class = classes[indexed[0][0]]
        info = DISEASE_INFO.get(best_class, {})

        return JSONResponse({
            "predicted_class": best_class,
            "label": best_class.replace("Tomato___", "").replace("_", " "),
            "confidence": round(indexed[0][1] * 100, 2),
            "top5": top5,
            "info": info,
            "model": model_loaded_name,
        })

    # Serve static HTML at root — mount last so API routes take priority
    api.mount("/", StaticFiles(directory="static", html=True), name="static")

    return api


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default=None)
    p.add_argument("--port",   type=int, default=8000)
    p.add_argument("--host",   default="127.0.0.1")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = create_app(model_name_arg=args.model)
    uvicorn.run(app, host=args.host, port=args.port, reload=False)

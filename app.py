#!/usr/bin/env python
"""Gradio demo app for Tomato Leaf Disease Detection.

Usage
-----
    python app.py
    python app.py --share          # public Gradio URL
    python app.py --model resnet50 # choose backbone explicitly
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from src.data.transforms import get_val_transforms
from src.models import build_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_PATH = Path("configs/config.yaml")
SAVED_MODELS_DIR = Path("saved_models")

DISEASE_INFO: dict[str, dict] = {
    "Tomato___Bacterial_spot": {
        "icon": "🦠",
        "severity": "Moderate",
        "description": "Caused by Xanthomonas bacteria. Look for small, water-soaked lesions that turn brown with yellow halos.",
        "treatment": "Apply copper-based bactericides. Remove infected plant debris. Use disease-free seed.",
    },
    "Tomato___Early_blight": {
        "icon": "🍂",
        "severity": "Moderate",
        "description": "Caused by Alternaria solani fungus. Characteristic dark, concentric ring lesions (target-board pattern).",
        "treatment": "Apply fungicides (chlorothalonil or mancozeb). Improve air circulation. Rotate crops.",
    },
    "Tomato___Late_blight": {
        "icon": "⚠️",
        "severity": "Severe",
        "description": "Caused by Phytophthora infestans — the same pathogen that caused the Irish potato famine. Spreads rapidly in cool, wet conditions.",
        "treatment": "Apply systemic fungicides immediately. Remove and destroy infected plants. Avoid overhead irrigation.",
    },
    "Tomato___Leaf_Mold": {
        "icon": "🌫️",
        "severity": "Moderate",
        "description": "Caused by Passalora fulva. Pale green/yellow patches on upper leaf surface with olive-green mold on underside.",
        "treatment": "Improve ventilation. Reduce humidity. Apply fungicides (mancozeb or chlorothalonil).",
    },
    "Tomato___Septoria_leaf_spot": {
        "icon": "⭕",
        "severity": "Moderate",
        "description": "Caused by Septoria lycopersici. Small, circular spots with dark borders and light grey centres.",
        "treatment": "Remove infected lower leaves. Apply fungicides. Avoid splashing water on leaves.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "icon": "🕷️",
        "severity": "Low–Moderate",
        "description": "Tiny arachnids that suck plant sap. Causes stippling and bronzing of leaves; fine webbing visible.",
        "treatment": "Use miticides or insecticidal soap. Introduce predatory mites. Keep plants well-watered.",
    },
    "Tomato___Target_Spot": {
        "icon": "🎯",
        "severity": "Moderate",
        "description": "Caused by Corynespora cassiicola. Brown lesions with concentric rings resembling a target.",
        "treatment": "Apply fungicides. Remove infected leaves. Ensure good air circulation.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "icon": "🌀",
        "severity": "Severe",
        "description": "Viral disease spread by whiteflies. Leaves curl upward and turn yellow; plant growth is severely stunted.",
        "treatment": "No cure — remove and destroy infected plants. Control whitefly populations. Use resistant varieties.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "icon": "🦋",
        "severity": "Moderate–Severe",
        "description": "Viral disease causing mottled light/dark green mosaic pattern on leaves. Spread by contact and tools.",
        "treatment": "No cure. Remove infected plants. Sanitise tools with bleach. Use resistant seed varieties.",
    },
    "Tomato___healthy": {
        "icon": "✅",
        "severity": "None",
        "description": "The plant appears healthy with no visible signs of disease or pest damage.",
        "treatment": "Continue regular maintenance: proper watering, fertilisation, and monitoring.",
    },
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _auto_detect_model() -> tuple[str, Path] | tuple[None, None]:
    """Return (model_name, checkpoint_path) for the first available checkpoint."""
    if not SAVED_MODELS_DIR.exists():
        return None, None
    for p in sorted(SAVED_MODELS_DIR.glob("*_best.pth")):
        name = p.stem.replace("_best", "")
        return name, p
    return None, None


def load_model(model_name: str | None, cfg: dict, device: torch.device) -> tuple[torch.nn.Module, str] | tuple[None, str]:
    """Load model weights. Returns (model, model_name) or (None, error_msg)."""
    if model_name is None:
        model_name, ckpt_path = _auto_detect_model()
    else:
        ckpt_path = SAVED_MODELS_DIR / f"{model_name}_best.pth"

    if model_name is None or ckpt_path is None or not Path(ckpt_path).exists():
        return None, (
            "No trained model found in `saved_models/`.\n\n"
            "Train a model first:\n"
            "  python scripts/train.py --model efficientnetb0\n\n"
            "Then restart the app."
        )

    try:
        model = build_model(
            name=model_name,
            num_classes=cfg["model"]["num_classes"],
            pretrained=False,
            dropout=cfg["model"].get("dropout", 0.3),
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model, model_name
    except Exception as e:
        return None, f"Error loading model: {e}"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(
    image: Image.Image,
    model: torch.nn.Module,
    classes: list[str],
    device: torch.device,
    image_size: int = 224,
    top_k: int = 5,
) -> tuple[dict, str, Image.Image]:
    """Run inference on a PIL image.

    Returns:
        confidences:   Dict {label: probability} for top_k classes (for gr.Label).
        info_markdown: Disease information as Markdown string.
        bar_chart:     Matplotlib figure as PIL image.
    """
    transform = get_val_transforms(image_size)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    top_idx = probs.argsort()[::-1][:top_k]
    top_labels = [classes[i] for i in top_idx]
    top_probs = probs[top_idx]

    predicted_class = top_labels[0]
    info = DISEASE_INFO.get(predicted_class, {})

    # ---- Confidence dict for gr.Label ----
    conf_dict = {
        lbl.replace("Tomato___", "").replace("_", " "): float(p)
        for lbl, p in zip(top_labels, top_probs)
    }

    # ---- Disease info markdown ----
    icon = info.get("icon", "🌿")
    short_name = predicted_class.replace("Tomato___", "").replace("_", " ")
    confidence_pct = top_probs[0] * 100
    severity = info.get("severity", "Unknown")
    description = info.get("description", "")
    treatment = info.get("treatment", "")

    severity_color = {
        "None": "green", "Low–Moderate": "orange",
        "Moderate": "orange", "Moderate–Severe": "red", "Severe": "red",
    }.get(severity, "gray")

    info_md = f"""
## {icon} {short_name}

**Confidence:** {confidence_pct:.1f}%
**Severity:** <span style="color:{severity_color}; font-weight:bold">{severity}</span>

### Description
{description}

### Recommended Treatment
{treatment}
"""

    # ---- Bar chart ----
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(top_labels))]
    short_names = [l.replace("Tomato___", "").replace("_", " ") for l in top_labels]
    bars = ax.barh(short_names[::-1], top_probs[::-1] * 100, color=colors[::-1])
    ax.set_xlabel("Confidence (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Top Predictions", fontweight="bold")
    for bar, prob in zip(bars, top_probs[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{prob*100:.1f}%", va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    # Convert figure to PIL image
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    chart_img = Image.frombytes("RGBA", fig.canvas.get_width_height(), buf).convert("RGB")
    plt.close(fig)

    return conf_dict, info_md, chart_img


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_app(model_name_arg: str | None = None) -> gr.Blocks:
    cfg = _load_config()
    classes = sorted(DISEASE_INFO.keys())

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    model, model_or_error = load_model(model_name_arg, cfg, device)
    model_ready = model is not None
    model_label = model_or_error if model_ready else None
    error_msg = None if model_ready else model_or_error

    def run_inference(image: Image.Image) -> tuple:
        if image is None:
            return {}, "Please upload an image.", None
        if not model_ready:
            return {}, f"**Model not loaded.**\n\n{error_msg}", None
        conf_dict, info_md, chart = predict(image, model, classes, device, cfg["data"]["image_size"])
        return conf_dict, info_md, chart

    # ---- Example images (from dataset if available) ----
    example_paths: list[list] = []
    dataset_root = Path(cfg["data"]["train_dir"])
    if dataset_root.exists():
        for cls_dir in sorted(dataset_root.iterdir()):
            if cls_dir.is_dir():
                imgs = sorted(cls_dir.glob("*.jpg"))[:1] + sorted(cls_dir.glob("*.JPG"))[:1]
                if imgs:
                    example_paths.append([str(imgs[0])])
                if len(example_paths) >= 6:
                    break

    # ---- Layout ----
    with gr.Blocks(title="Tomato Leaf Disease Detector") as demo:
        gr.Markdown(
            """
            # 🍅 Tomato Leaf Disease Detector
            Upload a photo of a tomato leaf to identify diseases and get treatment recommendations.
            """
        )

        if not model_ready:
            gr.Markdown(
                f"> **No model loaded.** {error_msg}",
                elem_id="warning",
            )
        else:
            gr.Markdown(f"> Model: **{model_label}** | Device: **{device}**")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Tomato Leaf Image",
                    height=320,
                )
                predict_btn = gr.Button("Diagnose", variant="primary", size="lg")

                if example_paths:
                    gr.Examples(
                        examples=example_paths,
                        inputs=image_input,
                        label="Example Images (from dataset)",
                    )

            with gr.Column(scale=1):
                label_output = gr.Label(
                    num_top_classes=5,
                    label="Confidence Scores",
                )
                chart_output = gr.Image(
                    label="Prediction Chart",
                    height=260,
                )

        with gr.Row():
            info_output = gr.Markdown(label="Disease Information")

        predict_btn.click(
            fn=run_inference,
            inputs=image_input,
            outputs=[label_output, info_output, chart_output],
        )
        # Also trigger on image upload
        image_input.change(
            fn=run_inference,
            inputs=image_input,
            outputs=[label_output, info_output, chart_output],
        )

        gr.Markdown(
            """
            ---
            **Supported Diseases:** Bacterial Spot · Early Blight · Late Blight · Leaf Mold ·
            Septoria Leaf Spot · Spider Mites · Target Spot · Yellow Leaf Curl Virus · Mosaic Virus · Healthy
            """
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default=None, help="Model name (e.g. efficientnetb0)")
    p.add_argument("--share",  action="store_true", help="Create public Gradio link")
    p.add_argument("--port",   type=int, default=7860)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = build_app(model_name_arg=args.model)
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="green"),
    )

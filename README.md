# Tomato Leaf Disease Detector

AI-powered web app that detects 10 tomato leaf diseases from a photo using EfficientNetB0.

------------------------------------------------------------------------

## Requirements

-   Python 3.10 or higher
-   pip3

------------------------------------------------------------------------

## Step 1 — Clone or Open the Project

Open a terminal and navigate to the project folder:

``` bash
cd "/Users/kongsattha/Documents/ITC/Personal Doc/PersoanlProject/Tomato_Leave_detection"
```

------------------------------------------------------------------------

## Step 2 — Install Dependencies

``` bash
pip3 install -r requirements.txt
```

This installs: PyTorch, torchvision, FastAPI, Uvicorn, Gradio, Pillow, and other required libraries.

------------------------------------------------------------------------

## Step 3 — Check the Trained Model

A pre-trained model is already included in `saved_models/`:

```         
saved_models/
└── efficientnetb0_best.pth   ← trained model weights (18 MB)
```

You can skip to **Step 5** to run the demo directly.

------------------------------------------------------------------------

## Step 4 — (Optional) Retrain the Model

If you want to train from scratch:

``` bash
python3 scripts/train.py --model efficientnetb0
```

Available model options:

| Model                    | Flag                     |
|--------------------------|--------------------------|
| EfficientNetB0 (default) | `--model efficientnetb0` |
| MobileNetV2              | `--model mobilenetv2`    |
| ResNet50                 | `--model resnet50`       |
| DenseNet121              | `--model densenet121`    |
| Custom CNN               | `--model custom_cnn`     |

Training config (epochs, learning rate, batch size) can be adjusted in `configs/config.yaml`.

------------------------------------------------------------------------

## Step 5 — Start the Web Demo

``` bash
python3 web_app.py
```

Expected output:

```         
Loaded model: efficientnetb0 | device: mps
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000
```

------------------------------------------------------------------------

## Step 6 — Open in Browser

Go to:

```         
http://127.0.0.1:8000
```

------------------------------------------------------------------------

## Step 7 — Try It Out

1.  Click **"Click to browse"** or drag & drop a tomato leaf image onto the upload area
2.  Click the **"Analyze"** button
3.  View the results:
    -   Predicted disease name and confidence score
    -   Severity level (Healthy / Moderate / Severe)
    -   **Confidence tab** — top 5 predictions with probability bars
    -   **About tab** — disease description
    -   **Treatment tab** — recommended treatment steps
4.  Scroll down to browse the **Disease Encyclopedia** for all 10 classes

------------------------------------------------------------------------

## Supported Diseases

| \#  | Disease                | Severity        |
|-----|------------------------|-----------------|
| 1   | Bacterial Spot         | Moderate        |
| 2   | Early Blight           | Moderate        |
| 3   | Late Blight            | Severe          |
| 4   | Leaf Mold              | Moderate        |
| 5   | Septoria Leaf Spot     | Moderate        |
| 6   | Spider Mites           | Low–Moderate    |
| 7   | Target Spot            | Moderate        |
| 8   | Yellow Leaf Curl Virus | Severe          |
| 9   | Mosaic Virus           | Moderate–Severe |
| 10  | Healthy                | —               |

------------------------------------------------------------------------

## Stop the Server

Press `Ctrl + C` in the terminal, or run:

``` bash
pkill -f web_app.py
```

------------------------------------------------------------------------

## Project Structure

```         
Tomato_Leave_detection/
├── Dataset/tomato/          # 6,000 labeled images (train + test)
├── saved_models/            # Trained model weights
├── src/
│   ├── data/                # Dataset loading, transforms
│   ├── models/              # Model architectures
│   ├── training/            # Trainer, callbacks
│   └── utils/               # Logger, visualization
├── scripts/
│   ├── train.py             # Training script
│   └── evaluate.py          # Evaluation script
├── static/index.html        # Web UI (HTML/CSS/JS)
├── configs/config.yaml      # Training configuration
├── app.py                   # Gradio demo (alternative UI)
├── web_app.py               # FastAPI backend + HTML demo
├── Tomato_LDD.ipynb         # Jupyter notebook
└── requirements.txt         # Python dependencies
```

------------------------------------------------------------------------

## Alternative Demo (Gradio)

If you prefer Gradio's interface instead:

``` bash
python3 app.py
```

Then open: `http://127.0.0.1:7860`

To generate a public shareable link:

``` bash
python3 app.py --share
```

# Assignment 3: Small Object Detection Using YOLO

**Course:** CSCI 611 — Computer Vision, Spring 2026  
**Student:** Amol Bhalerao  

---

## Objective

Detect traffic signs (small objects) from street images using YOLOv8.  
We compare a **pre-trained YOLOv8n baseline** against two **fine-tuned configurations** trained on the Roboflow Road Signs dataset, and evaluate using mAP50, Precision, Recall, and F1 metrics.

---

## Repository Structure

```
Assignment_3/
├── yolo_detection.ipynb      # Main notebook — all code, experiments, and outputs
├── requirements.txt          # Python dependencies
├── Readme.md                 # This file
├── dataset/                  # Downloaded Roboflow dataset (auto-created)
├── samples/                  # Annotated sample images and dataset analysis plots
├── results/                  # Detection outputs, comparison charts, sweeps
└── runs/                     # Trained model weights (Ultralytics format)
    ├── config_A/weights/best.pt
    └── config_B/weights/best.pt
```

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** For GPU training, ensure you have CUDA-compatible PyTorch installed.  
> Visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the right command for your system.

### 2. Get a Free Roboflow API Key

The notebook downloads the **Road Signs Detection** dataset automatically via the Roboflow API.

1. Go to [roboflow.com](https://roboflow.com) and create a free account.
2. Navigate to **Settings → API Keys** and copy your key.
3. Open `yolo_detection.ipynb` and paste your key in **Section 3**:

```python
ROBOFLOW_API_KEY = "your_api_key_here"   # Section 3, Cell 1
```

> **Note for grader:** The API key is already set in the submitted notebook.

### 3. Run the Notebook

```bash
jupyter notebook yolo_detection.ipynb
```

Or open it in VS Code / JupyterLab.  
Run all cells sequentially from top to bottom (`Kernel → Restart & Run All`).

---

## Running on Google Colab (Recommended for Training)

Training on CPU is slow. For faster results:

1. Upload this folder to your Google Drive under `MyDrive/Assignment_3/`.
2. Open `yolo_detection.ipynb` in [Google Colab](https://colab.research.google.com/).
3. In **Section 1**, the notebook automatically detects Colab and mounts Google Drive.
4. Set the runtime to **GPU** (`Runtime → Change runtime type → T4 GPU`).
5. Run all cells — training Config B (imgsz=1280) should complete in ~15–20 minutes.

---

## Dataset

**Road Signs Detection** — Roboflow Universe  
- URL: https://universe.roboflow.com/roboflow-100/road-signs-6ih4y  
- Format: YOLOv8 (pre-annotated)  
- Splits: ~80% train / ~10% val / ~10% test  
- Classes: Various traffic sign types

---

## Experiments

| Config | Image Size | Epochs | Batch | Augmentation | Notes |
|--------|-----------|--------|-------|--------------|-------|
| Baseline (Pre-trained) | — | — | — | — | COCO weights, no fine-tuning |
| Config A | 640 px | 30 | 8 | Default mosaic + flips | Standard fine-tuning |
| Config B | 1280 px | 30 | 4 | Enhanced (scale, erasing, mosaic) | Optimized for small objects |

Additional experiments in the notebook:
- Confidence threshold sweep (0.1 → 0.7)
- NMS IoU threshold sweep (0.3 → 0.8)
- Side-by-side visual comparison of all models

---

## Deliverables

| Item | Location |
|------|----------|
| Trained model weights | `runs/config_A/weights/best.pt`, `runs/config_B/weights/best.pt` |
| Annotated dataset samples | `samples/dataset_samples.png` |
| Class distribution plot | `samples/class_distribution.png` |
| Bounding box size analysis | `samples/bbox_size_analysis.png` |
| Baseline predictions | `results/baseline_test_predictions.png` |
| High-confidence detections | `results/high_conf_detections/` |
| Confidence sweep chart | `results/confidence_threshold_sweep.png` |
| NMS IoU sweep chart | `results/nms_iou_sweep.png` |
| Model comparison table | `results/results_comparison.csv` |
| Model comparison bar chart | `results/model_comparison_bar.png` |
| Training loss curves | `results/training_curves_config_A.png`, `results/training_curves_config_B.png` |
| Side-by-side comparison | `results/side_by_side_comparison.png` |
| PDF report | Submitted to Canvas |

---

## Key Results

| Model | mAP50 | mAP50-95 | Precision | Recall |
|-------|-------|---------|-----------|--------|
| Pre-trained YOLOv8n (COCO) | 0.000 | 0.000 | 0.000 | 0.000 |
| Fine-tuned Config A (640px) | 0.954 | 0.817 | 0.934 | 0.908 |
| Fine-tuned Config B (1280px) | 0.947 | 0.752 | 0.934 | 0.876 |

**Key finding:** Fine-tuning produced a massive improvement over the COCO pre-trained baseline (0 → 0.95 mAP50), confirming that domain-specific training is essential for traffic sign detection. Config A (640px) achieved slightly higher mAP50 while Config B (1280px) used more aggressive augmentation targeting small objects.

---

## References

- Ultralytics YOLOv8: https://docs.ultralytics.com
- Roboflow Road Signs Dataset: https://universe.roboflow.com/roboflow-100/road-signs-6ih4y
- Jocher, G. et al. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics


# CSCI 611 — Assignment 2: CNN on CIFAR-10

**Author:** A. Bhalerao  
**Course:** CSCI 611 — Spring 2026  
**University:** California State University, Chico

---

## Overview

This assignment trains a custom Convolutional Neural Network (CNN) on the CIFAR-10 dataset and visualizes its internal feature representations. It covers:

- Task 1: CNN design, training, and evaluation
- Task 2A: Feature map visualization from the first convolutional layer
- Task 2B: Maximally activating images for selected filters

All code, notes, and execution traces (outputs, plots) are contained in the Jupyter notebook.

---

## Files

| File | Description |
|------|-------------|
| `cnn_assignment2.ipynb` | Main notebook — all code and execution traces |
| `Assignment2_Report.pdf` | Written report (5–6 pages) |
| `model_trained.pt` | Saved model weights (best validation loss checkpoint) |
| `data/` | CIFAR-10 dataset (auto-downloaded on first run) |

---

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- numpy
- matplotlib

### Install dependencies

**Using pip:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib jupyter
```

**Using conda:**
```bash
conda install pytorch torchvision cpuonly -c pytorch
conda install numpy matplotlib jupyter
```

> If you have a GPU, replace `cpuonly` with the appropriate CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## How to Run

### Option 1: Jupyter Notebook (recommended)

```bash
jupyter notebook cnn_assignment2.ipynb
```

Then select **Kernel → Restart & Run All** to execute all cells from top to bottom.

### Option 2: Google Colab

1. Upload `cnn_assignment2.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU) for faster training
3. Run **Runtime → Run all**

> The notebook auto-downloads CIFAR-10 on first run — no manual dataset setup needed.

---

## Expected Outputs

| Cell | Output |
|------|--------|
| Setup | `Training on CPU` (or `GPU` on Colab) |
| Model | Printed architecture summary |
| Training | Loss per epoch, best model saved to `model_trained.pt` |
| Loss curves | Plot of training vs validation loss over 15 epochs |
| Test accuracy | Per-class and overall accuracy (~70–75%) |
| Predictions | 20 test images with predicted and true labels |
| Task 2A | Feature maps from `conv1` for 3 test images (8 maps each) |
| Task 2B | Top 5 activating images for 3 filters in `conv2` |

### Training setup used

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10 |
| Loss | CrossEntropyLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 20 |
| Epochs | 15 |
| Regularization | Dropout (p=0.25) |
| Augmentation | RandomHorizontalFlip + RandomCrop(32, padding=4) |

### Results obtained (CPU run)

- **Overall test accuracy: 74%**
- Best validation loss: 0.7248 (epoch 15)

---

## Notes

- Training on CPU takes approximately 25–35 minutes for 15 epochs. On Colab GPU it runs significantly faster.
- The notebook is fully self-contained — all execution outputs and plots are already saved inside it and visible without re-running.
- CIFAR-10 data is downloaded automatically to a local `data/` folder on first run.

# Assignment 4 – Neural Style Transfer

**CSCI 611 | Amol Bhalerao**  
Implementation of *Image Style Transfer Using Convolutional Neural Networks* (Gatys et al., CVPR 2016).

---

## Overview

This assignment implements neural style transfer using a pre-trained **VGG19** network. Given a *content image* and a *style image*, the algorithm synthesises a new image that preserves the high-level content of the content image while applying the visual texture/style of the style image.

---

## Files

| File | Description |
|------|-------------|
| `Style_Transfer_Exercise.ipynb` | Main Jupyter notebook with full implementation and experiments |
| `README.md` | This file |
| `exp1_style_weight.png` | Output – Experiment 1 (style weight ratio comparison) |
| `exp2_layer_weights.png` | Output – Experiment 2 (style layer weight configurations) |
| `exp3_init.png` | Output – Experiment 3 (initialisation strategy comparison) |
| `exp4_lr.png` | Output – Experiment 4 (learning rate comparison) |

---

## Requirements

- Python 3.8+
- PyTorch >= 1.9
- torchvision
- Pillow
- matplotlib
- numpy
- requests

Install all dependencies with:

```bash
pip install torch torchvision pillow matplotlib numpy requests
```

---

## How to Run

1. Open a terminal and navigate to this folder:
   ```bash
   cd Assignment_4
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Style_Transfer_Exercise.ipynb
   ```
   Or open with JupyterLab:
   ```bash
   jupyter lab Style_Transfer_Exercise.ipynb
   ```

3. Run all cells in order (**Kernel → Restart & Run All**).

   - **GPU is strongly recommended.** If CUDA is available, it is used automatically; the code falls back to CPU otherwise. On CPU, 2000 steps may take 20–40 minutes.
   - The notebook downloads the content and style images directly from Wikimedia Commons (internet connection required for the first run).

4. The notebook is structured as follows:

   | Section | Cells | Description |
   |---------|-------|-------------|
   | Part 1 – Implementation | 1–24 | Core style transfer (TODOs completed) |
   | Part 2 – Experiments | 25–35 | Hyperparameter sweeps |

---

## Content & Style Images

The notebook fetches images via URL:

- **Content:** Labrador Retriever photo (Wikimedia Commons)  
  `https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg`

- **Style:** *The Starry Night* by Vincent van Gogh  
  `https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg`



```python
content = load_image('path/to/your_content.jpg').to(device)
style   = load_image('path/to/your_style.jpg', shape=content.shape[-2:]).to(device)
```

---

## Key Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `content_weight` (α) | `1` | Weight for content loss |
| `style_weight` (β) | `1e6` | Weight for style loss; increase for stronger stylisation |
| `steps` | `2000` | Optimisation iterations; more = better quality |
| `lr` | `0.003` | Adam learning rate |
| `style_weights` | see cell 20 | Per-layer style weighting (early = coarse, late = fine) |

---

## Algorithm Summary

1. Load VGG19 (features only) with frozen weights.
2. Pass content and style images through VGG19 to extract feature maps at specific layers.
3. Compute **Gram matrices** for style layers (encode texture statistics).
4. Initialise target image as a copy of the content image.
5. Iteratively optimise the target image via Adam to minimise:
   - **Content loss:** MSE between target and content features at `conv4_2`
   - **Style loss:** weighted MSE between Gram matrices at `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`
   - **Total loss:** `α × content_loss + β × style_loss`

---

## Reference

Gatys, L. A., Ecker, A. S., & Bethge, M. (2016).  
*Image Style Transfer Using Convolutional Neural Networks.*  
CVPR 2016. https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf

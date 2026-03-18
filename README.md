# Masked Autoencoder (MAE) — Self-Supervised Image Representation Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.54-FF4B4B?style=for-the-badge&logo=streamlit)
![Kaggle](https://img.shields.io/badge/Kaggle-T4%20x2%20GPU-20BEFF?style=for-the-badge&logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)

**A complete from-scratch PyTorch implementation of Masked Autoencoders (MAE) for self-supervised visual representation learning.**

[Architecture](#architecture) • [Results](#results) • [Setup](#setup) • [Usage](#usage) • [References](#references)

</div>

---

## Author

| | |
|---|---|
| **Name** | Muneeb |
| **Course** | Generative AI (AI4009) |
| **University** | National University of Computer & Emerging Sciences (NUCES) |
| **Semester** | Spring 2026 |
| **Assignment** | No. 2 — Self-Supervised Image Representation Learning |

---

## Overview

This project implements the **Masked Autoencoder (MAE)** proposed by He et al. (2021), a powerful self-supervised learning framework that teaches a model to understand images by reconstructing **75% of randomly masked patches** from only 25% of visible patches.

The entire system is built **from scratch using base PyTorch** — no pretrained weights, no external ViT libraries. A polished **Streamlit** web app with a glassmorphism dark UI lets you upload any image and watch the MAE reconstruct it in real time.

```
Input Image (224×224)
       │
       ▼
  196 Patches (16×16 each)
       │
       ▼
  Mask 75% → Keep 49 patches only
       │
       ▼
  Encoder (ViT-Base, 86M params)
       │
       ▼
  Decoder (ViT-Small, 22M params)
       │
       ▼
  Reconstruct all 196 patches
       │
       ▼
  MSE Loss on masked patches only
```

---

## Architecture

The system uses an **asymmetric encoder-decoder design** — the key innovation that makes MAE computationally efficient.

### Encoder — ViT-Base (B/16)

| Parameter | Value |
|-----------|-------|
| Image Size | 224 × 224 |
| Patch Size | 16 × 16 |
| Total Patches | 196 |
| Visible Patches | 49 (25%) |
| Hidden Dimension | 768 |
| Transformer Layers | 12 |
| Attention Heads | 12 |
| Parameters | ~86 Million |

- Accepts **only 25% visible patches** (49 out of 196)
- Uses **learned positional embeddings** for spatial awareness
- Mask tokens are **never** fed to the encoder
- Outputs latent representations for visible tokens only

### Decoder — ViT-Small (S/16)

| Parameter | Value |
|-----------|-------|
| Hidden Dimension | 384 |
| Transformer Layers | 12 |
| Attention Heads | 6 |
| Parameters | ~22 Million |

- Receives encoder output + **learnable mask tokens** for missing patches
- Reconstructs the full 196-patch sequence
- Used **only during training** — discarded at inference

### Why Asymmetric?

The encoder processes 4× fewer tokens (49 vs 196), making training roughly **16× cheaper** due to quadratic attention complexity. The decoder is intentionally lightweight since reconstruction is simpler than understanding.

---

## Key Concepts

### Patchification
```
224×224 image → split into 14×14 grid → 196 patches of 16×16 pixels
Each patch flattened → 16 × 16 × 3 = 768 numbers per patch
```

### Random Masking
```
Shuffle all 196 patch indices randomly
Keep first 49 (25%) → visible patches
Discard last 147 (75%) → masked patches
Save ids_restore to unshuffle later
```

### Loss Function
```
Target = patchify(original image)
Target = normalize per-patch (mean=0, std=1)
Loss   = MSE(prediction, target)
Loss computed ONLY on 147 masked patches
Visible patches are ignored in loss
```

### Training Techniques
- **AdamW** optimizer with betas (0.9, 0.95) and weight decay 0.05
- **Cosine LR schedule** with 5-epoch linear warmup
- **Mixed Precision (AMP)** for 2× memory savings and faster training
- **Gradient clipping** at 1.0 to prevent instability
- **DataParallel** for dual T4 GPU utilization

---

## Results

### Training Metrics

| Metric | Value |
|--------|-------|
| Best Validation Loss | 0.3491 |
| Training Platform | Kaggle T4 x2 GPU |
| Dataset | TinyImageNet (100K images) |
| Batch Size | 64 |

### Quantitative Evaluation

| Metric | Score |
|--------|-------|
| Mean PSNR | ~22 dB |
| Mean SSIM | ~0.70 |

### Qualitative Results

The model successfully:
- Reconstructs major structures and shapes from 25% visible patches
- Infers approximate colors and textures of masked regions
- Maintains spatial coherence across reconstructed patches

---

## Project Structure

```
mae-vit-self-supervised-image-representation-main/
│
├── app.py                                          # Streamlit web app (dark glassmorphism UI)
├── assignment-2-generative-ai-22f-3875-1 (3).ipynb # Complete Kaggle training notebook
├── model_mae.pth                                   # Trained MAE checkpoint (~108M params)
├── requirements.txt                                # Python dependencies
└── README.md                                       # This file
```

---

## Setup

### Prerequisites
```bash
Python 3.11+
CUDA-enabled GPU (recommended, but CPU works)
```

### Installation
```bash
git clone https://github.com/Muneeb/mae-vit-self-supervised-image-representation.git
cd mae-vit-self-supervised-image-representation-main
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

### Requirements
```
torch>=2.0.0
torchvision>=0.15.0
streamlit
numpy
pillow
scikit-image
einops
tqdm
timm
gdown
```

---

## Usage

### Run the Streamlit App

```bash
python -m streamlit run app.py
# Opens at http://localhost:8501
```

The app will auto-download `model_mae.pth` from Google Drive on first launch if it is not already present.

**Features:**
- Dark glassmorphism UI with animated sidebar
- Adjustable masking ratio (10%–95%)
- Side-by-side comparison: Original → Masked → Reconstructed
- PSNR & SSIM quality metrics
- Interactive 14×14 patch visibility map
- One-click PNG download of the reconstruction

### Run on Kaggle (Training)

1. Open the notebook on Kaggle
2. Add **TinyImageNet** dataset: `akash2sharma/tiny-imagenet`
3. Enable **GPU T4 x2** accelerator
4. Run all cells in order

### Notebook Cell Guide

| Cell | Description |
|------|-------------|
| Cell 1 | Install dependencies |
| Cell 2 | Config & imports |
| Cell 3 | TinyImageNet dataloaders |
| Cell 4 | Transformer building blocks |
| Cell 5 | Patchify / Unpatchify / Masking |
| Cell 6 | MAE Encoder (ViT-Base) |
| Cell 7 | MAE Decoder (ViT-Small) |
| Cell 8 | Full MAE model |
| Cell 9 | Optimizer + Scheduler + AMP |
| Cell 10 | Training loop |
| Cell 11 | Loss curve plot |
| Cell 12 | Visualization (5 samples) |
| Cell 13 | PSNR & SSIM evaluation |
| Cell 14 | Save checkpoint |
| Cell 15 | Streamlit app |

---

## Dataset

**TinyImageNet** — a subset of ImageNet
- 200 object classes
- 100,000 training images
- 10,000 validation images
- Image size: resized to 224×224

Add to Kaggle notebook from:
`https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet`

---

## References

```bibtex
@article{he2021masked,
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  author  = {He, Kaiming and Chen, Xinlei and Xie, Saining and
             Li, Yanghao and Dollar, Piotr and Girshick, Ross},
  journal = {arXiv preprint arXiv:2111.06377},
}

@article{dosovitskiy2020image,
  title   = {An Image is Worth 16x16 Words: Transformers for
             Image Recognition at Scale},
  author  = {Dosovitskiy, Alexey and Beyer, Lucas and
             Kolesnikov, Alexander and others},
  journal = {arXiv preprint arXiv:2010.11929},
}
```

<div align="center">

Built with PyTorch & Streamlit | Trained on Kaggle T4 x2 | By **Muneeb**

</div>

# Brain Segmentation — Reproducible PyTorch Demo

A small **CPU-ready image-segmentation engineering demonstration** accompanying Leon Fang’s research interests in neuroimaging and machine learning.

> **Scope:** This repository uses **synthetic 2D images of geometric shapes**, *not* patient MRI or the BraTS dataset. The reported losses and segmentation scores are **toy-demo outputs**, not published medical-imaging results or evidence of clinical performance. This is a fresh, self-contained demonstration, not a reconstruction of the original research experiments.

## What it demonstrates

- A compact encoder–decoder **U-Net-style** model built in PyTorch, with skip connections and bilinear upsampling.
- A reproducible synthetic segmentation dataset with background, disk, and rectangle labels.
- An end-to-end supervised training loop with cross-entropy, Adam, and inference.
- Multiclass Dice and mean intersection-over-union (IoU) evaluation.
- Three offline smoke tests for dataset integrity, forward/backward gradients, and an optimizer step.

## Quick start

Python 3.10+ recommended. A CPU is sufficient.

```bash
python -m pip install -r requirements.txt
python segmentation_demo.py --epochs 1
python -m unittest discover -s tests -v
```

Expected output begins with `SYNTHETIC DATA ONLY` and prints a dictionary containing `loss`, `mean_dice`, and `mean_iou`. Values depend on the library version and runtime. The tests should report **3 tests OK**.

## Repository structure

| File | Role |
|---|---|
| `segmentation_demo.py` | Synthetic data, tiny U-Net, training, Dice/IoU |
| `tests/test_smoke.py` | Fast offline smoke tests |
| `requirements.txt` | Minimal Python dependencies |
| `.gitignore` | Prevents accidental addition of medical data and model weights |

## Design and limitations

The program creates simple noisy images with two overlapping foreground primitives, then trains a multiclass segmentation model. It is intended to demonstrate **software mechanics**, not the ability to segment real tumors. It uses a tiny sample and evaluates a subset of its synthetic training distribution, not an independent clinical test set. It has **no patient data, original project code, third-party training logs, checkpoints, or pretrained weights**.

To extend this to genuine medical-imaging research, a researcher would need a properly licensed dataset, MRI preprocessing, patient-level train/validation/test separation, class-aware evaluation, tuned training, and reproducibility controls. None of those experiments are claimed here.

## Related independent publication

Fang, L. (2025). “Application of Computer Image Processing Technology in Brain Tumor Segmentation.” In *AI in Multidimensional Signals Analysis and Processing*, pp. 119–130. Springer. [DOI: 10.1007/978-981-96-5408-6_9](https://doi.org/10.1007/978-981-96-5408-6_9).

**Important:** The publication and this newly assembled demonstration are separate artifacts. The demo does **not** reproduce the paper's experiments, and it must not be cited as validation of the paper's reported findings.

## Author and purpose

Prepared for Leon Fang's undergraduate research portfolio, November 2026. Project emphasis: Python, PyTorch, segmentation model engineering, and reproducible smoke testing.

# microscopy-qc

Reproducible, explainable quality control (QC) for microscopy images.

> **Status:** work in progress, Phase 2 of 7 (data and degradations).

## Problem

Microscopy images taken in the lab are sometimes out of focus, overexposed or
noisy. If those images go unnoticed into downstream analysis, results become
unreliable and hard to reproduce. This project builds a small, transparent
system that flags low-quality images and explains *why* each one was flagged.

## Approach (planned)

1. Take public cell images (MedMNIST) and apply **controlled, seeded
   degradations** (blur, exposure shift, noise), so the true QC label is known.
2. Compare three models:
   - interpretable QC features (sharpness, exposure, SNR) + a classical classifier
   - a small CNN on the raw image
   - a fusion of both
3. Report calibrated confidence and send low-confidence images to human review.
4. Explain decisions with feature importance and Grad-CAM.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
make install
make test
make data      # downloads BloodMNIST and builds the QC dataset
```

## Reproducibility

- All settings live in `configs/`; nothing is hard-coded.
- One seed in the config controls all randomness.
- Dependencies are pinned to exact versions in `pyproject.toml`.
- The source data is verified against its published MD5 checksum before use.
- Every generated dataset is saved with a metadata JSON (config, seed, source
  checksum, split sizes, creation time), so any result can be traced back to
  exactly how its data was made.
- Each degraded image stores its severity, so errors can be analysed by how
  strong the degradation was.

## Data

Source images: [MedMNIST v3](https://medmnist.com) BloodMNIST (CC BY 4.0),
28x28 images of blood cells, converted to grayscale. Only the official
training split is used as a pool; 2,000 images are sampled (500 per class)
and each is used once:

| QC class      | Simulates                      | Severity drawn from        |
|---------------|--------------------------------|----------------------------|
| `good`        | unchanged image                | -                          |
| `blurry`      | out-of-focus acquisition       | Gaussian sigma 0.8-2.0 px  |
| `overexposed` | too-long exposure / high gain  | brightness x1.5-2.5, clip  |
| `noisy`       | low light / sensor noise       | Gaussian noise std 0.05-0.15 |

The data is then split 70/15/15 into train/val/test, stratified by class.

## Project layout

```
configs/   experiment settings
src/mqc/   package source code
tests/     automated tests
data/      downloaded data (not committed)
reports/   figures and model card
docs/      design notes
```

## License

MIT

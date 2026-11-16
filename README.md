# microscopy-qc

Reproducible, explainable quality control (QC) for microscopy images.

> **Status:** work in progress, Phase 3 of 7 (QC features and classical models).

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
make train     # trains feature models; results go to runs/<timestamp>/
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
| `blurry`      | out-of-focus acquisition       | Gaussian sigma 0.4-1.0 px  |
| `overexposed` | too-long exposure / high gain  | brightness x1.05-1.3, clip |
| `noisy`       | low light / sensor noise       | Gaussian noise std 0.01-0.05 |

The data is then split 70/15/15 into train/val/test, stratified by class.

**Why such mild degradations?** A first version used stronger settings
(blur 0.8-2.0, gain 1.5-2.5, noise 0.05-0.15). A simple logistic regression
on the features below reached 99-100% test accuracy, which says more about the
benchmark than the model. Subtle failures are also the ones most likely to slip
past a person at the microscope, so the ranges were reduced until the task was
no longer trivial.

## QC features

Each feature mirrors something a scientist would check by eye:

| Feature | Measures |
|---|---|
| `laplacian_var` | sharpness (variance of the Laplacian) |
| `gradient_energy` | sharpness (mean squared Sobel gradient) |
| `mean_intensity`, `std_intensity` | brightness and contrast |
| `p99_intensity`, `saturated_frac` | highlights and clipped pixels |
| `noise_sigma` | noise level, Immerkaer (1996) fast noise estimator |

Two classical models use them: logistic regression (weights are directly
readable) and a random forest.

## Run outputs

Every `make train` creates `runs/<UTC timestamp>/` containing:

- `config.yaml`: the exact settings used
- `provenance.json`: git commit (and whether the code had uncommitted
  changes), dataset MD5, Python and package versions
- `summary.json`: accuracy and macro-F1 per model
- `<model>/metrics.json`: per-class precision/recall, confusion matrix,
  recall by degradation severity (mild / medium / severe), feature importance
- `<model>/model.joblib`: the trained model

Models are trained on `train` only. Hyperparameters are fixed in the config,
not tuned on the test set.

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

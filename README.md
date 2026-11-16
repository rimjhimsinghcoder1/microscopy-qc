# microscopy-qc

[![CI](https://github.com/YOUR_GITHUB_USERNAME/microscopy-qc/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_GITHUB_USERNAME/microscopy-qc/actions/workflows/ci.yml)

**Reproducible, explainable quality control for microscopy images: a small
decision-support system that flags out-of-focus, overexposed and noisy images,
says how confident it is, and sends uncertain cases to a person.**

![Coverage vs accuracy](reports/figures/coverage.png)

## Why

Low-quality microscopy images that slip into analysis make results unreliable
and hard to reproduce. Checking every image by eye is slow, and subtle defects
are easy to miss. This project asks: *how much of that checking can be
automated safely, and how do we know when not to trust the model?*

## Headline results

On a held-out test set of 300 images (reference run on macOS; full,
auto-generated tables in [`reports/results.md`](reports/results.md)):

| Model | Inputs | Test accuracy (95% CI) |
|---|---|---|
| Logistic regression | 7 QC features | 0.967 (0.943-0.987) |
| Random forest | 7 QC features | 0.940 (0.910-0.967) |
| CNN | image | 0.963 (0.940-0.983) |
| **Fusion** | image + features | **0.987 (0.973-0.997)** |

With a human-review policy chosen on validation data, the fusion model
**decides 94% of test images automatically at 99.6% accuracy**; the remaining
6% go to a person, and 3 of its 4 errors were among them.

## Quickstart

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
make install
make test      # 71 tests, no download needed
make all       # download data -> train 4 models -> report (~2 min on a laptop CPU)
```

`make all` runs three steps you can also run separately: `make data`,
`make train`, `make report`.

## How it works

```
BloodMNIST images --(MD5 check)--> seeded degradations --> 2,000 labelled images
      |                                                         | 70/15/15 split
      v                                                         v
 7 QC features ---------------------------> logistic regression, random forest
 raw pixels -----------------------------> small CNN
 raw pixels + QC features ---------------> fusion network
      |
      v
 temperature scaling + review threshold (fitted on validation only)
      |
      v
 report: metrics with CIs, calibration, review policy, Grad-CAM, weights
```

### 1. Data with known ground truth

Real "bad image" labels are expensive, so degradations are applied in a
controlled, seeded way to [MedMNIST v3](https://medmnist.com) BloodMNIST
(CC BY 4.0) blood-cell images (28x28, grayscale). Each source image is used
once, so no cell appears in two classes.

| QC class | Simulates | Severity drawn from |
|---|---|---|
| `good` | unchanged image | - |
| `blurry` | out-of-focus acquisition | Gaussian sigma 0.4-1.0 px |
| `overexposed` | too-long exposure / high gain | brightness x1.05-1.3, clipped |
| `noisy` | low light / sensor noise | Gaussian noise std 0.01-0.05 |

**Why such mild degradations?** A first version with stronger settings (blur
0.8-2.0, gain 1.5-2.5, noise 0.05-0.15) was solved at 99-100% by logistic
regression. That says more about the benchmark than the model, and subtle
defects are the ones that slip past a person, so the ranges were reduced until
the task was no longer trivial.

### 2. Interpretable QC features

| Feature | Measures |
|---|---|
| `laplacian_var`, `gradient_energy` | sharpness |
| `mean_intensity`, `std_intensity` | brightness, contrast |
| `p99_intensity`, `saturated_frac` | highlights, clipped pixels |
| `noise_sigma` | noise level (Immerkaer 1996 estimator) |

### 3. Four models

- **Logistic regression** and **random forest** on the features.
- **CNN** (~17k parameters) on raw pixels. No per-image normalisation, because
  brightness is the signal for overexposure; no augmentations that would change
  blur, brightness or noise, because those would change the label.
- **Fusion**: CNN embedding + standardised features -> small MLP.

The PyTorch training loop is written out explicitly in
[`src/mqc/torch_train.py`](src/mqc/torch_train.py), with early stopping on
validation loss.

### 4. From prediction to decision support

- **Calibration.** A temperature is fitted on validation log-loss. For the
  fusion model, test expected calibration error fell from 0.040 to 0.016. All
  models were slightly *under*-confident (T < 1).
- **Review policy.** On validation, the lowest confidence threshold whose
  auto-decided images reach 99% accuracy is chosen; images below it go to a
  person. The model that automates the most *on validation* is recommended.
  Test data is never used to make a choice.

![Fusion calibration](reports/figures/reliability.png)

## What the models learned

**Feature weights read like a lab checklist.** High `noise_sigma` pushes
towards *noisy*, high `p99_intensity` towards *overexposed*, high
`laplacian_var` (sharpness) towards *good*.

![Logistic regression weights](reports/figures/lr_weights.png)

**Grad-CAM shows the CNN looks in sensible places.** For *blurry* images it
attends to cell edges, where focus loss shows. For *overexposed* and *noisy*
images it attends to the flat background, where saturation and noise are
easiest to see.

![Grad-CAM](reports/figures/gradcam.png)

**Errors are subtle, not confused.** Every fusion error involves the `good`
class (a mild defect missed, or a good image flagged); no defect type is ever
mistaken for another. Medium and severe defects are caught by all models;
recall on the mildest third of each defect type is in `reports/results.md`.

![Confusion matrix](reports/figures/confusion.png)

**Honest caveats.** The random forest drops four points from validation to
test, a sign of overfitting; the simpler logistic regression generalises
better. The confidence intervals of the top models overlap, so with 300 test
images their ranking is suggestive, not conclusive.

## Reproducibility

- **Pinned environment:** exact package versions in `pyproject.toml`.
- **Configuration, not code:** every setting lives in `configs/baseline.yaml`.
- **Seeds everywhere:** data sampling, degradations, splits, model training.
- **Verified data:** source file checked against its published MD5; the built
  dataset is saved with a metadata JSON (config, seed, checksum, split sizes).
- **Run provenance:** every run saves its config, git commit (and whether the
  code had uncommitted changes), dataset MD5, Python and package versions.
- **Guarded reports:** `mqc report` refuses to run if the dataset changed since
  training.
- **Generated documentation:** `reports/results.md` is written by the code from
  the run's outputs, never by hand.
- **Tested:** 71 tests (degradations, features, splits, leakage, determinism,
  calibration, reporting) run on every push via GitHub Actions.
- **Checked across machines:** the scikit-learn results match to every reported
  digit on macOS (Apple Silicon) and Linux (x86-64); two runs on one machine
  give byte-identical results for all four models. Neural-network results can
  differ slightly across hardware (fusion test accuracy 0.987 on macOS vs 0.980
  on Linux).

## Limitations

Synthetic degradations are cleaner than real artefacts; each image has at most
one defect; important failure types (debris, uneven illumination, wrong focal
plane) are not modelled; and the images are small crops of one cell type. See
the [model card](reports/model_card.md) for details and intended use.

## Project layout

```
configs/             experiment settings
src/mqc/
  data.py            download + checksum
  degradations.py    seeded blur / exposure / noise
  dataset.py         build, split, save with metadata
  features.py        interpretable QC features
  models.py          scikit-learn models
  cnn.py             CNN and fusion network
  torch_train.py     explicit PyTorch training loop
  train.py           trains all models, writes run folders
  evaluate.py        metrics, bootstrap CIs, recall by severity
  calibration.py     ECE, temperature scaling, review policy
  explain.py         Grad-CAM
  report.py          report + figures; results_md.py writes results.md
  provenance.py      git commit, checksums, versions
  cli.py             `mqc prepare | train | report`
tests/               71 tests
reports/             figures, results.md, model card
.github/workflows/   CI
```

## License

MIT. BloodMNIST data: CC BY 4.0, Yang et al., *MedMNIST v2*, Scientific Data
(2023).

# microscopy-qc

Reproducible, explainable quality control (QC) for microscopy images.

> **Status:** work in progress, Phase 1 of 7 (project setup).

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
make run
```

## Reproducibility

- All settings live in `configs/`; nothing is hard-coded.
- One seed in the config controls all randomness.
- Dependencies are pinned to exact versions in `pyproject.toml`.

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

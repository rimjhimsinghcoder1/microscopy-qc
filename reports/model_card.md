# Model card: microscopy-qc fusion model

Structure follows Mitchell et al. (2019), *Model Cards for Model Reporting*.

## Model details

- **What it is:** a classifier that labels a grayscale microscopy image as
  `good`, `blurry`, `overexposed` or `noisy`, with a calibrated confidence.
- **Architecture:** a small CNN (~17k parameters) on the 28x28 image, whose
  embedding is concatenated with 7 interpretable QC features (sharpness,
  brightness, contrast, highlights, saturation, noise level) and passed to a
  two-layer MLP. ~20k parameters in total.
- **Training:** Adam, cross-entropy, early stopping on validation loss,
  CPU only. Settings in `configs/baseline.yaml`.
- **Post-processing:** temperature scaling fitted on validation; images below a
  confidence threshold (chosen on validation for 99% accuracy on the rest) are
  flagged for human review.
- **Version:** see `reports/results.md` for the commit and dataset checksum of
  the reported run.

## Intended use

- **Intended:** a research prototype showing how image-QC decision support can
  be built reproducibly and explained. Suitable for teaching, benchmarking and
  as a starting point for QC on a real lab's own images.
- **Not intended:** making accept/reject decisions on real experimental data
  without validation on that lab's images, instruments and failure modes. It is
  not a validated tool under GxP or any other quality framework.

## Data

- **Source:** MedMNIST v3 BloodMNIST (CC BY 4.0), 28x28 blood-cell images from
  normal donors, converted to grayscale. 2,000 images, each used once.
- **Labels:** synthetic. `good` images are unchanged; the other classes are
  made by applying seeded Gaussian blur, brightness gain with clipping, or
  Gaussian noise at deliberately mild strengths.
- **Split:** 70/15/15 train/validation/test, stratified by class.

## Performance

See `reports/results.md` (generated from the run). On the reference macOS run
the fusion model reached 0.987 test accuracy (95% CI 0.973-0.997). Under the
review policy it decided 94% of test images automatically with 99.6% accuracy;
3 of its 4 errors were among the images sent to review.

All errors involve the `good` class: mildly degraded images taken as `good`,
or `good` images flagged as degraded. No degradation type was mistaken for
another.

## Limitations

- **Synthetic degradations are not real artefacts.** Real defocus is not a
  perfect Gaussian, real noise depends on signal (Poisson), and real exposure
  problems interact with staining and the camera's response curve. Accuracy on
  real lab images will be lower until the model is trained on them.
- **One failure at a time.** Each image has at most one degradation. Real
  images can be blurry *and* noisy.
- **Missing failure types:** debris, bubbles, uneven illumination, stitching
  artefacts, wrong focal plane, contamination and staining problems.
- **Small images, one cell type.** 28x28 crops of blood cells only; organoids,
  tissue sections and full-field images will behave differently.
- **Small test set.** 300 images: the confidence intervals of the top models
  overlap, so their ranking is not conclusive.
- **Hardware variation.** Neural-network results can differ in the last
  decimals between machines (fusion test accuracy 0.987 on macOS, 0.980 on
  Linux). The scikit-learn baselines match exactly.

## Ethical and practical considerations

- A QC model that wrongly passes a bad image lets an error into downstream
  analysis silently. This is why the review policy is tuned for accuracy on
  auto-decided images, and why missed degradations are reported by severity.
- The human-review step is part of the design, not an optional extra: the
  model is meant to reduce the reviewer's workload, not to replace the review.

## How to reproduce

```bash
make install
make all   # download data, train, report
```

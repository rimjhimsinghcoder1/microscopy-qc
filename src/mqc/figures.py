"""Static report figures (PNG). One consistent, colour-blind-checked palette."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no screen needed; works on servers and in CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Categorical colours in fixed order (validated for colour-vision deficiency).
# Each model always gets the same colour across every figure.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3df", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#2a78d6", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#2a78d6", "#f0efec", "#e34948"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 9,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _save(fig, path: Path) -> Path:
    fig.savefig(path)
    plt.close(fig)
    return path


def reliability_diagram(
    curves: dict[str, list[dict]], path: Path, title: str, min_count: int = 5
) -> Path:
    """Confidence vs accuracy; the dashed diagonal is perfect calibration.

    Bins with fewer than `min_count` images are dropped: their accuracy is
    mostly noise. Marker area shows how many images each bin holds.
    """
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED)
    ax.text(0.5, 0.455, "perfect calibration", color=MUTED, rotation=45, fontsize=8)
    for color, (label, bins) in zip(SERIES, curves.items(), strict=False):
        bins = [b for b in bins if b["n"] >= min_count]
        x = [b["confidence"] for b in bins]
        y = [b["accuracy"] for b in bins]
        ax.plot(x, y, lw=2, color=color, label=label)
        ax.scatter(
            x,
            y,
            s=[8 + b["n"] / 3 for b in bins],
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=3,
        )
    ax.set(
        xlim=(0.4, 1.01),
        ylim=(0.4, 1.01),
        xlabel="Mean confidence",
        ylabel="Accuracy (test)",
        title=title,
    )
    ax.grid(color=GRID, lw=0.6)
    ax.legend(loc="upper left")
    return _save(fig, path)


def coverage_plot(curves: dict[str, dict], path: Path, marker=None) -> Path:
    """Accuracy of auto-decided images vs the share decided without a human."""
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for color, (label, c) in zip(SERIES, curves.items(), strict=False):
        ax.plot(c["coverage"], c["accuracy"], lw=2, color=color, label=label)
    if marker:
        ax.plot(
            *marker["xy"],
            "o",
            ms=8,
            color=INK,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            zorder=5,
        )
        ax.annotate(
            marker["text"],
            marker["xy"],
            xytext=(0.62, 0.965),
            textcoords="data",
            ha="left",
            color=INK,
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
        )
    ax.set(
        xlim=(0.5, 1.0),
        ylim=(0.94, 1.002),
        xlabel="Share of images decided automatically (rest go to human review)",
        ylabel="Accuracy on auto-decided images",
        title="Trading automation for accuracy",
    )
    ax.grid(color=GRID, lw=0.6)
    ax.legend(loc="lower left")
    return _save(fig, path)


def confusion_plot(cm: np.ndarray, classes: list[str], title: str, path: Path):
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    ax.imshow(cm, cmap=BLUES)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=10,
                color=SURFACE if cm[i, j] > cm.max() / 2 else INK,
            )
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    ax.spines[:].set_visible(False)
    return _save(fig, path)


def coefficient_plot(coefs: dict[str, dict[str, float]], path: Path) -> Path:
    """Logistic-regression weights: which feature pushes towards which class."""
    classes = list(coefs)
    feats = list(next(iter(coefs.values())))
    m = np.array([[coefs[c][f] for f in feats] for c in classes])
    lim = np.abs(m).max()
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    im = ax.imshow(m, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(classes)):
        for j in range(len(feats)):
            ax.text(
                j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=8, color=INK
            )
    ax.set_xticks(range(len(feats)), feats, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_title("Logistic regression weights (standardised features)")
    ax.spines[:].set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("pushes away  <->  pushes towards", color=MUTED, fontsize=8)
    cbar.outline.set_visible(False)
    return _save(fig, path)


def gradcam_grid(images, cams, titles, path: Path) -> Path:
    """Top row: input image. Bottom row: Grad-CAM overlay (brighter = more used)."""
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(1.45 * n, 3.3))
    for k in range(n):
        axes[0, k].imshow(images[k], cmap="gray", vmin=0, vmax=1)
        axes[0, k].set_title(titles[k], fontsize=7.5, loc="center", fontweight="normal")
        axes[1, k].imshow(images[k], cmap="gray", vmin=0, vmax=1)
        axes[1, k].imshow(cams[k], cmap="inferno", alpha=0.55, vmin=0, vmax=1)
        for ax in axes[:, k]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
    axes[0, 0].set_ylabel("image", color=MUTED)
    axes[1, 0].set_ylabel("Grad-CAM", color=MUTED)
    fig.suptitle(
        "Where the CNN looks (predicted class)",
        x=0.12,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    return _save(fig, path)

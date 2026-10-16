"""Generate the original synthetic scene and publication-ready comparison PNGs."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import cv2 as cv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from enhance import process, write_png


ASSETS = ROOT / "assets"
LABELS = {
    "original": "Original",
    "he": "HE (global)",
    "ahe": "AHE (tiles, no clipping)",
    "clahe": "CLAHE (clip=2)",
    "ace_lrs": "ACE / LRS (intensity bands)",
}


def synthetic_scene() -> np.ndarray:
    """A deterministic inspection-like scene with shadow, glare and noise."""
    width, height = 768, 512
    rng = np.random.default_rng(20261016)
    yy, xx = np.mgrid[:height, :width]
    base = 145 + 5 * np.sin(xx / 18) + 3 * np.cos(yy / 23)
    base = np.clip(base, 0, 255).astype(np.uint8)

    # Fine low-contrast objects deliberately span both lighting conditions.
    cv.rectangle(base, (42, 45), (330, 465), 139, 3)
    cv.circle(base, (184, 246), 113, 152, -1)
    cv.circle(base, (184, 246), 77, 142, 3)
    cv.circle(base, (184, 246), 28, 134, -1)
    cv.rectangle(base, (410, 65), (715, 448), 153, -1)
    for idx, x in enumerate(range(430, 700, 43)):
        cv.circle(base, (x, 153), 13 + idx % 3, 144, -1)
    for y in range(245, 380, 26):
        cv.line(base, (450, y), (674, y - 18), 145, 2, cv.LINE_AA)
    cv.putText(base, "LOW CONTRAST", (58, 97), cv.FONT_HERSHEY_SIMPLEX, 0.67, 155, 1, cv.LINE_AA)
    cv.putText(base, "DETAILS", (456, 421), cv.FONT_HERSHEY_SIMPLEX, 0.75, 142, 2, cv.LINE_AA)

    # Dark on the left; brighter on the right. This makes global HE struggle
    # to choose a good mapping for both regions at once.
    illumination = 0.43 + 0.88 * xx / (width - 1)
    illumination += 0.12 * np.exp(-((xx - 615) ** 2 + (yy - 160) ** 2) / (2 * 105**2))
    observed = base.astype(np.float64) * illumination
    observed += rng.normal(0, 2.4, size=(height, width))
    return np.rint(np.clip(observed, 0, 255)).astype(np.uint8)


def save_comparison(images: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(14, 17), gridspec_kw={"width_ratios": [3.1, 1.2]})
    for row, (name, image) in enumerate(images.items()):
        ax_image, ax_hist = axes[row]
        ax_image.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax_image.set_title(LABELS[name], loc="left", fontsize=15, fontweight="bold")
        ax_image.axis("off")
        ax_hist.hist(image.ravel(), bins=np.arange(257), color="#2367A5", alpha=0.85)
        ax_hist.set_xlim(0, 255)
        ax_hist.set_xlabel("Gray value")
        ax_hist.set_ylabel("Pixels")
        ax_hist.grid(alpha=0.2)
    fig.suptitle("Same input, five outputs: image and grayscale histogram", fontsize=19)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(ASSETS / "comparison.png", dpi=145, facecolor="white")
    plt.close(fig)


def save_detail(images: dict[str, np.ndarray]) -> None:
    # Magnify a dark area where the injected noise is visible.
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.0))
    for ax, (name, image) in zip(axes, images.items()):
        crop = image[180:305, 90:300]
        ax.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(LABELS[name], fontsize=10)
        ax.axis("off")
    fig.suptitle("Shadow-area detail: structure and amplified noise", fontsize=16)
    fig.tight_layout()
    fig.savefig(ASSETS / "shadow_detail.png", dpi=160, facecolor="white")
    plt.close(fig)


def save_mapping(images: dict[str, np.ndarray]) -> None:
    src = images["original"].ravel()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(src, bins=np.arange(257), color="#365B88")
    for x, label in ((85, "dark / mid"), (170, "mid / bright")):
        axes[0].axvline(x, color="#C34B34", ls="--", lw=1.5)
        axes[0].text(x + 3, axes[0].get_ylim()[1] * 0.72, label, rotation=90, color="#C34B34")
    axes[0].set(xlim=(0, 255), xlabel="Input gray value", ylabel="Pixels", title="Three intensity regions")

    # Both HE and ACE/LRS are global point mappings: all pixels with the same
    # input level receive the same output level. Local AHE/CLAHE are not.
    for name, color in (("he", "#CF6D28"), ("ace_lrs", "#216B8C")):
        dst = images[name].ravel()
        levels = np.unique(src)
        mapped = np.array([dst[np.flatnonzero(src == v)[0]] for v in levels])
        axes[1].plot(levels, mapped, lw=2, color=color, label=LABELS[name])
    axes[1].plot([0, 255], [0, 255], color="gray", ls=":", label="Identity")
    axes[1].set(xlim=(0, 255), ylim=(0, 255), xlabel="Input gray value", ylabel="Output gray value", title="One global LUT, two different designs")
    axes[1].legend(loc="lower right")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSETS / "intensity_regions.png", dpi=165, facecolor="white")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    image = synthetic_scene()
    write_png(ASSETS / "synthetic_input.png", image)
    outputs = process(image)
    save_comparison(outputs)
    save_detail(outputs)
    save_mapping(outputs)
    print("Generated:", *(p.name for p in sorted(ASSETS.glob("*.png"))))
    for name, result in outputs.items():
        print(f"{name:>8}: range=[{result.min():3d},{result.max():3d}], std={result.std():6.2f}")


if __name__ == "__main__":
    main()

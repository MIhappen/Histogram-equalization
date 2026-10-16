"""Five reproducible 8-bit grayscale contrast transforms.

The ACE/LRS function is a transparent teaching implementation inspired by
Srinivasan & Balram (2006); its strength curve is a documented choice, not
an asserted reproduction of the paper's unpublished numerical settings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np


def require_gray8(image: np.ndarray) -> None:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Expected a single-channel uint8 grayscale image")


def he(image: np.ndarray) -> np.ndarray:
    """Global histogram equalization."""
    require_gray8(image)
    return cv.equalizeHist(image)


def ahe_tiled(image: np.ndarray, grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Tile-based AHE with interpolation and an effectively inactive clip cap.

For an 8-bit tile, OpenCV's internal clip threshold is proportional to
clipLimit * tile_area / 256. Setting clipLimit=256 makes that threshold at
least the tile area, so no histogram bin can exceed it. This is tile AHE,
not the slower classical sliding-window AHE at every pixel.
    """
    require_gray8(image)
    return cv.createCLAHE(clipLimit=256.0, tileGridSize=grid).apply(image)


def clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """OpenCV contrast-limited tile AHE."""
    require_gray8(image)
    if clip_limit <= 0:
        raise ValueError("clip_limit must be positive")
    return cv.createCLAHE(clipLimit=clip_limit, tileGridSize=grid).apply(image)


def ace_lrs(
    image: np.ndarray,
    strengths: tuple[float, float, float] = (0.95, 0.65, 0.35),
) -> np.ndarray:
    """Adaptive contrast enhancement in three *intensity* regions.

Each region is equalized within its own intensity band. The result is blended
with the input. Its weight follows a triangular function of mean absolute
deviation (MAD): near-zero and near-maximal spread receive little weight.
This captures the paper's described design, while the exact band endpoints,
triangular peak and strengths are explicit teaching choices.
    """
    require_gray8(image)
    if len(strengths) != 3 or any(not 0 <= s <= 1 for s in strengths):
        raise ValueError("strengths must contain three numbers in [0, 1]")

    # [0, 85), [85, 170), [170, 256): contiguous, non-overlapping bands.
    edges = (0, 85, 170, 256)
    lut = np.arange(256, dtype=np.float64)
    hist = np.bincount(image.ravel(), minlength=256)
    for band in range(3):
        lo, hi = edges[band : band + 2]
        counts = hist[lo:hi]
        total = int(counts.sum())
        present = np.flatnonzero(counts)
        if total < 2 or len(present) < 2:
            continue

        levels = np.arange(lo, hi, dtype=np.float64)
        mean = float(np.dot(levels, counts) / total)
        mad = float(np.dot(np.abs(levels - mean), counts) / total)
        # For values in a fixed band, maximum MAD is approximately half
        # the band width. The triangle peaks at half this maximum.
        spread = np.clip(mad / ((hi - lo - 1) / 2), 0.0, 1.0)
        triangle = max(0.0, 1.0 - abs(2.0 * spread - 1.0))
        weight = strengths[band] * triangle
        if weight == 0:
            continue

        cdf = np.cumsum(counts, dtype=np.float64)
        first_cdf = cdf[present[0]]
        mapped = lo + (cdf - first_cdf) * (hi - lo - 1) / (total - first_cdf)
        mapped = np.clip(mapped, lo, hi - 1)
        lut[lo:hi] = (1.0 - weight) * levels + weight * mapped

    return cv.LUT(image, np.rint(np.clip(lut, 0, 255)).astype(np.uint8))


def process(image: np.ndarray) -> dict[str, np.ndarray]:
    require_gray8(image)
    return {
        "original": image,
        "he": he(image),
        "ahe": ahe_tiled(image),
        "clahe": clahe(image),
        "ace_lrs": ace_lrs(image),
    }


def read_gray(path: Path) -> np.ndarray | None:
    """Read paths containing non-ASCII characters on Windows, too."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv.imdecode(data, cv.IMREAD_GRAYSCALE) if data.size else None


def write_png(path: Path, image: np.ndarray) -> None:
    ok, data = cv.imencode(".png", image)
    if not ok:
        raise OSError(f"Cannot encode {path}")
    data.tofile(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="8-bit grayscale or BGR image")
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    image = read_gray(args.input)
    if image is None:
        parser.error(f"Cannot read image: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, result in process(image).items():
        write_png(args.output / f"{name}.png", result)
    print(f"Saved 5 images to {args.output.resolve()}")


if __name__ == "__main__":
    main()

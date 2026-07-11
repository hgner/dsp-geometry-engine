"""Synthetic 2-D image factories shared by the imaging-pack tests.

Deterministic, dependency-light (numpy + Pillow). Every factory returns a float
image in [0, 1]; the ``save_png*`` helpers write the exact formats the tools read
(8-bit and 16-bit grayscale PNG).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_png8(img: np.ndarray, path: Path | str) -> Path:
    """8-bit grayscale PNG."""
    path = Path(path)
    Image.fromarray((np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)).save(path)
    return path


def save_png16(img: np.ndarray, path: Path | str) -> Path:
    """16-bit grayscale PNG (Pillow mode I;16)."""
    path = Path(path)
    Image.fromarray((np.clip(img, 0.0, 1.0) * 65535.0).round().astype(np.uint16)).save(path)
    return path


def make_low_contrast(size: int = 128, lo: float = 0.35, hi: float = 0.55, seed: int = 0) -> np.ndarray:
    """A textured image squeezed into a narrow intensity band [lo, hi] — the
    canonical input where histogram equalization must widen the spread."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    texture = 0.5 + 0.5 * np.sin(2.0 * np.pi * 6.0 * xx / size) * np.cos(2.0 * np.pi * 4.0 * yy / size)
    texture = 0.7 * texture + 0.3 * rng.random((size, size))
    return lo + (hi - lo) * np.clip(texture, 0.0, 1.0)


def make_rects(size: int = 128) -> tuple[np.ndarray, list[dict]]:
    """Bright rectangles on a dark field, plus the ground-truth component list
    (area, bbox as (x0, y0, x1, y1) exclusive) — the segmentation golden."""
    img = np.full((size, size), 0.1, dtype=np.float64)
    specs = [
        {"x0": 10, "y0": 12, "x1": 30, "y1": 40},  # 20 x 28 = 560 px
        {"x0": 70, "y0": 60, "x1": 100, "y1": 80},  # 30 x 20 = 600 px
    ]
    truth = []
    for s in specs:
        img[s["y0"] : s["y1"], s["x0"] : s["x1"]] = 0.9
        truth.append(
            {
                "area_px": (s["x1"] - s["x0"]) * (s["y1"] - s["y0"]),
                "bbox": (s["x0"], s["y0"], s["x1"], s["y1"]),
            }
        )
    truth.sort(key=lambda t: -t["area_px"])  # component_stats sorts area-descending
    return img, truth


def make_oriented_ripple(
    size: int = 128, cycles: float = 24.0, orient_deg: float = 0.0, amp: float = 0.4, base: float = 0.5
) -> np.ndarray:
    """A single-frequency oriented sinusoid — used to measure filter attenuation
    (its amplitude before/after a known-gain filter is exactly checkable)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    theta = np.radians(orient_deg)
    freq_cpp = cycles / size
    phase = 2.0 * np.pi * freq_cpp * (xx * np.cos(theta) + yy * np.sin(theta))
    return base + amp * np.sin(phase)


def make_noisy(size: int = 128, sigma: float = 0.08, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(clean, noisy): a smooth ramp+blob image and its additive-Gaussian-noise
    version — the denoiser must move the noisy image back toward the clean one."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    clean = (
        0.3
        + 0.4 * (xx / size)
        + 0.2 * np.exp(-(((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (2.0 * (size / 6.0) ** 2)))
    )
    clean = np.clip(clean, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    noisy = np.clip(clean + rng.normal(0.0, sigma, clean.shape), 0.0, 1.0)
    return clean, noisy

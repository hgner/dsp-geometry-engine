"""2-D image lane (ELE407 week 14): spectral analysis of depth/AOV renders.

The corrugation defect was originally observed in depth renders, so this module
cross-validates the mesh-space findings in render space. A periodic corrugation
shows up as an oriented ridge in the centered 2-D power spectrum: the ridge
position gives the spatial frequency in cycles/pixel (convert to cycles/meter
via the render's camera geometry when available, else report per-image-width)
and its angle gives the ripple orientation.

Pure scipy + Pillow — no scikit-image/OpenCV. SSIM is hand-rolled on
``scipy.ndimage.uniform_filter`` windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import fft, ndimage
from scipy.signal import windows

__all__ = [
    "ImageComparison",
    "Roi",
    "Spectrum2D",
    "auto_roi",
    "compare_images",
    "crop_roi",
    "load_image_gray",
    "roi_spectrum_2d",
    "ssim",
]

# (x0, y0, x1, y1) in pixel coordinates, upper bounds EXCLUSIVE (numpy slicing:
# img[y0:y1, x0:x1]).
Roi = tuple[int, int, int, int]

_DC_RADIUS_BINS = 3  # zeroed disk around DC, in FFT bins
_N_ORIENTATION_BINS = 36  # 5 degrees per bin over the 180-degree half-plane
_SIXTEEN_BIT_MODES = ("I;16", "I;16L", "I;16B", "I;16N", "I")


def load_image_gray(path: Path | str) -> np.ndarray:
    """Load an image as an (h, w) float64 grayscale array in [0, 1].

    16-bit PNGs (Pillow modes ``I;16`` variants and ``I``) are normalized by
    65535; float images are clipped as-is; everything else goes through
    Pillow's ``convert("L")`` and /255.
    """
    with Image.open(path) as im:
        if im.mode in _SIXTEEN_BIT_MODES:
            arr = np.asarray(im, dtype=np.float64) / 65535.0
        elif im.mode == "F":
            arr = np.asarray(im, dtype=np.float64)
        else:
            arr = np.asarray(im.convert("L"), dtype=np.float64) / 255.0
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a single-channel image, got array shape {arr.shape}")
    return np.clip(arr, 0.0, 1.0)


def crop_roi(img: np.ndarray, roi: Roi) -> np.ndarray:
    """Crop ``img`` to ``roi`` = (x0, y0, x1, y1), clamping to image bounds.

    Raises ValueError if the ROI is empty after clamping.
    """
    h, w = img.shape
    x0, y0, x1, y1 = (int(v) for v in roi)
    x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
    y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"roi {roi} is empty after clamping to the {w}x{h} image")
    return img[y0:y1, x0:x1]


def auto_roi(img: np.ndarray, percentile: float = 60.0) -> Roi:
    """ROI around the largest connected region of strong gradient magnitude.

    gaussian_gradient_magnitude -> percentile threshold -> label -> bounding
    box of the largest component. Falls back to the full image when nothing
    usable stands out (flat image or a degenerate region).
    """
    h, w = img.shape
    full: Roi = (0, 0, w, h)
    grad = ndimage.gaussian_gradient_magnitude(np.asarray(img, dtype=np.float64), sigma=2.0)
    mask = grad > float(np.percentile(grad, percentile))
    if not mask.any():
        return full
    labels, _n_regions = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # background
    largest = int(np.argmax(sizes))
    sly, slx = ndimage.find_objects(labels, max_label=largest)[largest - 1]
    roi: Roi = (int(slx.start), int(sly.start), int(slx.stop), int(sly.stop))
    if roi[2] - roi[0] < 8 or roi[3] - roi[1] < 8:
        return full
    return roi


@dataclass
class Spectrum2D:
    """Centered 2-D power spectrum of one (optionally ROI-cropped) image.

    All frequencies are in cycles/pixel ("cpp"). Orientation is the angle of
    the dominant wavevector in image coordinates (x right, y down), folded to
    [0, 180): 0 deg = ripple varying along x (vertical stripes).
    """

    freqs_x_cpp: np.ndarray  # (w,) fftshifted frequency axis
    freqs_y_cpp: np.ndarray  # (h,) fftshifted frequency axis
    psd: np.ndarray  # (h, w) centered |F|^2, small DC disk zeroed
    dominant_freq_cpp: float  # radial frequency of the strongest bin
    dominant_orientation_deg: float  # [0, 180)
    dominant_prominence_db: float  # peak over median background, dB
    radial_freqs_cpp: np.ndarray  # (n_r,) annular bin centers
    radial_profile: np.ndarray  # (n_r,) mean psd per annulus
    orientation_bins_deg: np.ndarray  # (n_t,) angular bin centers in [0, 180)
    orientation_profile: np.ndarray  # (n_t,) mean psd per angular bin


def _detrend_plane(img: np.ndarray) -> np.ndarray:
    """Subtract the least-squares plane a + b*x + c*y (removes mean + tilt)."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    basis = np.column_stack([np.ones(img.size), xx.ravel(), yy.ravel()])
    coef, *_ = np.linalg.lstsq(basis, img.ravel(), rcond=None)
    return img - (basis @ coef).reshape(h, w)


def roi_spectrum_2d(img: np.ndarray, roi: Roi | None = None, detrend: bool = True) -> Spectrum2D:
    """Windowed 2-D power spectrum of ``img`` (or of ``roi`` within it).

    ``roi=None`` analyzes the full image (pass ``auto_roi(img)`` explicitly
    for detection). With ``detrend`` a plane fit is subtracted first so smooth
    depth gradients do not mask the corrugation band; a 2-D Hann window bounds
    leakage; a small disk around DC is zeroed so argmax lands on structure.
    """
    patch = crop_roi(img, roi) if roi is not None else np.asarray(img)
    if patch.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale array, got shape {patch.shape}")
    h, w = patch.shape
    if h < 8 or w < 8:
        raise ValueError(f"patch {w}x{h} is too small for spectral analysis (need >= 8x8)")

    patch = patch.astype(np.float64, copy=True)
    if detrend:
        patch = _detrend_plane(patch)
    win = np.outer(windows.hann(h, sym=False), windows.hann(w, sym=False))
    psd = fft.fftshift(np.abs(fft.fft2(patch * win)) ** 2) / (h * w)

    freqs_x = fft.fftshift(fft.fftfreq(w))
    freqs_y = fft.fftshift(fft.fftfreq(h))
    fx_grid, fy_grid = np.meshgrid(freqs_x, freqs_y)

    # Zero a small disk around DC (in bin distance, so h != w stays isotropic).
    kx_grid, ky_grid = np.meshgrid(np.arange(w) - w // 2, np.arange(h) - h // 2)
    bin_dist = np.hypot(kx_grid, ky_grid)
    dc_radius = min(_DC_RADIUS_BINS, max(1, min(h, w) // 8))
    dc_mask = bin_dist <= dc_radius
    psd[dc_mask] = 0.0

    iy, ix = np.unravel_index(int(np.argmax(psd)), psd.shape)
    peak = float(psd[iy, ix])
    if peak > 0.0:
        px, py = float(freqs_x[ix]), float(freqs_y[iy])
        dominant_freq = float(np.hypot(px, py))
        orientation = float(np.degrees(np.arctan2(py, px)) % 180.0)
        background = float(np.median(psd[~dc_mask]))
        prominence_db = 10.0 * float(np.log10(peak / max(background, peak * 1e-12)))
    else:  # constant patch: no structure at all
        dominant_freq = orientation = prominence_db = 0.0

    # Radial fold: mean psd in annular frequency bins.
    radial = np.hypot(fx_grid, fy_grid)
    n_r = max(8, min(h, w) // 4)
    edges = np.linspace(0.0, float(radial.max()), n_r + 1)
    r_idx = np.clip(np.digitize(radial.ravel(), edges) - 1, 0, n_r - 1)
    r_counts = np.bincount(r_idx, minlength=n_r)
    r_sums = np.bincount(r_idx, weights=psd.ravel(), minlength=n_r)

    # Orientation fold: mean psd in angular bins over the 180-degree half-plane.
    angles = np.degrees(np.arctan2(fy_grid, fx_grid)) % 180.0
    theta_width = 180.0 / _N_ORIENTATION_BINS
    t_idx = np.clip((angles.ravel() / theta_width).astype(np.int64), 0, _N_ORIENTATION_BINS - 1)
    valid = radial.ravel() > 0.0  # the DC point has no angle
    t_counts = np.bincount(t_idx[valid], minlength=_N_ORIENTATION_BINS)
    t_sums = np.bincount(t_idx[valid], weights=psd.ravel()[valid], minlength=_N_ORIENTATION_BINS)

    return Spectrum2D(
        freqs_x_cpp=freqs_x,
        freqs_y_cpp=freqs_y,
        psd=psd,
        dominant_freq_cpp=dominant_freq,
        dominant_orientation_deg=orientation,
        dominant_prominence_db=prominence_db,
        radial_freqs_cpp=0.5 * (edges[:-1] + edges[1:]),
        radial_profile=r_sums / np.maximum(r_counts, 1),
        orientation_bins_deg=(np.arange(_N_ORIENTATION_BINS) + 0.5) * theta_width,
        orientation_profile=t_sums / np.maximum(t_counts, 1),
    )


def _normalize01(img: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]; a constant image maps to all zeros."""
    img = np.asarray(img, dtype=np.float64)
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def ssim(a: np.ndarray, b: np.ndarray, window: int = 11, k1: float = 0.01, k2: float = 0.03) -> float:
    """Classic mean SSIM (Wang et al. 2004) on uniform-filter local statistics.

    Both images are min-max normalized to [0, 1] first, so data_range == 1.0.
    The mean is taken over the map interior (window//2 border cropped) to keep
    boundary-padding artifacts out of the score.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"ssim needs same-shape images, got {a.shape} vs {b.shape}")
    if a.ndim != 2:
        raise ValueError(f"expected 2-D grayscale arrays, got shape {a.shape}")
    a = _normalize01(a)
    b = _normalize01(b)

    def local_mean(x: np.ndarray) -> np.ndarray:
        return ndimage.uniform_filter(x, size=window, mode="reflect")

    mu_a, mu_b = local_mean(a), local_mean(b)
    var_a = local_mean(a * a) - mu_a**2
    var_b = local_mean(b * b) - mu_b**2
    cov = local_mean(a * b) - mu_a * mu_b
    c1 = k1 * k1  # (k1 * data_range)^2 with data_range == 1.0
    c2 = k2 * k2
    ssim_map = ((2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )
    pad = window // 2
    core = ssim_map[pad : ssim_map.shape[0] - pad, pad : ssim_map.shape[1] - pad]
    return float(core.mean() if core.size else ssim_map.mean())


@dataclass
class ImageComparison:
    """Result of :func:`compare_images` — difference stats plus per-image
    spectral summaries (dominant frequency / orientation / prominence live on
    the embedded :class:`Spectrum2D` objects)."""

    path_a: str
    path_b: str
    shape: tuple[int, int]  # (h, w) of the compared (ROI-cropped) arrays
    spectrum_a: Spectrum2D
    spectrum_b: Spectrum2D
    mae: float  # on [0,1]-normalized aligned arrays
    rms: float
    ssim: float


def compare_images(path_a: Path | str, path_b: Path | str, roi: Roi | None = None) -> ImageComparison:
    """Load two grayscale images and compare them (same ROI on both).

    Difference stats (MAE/RMS/SSIM) run on the [0,1] min-max-normalized ROI
    arrays; spectra run per image on the un-renormalized ROI. Shape mismatch
    raises ValueError naming both shapes.
    """
    a = load_image_gray(path_a)
    b = load_image_gray(path_b)
    if a.shape != b.shape:
        raise ValueError(
            f"image shapes differ: {Path(path_a).name} is {a.shape}, {Path(path_b).name} is {b.shape}"
        )
    if roi is not None:
        a = crop_roi(a, roi)
        b = crop_roi(b, roi)
    a_norm = _normalize01(a)
    b_norm = _normalize01(b)
    diff = a_norm - b_norm
    return ImageComparison(
        path_a=str(path_a),
        path_b=str(path_b),
        shape=(int(a.shape[0]), int(a.shape[1])),
        spectrum_a=roi_spectrum_2d(a),
        spectrum_b=roi_spectrum_2d(b),
        mae=float(np.mean(np.abs(diff))),
        rms=float(np.sqrt(np.mean(diff * diff))),
        ssim=ssim(a_norm, b_norm),
    )

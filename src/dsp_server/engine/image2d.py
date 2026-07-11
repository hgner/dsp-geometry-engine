"""2-D image lane (ELE407 week 14): spectral analysis of depth/AOV renders.

The corrugation defect was originally observed in depth renders, so this module
cross-validates the mesh-space findings in render space. A periodic corrugation
shows up as an oriented ridge in the centered 2-D power spectrum: the ridge
position gives the spatial frequency in cycles/pixel (convert to cycles/meter
via the render's camera geometry when available, else report per-image-width)
and its angle gives the ripple orientation.

Pure scipy + Pillow — no scikit-image/OpenCV. SSIM is hand-rolled on
``scipy.ndimage.uniform_filter`` windows.

The ELE490 section at the bottom (enhancement / filtering / segmentation /
restoration kernels + 16-bit PNG output via :func:`save_image_gray`) backs the
imaging v2 tools; everything above it is the frozen week-14 spectral lane.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import fft, ndimage, signal
from scipy.signal import windows

__all__ = [
    "MORPH_MODES",
    "ComponentStats",
    "HistStats",
    "ImageComparison",
    "Roi",
    "Spectrum2D",
    "adaptive_equalize",
    "adaptive_threshold_mask",
    "adjust_gamma",
    "auto_roi",
    "binary_cleanup",
    "butterworth_filter",
    "compare_images",
    "component_stats",
    "crop_roi",
    "equalize_hist",
    "gaussian_psf",
    "gaussian_smooth",
    "hist_stats",
    "histogram256",
    "kmeans_gray",
    "laplacian_sharpness",
    "load_image_gray",
    "log_transform",
    "median_smooth",
    "otsu_threshold",
    "richardson_lucy",
    "roi_spectrum_2d",
    "save_image_gray",
    "save_mask_png",
    "sobel_magnitude",
    "ssim",
    "unsharp_mask",
    "wiener_deconvolve",
    "wiener_denoise",
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


# --------------------------------------------------------------------------- #
# ELE490 imaging v2 (Gonzalez & Woods lanes): intensity transformations,
# spatial + frequency-domain filtering, segmentation + morphology, restoration.
# Pure numpy/scipy/Pillow, appended without touching the week-14 spectral lane.

MORPH_MODES = ("none", "open", "close", "open_close")


def _as_gray(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("expected a non-empty 2-D grayscale array")
    return arr


def _bin_index(arr: np.ndarray, bins: int) -> np.ndarray:
    """Histogram bin index per pixel for values nominally in [0, 1] (clipped)."""
    return np.clip((arr * bins).astype(np.int64), 0, bins - 1)


def save_image_gray(img: np.ndarray, path: Path | str) -> Path:
    """Write a float image as a 16-bit grayscale PNG (Pillow mode ``I;16``).

    Values are clipped to [0, 1] then quantized by 65535, so
    :func:`load_image_gray` round-trips within 1/65535. Parent directories are
    created on demand. Returns the Path written.
    """
    arr = _as_gray(img)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    u16 = np.round(np.clip(arr, 0.0, 1.0) * 65535.0).astype("<u2")
    im = Image.frombytes("I;16", (u16.shape[1], u16.shape[0]), np.ascontiguousarray(u16).tobytes())
    im.save(path)
    return path


def save_mask_png(mask: np.ndarray, path: Path | str) -> Path:
    """Write a boolean mask as an 8-bit {0, 255} PNG (the mask exception to the
    16-bit output convention). Returns the Path written."""
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {m.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(m.astype(bool), 255, 0).astype(np.uint8)).save(path)
    return path


@dataclass
class HistStats:
    """Compact intensity statistics of one grayscale image (values in [0, 1])."""

    mean: float
    std: float
    p01: float
    p50: float
    p99: float
    entropy_bits: float  # Shannon entropy of the 256-bin histogram


def histogram256(img: np.ndarray, bins: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """(counts, edges) of the ``bins``-bin histogram over [0, 1] — plot-ready."""
    arr = _as_gray(img)
    counts, edges = np.histogram(arr, bins=bins, range=(0.0, 1.0))
    return counts, edges


def hist_stats(img: np.ndarray, bins: int = 256) -> HistStats:
    """Mean/std, 1/50/99 percentiles, and histogram entropy in bits."""
    arr = _as_gray(img)
    counts, _edges = histogram256(arr, bins)
    total = float(counts.sum())
    if total > 0.0:
        q = counts[counts > 0].astype(np.float64) / total
        entropy = float(-(q * np.log2(q)).sum())
    else:
        entropy = 0.0
    p01, p50, p99 = (float(v) for v in np.percentile(arr, [1.0, 50.0, 99.0]))
    return HistStats(
        mean=float(arr.mean()), std=float(arr.std()), p01=p01, p50=p50, p99=p99, entropy_bits=entropy
    )


def equalize_hist(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """Global histogram equalization: each pixel maps to the CDF of its bin.

    A full-range uniform ramp is a fixed point within 1/bins; being a
    deterministic per-bin remap it can spread values but never increase the
    ``bins``-bin histogram entropy.
    """
    arr = _as_gray(img)
    counts, _edges = np.histogram(arr, bins=bins, range=(0.0, 1.0))
    cdf = np.cumsum(counts).astype(np.float64)
    if cdf[-1] <= 0.0:
        return np.zeros_like(arr)
    cdf /= cdf[-1]
    return cdf[_bin_index(arr, bins)]


def adaptive_equalize(
    img: np.ndarray, tiles: int = 8, clip_limit: float = 0.01, bins: int = 256
) -> np.ndarray:
    """CLAHE-like adaptive equalization on a ``tiles`` x ``tiles`` grid.

    The image is symmetric-padded to a tile multiple; each tile gets a clipped
    histogram (ceiling = ``clip_limit`` * tile pixel count, clipped excess
    redistributed uniformly; 0 disables clipping) whose CDF becomes the tile's
    mapping; per-pixel output bilinearly interpolates the four nearest tile
    mappings; the pad is cropped back off.
    """
    arr = _as_gray(img)
    tiles = int(tiles)
    if tiles < 1:
        raise ValueError(f"tiles must be >= 1, got {tiles}")
    if clip_limit < 0.0:
        raise ValueError(f"clip_limit must be >= 0, got {clip_limit}")
    h, w = arr.shape
    th = -(-h // tiles)  # ceil division
    tw = -(-w // tiles)
    pad_h, pad_w = th * tiles - h, tw * tiles - w
    padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="symmetric") if (pad_h or pad_w) else arr
    n_tile_px = th * tw

    mappings = np.zeros((tiles, tiles, bins), dtype=np.float64)
    for i in range(tiles):
        for j in range(tiles):
            tile = padded[i * th : (i + 1) * th, j * tw : (j + 1) * tw]
            counts = np.histogram(tile, bins=bins, range=(0.0, 1.0))[0].astype(np.float64)
            if clip_limit > 0.0:
                ceiling = max(1.0, clip_limit * n_tile_px)
                excess = float(np.maximum(counts - ceiling, 0.0).sum())
                counts = np.minimum(counts, ceiling) + excess / bins
            cdf = np.cumsum(counts)
            if cdf[-1] > 0.0:
                mappings[i, j] = cdf / cdf[-1]

    hh, ww = padded.shape
    gy = np.clip((np.arange(hh, dtype=np.float64) + 0.5) / th - 0.5, 0.0, tiles - 1.0)
    gx = np.clip((np.arange(ww, dtype=np.float64) + 0.5) / tw - 0.5, 0.0, tiles - 1.0)
    iy0 = np.minimum(np.floor(gy).astype(np.int64), tiles - 1)
    ix0 = np.minimum(np.floor(gx).astype(np.int64), tiles - 1)
    iy1 = np.minimum(iy0 + 1, tiles - 1)
    ix1 = np.minimum(ix0 + 1, tiles - 1)
    fy = (gy - iy0)[:, None]
    fx = (gx - ix0)[None, :]
    b = _bin_index(padded, bins)
    iy0c, iy1c = iy0[:, None], iy1[:, None]
    ix0c, ix1c = ix0[None, :], ix1[None, :]
    out = (
        (1.0 - fy) * (1.0 - fx) * mappings[iy0c, ix0c, b]
        + (1.0 - fy) * fx * mappings[iy0c, ix1c, b]
        + fy * (1.0 - fx) * mappings[iy1c, ix0c, b]
        + fy * fx * mappings[iy1c, ix1c, b]
    )
    return out[:h, :w]


def adjust_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Power-law intensity transform s = r**gamma on [0, 1] (gamma > 0)."""
    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    return np.clip(_as_gray(img), 0.0, 1.0) ** float(gamma)


def log_transform(img: np.ndarray, scale: float = 255.0) -> np.ndarray:
    """Log intensity transform s = log1p(scale*r)/log1p(scale): 0->0 and 1->1 exactly."""
    if scale <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    return np.log1p(float(scale) * np.clip(_as_gray(img), 0.0, 1.0)) / np.log1p(float(scale))


def gaussian_smooth(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian low-pass (thin ndimage wrapper keeps scipy out of toolsets/)."""
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    return ndimage.gaussian_filter(_as_gray(img), sigma=float(sigma), mode="reflect")


def median_smooth(img: np.ndarray, size: int = 3) -> np.ndarray:
    """Median filter over a ``size`` x ``size`` neighbourhood."""
    if int(size) < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    return ndimage.median_filter(_as_gray(img), size=int(size), mode="reflect")


def unsharp_mask(img: np.ndarray, sigma: float = 2.0, amount: float = 1.0) -> np.ndarray:
    """Unsharp masking: out = img + amount * (img - gaussian(img, sigma)). Unclipped."""
    arr = _as_gray(img)
    return arr + float(amount) * (arr - gaussian_smooth(arr, sigma=sigma))


def sobel_magnitude(img: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude normalized to [0, 1] by its max; flat input -> zeros."""
    arr = _as_gray(img)
    mag = np.hypot(ndimage.sobel(arr, axis=0, mode="reflect"), ndimage.sobel(arr, axis=1, mode="reflect"))
    peak = float(mag.max())
    return mag / peak if peak > 0.0 else np.zeros_like(arr)


def butterworth_filter(
    img: np.ndarray, cutoff_cpp: float, order: int = 2, highpass: bool = False
) -> np.ndarray:
    """Frequency-domain Butterworth filter, Gonzalez-Woods convention.

    Low-pass transfer function H = 1 / (1 + (D/D0)^(2*order)) on the radial
    fftfreq grid, D0 = ``cutoff_cpp`` in cycles/pixel (the unit shared with
    Spectrum2D). The high-pass is 1 - H, which equals the G&W high-pass
    1 / (1 + (D0/D)^(2*order)) identically. At D/D0 = 3, order 2 the low-pass
    gain is 1/82 (~82x attenuation). Output is real and unclipped.
    """
    arr = _as_gray(img)
    if cutoff_cpp <= 0.0:
        raise ValueError(f"cutoff_cpp must be > 0, got {cutoff_cpp}")
    if int(order) < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    h, w = arr.shape
    fx_grid, fy_grid = np.meshgrid(fft.fftfreq(w), fft.fftfreq(h))
    d = np.hypot(fx_grid, fy_grid)
    transfer = 1.0 / (1.0 + (d / float(cutoff_cpp)) ** (2 * int(order)))
    if highpass:
        transfer = 1.0 - transfer
    return fft.ifft2(fft.fft2(arr) * transfer).real


def otsu_threshold(img: np.ndarray, bins: int = 256) -> float:
    """Otsu's global threshold over a ``bins``-bin histogram on [0, 1].

    Maximizes the between-class variance; when the maximum is a plateau (an
    empty valley between two well-separated modes) the plateau MIDPOINT is
    returned, so a clean bimodal image thresholds midway between its modes.
    Foreground convention: bright pixels are ``img > threshold``.
    """
    arr = _as_gray(img)
    counts, edges = np.histogram(arr, bins=bins, range=(0.0, 1.0))
    total = float(counts.sum())
    if total <= 0.0:
        raise ValueError("otsu_threshold: image has no pixels in [0, 1]")
    p = counts.astype(np.float64) / total
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    w1 = 1.0 - w0
    valid = (w0 > 0.0) & (w1 > 0.0)
    if not valid.any():  # constant image: no split exists
        return float(centers[int(np.argmax(counts))])
    sigma_b = np.zeros(bins, dtype=np.float64)
    sigma_b[valid] = (mu_t * w0[valid] - mu[valid]) ** 2 / (w0[valid] * w1[valid])
    peak_bins = np.flatnonzero(sigma_b == sigma_b.max())
    k = int(round(float(peak_bins.mean())))
    return float(edges[k + 1])


def adaptive_threshold_mask(
    img: np.ndarray, block: int = 31, offset: float = 0.02, foreground: str = "bright"
) -> np.ndarray:
    """Local-mean adaptive threshold: bright foreground is img > mean_block + offset,
    dark foreground is img < mean_block - offset. Returns a bool mask."""
    arr = _as_gray(img)
    if int(block) < 3:
        raise ValueError(f"block must be >= 3, got {block}")
    if foreground not in ("bright", "dark"):
        raise ValueError(f"foreground must be 'bright' or 'dark', got '{foreground}'")
    local_mean = ndimage.uniform_filter(arr, size=int(block), mode="reflect")
    if foreground == "bright":
        return arr > local_mean + float(offset)
    return arr < local_mean - float(offset)


def kmeans_gray(
    img: np.ndarray, k: int = 2, bins: int = 256, iters: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """1-D Lloyd k-means on the ``bins``-bin gray histogram — deterministic.

    Centers initialize at the data quantiles ((i+0.5)/k), iterate histogram-
    weighted assignment/update until convergence or ``iters``. Returns
    ``(label_img, centers)`` with centers ascending and labels remapped so
    label 0 is the darkest cluster and label k-1 the brightest.
    """
    arr = _as_gray(img)
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > bins:
        raise ValueError(f"k must be <= bins ({bins}), got {k}")
    counts, edges = np.histogram(arr, bins=bins, range=(0.0, 1.0))
    centers_bins = 0.5 * (edges[:-1] + edges[1:])
    c = np.quantile(arr, [(i + 0.5) / k for i in range(k)])
    assign = np.zeros(bins, dtype=np.int64)
    for _ in range(int(iters)):
        assign = np.argmin(np.abs(centers_bins[:, None] - c[None, :]), axis=1)
        new_c = c.copy()
        for j in range(k):
            sel = (assign == j) & (counts > 0)
            weight = float(counts[sel].sum())
            if weight > 0.0:
                new_c[j] = float((counts[sel] * centers_bins[sel]).sum() / weight)
        if np.allclose(new_c, c, rtol=0.0, atol=1e-12):
            c = new_c
            break
        c = new_c
    assign = np.argmin(np.abs(centers_bins[:, None] - c[None, :]), axis=1)
    order = np.argsort(c, kind="stable")
    remap = np.empty(k, dtype=np.int64)
    remap[order] = np.arange(k)
    label_img = remap[assign][_bin_index(arr, bins)]
    return label_img, np.asarray(c, dtype=np.float64)[order]


def binary_cleanup(mask: np.ndarray, mode: str = "open_close", size: int = 3) -> np.ndarray:
    """Binary morphology cleanup with a ``size`` x ``size`` structuring element.

    ``mode`` in MORPH_MODES = (none, open, close, open_close); ``open_close``
    is opening (kills speckle) then closing (fills pinholes).
    """
    m = np.asarray(mask).astype(bool)
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {m.shape}")
    if mode not in MORPH_MODES:
        raise ValueError(f"unknown morph mode '{mode}' — expected one of {list(MORPH_MODES)}")
    if int(size) < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    if mode == "none" or int(size) == 1:
        return m
    structure = np.ones((int(size), int(size)), dtype=bool)
    if mode in ("open", "open_close"):
        m = ndimage.binary_opening(m, structure=structure)
    if mode in ("close", "open_close"):
        m = ndimage.binary_closing(m, structure=structure)
    return m


@dataclass
class ComponentStats:
    """One connected component: geometry in FULL-IMAGE pixel coordinates
    (``offset_xy`` re-adds any ROI crop offset), intensity from the source image."""

    label: int
    area_px: int
    bbox: Roi  # (x0, y0, x1, y1), exclusive upper bounds
    centroid_xy: tuple[float, float]
    mean_intensity: float


def component_stats(
    mask: np.ndarray,
    img: np.ndarray,
    offset_xy: tuple[int, int] = (0, 0),
    min_area_px: int = 1,
) -> list[ComponentStats]:
    """4-connected components of ``mask``, area-filtered, sorted by area descending
    (ties by label). bbox/centroid are shifted by ``offset_xy`` into full-image
    coordinates; ``mean_intensity`` averages ``img`` over the component."""
    m = np.asarray(mask).astype(bool)
    arr = _as_gray(img)
    if m.shape != arr.shape:
        raise ValueError(f"mask shape {m.shape} != image shape {arr.shape}")
    labels, n_labels = ndimage.label(m)
    if n_labels == 0:
        return []
    areas = np.bincount(labels.ravel(), minlength=n_labels + 1)
    objects = ndimage.find_objects(labels)
    off_x, off_y = int(offset_xy[0]), int(offset_xy[1])
    out: list[ComponentStats] = []
    for lab in range(1, n_labels + 1):
        area = int(areas[lab])
        if area < int(min_area_px):
            continue
        sly, slx = objects[lab - 1]
        comp = labels[sly, slx] == lab
        ys, xs = np.nonzero(comp)
        out.append(
            ComponentStats(
                label=lab,
                area_px=area,
                bbox=(
                    int(slx.start) + off_x,
                    int(sly.start) + off_y,
                    int(slx.stop) + off_x,
                    int(sly.stop) + off_y,
                ),
                centroid_xy=(
                    float(xs.mean() + slx.start + off_x),
                    float(ys.mean() + sly.start + off_y),
                ),
                mean_intensity=float(arr[sly, slx][comp].mean()),
            )
        )
    out.sort(key=lambda c: (-c.area_px, c.label))
    return out


def wiener_denoise(img: np.ndarray, kernel_size: int = 5, noise_power: float | None = None) -> np.ndarray:
    """Adaptive local-statistics Wiener DENOISER (scipy.signal.wiener) — this is
    NOT deconvolution; it never sharpens, it suppresses noise where the local
    variance is low. ``noise_power=None`` estimates the noise as the mean local
    variance. Non-finite outputs (zero-variance patches) fall back to the input."""
    arr = _as_gray(img)
    if int(kernel_size) < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    out = signal.wiener(arr, mysize=int(kernel_size), noise=noise_power)
    bad = ~np.isfinite(out)
    if bad.any():
        out = np.where(bad, arr, out)
    return out


def gaussian_psf(sigma: float, radius: int | None = None) -> np.ndarray:
    """Normalized 2-D Gaussian PSF; default radius ceil(4*sigma) (matches the
    ndimage truncate=4 default so blur/deblur kernels agree)."""
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    r = int(np.ceil(4.0 * float(sigma))) if radius is None else int(radius)
    if r < 1:
        raise ValueError(f"radius must be >= 1, got {r}")
    ax = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-(ax**2) / (2.0 * float(sigma) ** 2))
    kernel = np.outer(g, g)
    return kernel / kernel.sum()


def wiener_deconvolve(img: np.ndarray, psf_sigma: float, nsr: float = 0.01) -> np.ndarray:
    """Frequency-domain Wiener deconvolution assuming a Gaussian PSF:
    F_hat = conj(H) / (|H|^2 + nsr) * G. ``nsr`` is the assumed noise-to-signal
    power ratio (regularization; > 0). Output is real and unclipped."""
    arr = _as_gray(img)
    if nsr <= 0.0:
        raise ValueError(f"nsr must be > 0, got {nsr}")
    psf = gaussian_psf(psf_sigma)
    kh, kw = psf.shape
    if kh > arr.shape[0] or kw > arr.shape[1]:
        raise ValueError(f"PSF {kw}x{kh} larger than the {arr.shape[1]}x{arr.shape[0]} image")
    kernel = np.zeros(arr.shape, dtype=np.float64)
    kernel[:kh, :kw] = psf
    kernel = np.roll(kernel, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    transfer = fft.fft2(kernel)
    restored = np.conj(transfer) / (np.abs(transfer) ** 2 + float(nsr)) * fft.fft2(arr)
    return fft.ifft2(restored).real


def richardson_lucy(img: np.ndarray, psf_sigma: float, n_iter: int = 10) -> np.ndarray:
    """Richardson-Lucy deconvolution (fftconvolve-based) with a Gaussian PSF.

    Input is clipped to [0, 1] (RL needs non-negative data); the estimate starts
    flat at 0.5 and runs ``n_iter`` multiplicative updates. Deterministic."""
    arr = np.clip(_as_gray(img), 0.0, 1.0)
    if int(n_iter) < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    psf = gaussian_psf(psf_sigma)
    mirror = psf[::-1, ::-1]
    est = np.full_like(arr, 0.5)
    eps = 1e-12
    for _ in range(int(n_iter)):
        conv = signal.fftconvolve(est, psf, mode="same")
        ratio = arr / np.maximum(conv, eps)
        est = est * signal.fftconvolve(ratio, mirror, mode="same")
    return est


def laplacian_sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian — the standard scalar focus/sharpness proxy."""
    return float(np.var(ndimage.laplace(_as_gray(img))))

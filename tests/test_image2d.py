"""Golden tests for the 2-D image lane (image2d.py).

A synthetic 256x256 depth-like image carries a sinusoidal corrugation of known
frequency (12 cycles across the width) and orientation (30 degrees) on top of a
smooth background plane. After an 8-bit PNG round-trip, roi_spectrum_2d must
recover the frequency within one FFT bin and the orientation within 5 degrees;
SSIM and compare_images must behave classically.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from dsp_server.engine.image2d import (
    auto_roi,
    compare_images,
    crop_roi,
    load_image_gray,
    roi_spectrum_2d,
    ssim,
)

SIZE = 256
CYCLES = 12.0  # cycles across the image width
ORIENT_DEG = 30.0  # wavevector angle (0 = ripple varying along x)
FREQ_CPP = CYCLES / SIZE  # cycles/pixel
BIN_CPP = 1.0 / SIZE  # one FFT bin


def make_corrugated(size: int = SIZE, amp: float = 0.15) -> np.ndarray:
    """Depth-like image: smooth background plane + oriented sinusoidal ripple."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    theta = np.radians(ORIENT_DEG)
    phase = 2.0 * np.pi * FREQ_CPP * (xx * np.cos(theta) + yy * np.sin(theta))
    background = 0.15 * (xx / size) + 0.10 * (yy / size)
    return np.clip(0.45 + background + amp * np.sin(phase), 0.0, 1.0)


def save_png8(img: np.ndarray, path) -> None:
    Image.fromarray((np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)).save(path)


def wrapped_angle_delta(a_deg: float, b_deg: float) -> float:
    d = abs(a_deg - b_deg) % 180.0
    return min(d, 180.0 - d)


def test_roi_spectrum_recovers_frequency_and_orientation(tmp_path):
    png = tmp_path / "corrugated.png"
    save_png8(make_corrugated(), png)
    img = load_image_gray(png)
    assert img.shape == (SIZE, SIZE)
    assert img.dtype == np.float64
    assert img.min() >= 0.0 and img.max() <= 1.0

    spec = roi_spectrum_2d(img)
    assert abs(spec.dominant_freq_cpp - FREQ_CPP) <= BIN_CPP
    assert wrapped_angle_delta(spec.dominant_orientation_deg, ORIENT_DEG) <= 5.0
    assert spec.dominant_prominence_db > 6.0
    assert spec.psd.shape == (SIZE, SIZE)
    assert spec.freqs_x_cpp.shape == (SIZE,) and spec.freqs_y_cpp.shape == (SIZE,)

    # The folded profiles peak at the injected frequency / orientation too.
    r_peak = int(np.argmax(spec.radial_profile))
    r_bin_width = float(spec.radial_freqs_cpp[1] - spec.radial_freqs_cpp[0])
    assert abs(spec.radial_freqs_cpp[r_peak] - FREQ_CPP) <= r_bin_width
    t_peak = int(np.argmax(spec.orientation_profile))
    assert wrapped_angle_delta(float(spec.orientation_bins_deg[t_peak]), ORIENT_DEG) <= 5.0


def test_load_image_gray_16bit(tmp_path):
    img16 = (make_corrugated() * 65535.0).round().astype(np.uint16)
    path = tmp_path / "depth16.png"
    Image.fromarray(img16).save(path)
    loaded = load_image_gray(path)
    assert loaded.shape == (SIZE, SIZE)
    assert np.allclose(loaded, img16.astype(np.float64) / 65535.0, atol=1e-9)


def test_ssim_self_and_shuffled():
    img = make_corrugated()
    assert abs(ssim(img, img) - 1.0) <= 1e-9
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(img.ravel()).reshape(img.shape)
    assert ssim(img, shuffled) < 0.5
    with pytest.raises(ValueError):
        ssim(img, img[:-1, :])


def test_compare_images_identical(tmp_path):
    img = make_corrugated()
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    save_png8(img, path_a)
    save_png8(img, path_b)
    result = compare_images(path_a, path_b)
    assert result.mae == 0.0
    assert result.rms == 0.0
    assert abs(result.ssim - 1.0) <= 1e-9
    assert result.shape == (SIZE, SIZE)
    assert result.spectrum_a.dominant_freq_cpp == result.spectrum_b.dominant_freq_cpp
    assert result.spectrum_a.dominant_orientation_deg == result.spectrum_b.dominant_orientation_deg


def test_compare_images_shape_mismatch(tmp_path):
    path_a = tmp_path / "a.png"
    path_c = tmp_path / "small.png"
    save_png8(make_corrugated(), path_a)
    save_png8(make_corrugated(size=128), path_c)
    with pytest.raises(ValueError) as excinfo:
        compare_images(path_a, path_c)
    msg = str(excinfo.value)
    assert "(256, 256)" in msg and "(128, 128)" in msg


def test_crop_and_auto_roi():
    img = make_corrugated()
    # Out-of-bounds ROI clamps to the full image; an empty ROI raises.
    assert crop_roi(img, (-50, -50, 5000, 5000)).shape == img.shape
    with pytest.raises(ValueError):
        crop_roi(img, (10, 10, 10, 20))
    x0, y0, x1, y1 = auto_roi(img)
    assert 0 <= x0 < x1 <= SIZE and 0 <= y0 < y1 <= SIZE
    patch = crop_roi(img, (x0, y0, x1, y1))
    spec = roi_spectrum_2d(img, roi=(x0, y0, x1, y1))
    assert spec.psd.shape == patch.shape
    # A flat image has no gradient structure: auto_roi falls back to full frame.
    assert auto_roi(np.full((64, 64), 0.5)) == (0, 0, 64, 64)

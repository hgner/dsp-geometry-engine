"""Imaging v2 pack tests (ELE490): enhance/filter/segment/restore.

Tools are exercised through their module-level ``_impl`` functions (the geometry
pattern); the Butterworth attenuation golden is checked against the engine
function directly. Synthetic images come from tests/synth2d.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dsp_server.engine import image2d
from dsp_server.toolsets import AppContext, imaging
from synth2d import (
    make_low_contrast,
    make_noisy,
    make_oriented_ripple,
    make_rects,
    save_png8,
    save_png16,
)


@pytest.fixture
def ctx(tmp_path: Path) -> AppContext:
    data = tmp_path / "data"
    return AppContext(
        data_dir=data,
        dumps_dir=data / "dumps",
        cache_dir=data / "cache",
        plots_dir=data / "plots",
        logs_dir=data / "logs",
        engine_root=tmp_path / "engine",
    )


# --------------------------------------------------------------------------- #
# enhance_image


def test_enhance_histogram_eq_widens_spread(ctx, tmp_path):
    png = save_png8(make_low_contrast(), tmp_path / "lowc.png")
    out = json.loads(imaging._enhance_image(ctx, str(png), mode="histogram_eq"))
    assert "error" not in out
    before, after = out["stats_before"], out["stats_after"]
    spread_before = before["p99"] - before["p01"]
    spread_after = after["p99"] - after["p01"]
    assert spread_after > spread_before * 1.5  # narrow band -> full range
    assert Path(out["output_path"]).is_file()


def test_enhance_gamma_and_log_run(ctx, tmp_path):
    png = save_png8(make_low_contrast(), tmp_path / "lowc.png")
    for mode in ("gamma", "adaptive_eq", "log"):
        out = json.loads(imaging._enhance_image(ctx, str(png), mode=mode, gamma=0.5))
        assert "error" not in out, out
        assert Path(out["output_path"]).is_file()


def test_enhance_bad_mode_returns_toolerror(ctx, tmp_path):
    png = save_png8(make_low_contrast(), tmp_path / "lowc.png")
    out = json.loads(imaging._enhance_image(ctx, str(png), mode="nope"))
    assert "error" in out
    assert "histogram_eq" in (out.get("hint") or "")


# --------------------------------------------------------------------------- #
# filter_image + Butterworth attenuation golden (critique #11)


def test_butterworth_attenuation_ge_50x():
    """Gonzalez-Woods H = 1/(1+(D/D0)^(2n)); at D/D0 = 3, order 2 the gain is
    1/82 (~82x). A pure sinusoid at 3*D0 must lose >= 50x of its amplitude."""
    size = 128
    d0 = 8.0 / size  # cutoff cycles/pixel
    ripple = make_oriented_ripple(size=size, cycles=24.0, amp=0.4, base=0.5)  # freq = 3*D0
    out = image2d.butterworth_filter(ripple, cutoff_cpp=d0, order=2)
    amp_in = float(np.std(ripple - ripple.mean()))
    amp_out = float(np.std(out - out.mean()))
    assert amp_in > 0.0
    assert amp_out / amp_in <= 1.0 / 50.0


def test_filter_gaussian_and_sobel(ctx, tmp_path):
    png = save_png8(make_oriented_ripple(cycles=20.0), tmp_path / "rip.png")
    smoothed = json.loads(imaging._filter_image(ctx, str(png), mode="gaussian", sigma=3.0))
    assert "error" not in smoothed
    assert smoothed["rms_change"] > 0.0
    assert smoothed["ssim_vs_input"] < 1.0
    edges = json.loads(imaging._filter_image(ctx, str(png), mode="sobel"))
    assert "error" not in edges
    assert edges["edge_density"] is not None


# --------------------------------------------------------------------------- #
# segment_image + connected-component geometry golden


def test_segment_otsu_finds_two_rects(ctx, tmp_path):
    img, truth = make_rects()
    png = save_png16(img, tmp_path / "rects.png")
    out = json.loads(imaging._segment_image(ctx, str(png), mode="otsu", foreground="bright", morph="none"))
    assert "error" not in out, out
    assert out["n_components_total"] == 2
    comps = out["components"]
    assert len(comps) == 2
    for got, want in zip(comps, truth, strict=True):
        assert got["area_px"] == want["area_px"]
        assert tuple(got["bbox"]) == want["bbox"]
        assert got["mean_intensity"] > 0.8  # bright rects
    assert Path(out["mask_path"]).is_file()


def test_segment_roi_offset_reapplied(ctx, tmp_path):
    img, _ = make_rects()
    png = save_png16(img, tmp_path / "rects.png")
    roi = [60, 50, 110, 90]  # crops only the second rect (70..100, 60..80)
    out = json.loads(imaging._segment_image(ctx, str(png), mode="otsu", roi=roi, morph="none"))
    assert "error" not in out
    assert out["n_components_total"] == 1
    bbox = out["components"][0]["bbox"]
    assert bbox[0] >= roi[0] and bbox[1] >= roi[1]  # full-image coords, not roi-local
    assert (bbox[0], bbox[1], bbox[2], bbox[3]) == (70, 60, 100, 80)


def test_segment_kmeans_runs(ctx, tmp_path):
    img, _ = make_rects()
    png = save_png16(img, tmp_path / "rects.png")
    out = json.loads(imaging._segment_image(ctx, str(png), mode="kmeans", k=2, morph="none"))
    assert "error" not in out, out
    assert out["kmeans_centers"] is not None
    assert out["n_components_total"] == 2


# --------------------------------------------------------------------------- #
# restore_image


def test_restore_wiener_moves_toward_clean(ctx, tmp_path):
    clean, noisy = make_noisy(sigma=0.1, seed=1)
    png = save_png16(noisy, tmp_path / "noisy.png")
    out = json.loads(imaging._restore_image(ctx, str(png), mode="wiener", kernel_size=5))
    assert "error" not in out, out
    denoised = image2d.load_image_gray(out["output_path"])
    rms_noisy = float(np.sqrt(np.mean((noisy - clean) ** 2)))
    rms_denoised = float(np.sqrt(np.mean((denoised - clean) ** 2)))
    assert rms_denoised < rms_noisy  # denoiser pulled it back toward ground truth
    assert out["ssim_vs_input"] < 1.0


def test_restore_bad_mode_and_missing_file(ctx, tmp_path):
    png = save_png8(make_low_contrast(), tmp_path / "x.png")
    bad = json.loads(imaging._restore_image(ctx, str(png), mode="nope"))
    assert "error" in bad
    missing = json.loads(imaging._enhance_image(ctx, str(tmp_path / "nope.png"), mode="gamma"))
    assert "error" in missing


def test_16bit_round_trip_through_tool(ctx, tmp_path):
    img = make_oriented_ripple(cycles=10.0, amp=0.3)
    png = save_png16(img, tmp_path / "d16.png")
    out = json.loads(imaging._filter_image(ctx, str(png), mode="gaussian", sigma=0.5))
    assert "error" not in out
    assert out["height"] == img.shape[0] and out["width"] == img.shape[1]

"""compare_wavelet_signatures tests (ELE490 multiresolution lane).

The tool is exercised through its module-level ``_impl`` (the geometry pattern).
Goldens are physics-grounded, not empirically pinned:

* identical images -> every parity is exactly 1.0 (energy is trivially preserved);
* a Gaussian-blurred copy -> the finest detail band is annihilated while the coarse
  band survives, so ``micro_parity << macro_parity`` and the interpretation names
  smoothing;
* a shape mismatch -> ToolError naming both shapes.

The reference is a *multiscale* texture: a low-frequency tone (wavelength ~96 px)
plus broadband fine noise (which dominates the finest wavelet level and is what a
Gaussian blur destroys). A single-scale ripple would leave the coarse band empty, so
we build the texture explicitly here rather than reuse a one-frequency factory.

Wavelet detail level k spans wavelengths [2^k, 2^(k+1)] px, so at level 6 the
coarsest band is 64-128 px — where a sigma=2 Gaussian is essentially transparent, and
the 96 px tone lands squarely (macro parity ~ 0.98). The finest band (level 1, 2-4 px)
is carried by the noise, which the same Gaussian obliterates (micro parity ~ 0).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from dsp_server.toolsets import AppContext, imaging
from synth2d import save_png16

_SIZE = 512  # max db2 level is 7 here, so a level-6 decomposition has ample margin
_LEVELS = 6
_COARSE_WAVELENGTH = 96.0  # px -> f = 1/96 cpp, dead-center in the level-6 detail band


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


def _multiscale_texture(size: int = _SIZE, seed: int = 0) -> np.ndarray:
    """Low-frequency tone (coarse-scale energy) + fine broadband noise (finest-scale
    energy) — a texture whose finest wavelet band carries real energy for the blur to
    destroy while its coarsest band stays Gaussian-transparent."""
    rng = np.random.default_rng(seed)
    _yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    coarse = 0.25 * np.sin(2.0 * np.pi * xx / _COARSE_WAVELENGTH)  # coarse detail band
    fine = 0.08 * rng.standard_normal((size, size))  # broadband -> finest band
    return np.clip(0.5 + coarse + fine, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# (a) identical images -> perfect parity everywhere


def test_wavelet_identical_all_parity(ctx, tmp_path):
    png = save_png16(_multiscale_texture(), tmp_path / "ref.png")
    out = json.loads(imaging._compare_wavelet_signatures(ctx, str(png), str(png), levels=_LEVELS))
    assert "error" not in out, out
    assert out["n_levels"] == _LEVELS
    assert abs(out["approx_parity"] - 1.0) < 1e-6
    assert abs(out["macro_parity"] - 1.0) < 1e-6
    assert abs(out["micro_parity"] - 1.0) < 1e-6
    for parity in out["detail_parity_by_scale"]:
        assert abs(parity - 1.0) < 1e-6
    assert out["interpretation"] == "parity"


# --------------------------------------------------------------------------- #
# (b) Gaussian blur -> micro-texture loss / smoothing


def test_wavelet_gaussian_blur_is_micro_texture_loss(ctx, tmp_path):
    ref = _multiscale_texture()
    gen = ndimage.gaussian_filter(ref, sigma=2.0, mode="reflect")
    ref_png = save_png16(ref, tmp_path / "ref.png")
    gen_png = save_png16(gen, tmp_path / "gen.png")

    out = json.loads(imaging._compare_wavelet_signatures(ctx, str(ref_png), str(gen_png), levels=_LEVELS))
    assert "error" not in out, out
    # The blur destroys the finest scale but spares the coarse structure.
    assert out["micro_parity"] < 0.5
    assert out["macro_parity"] > 0.9
    assert out["micro_parity"] < out["macro_parity"]
    assert out["worst_scale"] == 1  # finest level lost the most energy
    assert "smoothing" in out["interpretation"]

    # detail_parity_by_scale is finest-first: it must line up with micro/macro.
    detail = out["detail_parity_by_scale"]
    assert len(detail) == _LEVELS
    assert detail[0] == out["micro_parity"]
    assert detail[-1] == out["macro_parity"]

    # save_plot writes a per-scale energy bar chart.
    out_plot = json.loads(
        imaging._compare_wavelet_signatures(ctx, str(ref_png), str(gen_png), levels=_LEVELS, save_plot=True)
    )
    assert "error" not in out_plot
    assert Path(out_plot["plot_path"]).is_file()


# --------------------------------------------------------------------------- #
# (c) shape mismatch -> ToolError naming both shapes


def test_wavelet_shape_mismatch_is_tool_error(ctx, tmp_path):
    big = save_png16(_multiscale_texture(size=256), tmp_path / "big.png")
    small = save_png16(_multiscale_texture(size=128), tmp_path / "small.png")
    out = json.loads(imaging._compare_wavelet_signatures(ctx, str(big), str(small)))
    assert "error" in out
    assert "256" in out["error"] and "128" in out["error"]

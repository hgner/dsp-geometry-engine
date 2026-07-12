"""Perceptual / semantic gate tests (the FR-VQA pack): CW-SSIM shift tolerance +
identity-coherence morph detection.

Tools are exercised through their module-level ``_impl`` functions. Synthetic frames
are written as PNGs into tmp_path. Deterministic (fixed-seed RNGs, analytic fields);
images are small (64-96 px) so each tool finishes well under a second.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from dsp_server.engine import image2d, perceptual
from dsp_server.toolsets import AppContext
from dsp_server.toolsets import perceptual as perceptual_pack
from synth2d import save_png8


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


def _save_rgb8(rgb: np.ndarray, path: Path | str) -> Path:
    path = Path(path)
    Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)).save(path)
    return path


def _tex(h: int = 96, w: int = 96, seed: int = 0, sigma: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = ndimage.gaussian_filter(rng.random((h, w)), sigma=sigma, mode="reflect")
    return (img - img.min()) / (img.max() - img.min())


# --------------------------------------------------------------------------- #
# Tool A: evaluate_perceptual_similarity


def test_perceptual_shift_is_equivalent_despite_low_ssim(ctx: AppContext, tmp_path: Path):
    ref = _tex()
    gen = ndimage.shift(ref, (0.0, 2.0), order=1, mode="reflect")  # 2 px grain shift
    ref_p = save_png8(ref, tmp_path / "ref.png")
    gen_p = save_png8(gen, tmp_path / "gen.png")
    rep = json.loads(perceptual_pack._evaluate_perceptual_similarity(ctx, str(ref_p), str(gen_p)))
    assert "error" not in rep, rep
    assert rep["verdict"] == "perceptually-equivalent"
    assert rep["cw_ssim"] > 0.85  # shift-tolerant: still looks identical
    assert rep["ssim"] < 0.6  # pixel SSIM collapses on the same shift
    assert rep["shift_tolerant_gap"] > 0.2  # cw_ssim >> ssim => a benign shift


def test_perceptual_identical_is_equivalent(ctx: AppContext, tmp_path: Path):
    ref = _tex()
    ref_p = save_png8(ref, tmp_path / "ref.png")
    gen_p = save_png8(ref, tmp_path / "gen.png")
    rep = json.loads(perceptual_pack._evaluate_perceptual_similarity(ctx, str(ref_p), str(gen_p)))
    assert "error" not in rep, rep
    assert rep["cw_ssim"] == pytest.approx(1.0, abs=1e-6)
    assert rep["verdict"] == "perceptually-equivalent"


def test_perceptual_unrelated_is_distinct(ctx: AppContext, tmp_path: Path):
    ref_p = save_png8(_tex(seed=0), tmp_path / "ref.png")
    gen_p = save_png8(_tex(seed=99), tmp_path / "gen.png")
    rep = json.loads(perceptual_pack._evaluate_perceptual_similarity(ctx, str(ref_p), str(gen_p)))
    assert "error" not in rep, rep
    assert rep["verdict"] == "distinct"
    assert rep["cw_ssim"] < 0.55


def test_perceptual_shape_mismatch_is_tool_error(ctx: AppContext, tmp_path: Path):
    ref_p = save_png8(_tex(h=96, w=96), tmp_path / "ref.png")
    gen_p = save_png8(_tex(h=64, w=64), tmp_path / "gen.png")
    rep = json.loads(perceptual_pack._evaluate_perceptual_similarity(ctx, str(ref_p), str(gen_p)))
    assert "error" in rep
    assert rep.get("hint")


def test_cw_ssim_beats_pixel_ssim_on_shift():
    ref = _tex()
    shifted = ndimage.shift(ref, (0.0, 3.0), order=1, mode="reflect")
    cw = perceptual.cw_ssim(ref, shifted)
    ss = image2d.ssim(ref, shifted)
    assert cw > 0.8 > ss  # the whole point: shift-tolerant vs shift-fragile


def test_perceptual_flat_frames_are_equivalent(ctx: AppContext, tmp_path: Path):
    # Two featureless frames differing only in brightness (a smooth sky/wall with a DC
    # offset): no structure to differ. The mean-subtraction fix keeps a zero-DC Gabor
    # bank's edge ring from faking a "degraded" verdict here.
    ref_p = save_png8(np.full((96, 96), 0.5), tmp_path / "ref.png")
    gen_p = save_png8(np.full((96, 96), 0.2), tmp_path / "gen.png")
    rep = json.loads(perceptual_pack._evaluate_perceptual_similarity(ctx, str(ref_p), str(gen_p)))
    assert "error" not in rep, rep
    assert rep["verdict"] == "perceptually-equivalent"
    assert rep["cw_ssim"] > 0.9


# --------------------------------------------------------------------------- #
# Tool B: verify_identity_coherence


def _obj_frame(color: tuple, tex_amp: float, tex_seed: int, h: int = 64, w: int = 64) -> np.ndarray:
    """An RGB frame whose object colour and texture contrast are controllable."""
    rng = np.random.default_rng(tex_seed)
    t = ndimage.gaussian_filter(rng.random((h, w)), sigma=0.8, mode="reflect")
    t = (t - t.min()) / (t.max() - t.min() + 1e-9)
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    for c in range(3):
        rgb[:, :, c] = np.clip(color[c] * (1.0 - tex_amp + tex_amp * t), 0.0, 1.0)
    return rgb


def _save_frames_rgb(frames: list[np.ndarray], d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        _save_rgb8(f, d / f"frame_{i:02d}.png")
    return d


def _central_mask(size: int = 64) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.float64)
    mask[20:44, 20:44] = 1.0
    return mask


def test_identity_coherent_object_holds(ctx: AppContext, tmp_path: Path):
    frames = [_obj_frame((0.8, 0.2, 0.2), 0.5, tex_seed=10 + i) for i in range(10)]
    d = _save_frames_rgb(frames, tmp_path / "coh")
    mask_p = save_png8(_central_mask(), tmp_path / "mask.png")
    rep = json.loads(perceptual_pack._verify_identity_coherence(ctx, str(d), str(mask_p)))
    assert "error" not in rep, rep
    assert rep["verdict"] == "coherent"
    assert rep["first_break_frame"] == -1
    assert rep["mean_coherence"] > 0.9
    # coherence array lives on disk, never inline
    npz = Path(rep["coherence_path"])
    assert npz.parent == ctx.data_dir / "perceptual"
    with np.load(npz, allow_pickle=False) as z:
        assert z["coherence_by_frame"].shape == (10,)
        assert z["coherence_by_frame"][0] == pytest.approx(1.0, abs=1e-9)


def test_identity_morph_localized_to_frame(ctx: AppContext, tmp_path: Path):
    # Frames 0-4: red, textured leather. Frames 5-9: blue, flat plastic. The masked
    # object's colour AND texture flip at frame 5 -> a morph the tool must localize.
    frames = [
        _obj_frame((0.8, 0.2, 0.2), 0.5, tex_seed=10 + i)
        if i < 5
        else _obj_frame((0.2, 0.3, 0.9), 0.1, tex_seed=100 + i)
        for i in range(10)
    ]
    d = _save_frames_rgb(frames, tmp_path / "morph")
    mask_p = save_png8(_central_mask(), tmp_path / "mask.png")
    rep = json.loads(perceptual_pack._verify_identity_coherence(ctx, str(d), str(mask_p)))
    assert "error" not in rep, rep
    assert rep["verdict"] in {"morph", "identity-drift"}
    assert rep["first_break_frame"] == 5  # the exact frame the identity broke
    assert rep["min_coherence"] < 0.8
    assert rep["min_coherence_frame"] >= 5


def test_identity_mask_mismatch_is_tool_error(ctx: AppContext, tmp_path: Path):
    frames = [_obj_frame((0.8, 0.2, 0.2), 0.5, tex_seed=10 + i) for i in range(4)]
    d = _save_frames_rgb(frames, tmp_path / "clip")
    mask_p = save_png8(np.ones((32, 32), dtype=np.float64), tmp_path / "mask.png")  # wrong size
    rep = json.loads(perceptual_pack._verify_identity_coherence(ctx, str(d), str(mask_p)))
    assert "error" in rep
    assert rep.get("hint")


def test_identity_low_value_id_mask_selects_object(ctx: AppContext, tmp_path: Path):
    # An ID-map / binary mask whose object label is a SMALL integer (3), not 255 — exactly
    # what the engine emits. "nonzero = object" (mask_threshold default 0) must still select
    # it, not normalize it below a 0.5 cutoff and report a false "0 px" error.
    frames = [_obj_frame((0.8, 0.2, 0.2), 0.5, tex_seed=10 + i) for i in range(6)]
    d = _save_frames_rgb(frames, tmp_path / "clip")
    m = np.zeros((64, 64), dtype=np.uint8)
    m[20:44, 20:44] = 3  # object label 3
    mask_p = tmp_path / "idmask.png"
    Image.fromarray(m, mode="L").save(mask_p)
    rep = json.loads(perceptual_pack._verify_identity_coherence(ctx, str(d), str(mask_p)))
    assert "error" not in rep, rep
    assert rep["mask_pixel_count"] == 24 * 24
    assert rep["verdict"] == "coherent"

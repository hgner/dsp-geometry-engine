"""End-to-end integration of the AI-video COMPARISON GATE.

The per-tool golden tests prove each gate in isolation. This proves the four build
rounds COMPOSE: every gate tool is driven end-to-end through its module-level ``_impl``
on realistic on-disk assets, each passes its clean input and flags its defect, and a
single combiner aggregates the per-axis verdicts into one clip pass/fail — the turnkey
flow a client (proje8) runs over ``DSP_TOOLSETS=video,imaging,geometry,perceptual``.

Design note (from the adversarial design critique): a SINGLE shared clip cannot satisfy
every gate's clean-case at once — the temporal gate needs spatially-smooth content (so a
pan doesn't read as flicker) while the camera/perceptual gates need broadband texture
with trackable corners; those requirements conflict. So each axis uses its own
purpose-built scenario, reusing the exact golden-test factories (imported here, not
re-tuned) so the e2e stays faithful to the verified per-tool behavior.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from dsp_server.toolsets import AppContext, imaging, video
from dsp_server.toolsets import perceptual as perceptual_pack
from synth2d import save_png8

# Reuse the PROVEN golden-test factories (tests/ is on sys.path) — zero re-tuning.
from test_geometry_gate import _LIGHT, _curl_warp, _fg_square, _lit, _normal_field
from test_geometry_gate import _save_rgb8 as save_rgb
from test_geometry_gate import _texture as gg_texture
from test_perceptual import _central_mask, _obj_frame, _save_frames_rgb
from test_perceptual import _tex as perc_tex
from test_video import _flicker_frames, _melting_frames, _pan_frames, _rigid_frames
from test_video import _save_frames as save_gray_frames


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
# One runner per gate axis: build (clean, defect) scenarios, drive the tool's
# _impl, and return (clean_passed, defect_flagged). Each is isolated so one axis's
# defect can never break another axis's assertion.


def _axis_temporal(ctx: AppContext, d: Path) -> tuple[bool, bool]:
    """evaluate_spatiotemporal_frequencies — temporal flicker/boil (video)."""
    clean = save_gray_frames(_pan_frames(), d / "clean")  # smooth pan -> low temporal HF
    defect = save_gray_frames(_flicker_frames(), d / "defect")  # every-other-frame brightness pump
    c = json.loads(video._evaluate_spatiotemporal_frequencies(ctx, str(clean)))
    f = json.loads(video._evaluate_spatiotemporal_frequencies(ctx, str(defect)))
    return c["flicker_verdict"] == "stable", f["flicker_verdict"] == "flicker"


def _axis_motion(ctx: AppContext, d: Path) -> tuple[bool, bool]:
    """verify_motion_consistency — melting / hallucinated motion (video)."""
    clean = save_gray_frames(_rigid_frames(), d / "clean")  # clean recoverable translation
    defect = save_gray_frames(_melting_frames(), d / "defect")  # per-frame fresh noise region
    c = json.loads(video._verify_motion_consistency(ctx, str(clean), tau=2.0))
    f = json.loads(video._verify_motion_consistency(ctx, str(defect), tau=0.5))
    clean_ok = c["inconsistent_fraction"] < 0.2
    defect_flag = f["max_fb_residual_px"] > 0.5 and f["inconsistent_fraction"] > 0.0
    return clean_ok, defect_flag


def _axis_camera(ctx: AppContext, d: Path) -> tuple[bool, bool]:
    """verify_camera_projection — camera drift / warp (video)."""
    d.mkdir(parents=True, exist_ok=True)
    ref = gg_texture()  # band-limited noise: trackable corners
    ref_p = save_png8(ref, d / "ref.png")
    clean_p = save_png8(ref, d / "clean.png")  # identical camera
    defect_p = save_png8(_curl_warp(ref), d / "defect.png")  # per-quadrant curl: no global camera
    c = json.loads(video._verify_camera_projection(ctx, str(ref_p), str(clean_p)))
    f = json.loads(video._verify_camera_projection(ctx, str(ref_p), str(defect_p)))
    return c["verdict"] == "camera-consistent", f["verdict"] in {"camera-drift", "geometry-inconsistent"}


def _axis_lighting(ctx: AppContext, d: Path) -> tuple[bool, bool]:
    """analyze_photometric_consistency — shading vs the true normals (video)."""
    d.mkdir(parents=True, exist_ok=True)
    normals = _normal_field()
    nrm_p = save_rgb((normals + 1.0) / 2.0, d / "normals.png")
    clean_p = save_png8(_lit(normals, _LIGHT), d / "clean.png")  # Lambertian from the true normals
    hallucinated = np.random.default_rng(0).random(normals.shape[:2])  # shading uncorrelated with normals
    defect_p = save_png8(hallucinated, d / "defect.png")
    c = json.loads(video._analyze_photometric_consistency(ctx, str(clean_p), str(nrm_p)))
    f = json.loads(video._analyze_photometric_consistency(ctx, str(defect_p), str(nrm_p)))
    return c["verdict"] == "consistent", f["verdict"] == "inconsistent"


def _axis_occlusion(ctx: AppContext, d: Path) -> tuple[bool, bool]:
    """evaluate_occlusion_boundaries — depth-edge sharpness (video)."""
    d.mkdir(parents=True, exist_ok=True)
    depth, sharp = _fg_square()
    depth_p = save_png8(depth, d / "depth.png")
    ref_p = save_png8(sharp, d / "ref.png")
    blur = ndimage.gaussian_filter(sharp, sigma=2.0, mode="reflect")
    defect_p = save_png8(blur, d / "defect.png")
    occ = video._evaluate_occlusion_boundaries
    c = json.loads(occ(ctx, str(ref_p), str(depth_p), reference=str(ref_p)))
    f = json.loads(occ(ctx, str(defect_p), str(depth_p), reference=str(ref_p)))
    return c["verdict"] == "sharp", f["verdict"] == "soft-boundaries"


def _axis_spatial(ctx: AppContext, d: Path) -> tuple[bool, bool]:
    """compare_wavelet_signatures — micro-texture loss (imaging). 256px so the coarse
    band survives a blur that erases the fine band (per the design critique)."""
    d.mkdir(parents=True, exist_ok=True)
    ref = perc_tex(h=256, w=256, seed=1, sigma=1.0)  # broadband
    ref_p = save_png8(ref, d / "ref.png")
    clean_p = save_png8(ref, d / "clean.png")
    defect_p = save_png8(ndimage.gaussian_filter(ref, sigma=1.6, mode="reflect"), d / "defect.png")
    c = json.loads(imaging._compare_wavelet_signatures(ctx, str(ref_p), str(clean_p), levels=5))
    f = json.loads(imaging._compare_wavelet_signatures(ctx, str(ref_p), str(defect_p), levels=5))
    clean_ok = c["interpretation"] == "parity"
    defect_flag = f["micro_parity"] < f["macro_parity"] and f["micro_parity"] < 0.7
    return clean_ok, defect_flag


def _axis_perceptual(ctx: AppContext, d: Path) -> tuple[bool, bool]:
    """evaluate_perceptual_similarity — shift-tolerant perceptual parity (perceptual)."""
    d.mkdir(parents=True, exist_ok=True)
    ref = perc_tex(seed=0)  # broadband: a small shift stays perceptually equivalent
    ref_p = save_png8(ref, d / "ref.png")
    shifted = ndimage.shift(ref, (0.0, 2.0), order=1, mode="reflect")  # benign 2 px grain shift
    clean_p = save_png8(shifted, d / "clean.png")
    defect_p = save_png8(perc_tex(seed=99), d / "defect.png")  # decorrelated content
    c = json.loads(perceptual_pack._evaluate_perceptual_similarity(ctx, str(ref_p), str(clean_p)))
    f = json.loads(perceptual_pack._evaluate_perceptual_similarity(ctx, str(ref_p), str(defect_p)))
    return c["verdict"] == "perceptually-equivalent", f["verdict"] == "distinct"


def _axis_identity(ctx: AppContext, d: Path) -> tuple[bool, bool]:
    """verify_identity_coherence — object morph across the clip (perceptual)."""
    coherent = _save_frames_rgb(
        [_obj_frame((0.8, 0.2, 0.2), 0.5, tex_seed=10 + i) for i in range(10)], d / "clean"
    )
    morph = _save_frames_rgb(
        [
            _obj_frame((0.8, 0.2, 0.2), 0.5, tex_seed=10 + i)
            if i < 5
            else _obj_frame((0.2, 0.3, 0.9), 0.1, tex_seed=100 + i)
            for i in range(10)
        ],
        d / "defect",
    )
    mask_p = save_png8(_central_mask(), d / "mask.png")
    c = json.loads(perceptual_pack._verify_identity_coherence(ctx, str(coherent), str(mask_p)))
    f = json.loads(perceptual_pack._verify_identity_coherence(ctx, str(morph), str(mask_p)))
    clean_ok = c["verdict"] == "coherent"
    defect_flag = f["first_break_frame"] == 5 and f["verdict"] in {"morph", "identity-drift"}
    return clean_ok, defect_flag


AXES: list[tuple[str, Callable[[AppContext, Path], tuple[bool, bool]]]] = [
    ("temporal", _axis_temporal),
    ("motion", _axis_motion),
    ("camera", _axis_camera),
    ("lighting", _axis_lighting),
    ("occlusion", _axis_occlusion),
    ("spatial", _axis_spatial),
    ("perceptual", _axis_perceptual),
    ("identity", _axis_identity),
]


def _overall_pass(axis_pass: dict[str, bool]) -> bool:
    """The clip PASSES the comparison gate only if every axis passes (any flagged axis
    fails the clip). This is the combiner a client applies over the per-tool verdicts."""
    return all(axis_pass.values())


@pytest.mark.parametrize("name,runner", AXES, ids=[n for n, _ in AXES])
def test_gate_axis_clean_passes_and_defect_flags(
    ctx: AppContext, tmp_path: Path, name: str, runner: Callable[[AppContext, Path], tuple[bool, bool]]
):
    clean_ok, defect_flag = runner(ctx, tmp_path)
    assert clean_ok, f"{name}: a CLEAN generated input was false-flagged by the gate"
    assert defect_flag, f"{name}: the gate MISSED a {name} defect"


def test_full_comparison_gate_composes(ctx: AppContext, tmp_path: Path):
    """Every gate runs end-to-end and the per-axis verdicts combine into one clip verdict:
    a clean bundle -> PASS; a defect bundle -> FAIL with every axis flagged."""
    clean_pass: dict[str, bool] = {}
    defect_flag: dict[str, bool] = {}
    for name, runner in AXES:
        c_ok, f_flag = runner(ctx, tmp_path / name)
        clean_pass[name] = c_ok
        defect_flag[name] = f_flag

    # Each gate is callable end-to-end and correct on its golden scenario.
    false_flagged = [n for n, v in clean_pass.items() if not v]
    missed = [n for n, v in defect_flag.items() if not v]
    assert not false_flagged, f"clean inputs false-flagged: {false_flagged}"
    assert not missed, f"defects missed: {missed}"

    # Composition: the clean bundle passes the whole gate; the defect bundle fails it,
    # and every axis is a failing axis (defect_flag True -> that axis's pass is False).
    assert _overall_pass(clean_pass) is True
    defect_axis_pass = {n: not flagged for n, flagged in defect_flag.items()}
    assert _overall_pass(defect_axis_pass) is False
    assert [n for n, ok in defect_axis_pass.items() if not ok] == [n for n, _ in AXES]

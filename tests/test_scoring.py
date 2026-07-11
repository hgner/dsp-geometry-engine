"""Tests for the bake-scoring metric (engine/scoring.py + the score_bake tool).

Synthetic silhouettes only — no engine, no GPU. The load-bearing property is
DISCRIMINATION: a reference that matches the bake's shape must out-score a
mismatched one, and the pure metrics must satisfy their invariances (IoU=1 on
identical masks, Hu distance ~0 on identical shapes, scale/translation
invariance of the normalized silhouette)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dsp_server.engine import scoring
from dsp_server.toolsets import AppContext
from dsp_server.toolsets.geometry import _score_bake
from synth2d import save_png8


def _ellipse(size: int, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    return (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0


def _depth_from_mask(mask: np.ndarray) -> np.ndarray:
    """Bright body on a dark field (the reference-depth convention)."""
    return np.where(mask, 0.85, 0.05)


def _ctx(tmp_path: Path) -> AppContext:
    return AppContext(
        data_dir=tmp_path,
        dumps_dir=tmp_path,
        cache_dir=tmp_path,
        plots_dir=tmp_path,
        logs_dir=tmp_path,
        engine_root=tmp_path,
    )


# --------------------------------------------------------------------------- #
# Pure metrics


def test_iou_identical_and_disjoint():
    a = _ellipse(64, 32, 32, 20, 12)
    assert scoring.iou(a, a) == pytest.approx(1.0)
    b = np.zeros_like(a)
    b[:5, :5] = True
    assert scoring.iou(a, b) == 0.0


def test_hu_distance_zero_on_identical_shape():
    a = _ellipse(96, 48, 48, 30, 16)
    d = scoring.match_shapes_i2(scoring.hu_moments(a), scoring.hu_moments(a))
    assert d == pytest.approx(0.0, abs=1e-9)


def test_normalize_silhouette_scale_translation_invariant():
    small = _ellipse(96, 30, 30, 16, 10)
    big = _ellipse(240, 150, 100, 48, 30)  # same 8:5 aspect, ~3x scale + moved
    ns, nb = scoring.normalize_silhouette(small), scoring.normalize_silhouette(big)
    assert scoring.iou(ns, nb) > 0.95


def test_silhouette_from_depth_recovers_body():
    mask = _ellipse(80, 40, 40, 24, 14)
    sil = scoring.silhouette_from_depth(_depth_from_mask(mask))
    assert scoring.iou(sil, mask) > 0.98


# --------------------------------------------------------------------------- #
# score_bake discrimination


def test_score_bake_discriminates(tmp_path: Path):
    bake_mask = _ellipse(128, 64, 64, 40, 22)
    match_ref = _ellipse(128, 70, 60, 44, 24)  # near-identical shape
    other_ref = _ellipse(128, 64, 64, 12, 12)  # small circle — very different

    depth = save_png8(_depth_from_mask(bake_mask), tmp_path / "bake_depth.png")
    mask = save_png8(bake_mask.astype(float), tmp_path / "bake_mask.png")
    refdir = tmp_path / "refs"
    refdir.mkdir()
    save_png8(_depth_from_mask(match_ref), refdir / "match_depth.png")
    save_png8(_depth_from_mask(other_ref), refdir / "other_depth.png")

    res = scoring.score_bake(depth, mask, str(refdir / "*_depth.png"))
    assert res.n_refs == 2
    # the matching ellipse must be the best IoU, well above the small circle
    assert res.top_matches[0].ref == "match_depth.png"
    assert res.iou_best > 0.7
    assert res.iou_best > res.iou_mean  # the mismatch drags the mean down


def test_score_bake_verdict_and_corrugation_band(tmp_path: Path):
    bake_mask = _ellipse(128, 64, 64, 40, 22)
    depth = save_png8(_depth_from_mask(bake_mask), tmp_path / "d.png")
    mask = save_png8(bake_mask.astype(float), tmp_path / "m.png")
    refdir = tmp_path / "r"
    refdir.mkdir()
    save_png8(_depth_from_mask(_ellipse(128, 64, 64, 40, 22)), refdir / "a_depth.png")

    # a perfect self-match clears the pose gate at a lenient target
    res = scoring.score_bake(depth, mask, str(refdir / "*_depth.png"), iou_target=0.6, hu_target=5.0)
    assert res.pose_ok
    assert res.verdict.startswith("pass")
    # a smooth synthetic depth has no ripple -> the advisory corrugation stays finite/low
    assert res.corrugation_db >= 0.0


# --------------------------------------------------------------------------- #
# Tool layer


def test_tool_score_bake_ok_and_toolerror(tmp_path: Path):
    frame = 8
    bake = tmp_path / "bake"
    (bake / "depth").mkdir(parents=True)
    (bake / "mask").mkdir(parents=True)
    body = _ellipse(96, 48, 48, 30, 18)
    save_png8(_depth_from_mask(body), bake / "depth" / f"frame_{frame:05d}.png")
    save_png8(body.astype(float), bake / "mask" / f"frame_{frame:05d}_p1.png")
    refdir = tmp_path / "refs"
    refdir.mkdir()
    save_png8(_depth_from_mask(_ellipse(96, 50, 46, 32, 19)), refdir / "r_depth.png")

    out = json.loads(
        _score_bake(_ctx(tmp_path), str(bake), str(refdir / "*_depth.png"), frame=frame, person="p1")
    )
    assert "error" not in out
    assert out["n_refs"] == 1
    assert out["verdict"].startswith(("pass", "pose-weak"))

    bad = json.loads(_score_bake(_ctx(tmp_path), str(tmp_path / "nope"), str(refdir / "*_depth.png")))
    assert "error" in bad

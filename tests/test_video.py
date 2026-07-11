"""Video pack tests (temporal comparison gate): spatio-temporal FFT + optical-flow
forward-backward consistency.

Tools are exercised through their module-level ``_impl`` functions (the geometry
pattern). Synthetic frame stacks are written as 8-bit PNGs (synth2d.save_png8) into
tmp_path directories, then fed to the tools by directory path.

Determinism: the flicker/pan stacks are analytic; the "melting" stack's noise region
uses a fixed-seed RNG. Frames are small (48-64 px, <=16 frames) so both tools finish
well under a second.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from dsp_server.engine import optflow
from dsp_server.toolsets import AppContext, video
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


# --------------------------------------------------------------------------- #
# Synthetic frame-stack factories


def _save_frames(frames: list[np.ndarray], d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        save_png8(f, d / f"frame_{i:02d}.png")
    return d


def _pan_frames(
    n: int = 16, w: int = 64, amp: float = 0.1, base: float = 0.4, cycles: int = 2
) -> list[np.ndarray]:
    """A seamless horizontal sinusoid gradient panned 1 px/frame (period divides w, so
    np.roll wraps without a seam) — each pixel sees a pure LOW temporal frequency."""
    x = np.arange(w)
    row = base + amp * np.sin(2.0 * np.pi * cycles * x / w)
    canvas = np.tile(row, (w, 1))
    return [np.roll(canvas, i, axis=1) for i in range(n)]


def _flicker_frames(n: int = 16, w: int = 64) -> list[np.ndarray]:
    """The pan, but every other frame brightened +0.3 -> a Nyquist (f_t=0.5) flicker."""
    base = _pan_frames(n, w)
    return [np.clip(f + (0.3 if i % 2 == 0 else 0.0), 0.0, 1.0) for i, f in enumerate(base)]


def _texture(
    h: int, w: int, dx: float = 0.0, dy: float = 0.0, freq: float = 6.0, amp: float = 0.2
) -> np.ndarray:
    """A 2-D texture with gradients in BOTH axes (no aperture problem), shifted by (dx, dy)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    sx = amp * np.sin(2.0 * np.pi * freq * (xx - dx) / w)
    sy = amp * np.sin(2.0 * np.pi * freq * (yy - dy) / h)
    return 0.5 + sx + sy


def _rigid_frames(n: int = 8, h: int = 64, w: int = 64, shift: float = 2.0) -> list[np.ndarray]:
    """The 2-D texture translated +shift px/frame in x — a clean recoverable motion."""
    return [_texture(h, w, dx=shift * i) for i in range(n)]


def _melting_frames(
    n: int = 8,
    h: int = 64,
    w: int = 64,
    shift: float = 2.0,
    region: tuple[int, int] = (16, 48),
    seed: int = 0,
) -> list[np.ndarray]:
    """Rigid translation everywhere EXCEPT a central square replaced by fresh per-frame
    blobby noise (no consistent motion -> forward/backward flow cannot cancel there)."""
    rng = np.random.default_rng(seed)
    r0, r1 = region
    size = r1 - r0
    frames = []
    for i in range(n):
        fr = _texture(h, w, dx=shift * i)
        blob = ndimage.zoom(rng.random((4, 4)), size / 4.0, order=1)
        blob = (blob - blob.min()) / (blob.max() - blob.min() + 1e-9)
        fr[r0:r1, r0:r1] = blob
        frames.append(fr)
    return frames


# --------------------------------------------------------------------------- #
# Tool A: evaluate_spatiotemporal_frequencies


def test_spatiotemporal_pan_is_stable(ctx: AppContext, tmp_path: Path):
    d = _save_frames(_pan_frames(), tmp_path / "pan")
    rep = json.loads(video._evaluate_spatiotemporal_frequencies(ctx, str(d)))
    assert "error" not in rep, rep
    assert rep["n_frames"] == 16
    assert rep["temporal_hf_energy_fraction"] < 0.2  # slow pan -> little fast temporal energy
    assert rep["flicker_verdict"] == "stable"
    assert rep["reference"] is None
    # spectrum npz lives on disk under data/video/, never inline.
    npz = Path(rep["spectrum_path"])
    assert npz.parent == ctx.data_dir / "video"
    with np.load(npz, allow_pickle=False) as z:
        assert {"freqs_cpf", "power"} <= set(z.files)
        assert float(z["freqs_cpf"].max()) == pytest.approx(0.5, abs=1e-9)


def test_spatiotemporal_flicker_detected(ctx: AppContext, tmp_path: Path):
    d = _save_frames(_flicker_frames(), tmp_path / "flick")
    rep = json.loads(video._evaluate_spatiotemporal_frequencies(ctx, str(d), save_plot=True))
    assert "error" not in rep, rep
    assert rep["temporal_hf_energy_fraction"] > 0.5  # alternating-frame flicker dominates
    assert rep["dominant_temporal_freq_cpf"] == pytest.approx(0.5, abs=0.05)  # Nyquist
    assert rep["flicker_verdict"] == "flicker"
    assert Path(rep["plot_path"]).is_file()


def test_spatiotemporal_two_stack_flicker_vs_ref(ctx: AppContext, tmp_path: Path):
    gen = _save_frames(_flicker_frames(), tmp_path / "gen")
    ref = _save_frames(_pan_frames(), tmp_path / "ref")
    rep = json.loads(video._evaluate_spatiotemporal_frequencies(ctx, str(gen), reference=str(ref)))
    assert "error" not in rep, rep
    assert rep["flicker_verdict"] != "stable"
    assert rep["flicker_verdict"] == "flicker-vs-ref"
    assert rep["delta_temporal_hf_energy_fraction"] > 0.0
    assert rep["ref_temporal_hf_energy_fraction"] < rep["temporal_hf_energy_fraction"]
    with np.load(rep["spectrum_path"], allow_pickle=False) as z:
        assert {"freqs_cpf", "power", "ref_freqs_cpf", "ref_power"} <= set(z.files)


def test_spatiotemporal_too_few_frames_is_tool_error(ctx: AppContext, tmp_path: Path):
    d = tmp_path / "single"
    d.mkdir()
    save_png8(_texture(48, 48), d / "frame_00.png")
    rep = json.loads(video._evaluate_spatiotemporal_frequencies(ctx, str(d)))
    assert "error" in rep
    assert rep.get("hint")


# --------------------------------------------------------------------------- #
# Tool B: verify_motion_consistency


def test_lucas_kanade_recovers_translation():
    frames = _rigid_frames(n=2, shift=2.0)
    points = np.array([[20.0, 20.0], [30.0, 30.0], [40.0, 40.0], [24.0, 36.0], [36.0, 24.0]])
    flow = optflow.lucas_kanade(frames[0], frames[1], points, window=15)
    assert flow.shape == (5, 2)
    assert float(np.median(flow[:, 0])) == pytest.approx(2.0, abs=0.6)  # +2 px in x
    assert float(np.median(flow[:, 1])) == pytest.approx(0.0, abs=0.6)  # ~0 in y


def test_motion_consistency_rigid_is_clean(ctx: AppContext, tmp_path: Path):
    d = _save_frames(_rigid_frames(), tmp_path / "rigid")
    rep = json.loads(video._verify_motion_consistency(ctx, str(d), tau=2.0, save_plot=True))
    assert "error" not in rep, rep
    assert rep["n_pairs"] == 7
    assert rep["n_grid_points"] == 16
    assert rep["mean_fb_residual_px"] < 0.5  # translation round-trips cleanly
    assert rep["inconsistent_fraction"] < 0.2
    assert 0 <= rep["worst_frame_index"] < rep["n_pairs"]
    assert Path(rep["plot_path"]).is_file()


def test_motion_consistency_melting_flagged(ctx: AppContext, tmp_path: Path):
    melt = _save_frames(_melting_frames(), tmp_path / "melt")
    rigid = _save_frames(_rigid_frames(), tmp_path / "rigid_ctrl")

    rep = json.loads(video._verify_motion_consistency(ctx, str(melt), tau=0.5))
    ctrl = json.loads(video._verify_motion_consistency(ctx, str(rigid), tau=0.5))
    assert "error" not in rep, rep

    # The melting region breaks forward-backward consistency; the clean control does not.
    assert rep["inconsistent_fraction"] > 0.0
    assert rep["inconsistent_fraction"] > ctrl["inconsistent_fraction"]
    assert rep["mean_fb_residual_px"] > ctrl["mean_fb_residual_px"]
    assert rep["max_fb_residual_px"] > 0.5
    assert rep["n_invalid_total"] >= 1
    assert len(rep["invalid_points"]) >= 1
    assert 0 <= rep["worst_frame_index"] < rep["n_pairs"]
    assert rep["worst_frame_inconsistent_fraction"] > 0.0

    # The residual map localizes the defect: the central (noise) grid cells carry far
    # more residual than the surrounding (clean-translation) cells.
    with np.load(rep["residual_map_path"], allow_pickle=False) as z:
        residual_grid = z["residual_grid"].copy()  # (n_pairs, 4, 4)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True  # grid cells whose windows fall inside the [16:48] noise square
    in_region = residual_grid[:, mask]
    out_region = residual_grid[:, ~mask]
    assert float(in_region.mean()) > float(out_region.mean())
    assert float(in_region.mean()) > 0.2


def test_motion_consistency_missing_source_is_tool_error(ctx: AppContext, tmp_path: Path):
    rep = json.loads(video._verify_motion_consistency(ctx, str(tmp_path / "nope")))
    assert "error" in rep
    assert rep.get("hint")

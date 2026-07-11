"""Geometric / photometric comparison-gate tests (the per-frame half of the video
pack): camera-projection, photometric consistency, occlusion boundaries.

Tools are exercised through their module-level ``_impl`` functions (the geometry
pattern). Synthetic frames/passes are written as PNGs into tmp_path directories and
fed to the tools by path. Everything is deterministic (fixed-seed RNGs, analytic
fields); images are small (80-160 px) so each tool finishes well under a second.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from dsp_server.engine import epipolar, occlusion, photometric
from dsp_server.toolsets import AppContext, video
from synth2d import save_png8, save_png16


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


# --------------------------------------------------------------------------- #
# Camera-gate factories: a corner-rich band-limited-noise texture + rigid / curl warps


def _texture(h: int = 160, w: int = 160, seed: int = 0, sigma: float = 1.0) -> np.ndarray:
    """Band-limited noise — strong 2-D gradients everywhere, so Lucas-Kanade tracks
    reliably and Shi-Tomasi finds tens of well-localized, uniquely matchable corners
    (a global shift is then unambiguous, unlike a periodic pattern)."""
    rng = np.random.default_rng(seed)
    img = ndimage.gaussian_filter(rng.random((h, w)), sigma=sigma, mode="reflect")
    lo, hi = float(img.min()), float(img.max())
    return (img - lo) / (hi - lo)


def _curl_warp(img: np.ndarray, s: float = 4.0) -> np.ndarray:
    """A per-quadrant rotational displacement (TL:+x, TR:+y, BR:-x, BL:-y). Each
    quadrant is a clean rigid shift (so correspondences survive the flow gate), but
    the four directions form a curl that NO single homography can fit — the
    canonical 'geometry-inconsistent' case."""
    h, w = img.shape
    hy, hx = h // 2, w // 2
    tl = ndimage.shift(img, (0.0, s), order=1, mode="nearest")
    tr = ndimage.shift(img, (s, 0.0), order=1, mode="nearest")
    br = ndimage.shift(img, (0.0, -s), order=1, mode="nearest")
    bl = ndimage.shift(img, (-s, 0.0), order=1, mode="nearest")
    g = img.copy()
    g[:hy, :hx] = tl[:hy, :hx]
    g[:hy, hx:] = tr[:hy, hx:]
    g[hy:, hx:] = br[hy:, hx:]
    g[hy:, :hx] = bl[hy:, :hx]
    return g


# --------------------------------------------------------------------------- #
# Tool C: verify_camera_projection


def test_camera_projection_identity_is_consistent(ctx: AppContext, tmp_path: Path):
    ref = _texture()
    ref_p = save_png8(ref, tmp_path / "ref.png")
    gen_p = save_png8(ref, tmp_path / "gen.png")  # identical -> camera matches
    rep = json.loads(video._verify_camera_projection(ctx, str(ref_p), str(gen_p)))
    assert "error" not in rep, rep
    assert rep["verdict"] == "camera-consistent"
    assert rep["median_displacement_px"] < 0.75
    assert rep["epipolar_degenerate"] is True  # no parallax -> F unidentifiable
    assert rep["n_correspondences"] >= 8


def test_camera_projection_global_shift_is_drift(ctx: AppContext, tmp_path: Path):
    ref = _texture()
    gen = ndimage.shift(ref, (0.0, 3.0), order=1, mode="nearest")  # content moves +3 px in x
    ref_p = save_png8(ref, tmp_path / "ref.png")
    gen_p = save_png8(gen, tmp_path / "gen.png")
    rep = json.loads(video._verify_camera_projection(ctx, str(ref_p), str(gen_p)))
    assert "error" not in rep, rep
    assert rep["verdict"] == "camera-drift"
    assert rep["camera_shift_px"] == pytest.approx(3.0, abs=1.5)
    assert rep["median_displacement_px"] == pytest.approx(3.0, abs=1.0)
    assert rep["homography_inlier_fraction"] > 0.7
    # a pure global translation is a homography (zero baseline) -> F unidentifiable
    assert rep["epipolar_degenerate"] is True


def test_camera_projection_low_contrast_shift_still_detected(ctx: AppContext, tmp_path: Path):
    # A very low-contrast but valid frame (subtle depth/AOV render). The joint contrast
    # stretch must let Lucas-Kanade track it, else a real drift reads as false 'consistent'.
    ref = 0.5 + 0.03 * (_texture() - 0.5)  # peak-to-peak ~0.03, well inside [0, 1]
    gen = ndimage.shift(ref, (0.0, 3.0), order=1, mode="nearest")
    ref_p = save_png16(ref, tmp_path / "ref.png")  # 16-bit so the low amplitude survives
    gen_p = save_png16(gen, tmp_path / "gen.png")
    rep = json.loads(video._verify_camera_projection(ctx, str(ref_p), str(gen_p)))
    assert "error" not in rep, rep
    assert rep["verdict"] == "camera-drift"  # NOT a false 'camera-consistent'
    assert rep["median_displacement_px"] > 1.0


def test_camera_projection_curl_warp_is_inconsistent(ctx: AppContext, tmp_path: Path):
    ref = _texture()
    gen = _curl_warp(ref)  # per-quadrant curl -> no global camera model fits
    ref_p = save_png8(ref, tmp_path / "ref.png")
    gen_p = save_png8(gen, tmp_path / "gen.png")
    rep = json.loads(video._verify_camera_projection(ctx, str(ref_p), str(gen_p)))
    assert "error" not in rep, rep
    assert rep["verdict"] == "geometry-inconsistent"
    assert rep["homography_inlier_fraction"] < 0.6
    assert rep["median_displacement_px"] > 0.75


def test_camera_projection_flat_reference_is_tool_error(ctx: AppContext, tmp_path: Path):
    flat = np.full((160, 160), 0.5, dtype=np.float64)  # same size as _texture(): reach the corner path
    ref_p = save_png8(flat, tmp_path / "flat.png")
    gen_p = save_png8(_texture(), tmp_path / "gen.png")
    rep = json.loads(video._verify_camera_projection(ctx, str(ref_p), str(gen_p)))
    assert "error" in rep
    assert "corner" in rep["error"]  # the textureless path, NOT a shape mismatch
    assert rep.get("hint")


def test_shi_tomasi_finds_texture_corners():
    corners = epipolar.shi_tomasi_corners(_texture(), max_corners=200)
    assert corners.ndim == 2 and corners.shape[1] == 2
    assert corners.shape[0] >= 20


# --------------------------------------------------------------------------- #
# Photometric-gate factories: a bumpy normal field + a Lambertian-lit frame


def _normal_field(h: int = 80, w: int = 80, freq: float = 3.0, amp: float = 0.35) -> np.ndarray:
    """Unit normals of a smooth sinusoidal bump surface (nz > 0 everywhere)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    height = amp * np.sin(2.0 * np.pi * freq * xx / w) * np.cos(2.0 * np.pi * freq * yy / h)
    dzdy, dzdx = np.gradient(height)
    n = np.stack([-dzdx, -dzdy, np.ones_like(height)], axis=2)
    return n / np.linalg.norm(n, axis=2, keepdims=True)


def _lit(normals: np.ndarray, light: np.ndarray, intensity: float = 0.5, ambient: float = 0.3) -> np.ndarray:
    """Lambertian shading ambient + intensity*(N.L_unit); intensity is kept small so
    the result stays inside [0, 1] and never saturates (a clip would destroy the
    linear normal->shading relationship the tool fits)."""
    return np.clip(ambient + intensity * (normals @ light), 0.0, 1.0)


_LIGHT = np.array([0.3, 0.2, 0.9]) / np.linalg.norm([0.3, 0.2, 0.9])


def test_photometric_consistent_shading_matches_normals(ctx: AppContext, tmp_path: Path):
    normals = _normal_field()
    gen = _lit(normals, _LIGHT)
    nrm_p = _save_rgb8((normals + 1.0) / 2.0, tmp_path / "normals.png")
    gen_p = save_png8(gen, tmp_path / "gen.png")
    rep = json.loads(video._analyze_photometric_consistency(ctx, str(gen_p), str(nrm_p)))
    assert "error" not in rep, rep
    assert rep["verdict"] == "consistent"
    assert rep["photometric_r2"] > 0.85
    assert rep["used_albedo"] is False
    # recovered light direction aligns with the true light
    d = np.array(rep["light_direction"])
    assert float(d @ _LIGHT) > 0.8


def test_photometric_hallucinated_shading_flagged(ctx: AppContext, tmp_path: Path):
    normals = _normal_field()
    rng = np.random.default_rng(0)
    gen = rng.random(normals.shape[:2])  # shading uncorrelated with the true normals
    nrm_p = _save_rgb8((normals + 1.0) / 2.0, tmp_path / "normals.png")
    gen_p = save_png8(gen, tmp_path / "gen.png")
    rep = json.loads(video._analyze_photometric_consistency(ctx, str(gen_p), str(nrm_p)))
    assert "error" not in rep, rep
    assert rep["verdict"] == "inconsistent"
    assert rep["photometric_r2"] < 0.4


def test_photometric_flat_frame_is_consistent(ctx: AppContext, tmp_path: Path):
    # A flat/ambient frame has no shading variation, so there is nothing to CONTRADICT
    # the geometry: it must read 'consistent' (R^2 -> 1.0), not a false 'inconsistent'.
    normals = _normal_field()
    gen = np.full(normals.shape[:2], 0.5, dtype=np.float64)
    nrm_p = _save_rgb8((normals + 1.0) / 2.0, tmp_path / "normals.png")
    gen_p = save_png8(gen, tmp_path / "gen.png")
    rep = json.loads(video._analyze_photometric_consistency(ctx, str(gen_p), str(nrm_p)))
    assert "error" not in rep, rep
    assert rep["verdict"] == "consistent"
    assert rep["photometric_r2"] == pytest.approx(1.0, abs=1e-6)


def test_photometric_albedo_leak_detected(ctx: AppContext, tmp_path: Path):
    normals = _normal_field()
    yy, xx = np.mgrid[0 : normals.shape[0], 0 : normals.shape[1]].astype(np.float64)
    albedo = 0.4 + 0.4 * (0.5 + 0.5 * np.sin(2.0 * np.pi * 8.0 * xx / normals.shape[1]))  # textured
    lit = _lit(normals, _LIGHT)
    nrm_p = _save_rgb8((normals + 1.0) / 2.0, tmp_path / "normals.png")
    alb_p = save_png8(albedo, tmp_path / "albedo.png")

    clean = save_png8(np.clip(albedo * lit, 0.0, 1.0), tmp_path / "clean.png")  # correct I = R*S
    # texture baked in twice -> albedo gradients leak into the recovered shading
    double = save_png8(np.clip(albedo * albedo * lit, 0.0, 1.0), tmp_path / "double.png")
    clean_rep = json.loads(
        video._analyze_photometric_consistency(ctx, str(clean), str(nrm_p), albedo=str(alb_p))
    )
    double_rep = json.loads(
        video._analyze_photometric_consistency(ctx, str(double), str(nrm_p), albedo=str(alb_p))
    )
    assert "error" not in clean_rep and "error" not in double_rep
    assert clean_rep["used_albedo"] is True
    assert clean_rep["albedo_shading_leak"] is not None
    # double-textured shading leaks albedo gradients; clean intrinsic separation does not
    assert double_rep["albedo_shading_leak"] > clean_rep["albedo_shading_leak"]
    assert double_rep["albedo_shading_leak"] > 0.3


def test_decode_normal_map_unit_length():
    normals = _normal_field()
    decoded = photometric.decode_normal_map((normals + 1.0) / 2.0)
    lengths = np.linalg.norm(decoded, axis=2)
    assert np.allclose(lengths, 1.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# Occlusion-gate factories: a two-level depth pass + a crisp beauty step


def _fg_square(size: int = 80, box: tuple[int, int, int, int] = (24, 24, 56, 56)) -> tuple:
    """(depth, sharp beauty): a foreground square (near) on a far background, and a
    crisp intensity step image aligned to it (bright fg on dark bg)."""
    x0, y0, x1, y1 = box
    depth = np.full((size, size), 0.8, dtype=np.float64)
    depth[y0:y1, x0:x1] = 0.3  # near square -> a depth discontinuity at its border
    beauty = np.full((size, size), 0.2, dtype=np.float64)
    beauty[y0:y1, x0:x1] = 0.9
    return depth, beauty


def test_occlusion_sharp_frame_is_sharp(ctx: AppContext, tmp_path: Path):
    depth, sharp = _fg_square()
    depth_p = save_png8(depth, tmp_path / "depth.png")
    gen_p = save_png8(sharp, tmp_path / "gen.png")
    rep = json.loads(video._evaluate_occlusion_boundaries(ctx, str(gen_p), str(depth_p)))
    assert "error" not in rep, rep
    assert rep["verdict"] == "sharp"
    assert rep["edge_pixel_count"] > 0
    assert Path(rep["edge_mask_path"]).is_file()
    assert Path(rep["edge_mask_path"]).parent == ctx.data_dir / "video"


def test_occlusion_blurred_frame_soft_vs_reference(ctx: AppContext, tmp_path: Path):
    depth, sharp = _fg_square()
    blurred = ndimage.gaussian_filter(sharp, sigma=2.0, mode="reflect")
    depth_p = save_png8(depth, tmp_path / "depth.png")
    ref_p = save_png8(sharp, tmp_path / "ref.png")
    blur_p = save_png8(blurred, tmp_path / "blur.png")

    soft = json.loads(
        video._evaluate_occlusion_boundaries(ctx, str(blur_p), str(depth_p), reference=str(ref_p))
    )
    keep = json.loads(
        video._evaluate_occlusion_boundaries(ctx, str(ref_p), str(depth_p), reference=str(ref_p))
    )
    assert "error" not in soft and "error" not in keep
    assert soft["verdict"] == "soft-boundaries"
    assert soft["gen_vs_ref_ratio"] < 0.6
    assert keep["verdict"] == "sharp"
    assert keep["gen_vs_ref_ratio"] == pytest.approx(1.0, abs=0.05)
    # the blurred frame really is less sharp at the occlusion band
    assert soft["boundary_sharpness"] < keep["boundary_sharpness"]


def test_occlusion_flat_depth_is_tool_error(ctx: AppContext, tmp_path: Path):
    flat = np.full((64, 64), 0.5, dtype=np.float64)
    _, sharp = _fg_square(size=64, box=(16, 16, 48, 48))
    depth_p = save_png8(flat, tmp_path / "flat_depth.png")
    gen_p = save_png8(sharp, tmp_path / "gen.png")
    rep = json.loads(video._evaluate_occlusion_boundaries(ctx, str(gen_p), str(depth_p)))
    assert "error" in rep
    assert rep.get("hint")


def test_occlusion_smooth_ramp_depth_is_tool_error(ctx: AppContext, tmp_path: Path):
    # A pervasive smooth depth gradient (a receding ground plane) has NO localized
    # discontinuity: the range-floor must reject it instead of flooding the band and
    # faking a 'sharp' verdict on the whole frame.
    size = 80
    _, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    ramp = 0.2 + 0.6 * (xx / (size - 1))  # smooth left-to-right depth ramp
    _, sharp = _fg_square()
    depth_p = save_png16(ramp, tmp_path / "ramp_depth.png")  # 16-bit: no quantization steps
    gen_p = save_png8(sharp, tmp_path / "gen.png")
    rep = json.loads(video._evaluate_occlusion_boundaries(ctx, str(gen_p), str(depth_p)))
    assert "error" in rep
    assert rep.get("hint")


def test_depth_edge_band_localizes_border():
    depth, _ = _fg_square()
    band, edges = occlusion.depth_edge_band(depth, percentile=95.0, band_px=2)
    assert edges.sum() > 0
    assert band.sum() >= edges.sum()  # dilation only grows the set
    # the band hugs the square's border, not its flat interior or the far field
    assert not band[0:10, 0:10].any()
    assert not band[38:42, 38:42].any()  # centre of the 24..56 square is interior

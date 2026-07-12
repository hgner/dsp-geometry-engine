"""Bake scoring: one objective number for a posed engine bake vs a library
reference pack. Two independent lanes, both off render-side PNGs already on disk:

  pose-correctness  — silhouette IoU + Hu-moment shape distance + Sobel
                      edge-orientation correlation, facing-normalized (best of
                      mirror), against every depth in a reference glob.
  mesh-quality      — oriented 2-D FFT prominence (dB) of the bake depth over the
                      body ROI: the render-side proxy for the edge-loop corrugation
                      whose authoritative measure is the vertex-dump analyzer
                      (analyze_corrugation). Kept here so the score is a single
                      GPU-free call inside an iterate-bake-score loop.

Pure numpy/scipy/pillow (no OpenCV): Hu moments and IoU are closed-form, so the
metric adds no dependency and stays deterministic. All heavy arrays stay in
memory; the tool layer persists only the compact report.
"""

from __future__ import annotations

import glob
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from dsp_server.engine import image2d

# Provisional gate thresholds. These are COMPARATIVE aids, not absolute truth:
# silhouette IoU against a different body tops out well below 1.0, and the depth
# FFT dB is a distinct scale from the vertex-dump dB. Calibrate against a bake the
# operator has ACCEPTED; until then the verdict's job is "does iter N beat N-1".
IOU_TARGET = 0.50
HU_TARGET = 0.30  # matchShapes-I2, lower is closer
# ADVISORY only: the depth-render FFT is dominated by the body's bulk shape and does
# NOT resolve the ~8 mm edge-loop ripple as a distinct peak (measured 2026-07-11:
# ~30 dB in-band floor, stable across poses of the same mesh). The AUTHORITATIVE
# mesh-corrugation measure is analyze_corrugation on the vertex DUMP (edge-loop peak
# at ~119 cy/m). So the score_bake GATE is pose-driven; mesh_ok is reported, not gating.
CORRUGATION_DB_ADVISORY = 30.0
NORM_CANVAS = 256
# Fine-ripple frequency band (cycles/pixel) for the depth-FFT corrugation proxy.
# BELOW ~0.03 cpp the spectrum is dominated by the body's bulk depth shape (a
# 100-px-wavelength ramp reads as 50 dB and swamps the ripple); the edge-loop
# ripple at render scale sits well inside this band. ABOVE 0.25 is pixel noise.
CORRUGATION_BAND_CPP = (0.03, 0.25)


# --------------------------------------------------------------------------- #
# Silhouette extraction + facing-normalized canvas


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 4-connected blob (drops floor gradients / text /
    speckle that survive a global threshold on a reference depth)."""
    m = np.asarray(mask, dtype=bool)
    labels, n = ndimage.label(m)
    if n <= 1:
        return m
    areas = np.bincount(labels.ravel())
    areas[0] = 0  # background
    return labels == int(np.argmax(areas))


def silhouette_from_depth(depth01: np.ndarray, foreground: str = "auto") -> np.ndarray:
    """Body mask from a reference depth: Otsu split + largest component. With
    foreground='auto' the bright split is tried first and flipped to dark when it
    claims an implausible (>60%) fraction of the frame (inverted depth packs)."""
    thr = image2d.otsu_threshold(depth01)
    bright = _largest_component(depth01 > thr)
    if foreground == "dark":
        return _largest_component(depth01 <= thr)
    if foreground == "bright":
        return bright
    if bright.mean() > 0.60:  # 'body' filled most of the frame -> wrong polarity
        return _largest_component(depth01 <= thr)
    return bright


def normalize_silhouette(mask: np.ndarray, canvas: int = NORM_CANVAS) -> np.ndarray:
    """Crop to the mask bbox, scale the longest side to ``canvas`` (aspect kept),
    pad-center on a canvas x canvas bool image. Facing/scale/translation
    invariant except for handedness (the caller tries both mirrors)."""
    m = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return np.zeros((canvas, canvas), dtype=bool)
    crop = m[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = crop.shape
    scale = canvas / max(h, w)
    zoomed = ndimage.zoom(crop.astype(np.float32), scale, order=0) > 0.5
    zh, zw = zoomed.shape
    zh, zw = min(zh, canvas), min(zw, canvas)
    zoomed = zoomed[:zh, :zw]
    out = np.zeros((canvas, canvas), dtype=bool)
    y0, x0 = (canvas - zh) // 2, (canvas - zw) // 2
    out[y0 : y0 + zh, x0 : x0 + zw] = zoomed
    return out


# --------------------------------------------------------------------------- #
# Shape metrics


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def hu_moments(mask: np.ndarray) -> np.ndarray:
    """The 7 Hu invariant moments of a binary silhouette (Hu 1962), log-scaled the
    way OpenCV's matchShapes consumes them: m_i = sign(h_i)*log10(|h_i|)."""
    m = np.asarray(mask, dtype=np.float64)
    total = m.sum()
    if total <= 0.0:
        return np.zeros(7, dtype=np.float64)
    ys, xs = np.mgrid[0 : m.shape[0], 0 : m.shape[1]].astype(np.float64)
    x_bar = float((xs * m).sum() / total)
    y_bar = float((ys * m).sum() / total)
    xc, yc = xs - x_bar, ys - y_bar

    def mu(p: int, q: int) -> float:
        return float(((xc**p) * (yc**q) * m).sum())

    def nu(p: int, q: int) -> float:
        return mu(p, q) / (total ** (1.0 + (p + q) / 2.0))

    n20, n02, n11 = nu(2, 0), nu(0, 2), nu(1, 1)
    n30, n12, n21, n03 = nu(3, 0), nu(1, 2), nu(2, 1), nu(0, 3)
    h = np.empty(7, dtype=np.float64)
    h[0] = n20 + n02
    h[1] = (n20 - n02) ** 2 + 4.0 * n11**2
    h[2] = (n30 - 3.0 * n12) ** 2 + (3.0 * n21 - n03) ** 2
    h[3] = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h[4] = (n30 - 3.0 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3.0 * (n21 + n03) ** 2) + (
        3.0 * n21 - n03
    ) * (n21 + n03) * (3.0 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    h[5] = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4.0 * n11 * (n30 + n12) * (n21 + n03)
    h[6] = (3.0 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3.0 * (n21 + n03) ** 2) - (
        n30 - 3.0 * n12
    ) * (n21 + n03) * (3.0 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_h = np.where(h != 0.0, np.sign(h) * np.log10(np.abs(h)), 0.0)
    return log_h


def match_shapes_i2(hu_a: np.ndarray, hu_b: np.ndarray) -> float:
    """OpenCV CONTOURS_MATCH_I2 distance over log-scaled Hu moments (0 = identical).

    I2 is the DIRECT difference ``sum |m_a - m_b|`` (which is what HU_TARGET=0.30 was
    calibrated for). The reciprocal ``|1/m_a - 1/m_b|`` form is OpenCV's I1, and it
    blows up when a log-Hu moment is near 0 (|h_i| ~ 1, reachable for elongated human
    silhouettes) — a spurious huge distance that falsely flips the pose verdict."""
    d = 0.0
    for a, b in zip(hu_a, hu_b, strict=True):
        if a != 0.0 and b != 0.0:
            d += abs(a - b)
    return float(d)


def edge_orientation_hist(gray01: np.ndarray, mask: np.ndarray, bins: int = 18) -> np.ndarray:
    """Magnitude-weighted Sobel edge-orientation histogram over the masked region,
    folded to [0,180) and L2-normalized (a direction-of-surface-detail signature)."""
    gx = ndimage.sobel(gray01, axis=1, mode="reflect")
    gy = ndimage.sobel(gray01, axis=0, mode="reflect")
    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 180.0
    sel = np.asarray(mask, dtype=bool)
    if not sel.any():
        return np.zeros(bins, dtype=np.float64)
    hist, _ = np.histogram(ang[sel], bins=bins, range=(0.0, 180.0), weights=mag[sel])
    norm = float(np.linalg.norm(hist))
    return hist / norm if norm > 0.0 else hist


# --------------------------------------------------------------------------- #
# Composite


@dataclass
class RefMatch:
    ref: str
    iou: float
    hu: float
    edge_corr: float


@dataclass
class BakeScore:
    bake_depth: str
    bake_mask: str
    n_refs: int
    iou_best: float
    iou_mean: float
    hu_best: float
    hu_mean: float
    edge_corr_best: float
    edge_corr_mean: float
    corrugation_db: float
    corrugation_freq_cpp: float
    corrugation_orientation_deg: float
    pose_ok: bool
    mesh_ok: bool
    verdict: str
    top_matches: list[RefMatch]


def _pose_metrics_vs_ref(
    bake_sil: np.ndarray, bake_gray: np.ndarray, ref_sil: np.ndarray, ref_gray: np.ndarray
) -> tuple[float, float, float]:
    """Best-of-mirror silhouette metrics (mirror chosen by IoU, the least noisy)."""
    bh_bake = hu_moments(bake_sil)
    eh_bake = edge_orientation_hist(bake_gray, bake_sil)
    best = (-1.0, math.inf, 0.0)
    for flip in (False, True):
        rs = ref_sil[:, ::-1] if flip else ref_sil
        rg = ref_gray[:, ::-1] if flip else ref_gray
        this_iou = iou(bake_sil, rs)
        if this_iou > best[0]:
            hu = match_shapes_i2(bh_bake, hu_moments(rs))
            ec = float(np.dot(eh_bake, edge_orientation_hist(rg, rs)))
            best = (this_iou, hu, ec)
    return best


def depth_corrugation_db(
    depth01: np.ndarray, mask: np.ndarray, band: tuple[float, float] = CORRUGATION_BAND_CPP
) -> tuple[float, float, float]:
    """Render-side corrugation proxy: 2-D FFT prominence over the masked body bbox,
    peak searched ONLY within the fine-ripple band (cycles/pixel). The global
    argmax would lock onto the bulk depth ramp (~0.01 cpp, tens of dB) and never
    see the edge-loop ripple; band-limiting isolates the periodic surface detail.
    Returns (prominence_db, dominant cpp, orientation degrees) in-band."""
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if ys.size == 0:
        return 0.0, 0.0, 0.0
    roi = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    spec = image2d.roi_spectrum_2d(depth01, roi=roi, detrend=True)
    fx, fy = np.meshgrid(spec.freqs_x_cpp, spec.freqs_y_cpp)
    radial = np.hypot(fx, fy)
    in_band = (radial >= band[0]) & (radial <= band[1])
    if not in_band.any() or spec.psd[in_band].max() <= 0.0:
        return 0.0, 0.0, 0.0
    psd_band = np.where(in_band, spec.psd, 0.0)
    iy, ix = np.unravel_index(int(np.argmax(psd_band)), psd_band.shape)
    peak = float(spec.psd[iy, ix])
    background = float(np.median(spec.psd[in_band]))
    prominence_db = 10.0 * math.log10(peak / max(background, peak * 1e-12))
    freq = float(radial[iy, ix])
    orientation = float(np.degrees(np.arctan2(spec.freqs_y_cpp[iy], spec.freqs_x_cpp[ix])) % 180.0)
    return prominence_db, freq, orientation


def score_bake(
    bake_depth_path: str | Path,
    bake_mask_path: str | Path,
    reference_glob: str,
    *,
    top_k: int = 3,
    iou_target: float = IOU_TARGET,
    hu_target: float = HU_TARGET,
    corrugation_db_advisory: float = CORRUGATION_DB_ADVISORY,
    ref_foreground: str = "auto",
) -> BakeScore:
    """Score one bake frame (depth PNG + person mask PNG) against a reference glob.
    Raises ValueError on missing inputs / empty glob / empty silhouette; the tool
    layer converts that to a ToolError envelope."""
    bake_depth_path = Path(bake_depth_path)
    bake_mask_path = Path(bake_mask_path)
    if not bake_depth_path.is_file():
        raise ValueError(f"bake depth not found: {bake_depth_path}")
    if not bake_mask_path.is_file():
        raise ValueError(f"bake mask not found: {bake_mask_path}")
    ref_paths = sorted(glob.glob(reference_glob))
    if not ref_paths:
        raise ValueError(f"no references match glob: {reference_glob}")

    bake_gray_full = image2d.load_image_gray(bake_depth_path)
    bake_mask_full = image2d.load_image_gray(bake_mask_path) > 0.5
    if not bake_mask_full.any():
        raise ValueError(f"bake mask is empty (all background): {bake_mask_path}")
    bake_mask_full = _largest_component(bake_mask_full)

    # The mesh corrugation lane is ADVISORY (the pose lane is the reliable gate), so a
    # too-small / degenerate mask bbox (roi_spectrum_2d needs >= 8x8) must NOT sink the
    # whole tool — degrade to "not measured" (0 dB -> mesh_ok) and let pose drive.
    try:
        corr_db, corr_freq, corr_deg = depth_corrugation_db(bake_gray_full, bake_mask_full)
    except ValueError:
        corr_db, corr_freq, corr_deg = 0.0, 0.0, 0.0

    bake_sil = normalize_silhouette(bake_mask_full)
    bake_gray = _normalize_gray(bake_gray_full, bake_mask_full)

    matches: list[RefMatch] = []
    for rp in ref_paths:
        ref_gray_full = image2d.load_image_gray(rp)
        ref_mask_full = silhouette_from_depth(ref_gray_full, foreground=ref_foreground)
        if not ref_mask_full.any():
            continue
        ref_sil = normalize_silhouette(ref_mask_full)
        ref_gray = _normalize_gray(ref_gray_full, ref_mask_full)
        m_iou, m_hu, m_ec = _pose_metrics_vs_ref(bake_sil, bake_gray, ref_sil, ref_gray)
        matches.append(RefMatch(ref=Path(rp).name, iou=m_iou, hu=m_hu, edge_corr=m_ec))
    if not matches:
        raise ValueError("every reference silhouette was empty after thresholding")

    ious = np.array([m.iou for m in matches])
    hus = np.array([m.hu for m in matches])
    ecs = np.array([m.edge_corr for m in matches])
    iou_best = float(ious.max())
    hu_best = float(hus.min())
    pose_ok = iou_best >= iou_target and hu_best <= hu_target
    mesh_ok = corr_db < corrugation_db_advisory  # advisory: see CORRUGATION_DB_ADVISORY
    verdict = _verdict(pose_ok, mesh_ok)
    top = sorted(matches, key=lambda m: m.iou, reverse=True)[:top_k]
    return BakeScore(
        bake_depth=str(bake_depth_path),
        bake_mask=str(bake_mask_path),
        n_refs=len(matches),
        iou_best=iou_best,
        iou_mean=float(ious.mean()),
        hu_best=hu_best,
        hu_mean=float(hus.mean()),
        edge_corr_best=float(ecs.max()),
        edge_corr_mean=float(ecs.mean()),
        corrugation_db=corr_db,
        corrugation_freq_cpp=corr_freq,
        corrugation_orientation_deg=corr_deg,
        pose_ok=pose_ok,
        mesh_ok=mesh_ok,
        verdict=verdict,
        top_matches=top,
    )


def _normalize_gray(gray01: np.ndarray, mask: np.ndarray, canvas: int = NORM_CANVAS) -> np.ndarray:
    """Same bbox-crop + longest-side scale + pad-center as the silhouette, applied
    to the masked gray image (background zeroed) so the edge histogram sees only
    the body's own surface detail."""
    m = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return np.zeros((canvas, canvas), dtype=np.float64)
    g = (gray01 * m)[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    scale = canvas / max(g.shape)
    z = ndimage.zoom(g, scale, order=1)
    zh, zw = min(z.shape[0], canvas), min(z.shape[1], canvas)
    z = z[:zh, :zw]
    out = np.zeros((canvas, canvas), dtype=np.float64)
    y0, x0 = (canvas - zh) // 2, (canvas - zw) // 2
    out[y0 : y0 + zh, x0 : x0 + zw] = z
    return out


def _verdict(pose_ok: bool, mesh_ok: bool) -> str:
    """Pose-driven gate (the reliable lane). The mesh flag rides along advisorily:
    'pass*' / 'pose-weak*' mark that the advisory mesh corrugation is also high, so
    the caller knows to run analyze_corrugation on the vertex dump before trusting it."""
    base = "pass" if pose_ok else "pose-weak"
    return base if mesh_ok else base + "*"

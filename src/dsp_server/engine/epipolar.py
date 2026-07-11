"""Epipolar / camera-projection consistency for the video comparison gate.

The engine renders with a strict projection matrix ``P = K[R|t]``; a generative
video model can imperceptibly drift the camera (focal length, field of view, nodal
point) while every frame still looks plausible. This module extracts point
correspondences between a REFERENCE render (ground truth) and a GENERATED frame —
Shi-Tomasi corners in the reference, tracked into the generated frame by the repo's
own pyramidal Lucas-Kanade (:mod:`dsp_server.engine.optflow`) with a
forward-backward gate — then fits two global camera models to them:

* a **homography** ``H`` (normalized DLT + RANSAC): the model a genuine same-view
  camera drift produces. Its reprojection residual plus the induced image-centre
  shift / scale / rotation quantify the drift, and it is well-defined even at zero
  baseline. This is the RELIABLE metric and drives the verdict.
* a **fundamental matrix** ``F`` (normalized 8-point + RANSAC): the general
  two-view epipolar model. Its mean SYMMETRIC epipolar distance is what the caller
  asked for, but ``F`` is only identifiable with real parallax, so a near-zero-
  baseline (same-view) pair is flagged ``epipolar_degenerate`` instead of being
  reported as a spurious residual.

Pure numpy (SVD) + the repo's optical flow. RANSAC uses a fixed-seed RNG, so the
whole analysis is deterministic. Frame I/O is owned by
:mod:`dsp_server.engine.image2d`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from dsp_server.engine import optflow

__all__ = [
    "CameraProjectionResult",
    "analyze_camera_projection",
    "correspondences",
    "shi_tomasi_corners",
]

_TINY = 1e-12
_MIN_CORRESPONDENCES = 8  # 8-point algorithm floor (homography needs only 4)
_RANSAC_SEED = 0  # deterministic sampling


@dataclass
class CameraProjectionResult:
    """Camera-projection consistency of a generated frame vs a reference render.

    ``verdict``: ``camera-consistent`` (correspondences are near-identity — the
    generated camera matches the reference), ``camera-drift`` (a single global
    homography explains a non-trivial shift — the camera transform drifted:
    pan / zoom / FOV), or ``geometry-inconsistent`` (no global camera model fits —
    local warping / hallucinated geometry). ``camera_shift_px`` / ``camera_scale``
    / ``camera_rotation_deg`` decompose the homography at the image centre (a
    near-affine approximation). ``mean_symmetric_epipolar_distance_px`` is the
    requested epipolar residual; trust it only when ``epipolar_degenerate`` is
    False (i.e. there is genuine parallax).
    """

    height: int
    width: int
    n_corners: int
    n_correspondences: int
    median_displacement_px: float
    max_displacement_px: float
    homography_inlier_fraction: float
    homography_residual_px: float
    camera_shift_px: float
    camera_scale: float
    camera_rotation_deg: float
    epipolar_inlier_fraction: float
    mean_symmetric_epipolar_distance_px: float
    epipolar_degenerate: bool
    verdict: str
    verdict_reason: str


# --------------------------------------------------------------------------- #
# Feature extraction + correspondences


def shi_tomasi_corners(
    img: np.ndarray,
    max_corners: int = 200,
    quality: float = 0.01,
    min_distance: int = 8,
    border: int = 16,
) -> np.ndarray:
    """Shi-Tomasi corners of ``img`` as an ``(N, 2)`` array of ``(x, y)`` pixels.

    The corner response is the SMALLER eigenvalue of the Gaussian-windowed
    structure tensor (good features to track). Non-maximum suppression keeps only
    local maxima at least ``min_distance`` apart; corners within ``border`` pixels
    of the edge are dropped so a Lucas-Kanade window stays in-bounds. Returns the
    strongest ``max_corners``, response-descending.
    """
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale frame, got shape {arr.shape}")
    h, w = arr.shape
    ix = ndimage.sobel(arr, axis=1, mode="reflect") / 8.0
    iy = ndimage.sobel(arr, axis=0, mode="reflect") / 8.0
    sxx = ndimage.gaussian_filter(ix * ix, sigma=2.0, mode="reflect")
    syy = ndimage.gaussian_filter(iy * iy, sigma=2.0, mode="reflect")
    sxy = ndimage.gaussian_filter(ix * iy, sigma=2.0, mode="reflect")
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    response = 0.5 * (tr - np.sqrt(disc))  # smaller eigenvalue

    b = min(int(border), h // 2, w // 2)
    if b > 0:
        response[:b, :] = 0.0
        response[-b:, :] = 0.0
        response[:, :b] = 0.0
        response[:, -b:] = 0.0
    peak_max = float(response.max())
    if peak_max <= 0.0:
        return np.empty((0, 2), dtype=np.float64)

    thresh = float(quality) * peak_max
    size = 2 * int(min_distance) + 1
    local_max = ndimage.maximum_filter(response, size=size, mode="constant", cval=0.0)
    peaks = (response == local_max) & (response > thresh)
    ys, xs = np.nonzero(peaks)
    if xs.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    order = np.argsort(response[ys, xs], kind="stable")[::-1][: int(max_corners)]
    return np.column_stack([xs[order], ys[order]]).astype(np.float64)


def _stretch(ref: np.ndarray, gen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rescale both frames by the reference's [min, max] to unit range (a no-op for a
    full-contrast frame). A constant reference is returned unchanged (it has no
    corners and is caught downstream)."""
    lo, hi = float(ref.min()), float(ref.max())
    span = hi - lo
    if span < _TINY:
        return ref, gen
    inv = 1.0 / span
    return (ref - lo) * inv, (gen - lo) * inv


def correspondences(
    reference: np.ndarray,
    generated: np.ndarray,
    max_corners: int = 200,
    window: int = 15,
    fb_tol: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Point correspondences ``(src, dst)`` between two same-shape frames.

    Corners in ``reference`` are tracked into ``generated`` by pyramidal
    Lucas-Kanade, then gated by a forward-backward round trip (drop any point whose
    ``ref -> gen -> ref`` error exceeds ``fb_tol`` px, or that leaves the frame).
    Returns two ``(M, 2)`` arrays of matched ``(x, y)`` pixels.
    """
    ref = np.asarray(reference, dtype=np.float64)
    gen = np.asarray(generated, dtype=np.float64)
    if ref.shape != gen.shape:
        raise ValueError(f"frames must match: {ref.shape} vs {gen.shape}")
    if ref.ndim != 2:
        raise ValueError(f"expected 2-D grayscale frames, got shape {ref.shape}")
    # Joint contrast stretch by the reference's range: Lucas-Kanade's structure-tensor
    # guard is an ABSOLUTE eigenvalue floor, so a low-contrast (but valid [0,1]) frame
    # would otherwise get zero flow at every corner and read as a false "consistent".
    # The same affine on both frames preserves corner locations and displacements.
    ref, gen = _stretch(ref, gen)
    src = shi_tomasi_corners(ref, max_corners=max_corners)
    if src.shape[0] < _MIN_CORRESPONDENCES:
        raise ValueError(
            f"only {src.shape[0]} trackable corners in the reference frame (need "
            f">= {_MIN_CORRESPONDENCES}) — the reference is too flat/low-texture for a "
            "camera-projection check"
        )
    fwd = optflow.lucas_kanade(ref, gen, src, window=window)
    dst = src + fwd
    back = optflow.lucas_kanade(gen, ref, dst, window=window)
    fb_err = np.hypot((dst + back)[:, 0] - src[:, 0], (dst + back)[:, 1] - src[:, 1])
    h, w = ref.shape
    in_bounds = (dst[:, 0] >= 0.0) & (dst[:, 0] <= w - 1.0) & (dst[:, 1] >= 0.0) & (dst[:, 1] <= h - 1.0)
    keep = (fb_err < float(fb_tol)) & in_bounds
    return src[keep], dst[keep]


# --------------------------------------------------------------------------- #
# Geometry: normalized DLT homography + normalized 8-point fundamental matrix


def _normalize_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalization: centroid to origin, mean distance sqrt(2). Returns
    the normalized points and the 3x3 similarity ``T`` that produced them."""
    c = pts.mean(axis=0)
    d = pts - c
    mean_dist = float(np.sqrt((d * d).sum(axis=1)).mean())
    scale = np.sqrt(2.0) / mean_dist if mean_dist > _TINY else 1.0
    t = np.array([[scale, 0.0, -scale * c[0]], [0.0, scale, -scale * c[1]], [0.0, 0.0, 1.0]])
    normed = d * scale
    return normed, t


def _fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """DLT homography ``dst ~ H src`` from >= 4 correspondences (normalized)."""
    ns, ts = _normalize_points(src)
    nd, td = _normalize_points(dst)
    rows = []
    for (x, y), (u, v) in zip(ns, nd, strict=True):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _u, _s, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    hn = vt[-1].reshape(3, 3)
    h = np.linalg.inv(td) @ hn @ ts
    return h / h[2, 2] if abs(h[2, 2]) > _TINY else h


def _reprojection_error(h: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Per-point reprojection residual ``|| dehom(H src) - dst ||`` in pixels."""
    srch = np.column_stack([src, np.ones(len(src))])
    proj = srch @ h.T
    wz = proj[:, 2]
    wz = np.where(np.abs(wz) < _TINY, _TINY, wz)
    xy = proj[:, :2] / wz[:, None]
    return np.hypot(xy[:, 0] - dst[:, 0], xy[:, 1] - dst[:, 1])


def _fit_fundamental(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Normalized 8-point fundamental matrix ``dst^T F src = 0`` (rank-2 enforced)."""
    ns, ts = _normalize_points(src)
    nd, td = _normalize_points(dst)
    x, y = ns[:, 0], ns[:, 1]
    xp, yp = nd[:, 0], nd[:, 1]
    a = np.column_stack([xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, np.ones_like(x)])
    _u, _s, vt = np.linalg.svd(a)
    f = vt[-1].reshape(3, 3)
    uf, sf, vtf = np.linalg.svd(f)
    sf[2] = 0.0  # enforce rank 2 (a valid fundamental matrix is singular)
    f = uf @ np.diag(sf) @ vtf
    f = td.T @ f @ ts
    norm = float(np.linalg.norm(f))
    return f / norm if norm > _TINY else f


def _symmetric_epipolar_distance(f: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Symmetric epipolar distance (px) per correspondence for fundamental ``F``."""
    srch = np.column_stack([src, np.ones(len(src))])
    dsth = np.column_stack([dst, np.ones(len(dst))])
    fx = srch @ f.T  # epipolar lines in the dst image
    ftx = dsth @ f  # epipolar lines in the src image
    num = np.sum(dsth * fx, axis=1) ** 2  # (x'^T F x)^2
    d1 = fx[:, 0] ** 2 + fx[:, 1] ** 2
    d2 = ftx[:, 0] ** 2 + ftx[:, 1] ** 2
    dist_sq = num * (1.0 / np.maximum(d1, _TINY) + 1.0 / np.maximum(d2, _TINY))
    return np.sqrt(np.maximum(dist_sq, 0.0))


def _ransac(
    src: np.ndarray,
    dst: np.ndarray,
    n_sample: int,
    fit_fn,
    error_fn,
    thresh: float,
    iters: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generic RANSAC: returns the model refit on its inliers and the inlier mask."""
    n = len(src)
    best_mask = np.zeros(n, dtype=bool)
    best_count = -1
    for _ in range(int(iters)):
        idx = rng.choice(n, n_sample, replace=False)
        try:
            model = fit_fn(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        mask = error_fn(model, src, dst) < thresh
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask
    if best_mask.sum() < n_sample:  # RANSAC found nothing better than a minimal set
        best_mask = np.ones(n, dtype=bool)
    model = fit_fn(src[best_mask], dst[best_mask])
    return model, best_mask


def _decompose_homography(h: np.ndarray, width: int, height: int) -> tuple[float, float, float]:
    """Image-centre shift (px), scale, and rotation (deg) of a near-affine ``H``."""
    center = np.array([width / 2.0, height / 2.0, 1.0])
    mapped = h @ center
    mapped = mapped[:2] / (mapped[2] if abs(mapped[2]) > _TINY else _TINY)
    shift = float(np.hypot(mapped[0] - center[0], mapped[1] - center[1]))
    a2 = h[:2, :2]
    scale = float(np.sqrt(abs(np.linalg.det(a2))))
    rotation = float(np.degrees(np.arctan2(a2[1, 0], a2[0, 0])))
    return shift, scale, rotation


# --------------------------------------------------------------------------- #
# Top-level analysis


def analyze_camera_projection(
    reference: np.ndarray,
    generated: np.ndarray,
    max_corners: int = 200,
    window: int = 15,
    ransac_px: float = 2.0,
    ransac_iters: int = 200,
    consistent_px: float = 0.75,
    fb_tol: float = 1.0,
) -> CameraProjectionResult:
    """Fit global camera models to reference->generated correspondences and verdict.

    Correspondences come from :func:`correspondences`; a homography and a
    fundamental matrix are each fit by RANSAC at a ``ransac_px`` inlier threshold.
    The verdict is driven by the homography (reliable at any baseline); the
    fundamental matrix's symmetric epipolar distance is reported but flagged
    ``epipolar_degenerate`` when the motion is near-identity OR a single homography
    already explains the correspondences (a pure pan / zoom / planar scene has zero
    baseline, so the epipolar geometry is not identifiable however large the
    displacement).
    """
    ref = np.asarray(reference, dtype=np.float64)
    gen = np.asarray(generated, dtype=np.float64)
    if ref.shape != gen.shape:
        raise ValueError(f"frames must match: {ref.shape} vs {gen.shape}")
    h, w = ref.shape
    n_corners = int(shi_tomasi_corners(ref, max_corners=max_corners).shape[0])
    src, dst = correspondences(ref, gen, max_corners=max_corners, window=window, fb_tol=fb_tol)
    m = int(src.shape[0])
    if m < _MIN_CORRESPONDENCES:
        raise ValueError(
            f"only {m} surviving correspondences after the forward-backward gate (need "
            f">= {_MIN_CORRESPONDENCES}) — the generated frame diverges too much from the "
            "reference for a camera-projection fit (likely a hallucinated/very different frame)"
        )

    disp = np.hypot(dst[:, 0] - src[:, 0], dst[:, 1] - src[:, 1])
    median_disp = float(np.median(disp))
    max_disp = float(disp.max())
    rng = np.random.default_rng(_RANSAC_SEED)

    hmat, h_inliers = _ransac(
        src, dst, 4, _fit_homography, _reprojection_error, float(ransac_px), ransac_iters, rng
    )
    h_resid = float(np.median(_reprojection_error(hmat, src[h_inliers], dst[h_inliers])))
    h_frac = float(h_inliers.mean())
    shift, scale, rotation = _decompose_homography(hmat, w, h)

    fmat, f_inliers = _ransac(
        src,
        dst,
        _MIN_CORRESPONDENCES,
        _fit_fundamental,
        _symmetric_epipolar_distance,
        float(ransac_px),
        ransac_iters,
        rng,
    )
    f_dist = float(np.mean(_symmetric_epipolar_distance(fmat, src[f_inliers], dst[f_inliers])))
    f_frac = float(f_inliers.mean())
    # F is only identifiable with genuine parallax. It is NOT identifiable when the
    # motion is trivial (near-identity) OR when a single homography already explains
    # the correspondences — a pure pan / zoom / planar scene has zero baseline, so a
    # large displacement alone does not make the epipolar residual meaningful.
    homography_explains = h_frac >= 0.6 and h_resid < float(ransac_px)
    epipolar_degenerate = median_disp < float(consistent_px) or homography_explains

    if median_disp < float(consistent_px):
        verdict = "camera-consistent"
        reason = (
            f"median correspondence displacement {median_disp:.3g} px < {consistent_px} px — "
            "the generated camera matches the reference (no identifiable parallax, so the "
            "epipolar residual is not meaningful here)"
        )
    elif h_frac >= 0.6 and h_resid < float(ransac_px):
        verdict = "camera-drift"
        reason = (
            f"a single homography explains {h_frac:.0%} of correspondences at {h_resid:.3g} px "
            f"residual with a {shift:.3g} px centre shift / scale {scale:.4g} — the camera "
            "transform drifted (pan / zoom / FOV) but geometry stayed globally consistent"
        )
    else:
        verdict = "geometry-inconsistent"
        reason = (
            f"no global homography fits (inliers {h_frac:.0%}, residual {h_resid:.3g} px) — the "
            "correspondences disobey any single camera transform, i.e. local warping / "
            "hallucinated geometry"
        )

    return CameraProjectionResult(
        height=int(h),
        width=int(w),
        n_corners=n_corners,
        n_correspondences=m,
        median_displacement_px=median_disp,
        max_displacement_px=max_disp,
        homography_inlier_fraction=h_frac,
        homography_residual_px=h_resid,
        camera_shift_px=shift,
        camera_scale=scale,
        camera_rotation_deg=rotation,
        epipolar_inlier_fraction=f_frac,
        mean_symmetric_epipolar_distance_px=f_dist,
        epipolar_degenerate=bool(epipolar_degenerate),
        verdict=verdict,
        verdict_reason=reason,
    )

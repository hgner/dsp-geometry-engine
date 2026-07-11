"""Bidirectional Lucas-Kanade optical flow + forward-backward consistency.

Pure scipy (``ndimage``) + numpy — deliberately NOT OpenCV: this repo is
scipy+Pillow by policy, so this is a deterministic pyramidal Lucas-Kanade flow,
adequate for smooth generative video, not a calibrated OpenCV flow.

:func:`lucas_kanade` solves the windowed 2x2 normal equations per grid point over a
coarse-to-fine Gaussian pyramid (so a few-pixel motion is handled), guarding
rank-deficient (aperture) points by returning zero flow. :func:`forward_backward_residual`
runs the flow both directions per consecutive pair and measures the round-trip
residual ``||f(p) + b(p + f(p))||`` — a "hallucination map": where forward and
backward flow fail to cancel, the motion disobeys geometry (melting / drift).

Frame I/O is owned by :mod:`dsp_server.engine.video`; this module consumes the
``(T, H, W)`` volume it produces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# Below this minimum structure-tensor eigenvalue the 2x2 system is rank-deficient
# (aperture problem / flat patch): report zero flow rather than an amplified guess.
_EPS_EIG = 1e-4
_MAX_LEVELS = 3  # Gaussian-pyramid levels (coarse-to-fine)
_ITERS_PER_LEVEL = 5  # Newton refinements per level
_MIN_LEVEL_SIZE = 8  # do not downsample a pyramid level below this


@dataclass
class FBResult:
    """Forward-backward residual field over the analysis grid.

    ``residual_grid`` is (n_pairs, n_rows, n_cols) of ``||f + b(p+f)||`` in pixels;
    ``grid_x``/``grid_y`` are the pixel coordinates of the grid columns/rows.
    """

    residual_grid: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    n_pairs: int
    n_rows: int
    n_cols: int


def _levels_for(shape: tuple[int, int], requested: int) -> int:
    """Clamp the pyramid depth so the coarsest level stays >= _MIN_LEVEL_SIZE."""
    m = int(min(shape))
    levels = 1
    while levels < int(requested) and m // 2 >= _MIN_LEVEL_SIZE:
        m //= 2
        levels += 1
    return levels


def _pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """Gaussian pyramid, index 0 = finest (full resolution)."""
    pyr = [np.asarray(img, dtype=np.float64)]
    for _ in range(levels - 1):
        smoothed = ndimage.gaussian_filter(pyr[-1], sigma=1.0, mode="reflect")
        pyr.append(ndimage.zoom(smoothed, 0.5, order=1))
    return pyr


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample ``img`` at fractional (x, y), clamping to image bounds."""
    h, w = img.shape
    x = np.clip(x, 0.0, w - 1.0)
    y = np.clip(y, 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    v00 = img[y0, x0]
    v01 = img[y0, x1]
    v10 = img[y1, x0]
    v11 = img[y1, x1]
    return v00 * (1.0 - fx) * (1.0 - fy) + v01 * fx * (1.0 - fy) + v10 * (1.0 - fx) * fy + v11 * fx * fy


def _lk_point(
    l0: np.ndarray,
    l1: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    px: float,
    py: float,
    guess: np.ndarray,
    half: int,
) -> np.ndarray:
    """One point's flow at a single pyramid level (template-Jacobian LK, warped l1)."""
    h, w = l0.shape
    cx = int(round(px))
    cy = int(round(py))
    xs = np.clip(np.arange(cx - half, cx + half + 1), 0, w - 1)
    ys = np.clip(np.arange(cy - half, cy + half + 1), 0, h - 1)
    gx, gy = np.meshgrid(xs, ys)
    template = l0[gy, gx]
    iwx = ix[gy, gx]
    iwy = iy[gy, gx]
    sxx = float(np.sum(iwx * iwx))
    sxy = float(np.sum(iwx * iwy))
    syy = float(np.sum(iwy * iwy))
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = max(tr * tr - 4.0 * det, 0.0)
    eig_min = 0.5 * (tr - np.sqrt(disc))
    if eig_min < _EPS_EIG:  # rank-deficient: aperture problem / flat patch
        return np.zeros(2, dtype=np.float64)
    amat = np.array([[sxx, sxy], [sxy, syy]], dtype=np.float64)
    maxd = 0.5 * float(min(h, w))
    d = np.array(guess, dtype=np.float64)
    for _ in range(_ITERS_PER_LEVEL):
        warped = _bilinear(l1, gx + d[0], gy + d[1])
        it = warped - template
        rhs = np.array([-float(np.sum(iwx * it)), -float(np.sum(iwy * it))], dtype=np.float64)
        delta = np.linalg.solve(amat, rhs)
        d = np.clip(d + delta, -maxd, maxd)
        if float(np.hypot(delta[0], delta[1])) < 1e-3:
            break
    return d


def _patch_ssd(l0: np.ndarray, l1: np.ndarray, px: float, py: float, d: np.ndarray, half: int) -> float:
    """Mean-squared photometric error of warping ``l1`` by ``d`` onto ``l0``'s window."""
    h, w = l0.shape
    cx, cy = int(round(px)), int(round(py))
    xs = np.clip(np.arange(cx - half, cx + half + 1), 0, w - 1)
    ys = np.clip(np.arange(cy - half, cy + half + 1), 0, h - 1)
    gx, gy = np.meshgrid(xs, ys)
    warped = _bilinear(l1, gx + d[0], gy + d[1])
    diff = warped - l0[gy, gx]
    return float(np.mean(diff * diff))


def lucas_kanade(
    frame0: np.ndarray,
    frame1: np.ndarray,
    points: np.ndarray,
    window: int = 15,
    levels: int = _MAX_LEVELS,
) -> np.ndarray:
    """Pyramidal Lucas-Kanade flow at ``points`` (each ``(x, y)``) from frame0 to frame1.

    Returns an ``(N, 2)`` array of ``(u, v)`` displacements in pixels: ``frame1(p + flow)
    ~= frame0(p)``. Spatial gradients come from a Sobel operator on the template
    (frame0); the 2x2 normal equations are solved per point and refined over a
    coarse-to-fine Gaussian pyramid. Rank-deficient points (min structure-tensor
    eigenvalue < ``_EPS_EIG``) get zero flow.

    A coarse pyramid level can alias fine texture and inject a large spurious flow that
    the finer levels cannot escape (a local method's convergence basin). To stay robust
    on small motion, each point is ALSO solved from a zero guess at full resolution and
    the candidate with the lower photometric error is kept.
    """
    f0 = np.asarray(frame0, dtype=np.float64)
    f1 = np.asarray(frame1, dtype=np.float64)
    if f0.shape != f1.shape:
        raise ValueError(f"frames must match: {f0.shape} vs {f1.shape}")
    if f0.ndim != 2:
        raise ValueError(f"expected 2-D grayscale frames, got shape {f0.shape}")
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = pts.shape[0]
    n_levels = _levels_for(f0.shape, int(levels))
    pyr0 = _pyramid(f0, n_levels)
    pyr1 = _pyramid(f1, n_levels)
    half = int(window) // 2
    flow = np.zeros((n, 2), dtype=np.float64)
    for level in range(len(pyr0) - 1, -1, -1):
        scale = 0.5**level
        l0 = pyr0[level]
        l1 = pyr1[level]
        grad_x = ndimage.sobel(l0, axis=1, mode="reflect") / 8.0
        grad_y = ndimage.sobel(l0, axis=0, mode="reflect") / 8.0
        for j in range(n):
            d = _lk_point(l0, l1, grad_x, grad_y, pts[j, 0] * scale, pts[j, 1] * scale, flow[j] * scale, half)
            flow[j] = d / scale
    # Fine-level from-zero fallback: keep it where it matches better than the pyramid
    # result (fixes coarse-alias garbage on small motion; large motion keeps the pyramid).
    gx0 = ndimage.sobel(pyr0[0], axis=1, mode="reflect") / 8.0
    gy0 = ndimage.sobel(pyr0[0], axis=0, mode="reflect") / 8.0
    for j in range(n):
        d_zero = _lk_point(pyr0[0], pyr1[0], gx0, gy0, pts[j, 0], pts[j, 1], np.zeros(2), half)
        if _patch_ssd(pyr0[0], pyr1[0], pts[j, 0], pts[j, 1], d_zero, half) < _patch_ssd(
            pyr0[0], pyr1[0], pts[j, 0], pts[j, 1], flow[j], half
        ):
            flow[j] = d_zero
    return flow


def _grid_axes(shape: tuple[int, int], grid_step: int, half: int) -> tuple[np.ndarray, np.ndarray]:
    """Grid pixel coordinates with a ``half``-pixel margin so windows stay in-bounds."""
    h, w = shape
    xs = np.arange(half, max(half + 1, w - half), grid_step)
    ys = np.arange(half, max(half + 1, h - half), grid_step)
    if xs.size == 0:
        xs = np.array([w // 2])
    if ys.size == 0:
        ys = np.array([h // 2])
    return xs, ys


def _sample_field(
    field: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    x0: float,
    y0: float,
    step_x: float,
    step_y: float,
) -> np.ndarray:
    """Bilinearly sample an ``(n_rows, n_cols, 2)`` grid field at pixel points (qx, qy)."""
    n_rows, n_cols, _ = field.shape
    col = np.clip((qx - x0) / step_x, 0.0, n_cols - 1.0)
    row = np.clip((qy - y0) / step_y, 0.0, n_rows - 1.0)
    c0 = np.floor(col).astype(np.int64)
    r0 = np.floor(row).astype(np.int64)
    c1 = np.minimum(c0 + 1, n_cols - 1)
    r1 = np.minimum(r0 + 1, n_rows - 1)
    fc = (col - c0)[:, None]
    fr = (row - r0)[:, None]
    v00 = field[r0, c0]
    v01 = field[r0, c1]
    v10 = field[r1, c0]
    v11 = field[r1, c1]
    return v00 * (1.0 - fc) * (1.0 - fr) + v01 * fc * (1.0 - fr) + v10 * (1.0 - fc) * fr + v11 * fc * fr


def forward_backward_residual(frames: np.ndarray, grid_step: int = 16, window: int = 15) -> FBResult:
    """Forward-backward flow consistency over a regular grid, per consecutive pair.

    For each pair ``(i, i+1)``: forward flow ``f`` (i -> i+1) and backward flow ``b``
    (i+1 -> i) at the grid points, then ``residual = ||f(p) + b(p + f(p))||`` with the
    backward field bilinearly sampled at ``p + f``. A consistent (geometry-obeying)
    motion round-trips to near zero; a large residual marks hallucinated motion.
    """
    vol = np.asarray(frames, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"expected a (T, H, W) volume, got shape {vol.shape}")
    t, h, w = vol.shape
    if t < 2:
        raise ValueError(f"need >= 2 frames for a flow analysis, got {t}")
    half = int(window) // 2
    xs, ys = _grid_axes((h, w), int(grid_step), half)
    n_cols = int(xs.size)
    n_rows = int(ys.size)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel().astype(np.float64), gy.ravel().astype(np.float64)])
    step_x = float(xs[1] - xs[0]) if n_cols > 1 else 1.0
    step_y = float(ys[1] - ys[0]) if n_rows > 1 else 1.0
    x0 = float(xs[0])
    y0 = float(ys[0])

    n_pairs = t - 1
    residual = np.zeros((n_pairs, n_rows, n_cols), dtype=np.float64)
    for i in range(n_pairs):
        f = lucas_kanade(vol[i], vol[i + 1], pts, window)
        b = lucas_kanade(vol[i + 1], vol[i], pts, window)
        b_field = b.reshape(n_rows, n_cols, 2)
        target = pts + f
        b_samp = _sample_field(b_field, target[:, 0], target[:, 1], x0, y0, step_x, step_y)
        res_vec = f + b_samp
        residual[i] = np.hypot(res_vec[:, 0], res_vec[:, 1]).reshape(n_rows, n_cols)

    return FBResult(
        residual_grid=residual,
        grid_x=xs.astype(np.float64),
        grid_y=ys.astype(np.float64),
        n_pairs=n_pairs,
        n_rows=n_rows,
        n_cols=n_cols,
    )

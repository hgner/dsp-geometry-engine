"""Photometric (intrinsic-shading) consistency for the video comparison gate.

Rendering forms an image as ``I = R * S`` — reflectance (albedo) times shading,
where shading depends on the surface normal ``N`` and the light ``L``. A generative
model routinely bakes lighting into texture and produces shading that does NOT obey
the true surface geometry. Given the engine's ground-truth NORMAL pass (and,
optionally, its ALBEDO pass), this module recovers the observed shading of a
generated frame and fits the Lambertian model ``S ~ ambient + N.L`` over all
(masked) pixels by linear least squares. The fit's ``R^2`` is the verdict: shading
the true normals explain well (high ``R^2``) obeys the geometry; shading they
cannot explain (low ``R^2``) is hallucinated lighting. With an albedo pass it also
reports the albedo->shading gradient LEAK — texture bleeding into the shading
channel, i.e. a failed intrinsic decomposition.

This is a linear Lambertian PROXY: it does not model the ``max(0, .)`` light clamp
or cast shadows, and the recovered light lives in whatever coordinate frame the
normal map uses. It is a consistency GATE, not a photometric-stereo light solver.
Pure numpy least squares; frame I/O is owned by :mod:`dsp_server.engine.image2d`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

__all__ = [
    "PhotometricResult",
    "analyze_photometric_consistency",
    "decode_normal_map",
]

_TINY = 1e-12
_ALBEDO_FLOOR = 0.05  # albedo below this is too dark to divide out reliably; pixel dropped


@dataclass
class PhotometricResult:
    """Photometric consistency of a generated frame vs the engine normal/albedo pass.

    ``photometric_r2`` is the coefficient of determination of the Lambertian fit
    ``S ~ ambient + N.L`` (can be negative when shading anti-correlates with the
    normals). ``light_direction`` / ``ambient`` / ``light_elevation_deg`` /
    ``light_azimuth_deg`` describe the recovered light in the normal map's frame.
    ``albedo_shading_leak`` (albedo mode only) is the Pearson correlation of
    ``|grad shading|`` with ``|grad albedo|`` — high means texture leaked into the
    shading channel. ``verdict`` is ``consistent`` or ``inconsistent`` (optionally
    with an ``albedo leak`` note).
    """

    n_pixels: int
    used_albedo: bool
    photometric_r2: float
    residual_rms: float
    light_direction: list[float]
    ambient: float
    light_elevation_deg: float
    light_azimuth_deg: float
    albedo_shading_leak: float | None
    verdict: str
    verdict_reason: str


def decode_normal_map(rgb: np.ndarray) -> np.ndarray:
    """Decode an ``(h, w, 3)`` normal map in [0, 1] to unit normals in [-1, 1].

    Standard packing ``N = 2*rgb - 1``; each pixel is renormalized to unit length
    (a degenerate zero vector maps to ``[0, 0, 1]``)."""
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected an (h, w, 3) normal map, got shape {arr.shape}")
    n = 2.0 * arr[:, :, :3] - 1.0
    norm = np.sqrt((n * n).sum(axis=2, keepdims=True))
    unit = np.where(norm < 1e-8, np.array([0.0, 0.0, 1.0]), n / np.maximum(norm, _TINY))
    return unit


def _grad_mag(img: np.ndarray) -> np.ndarray:
    return np.hypot(ndimage.sobel(img, axis=0, mode="reflect"), ndimage.sobel(img, axis=1, mode="reflect"))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    return float((a * b).sum() / denom) if denom > _TINY else 0.0


def analyze_photometric_consistency(
    generated_lum: np.ndarray,
    normals: np.ndarray,
    albedo_lum: np.ndarray | None = None,
    r2_ok: float = 0.5,
    leak_warn: float = 0.6,
) -> PhotometricResult:
    """Fit ``S ~ ambient + N.L`` on a generated frame's shading and verdict it.

    ``generated_lum`` and ``albedo_lum`` are luminance (2-D); ``normals`` is an
    ``(h, w, 3)`` unit-normal field (see :func:`decode_normal_map`). With an albedo
    pass the shading is ``S = I / albedo`` (dark-albedo pixels dropped); without
    one, ``S = I`` directly. Pixels that are non-finite (or too-dark albedo) are
    excluded from the fit and the leak correlation.
    """
    lum = np.asarray(generated_lum, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64)
    if lum.ndim != 2:
        raise ValueError(f"generated luminance must be 2-D, got shape {lum.shape}")
    if nrm.ndim != 3 or nrm.shape[2] < 3 or nrm.shape[:2] != lum.shape:
        raise ValueError(f"normals must be (h, w, 3) matching the frame {lum.shape}, got shape {nrm.shape}")

    used_albedo = albedo_lum is not None
    valid = np.ones(lum.shape, dtype=bool)
    if used_albedo:
        alb = np.asarray(albedo_lum, dtype=np.float64)
        if alb.shape != lum.shape:
            raise ValueError(f"albedo {alb.shape} must match the frame {lum.shape}")
        valid = alb >= _ALBEDO_FLOOR
        shading = np.divide(lum, np.maximum(alb, _ALBEDO_FLOOR))
    else:
        shading = lum
    valid &= np.isfinite(shading)
    if int(valid.sum()) < 16:
        raise ValueError(
            "too few valid pixels for a photometric fit (albedo too dark or frame non-finite) — "
            "supply a brighter albedo pass or drop the albedo argument"
        )

    n_flat = nrm[:, :, :3][valid]  # (M, 3)
    s_flat = shading[valid]  # (M,)
    amat = np.column_stack([np.ones(n_flat.shape[0]), n_flat[:, 0], n_flat[:, 1], n_flat[:, 2]])
    coef, *_ = np.linalg.lstsq(amat, s_flat, rcond=None)
    pred = amat @ coef
    resid = s_flat - pred
    ss_res = float((resid * resid).sum())
    ss_tot = float(((s_flat - s_flat.mean()) ** 2).sum())
    # OLS gives ss_res <= ss_tot, so constant shading (ss_tot ~ 0 => ss_res ~ 0) is a
    # PERFECT fit by the ambient term alone: R^2 -> 1.0 (there is no shading variation
    # to contradict the geometry), not 0.0 (which would falsely read "inconsistent").
    r2 = 1.0 - ss_res / ss_tot if ss_tot > _TINY else 1.0
    residual_rms = float(np.sqrt(ss_res / s_flat.size))

    ambient = float(coef[0])
    light_vec = coef[1:]
    light_norm = float(np.linalg.norm(light_vec))
    direction = (light_vec / light_norm) if light_norm > _TINY else np.array([0.0, 0.0, 1.0])
    elevation = float(np.degrees(np.arcsin(np.clip(direction[2], -1.0, 1.0))))
    azimuth = float(np.degrees(np.arctan2(direction[1], direction[0])))

    leak: float | None = None
    if used_albedo:
        # Correlate the gradient magnitudes over the VALID pixels only: dark-albedo
        # regions carry a floor-clamped shading artifact whose gradients would
        # otherwise contaminate the leak (and its docstring promises the mask).
        leak = _pearson(_grad_mag(shading)[valid], _grad_mag(alb)[valid])

    if r2 >= float(r2_ok):
        verdict = "consistent"
        reason = (
            f"shading is well explained by the true normals (R^2 = {r2:.3g} >= {r2_ok}) — the "
            "lighting obeys the surface geometry"
        )
    else:
        verdict = "inconsistent"
        reason = (
            f"shading is NOT explained by the true normals (R^2 = {r2:.3g} < {r2_ok}) — the "
            "lighting does not obey the surface geometry (hallucinated / baked-in shading)"
        )
    if leak is not None and leak >= float(leak_warn):
        verdict = f"{verdict} (albedo leak)"
        reason += (
            f"; albedo->shading gradient leak {leak:.3g} >= {leak_warn} — texture is bleeding "
            "into the shading channel (failed intrinsic decomposition)"
        )

    return PhotometricResult(
        n_pixels=int(s_flat.size),
        used_albedo=used_albedo,
        photometric_r2=float(r2),
        residual_rms=residual_rms,
        light_direction=[float(v) for v in direction],
        ambient=ambient,
        light_elevation_deg=elevation,
        light_azimuth_deg=azimuth,
        albedo_shading_leak=leak,
        verdict=verdict,
        verdict_reason=reason,
    )

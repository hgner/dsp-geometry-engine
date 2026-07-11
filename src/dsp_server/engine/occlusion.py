"""Occlusion-boundary sharpness for the video comparison gate.

A generative model has no Z-buffer, so at occlusion boundaries — where a foreground
object crosses a background one — the foreground and background pixels bleed
together. The engine's DEPTH pass marks those boundaries exactly: a depth
discontinuity is a sharp jump in Z. This module takes the Sobel gradient magnitude
of the depth pass, thresholds it at a high percentile to a boundary band, and
measures the variance of the Laplacian (the standard focus / sharpness proxy) of a
generated frame INSIDE that band versus the flat interior. A crisp frame is sharper
at the true edges than in the interior; a melting frame is not. With a reference
render it reports the gen/ref boundary-sharpness ratio (< 1 = the generator softened
the occlusion edges) — the reliable mode, since variance-of-Laplacian is
content-dependent.

Pure scipy.ndimage. Frame I/O is owned by :mod:`dsp_server.engine.image2d`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

__all__ = [
    "OcclusionResult",
    "depth_edge_band",
    "evaluate_occlusion_boundaries",
]

_TINY = 1e-12


@dataclass
class OcclusionResult:
    """Occlusion-boundary sharpness of a generated frame against a depth pass.

    ``boundary_sharpness`` / ``interior_sharpness`` are the variance of the frame's
    Laplacian inside the depth-edge band and in the interior; their ratio > 1 means
    edges are crisper than flat regions (expected for real occlusions). When a
    reference render is supplied, ``gen_vs_ref_ratio`` (gen boundary sharpness over
    ref boundary sharpness) is the reliable read: < ``ref_ok`` => the generator
    blurred the occlusion edges. ``verdict`` is ``sharp`` or ``soft-boundaries``;
    ``edge_band`` is the boolean band mask (written to disk by the tool layer).
    """

    height: int
    width: int
    edge_percentile: float
    band_px: int
    edge_pixel_count: int
    band_pixel_count: int
    edge_fraction: float
    boundary_sharpness: float
    interior_sharpness: float
    boundary_sharpness_ratio: float
    ref_boundary_sharpness: float | None
    gen_vs_ref_ratio: float | None
    verdict: str
    verdict_reason: str
    edge_band: np.ndarray


def depth_edge_band(
    depth: np.ndarray, percentile: float = 95.0, band_px: int = 2, min_range_frac: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean ``(band, edges)`` masks of depth DISCONTINUITIES.

    ``edges`` are pixels whose depth Sobel-gradient magnitude clears BOTH the
    ``percentile`` of the positive gradient AND an absolute floor of ``min_range_frac``
    of the depth's dynamic range. The floor is what distinguishes a true occlusion
    jump from a pervasive SMOOTH gradient (a receding ground plane, a tilted wall):
    a percentile alone floods the whole frame when the gradient is roughly uniform,
    collapsing the interior and faking a "sharp" verdict. ``band`` dilates ``edges``
    by ``band_px`` so the sharpness window straddles the boundary. A flat depth pass —
    or one with only smooth gradients (no localized jump) — yields empty masks.
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"expected a 2-D depth pass, got shape {d.shape}")
    grad = np.hypot(ndimage.sobel(d, axis=0, mode="reflect"), ndimage.sobel(d, axis=1, mode="reflect"))
    positive = grad[grad > 0.0]
    if positive.size == 0:  # perfectly flat depth: no occlusion boundaries
        empty = np.zeros(d.shape, dtype=bool)
        return empty, empty
    # The Sobel magnitude is ~8x the per-pixel depth step, so a smooth ramp over W px
    # sits near (8/W)*range (a few % of the range) while a real discontinuity is several
    # x the range. The floor at min_range_frac*range rejects the ramp, keeps the jump.
    depth_range = float(d.max() - d.min())
    floor = float(min_range_frac) * depth_range
    thresh = max(float(np.percentile(positive, float(percentile))), floor, _TINY)
    edges = grad >= thresh
    if int(band_px) > 0:
        band = ndimage.binary_dilation(edges, iterations=int(band_px))
    else:
        band = edges
    return band, edges


def _laplacian_variance(img: np.ndarray, mask: np.ndarray) -> float:
    """Variance of ``img``'s Laplacian over ``mask`` (0.0 if the mask is empty)."""
    if not mask.any():
        return 0.0
    lap = ndimage.laplace(np.asarray(img, dtype=np.float64))
    return float(np.var(lap[mask]))


def evaluate_occlusion_boundaries(
    generated: np.ndarray,
    depth: np.ndarray,
    reference: np.ndarray | None = None,
    percentile: float = 95.0,
    band_px: int = 2,
    ref_ok: float = 0.6,
    interior_ratio_ok: float = 1.0,
) -> OcclusionResult:
    """Measure generated-frame sharpness at the depth pass's occlusion boundaries.

    The verdict is ``sharp`` / ``soft-boundaries``. With a ``reference`` it is
    driven by ``gen_vs_ref_ratio`` (>= ``ref_ok`` is sharp); without one it falls
    back to the interior ratio (edges expected at least ``interior_ratio_ok`` times
    as sharp as flat interior) — a weaker, content-dependent read, so pass a
    reference when possible.
    """
    gen = np.asarray(generated, dtype=np.float64)
    d = np.asarray(depth, dtype=np.float64)
    if gen.ndim != 2:
        raise ValueError(f"generated frame must be 2-D grayscale, got shape {gen.shape}")
    if d.shape != gen.shape:
        raise ValueError(f"depth pass {d.shape} must match the frame {gen.shape}")
    band, edges = depth_edge_band(d, percentile=percentile, band_px=band_px)
    if not band.any():
        raise ValueError(
            "the depth pass has no discontinuities (flat depth) — no occlusion boundaries to "
            "evaluate; check that a real depth/Z pass was supplied"
        )
    interior = ~band
    h, w = gen.shape

    boundary_sharp = _laplacian_variance(gen, band)
    interior_sharp = _laplacian_variance(gen, interior)
    ratio = boundary_sharp / max(interior_sharp, _TINY)

    ref_sharp: float | None = None
    gen_vs_ref: float | None = None
    if reference is not None:
        ref = np.asarray(reference, dtype=np.float64)
        if ref.shape != gen.shape:
            raise ValueError(f"reference {ref.shape} must match the frame {gen.shape}")
        ref_sharp = _laplacian_variance(ref, band)
        gen_vs_ref = boundary_sharp / max(ref_sharp, _TINY)

    if gen_vs_ref is not None:
        if gen_vs_ref >= float(ref_ok):
            verdict = "sharp"
            reason = (
                f"generated boundary sharpness is {gen_vs_ref:.0%} of the reference "
                f"(>= {ref_ok:.0%}) — occlusion edges are preserved"
            )
        else:
            verdict = "soft-boundaries"
            reason = (
                f"generated boundary sharpness is only {gen_vs_ref:.0%} of the reference "
                f"(< {ref_ok:.0%}) — the generator softened the occlusion edges (depth-layer "
                "bleeding)"
            )
    elif ratio >= float(interior_ratio_ok):
        verdict = "sharp"
        reason = (
            f"boundary/interior sharpness ratio {ratio:.3g} >= {interior_ratio_ok} — edges are "
            "crisper than flat interior (no reference given; this read is content-dependent)"
        )
    else:
        verdict = "soft-boundaries"
        reason = (
            f"boundary/interior sharpness ratio {ratio:.3g} < {interior_ratio_ok} — occlusion "
            "edges are no sharper than flat interior (no reference given; pass one for a firm read)"
        )

    return OcclusionResult(
        height=int(h),
        width=int(w),
        edge_percentile=float(percentile),
        band_px=int(band_px),
        edge_pixel_count=int(edges.sum()),
        band_pixel_count=int(band.sum()),
        edge_fraction=float(edges.mean()),
        boundary_sharpness=boundary_sharp,
        interior_sharpness=interior_sharp,
        boundary_sharpness_ratio=float(ratio),
        ref_boundary_sharpness=ref_sharp,
        gen_vs_ref_ratio=gen_vs_ref,
        verdict=verdict,
        verdict_reason=reason,
        edge_band=band,
    )

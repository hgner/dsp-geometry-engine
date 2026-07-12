"""Perceptual / semantic tool pack — the FR-VQA layer (classical, pure-scipy).

Two tools that bridge MATHEMATICAL parity (what the DSP/geometry gates measure) and
PERCEPTUAL parity (what a human eye accepts):

* ``evaluate_perceptual_similarity`` — Complex-Wavelet SSIM (shift-tolerant) + MS-SSIM
  + pixel SSIM of a reference vs a generated frame. The tie-breaker: when the FFT /
  wavelet tools flag a huge error but CW-SSIM is high, the difference is a benign
  small shift, not a real distortion.
* ``verify_identity_coherence`` — cosine drift of a masked object's colour+texture
  signature across a clip: the exact frame an object MORPHS (a jacket's material
  identity flips), which optical flow — tracking where pixels move, not what they are
  — cannot see.

Classical stand-ins for LPIPS / DINOv2: no torch, no downloaded weights — the server
stays a pure-wheel, deterministic, cloud/cron-safe measurement product. Same contract
as the other packs: JSON summaries only (arrays live on disk under data/perceptual/),
ToolError envelope on failure, module-level ``_impl`` the tests call directly.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP

from dsp_server.engine import image2d, perceptual, video
from dsp_server.schemas import SchemaBase, ToolError
from dsp_server.toolsets import AppContext

_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]")

_SIMILARITY_HINT = (
    "reference and generated must be readable, identically shaped single frames "
    "(the reference is the engine render, ground truth)"
)
_COHERENCE_HINT = (
    "frames is a directory of same-size frames or an explicit path list (>= 2 frames); "
    "mask is a single-channel image (nonzero = the tracked object), same H x W as the frames"
)


def _safe(text: str) -> str:
    return _SAFE_RE.sub("_", text) or "x"


def _error(exc: Exception, hint: str) -> str:
    return ToolError(error=f"{type(exc).__name__}: {exc}", hint=hint).model_dump_json()


def _perceptual_dir(ctx: AppContext) -> Path:
    d = ctx.data_dir / "perceptual"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _source_stem(source: str | list[str]) -> str:
    if isinstance(source, list):
        base = Path(source[0]).parent.name if source else "framelist"
        return _safe(base or "framelist")
    p = Path(source)
    return _safe(p.name if p.is_dir() else p.stem)


def _source_label(source: str | list[str]) -> str:
    return f"[{len(source)} frames]" if isinstance(source, list) else str(source)


# --------------------------------------------------------------------------- #
# Response models


class PerceptualSimilarityReport(SchemaBase):
    """evaluate_perceptual_similarity: shift-tolerant CW-SSIM (+ MS-SSIM + pixel SSIM)."""

    reference: str
    generated: str
    height: int
    width: int
    cw_ssim: float
    ms_ssim: float
    ssim: float
    shift_tolerant_gap: float
    verdict: str
    verdict_reason: str


class IdentityCoherenceReport(SchemaBase):
    """verify_identity_coherence: per-frame colour+texture drift of a masked object."""

    frames: str
    mask: str
    n_frames: int
    mask_pixel_count: int
    break_threshold: float
    mean_coherence: float
    min_coherence: float
    min_coherence_frame: int
    first_break_frame: int
    verdict: str
    verdict_reason: str
    coherence_path: str
    n_dropped: int = 0


# --------------------------------------------------------------------------- #
# Tool A: evaluate_perceptual_similarity


def _evaluate_perceptual_similarity(
    ctx: AppContext,
    reference: str,
    generated: str,
    equivalent_cw: float = 0.85,
    distinct_cw: float = 0.55,
) -> str:
    try:
        ref = image2d.load_image_gray(reference)
        gen = image2d.load_image_gray(generated)
        if ref.shape != gen.shape:
            return ToolError(
                error=(
                    f"frame shapes differ: {Path(reference).name} is {ref.shape}, "
                    f"{Path(generated).name} is {gen.shape}"
                ),
                hint=_SIMILARITY_HINT,
            ).model_dump_json()
        res = perceptual.analyze_perceptual_similarity(
            ref, gen, equivalent_cw=float(equivalent_cw), distinct_cw=float(distinct_cw)
        )
        return PerceptualSimilarityReport(
            reference=str(reference),
            generated=str(generated),
            height=res.height,
            width=res.width,
            cw_ssim=res.cw_ssim,
            ms_ssim=res.ms_ssim,
            ssim=res.ssim,
            shift_tolerant_gap=res.shift_tolerant_gap,
            verdict=res.verdict,
            verdict_reason=res.verdict_reason,
        ).model_dump_json()
    except Exception as exc:  # the MCP boundary must never see a raise
        return _error(exc, _SIMILARITY_HINT)


# --------------------------------------------------------------------------- #
# Tool B: verify_identity_coherence


def _verify_identity_coherence(
    ctx: AppContext,
    frames: str | list[str],
    mask: str,
    break_threshold: float = 0.8,
    mask_threshold: float = 0.0,
    max_frames: int | None = None,
) -> str:
    try:
        stack = video.load_rgb_frame_stack(frames, max_frames=max_frames)
        mask_img = image2d.load_image_gray(mask) > float(mask_threshold)
        if mask_img.shape != stack.frames.shape[1:3]:
            return ToolError(
                error=(f"mask {mask_img.shape} must match the frame {stack.frames.shape[1:3]} (H x W)"),
                hint=_COHERENCE_HINT,
            ).model_dump_json()
        res = perceptual.identity_coherence(stack.frames, mask_img, break_threshold=float(break_threshold))
        outdir = _perceptual_dir(ctx)
        npz_path = outdir / f"coherence_{_source_stem(frames)}.npz"
        np.savez(npz_path, coherence_by_frame=res.coherence_by_frame)
        return IdentityCoherenceReport(
            frames=_source_label(frames),
            mask=str(mask),
            n_frames=res.n_frames,
            mask_pixel_count=res.mask_pixel_count,
            break_threshold=res.break_threshold,
            mean_coherence=res.mean_coherence,
            min_coherence=res.min_coherence,
            min_coherence_frame=res.min_coherence_frame,
            first_break_frame=res.first_break_frame,
            verdict=res.verdict,
            verdict_reason=res.verdict_reason,
            coherence_path=str(npz_path),
            n_dropped=stack.n_dropped,
        ).model_dump_json()
    except Exception as exc:  # the MCP boundary must never see a raise
        return _error(exc, _COHERENCE_HINT)


# --------------------------------------------------------------------------- #
# Registration


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the perceptual tools on ``mcp``, bound to ``ctx``."""

    @mcp.tool()
    def evaluate_perceptual_similarity(
        reference: str,
        generated: str,
        equivalent_cw: float = 0.85,
        distinct_cw: float = 0.55,
    ) -> str:
        """PERCEPTUAL tie-breaker: does a generated frame LOOK like the reference to a human, even
        where the pixel/FFT math diverges? Reports Complex-Wavelet SSIM (cw_ssim, the headline) — a
        classical shift-tolerant metric computed over a complex Gabor bank, so a few-pixel texture
        shift (a huge FFT/wavelet error) still scores ~1 — alongside MS-SSIM (multi-scale
        structural) and single-scale pixel SSIM. shift_tolerant_gap = cw_ssim - ssim is large
        exactly when the frames differ mainly by a benign spatial shift (mathematically divergent,
        perceptually equivalent). verdict: 'perceptually-equivalent' (cw_ssim >= equivalent_cw),
        'distinct' (< distinct_cw), else 'perceptually-degraded'. This is a CLASSICAL stand-in for a
        neural LPIPS metric (no torch/weights) — honest about being pure-scipy. Use it as the
        tie-breaker: when compare_depth_renders / compare_wavelet_signatures flag a big error but
        cw_ssim is high, the generator preserved the perceptual intent. reference and generated must
        be identically shaped single frames. Returns PerceptualSimilarityReport JSON, or ToolError
        JSON on failure."""
        return _evaluate_perceptual_similarity(ctx, reference, generated, equivalent_cw, distinct_cw)

    @mcp.tool()
    def verify_identity_coherence(
        frames: str | list[str],
        mask: str,
        break_threshold: float = 0.8,
        mask_threshold: float = 0.0,
        max_frames: int | None = None,
    ) -> str:
        """MORPHING gate: does a tracked object keep its identity across a clip, or does it morph
        (a leather jacket becomes a plastic raincoat)? Optical flow tracks WHERE pixels move;
        this tracks WHAT they are. Given a frame sequence (a directory or path list) and an object
        mask from the engine (an ID map / segmentation pass — a pixel > mask_threshold, default 0 so
        ANY nonzero pixel, is the object; raise it to ignore faint/antialiased edges), it extracts a
        colour+texture signature of the masked region in each frame (per-channel colour histograms +
        Gabor-magnitude texture features) and reports the cosine similarity of every frame's
        signature to frame 0. A drop below break_threshold marks a morph; min_coherence_frame
        localizes the worst change and first_break_frame is the exact frame the identity broke.
        verdict: 'coherent' (holds), 'identity-drift' (partial), or 'morph' (a hard colour/texture
        flip). A CLASSICAL stand-in for a neural DINOv2 semantic tracker (no torch/weights) — it
        flags gross colour/texture/shape morphing, not fine semantic identity, and uses a FIXED mask
        region (supply a per-frame region or pair with verify_motion_consistency if the object moves
        a lot). The full per-frame coherence array is written to an npz under data/perceptual/.
        Returns IdentityCoherenceReport JSON, or ToolError JSON on failure."""
        return _verify_identity_coherence(ctx, frames, mask, break_threshold, mask_threshold, max_frames)

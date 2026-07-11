"""Imaging tool pack (ELE407 week-14 2-D lane): depth/AOV render cross-validation.

One tool, compare_depth_renders, operating strictly on PNGs already on disk (the
renders are produced outside the MCP call, or via the optional M8 render bridge) —
it never touches the GPU. Same contract as the geometry pack: JSON summaries only,
ToolError envelope on failure, module-level ``_impl`` for direct test calls.
"""

from __future__ import annotations

import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from dsp_server import plots
from dsp_server.engine import image2d
from dsp_server.schemas import ImageComparisonReport, Spectrum2DSchema, ToolError
from dsp_server.toolsets import AppContext

_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]")

_ROI_HINT = (
    "images must be readable single-channel (grayscale/depth) images; "
    "roi is [x0, y0, x1, y1] in pixels with exclusive upper bounds"
)


def _safe(text: str) -> str:
    return _SAFE_RE.sub("_", text) or "x"


def _roi_tuple(roi: list[int] | None) -> image2d.Roi | None:
    if roi is None:
        return None
    if len(roi) != 4:
        raise ValueError(f"roi must be [x0, y0, x1, y1], got {roi}")
    return (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))


def _spec_schema(spec: image2d.Spectrum2D) -> Spectrum2DSchema:
    freq = spec.dominant_freq_cpp
    return Spectrum2DSchema(
        dominant_freq_cpp=freq,
        dominant_wavelength_px=1.0 / freq if freq > 0.0 else 0.0,
        dominant_orientation_deg=spec.dominant_orientation_deg,
        dominant_prominence_db=spec.dominant_prominence_db,
    )


def _compare_depth_renders(
    ctx: AppContext,
    image_a: str,
    image_b: str | None = None,
    roi: list[int] | None = None,
    save_plot: bool = False,
) -> str:
    try:
        roi_t = _roi_tuple(roi)
        if image_b is None:
            img = image2d.load_image_gray(image_a)
            patch = image2d.crop_roi(img, roi_t) if roi_t is not None else img
            spec = image2d.roi_spectrum_2d(patch)
            plot_path: str | None = None
            if save_plot:
                out = ctx.plots_dir / f"spectrum2d_{_safe(Path(image_a).stem)}.png"
                plot_path = str(plots.save_spectrum2d_plot(spec.psd, out, Path(image_a).name))
            return ImageComparisonReport(
                image_a=str(image_a),
                image_b=None,
                roi=list(roi_t) if roi_t is not None else None,
                height=int(patch.shape[0]),
                width=int(patch.shape[1]),
                spectrum_a=_spec_schema(spec),
                plot_path=plot_path,
            ).model_dump_json()

        comparison = image2d.compare_images(image_a, image_b, roi=roi_t)
        plot_path = None
        if save_plot:
            name = f"spectrum2d_{_safe(Path(image_a).stem)}_vs_{_safe(Path(image_b).stem)}.png"
            plot_path = str(
                plots.save_spectrum2d_plot(
                    comparison.spectrum_a.psd,
                    ctx.plots_dir / name,
                    f"{Path(image_a).name} vs {Path(image_b).name}",
                    psd_b=comparison.spectrum_b.psd,
                )
            )
        return ImageComparisonReport(
            image_a=comparison.path_a,
            image_b=comparison.path_b,
            roi=list(roi_t) if roi_t is not None else None,
            height=comparison.shape[0],
            width=comparison.shape[1],
            spectrum_a=_spec_schema(comparison.spectrum_a),
            spectrum_b=_spec_schema(comparison.spectrum_b),
            mae=comparison.mae,
            rms=comparison.rms,
            ssim=comparison.ssim,
            plot_path=plot_path,
        ).model_dump_json()
    except Exception as exc:  # the MCP boundary must never see a raise
        return ToolError(error=f"{type(exc).__name__}: {exc}", hint=_ROI_HINT).model_dump_json()


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the imaging tools on ``mcp``, bound to ``ctx``."""

    @mcp.tool()
    def compare_depth_renders(
        image_a: str,
        image_b: str | None = None,
        roi: list[int] | None = None,
        save_plot: bool = False,
    ) -> str:
        """2-D spectral cross-validation on depth/AOV renders already on disk (never blocks on a
        GPU). With image_a alone: windowed 2-D FFT over roi (or the whole image) reporting the
        dominant spatial frequency in cycles/pixel (plus its wavelength in pixels), its
        orientation in degrees (0 = ripple varying along x, i.e. vertical stripes), and its
        prominence in dB — a periodic mesh corrugation shows an oriented spectral ridge. With
        image_b as well: additionally MAE/RMS/SSIM difference stats on the [0,1]-normalized
        images and a spectral summary per image. roi is [x0, y0, x1, y1] in pixels (exclusive
        upper bounds, same crop on both images); save_plot writes the log-power spectrum image(s)
        and returns the path. Returns ImageComparisonReport JSON, or ToolError JSON on failure."""
        return _compare_depth_renders(ctx, image_a, image_b, roi, save_plot)

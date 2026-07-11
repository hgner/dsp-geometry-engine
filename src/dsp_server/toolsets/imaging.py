"""Imaging tool pack (ELE407 week-14 2-D lane + ELE490 image processing).

Five tools operating strictly on images already on disk (never the GPU):
compare_depth_renders (2-D spectral cross-validation, unchanged) plus the ELE490
Gonzalez & Woods lanes — enhance_image (intensity transformations), filter_image
(spatial + frequency-domain filtering), segment_image (thresholding + morphology
+ connected components: THE defect-mask tool), restore_image (honest-scipy
restoration). Processed outputs are 16-bit grayscale PNGs under data/images/
(masks are 8-bit {0, 255}), so every tool's output is a valid input to every
other imaging tool. Same contract as the geometry pack: JSON summaries only,
ToolError envelope on failure, module-level ``_impl`` for direct test calls.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP

from dsp_server import plots
from dsp_server.engine import image2d
from dsp_server.schemas import ImageComparisonReport, SchemaBase, Spectrum2DSchema, ToolError
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


# --------------------------------------------------------------------------- #
# ELE490 imaging v2 — allowed modes, hints, pack-local output dir


_ENHANCE_MODES = ("histogram_eq", "adaptive_eq", "gamma", "log")
_FILTER_MODES = ("gaussian", "median", "unsharp", "sobel", "butterworth_lp", "butterworth_hp")
_SEGMENT_MODES = ("otsu", "adaptive", "kmeans")
_RESTORE_MODES = ("wiener", "wiener_deconv", "richardson_lucy")
_FOREGROUNDS = ("bright", "dark")

_IMG_HINT = (
    "image must be a readable single-channel (grayscale/depth) image on disk; "
    "processed outputs are 16-bit grayscale PNGs under data/images/ and are valid "
    "inputs to every imaging tool"
)


def _images_dir(ctx: AppContext) -> Path:
    """Pack-local output dir under data/ (critique #15: AppContext stays frozen)."""
    d = ctx.data_dir / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _out_name(image: str, mode: str, out_label: str | None) -> str:
    """Deterministic output name — re-runs overwrite, same as dump labels."""
    label = f"__{_safe(out_label)}" if out_label else ""
    return f"{_safe(Path(image).stem)}__{mode}{label}.png"


def _unknown(kind: str, value: str, allowed: tuple[str, ...]) -> str:
    return ToolError(
        error=f"unknown {kind} '{value}'", hint=f"expected one of {list(allowed)}"
    ).model_dump_json()


def _error(exc: Exception, hint: str) -> str:
    return ToolError(error=f"{type(exc).__name__}: {exc}", hint=hint).model_dump_json()


# --------------------------------------------------------------------------- #
# Response models (pack-local, on the shared SchemaBase contract)


class HistStatsSchema(SchemaBase):
    """1:1 mirror of :class:`dsp_server.engine.image2d.HistStats`."""

    mean: float
    std: float
    p01: float
    p50: float
    p99: float
    entropy_bits: float


class ImageEnhanceReport(SchemaBase):
    """enhance_image: intensity transformation + before/after histogram stats."""

    image: str
    mode: str
    params: dict[str, float]
    output_path: str
    height: int
    width: int
    stats_before: HistStatsSchema
    stats_after: HistStatsSchema
    delta_entropy_bits: float
    plot_path: str | None = None


class ImageFilterReport(SchemaBase):
    """filter_image: spatial/frequency-domain filtering + change stats."""

    image: str
    mode: str
    params: dict[str, float]
    output_path: str
    height: int
    width: int
    stats_before: HistStatsSchema
    stats_after: HistStatsSchema
    rms_change: float
    ssim_vs_input: float
    edge_density: float | None = None


class ComponentSchema(SchemaBase):
    """One connected component; bbox = [x0, y0, x1, y1], exclusive upper bounds,
    ALWAYS in full-image pixel coordinates (roi offset re-added)."""

    label: int
    area_px: int
    bbox: list[int]
    centroid_xy: list[float]
    mean_intensity: float


class SegmentationReport(SchemaBase):
    """segment_image: threshold + morphology + connected-component table."""

    image: str
    mode: str
    foreground: str
    threshold: float | None = None
    kmeans_centers: list[float] | None = None
    morph: str
    morph_size: int
    roi: list[int] | None = None
    mask_path: str
    height: int
    width: int
    foreground_fraction: float
    n_components_total: int
    components: list[ComponentSchema]
    components_truncated: bool


class RestorationReport(SchemaBase):
    """restore_image: restoration + sharpness delta so the caller sees whether it helped."""

    image: str
    mode: str
    params: dict[str, float]
    output_path: str
    height: int
    width: int
    stats_before: HistStatsSchema
    stats_after: HistStatsSchema
    sharpness_before: float
    sharpness_after: float
    rms_change: float
    ssim_vs_input: float


def _hist_schema(stats: image2d.HistStats) -> HistStatsSchema:
    return HistStatsSchema(**asdict(stats))


def _save_histogram_pair(
    counts_before: np.ndarray,
    counts_after: np.ndarray,
    bin_edges: np.ndarray,
    path: Path,
    title: str,
) -> Path:
    """Before/after step-histogram overlay (lives here, not in the frozen plots.py;
    dsp_server.plots locked the Agg backend before pyplot was imported)."""
    plt = plots.plt
    path.parent.mkdir(parents=True, exist_ok=True)
    centers = 0.5 * (np.asarray(bin_edges)[:-1] + np.asarray(bin_edges)[1:])
    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    ax.step(centers, counts_before, where="mid", lw=1.0, color="tab:blue", label="before")
    ax.step(centers, counts_after, where="mid", lw=1.0, color="tab:orange", label="after")
    ax.set_xlabel("intensity")
    ax.set_ylabel("pixel count")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Tool: enhance_image


def _enhance_image(
    ctx: AppContext,
    image: str,
    mode: str = "histogram_eq",
    gamma: float = 1.0,
    tiles: int = 8,
    clip_limit: float = 0.01,
    out_label: str | None = None,
    save_plot: bool = False,
) -> str:
    try:
        if mode not in _ENHANCE_MODES:
            return _unknown("mode", mode, _ENHANCE_MODES)
        img = image2d.load_image_gray(image)
        before = image2d.hist_stats(img)
        params: dict[str, float]
        if mode == "histogram_eq":
            out = image2d.equalize_hist(img)
            params = {}
        elif mode == "adaptive_eq":
            out = image2d.adaptive_equalize(img, tiles=int(tiles), clip_limit=clip_limit)
            params = {"tiles": float(tiles), "clip_limit": float(clip_limit)}
        elif mode == "gamma":
            out = image2d.adjust_gamma(img, gamma)
            params = {"gamma": float(gamma)}
        else:  # log
            out = image2d.log_transform(img)
            params = {"scale": 255.0}
        out = np.clip(out, 0.0, 1.0)
        after = image2d.hist_stats(out)
        out_path = image2d.save_image_gray(out, _images_dir(ctx) / _out_name(image, mode, out_label))
        plot_path: str | None = None
        if save_plot:
            counts_before, edges = image2d.histogram256(img)
            counts_after, _ = image2d.histogram256(out)
            name = f"hist_{_safe(Path(image).stem)}_{_safe(mode)}.png"
            plot_path = str(
                _save_histogram_pair(
                    counts_before,
                    counts_after,
                    edges,
                    ctx.plots_dir / name,
                    f"{Path(image).name}: {mode}",
                )
            )
        return ImageEnhanceReport(
            image=str(image),
            mode=mode,
            params=params,
            output_path=str(out_path),
            height=int(img.shape[0]),
            width=int(img.shape[1]),
            stats_before=_hist_schema(before),
            stats_after=_hist_schema(after),
            delta_entropy_bits=after.entropy_bits - before.entropy_bits,
            plot_path=plot_path,
        ).model_dump_json()
    except Exception as exc:  # the MCP boundary must never see a raise
        return _error(exc, _IMG_HINT)


# --------------------------------------------------------------------------- #
# Tool: filter_image


def _filter_image(
    ctx: AppContext,
    image: str,
    mode: str = "gaussian",
    sigma: float = 2.0,
    size: int = 3,
    amount: float = 1.0,
    cutoff_cpp: float = 0.1,
    order: int = 2,
    out_label: str | None = None,
) -> str:
    try:
        if mode not in _FILTER_MODES:
            return _unknown("mode", mode, _FILTER_MODES)
        img = image2d.load_image_gray(image)
        before = image2d.hist_stats(img)
        params: dict[str, float]
        if mode == "gaussian":
            out = image2d.gaussian_smooth(img, sigma=sigma)
            params = {"sigma": float(sigma)}
        elif mode == "median":
            out = image2d.median_smooth(img, size=int(size))
            params = {"size": float(size)}
        elif mode == "unsharp":
            out = image2d.unsharp_mask(img, sigma=sigma, amount=amount)
            params = {"sigma": float(sigma), "amount": float(amount)}
        elif mode == "sobel":
            out = image2d.sobel_magnitude(img)
            params = {}
        else:  # butterworth_lp / butterworth_hp
            out = image2d.butterworth_filter(
                img, cutoff_cpp=cutoff_cpp, order=int(order), highpass=(mode == "butterworth_hp")
            )
            params = {"cutoff_cpp": float(cutoff_cpp), "order": float(order)}
        out = np.clip(out, 0.0, 1.0)
        edge_density: float | None = None
        if mode == "sobel":
            threshold = image2d.otsu_threshold(out)
            edge_density = float(np.mean(out > threshold))
        after = image2d.hist_stats(out)
        out_path = image2d.save_image_gray(out, _images_dir(ctx) / _out_name(image, mode, out_label))
        return ImageFilterReport(
            image=str(image),
            mode=mode,
            params=params,
            output_path=str(out_path),
            height=int(img.shape[0]),
            width=int(img.shape[1]),
            stats_before=_hist_schema(before),
            stats_after=_hist_schema(after),
            rms_change=float(np.sqrt(np.mean((out - img) ** 2))),
            ssim_vs_input=image2d.ssim(img, out),
            edge_density=edge_density,
        ).model_dump_json()
    except Exception as exc:
        return _error(exc, _IMG_HINT)


# --------------------------------------------------------------------------- #
# Tool: segment_image


def _segment_image(
    ctx: AppContext,
    image: str,
    mode: str = "otsu",
    foreground: str = "bright",
    block: int = 31,
    offset: float = 0.02,
    k: int = 2,
    morph: str = "open_close",
    morph_size: int = 3,
    roi: list[int] | None = None,
    min_area_px: int = 8,
    max_components: int = 20,
    out_label: str | None = None,
) -> str:
    try:
        if mode not in _SEGMENT_MODES:
            return _unknown("mode", mode, _SEGMENT_MODES)
        if foreground not in _FOREGROUNDS:
            return _unknown("foreground", foreground, _FOREGROUNDS)
        if morph not in image2d.MORPH_MODES:
            return _unknown("morph", morph, image2d.MORPH_MODES)
        if int(max_components) < 1:
            raise ValueError(f"max_components must be >= 1, got {max_components}")
        if int(min_area_px) < 0:
            raise ValueError(f"min_area_px must be >= 0, got {min_area_px}")
        img = image2d.load_image_gray(image)
        roi_t = _roi_tuple(roi)
        if roi_t is not None:
            region = image2d.crop_roi(img, roi_t)
            h_img, w_img = img.shape
            offset_xy = (max(0, min(roi_t[0], w_img)), max(0, min(roi_t[1], h_img)))
        else:
            region = img
            offset_xy = (0, 0)

        threshold: float | None = None
        centers_list: list[float] | None = None
        if mode == "otsu":
            threshold = image2d.otsu_threshold(region)
            mask = region > threshold if foreground == "bright" else region < threshold
        elif mode == "adaptive":
            mask = image2d.adaptive_threshold_mask(
                region, block=int(block), offset=offset, foreground=foreground
            )
        else:  # kmeans
            kk = int(k)
            if not 2 <= kk <= 16:
                raise ValueError(f"k must be in [2, 16], got {k}")
            label_img, centers = image2d.kmeans_gray(region, k=kk)
            centers_list = [float(c) for c in centers]
            target = kk - 1 if foreground == "bright" else 0
            mask = label_img == target

        mask = image2d.binary_cleanup(mask, mode=morph, size=int(morph_size))
        comps = image2d.component_stats(mask, region, offset_xy=offset_xy, min_area_px=int(min_area_px))
        n_total = len(comps)
        truncated = n_total > int(max_components)
        mask_path = image2d.save_mask_png(mask, _images_dir(ctx) / _out_name(image, mode, out_label))
        return SegmentationReport(
            image=str(image),
            mode=mode,
            foreground=foreground,
            threshold=threshold,
            kmeans_centers=centers_list,
            morph=morph,
            morph_size=int(morph_size),
            roi=list(roi_t) if roi_t is not None else None,
            mask_path=str(mask_path),
            height=int(region.shape[0]),
            width=int(region.shape[1]),
            foreground_fraction=float(mask.mean()),
            n_components_total=n_total,
            components=[
                ComponentSchema(
                    label=c.label,
                    area_px=c.area_px,
                    bbox=[int(v) for v in c.bbox],
                    centroid_xy=[float(c.centroid_xy[0]), float(c.centroid_xy[1])],
                    mean_intensity=c.mean_intensity,
                )
                for c in comps[: int(max_components)]
            ],
            components_truncated=truncated,
        ).model_dump_json()
    except Exception as exc:
        return _error(exc, _ROI_HINT)


# --------------------------------------------------------------------------- #
# Tool: restore_image


def _restore_image(
    ctx: AppContext,
    image: str,
    mode: str = "wiener",
    kernel_size: int = 5,
    noise_power: float | None = None,
    psf_sigma: float = 2.0,
    nsr: float = 0.01,
    n_iter: int = 10,
    out_label: str | None = None,
) -> str:
    try:
        if mode not in _RESTORE_MODES:
            return _unknown("mode", mode, _RESTORE_MODES)
        img = image2d.load_image_gray(image)
        before = image2d.hist_stats(img)
        params: dict[str, float]
        if mode == "wiener":
            out = image2d.wiener_denoise(img, kernel_size=int(kernel_size), noise_power=noise_power)
            params = {"kernel_size": float(kernel_size)}
            if noise_power is not None:
                params["noise_power"] = float(noise_power)
        elif mode == "wiener_deconv":
            out = image2d.wiener_deconvolve(img, psf_sigma=psf_sigma, nsr=nsr)
            params = {"psf_sigma": float(psf_sigma), "nsr": float(nsr)}
        else:  # richardson_lucy
            out = image2d.richardson_lucy(img, psf_sigma=psf_sigma, n_iter=int(n_iter))
            params = {"psf_sigma": float(psf_sigma), "n_iter": float(n_iter)}
        out = np.clip(out, 0.0, 1.0)
        after = image2d.hist_stats(out)
        out_path = image2d.save_image_gray(out, _images_dir(ctx) / _out_name(image, mode, out_label))
        return RestorationReport(
            image=str(image),
            mode=mode,
            params=params,
            output_path=str(out_path),
            height=int(img.shape[0]),
            width=int(img.shape[1]),
            stats_before=_hist_schema(before),
            stats_after=_hist_schema(after),
            sharpness_before=image2d.laplacian_sharpness(img),
            sharpness_after=image2d.laplacian_sharpness(out),
            rms_change=float(np.sqrt(np.mean((out - img) ** 2))),
            ssim_vs_input=image2d.ssim(img, out),
        ).model_dump_json()
    except Exception as exc:
        return _error(exc, _IMG_HINT)


# --------------------------------------------------------------------------- #
# Tool: compare_depth_renders (week-14 lane, unchanged)


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

    @mcp.tool()
    def enhance_image(
        image: str,
        mode: str = "histogram_eq",
        gamma: float = 1.0,
        tiles: int = 8,
        clip_limit: float = 0.01,
        out_label: str | None = None,
        save_plot: bool = False,
    ) -> str:
        """Intensity transformation of a grayscale/depth image on disk. mode: 'histogram_eq'
        (global equalization), 'adaptive_eq' (CLAHE-like — tiles x tiles grid, per-tile clipped
        histogram equalization with clip_limit as the ceiling fraction of tile pixels, 0 disables
        clipping, bilinear mapping interpolation), 'gamma' (s = r^gamma), 'log'
        (s = log1p(255*r)/log1p(255)). Writes the result as a 16-bit grayscale PNG under
        data/images/ (a valid input to every imaging tool; out_label suffixes the filename) and
        reports before/after histogram statistics plus the entropy delta in bits. save_plot writes
        a before/after histogram figure and returns its path. Returns ImageEnhanceReport JSON, or
        ToolError JSON on failure."""
        return _enhance_image(ctx, image, mode, gamma, tiles, clip_limit, out_label, save_plot)

    @mcp.tool()
    def filter_image(
        image: str,
        mode: str = "gaussian",
        sigma: float = 2.0,
        size: int = 3,
        amount: float = 1.0,
        cutoff_cpp: float = 0.1,
        order: int = 2,
        out_label: str | None = None,
    ) -> str:
        """Spatial or frequency-domain filtering of a grayscale/depth image on disk. mode:
        'gaussian' (sigma), 'median' (size), 'unsharp' (out = img + amount*(img - gaussian);
        sigma, amount), 'sobel' (gradient magnitude normalized to [0,1], plus edge_density =
        fraction of pixels above the Otsu threshold of the magnitude), 'butterworth_lp' /
        'butterworth_hp' (frequency-domain H = 1/(1+(D/D0)^(2*order)) on the radial fftfreq grid;
        cutoff_cpp is D0 in cycles/pixel — the same unit as compare_depth_renders spectra).
        Params unused by the chosen mode are ignored and omitted from the report. Output is
        clipped to [0,1] and written as a 16-bit PNG under data/images/. Reports before/after
        stats, the RMS of (out - in), and SSIM vs the input. Returns ImageFilterReport JSON, or
        ToolError JSON on failure."""
        return _filter_image(ctx, image, mode, sigma, size, amount, cutoff_cpp, order, out_label)

    @mcp.tool()
    def segment_image(
        image: str,
        mode: str = "otsu",
        foreground: str = "bright",
        block: int = 31,
        offset: float = 0.02,
        k: int = 2,
        morph: str = "open_close",
        morph_size: int = 3,
        roi: list[int] | None = None,
        min_area_px: int = 8,
        max_components: int = 20,
        out_label: str | None = None,
    ) -> str:
        """THE defect-mask tool for depth renders: threshold -> binary morphology -> connected
        components. mode: 'otsu' (global; threshold reported), 'adaptive' (local mean over a
        block x block window +/- offset; threshold null), 'kmeans' (1-D Lloyd on the 256-bin gray
        histogram, k clusters; foreground = brightest cluster for foreground='bright', darkest for
        'dark'; sorted centers reported). morph in {none, open, close, open_close} with a
        morph_size^2 structuring element cleans the mask before ndimage.label. roi is
        [x0, y0, x1, y1] in pixels (exclusive upper bounds) — component bbox/centroid are ALWAYS
        reported in full-image pixel coordinates even with roi. Components smaller than
        min_area_px are dropped; survivors are sorted by area descending and capped at
        max_components with a truncation flag. The mask is written as an 8-bit {0,255} PNG under
        data/images/. Returns SegmentationReport JSON, or ToolError JSON on failure."""
        return _segment_image(
            ctx,
            image,
            mode,
            foreground,
            block,
            offset,
            k,
            morph,
            morph_size,
            roi,
            min_area_px,
            max_components,
            out_label,
        )

    @mcp.tool()
    def restore_image(
        image: str,
        mode: str = "wiener",
        kernel_size: int = 5,
        noise_power: float | None = None,
        psf_sigma: float = 2.0,
        nsr: float = 0.01,
        n_iter: int = 10,
        out_label: str | None = None,
    ) -> str:
        """Honest-scipy image restoration. mode: 'wiener' (scipy.signal.wiener adaptive
        local-statistics DENOISER — explicitly not deconvolution; kernel_size, optional
        noise_power), 'wiener_deconv' (frequency-domain Wiener deconvolution assuming a Gaussian
        PSF of psf_sigma pixels: F = conj(H)/(|H|^2 + nsr) * G), 'richardson_lucy' (n_iter
        multiplicative iterations, Gaussian PSF psf_sigma, fftconvolve-based). No blind
        deconvolution and no non-local means — pure-scipy NLM at render resolution would be
        dishonestly slow; use 'wiener' for plain denoising. Output is clipped to [0,1] and written
        as a 16-bit PNG under data/images/. Reports before/after stats plus a sharpness delta
        (variance of the Laplacian) so the caller can see whether restoration helped. Returns
        RestorationReport JSON, or ToolError JSON on failure."""
        return _restore_image(ctx, image, mode, kernel_size, noise_power, psf_sigma, nsr, n_iter, out_label)

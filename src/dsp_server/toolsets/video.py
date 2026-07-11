"""Video tool pack — the AI-video comparison gate (generated clip vs engine truth).

Two TEMPORAL tools operate on a frame stack already on disk (a directory of frames
or an explicit path list):

* ``evaluate_spatiotemporal_frequencies`` — treats the sequence as an (T, H, W)
  volume and measures temporal frequency content (flicker / boil) that per-frame
  metrics cannot see.
* ``verify_motion_consistency`` — a deterministic optical-flow forward-backward
  hallucination detector (pure-scipy Lucas-Kanade, no OpenCV).

Three per-frame GEOMETRIC / PHOTOMETRIC gates validate one generated frame against
the engine's ground-truth AOV passes (the engine renders exact camera / normal /
depth buffers a generator cannot):

* ``verify_camera_projection`` — reference-vs-generated correspondences fit a global
  homography + fundamental matrix (camera drift / FOV warp).
* ``analyze_photometric_consistency`` — does the generated shading obey the true
  normal pass (Lambertian ``S ~ ambient + N.L`` fit) — a lighting gate.
* ``evaluate_occlusion_boundaries`` — sharpness (variance of Laplacian) at the depth
  pass's discontinuities — a depth-layer-bleeding gate.

Same contract as the other packs: every tool returns a pydantic model's
``model_dump_json()``; failures return a :class:`ToolError` envelope instead of
raising; arrays/plots/masks live under ``data/`` (npz + png), never inline. Each
tool is a thin closure over :class:`AppContext` delegating to a module-level
``_impl`` the tests call directly.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP

from dsp_server import plots
from dsp_server.engine import epipolar, image2d, occlusion, optflow, photometric, stfreq, video
from dsp_server.schemas import SchemaBase, ToolError
from dsp_server.toolsets import AppContext

_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]")

# Flicker verdict thresholds (temporal_hf_energy_fraction is in [0, 1]).
_FLICKER_ABS_FLOOR = 0.35  # no reference: above this the generated stack flickers
_FLICKER_REF_MARGIN = 0.1  # with a reference: exceed it by this much to flag vs-ref

_MAX_INVALID = 64  # cap on the inline invalid-points list

_STFREQ_HINT = (
    "pass a directory of same-size frames or an explicit path list (>= 2 frames); "
    "per-frame spatial quality is a separate lane (compare_depth_renders / "
    "compare_wavelet_signatures) — this tool is the temporal gate"
)
_MOTION_HINT = (
    "pass a directory of same-size frames or an explicit path list (>= 2 frames); "
    "grid_step/window control the flow grid and Lucas-Kanade window size"
)
_CAMERA_HINT = (
    "reference and generated must be readable, identically shaped single frames "
    "(the reference is the engine render, ground truth); needs textured content "
    "(>= 8 trackable corners)"
)
_PHOTOMETRIC_HINT = (
    "generated is a frame image; normal_map is the engine NORMAL pass (rgb-packed, "
    "same size); optional albedo is the engine ALBEDO pass — all identically shaped"
)
_OCCLUSION_HINT = (
    "generated is a frame image; depth is the engine DEPTH pass (same size, with real "
    "discontinuities); optional reference is the engine beauty render for a sharpness baseline"
)


def _safe(text: str) -> str:
    return _SAFE_RE.sub("_", text) or "x"


def _error(exc: Exception, hint: str) -> str:
    return ToolError(error=f"{type(exc).__name__}: {exc}", hint=hint).model_dump_json()


def _video_dir(ctx: AppContext) -> Path:
    """Pack-local output dir under data/ (AppContext stays frozen)."""
    d = ctx.data_dir / "video"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _source_label(source: str | list[str]) -> str:
    if isinstance(source, list):
        return f"[{len(source)} frames]"
    return str(source)


def _source_stem(source: str | list[str]) -> str:
    if isinstance(source, list):
        base = Path(source[0]).parent.name if source else "framelist"
        return _safe(base or "framelist")
    p = Path(source)
    return _safe(p.name if p.is_dir() else p.stem)


# --------------------------------------------------------------------------- #
# Response models (pack-local, on the shared SchemaBase contract)


class SpatioTemporalReport(SchemaBase):
    """evaluate_spatiotemporal_frequencies: temporal flicker/boil summary (+ delta vs a
    reference stack when given)."""

    generated: str
    reference: str | None = None
    n_frames: int
    height: int
    width: int
    hf_cutoff_cpf: float
    temporal_hf_energy_fraction: float
    dominant_temporal_freq_cpf: float
    boil_energy_fraction: float
    ref_temporal_hf_energy_fraction: float | None = None
    ref_dominant_temporal_freq_cpf: float | None = None
    ref_boil_energy_fraction: float | None = None
    delta_temporal_hf_energy_fraction: float | None = None
    delta_dominant_temporal_freq_cpf: float | None = None
    delta_boil_energy_fraction: float | None = None
    flicker_verdict: str
    n_dropped: int = 0
    spectrum_path: str
    plot_path: str | None = None


class InvalidPoint(SchemaBase):
    """One grid point whose forward-backward residual exceeded tau. ``frame`` is the
    consecutive-pair index i (pair i -> i+1); x/y are pixel coordinates."""

    frame: int
    x: int
    y: int
    residual: float


class MotionConsistencyReport(SchemaBase):
    """verify_motion_consistency: forward-backward optical-flow hallucination summary."""

    frames: str
    n_frames: int
    n_pairs: int
    grid_step: int
    window: int
    tau: float
    n_grid_points: int
    mean_fb_residual_px: float
    max_fb_residual_px: float
    inconsistent_fraction: float
    worst_frame_index: int
    worst_frame_inconsistent_fraction: float
    invalid_points: list[InvalidPoint]
    invalid_points_truncated: bool
    n_invalid_total: int
    residual_map_path: str
    plot_path: str | None = None


class CameraProjectionReport(SchemaBase):
    """verify_camera_projection: camera-projection consistency of a generated frame vs a
    reference render (homography-driven verdict + requested epipolar residual)."""

    reference: str
    generated: str
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


class PhotometricConsistencyReport(SchemaBase):
    """analyze_photometric_consistency: Lambertian normal-vs-shading fit of a generated
    frame against the engine normal (and optional albedo) pass."""

    generated: str
    normal_map: str
    albedo: str | None = None
    height: int
    width: int
    n_pixels: int
    used_albedo: bool
    photometric_r2: float
    residual_rms: float
    light_direction: list[float]
    ambient: float
    light_elevation_deg: float
    light_azimuth_deg: float
    albedo_shading_leak: float | None = None
    verdict: str
    verdict_reason: str


class OcclusionBoundaryReport(SchemaBase):
    """evaluate_occlusion_boundaries: generated-frame sharpness at the depth pass's
    occlusion boundaries (+ gen/ref ratio when a reference is given)."""

    generated: str
    depth: str
    reference: str | None = None
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
    ref_boundary_sharpness: float | None = None
    gen_vs_ref_ratio: float | None = None
    verdict: str
    verdict_reason: str
    edge_mask_path: str


# --------------------------------------------------------------------------- #
# Plot helpers (live here, not in frozen plots.py: dsp_server.plots locked the Agg
# backend before pyplot loaded, and exposes the module as plots.plt).


def _save_spectrum_plot(
    spec_g: stfreq.SpatioTemporalSpectrum,
    spec_r: stfreq.SpatioTemporalSpectrum | None,
    path: Path,
    title: str,
) -> Path:
    plt = plots.plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    ax.semilogy(spec_g.freqs_cpf, spec_g.power + 1e-30, lw=1.0, color="tab:blue", label="generated")
    if spec_r is not None:
        ax.semilogy(spec_r.freqs_cpf, spec_r.power + 1e-30, lw=1.0, color="tab:orange", label="reference")
    ax.axvline(spec_g.hf_cutoff_cpf, color="tab:gray", ls="--", lw=0.8, label="HF cutoff")
    ax.set_xlabel("temporal frequency [cycles/frame]")
    ax.set_ylabel("mean temporal power")
    ax.set_title("temporal power spectrum P_t(f_t)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _save_residual_plot(
    mean_grid: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, path: Path, title: str
) -> Path:
    plt = plots.plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    extent = [float(grid_x[0]), float(grid_x[-1]), float(grid_y[-1]), float(grid_y[0])]
    im = ax.imshow(mean_grid, origin="upper", cmap="magma", aspect="auto", extent=extent)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.set_title("mean forward-backward residual")
    fig.colorbar(im, ax=ax, shrink=0.85, label="residual [px]")
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Tool A: evaluate_spatiotemporal_frequencies


def _evaluate_spatiotemporal_frequencies(
    ctx: AppContext,
    generated: str | list[str],
    reference: str | list[str] | None = None,
    hf_cutoff_cpf: float = stfreq.DEFAULT_HF_CUTOFF_CPF,
    max_frames: int | None = None,
    save_plot: bool = False,
) -> str:
    try:
        gen = video.load_frame_stack(generated, max_frames=max_frames)
        spec_g = stfreq.spatiotemporal_spectrum(gen.frames, hf_cutoff_cpf=hf_cutoff_cpf)

        spec_r: stfreq.SpatioTemporalSpectrum | None = None
        if reference is not None:
            ref = video.load_frame_stack(reference, max_frames=max_frames)
            spec_r = stfreq.spatiotemporal_spectrum(ref.frames, hf_cutoff_cpf=hf_cutoff_cpf)

        gen_hf = spec_g.temporal_hf_energy_fraction
        if spec_r is None:
            verdict = "flicker" if gen_hf > _FLICKER_ABS_FLOOR else "stable"
        else:
            delta_hf = gen_hf - spec_r.temporal_hf_energy_fraction
            if delta_hf > _FLICKER_REF_MARGIN:
                verdict = "flicker-vs-ref"
            elif gen_hf > _FLICKER_ABS_FLOOR:
                verdict = "flicker"
            else:
                verdict = "stable"

        outdir = _video_dir(ctx)
        npz_path = outdir / f"stfreq_{_source_stem(generated)}.npz"
        arrays: dict[str, np.ndarray] = {"freqs_cpf": spec_g.freqs_cpf, "power": spec_g.power}
        if spec_r is not None:
            arrays["ref_freqs_cpf"] = spec_r.freqs_cpf
            arrays["ref_power"] = spec_r.power
        np.savez(npz_path, **arrays)

        plot_path: str | None = None
        if save_plot:
            out = ctx.plots_dir / f"stfreq_{_source_stem(generated)}.png"
            title = f"temporal spectrum: {_source_stem(generated)}"
            plot_path = str(_save_spectrum_plot(spec_g, spec_r, out, title))

        # ref_* mirror + delta_* (generated minus reference) only exist in two-stack mode.
        ref_hf = ref_dom = ref_boil = None
        delta_hf = delta_dom = delta_boil = None
        if spec_r is not None:
            ref_hf = spec_r.temporal_hf_energy_fraction
            ref_dom = spec_r.dominant_temporal_freq_cpf
            ref_boil = spec_r.boil_energy_fraction
            delta_hf = spec_g.temporal_hf_energy_fraction - ref_hf
            delta_dom = spec_g.dominant_temporal_freq_cpf - ref_dom
            delta_boil = spec_g.boil_energy_fraction - ref_boil

        return SpatioTemporalReport(
            generated=_source_label(generated),
            reference=_source_label(reference) if reference is not None else None,
            n_frames=spec_g.n_frames,
            height=spec_g.height,
            width=spec_g.width,
            hf_cutoff_cpf=spec_g.hf_cutoff_cpf,
            temporal_hf_energy_fraction=spec_g.temporal_hf_energy_fraction,
            dominant_temporal_freq_cpf=spec_g.dominant_temporal_freq_cpf,
            boil_energy_fraction=spec_g.boil_energy_fraction,
            ref_temporal_hf_energy_fraction=ref_hf,
            ref_dominant_temporal_freq_cpf=ref_dom,
            ref_boil_energy_fraction=ref_boil,
            delta_temporal_hf_energy_fraction=delta_hf,
            delta_dominant_temporal_freq_cpf=delta_dom,
            delta_boil_energy_fraction=delta_boil,
            flicker_verdict=verdict,
            n_dropped=gen.n_dropped,
            spectrum_path=str(npz_path),
            plot_path=plot_path,
        ).model_dump_json()
    except Exception as exc:  # the MCP boundary must never see a raise
        return _error(exc, _STFREQ_HINT)


# --------------------------------------------------------------------------- #
# Tool B: verify_motion_consistency


def _verify_motion_consistency(
    ctx: AppContext,
    frames: str | list[str],
    grid_step: int = 16,
    window: int = 15,
    tau: float = 2.0,
    max_frames: int | None = None,
    save_plot: bool = False,
) -> str:
    try:
        stack = video.load_frame_stack(frames, max_frames=max_frames)
        fb = optflow.forward_backward_residual(stack.frames, grid_step=int(grid_step), window=int(window))
        res = fb.residual_grid  # (n_pairs, n_rows, n_cols)
        inconsistent = res > float(tau)
        per_pair_frac = inconsistent.reshape(fb.n_pairs, -1).mean(axis=1)
        worst_frame = int(np.argmax(per_pair_frac))

        idx = np.argwhere(inconsistent)  # (K, 3): pair, row, col
        flagged = res[inconsistent]
        order = np.argsort(flagged, kind="stable")[::-1]
        n_invalid = int(idx.shape[0])
        invalid_points = [
            InvalidPoint(
                frame=int(idx[k, 0]),
                x=int(fb.grid_x[idx[k, 2]]),
                y=int(fb.grid_y[idx[k, 1]]),
                residual=float(res[idx[k, 0], idx[k, 1], idx[k, 2]]),
            )
            for k in order[:_MAX_INVALID]
        ]

        outdir = _video_dir(ctx)
        npz_path = outdir / f"fbresidual_{_source_stem(frames)}.npz"
        np.savez(npz_path, residual_grid=res, grid_x=fb.grid_x, grid_y=fb.grid_y)

        plot_path: str | None = None
        if save_plot:
            out = ctx.plots_dir / f"fbresidual_{_source_stem(frames)}.png"
            title = f"FB residual: {_source_stem(frames)}"
            plot_path = str(_save_residual_plot(res.mean(axis=0), fb.grid_x, fb.grid_y, out, title))

        return MotionConsistencyReport(
            frames=_source_label(frames),
            n_frames=stack.n_frames,
            n_pairs=fb.n_pairs,
            grid_step=int(grid_step),
            window=int(window),
            tau=float(tau),
            n_grid_points=fb.n_rows * fb.n_cols,
            mean_fb_residual_px=float(res.mean()),
            max_fb_residual_px=float(res.max()),
            inconsistent_fraction=float(inconsistent.mean()),
            worst_frame_index=worst_frame,
            worst_frame_inconsistent_fraction=float(per_pair_frac[worst_frame]),
            invalid_points=invalid_points,
            invalid_points_truncated=n_invalid > _MAX_INVALID,
            n_invalid_total=n_invalid,
            residual_map_path=str(npz_path),
            plot_path=plot_path,
        ).model_dump_json()
    except Exception as exc:  # the MCP boundary must never see a raise
        return _error(exc, _MOTION_HINT)


# --------------------------------------------------------------------------- #
# Tool C: verify_camera_projection (epipolar / camera gate)


def _verify_camera_projection(
    ctx: AppContext,
    reference: str,
    generated: str,
    max_corners: int = 200,
    window: int = 15,
    ransac_px: float = 2.0,
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
                hint=_CAMERA_HINT,
            ).model_dump_json()
        res = epipolar.analyze_camera_projection(
            ref, gen, max_corners=int(max_corners), window=int(window), ransac_px=float(ransac_px)
        )
        return CameraProjectionReport(
            reference=str(reference),
            generated=str(generated),
            height=res.height,
            width=res.width,
            n_corners=res.n_corners,
            n_correspondences=res.n_correspondences,
            median_displacement_px=res.median_displacement_px,
            max_displacement_px=res.max_displacement_px,
            homography_inlier_fraction=res.homography_inlier_fraction,
            homography_residual_px=res.homography_residual_px,
            camera_shift_px=res.camera_shift_px,
            camera_scale=res.camera_scale,
            camera_rotation_deg=res.camera_rotation_deg,
            epipolar_inlier_fraction=res.epipolar_inlier_fraction,
            mean_symmetric_epipolar_distance_px=res.mean_symmetric_epipolar_distance_px,
            epipolar_degenerate=res.epipolar_degenerate,
            verdict=res.verdict,
            verdict_reason=res.verdict_reason,
        ).model_dump_json()
    except Exception as exc:  # the MCP boundary must never see a raise
        return _error(exc, _CAMERA_HINT)


# --------------------------------------------------------------------------- #
# Tool D: analyze_photometric_consistency (lighting gate)


def _analyze_photometric_consistency(
    ctx: AppContext,
    generated: str,
    normal_map: str,
    albedo: str | None = None,
    r2_ok: float = 0.5,
) -> str:
    try:
        gen_lum = image2d.load_image_gray(generated)
        normals = photometric.decode_normal_map(image2d.load_image_rgb(normal_map))
        if normals.shape[:2] != gen_lum.shape:
            return ToolError(
                error=(
                    f"shapes differ: {Path(generated).name} is {gen_lum.shape}, normal map "
                    f"{Path(normal_map).name} is {normals.shape[:2]}"
                ),
                hint=_PHOTOMETRIC_HINT,
            ).model_dump_json()
        albedo_lum = image2d.load_image_gray(albedo) if albedo is not None else None
        res = photometric.analyze_photometric_consistency(
            gen_lum, normals, albedo_lum=albedo_lum, r2_ok=float(r2_ok)
        )
        return PhotometricConsistencyReport(
            generated=str(generated),
            normal_map=str(normal_map),
            albedo=str(albedo) if albedo is not None else None,
            height=int(gen_lum.shape[0]),
            width=int(gen_lum.shape[1]),
            n_pixels=res.n_pixels,
            used_albedo=res.used_albedo,
            photometric_r2=res.photometric_r2,
            residual_rms=res.residual_rms,
            light_direction=list(res.light_direction),
            ambient=res.ambient,
            light_elevation_deg=res.light_elevation_deg,
            light_azimuth_deg=res.light_azimuth_deg,
            albedo_shading_leak=res.albedo_shading_leak,
            verdict=res.verdict,
            verdict_reason=res.verdict_reason,
        ).model_dump_json()
    except Exception as exc:  # the MCP boundary must never see a raise
        return _error(exc, _PHOTOMETRIC_HINT)


# --------------------------------------------------------------------------- #
# Tool E: evaluate_occlusion_boundaries (occlusion gate)


def _evaluate_occlusion_boundaries(
    ctx: AppContext,
    generated: str,
    depth: str,
    reference: str | None = None,
    edge_percentile: float = 95.0,
    band_px: int = 2,
) -> str:
    try:
        gen = image2d.load_image_gray(generated)
        dep = image2d.load_image_gray(depth)
        if dep.shape != gen.shape:
            return ToolError(
                error=(
                    f"shapes differ: {Path(generated).name} is {gen.shape}, depth "
                    f"{Path(depth).name} is {dep.shape}"
                ),
                hint=_OCCLUSION_HINT,
            ).model_dump_json()
        ref = image2d.load_image_gray(reference) if reference is not None else None
        res = occlusion.evaluate_occlusion_boundaries(
            gen, dep, reference=ref, percentile=float(edge_percentile), band_px=int(band_px)
        )
        # The band is derived from the DEPTH pass (+ percentile/band_px), not the
        # generated frame — key the filename off depth so distinct depths don't collide.
        mask_name = f"occlusion_band_{_safe(Path(generated).stem)}__{_safe(Path(depth).stem)}.png"
        mask_path = image2d.save_mask_png(res.edge_band, _video_dir(ctx) / mask_name)
        return OcclusionBoundaryReport(
            generated=str(generated),
            depth=str(depth),
            reference=str(reference) if reference is not None else None,
            height=res.height,
            width=res.width,
            edge_percentile=res.edge_percentile,
            band_px=res.band_px,
            edge_pixel_count=res.edge_pixel_count,
            band_pixel_count=res.band_pixel_count,
            edge_fraction=res.edge_fraction,
            boundary_sharpness=res.boundary_sharpness,
            interior_sharpness=res.interior_sharpness,
            boundary_sharpness_ratio=res.boundary_sharpness_ratio,
            ref_boundary_sharpness=res.ref_boundary_sharpness,
            gen_vs_ref_ratio=res.gen_vs_ref_ratio,
            verdict=res.verdict,
            verdict_reason=res.verdict_reason,
            edge_mask_path=str(mask_path),
        ).model_dump_json()
    except Exception as exc:  # the MCP boundary must never see a raise
        return _error(exc, _OCCLUSION_HINT)


# --------------------------------------------------------------------------- #
# Registration


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the video tools on ``mcp``, bound to ``ctx``."""

    @mcp.tool()
    def evaluate_spatiotemporal_frequencies(
        generated: str | list[str],
        reference: str | list[str] | None = None,
        hf_cutoff_cpf: float = stfreq.DEFAULT_HF_CUTOFF_CPF,
        max_frames: int | None = None,
        save_plot: bool = False,
    ) -> str:
        """TEMPORAL gate: treat a frame sequence (a directory of frames or a path list) as an
        (T, H, W) volume and measure temporal frequency content that PER-FRAME spatial quality
        tools (compare_depth_renders / compare_wavelet_signatures) cannot see. Subtracts the
        per-pixel temporal mean (removes static content), FFTs along time per pixel, and averages
        power over pixels to a temporal spectrum P_t(f_t) in cycles/frame [0, 0.5]. Reports
        temporal_hf_energy_fraction (AC energy above hf_cutoff_cpf over total AC energy —
        flicker/rapid oscillation), dominant_temporal_freq_cpf (argmax AC bin), and
        boil_energy_fraction (energy in the high-spatial x high-temporal corner — rapid change of
        FINE detail, which a spatially-uniform flicker is not). With a reference stack it also
        reports ref_* mirrors and delta_* (generated minus reference). flicker_verdict is 'stable',
        'flicker' (generated HF fraction exceeds an absolute floor), or 'flicker-vs-ref' (exceeds
        the reference by a margin). The full P_t array is written to an npz under data/video/;
        save_plot writes the spectrum figure. Returns SpatioTemporalReport JSON, or ToolError JSON
        on failure."""
        return _evaluate_spatiotemporal_frequencies(
            ctx, generated, reference, hf_cutoff_cpf, max_frames, save_plot
        )

    @mcp.tool()
    def verify_motion_consistency(
        frames: str | list[str],
        grid_step: int = 16,
        window: int = 15,
        tau: float = 2.0,
        max_frames: int | None = None,
        save_plot: bool = False,
    ) -> str:
        """Deterministic optical-flow HALLUCINATION detector over a frame sequence (a directory of
        frames or a path list). For each consecutive pair it runs a pure-scipy pyramidal
        Lucas-Kanade flow forward (i -> i+1) and backward (i+1 -> i) on a grid_step-spaced grid and
        measures the round-trip residual ||f(p) + b(p + f(p))|| in pixels: a geometry-obeying motion
        cancels to near zero, so a large residual marks motion that disobeys geometry (melting /
        drift). NOTE: this is a pure-scipy flow (no OpenCV, per repo policy) — adequate for smooth
        generative video, not a calibrated OpenCV flow. Reports mean/max residual, inconsistent_fraction
        (residual > tau over grid x pairs), the worst pair index and its inconsistent fraction, and a
        capped list of the worst invalid points {frame, x, y, residual}. The full (n_pairs, n_rows,
        n_cols) residual map is written to an npz under data/video/; save_plot writes a mean-residual
        heatmap. Returns MotionConsistencyReport JSON, or ToolError JSON on failure."""
        return _verify_motion_consistency(ctx, frames, grid_step, window, tau, max_frames, save_plot)

    @mcp.tool()
    def verify_camera_projection(
        reference: str,
        generated: str,
        max_corners: int = 200,
        window: int = 15,
        ransac_px: float = 2.0,
    ) -> str:
        """CAMERA gate: does a generated frame's camera match the engine's reference render? The
        engine renders with a strict projection matrix; a generator can imperceptibly drift focal
        length / FOV / nodal point. Shi-Tomasi corners in the reference are tracked into the
        generated frame by the repo's pyramidal Lucas-Kanade with a forward-backward gate, then two
        global camera models are fit by RANSAC: a homography (the reliable metric — well-defined at
        any baseline; its reprojection residual + image-centre shift/scale/rotation quantify the
        drift) and a fundamental matrix (the requested epipolar model; its
        mean_symmetric_epipolar_distance_px is reported but flagged epipolar_degenerate when there
        is no parallax, since F is unidentifiable on a same-view pair). verdict: 'camera-consistent'
        (correspondences near-identity — camera matches), 'camera-drift' (a single homography
        explains a non-trivial shift — pan/zoom/FOV drift), or 'geometry-inconsistent' (no global
        camera model fits — local warping / hallucinated geometry). reference and generated must be
        identically shaped single frames with enough texture (>= 8 trackable corners). Deterministic
        (fixed-seed RANSAC). Returns CameraProjectionReport JSON, or ToolError JSON on failure."""
        return _verify_camera_projection(ctx, reference, generated, max_corners, window, ransac_px)

    @mcp.tool()
    def analyze_photometric_consistency(
        generated: str,
        normal_map: str,
        albedo: str | None = None,
        r2_ok: float = 0.5,
    ) -> str:
        """LIGHTING gate: does a generated frame's shading obey the engine's true surface normals?
        Rendering forms I = R*S (albedo x shading); a generator often bakes lighting into texture,
        producing shading that ignores the geometry. Given the engine NORMAL pass (rgb-packed,
        decoded as N = 2*rgb-1) and optionally the ALBEDO pass, the observed shading (S = I/albedo,
        or S = I when no albedo) is fit to the Lambertian model S ~ ambient + N.L by linear least
        squares over all valid pixels; photometric_r2 is the verdict signal. It also recovers the
        light_direction / elevation / azimuth (in the normal map's frame) and, with albedo, the
        albedo_shading_leak (Pearson correlation of |grad shading| with |grad albedo| — high means
        texture bled into shading, a failed intrinsic decomposition). verdict: 'consistent'
        (r2 >= r2_ok — lighting obeys geometry) or 'inconsistent' (hallucinated / baked-in shading),
        optionally with an 'albedo leak' note. This is a linear Lambertian PROXY (no cast-shadow or
        max(0,.) clamp modeling), a consistency gate not a light solver. All inputs identically
        shaped. Returns PhotometricConsistencyReport JSON, or ToolError JSON on failure."""
        return _analyze_photometric_consistency(ctx, generated, normal_map, albedo, r2_ok)

    @mcp.tool()
    def evaluate_occlusion_boundaries(
        generated: str,
        depth: str,
        reference: str | None = None,
        edge_percentile: float = 95.0,
        band_px: int = 2,
    ) -> str:
        """OCCLUSION gate: is a generated frame blurred where objects occlude each other? A
        generator has no Z-buffer, so foreground/background pixels bleed at occlusion boundaries.
        The engine DEPTH pass marks those boundaries exactly: this takes the Sobel gradient of the
        depth pass, thresholds it at edge_percentile to an occlusion band (dilated by band_px), and
        measures the variance of the generated frame's Laplacian (the standard sharpness proxy)
        inside the band vs the flat interior. boundary_sharpness_ratio > 1 means edges are crisper
        than interior (expected for real occlusions). With a reference beauty render,
        gen_vs_ref_ratio (gen boundary sharpness over ref) is the reliable read — below ref_ok
        means the generator softened the occlusion edges. verdict: 'sharp' or 'soft-boundaries'
        (depth-layer bleeding). The depth pass must have real discontinuities (a flat depth is a
        ToolError). The edge-band mask is written as an 8-bit PNG under data/video/. Returns
        OcclusionBoundaryReport JSON, or ToolError JSON on failure."""
        return _evaluate_occlusion_boundaries(ctx, generated, depth, reference, edge_percentile, band_px)

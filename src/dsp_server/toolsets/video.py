"""Video tool pack — the TEMPORAL half of the comparison gate.

Two tools operating on a frame stack already on disk (a directory of frames or an
explicit path list), the temporal counterpart to the imaging pack's per-frame
spatial lane:

* ``evaluate_spatiotemporal_frequencies`` — treats the sequence as an (T, H, W)
  volume and measures temporal frequency content (flicker / boil) that per-frame
  metrics cannot see.
* ``verify_motion_consistency`` — a deterministic optical-flow forward-backward
  hallucination detector (pure-scipy Lucas-Kanade, no OpenCV).

Same contract as the other packs: every tool returns a pydantic model's
``model_dump_json()``; failures return a :class:`ToolError` envelope instead of
raising; arrays/plots live under ``data/`` (npz + png), never inline. Each tool is
a thin closure over :class:`AppContext` delegating to a module-level ``_impl`` the
tests call directly.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP

from dsp_server import plots
from dsp_server.engine import optflow, stfreq, video
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

"""Systems tool pack (ELE301): LTI responses, pole-zero maps, Bode margins, the
sampling/aliasing verdict, convolution/correlation of series, group delay, and the
state-space controllability/observability verdict.

Seven tools: lti_response, pole_zero, bode, sampling_check, convolve_signals,
group_delay, state_space_analysis. Same contract as the geometry pack: every tool
returns a pydantic model's ``model_dump_json()``; every failure returns ToolError
JSON (TabularError messages land in the hint verbatim); arrays live on disk under
data/series/, never inline. Module-level ``_impl`` functions are called directly by
tests.

CT vs DT is the ``domain`` parameter ('s' | 'z'; 'z' requires dt). DT num/den
are polynomials in z^-1 (den=[1,-0.5] means 1-0.5z^-1). ``bode`` margins treat
the given H as the OPEN-loop transfer function. ``sampling_check`` is the bridge
onto the ELE407 lane: it accepts an engine dump + joint, flipping units to
cycles/meter with fs read as samples/meter.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP

# Importing dsp_server.plots locks the matplotlib Agg backend BEFORE this module's
# plot helpers lazily import pyplot (stdout is MCP JSON-RPC; no GUI may ever start).
from dsp_server import plots  # noqa: F401
from dsp_server.engine import filters, ltisys, ply, statespace, tabular, transforms
from dsp_server.engine.transforms import Signal1D
from dsp_server.schemas import SchemaBase, ToolError
from dsp_server.toolsets import AppContext
from dsp_server.toolsets.geometry import _bone_map_of, _histogram_hint

_MAX_INLINE = 4096
_NO_DT_MSG = "this format carries no dt — pass dt="
_NO_DT_HINT = (
    "only .npz (reserved scalar key 'dt') and .ply dump profiles carry a sample spacing; "
    "csv/tsv/json/npy series need an explicit dt= (seconds per sample, or meters per sample "
    "for geometry profiles)"
)
_OPERAND_HINT = (
    "pass each signal as path_X= (csv/tsv/json/npz/npy/ply file) or values_X= (inline list, "
    f"<= {_MAX_INLINE} samples); files carrying different dt are resampled onto a's grid"
)
_GROUP_DELAY_HINT = (
    "group_delay runs in one of two modes: (a) LTI — pass num=+den= or zeros=/poles=/gain= with "
    "domain='s'|'z' (z requires dt=); (b) cross-spectrum — pass series_a= and series_b= "
    "(equal-length series files; add dt= when the format carries no sample spacing)"
)
_STATE_SPACE_HINT = (
    "pass A (square n x n) and B (n x m) as inline rows a=[[...]] / b=[[...]] OR as files "
    "a_path=/b_path= (csv/tsv/json/npz/npy); add C (p x n) via c=/c_path= to also test observability"
)


# --------------------------------------------------------------------------- #
# Response models (pack-local, on the shared SchemaBase contract)


class StepMetricsSchema(SchemaBase):
    """Step-response metrics; all None when the response has not converged."""

    final_value: float | None = None
    rise_time_10_90: float | None = None
    settling_time_2pct: float | None = None
    overshoot_pct: float | None = None
    peak_time: float | None = None


class PoleDynamicsSchema(SchemaBase):
    """One pole: location + damping ratio / natural frequency / time constant."""

    re: float
    im: float
    magnitude: float | None = None  # DT only: |z|
    damping_ratio: float | None = None
    natural_freq_rad_s: float | None = None
    time_constant_s: float | None = None


class FoldedPeakSchema(SchemaBase):
    """A spectral peak of the reference signal and where sampling at fs folds it."""

    f_true: float
    f_alias: float
    prominence_db: float
    aliased: bool


class LtiResponseReport(SchemaBase):
    """lti_response: time response + step metrics + saved trajectory path."""

    domain: str
    dt: float | None = None
    input_kind: str
    stable: bool
    poles: list[list[float]]
    dc_gain: float | None = None
    step: StepMetricsSchema | None = None
    y_min: float
    y_max: float
    t_at_peak: float
    response_path: str
    plot_path: str | None = None


class PoleZeroReport(SchemaBase):
    """pole_zero: per-pole dynamics + stability/ROC/minimum-phase reading."""

    domain: str
    dt: float | None = None
    gain: float
    poles: list[PoleDynamicsSchema]
    zeros: list[list[float]]
    stable: bool
    marginally_stable: bool
    causal_roc: str
    minimum_phase: bool
    proper: bool
    plot_path: str | None = None


class BodeReport(SchemaBase):
    """bode: open-loop margins + bandwidth/resonance/HF-slope summary."""

    domain: str
    dt: float | None = None
    dc_gain_db: float | None = None
    gain_margin_db: float | None = None
    gain_margin_freq: float | None = None
    phase_margin_deg: float | None = None
    phase_margin_freq: float | None = None
    bandwidth_3db: float | None = None
    resonant_peak_db: float | None = None
    resonant_freq: float | None = None
    hf_slope_db_per_decade: float
    arrays_path: str
    plot_path: str | None = None


class SamplingCheckReport(SchemaBase):
    """sampling_check: aliasing verdict for a proposed rate against a reference."""

    source: str
    n_samples: int
    dt: float
    freq_unit: str  # "Hz" | "cycles/m"
    fs_proposed: float
    nyquist: float
    energy_fraction_above_nyquist: float
    bandwidth_99pct: float
    verdict: str  # safe | marginal | aliased
    folded_peaks: list[FoldedPeakSchema]
    fs_suggested: float
    plot_path: str | None = None


class ConvolutionReport(SchemaBase):
    """convolve_signals: convolution/correlation summary + saved result path."""

    mode: str
    scale: str
    n_a: int
    n_b: int
    dt: float
    n_out: int
    y_peak: float
    t_at_peak: float | None = None
    lag_at_peak: float | None = None
    peak_corrcoef: float | None = None
    energy_out: float
    resampled_b: bool
    result_path: str
    plot_path: str | None = None


class GroupDelayReport(SchemaBase):
    """group_delay: tau_g(w) summary + saved (w, tau) arrays path."""

    mode: str  # "lti" | "cross_spectrum"
    domain: str  # "s" | "z" | "cross-spectrum"
    dt: float | None = None
    n_points: int
    tau_unit: str  # "samples" (digital LTI) | "seconds"
    tau_mean: float
    tau_max: float
    tau_min: float
    tau_variance: float
    freq_at_max_tau: float  # w [rad/s] at the tau maximum
    tau_at_dc: float  # tau at the lowest analyzed frequency
    tau_mean_s: float  # tau_mean expressed in seconds (= tau_mean*dt for the digital lane)
    group_delay_path: str
    plot_path: str | None = None


class StateSpaceReport(SchemaBase):
    """state_space_analysis: controllability (+ optional observability) verdict."""

    n_states: int
    n_inputs: int
    ctrb_rank: int
    controllable: bool
    uncontrollable_dim: int
    ctrb_min_singular: float
    n_outputs: int | None = None
    obsv_rank: int | None = None
    observable: bool | None = None
    unobservable_dim: int | None = None
    obsv_min_singular: float | None = None
    note: str


# --------------------------------------------------------------------------- #
# Shared helpers


def _series_dir(ctx: AppContext) -> Path:
    d = ctx.data_dir / "series"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stamp(*parts: object) -> str:
    return hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:8]


def _fail(exc: Exception, default_hint: str | None = None) -> str:
    if isinstance(exc, tabular.TabularError):
        hint: str | None = str(exc)
    elif _NO_DT_MSG in str(exc):
        hint = _NO_DT_HINT
    else:
        hint = default_hint
    return ToolError(error=f"{type(exc).__name__}: {exc}", hint=hint).model_dump_json()


def _prepare_plot(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_xy_plot(
    curves: list[tuple[np.ndarray, np.ndarray, str]],
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> Path:
    import matplotlib.pyplot as plt  # Agg locked via dsp_server.plots at module import

    path = _prepare_plot(path)
    fig, ax = plt.subplots(figsize=(9.0, 4.5), constrained_layout=True)
    for x, y, label in curves:
        ax.plot(x, y, lw=1.0, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _save_pole_zero_plot(poles: np.ndarray, zeros: np.ndarray, domain: str, path: Path, title: str) -> Path:
    import matplotlib.pyplot as plt

    path = _prepare_plot(path)
    fig, ax = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)
    p = np.atleast_1d(np.asarray(poles, dtype=np.complex128))
    z = np.atleast_1d(np.asarray(zeros, dtype=np.complex128))
    if p.size:
        ax.plot(p.real, p.imag, "x", ms=9, color="tab:red", label="poles")
    if z.size:
        ax.plot(z.real, z.imag, "o", mfc="none", ms=9, color="tab:blue", label="zeros")
    if domain == "z":
        th = np.linspace(0.0, 2.0 * np.pi, 256)
        ax.plot(np.cos(th), np.sin(th), lw=0.8, color="gray", label="|z| = 1")
    else:
        ax.axvline(0.0, lw=0.8, color="gray")
    ax.axhline(0.0, lw=0.5, color="gray", alpha=0.5)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _save_bode_plot(w: np.ndarray, mag_db: np.ndarray, phase_deg: np.ndarray, path: Path, title: str) -> Path:
    import matplotlib.pyplot as plt

    path = _prepare_plot(path)
    fig, (ax_m, ax_p) = plt.subplots(2, 1, figsize=(9.0, 6.0), constrained_layout=True)
    ax_m.semilogx(w, mag_db, lw=1.0, color="tab:blue")
    ax_m.axhline(0.0, lw=0.6, color="gray")
    ax_m.set_ylabel("|H| [dB]")
    ax_m.grid(True, alpha=0.3, which="both")
    ax_p.semilogx(w, phase_deg, lw=1.0, color="tab:red")
    ax_p.axhline(-180.0, lw=0.6, color="gray")
    ax_p.set_xlabel("w [rad/s]")
    ax_p.set_ylabel("phase [deg]")
    ax_p.grid(True, alpha=0.3, which="both")
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _save_nyquist_plot(
    freqs: np.ndarray,
    psd: np.ndarray,
    nyquist: float,
    folded: list[FoldedPeakSchema],
    unit: str,
    path: Path,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt

    path = _prepare_plot(path)
    fig, ax = plt.subplots(figsize=(9.0, 4.5), constrained_layout=True)
    db = 10.0 * np.log10(np.asarray(psd, dtype=np.float64) + 1e-30)
    ax.plot(freqs[1:], db[1:], lw=1.0, color="tab:blue")
    ax.axvline(nyquist, lw=1.2, color="tab:red", label=f"proposed Nyquist = {nyquist:g} {unit}")
    for pk in folded:
        if pk.aliased:
            level = float(np.interp(pk.f_true, freqs, db))
            ax.annotate(
                "",
                xy=(pk.f_alias, level),
                xytext=(pk.f_true, level),
                arrowprops={"arrowstyle": "->", "color": "tab:orange"},
            )
    ax.set_xlabel(f"frequency [{unit}]")
    ax.set_ylabel("PSD [dB]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _load_operand(
    ctx: AppContext,
    path: str | None,
    values: list[float] | None,
    key: str | None,
    column: str | None,
    side: str,
) -> tuple[np.ndarray, float | None, str]:
    """One convolution operand: (samples, dt-or-None, source label)."""
    if (path is None) == (values is None):
        raise ValueError(f"pass exactly one of path_{side}= or values_{side}= for signal {side}")
    if values is not None:
        if len(values) > _MAX_INLINE:
            raise ValueError(
                f"values_{side} has {len(values)} samples — inline cap is {_MAX_INLINE}; "
                f"write an .npz and pass path_{side}="
            )
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError(f"values_{side} must be a flat list with >= 2 samples")
        return arr, None, f"inline values_{side}"
    series = tabular.load_series(path, column=column, key=key, cache_dir=ctx.cache_dir)
    return series.y, series.dt, str(path)


# --------------------------------------------------------------------------- #
# Tool 1: lti_response


def _lti_response(
    ctx: AppContext,
    num: list[float] | None = None,
    den: list[float] | None = None,
    zeros: list[list[float]] | None = None,
    poles: list[list[float]] | None = None,
    gain: float | None = None,
    domain: str = "s",
    dt: float | None = None,
    input_kind: str = "step",
    input_path: str | None = None,
    input_key: str | None = None,
    t_end: float | None = None,
    n_samples: int = 500,
    save_plot: bool = False,
) -> str:
    try:
        sys_ = ltisys.build_system(num, den, zeros, poles, gain, domain, dt)
        z, p, k = ltisys.zpk_of(sys_)
        stable, _marginal = ltisys.stability_of(p, domain)
        n_samples = int(n_samples)
        if not 8 <= n_samples <= 100_000:
            raise ValueError(f"n_samples must be in [8, 100000], got {n_samples}")

        u: np.ndarray | None = None
        if input_kind == "custom":
            if input_path is None:
                raise ValueError("input_kind='custom' needs input_path= (any series file)")
            series = tabular.load_series(input_path, key=input_key, cache_dir=ctx.cache_dir)
            u = series.y
            if domain == "z":
                grid_dt = float(sys_.dt)
            else:
                grid_dt_opt = series.dt if series.dt is not None else dt
                if grid_dt_opt is None:
                    raise ValueError(_NO_DT_MSG)
                grid_dt = float(grid_dt_opt)
            t = np.arange(u.size, dtype=np.float64) * grid_dt
        elif domain == "z":
            sys_dt = float(sys_.dt)
            n = n_samples if t_end is None else min(max(int(round(float(t_end) / sys_dt)) + 1, 8), 100_000)
            t = np.arange(n, dtype=np.float64) * sys_dt
        else:
            end = float(t_end) if t_end is not None else ltisys.default_t_end(p, domain)
            t = np.linspace(0.0, end, n_samples)

        tout, uu, y = ltisys.time_response(sys_, input_kind, t, u=u)
        step_schema: StepMetricsSchema | None = None
        if input_kind == "step" and stable:
            m = ltisys.step_metrics(tout, y)
            step_schema = StepMetricsSchema(
                final_value=m.final_value,
                rise_time_10_90=m.rise_time_10_90,
                settling_time_2pct=m.settling_time_2pct,
                overshoot_pct=m.overshoot_pct,
                peak_time=m.peak_time,
            )
        stamp = _stamp(
            "lti", num, den, zeros, poles, gain, domain, dt, input_kind, input_path, t_end, n_samples
        )
        out = _series_dir(ctx) / f"lti_{stamp}.npz"
        np.savez(out, t=tout, u=uu, y=y)
        plot_path: str | None = None
        if save_plot:
            plot_path = str(
                _save_xy_plot(
                    [(tout, uu, "u (input)"), (tout, y, "y (output)")],
                    ctx.plots_dir / f"lti_{stamp}.png",
                    f"{input_kind} response (domain={domain})",
                    "t [s]",
                    "amplitude",
                )
            )
        ip = int(np.argmax(np.abs(y)))
        return LtiResponseReport(
            domain=domain,
            dt=float(sys_.dt) if domain == "z" else None,
            input_kind=input_kind,
            stable=stable,
            poles=ltisys.complex_to_pairs(p),
            dc_gain=ltisys.dc_gain(z, p, k, domain),
            step=step_schema,
            y_min=float(np.min(y)),
            y_max=float(np.max(y)),
            t_at_peak=float(tout[ip]),
            response_path=str(out),
            plot_path=plot_path,
        ).model_dump_json()
    except Exception as exc:  # the MCP boundary must never see a raise
        return _fail(exc, default_hint=ltisys.SYSTEM_FORM_HINT)


# --------------------------------------------------------------------------- #
# Tool 2: pole_zero


def _pole_zero(
    ctx: AppContext,
    num: list[float] | None = None,
    den: list[float] | None = None,
    zeros: list[list[float]] | None = None,
    poles: list[list[float]] | None = None,
    gain: float | None = None,
    domain: str = "s",
    dt: float | None = None,
    save_plot: bool = False,
) -> str:
    try:
        sys_ = ltisys.build_system(num, den, zeros, poles, gain, domain, dt)
        z, p, k = ltisys.zpk_of(sys_)
        stable, marginal = ltisys.stability_of(p, domain)
        sys_dt = float(sys_.dt) if domain == "z" else None
        dyn = ltisys.pole_dynamics(p, domain, sys_dt)
        pole_schemas = [
            PoleDynamicsSchema(
                re=d.re,
                im=d.im,
                magnitude=d.magnitude,
                damping_ratio=d.damping_ratio,
                natural_freq_rad_s=d.natural_freq_rad_s,
                time_constant_s=d.time_constant_s,
            )
            for d in dyn
        ]
        if domain == "s":
            minimum_phase = bool(np.all(z.real < 0.0)) if z.size else True
        else:
            minimum_phase = bool(np.all(np.abs(z) < 1.0)) if z.size else True
        plot_path: str | None = None
        if save_plot:
            stamp = _stamp("pz", num, den, zeros, poles, gain, domain, dt)
            plot_path = str(
                _save_pole_zero_plot(
                    p, z, domain, ctx.plots_dir / f"pole_zero_{stamp}.png", f"pole-zero map ({domain})"
                )
            )
        return PoleZeroReport(
            domain=domain,
            dt=sys_dt,
            gain=k,
            poles=pole_schemas,
            zeros=ltisys.complex_to_pairs(z),
            stable=stable,
            marginally_stable=marginal,
            causal_roc=ltisys.roc_text(p, domain),
            minimum_phase=minimum_phase,
            proper=bool(z.size <= p.size),
            plot_path=plot_path,
        ).model_dump_json()
    except Exception as exc:
        return _fail(exc, default_hint=ltisys.SYSTEM_FORM_HINT)


# --------------------------------------------------------------------------- #
# Tool 3: bode


def _bode(
    ctx: AppContext,
    num: list[float] | None = None,
    den: list[float] | None = None,
    zeros: list[list[float]] | None = None,
    poles: list[list[float]] | None = None,
    gain: float | None = None,
    domain: str = "s",
    dt: float | None = None,
    w_min: float | None = None,
    w_max: float | None = None,
    n_points: int = 400,
    save_plot: bool = False,
) -> str:
    try:
        sys_ = ltisys.build_system(num, den, zeros, poles, gain, domain, dt)
        z, p, k = ltisys.zpk_of(sys_)
        n_points = int(n_points)
        if not 16 <= n_points <= 20_000:
            raise ValueError(f"n_points must be in [16, 20000], got {n_points}")
        sys_dt = float(sys_.dt) if domain == "z" else None
        w = ltisys.default_w_grid(z, p, domain, sys_dt, n_points, w_min=w_min, w_max=w_max)
        wout, mag_db, phase_deg = ltisys.freq_response(sys_, w)
        margins = ltisys.stability_margins(wout, mag_db, phase_deg)

        dcg = ltisys.dc_gain(z, p, k, domain)
        dc_db: float | None = None
        if dcg is not None and abs(dcg) > 1e-30:
            dc_db = float(20.0 * np.log10(abs(dcg)))
        bandwidth = ltisys.bandwidth_3db(wout, mag_db, ref_db=dc_db) if dc_db is not None else None
        res_peak_db = res_freq = None
        i_max = int(np.argmax(mag_db))
        if dc_db is not None and 0 < i_max < mag_db.size - 1 and mag_db[i_max] > dc_db + 0.01:
            res_peak_db = float(mag_db[i_max])
            res_freq = float(wout[i_max])

        stamp = _stamp("bode", num, den, zeros, poles, gain, domain, dt, w_min, w_max, n_points)
        out = _series_dir(ctx) / f"bode_{stamp}.npz"
        np.savez(out, w=wout, mag_db=mag_db, phase_deg=phase_deg)
        plot_path: str | None = None
        if save_plot:
            plot_path = str(
                _save_bode_plot(
                    wout, mag_db, phase_deg, ctx.plots_dir / f"bode_{stamp}.png", f"Bode ({domain})"
                )
            )
        return BodeReport(
            domain=domain,
            dt=sys_dt,
            dc_gain_db=dc_db,
            gain_margin_db=margins.gm_db,
            gain_margin_freq=margins.w_gain_cross,
            phase_margin_deg=margins.pm_deg,
            phase_margin_freq=margins.w_phase_cross,
            bandwidth_3db=bandwidth,
            resonant_peak_db=res_peak_db,
            resonant_freq=res_freq,
            hf_slope_db_per_decade=ltisys.hf_slope(wout, mag_db),
            arrays_path=str(out),
            plot_path=plot_path,
        ).model_dump_json()
    except Exception as exc:
        return _fail(exc, default_hint=ltisys.SYSTEM_FORM_HINT)


# --------------------------------------------------------------------------- #
# Tool 4: sampling_check


def _sampling_check(
    ctx: AppContext,
    fs: float,
    series_path: str | None = None,
    column: str | None = None,
    key: str | None = None,
    values: list[float] | None = None,
    dt: float | None = None,
    dump: str | None = None,
    joint: str = "armLowerL",
    channel: str = "posed",
    bone_map_path: str | None = None,
    save_plot: bool = False,
) -> str:
    loaded: ply.EngineDump | None = None
    bone_map: dict[int, str] = {}
    try:
        fs = float(fs)
        if fs <= 0.0:
            raise ValueError(f"fs must be positive, got {fs}")
        n_sources = sum(x is not None for x in (series_path, values, dump))
        if n_sources != 1:
            raise ValueError(
                "pass exactly one source: series_path= (csv/tsv/json/npz/npy/ply), "
                "values= (inline list, requires dt=), or dump= (engine PLY + joint/channel)"
            )

        freq_unit = "Hz"
        if dump is not None:
            dump_path = Path(dump)
            loaded = ply.load_dump_cached(dump_path, ctx.cache_dir)
            bone_map = _bone_map_of(dump_path)
            forearm = transforms.extract_forearm_signal(
                loaded, joint=joint, bone_map=bone_map or None, bone_map_path=bone_map_path, channel=channel
            )
            y, sig_dt = forearm.signal.y, float(forearm.signal.dt)
            freq_unit = "cycles/m"
            source = str(dump_path)
        elif values is not None:
            if len(values) > _MAX_INLINE:
                raise ValueError(
                    f"values has {len(values)} samples — inline cap is {_MAX_INLINE}; "
                    "write an .npz (series + reserved key 'dt') and pass series_path="
                )
            y = np.asarray(values, dtype=np.float64)
            if y.ndim != 1:
                raise ValueError("values must be a flat list of numbers")
            if dt is None:
                raise ValueError(_NO_DT_MSG)
            sig_dt = float(dt)
            source = "inline values"
        else:
            series = tabular.load_series(series_path, column=column, key=key, cache_dir=ctx.cache_dir)
            y = series.y
            sig_dt_opt = series.dt if series.dt is not None else dt
            if sig_dt_opt is None:
                raise ValueError(_NO_DT_MSG)
            sig_dt = float(sig_dt_opt)
            if Path(str(series_path)).suffix.lower() == ".ply":
                freq_unit = "cycles/m"
            source = str(series_path)

        if sig_dt <= 0.0:
            raise ValueError(f"dt must be positive, got {sig_dt}")
        if y.size < 16:
            raise ValueError(f"need >= 16 samples to form a reference spectrum, got {y.size}")

        nyq_ref = 0.5 / sig_dt
        if 0.5 * fs >= 0.9 * nyq_ref:
            return ToolError(
                error=(
                    f"insufficient_resolution: proposed Nyquist fs/2 = {0.5 * fs:g} {freq_unit} is "
                    f"at/above 90% of the reference signal's own Nyquist ({nyq_ref:g} {freq_unit})"
                ),
                hint=(
                    "the reference must be sampled well above the proposed rate — energy beyond the "
                    "proposed Nyquist is invisible otherwise. Supply a reference with "
                    f"dt < {0.9 / fs:g} (its Nyquist > {fs / 1.8:g} {freq_unit}), or propose a lower fs"
                ),
            ).model_dump_json()

        sig = Signal1D(y=np.ascontiguousarray(y, dtype=np.float64), dt=sig_dt, t0=0.0)
        freqs, psd = transforms.spectrum(sig)
        met = ltisys.sampling_metrics(freqs, psd, fs)
        span = sig.span_m if sig.span_m > 0.0 else sig_dt
        # Band floor: 2/span (first couple of non-DC bins) — geometry's 12 cpm is
        # meaningless for the Hz lane and is deliberately NOT reused here.
        peaks = filters.spectral_peaks(freqs, psd, f_min_cpm=2.0 / span, max_peaks=5)
        folded = [
            FoldedPeakSchema(
                f_true=f,
                f_alias=float(ltisys.alias_fold(np.asarray([f]), fs)[0]),
                prominence_db=prom,
                aliased=bool(f > 0.5 * fs),
            )
            for f, prom in peaks
        ]
        if met.frac_above < 0.01:
            verdict = "safe"
        elif met.frac_above < 0.05:
            verdict = "marginal"
        else:
            verdict = "aliased"

        plot_path: str | None = None
        if save_plot:
            stamp = _stamp("sampling", source, fs, sig_dt, y.size)
            plot_path = str(
                _save_nyquist_plot(
                    freqs,
                    psd,
                    met.nyquist,
                    folded,
                    freq_unit,
                    ctx.plots_dir / f"sampling_{stamp}.png",
                    f"sampling check @ fs={fs:g} ({verdict})",
                )
            )
        return SamplingCheckReport(
            source=source,
            n_samples=int(y.size),
            dt=sig_dt,
            freq_unit=freq_unit,
            fs_proposed=fs,
            nyquist=met.nyquist,
            energy_fraction_above_nyquist=met.frac_above,
            bandwidth_99pct=met.bw_99,
            verdict=verdict,
            folded_peaks=folded,
            fs_suggested=2.5 * met.bw_99,
            plot_path=plot_path,
        ).model_dump_json()
    except Exception as exc:
        if dump is not None:
            return ToolError(
                error=f"{type(exc).__name__}: {exc}", hint=_histogram_hint(loaded, bone_map)
            ).model_dump_json()
        return _fail(exc)


# --------------------------------------------------------------------------- #
# Tool 5: convolve_signals


def _convolve_signals(
    ctx: AppContext,
    path_a: str | None = None,
    path_b: str | None = None,
    key_a: str | None = None,
    key_b: str | None = None,
    column_a: str | None = None,
    column_b: str | None = None,
    values_a: list[float] | None = None,
    values_b: list[float] | None = None,
    dt: float | None = None,
    mode: str = "convolve",
    scale: str = "continuous",
    normalize: bool = False,
    save_plot: bool = False,
) -> str:
    try:
        if mode not in ("convolve", "correlate"):
            raise ValueError(f"unknown mode '{mode}' (expected convolve|correlate)")
        if scale not in ("continuous", "discrete"):
            raise ValueError(f"unknown scale '{scale}' (expected continuous|discrete)")
        a, dt_a_file, src_a = _load_operand(ctx, path_a, values_a, key_a, column_a, "a")
        b, dt_b_file, src_b = _load_operand(ctx, path_b, values_b, key_b, column_b, "b")
        n_a, n_b = int(a.size), int(b.size)

        dt_a = dt_a_file if dt_a_file is not None else (float(dt) if dt is not None else None)
        if dt_a is None:
            raise ValueError(_NO_DT_MSG)
        dt_a = float(dt_a)
        dt_b = float(dt_b_file) if dt_b_file is not None else (float(dt) if dt is not None else dt_a)
        resampled_b = False
        if abs(dt_b - dt_a) > 1e-12 * max(dt_a, dt_b):
            frac = Fraction(dt_b / dt_a).limit_denominator(1000)
            if frac.numerator <= 0:
                raise ValueError(f"cannot rationalize dt ratio {dt_b}/{dt_a}")
            b, dt_b = filters.resample_profile(b, dt_b, frac.numerator, frac.denominator)
            resampled_b = True

        axis, y = ltisys.convolve_uniform(a, b, dt_a, mode=mode, scale=scale)
        ip = int(np.argmax(y))
        t_at_peak = lag_at_peak = corr = None
        if mode == "convolve":
            t_at_peak = float(axis[ip])
        else:
            lag_at_peak = float(axis[ip])
            if normalize:
                corr = ltisys.ncc_at_lag(a, np.asarray(b, dtype=np.float64), ip - (int(b.size) - 1))
        energy = float(np.sum(y**2)) * (dt_a if scale == "continuous" else 1.0)

        stamp = _stamp("conv", src_a, src_b, n_a, n_b, dt_a, mode, scale)
        out = _series_dir(ctx) / f"conv_{stamp}.npz"
        if mode == "convolve":
            np.savez(out, t=axis, y=y)
        else:
            np.savez(out, lag=axis, y=y)
        plot_path: str | None = None
        if save_plot:
            xlabel = "t [s]" if mode == "convolve" else "lag [s] (a delayed copy in b peaks at lag<0)"
            plot_path = str(
                _save_xy_plot(
                    [(axis, y, mode)],
                    ctx.plots_dir / f"conv_{stamp}.png",
                    f"{mode} ({scale})",
                    xlabel,
                    "amplitude",
                )
            )
        return ConvolutionReport(
            mode=mode,
            scale=scale,
            n_a=n_a,
            n_b=n_b,
            dt=dt_a,
            n_out=int(y.size),
            y_peak=float(y[ip]),
            t_at_peak=t_at_peak,
            lag_at_peak=lag_at_peak,
            peak_corrcoef=corr,
            energy_out=energy,
            resampled_b=resampled_b,
            result_path=str(out),
            plot_path=plot_path,
        ).model_dump_json()
    except Exception as exc:
        return _fail(exc, default_hint=_OPERAND_HINT)


# --------------------------------------------------------------------------- #
# Tool 6: group_delay


def _group_delay(
    ctx: AppContext,
    num: list[float] | None = None,
    den: list[float] | None = None,
    zeros: list[list[float]] | None = None,
    poles: list[list[float]] | None = None,
    gain: float | None = None,
    domain: str = "s",
    dt: float | None = None,
    series_a: str | None = None,
    series_b: str | None = None,
    column_a: str | None = None,
    column_b: str | None = None,
    key_a: str | None = None,
    key_b: str | None = None,
    n_points: int = 512,
    save_plot: bool = False,
) -> str:
    try:
        cross_mode = series_a is not None or series_b is not None
        if cross_mode:
            if series_a is None or series_b is None:
                raise ValueError("cross-spectrum mode needs BOTH series_a= and series_b=")
            sa = tabular.load_series(series_a, column=column_a, key=key_a, cache_dir=ctx.cache_dir)
            sb = tabular.load_series(series_b, column=column_b, key=key_b, cache_dir=ctx.cache_dir)
            if sa.y.size != sb.y.size:
                raise ValueError(
                    f"series_a has {sa.y.size} samples but series_b has {sb.y.size} — "
                    "cross-spectral group delay needs equal-length, vertex-correspondent signals"
                )
            dt_opt = sa.dt if sa.dt is not None else (sb.dt if sb.dt is not None else dt)
            if dt_opt is None:
                raise ValueError(_NO_DT_MSG)
            grid_dt = float(dt_opt)
            if grid_dt <= 0.0:
                raise ValueError(f"dt must be positive, got {grid_dt}")
            w, tau = ltisys.group_delay_xspec(sa.y, sb.y, grid_dt)
            tau_seconds = tau
            unit = "seconds"
            mode = "cross_spectrum"
            report_domain = "cross-spectrum"
            report_dt: float | None = grid_dt
            stamp = _stamp("gdxs", str(series_a), str(series_b), sa.y.size, grid_dt)
            title = f"cross-spectral group delay ({Path(str(series_a)).name} vs {Path(str(series_b)).name})"
        else:
            n_points = int(n_points)
            if not 16 <= n_points <= 20_000:
                raise ValueError(f"n_points must be in [16, 20000], got {n_points}")
            sys_ = ltisys.build_system(num, den, zeros, poles, gain, domain, dt)
            z, p, k = ltisys.zpk_of(sys_)
            sys_dt = float(sys_.dt) if domain == "z" else None
            grid = ltisys.default_w_grid(z, p, domain, sys_dt, n_points)
            # Feed the raw num/den so scipy's normalize() cannot trim a small leading
            # numerator term (which would corrupt an FIR's group delay); zpk input
            # (no small-leading-coefficient pathology) falls back to the system's tf.
            ba = (num, den) if num is not None and den is not None else None
            w, tau, unit = ltisys.group_delay_lti(sys_, grid, domain, sys_dt, ba=ba)
            tau_seconds = tau * sys_dt if unit == "samples" and sys_dt is not None else tau
            mode = "lti"
            report_domain = domain
            report_dt = sys_dt
            stamp = _stamp("gdlti", num, den, zeros, poles, gain, domain, dt, n_points)
            title = f"group delay (domain={domain})"

        tau = np.asarray(tau, dtype=np.float64)
        if tau.size == 0:
            raise ValueError("group delay grid is empty")
        i_max = int(np.argmax(tau))
        out = _series_dir(ctx) / f"group_delay_{stamp}.npz"
        np.savez(out, w=w, tau=tau, tau_seconds=np.asarray(tau_seconds, dtype=np.float64))
        plot_path: str | None = None
        if save_plot:
            plot_path = str(
                _save_xy_plot(
                    [(w, tau, f"tau_g [{unit}]")],
                    ctx.plots_dir / f"group_delay_{stamp}.png",
                    title,
                    "w [rad/s]",
                    f"group delay [{unit}]",
                )
            )
        return GroupDelayReport(
            mode=mode,
            domain=report_domain,
            dt=report_dt,
            n_points=int(tau.size),
            tau_unit=unit,
            tau_mean=float(np.mean(tau)),
            tau_max=float(np.max(tau)),
            tau_min=float(np.min(tau)),
            tau_variance=float(np.var(tau)),
            freq_at_max_tau=float(w[i_max]),
            tau_at_dc=float(tau[0]),
            tau_mean_s=float(np.mean(np.asarray(tau_seconds, dtype=np.float64))),
            group_delay_path=str(out),
            plot_path=plot_path,
        ).model_dump_json()
    except Exception as exc:  # the MCP boundary must never see a raise
        return _fail(exc, default_hint=_GROUP_DELAY_HINT)


# --------------------------------------------------------------------------- #
# Tool 7: state_space_analysis


def _load_ss_matrix(
    ctx: AppContext, name: str, inline: list[list[float]] | None, path: str | None
) -> np.ndarray:
    if (inline is None) == (path is None):
        raise ValueError(
            f"pass exactly one of {name}= (inline rows) or {name}_path= (file) for matrix {name.upper()}"
        )
    if inline is not None:
        arr = np.asarray(inline, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                f"inline {name.upper()} must be a list of equal-length rows, got shape {arr.shape}"
            )
        return arr
    x, _names = tabular.load_matrix(path, cache_dir=ctx.cache_dir)
    return x


def _state_space_analysis(
    ctx: AppContext,
    a: list[list[float]] | None = None,
    b: list[list[float]] | None = None,
    c: list[list[float]] | None = None,
    a_path: str | None = None,
    b_path: str | None = None,
    c_path: str | None = None,
) -> str:
    try:
        mat_a = _load_ss_matrix(ctx, "a", a, a_path)
        mat_b = _load_ss_matrix(ctx, "b", b, b_path)
        mat_c: np.ndarray | None = None
        if c is not None or c_path is not None:
            mat_c = _load_ss_matrix(ctx, "c", c, c_path)
        res = statespace.analyze(mat_a, mat_b, mat_c)

        notes: list[str] = []
        if res.controllable:
            notes.append("controllable")
        else:
            notes.append(
                f"NOT controllable: {res.uncontrollable_dim} uncontrollable state direction(s) "
                f"(ctrb rank {res.ctrb_rank} < n_states {res.n_states})"
            )
        if res.observable is not None:
            if res.observable:
                notes.append("observable")
            else:
                notes.append(
                    f"NOT observable: {res.unobservable_dim} unobservable state direction(s) "
                    f"(obsv rank {res.obsv_rank} < n_states {res.n_states})"
                )
        return StateSpaceReport(
            n_states=res.n_states,
            n_inputs=res.n_inputs,
            ctrb_rank=res.ctrb_rank,
            controllable=res.controllable,
            uncontrollable_dim=res.uncontrollable_dim,
            ctrb_min_singular=res.ctrb_min_singular,
            n_outputs=res.n_outputs,
            obsv_rank=res.obsv_rank,
            observable=res.observable,
            unobservable_dim=res.unobservable_dim,
            obsv_min_singular=res.obsv_min_singular,
            note="; ".join(notes),
        ).model_dump_json()
    except Exception as exc:
        return _fail(exc, default_hint=_STATE_SPACE_HINT)


# --------------------------------------------------------------------------- #
# Registration


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the seven ELE301 systems tools on ``mcp``, bound to ``ctx``."""

    @mcp.tool()
    def lti_response(
        num: list[float] | None = None,
        den: list[float] | None = None,
        zeros: list[list[float]] | None = None,
        poles: list[list[float]] | None = None,
        gain: float | None = None,
        domain: str = "s",
        dt: float | None = None,
        input_kind: str = "step",
        input_path: str | None = None,
        input_key: str | None = None,
        t_end: float | None = None,
        n_samples: int = 500,
        save_plot: bool = False,
    ) -> str:
        """Time response of a CT ('s') or DT ('z', requires dt=sample period) LTI system given as
        num=+den= coefficient lists (descending powers of s; for 'z' polynomials in z^-1, so
        den=[1,-0.5] means 1-0.5z^-1) OR zeros=/poles= as [re,im] pairs plus gain= — exactly one
        form. input_kind is step|impulse|ramp|custom (custom loads u from input_path via the
        table loader; dt comes from the file or the dt= parameter — formats without one raise a
        ToolError saying to pass dt=). t_end defaults to ~7 slowest time constants (20 when all
        poles sit on the axis; n_samples*dt for DT). Returns LtiResponseReport JSON: stability,
        poles as [re,im], DC gain (None when a pole sits at s=0 / z=1), step metrics (final value,
        10-90% rise, 2% settling, overshoot %, peak time — step input on a stable system only),
        y range, and the path of the saved t/u/y arrays under data/series/ (never inline)."""
        return _lti_response(
            ctx,
            num,
            den,
            zeros,
            poles,
            gain,
            domain,
            dt,
            input_kind,
            input_path,
            input_key,
            t_end,
            n_samples,
            save_plot,
        )

    @mcp.tool()
    def pole_zero(
        num: list[float] | None = None,
        den: list[float] | None = None,
        zeros: list[list[float]] | None = None,
        poles: list[list[float]] | None = None,
        gain: float | None = None,
        domain: str = "s",
        dt: float | None = None,
        save_plot: bool = False,
    ) -> str:
        """Pole-zero map and dynamics reading of a CT/DT LTI system (same two input forms as
        lti_response). Per pole: damping ratio, natural frequency in rad/s, time constant in
        seconds (DT poles mapped via s = ln(z)/dt), and |z| for DT. Also: stability and marginal
        stability, the causal ROC as text ('Re(s) > -1' / '|z| > 0.9'), minimum-phase and proper
        flags. save_plot draws the s-plane (or z-plane with unit circle) map. Together with
        bode(domain='z') this covers geometric DTFT evaluation from pole-zero plots. Returns
        PoleZeroReport JSON."""
        return _pole_zero(ctx, num, den, zeros, poles, gain, domain, dt, save_plot)

    @mcp.tool()
    def bode(
        num: list[float] | None = None,
        den: list[float] | None = None,
        zeros: list[list[float]] | None = None,
        poles: list[list[float]] | None = None,
        gain: float | None = None,
        domain: str = "s",
        dt: float | None = None,
        w_min: float | None = None,
        w_max: float | None = None,
        n_points: int = 400,
        save_plot: bool = False,
    ) -> str:
        """Bode magnitude/phase over a log frequency grid (auto-ranged 2 decades beyond the
        pole/zero spread, DT capped at pi/dt) with stability margins computed by interpolated
        crossover search, treating the given H as the OPEN-loop transfer function: gain margin at
        the -180 deg phase crossover, phase margin at the 0 dB gain crossover (None when there is
        no crossing). Also -3 dB bandwidth relative to the DC gain, resonant peak, and the
        high-frequency slope in dB/decade (~ -20*(#poles - #zeros)). dc_gain_db is None when a
        pole sits at the origin (s=0 / z=1). The w/mag_db/phase_deg arrays are saved under
        data/series/ (path in the JSON). Returns BodeReport JSON."""
        return _bode(ctx, num, den, zeros, poles, gain, domain, dt, w_min, w_max, n_points, save_plot)

    @mcp.tool()
    def sampling_check(
        fs: float,
        series_path: str | None = None,
        column: str | None = None,
        key: str | None = None,
        values: list[float] | None = None,
        dt: float | None = None,
        dump: str | None = None,
        joint: str = "armLowerL",
        channel: str = "posed",
        bone_map_path: str | None = None,
        save_plot: bool = False,
    ) -> str:
        """Aliasing verdict for a proposed sampling rate fs against a well-sampled reference
        signal — exactly one source: series_path= (any table/series file; dt from the file or the
        dt= parameter — a ToolError says 'pass dt=' when the format has none), values= (inline
        list <= 4096, requires dt=), or dump= + joint/channel (engine PLY forearm profile:
        frequencies become cycles/meter and fs is samples/meter — the geometry-lane bridge; a bad
        joint gets the per-joint vertex histogram hint). Verdict thresholds on the energy fraction
        above fs/2: safe < 1%, marginal < 5%, else aliased; fs_suggested = 2.5x the 99%-energy
        bandwidth. Top spectral peaks are reported with their folded alias frequencies. Fails with
        'insufficient_resolution' when fs/2 sits at/above 90% of the reference's own Nyquist.
        Related: harmonic decomposition -> fourier_series (engmath); geometry corrugation verdicts
        -> analyze_corrugation. Returns SamplingCheckReport JSON."""
        return _sampling_check(
            ctx, fs, series_path, column, key, values, dt, dump, joint, channel, bone_map_path, save_plot
        )

    @mcp.tool()
    def convolve_signals(
        path_a: str | None = None,
        path_b: str | None = None,
        key_a: str | None = None,
        key_b: str | None = None,
        column_a: str | None = None,
        column_b: str | None = None,
        values_a: list[float] | None = None,
        values_b: list[float] | None = None,
        dt: float | None = None,
        mode: str = "convolve",
        scale: str = "continuous",
        normalize: bool = False,
        save_plot: bool = False,
    ) -> str:
        """Convolution or cross-correlation of two series (each from a file via the table loader
        or an inline values list <= 4096; formats without a dt need the dt= parameter). scale=
        'continuous' multiplies the discrete sum by dt to approximate the CT integral (rect*rect
        gives a unit triangle, not N); 'discrete' is the raw sum. Files carrying different dt get
        b resampled onto a's grid (rational anti-aliased resampling; resampled_b=true in the
        report). mode='correlate' reports lag_at_peak with scipy's convention c[k] =
        sum a[n]*b[n-k]: a delayed copy in b peaks at NEGATIVE lag. normalize=true additionally
        reports the overlap-normalized correlation coefficient at that peak (1.0 = perfect aligned
        match). The result array is saved under data/series/ (path in the JSON). Returns
        ConvolutionReport JSON."""
        return _convolve_signals(
            ctx,
            path_a,
            path_b,
            key_a,
            key_b,
            column_a,
            column_b,
            values_a,
            values_b,
            dt,
            mode,
            scale,
            normalize,
            save_plot,
        )

    @mcp.tool()
    def group_delay(
        num: list[float] | None = None,
        den: list[float] | None = None,
        zeros: list[list[float]] | None = None,
        poles: list[list[float]] | None = None,
        gain: float | None = None,
        domain: str = "s",
        dt: float | None = None,
        series_a: str | None = None,
        series_b: str | None = None,
        column_a: str | None = None,
        column_b: str | None = None,
        key_a: str | None = None,
        key_b: str | None = None,
        n_points: int = 512,
        save_plot: bool = False,
    ) -> str:
        """Group delay tau_g(w) = -d(phase)/dw in one of two modes. LTI mode: pass a system as
        num=+den= or zeros=/poles=/gain= (same two forms as lti_response/pole_zero/bode) with
        domain='s'|'z' (z requires dt=), evaluated over the auto-ranged log grid from bode.
        Digital reports tau in SAMPLES (scipy.signal.group_delay) plus tau_mean_s in seconds; analog
        reports SECONDS via -gradient(unwrap(angle(H(jw))), w). Cross-spectrum mode: pass series_a=
        and series_b= (equal-length, vertex-correspondent series files — e.g. an engine-posed forearm
        profile vs its pure-numpy LBS reconstruction; add dt= when the format carries no sample
        spacing) to measure the frequency-dependent LAG between them (positive tau = series_a lags
        series_b) via Sab = rfft(a)*conj(rfft(b)). A linear-phase FIR gives a flat tau = (N-1)/2
        samples. The (w, tau, tau_seconds) arrays are saved under data/series/ (path in the JSON).
        Related: pole_zero for the phase's pole/zero origin, bode for the magnitude/phase curves.
        Returns GroupDelayReport JSON."""
        return _group_delay(
            ctx,
            num,
            den,
            zeros,
            poles,
            gain,
            domain,
            dt,
            series_a,
            series_b,
            column_a,
            column_b,
            key_a,
            key_b,
            n_points,
            save_plot,
        )

    @mcp.tool()
    def state_space_analysis(
        a: list[list[float]] | None = None,
        b: list[list[float]] | None = None,
        c: list[list[float]] | None = None,
        a_path: str | None = None,
        b_path: str | None = None,
        c_path: str | None = None,
    ) -> str:
        """Definitively answer, for a linear state model x' = A x + B u (y = C x): can the inputs
        drive the system to ANY state (CONTROLLABLE) and — when C is given — can the outputs
        reconstruct the full state (OBSERVABLE)? Pass A (square n x n) and B (n x m) as inline rows
        a=[[...]] / b=[[...]] OR as files a_path=/b_path= (csv/tsv/json/npz/npy); add C (p x n) via
        c=/c_path= to also test observability. Uses the Kalman rank tests: controllability matrix
        [B, AB, ..., A^(n-1)B] and observability matrix [C; CA; ...; CA^(n-1)], each ranked by SVD.
        When a rank is deficient the note names the deficient dimension (uncontrollable_dim /
        unobservable_dim), and ctrb_min_singular / obsv_min_singular give the numerical margin to
        rank loss. Complements lti_response / pole_zero (the transfer-function view of the same
        dynamics). Returns StateSpaceReport JSON; shape mismatches return ToolError JSON."""
        return _state_space_analysis(ctx, a, b, c, a_path, b_path, c_path)

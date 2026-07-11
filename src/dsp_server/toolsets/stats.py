"""Stats tool pack (ELE320): descriptive statistics, distribution fitting,
hypothesis tests, OLS regression, and the two-dump ripple significance test.

Five tools, same contract as the geometry pack: every tool returns a pydantic
model's ``model_dump_json()``; every failure returns a :class:`ToolError` whose
hint carries the TabularError message verbatim (it already lists the columns/keys
found) or, for joint-selection failures, the dump's per-joint vertex histogram.
Data tools route through :mod:`dsp_server.engine.tabular` — .csv/.tsv/.json/
.npz/.npy plus engine ``.ply`` dumps via ``column="<joint>[:posed|rest]"``.

Testability: each registered tool is a thin closure over :class:`AppContext`
delegating to a module-level ``_impl`` function that tests call directly.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP

from dsp_server import plots
from dsp_server.engine import ply, statskit, tabular, transforms
from dsp_server.engine.filters import spectral_peaks
from dsp_server.engine.tabular import TabularError
from dsp_server.schemas import SchemaBase, ToolError
from dsp_server.toolsets import AppContext

_TINY = 1e-30
_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]")
_PLY_SOURCE_NOTE = "n = axial profile samples, not vertices"


# --------------------------------------------------------------------------- #
# Response models (pack-local, on the shared SchemaBase contract)


class DescribeReport(SchemaBase):
    """describe: moments, quantiles, and CIs for one numeric series."""

    path: str
    column: str
    n: int
    n_dropped_nan: int
    mean: float
    std: float
    variance: float
    min: float
    max: float
    median: float
    skewness: float
    kurtosis_excess: float
    quantiles: dict[str, float]
    ci_level: float
    ci_mean_lo: float
    ci_mean_hi: float
    ci_std_lo: float
    ci_std_hi: float
    normality_test: str
    normality_stat: float
    normality_p: float
    source_note: str | None = None  # .ply inputs: n counts profile samples, not vertices


class DistFitSchema(SchemaBase):
    """One fitted candidate distribution (params in scipy shape order + loc/scale)."""

    name: str
    params: dict[str, float]
    loglik: float
    k_params: int
    aic: float
    bic: float
    ks_stat: float
    ks_p: float
    chi2_stat: float | None = None
    chi2_p: float | None = None
    chi2_dof: int | None = None
    loc_fixed: bool


class FitDistributionReport(SchemaBase):
    """fit_distribution: candidates AIC-ascending; best_dist = lowest AIC."""

    path: str
    column: str
    n: int
    n_dropped_nan: int
    candidates: list[DistFitSchema]
    best_dist: str
    skipped: dict[str, str]
    plot_path: str | None = None


class HypothesisTestReport(SchemaBase):
    """hypothesis_test: one classical test, effect size, and mean-difference CI."""

    test_used: str
    alternative: str
    alpha: float
    n_a: int
    n_b: int | None = None
    n_dropped_a: int
    n_dropped_b: int | None = None
    mean_a: float
    mean_b: float | None = None
    std_a: float
    std_b: float | None = None
    statistic: float
    p_value: float
    df: float | None = None
    reject_h0: bool
    effect_size: float | None = None
    effect_size_kind: str | None = None
    ci_diff_lo: float | None = None
    ci_diff_hi: float | None = None


class CoefficientSchema(SchemaBase):
    name: str
    estimate: float
    std_err: float
    t_stat: float
    p_value: float
    ci_lo: float
    ci_hi: float


class RegressionReport(SchemaBase):
    """regression_fit: OLS with intercept + the full ELE320 diagnostic block."""

    path: str
    y_column: str
    x_columns: list[str]
    n: int
    n_dropped_nan: int
    coefficients: list[CoefficientSchema]
    r2: float
    r2_adj: float
    f_stat: float
    f_p: float
    rmse: float
    durbin_watson: float
    resid_normality_test: str
    resid_normality_stat: float
    resid_normality_p: float
    condition_number: float
    plot_path: str | None = None


class BootstrapRatioSchema(SchemaBase):
    n_boot: int
    ci_ratio_lo: float
    ci_ratio_hi: float
    ci_a_lo: float
    ci_a_hi: float
    ci_b_lo: float
    ci_b_hi: float
    significant: bool


class SectorTestSchema(SchemaBase):
    n_sectors_used: int
    test_name: str
    statistic: float
    p_value: float
    significant: bool


class RippleComparisonReport(SchemaBase):
    """compare_dump_ripples: band-RMS ratio + bootstrap CI + unpaired sector test."""

    dump_a: str
    dump_b: str
    channel_a: str
    channel_b: str
    joint: str
    joint_ids: list[int]
    vertex_count_a: int
    vertex_count_b: int
    f_lo_cpm: float
    f_hi_cpm: float
    band_source: str
    band_rms_a: float
    band_rms_b: float
    ratio_b_over_a: float
    bootstrap: BootstrapRatioSchema | None = None
    sector: SectorTestSchema | None = None
    verdict: str
    assumptions: list[str]


# --------------------------------------------------------------------------- #
# Shared helpers


def _safe(text: str) -> str:
    return _SAFE_RE.sub("_", text) or "x"


def _error(exc: Exception, hint: str | None = None) -> str:
    if hint is None and isinstance(exc, TabularError):
        hint = str(exc)  # tabular messages are hint-ready (they list what WAS found)
    return ToolError(error=f"{type(exc).__name__}: {exc}", hint=hint).model_dump_json()


def _joint_histogram(dump: ply.EngineDump, bone_map: dict[int, str]) -> dict[str, int]:
    uniq, counts = np.unique(dump.dominant_joint, return_counts=True)
    out: dict[str, int] = {}
    for j, c in zip(uniq.tolist(), counts.tolist(), strict=True):
        name = bone_map.get(int(j))
        key = f"{name} ({int(j)})" if name else str(int(j))
        out[key] = int(c)
    return out


def _histogram_hint(dump: ply.EngineDump | None, bone_map: dict[int, str]) -> str | None:
    if dump is None:
        return "could not load the dump — check the path points at an engine ASCII PLY"
    hist = ", ".join(f"{k}={v}" for k, v in _joint_histogram(dump, bone_map).items())
    return f"per-joint vertex histogram of this dump: {{{hist}}} — pick a joint name or integer id from it"


def _pick_column(table: dict[str, np.ndarray], name: str) -> str:
    """Kernel-compatible column resolution (exact, unique case-insensitive, index)."""
    names = list(table)
    if name in table:
        return name
    ci = [n for n in names if n.lower() == str(name).lower()]
    if len(ci) == 1:
        return ci[0]
    if str(name).isdigit() and 0 <= int(name) < len(names):
        return names[int(name)]
    raise TabularError(
        f"column '{name}' not found — columns present: {names} "
        f"(pass a name or an integer index 0..{len(names) - 1})"
    )


def _source_note(source: str) -> str | None:
    return _PLY_SOURCE_NOTE if Path(source).suffix.lower() == ".ply" else None


# --------------------------------------------------------------------------- #
# Plot helpers (pack-local; dsp_server.plots locked the Agg backend already)


def _save_fit_plot(y: np.ndarray, fits: list[statskit.DistFit], path: Path, title: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = plots.plt
    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    n_bins = min(60, max(10, int(math.sqrt(y.size))))
    ax.hist(y, bins=n_bins, density=True, alpha=0.35, color="tab:gray", label="data")
    xs = np.linspace(float(np.min(y)), float(np.max(y)), 400)
    for f in fits:
        frozen = statskit.frozen_from_params(f.name, f.params)
        with np.errstate(all="ignore"):
            pdf = np.asarray(frozen.pdf(xs), dtype=np.float64)
        ax.plot(xs, np.where(np.isfinite(pdf), pdf, np.nan), lw=1.2, label=f"{f.name} (AIC {f.aic:.1f})")
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _save_regression_plot(
    x_axis: np.ndarray,
    xlabel: str,
    y: np.ndarray,
    y_hat: np.ndarray,
    resid: np.ndarray,
    path: Path,
    title: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = plots.plt
    fig, (ax_fit, ax_res) = plt.subplots(2, 1, figsize=(9.0, 6.5), constrained_layout=True)
    order = np.argsort(x_axis)
    ax_fit.scatter(x_axis, y, s=12, alpha=0.6, color="tab:blue", label="data")
    ax_fit.plot(x_axis[order], y_hat[order], lw=1.5, color="tab:red", label="fit")
    ax_fit.set_xlabel(xlabel)
    ax_fit.set_ylabel("y")
    ax_fit.set_title("fit")
    ax_fit.grid(True, alpha=0.3)
    ax_fit.legend(loc="best", fontsize=8)
    ax_res.scatter(y_hat, resid, s=12, alpha=0.6, color="tab:blue")
    ax_res.axhline(0.0, color="tab:red", lw=1.0)
    ax_res.set_xlabel("fitted")
    ax_res.set_ylabel("residual")
    ax_res.set_title("residuals vs fitted")
    ax_res.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Tool 1: describe


def _describe(
    ctx: AppContext,
    path: str,
    column: str | None = None,
    key: str | None = None,
    ci_level: float = 0.95,
) -> str:
    try:
        series = tabular.load_series(path, column=column, key=key, cache_dir=ctx.cache_dir)
        st = statskit.describe_series(series.y, ci_level=ci_level)
        return DescribeReport(
            path=series.source,
            column=series.column,
            n_dropped_nan=series.n_dropped,
            source_note=_source_note(series.source),
            **asdict(st),
        ).model_dump_json()
    except Exception as exc:  # the MCP boundary must never see a raise
        return _error(exc)


# --------------------------------------------------------------------------- #
# Tool 2: fit_distribution


def _fit_distribution(
    ctx: AppContext,
    path: str,
    column: str | None = None,
    key: str | None = None,
    distributions: list[str] | None = None,
    save_plot: bool = False,
) -> str:
    try:
        series = tabular.load_series(path, column=column, key=key, cache_dir=ctx.cache_dir)
        names = list(distributions) if distributions else list(statskit.DEFAULT_DISTRIBUTIONS)
        fits, skipped = statskit.fit_candidates(series.y, names)
        if not fits:
            details = "; ".join(f"{k}: {v}" for k, v in skipped.items())
            return ToolError(
                error="every candidate distribution failed to fit",
                hint=details or "the distribution shortlist was empty",
            ).model_dump_json()
        plot_path: str | None = None
        if save_plot:
            out = ctx.plots_dir / f"fit_{_safe(Path(series.source).stem)}_{_safe(series.column)}.png"
            plot_path = str(_save_fit_plot(series.y, fits[:4], out, f"{series.column} — top fits by AIC"))
        return FitDistributionReport(
            path=series.source,
            column=series.column,
            n=int(series.y.size),
            n_dropped_nan=series.n_dropped,
            candidates=[DistFitSchema(**asdict(f)) for f in fits],
            best_dist=fits[0].name,
            skipped=skipped,
            plot_path=plot_path,
        ).model_dump_json()
    except Exception as exc:
        return _error(exc)


# --------------------------------------------------------------------------- #
# Tool 3: hypothesis_test


def _hypothesis_test(
    ctx: AppContext,
    path_a: str,
    path_b: str | None = None,
    column_a: str | None = None,
    column_b: str | None = None,
    key_a: str | None = None,
    key_b: str | None = None,
    test: str = "auto",
    alternative: str = "two-sided",
    mu0: float = 0.0,
    alpha: float = 0.05,
) -> str:
    try:
        series_a = tabular.load_series(path_a, column=column_a, key=key_a, cache_dir=ctx.cache_dir)
        series_b = (
            tabular.load_series(path_b, column=column_b, key=key_b, cache_dir=ctx.cache_dir)
            if path_b is not None
            else None
        )
        a = series_a.y
        b = series_b.y if series_b is not None else None
        result = statskit.run_test(a, b, test=test, alternative=alternative, mu0=mu0, alpha=alpha)
        return HypothesisTestReport(
            test_used=result.test_used,
            alternative=result.alternative_used,
            alpha=alpha,
            n_a=int(a.size),
            n_b=int(b.size) if b is not None else None,
            n_dropped_a=series_a.n_dropped,
            n_dropped_b=series_b.n_dropped if series_b is not None else None,
            mean_a=float(a.mean()),
            mean_b=float(b.mean()) if b is not None else None,
            std_a=float(a.std(ddof=1)),
            std_b=float(b.std(ddof=1)) if b is not None else None,
            statistic=result.statistic,
            p_value=result.p_value,
            df=result.df,
            reject_h0=bool(result.p_value < alpha),
            effect_size=result.effect_size,
            effect_size_kind=result.effect_size_kind,
            ci_diff_lo=result.ci_diff[0] if result.ci_diff is not None else None,
            ci_diff_hi=result.ci_diff[1] if result.ci_diff is not None else None,
        ).model_dump_json()
    except Exception as exc:
        return _error(exc)


# --------------------------------------------------------------------------- #
# Tool 4: regression_fit

_MAX_REGRESSORS = 32  # coefficient rows stay a compact inline list, never an array dump


def _regression_fit(
    ctx: AppContext,
    path: str,
    y_column: str,
    x_columns: list[str] | None = None,
    key: str | None = None,
    ci_level: float = 0.95,
    save_plot: bool = False,
) -> str:
    try:
        table = tabular.load_table(path, key=key, cache_dir=ctx.cache_dir)
        y_name = _pick_column(table, y_column)
        if x_columns is None:
            x_names = [c for c in table if c != y_name]
        else:
            x_names = []
            for c in x_columns:
                resolved = _pick_column(table, c)
                if resolved != y_name and resolved not in x_names:
                    x_names.append(resolved)
        if not x_names:
            return ToolError(
                error="no regressor columns",
                hint=(f"columns present: {list(table)} — pass x_columns different from y_column '{y_name}'"),
            ).model_dump_json()
        if len(x_names) > _MAX_REGRESSORS:
            return ToolError(
                error=f"{len(x_names)} regressors requested — cap is {_MAX_REGRESSORS}",
                hint="pass x_columns to select a smaller design",
            ).model_dump_json()
        sub = {y_name: table[y_name], **{c: table[c] for c in x_names}}
        clean, n_dropped = tabular.drop_nan_rows(sub)
        y = clean[y_name]
        n = int(y.size)
        design = np.column_stack([np.ones(n), *[clean[c] for c in x_names]])
        names = ["intercept", *x_names]
        res = statskit.ols(design, y, names, ci_level=ci_level)
        norm_name, norm_stat, norm_p = statskit.normality_test(res.resid)
        plot_path: str | None = None
        if save_plot:
            out = ctx.plots_dir / f"regression_{_safe(Path(path).stem)}_{_safe(y_name)}.png"
            y_hat = design @ res.coef
            if len(x_names) == 1:
                plot_path = str(
                    _save_regression_plot(
                        clean[x_names[0]],
                        x_names[0],
                        y,
                        y_hat,
                        res.resid,
                        out,
                        f"{y_name} ~ {x_names[0]}",
                    )
                )
            else:
                plot_path = str(
                    _save_regression_plot(
                        y_hat,
                        "fitted",
                        y,
                        y_hat,
                        res.resid,
                        out,
                        f"{y_name} ~ {len(x_names)} regressors (fitted vs actual)",
                    )
                )
        coefficients = [
            CoefficientSchema(
                name=res.names[i],
                estimate=float(res.coef[i]),
                std_err=float(res.std_err[i]),
                t_stat=float(res.t_stat[i]),
                p_value=float(res.p_value[i]),
                ci_lo=float(res.ci_lo[i]),
                ci_hi=float(res.ci_hi[i]),
            )
            for i in range(len(res.names))
        ]
        return RegressionReport(
            path=str(path),
            y_column=y_name,
            x_columns=x_names,
            n=n,
            n_dropped_nan=n_dropped,
            coefficients=coefficients,
            r2=res.r2,
            r2_adj=res.r2_adj,
            f_stat=res.f_stat,
            f_p=res.f_p,
            rmse=res.rmse,
            durbin_watson=res.durbin_watson,
            resid_normality_test=norm_name,
            resid_normality_stat=norm_stat,
            resid_normality_p=norm_p,
            condition_number=res.cond,
            plot_path=plot_path,
        ).model_dump_json()
    except Exception as exc:
        return _error(exc)


# --------------------------------------------------------------------------- #
# Tool 5: compare_dump_ripples


def _compare_dump_ripples(
    ctx: AppContext,
    dump_a: str,
    dump_b: str,
    joint: str = "armLowerL",
    channel_a: str = "posed",
    channel_b: str = "posed",
    method: str = "both",
    n_boot: int = 500,
    n_sectors: int = 8,
    alpha: float = 0.05,
    f_lo_cpm: float | None = None,
    f_hi_cpm: float | None = None,
    bone_map_path: str | None = None,
    seed: int = 0,
) -> str:
    loaded_a: ply.EngineDump | None = None
    bone_map: dict[int, str] = {}
    try:
        if method not in ("bootstrap", "sector", "both"):
            return ToolError(
                error=f"unknown method '{method}'",
                hint="expected one of bootstrap|sector|both",
            ).model_dump_json()
        if (f_lo_cpm is None) != (f_hi_cpm is None):
            return ToolError(
                error="explicit band needs BOTH f_lo_cpm and f_hi_cpm",
                hint="pass both bounds or neither (auto band = +/-30% around side B's dominant peak)",
            ).model_dump_json()
        path_a, path_b = Path(dump_a), Path(dump_b)
        loaded_a = ply.load_dump_cached(path_a, ctx.cache_dir)
        loaded_b = ply.load_dump_cached(path_b, ctx.cache_dir)
        # Resolve the joint PER DUMP (two skeletons may index the same name differently).
        map_a, map_b = ply.bone_map_for(path_a), ply.bone_map_for(path_b)
        bone_map = {**map_a, **map_b}  # for the error-hint histogram only
        joint_ids_a = transforms.resolve_joint_ids(
            joint, bone_map=(map_a or map_b) or None, bone_map_path=bone_map_path
        )
        joint_ids_b = transforms.resolve_joint_ids(
            joint, bone_map=(map_b or map_a) or None, bone_map_path=bone_map_path
        )
        mask_a = transforms.select_segment(loaded_a, joint_ids_a, channel=channel_a)
        mask_b = transforms.select_segment(loaded_b, joint_ids_b, channel=channel_b)
        pts_a = loaded_a.channel(channel_a)[mask_a]
        pts_b = loaded_b.channel(channel_b)[mask_b]
        # ONE cylinder frame, fitted on side A and REUSED for side B, so the axial
        # coordinate and the angular zero mean the same thing on both sides
        # (same-joint comparison; see `assumptions` in the report).
        frame = transforms.fit_axis(pts_a)
        t_a, r_a, th_a = transforms.cylindrical_coords(pts_a, frame)
        t_b, r_b, th_b = transforms.cylindrical_coords(pts_b, frame)
        sig_a = transforms.axial_profile(t_a, r_a)
        sig_b = transforms.axial_profile(t_b, r_b)

        if f_lo_cpm is not None and f_hi_cpm is not None:
            band_source = "explicit"
            f_lo, f_hi = float(f_lo_cpm), float(f_hi_cpm)
        else:
            freqs_b, psd_b = transforms.spectrum(statskit.rel_signal(sig_b))
            peaks = spectral_peaks(freqs_b, psd_b, f_min_cpm=12.0, max_peaks=1)
            if peaks:
                dom = peaks[0][0]
                band_source = "dominant_peak_b"
                f_lo, f_hi = 0.7 * dom, 1.3 * dom
            else:
                band_source = "fallback_highband"
                f_lo, f_hi = 12.0, 0.9 * (0.5 / sig_b.dt)
        rms_a = statskit.band_rms(sig_a, f_lo, f_hi)
        rms_b = statskit.band_rms(sig_b, f_lo, f_hi)
        ratio = rms_b / max(rms_a, _TINY)

        boot_schema: BootstrapRatioSchema | None = None
        sector_schema: SectorTestSchema | None = None
        votes: list[tuple[bool, bool]] = []  # (significant at alpha, direction b_larger)
        assumptions = [
            "axis frame fitted on dump A's masked vertices and reused for dump B — axial and "
            "angular coordinates are comparable only when both dumps pose the same joint similarly"
        ]
        if method in ("bootstrap", "both"):
            boot = statskit.bootstrap_band_ratio(
                t_a, r_a, t_b, r_b, f_lo, f_hi, n_boot=n_boot, seed=seed, alpha=alpha
            )
            boot_significant = not (boot.ci_ratio[0] <= 1.0 <= boot.ci_ratio[1])
            boot_schema = BootstrapRatioSchema(
                n_boot=boot.n_boot,
                ci_ratio_lo=boot.ci_ratio[0],
                ci_ratio_hi=boot.ci_ratio[1],
                ci_a_lo=boot.ci_a[0],
                ci_a_hi=boot.ci_a[1],
                ci_b_lo=boot.ci_b[0],
                ci_b_hi=boot.ci_b[1],
                significant=boot_significant,
            )
            votes.append((boot_significant, boot.ratio > 1.0))
            assumptions.append(
                "bootstrap resamples vertex indices within each side's joint mask — vertices are "
                "treated as exchangeable measurement samples of ONE surface realization (it "
                "quantifies sampling noise, not pose-to-pose variance)"
            )
        if method in ("sector", "both"):
            sectors_a = transforms.sector_profiles(t_a, r_a, th_a, n_sectors=n_sectors)
            sectors_b = transforms.sector_profiles(t_b, r_b, th_b, n_sectors=n_sectors)
            sec = statskit.sector_band_test(sectors_a, sectors_b, f_lo, f_hi)
            sec_significant = bool(sec.p_value < alpha)
            sector_schema = SectorTestSchema(
                n_sectors_used=sec.n_sectors_used,
                test_name=sec.test_name,
                statistic=sec.statistic,
                p_value=sec.p_value,
                significant=sec_significant,
            )
            votes.append((sec_significant, sec.b_larger))
            assumptions.append(
                "sector test is UNPAIRED (Mann-Whitney U across per-sector band RMS): "
                "sector_profiles drops empty wedges without index bookkeeping and cross-dump "
                "angular alignment is frame-deep only, so positional pairing would pair noise; "
                "sectors also share one deformation field (quasi-independence)"
            )

        if not any(sig for sig, _ in votes):
            verdict = "no_significant_difference"
        elif all(sig for sig, _ in votes) and len({direction for _, direction in votes}) == 1:
            verdict = "b_ripplier" if votes[0][1] else "a_ripplier"
        else:
            verdict = "methods_disagree"

        return RippleComparisonReport(
            dump_a=str(path_a),
            dump_b=str(path_b),
            channel_a=channel_a,
            channel_b=channel_b,
            joint=joint,
            joint_ids=sorted(set(joint_ids_a) | set(joint_ids_b)),
            vertex_count_a=int(mask_a.sum()),
            vertex_count_b=int(mask_b.sum()),
            f_lo_cpm=f_lo,
            f_hi_cpm=f_hi,
            band_source=band_source,
            band_rms_a=rms_a,
            band_rms_b=rms_b,
            ratio_b_over_a=ratio,
            bootstrap=boot_schema,
            sector=sector_schema,
            verdict=verdict,
            assumptions=assumptions,
        ).model_dump_json()
    except Exception as exc:
        return _error(exc, hint=_histogram_hint(loaded_a, bone_map))


# --------------------------------------------------------------------------- #
# Registration


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the five stats tools on ``mcp``, bound to ``ctx``."""

    @mcp.tool()
    def describe(
        path: str,
        column: str | None = None,
        key: str | None = None,
        ci_level: float = 0.95,
    ) -> str:
        """Moments, quantiles, and confidence intervals for one numeric series from
        .csv/.tsv/.json/.npz/.npy — or an engine .ply dump via column="<joint>[:posed|rest]"
        (the 256-sample binned axial radius profile; the report's source_note then reads
        'n = axial profile samples, not vertices' — no vertex-level inference is implied).
        Returns DescribeReport JSON: n and n_dropped_nan (non-finite values dropped by the
        loader), mean/std/variance/min/max/median, unbiased skewness and excess kurtosis,
        p05..p95 quantiles, a Student-t interval on the mean and a chi-square interval on the
        std at ci_level, plus a normality test (shapiro for n <= 5000, else normaltest;
        normality_test names which ran). key selects an .npz member. On a bad column the
        ToolError hint lists the columns actually present in the file."""
        return _describe(ctx, path, column, key, ci_level)

    @mcp.tool()
    def fit_distribution(
        path: str,
        column: str | None = None,
        key: str | None = None,
        distributions: list[str] | None = None,
        save_plot: bool = False,
    ) -> str:
        """MLE-fit a shortlist of scipy distributions to one numeric series and rank them by
        AIC (best_dist = lowest). Default shortlist: norm, lognorm, expon, gamma, weibull_min,
        rayleigh, uniform (rayleigh fits ripple envelopes — the RCA hook). Positive-support
        candidates are fitted with loc pinned at 0 iff min(y) > 0 (echoed as loc_fixed), else
        unconstrained. Each candidate carries params (scipy shape names + loc/scale), loglik,
        k_params (free params only), aic, bic, a KS test (ks_p is OPTIMISTIC because the params
        were fitted on the same data — rank by AIC and confirm with chi2_p) and an
        equal-probability chi-square GOF (the pack's GLR/goodness-of-fit lane; chi2_* are null
        when n is too small for valid bins). Unfittable candidates land in skipped with the
        reason; if every candidate is skipped the tool returns ToolError. save_plot writes a
        density histogram overlaid with the top-4 fitted pdfs and returns its path."""
        return _fit_distribution(ctx, path, column, key, distributions, save_plot)

    @mcp.tool()
    def hypothesis_test(
        path_a: str,
        path_b: str | None = None,
        column_a: str | None = None,
        column_b: str | None = None,
        key_a: str | None = None,
        key_b: str | None = None,
        test: str = "auto",
        alternative: str = "two-sided",
        mu0: float = 0.0,
        alpha: float = 0.05,
    ) -> str:
        """One classical hypothesis test per call. test in {auto, t_one_sample, t_two_sample,
        welch, paired_t, mannwhitney, ks2, f_var, levene}; auto routes one sample to
        t_one_sample against mu0 and two samples to welch (variance-robust default) — the
        choice is echoed in test_used. f_var is the variance-ratio F test (F = s_a^2/s_b^2,
        dof (n_a-1, n_b-1), two-sided p = 2*min(sf, cdf) clipped to 1); levene(center='median')
        ignores 'alternative' and echoes alternative='two-sided (forced)'. paired_t requires
        equal n (the error names both lengths); every two-sample test requires path_b. Effect
        size: pooled Cohen's d for the t family, rank-biserial r = 2U/(n_a*n_b)-1 for
        mannwhitney (effect_size_kind names it). The t family also reports the CI on the
        (difference of) means at level 1-alpha (ci_diff_lo/hi). Returns HypothesisTestReport
        JSON with per-sample n/mean/std, statistic, p_value, df, and reject_h0 = p < alpha."""
        return _hypothesis_test(
            ctx, path_a, path_b, column_a, column_b, key_a, key_b, test, alternative, mu0, alpha
        )

    @mcp.tool()
    def regression_fit(
        path: str,
        y_column: str,
        x_columns: list[str] | None = None,
        key: str | None = None,
        ci_level: float = 0.95,
        save_plot: bool = False,
    ) -> str:
        """OLS with an always-included intercept and the full diagnostic block.
        x_columns=None regresses y_column on every other numeric column; NaN rows are dropped
        jointly across the selected columns (count in n_dropped_nan). Per coefficient
        ('intercept' first): estimate, std_err, t_stat, two-sided p_value, and a ci_level CI.
        Global: r2/r2_adj, the overall F test (f_stat/f_p), rmse, the Durbin-Watson statistic
        (serial correlation of residuals), residual normality (shapiro for n <= 5000 else
        normaltest), and the condition number of the standardized design (multicollinearity
        flag; > ~30 is trouble). save_plot writes scatter+fit over residuals-vs-fitted (single
        regressor) or fitted-vs-actual over residuals (multiple). At most 32 regressors —
        the coefficient table stays a compact inline list. On a bad column the ToolError hint
        lists the columns present."""
        return _regression_fit(ctx, path, y_column, x_columns, key, ci_level, save_plot)

    @mcp.tool()
    def compare_dump_ripples(
        dump_a: str,
        dump_b: str,
        joint: str = "armLowerL",
        channel_a: str = "posed",
        channel_b: str = "posed",
        method: str = "both",
        n_boot: int = 500,
        n_sectors: int = 8,
        alpha: float = 0.05,
        f_lo_cpm: float | None = None,
        f_hi_cpm: float | None = None,
        bone_map_path: str | None = None,
        seed: int = 0,
    ) -> str:
        """Is the ripple difference between two engine dumps STATISTICALLY SIGNIFICANT?
        (For descriptive per-side roughness and deltas use compare_geometry_signals; this tool
        adds the inference on top.) Statistic: band RMS of the relative-radius profile
        (r/median(r)-1, cubic-detrended) over a band chosen as +/-30% around side B's dominant
        spectral peak (f_min 12 cpm), falling back to [12 cpm, 0.9*Nyquist], or explicit
        f_lo_cpm/f_hi_cpm — band_source echoes the choice. The cylinder frame is fitted on
        dump A and REUSED for dump B so axial/angular zero match (same-joint comparison; see
        assumptions). Two approximations, run per `method` (bootstrap|sector|both): a seeded
        vertex-index bootstrap CI on the band-RMS ratio b/a (significant when the CI excludes
        1), and an UNPAIRED Mann-Whitney U across per-sector band RMS (pairing was dropped:
        sector_profiles discards empty wedges without index bookkeeping and cross-dump angular
        alignment is frame-deep only, so positional pairing would pair noise). Agreement of the
        two approximations is the actionable signal: verdict is b_ripplier/a_ripplier only when
        the run methods agree in direction and are each significant at alpha;
        methods_disagree means collect more poses, not a verdict. On a bad joint the ToolError
        hint carries the dump's per-joint vertex histogram."""
        return _compare_dump_ripples(
            ctx,
            dump_a,
            dump_b,
            joint,
            channel_a,
            channel_b,
            method,
            n_boot,
            n_sectors,
            alpha,
            f_lo_cpm,
            f_hi_cpm,
            bone_map_path,
            seed,
        )

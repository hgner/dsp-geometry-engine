"""Statistics kernel for the ELE320 stats pack: descriptive summaries, MLE
distribution fitting with AIC/KS/chi-square goodness of fit, classical hypothesis
tests, OLS with the full diagnostic block, and the band-RMS bootstrap / sector
machinery behind ``compare_dump_ripples``.

Pure numpy/scipy in and out — no MCP, no pydantic, no file I/O. Failures are
plain ``ValueError`` with hint-ready messages; the toolset layer converts them
into ToolError envelopes. Everything stochastic takes an explicit ``seed``
(``np.random.default_rng(seed)``) so identical calls give identical numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as scistats

from dsp_server.engine.filters import band_energy
from dsp_server.engine.transforms import Signal1D, axial_profile, spectrum

_TINY = 1e-30

DEFAULT_DISTRIBUTIONS: tuple[str, ...] = (
    "norm",
    "lognorm",
    "expon",
    "gamma",
    "weibull_min",
    "rayleigh",
    "uniform",
)
# Positive-support candidates are fitted with loc pinned at 0 iff min(y) > 0 —
# deterministic, and avoids the degenerate free-loc MLE (loc -> min(y), scale -> 0).
_POSITIVE_SUPPORT = frozenset({"lognorm", "expon", "gamma", "weibull_min", "rayleigh"})

VALID_TESTS: tuple[str, ...] = (
    "auto",
    "t_one_sample",
    "t_two_sample",
    "welch",
    "paired_t",
    "mannwhitney",
    "ks2",
    "f_var",
    "levene",
)
_ALTERNATIVES = ("two-sided", "less", "greater")


# --------------------------------------------------------------------------- #
# Descriptive statistics


def normality_test(y: np.ndarray) -> tuple[str, float, float]:
    """(test_name, statistic, p): Shapiro-Wilk for n <= 5000, else D'Agostino's
    normaltest (Shapiro loses calibration on huge n; normaltest needs n >= 8)."""
    y = np.asarray(y, dtype=np.float64)
    if y.size <= 5000:
        res = scistats.shapiro(y)
        return "shapiro", float(res.statistic), float(res.pvalue)
    res = scistats.normaltest(y)
    return "normaltest", float(res.statistic), float(res.pvalue)


@dataclass
class DescribeStats:
    """Mirror of the DescribeReport numeric fields (kernel-side, no pydantic)."""

    n: int
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


def describe_series(y: np.ndarray, ci_level: float = 0.95) -> DescribeStats:
    """Moments (unbiased skew / excess kurtosis), quantiles, t-interval on the
    mean, chi-square interval on the std, and a normality verdict."""
    y = np.asarray(y, dtype=np.float64)
    n = int(y.size)
    if n < 3:
        raise ValueError(f"need >= 3 samples, got {n}")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    mean = float(y.mean())
    std = float(y.std(ddof=1))
    pcts = (5, 25, 50, 75, 95)
    quantiles = {f"p{p:02d}": float(v) for p, v in zip(pcts, np.percentile(y, pcts), strict=True)}
    sem = std / math.sqrt(n)
    if sem > 0.0:
        ci_mean_lo, ci_mean_hi = scistats.t.interval(ci_level, df=n - 1, loc=mean, scale=sem)
    else:
        ci_mean_lo = ci_mean_hi = mean
    tail = 0.5 * (1.0 - ci_level)
    s2 = std * std
    ci_std_lo = math.sqrt((n - 1) * s2 / float(scistats.chi2.ppf(1.0 - tail, n - 1)))
    ci_std_hi = math.sqrt((n - 1) * s2 / float(scistats.chi2.ppf(tail, n - 1)))
    test_name, test_stat, test_p = normality_test(y)
    return DescribeStats(
        n=n,
        mean=mean,
        std=std,
        variance=std * std,
        min=float(y.min()),
        max=float(y.max()),
        median=float(np.median(y)),
        skewness=float(scistats.skew(y, bias=False)),
        kurtosis_excess=float(scistats.kurtosis(y, fisher=True, bias=False)),
        quantiles=quantiles,
        ci_level=float(ci_level),
        ci_mean_lo=float(ci_mean_lo),
        ci_mean_hi=float(ci_mean_hi),
        ci_std_lo=float(ci_std_lo),
        ci_std_hi=float(ci_std_hi),
        normality_test=test_name,
        normality_stat=test_stat,
        normality_p=test_p,
    )


# --------------------------------------------------------------------------- #
# Distribution fitting + goodness of fit


@dataclass
class DistFit:
    name: str
    params: dict[str, float]
    loglik: float
    k_params: int
    aic: float
    bic: float
    ks_stat: float
    ks_p: float
    chi2_stat: float | None
    chi2_p: float | None
    chi2_dof: int | None
    loc_fixed: bool


def gof_chi2(y: np.ndarray, frozen, k_params: int) -> tuple[float, float, int]:
    """Equal-probability chi-square GOF: k_bins = clip(int(sqrt(n)), 8, 40),
    reduced until expected n/k_bins >= 5; bin edges from the fitted ppf;
    ``chisquare(obs, exp, ddof=k_params)`` — dof = k_bins - 1 - k_params."""
    y = np.asarray(y, dtype=np.float64)
    n = int(y.size)
    k_bins = int(np.clip(int(math.sqrt(n)), 8, 40))
    while k_bins > k_params + 2 and n / k_bins < 5.0:
        k_bins -= 1
    if n / k_bins < 5.0 or k_bins - 1 - k_params < 1:
        raise ValueError(
            f"n={n} is too small for a chi2 GOF with {k_params} fitted params "
            "(need >= 5 expected per bin and dof >= 1)"
        )
    q = np.linspace(0.0, 1.0, k_bins + 1)
    with np.errstate(all="ignore"):
        interior = np.asarray(frozen.ppf(q[1:-1]), dtype=np.float64)
    if not np.all(np.isfinite(interior)) or np.any(np.diff(interior) <= 0.0):
        raise ValueError("degenerate fitted quantiles — chi2 bin edges are not monotone")
    edges = np.concatenate(([-np.inf], interior, [np.inf]))
    obs = np.histogram(y, bins=edges)[0].astype(np.float64)
    exp = np.full(k_bins, n / k_bins, dtype=np.float64)
    stat, p = scistats.chisquare(obs, exp, ddof=k_params)
    return float(stat), float(p), int(k_bins - 1 - k_params)


def fit_candidates(y: np.ndarray, names: list[str]) -> tuple[list[DistFit], dict[str, str]]:
    """MLE-fit every named scipy distribution; returns (fits AIC-ascending,
    skipped {name: reason}). The raw positional fit tuple is used for kstest
    (scipy shape order); the params dict exists only for the JSON report."""
    y = np.asarray(y, dtype=np.float64)
    n = int(y.size)
    positive = bool(n > 0 and float(np.min(y)) > 0.0)
    fits: list[DistFit] = []
    skipped: dict[str, str] = {}
    for name in names:
        dist = getattr(scistats, name, None)
        if dist is None or not hasattr(dist, "fit") or not hasattr(dist, "logpdf"):
            skipped[name] = "not a scipy.stats continuous distribution"
            continue
        loc_fixed = name in _POSITIVE_SUPPORT and positive
        try:
            with np.errstate(all="ignore"):
                raw = dist.fit(y, floc=0.0) if loc_fixed else dist.fit(y)
                loglik = float(np.sum(dist.logpdf(y, *raw)))
        except Exception as exc:  # scipy fit failures are wildly heterogeneous
            skipped[name] = f"{type(exc).__name__}: {exc}"
            continue
        if not math.isfinite(loglik):
            skipped[name] = "non-finite log-likelihood at the MLE"
            continue
        k = len(raw) - (1 if loc_fixed else 0)
        aic = 2.0 * k - 2.0 * loglik
        bic = k * math.log(n) - 2.0 * loglik
        # kstest consumes the RAW tuple positionally (dist.shapes order) via the
        # frozen cdf — scipy 1.18's string-name path mis-dispatches loc/scale for
        # norm. The p is optimistic: the params were fitted on this very sample.
        frozen = dist(*raw)
        ks = scistats.kstest(y, frozen.cdf)
        chi2_stat: float | None
        chi2_p: float | None
        chi2_dof: int | None
        try:
            chi2_stat, chi2_p, chi2_dof = gof_chi2(y, frozen, k)
        except ValueError:
            chi2_stat = chi2_p = chi2_dof = None
        shape_names = [s.strip() for s in dist.shapes.split(",")] if dist.shapes else []
        keys = [*shape_names, "loc", "scale"]
        params = {kname: float(v) for kname, v in zip(keys, raw, strict=True)}
        fits.append(
            DistFit(
                name=name,
                params=params,
                loglik=loglik,
                k_params=k,
                aic=aic,
                bic=bic,
                ks_stat=float(ks.statistic),
                ks_p=float(ks.pvalue),
                chi2_stat=chi2_stat,
                chi2_p=chi2_p,
                chi2_dof=chi2_dof,
                loc_fixed=loc_fixed,
            )
        )
    fits.sort(key=lambda f: f.aic)
    return fits, skipped


def frozen_from_params(name: str, params: dict[str, float]):
    """Rebuild the frozen scipy distribution from a DistFit params dict (whose
    insertion order is: shape params, then loc, then scale)."""
    dist = getattr(scistats, name)
    vals = list(params.values())
    return dist(*vals[:-2], loc=vals[-2], scale=vals[-1])


# --------------------------------------------------------------------------- #
# Hypothesis tests


@dataclass
class TestResult:
    test_used: str
    alternative_used: str  # "two-sided (forced)" when the test ignores the request
    statistic: float
    p_value: float
    df: float | None
    effect_size: float | None
    effect_size_kind: str | None
    ci_diff: tuple[float, float] | None


def _pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
    n_a, n_b = a.size, b.size
    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    return math.sqrt(((n_a - 1) * va + (n_b - 1) * vb) / (n_a + n_b - 2))


def run_test(
    a: np.ndarray,
    b: np.ndarray | None,
    test: str,
    alternative: str,
    mu0: float,
    alpha: float,
) -> TestResult:
    """Dispatch one classical test. ``auto``: one sample -> t_one_sample against
    mu0; two samples -> welch. Effect size: pooled Cohen's d for the t family,
    rank-biserial r = 2U/(n_a*n_b) - 1 for mannwhitney, None otherwise. CIs on
    the (difference of) means come from scipy's TtestResult."""
    if test not in VALID_TESTS:
        raise ValueError(f"unknown test '{test}' — expected one of {list(VALID_TESTS)}")
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"unknown alternative '{alternative}' — expected one of {list(_ALTERNATIVES)}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    a = np.asarray(a, dtype=np.float64)
    test_used = test
    if test == "auto":
        test_used = "t_one_sample" if b is None else "welch"
    if test_used == "t_one_sample":
        if b is not None:
            raise ValueError("t_one_sample takes one sample — omit path_b or pick a two-sample test")
    elif b is None:
        raise ValueError(f"test '{test_used}' needs two samples — pass path_b")
    if b is not None:
        b = np.asarray(b, dtype=np.float64)

    df: float | None = None
    effect: float | None = None
    kind: str | None = None
    ci: tuple[float, float] | None = None
    alternative_used = alternative

    if test_used == "t_one_sample":
        res = scistats.ttest_1samp(a, mu0, alternative=alternative)
        df = float(res.df)
        sd = float(a.std(ddof=1))
        if sd > 0.0:
            effect = float((a.mean() - mu0) / sd)
            kind = "cohen_d"
        lo, hi = res.confidence_interval(1.0 - alpha)
        ci = (float(lo), float(hi))
    elif test_used in ("t_two_sample", "welch", "paired_t"):
        assert b is not None
        if test_used == "paired_t":
            if a.size != b.size:
                raise ValueError(f"paired_t needs equal sample sizes — got n_a={a.size} and n_b={b.size}")
            res = scistats.ttest_rel(a, b, alternative=alternative)
        else:
            res = scistats.ttest_ind(a, b, equal_var=(test_used == "t_two_sample"), alternative=alternative)
        df = float(res.df)
        pooled = _pooled_sd(a, b)
        if pooled > 0.0:
            effect = float((a.mean() - b.mean()) / pooled)
            kind = "cohen_d"
        lo, hi = res.confidence_interval(1.0 - alpha)
        ci = (float(lo), float(hi))
    elif test_used == "mannwhitney":
        assert b is not None
        res = scistats.mannwhitneyu(a, b, alternative=alternative)
        effect = float(2.0 * float(res.statistic) / (a.size * b.size) - 1.0)
        kind = "rank_biserial"
    elif test_used == "ks2":
        assert b is not None
        res = scistats.ks_2samp(a, b, alternative=alternative)
    elif test_used == "levene":
        assert b is not None
        res = scistats.levene(a, b, center="median")
        alternative_used = "two-sided (forced)"  # scipy's levene takes no alternative
    else:  # f_var — no scipy one-liner; spec'd explicitly
        assert b is not None
        va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
        if vb <= 0.0:
            raise ValueError("f_var needs a positive variance in sample b")
        f_stat = va / vb
        dfa, dfb = a.size - 1, b.size - 1
        sf = float(scistats.f.sf(f_stat, dfa, dfb))
        cdf = float(scistats.f.cdf(f_stat, dfa, dfb))
        if alternative == "greater":
            p = sf
        elif alternative == "less":
            p = cdf
        else:
            p = min(1.0, 2.0 * min(sf, cdf))
        return TestResult(
            test_used="f_var",
            alternative_used=alternative,
            statistic=float(f_stat),
            p_value=float(p),
            df=None,
            effect_size=None,
            effect_size_kind=None,
            ci_diff=None,
        )

    return TestResult(
        test_used=test_used,
        alternative_used=alternative_used,
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        df=df,
        effect_size=effect,
        effect_size_kind=kind,
        ci_diff=ci,
    )


# --------------------------------------------------------------------------- #
# OLS regression


@dataclass
class OlsResult:
    names: list[str]
    coef: np.ndarray
    std_err: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray
    ci_lo: np.ndarray
    ci_hi: np.ndarray
    r2: float
    r2_adj: float
    f_stat: float
    f_p: float
    rmse: float
    durbin_watson: float
    cond: float
    resid: np.ndarray


def ols(x: np.ndarray, y: np.ndarray, names: list[str], ci_level: float = 0.95) -> OlsResult:
    """OLS on a design matrix that ALREADY carries its intercept column of ones;
    ``names`` aligns with the columns ('intercept' first by convention).
    Coefficient inference from sigma^2 (X'X)^-1 with df = n - p; the condition
    number is taken on the standardized design (multicollinearity flag)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size:
        raise ValueError(f"shape mismatch: design {x.shape} vs y ({y.size},)")
    n, p = x.shape
    if len(names) != p:
        raise ValueError(f"{len(names)} names for {p} design columns")
    if n <= p:
        raise ValueError(f"need more rows than parameters — got n={n} rows for p={p} coefficients")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < p:
        raise ValueError(
            f"design matrix is rank-deficient (rank {rank} < {p} columns) — drop collinear x_columns"
        )
    y_hat = x @ beta
    resid = y - y_hat
    rss = float(resid @ resid)
    dof = n - p
    sigma2 = rss / dof
    xtx_inv = np.linalg.inv(x.T @ x)
    std_err = np.sqrt(np.clip(sigma2 * np.diag(xtx_inv), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.where(std_err > 0.0, beta / np.maximum(std_err, _TINY), np.inf)
    p_value = 2.0 * scistats.t.sf(np.abs(t_stat), dof)
    t_crit = float(scistats.t.ppf(0.5 + 0.5 * ci_level, dof))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0.0 else 0.0
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / dof
    k_reg = p - 1
    if k_reg >= 1 and rss > 0.0:
        f_stat = ((tss - rss) / k_reg) / (rss / dof)
        f_p = float(scistats.f.sf(f_stat, k_reg, dof))
    else:
        f_stat, f_p = math.inf, 0.0
    d = np.diff(resid)
    dw = float((d @ d) / rss) if rss > 0.0 else 0.0
    xs = x.copy()
    for j in range(p):
        col = xs[:, j]
        sd = float(col.std())
        if sd > 0.0:
            xs[:, j] = (col - col.mean()) / sd
    return OlsResult(
        names=list(names),
        coef=beta,
        std_err=std_err,
        t_stat=t_stat,
        p_value=p_value,
        ci_lo=beta - t_crit * std_err,
        ci_hi=beta + t_crit * std_err,
        r2=float(r2),
        r2_adj=float(r2_adj),
        f_stat=float(f_stat),
        f_p=float(f_p),
        rmse=math.sqrt(rss / n),
        durbin_watson=dw,
        cond=float(np.linalg.cond(xs)),
        resid=resid,
    )


# --------------------------------------------------------------------------- #
# Band RMS + bootstrap + sector machinery (compare_dump_ripples)


def rel_signal(sig: Signal1D) -> Signal1D:
    """r/median(r) - 1 as a Signal1D — the rel_ripple convention used by
    ``roughness`` (scale- and pose-fair, dimensionless)."""
    y = np.asarray(sig.y, dtype=np.float64)
    med = float(np.median(y))
    scale = med if abs(med) > 1e-12 else 1.0
    return Signal1D(y=y / scale - 1.0, dt=sig.dt, t0=sig.t0)


def band_rms(sig: Signal1D, f_lo: float, f_hi: float, detrend_order: int = 3) -> float:
    """sqrt(integrated PSD over [f_lo, f_hi]) of the RELATIVE radius profile
    (cubic-detrended, Hann PSD) — dimensionless, comparable across dumps."""
    if f_hi <= f_lo:
        raise ValueError(f"empty band [{f_lo}, {f_hi}] cpm — need f_lo < f_hi")
    freqs, psd = spectrum(rel_signal(sig), detrend_order=detrend_order)
    return math.sqrt(max(band_energy(freqs, psd, f_lo, f_hi), 0.0))


@dataclass
class BootstrapRatio:
    rms_a: float
    rms_b: float
    ratio: float
    ci_ratio: tuple[float, float]
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]
    n_boot: int


def bootstrap_band_ratio(
    t_a: np.ndarray,
    r_a: np.ndarray,
    t_b: np.ndarray,
    r_b: np.ndarray,
    f_lo: float,
    f_hi: float,
    n_boot: int = 500,
    seed: int = 0,
    n_samples: int = 256,
    alpha: float = 0.05,
) -> BootstrapRatio:
    """Vertex-index bootstrap: resample (t, r) pairs with replacement WITHIN each
    side's joint mask, rebuild the axial profile, take band RMS per replicate per
    side, and form ratio = rms_b/rms_a. Percentile CIs at [alpha/2, 1-alpha/2].
    Honest caveat: treats vertices as exchangeable measurement samples of one
    surface realization — it quantifies sampling noise, not pose-to-pose variance."""
    t_a = np.asarray(t_a, dtype=np.float64)
    r_a = np.asarray(r_a, dtype=np.float64)
    t_b = np.asarray(t_b, dtype=np.float64)
    r_b = np.asarray(r_b, dtype=np.float64)
    if n_boot < 20:
        raise ValueError(f"n_boot must be >= 20, got {n_boot}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    sig_a = axial_profile(t_a, r_a, n_samples=n_samples)
    sig_b = axial_profile(t_b, r_b, n_samples=n_samples)
    rms_a = band_rms(sig_a, f_lo, f_hi)
    rms_b = band_rms(sig_b, f_lo, f_hi)
    rng = np.random.default_rng(seed)
    n_a, n_b = t_a.size, t_b.size
    boots_a: list[float] = []
    boots_b: list[float] = []
    for _ in range(n_boot):
        ia = rng.integers(0, n_a, n_a)
        ib = rng.integers(0, n_b, n_b)
        try:
            sa = axial_profile(t_a[ia], r_a[ia], n_samples=n_samples)
            sb = axial_profile(t_b[ib], r_b[ib], n_samples=n_samples)
        except ValueError:
            continue  # degenerate resample (axial range collapsed) — skip it
        boots_a.append(band_rms(sa, f_lo, f_hi))
        boots_b.append(band_rms(sb, f_lo, f_hi))
    if len(boots_a) < max(20, n_boot // 2):
        raise ValueError(
            f"only {len(boots_a)}/{n_boot} bootstrap replicates were computable — "
            "the segment is too degenerate for a resampled axial profile"
        )
    ba = np.asarray(boots_a, dtype=np.float64)
    bb = np.asarray(boots_b, dtype=np.float64)
    ratios = bb / np.maximum(ba, _TINY)
    q_lo, q_hi = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)

    def _ci(v: np.ndarray) -> tuple[float, float]:
        lo, hi = np.percentile(v, [q_lo, q_hi])
        return (float(lo), float(hi))

    return BootstrapRatio(
        rms_a=rms_a,
        rms_b=rms_b,
        ratio=rms_b / max(rms_a, _TINY),
        ci_ratio=_ci(ratios),
        ci_a=_ci(ba),
        ci_b=_ci(bb),
        n_boot=int(ba.size),
    )


@dataclass
class SectorTest:
    n_sectors_used: int  # the SMALLER side's usable sector count (drives power)
    test_name: str
    statistic: float
    p_value: float
    b_larger: bool  # direction read off the test's own statistic


def sector_band_test(
    sectors_a: list[Signal1D],
    sectors_b: list[Signal1D],
    f_lo: float,
    f_hi: float,
) -> SectorTest:
    """UNPAIRED Mann-Whitney U across the per-sector band-RMS values of the two
    sides. Pairing was dropped deliberately: ``sector_profiles`` silently drops
    empty wedges without index bookkeeping, and cross-dump angular zero is only
    frame-aligned (not vertex-aligned), so positional pairing would pair noise.
    Falls back to Welch's t when either side has fewer than 4 usable sectors.
    Degenerate all-tie inputs (identical dumps) short-circuit to p = 1."""
    rms_a = [band_rms(s, f_lo, f_hi) for s in sectors_a]
    rms_b = [band_rms(s, f_lo, f_hi) for s in sectors_b]
    n_a, n_b = len(rms_a), len(rms_b)
    if min(n_a, n_b) < 2:
        raise ValueError(f"need >= 2 usable sectors per side, got {n_a} vs {n_b}")
    if len(set(rms_a) | set(rms_b)) == 1:
        return SectorTest(
            n_sectors_used=min(n_a, n_b),
            test_name="mannwhitneyu (degenerate: all values tied)",
            statistic=n_a * n_b / 2.0,
            p_value=1.0,
            b_larger=False,
        )
    if min(n_a, n_b) >= 4:
        res = scistats.mannwhitneyu(rms_a, rms_b, alternative="two-sided")
        return SectorTest(
            n_sectors_used=min(n_a, n_b),
            test_name="mannwhitneyu",
            statistic=float(res.statistic),
            p_value=float(res.pvalue),
            b_larger=bool(float(res.statistic) < n_a * n_b / 2.0),
        )
    res = scistats.ttest_ind(rms_a, rms_b, equal_var=False)
    return SectorTest(
        n_sectors_used=min(n_a, n_b),
        test_name="welch_t",
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        b_larger=bool(float(res.statistic) < 0.0),
    )

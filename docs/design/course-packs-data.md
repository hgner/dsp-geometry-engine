> **HISTORICAL — superseded design notes, written 2026-07-11, banner added 2026-07-31.**
>
> A pre-implementation design document for the `stats` and `ml` packs, which have since shipped;
> retained for provenance only. It is **not** a description of the shipped system and **must not**
> be read as a live specification, finding list, or backlog. Tool names, signatures, schema fields,
> and constants proposed here were revised during implementation. Most importantly for anyone
> reading the `ml` section: the `predict()` deserialization hazard raised as item 7 of
> `course-packs-critique.md` was **resolved** — `_predict` in `src/dsp_server/toolsets/ml.py`
> path-confines the model to `data/models/` and requires the server-written `.meta.json` sidecar
> before joblib unpickles anything, with `tests/test_ml.py::test_predict_security_rejections`
> covering the rejection paths. The shipped code, its docstrings, `llms.txt`, and the test suite are
> the only ground truth.

# Design — shared tabular loader + stats (ELE320) pack + ml (ELE489) pack

Grounded against the frozen architecture: `toolsets/__init__.py` (AppContext + TOOLSETS registry), the `geometry.py` pattern (module-level `_impl(ctx, ...)` + thin `@mcp.tool()` closures, ToolError never-raise, histogram-carrying hints), `schemas.py` (`_SchemaBase` with round6 + `extra="forbid"`), `engine/transforms.py` (`Signal1D`, `extract_forearm_signal`, `sector_profiles`), `engine/filters.py` (`detrend_poly`, `apply_highpass`, `band_energy`, `spectral_peaks`), `engine/ply.py` (`write_engine_ply`, `load_dump_cached`, `load_meta`), `plots.py` (Agg-only, Path-returning), `tests/synth.py` (`make_cylinder`, `STUB_BONE_MAP`), and the DEVELOPMENT.md 5-step checklist.

---

## 0. SHARED: `src/dsp_server/engine/tabular.py` (owned spec)

One module, no MCP imports, no pydantic. All exceptions raised here are `TabularError(ValueError)` so toolsets catch one type. Every error message states **what was found**, not just what was missing — toolsets pass `str(exc)` straight into `ToolError.hint`.

### 0.1 Module constants

```python
MAX_CELLS: int = 50_000_000        # module-level so tests can monkeypatch it small
MAX_FILE_BYTES: int = 400 * 2**20  # pre-parse stat() guard, cheaper than parsing 400 MB
SUPPORTED_SUFFIXES = (".csv", ".tsv", ".json", ".npz", ".npy", ".ply")

class TabularError(ValueError): ...
```

### 0.2 Public surface (4 loaders + 2 helpers)

```python
def load_series(
    path: str | Path,
    column: str | int | None = None,
    key: str | None = None,           # npz member name
    dropna: bool = True,
    cache_dir: Path | None = None,    # forwarded to ply.load_dump_cached for .ply
) -> np.ndarray                       # (n,) float64, C-contiguous

def load_table(
    path: str | Path,
    columns: list[str] | None = None,
    key: str | None = None,
    cache_dir: Path | None = None,
) -> dict[str, np.ndarray]            # insertion-ordered, equal-length (n,) float64 arrays

def load_matrix(
    path: str | Path,
    columns: list[str] | None = None,
    key: str | None = None,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, list[str]]     # (n, d) float64 X + column names (ml pack's loader)

def load_labels(
    path: str | Path,
    column: str | int,
    key: str | None = None,
) -> np.ndarray                       # (n,) — float64 if numeric, else <U str; NO numeric coercion

def columns_of(path: str | Path, key: str | None = None) -> list[str]   # cheap introspection for hints
def drop_nan_rows(table: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]  # jointly row-filter; returns (table, n_dropped)
```

### 0.3 Format dispatch (case-insensitive suffix)

| Suffix | Behavior |
|---|---|
| `.csv` / `.tsv` | delimiter `,` / `\t`. Header sniff: split the first non-empty, non-`#` line; if **every** token parses as float there is no header and columns get synthetic names `col0..col{N-1}`; otherwise the line is the header (`np.genfromtxt(..., names=True, dtype=np.float64, encoding="utf-8", deletechars="")` so names survive verbatim). Non-numeric cells become NaN (genfromtxt default). |
| `.npz` | `np.load(path)`. `key` selects the member; `key=None` with exactly one member uses it; with several → error listing keys. **Feature-matrix convention**: if the archive contains both `X` and `feature_names`, `load_table`/`load_matrix` treat it as a named matrix (columns = `feature_names` decoded to str); `load_series` then requires `column`. Size guard checks `arr.size` **before** materializing (npz members are lazy). |
| `.npy` | single array; `key` must be None (error otherwise: `"'.npy' holds one unnamed array — key= is only for .npz"`). |
| `.json` | `json.loads`. Accepted shapes: (a) list of numbers → one column named `"value"`; (b) list of flat objects (records) → table; (c) object of equal-length lists `{col: [...]}` → table. `null`→NaN, bool→float. A column with any non-coercible string is **excluded** from numeric output; requesting it explicitly errors: `"column 'name' exists but is not numeric — numeric columns present: [...] (use load_labels for string columns)"`. Records with missing keys fill NaN. |
| `.ply` | engine-dump convenience. `column` syntax `"<joint>"` or `"<joint>:<channel>"` (channel `posed`\|`rest`, default `posed`; joint default `armLowerL`). Loads via `ply.load_dump_cached(path, cache_dir)`, resolves the bone map via the promoted `ply.bone_map_for(path)` (see §3), calls `transforms.extract_forearm_signal(dump, joint, bone_map, channel=channel)`. `load_series` returns `signal.y` (the 256-sample binned radius profile). `load_table` returns `{"t": t0 + arange(n)*dt, "r": signal.y}` (regression-ready). `load_matrix`/`load_labels` on `.ply` → error pointing at `feature_engineer_dump`. |

2-D arrays from `.npz`/`.npy` in `load_series`: `column` as int (or all-digit string) selects `arr[:, i]`; missing column → `"array has shape (n, m) — pass column= an index 0..m-1"`.

### 0.4 Column-selection semantics (all loaders)

1. Exact name match wins.
2. Else unique case-insensitive name match.
3. Else, if `column` is an int or an all-digit string, positional index into file column order (`IndexError` → error listing the valid range and names).
4. `load_series(column=None)`: single-column source → that column; multi-column → `TabularError("table has 4 columns — pass column= one of ['t', 'r', 'temp', 'v'] (name or index 0..3)")`.
5. `load_table(columns=[...])`: subset **in requested order**; first unknown name errors with the full found-columns list.

### 0.5 NaN and size policy

- `load_series(dropna=True)`: drop non-finite values; if `< 3` finite remain → `"only {k} finite values in '{column}' after dropping NaN/inf — need >= 3"`.
- `load_table`/`load_matrix` preserve NaN (paired columns must stay row-aligned); callers use `drop_nan_rows` and report the count.
- Size guard, in order: (1) `stat().st_size > MAX_FILE_BYTES` → refuse pre-parse; (2) `rows * cols > MAX_CELLS` → `"table is {rows} x {cols} = {cells:.2e} cells — limit {MAX_CELLS:.0e}; pass columns= to select fewer, or downsample the file"`.

### 0.6 Error-message catalog (verbatim templates the implementer copies)

- unsupported: `"unsupported table format '{suffix}' — supported: .csv, .tsv, .json, .npz, .npy, .ply"`
- missing column: `"column '{name}' not found — columns present: {names} (pass a name or an integer index 0..{d-1})"`
- npz key: `"key '{key}' not in archive — keys present: {keys}"` / `"archive has {k} arrays — pass key= one of {keys}"`
- json shape: `"JSON must be a list of numbers, a list of flat objects, or an object of column lists — got {type}"`
- empty: `"no numeric rows parsed from {path}"`
- ragged json columns: `"column lists have unequal lengths: { 'a': 100, 'b': 99 }"`

---

## 1. STATS PACK (ELE320) — `toolsets/stats.py` + `engine/statskit.py`

Five tools. All data tools take `(path, column, key)` routed through `tabular.load_series`/`load_table`; on TabularError the ToolError hint is `str(exc)` (which already lists found columns). All p-values/statistics via scipy.stats; every report echoes `n` and `n_dropped_nan`.

### (a) Tool inventory

**1. `describe(path: str, column: str | None = None, key: str | None = None, ci_level: float = 0.95) -> str`**
Moments, quantiles, and CIs for one numeric series from any supported file (or a dump profile via `"dump.ply"` + `column="armLowerL:posed"`).
Returns `DescribeReport`: `path, column, n, n_dropped_nan, mean, std, variance, min, max, median, skewness, kurtosis_excess, quantiles (dict p05/p25/p50/p75/p95), ci_level, ci_mean_lo, ci_mean_hi, ci_std_lo, ci_std_hi, normality_test, normality_stat, normality_p`.
scipy: `skew(bias=False)`, `kurtosis(fisher=True, bias=False)`, `sem` + `t.interval` (mean CI), `chi2.ppf` (std CI: `sqrt((n-1)s²/χ²)`), `shapiro` if `n<=5000` else `normaltest` (field `normality_test` names which ran).

**2. `fit_distribution(path: str, column: str | None = None, key: str | None = None, distributions: list[str] | None = None, save_plot: bool = False) -> str`**
MLE-fit a shortlist of scipy distributions, rank by AIC, attach KS and equal-probability χ² goodness-of-fit per candidate. Default shortlist: `["norm", "lognorm", "expon", "gamma", "weibull_min", "rayleigh", "uniform"]` (rayleigh fits ripple envelopes — RCA hook). Fit rule: positive-support dists (`lognorm, expon, gamma, weibull_min, rayleigh`) fit with `floc=0.0` **iff** `min(y) > 0` (recorded as `loc_fixed`), else unconstrained — deterministic, avoids degenerate free-loc fits.
Per candidate: `params` (scipy shape names from `dist.shapes` + `loc`/`scale`), `loglik = sum(logpdf)`, `k_params` (free params only), `aic = 2k − 2·loglik`, `bic = k·ln n − 2·loglik`, `ks_stat/ks_p` (`stats.kstest(y, name, args=params)` — p is optimistic since params come from the same data; the docstring and llms rule say so), `chi2_stat/chi2_p/chi2_dof` (equal-probability bins: `k_bins = clip(int(sqrt(n)), 8, 40)` reduced until expected `n/k_bins >= 5`; edges from fitted `ppf`; `stats.chisquare(obs, exp, ddof=k_params)` → this is the syllabus GLR/goodness-of-fit lane). Candidates whose fit raises or yields non-finite loglik land in `skipped: dict[str,str]`; all skipped → ToolError.
Returns `FitDistributionReport`: `path, column, n, n_dropped_nan, candidates (AIC-ascending list of DistFitSchema), best_dist, skipped, plot_path`.

**3. `hypothesis_test(path_a: str, path_b: str | None = None, column_a: str | None = None, column_b: str | None = None, key_a: str | None = None, key_b: str | None = None, test: str = "auto", alternative: str = "two-sided", mu0: float = 0.0, alpha: float = 0.05) -> str`**
One tool, mode param (tool-count discipline). `test` ∈ `{auto, t_one_sample, t_two_sample, welch, paired_t, mannwhitney, ks2, f_var, levene}`. `auto`: one sample → `t_one_sample` against `mu0`; two samples → `welch` (variance-robust default); choice echoed in `test_used`. `path_b=None` with a two-sample test → ToolError. `paired_t` requires equal n (error names both lengths).
scipy mapping: `ttest_1samp`, `ttest_ind(equal_var=True)`, `ttest_ind(equal_var=False)` (df from the result's `.df`), `ttest_rel`, `mannwhitneyu`, `ks_2samp`, `levene(center="median")`. `f_var` has no scipy one-liner — spec exactly: `F = s_a²/s_b²` (ddof=1), `dof=(n_a−1, n_b−1)`, two-sided `p = 2·min(f.sf(F,·), f.cdf(F,·))` clipped to ≤1; one-sided uses `sf`/`cdf` per `alternative`.
Effect size: Cohen's d (pooled) for the t-family, rank-biserial `r = 2U/(n_a·n_b) − 1` for mannwhitney, `None` otherwise; kind named in `effect_size_kind`. CI on the (difference of) means for the t-family via the scipy result's `confidence_interval(1−alpha)`.
Returns `HypothesisTestReport`: `test_used, alternative, alpha, n_a, n_b, mean_a, mean_b, std_a, std_b, statistic, p_value, df, reject_h0, effect_size, effect_size_kind, ci_diff_lo, ci_diff_hi` (df/effect/CI nullable).
Note: formal Neyman-Pearson/LR machinery is deliberately not a seventh tool — the χ²-GLR lives in `fit_distribution`, and `reject_h0`+`alpha` covers the type-I framing (llms rule states this).

**4. `regression_fit(path: str, y_column: str, x_columns: list[str] | None = None, key: str | None = None, ci_level: float = 0.95, save_plot: bool = False) -> str`**
OLS with intercept (always), full ELE320 diagnostics. `x_columns=None` → all numeric columns except `y_column`. NaN rows dropped jointly via `drop_nan_rows` (count reported). Solve `lstsq`; `σ² = RSS/(n−p)`; coefficient covariance `σ²(XᵀX)⁻¹` → per-coefficient `std_err, t_stat, p_value` (two-sided, `stats.t`, df `n−p`), `ci_lo/ci_hi`. Global: `r2, r2_adj, f_stat, f_p` (`stats.f`), `rmse`, `durbin_watson` (`Σ(Δe)²/Σe²`, no new dep), residual normality (`shapiro`, n≤5000, else `normaltest`), `condition_number` (`np.linalg.cond` of the standardized design — multicollinearity flag). Plot: single regressor → scatter+fit line panel over residual-vs-fitted panel; multiple → fitted-vs-actual over residuals.
Returns `RegressionReport`: `path, y_column, x_columns, n, n_dropped_nan, coefficients (list of CoefficientSchema: name, estimate, std_err, t_stat, p_value, ci_lo, ci_hi — "intercept" first), r2, r2_adj, f_stat, f_p, rmse, durbin_watson, resid_normality_test, resid_normality_stat, resid_normality_p, condition_number, plot_path`.

**5. `compare_dump_ripples(dump_a: str, dump_b: str, joint: str = "armLowerL", channel_a: str = "posed", channel_b: str = "posed", method: str = "both", n_boot: int = 500, n_sectors: int = 16, alpha: float = 0.05, f_lo_cpm: float | None = None, f_hi_cpm: float | None = None, bone_map_path: str | None = None, seed: int = 0) -> str`**
The RCA convenience: *is the ripple difference between two dumps statistically significant?* Exact statistic, stated honestly: `band_rms = sqrt(band_energy(freqs, psd, f_lo, f_hi))` of the **relative-radius** profile (`r/median(r) − 1`, cubic-detrended, matching `roughness`'s rel_ripple convention), band chosen as ±30% around side-B's dominant peak (`spectral_peaks` on B, `f_min=12` cpm), fallback `[12 cpm, 0.9·Nyquist]` when B has no peak, or explicit `f_lo_cpm/f_hi_cpm`; the choice is echoed in `band_source ∈ {dominant_peak_b, explicit, fallback_highband}`.
Two methods, both reported (`method` selects `bootstrap|sector|both`):
- **bootstrap** — resample vertex indices with replacement *within each side's joint mask* (`np.random.default_rng(seed)`, `n_boot` replicates), rebuild `axial_profile` → detrend → band_rms per replicate per side, form `ratio = rms_b/rms_a`; percentile CI `[α/2, 1−α/2]`; `significant = not (ci_lo <= 1.0 <= ci_hi)`. Honest caveat (docstring + report field `assumptions`): treats vertices as exchangeable measurement samples of one surface realization.
- **sector** — `sector_profiles(n_sectors=16)` per side; per-sector band_rms; sectors are angularly index-matched → **paired Wilcoxon signed-rank** (`stats.wilcoxon`) on the per-sector pairs; if fewer than 6 shared usable sectors, fall back to `ttest_rel` (recorded in `sector_test`). Caveat: sectors share one deformation field, so this is quasi-independence.
`verdict`: `b_ripplier` / `a_ripplier` when the run methods agree in direction and each is significant at `alpha`; `no_significant_difference` when none significant; `methods_disagree` otherwise (with `method="bootstrap"|"sector"`, the single method decides). Agreement of the two approximations is the actionable signal — that sentence goes in the docstring.
Returns `RippleComparisonReport`: `dump_a, dump_b, channel_a, channel_b, joint, joint_ids, vertex_count_a, vertex_count_b, f_lo_cpm, f_hi_cpm, band_source, band_rms_a, band_rms_b, ratio_b_over_a, bootstrap (BootstrapRatioSchema | None), sector (SectorTestSchema | None), verdict, assumptions (list[str])`. Bad joint → ToolError with the per-joint histogram hint (reuse geometry's `_histogram_hint` pattern via the promoted helper, §3).

### (b) `engine/statskit.py` (pure; may import transforms/filters like transforms imports filters)

```python
@dataclass class DescribeStats: ...            # mirror of DescribeReport numeric fields
def describe_series(y: np.ndarray, ci_level: float = 0.95) -> DescribeStats
@dataclass class DistFit: name; params: dict[str, float]; loglik; k_params; aic; bic;
                          ks_stat; ks_p; chi2_stat; chi2_p; chi2_dof: int; loc_fixed: bool
def fit_candidates(y: np.ndarray, names: list[str]) -> tuple[list[DistFit], dict[str, str]]  # (fits AIC-asc, skipped)
def gof_chi2(y: np.ndarray, frozen, k_params: int) -> tuple[float, float, int]
@dataclass class TestResult: test_used; statistic; p_value; df: float | None;
                             effect_size: float | None; effect_size_kind: str | None;
                             ci_diff: tuple[float, float] | None
def run_test(a: np.ndarray, b: np.ndarray | None, test: str, alternative: str,
             mu0: float, alpha: float) -> TestResult
@dataclass class OlsResult: names; coef; std_err; t_stat; p_value; ci_lo; ci_hi;  # arrays
                            r2; r2_adj; f_stat; f_p; rmse; durbin_watson; cond; resid: np.ndarray
def ols(x: np.ndarray, y: np.ndarray, names: list[str], ci_level: float = 0.95) -> OlsResult
def band_rms(sig: Signal1D, f_lo: float, f_hi: float, detrend_order: int = 3) -> float
@dataclass class BootstrapRatio: rms_a; rms_b; ratio; ci_ratio: tuple[float,float];
                                 ci_a: tuple[float,float]; ci_b: tuple[float,float]; n_boot: int
def bootstrap_band_ratio(t_a, r_a, t_b, r_b, f_lo, f_hi, n_boot=500, seed=0,
                         n_samples=256) -> BootstrapRatio
@dataclass class SectorTest: n_sectors_used: int; test_name: str; statistic: float; p_value: float
def sector_band_test(sectors_a: list[Signal1D], sectors_b: list[Signal1D],
                     f_lo: float, f_hi: float) -> SectorTest
```

### (c) Pydantic schemas (append to `schemas.py`, all on `_SchemaBase`)

`DescribeReport`, `DistFitSchema`, `FitDistributionReport`, `HypothesisTestReport`, `CoefficientSchema`, `RegressionReport`, `BootstrapRatioSchema` (`n_boot, ci_ratio_lo, ci_ratio_hi, ci_a_lo, ci_a_hi, ci_b_lo, ci_b_hi, significant: bool`), `SectorTestSchema` (`n_sectors_used, test_name, statistic, p_value, significant`), `RippleComparisonReport` — fields exactly as listed in (a).

### (d) Test plan — `tests/test_stats.py` + `tests/test_tabular.py` (no engine, `_impl(ctx,...)` style)

tabular:
- csv with header / headerless csv (synthetic names `col0..`), tsv, npz single + multi-key, npy, json list / records / column-dict, `.ply` written via `write_engine_ply(make_cylinder(...))` with `column="5:posed"` → 256 samples ≈ 0.04 m radius.
- error contracts: missing column message **contains** `"columns present: ['t', 'r']"`; npz key message contains the key list; unsupported `.xlsx` message lists supported suffixes.
- size guard: `monkeypatch.setattr(tabular, "MAX_CELLS", 100)` then a 20×10 csv passes and 20×6=120-cell csv refuses (never allocate 5e7 cells in CI).
- `load_labels` returns `<U` dtype for a string class column; `load_matrix` on a feature-convention npz honors `feature_names` order.

stats (all seeds fixed; tolerance assertions; any p-threshold seed verified at implementation time and pinned):
- describe: `default_rng(42).normal(5, 2, 400)` → `abs(mean−5) < 0.25`, `abs(std−2) < 0.2`, `ci_mean_lo < 5 < ci_mean_hi`, `normality_p > 0.05`, `normality_test == "shapiro"`.
- fit_distribution: `expon(scale=2)` sample n=500 seed=1 → `best_dist == "expon"`, `aic(expon) < aic(norm)`, expon `loc_fixed is True`, fitted `scale` within `[1.75, 2.25]`; norm sample → best `norm` with `chi2_p > 0.01`.
- hypothesis_test: N(0,1,200,seed 0) vs N(0.5,1,200,seed 1) → welch `p < 1e-3`, `reject_h0`, `0.3 < effect_size < 0.7`, `ci_diff` excludes 0; same-distribution seeds → `p > 0.05`; `f_var` N(0,1) vs N(0,3) → `p < 1e-6`; `test="auto"` one-sample routes to `t_one_sample` (`test_used` asserted); `paired_t` unequal n → ToolError naming both lengths.
- regression_fit: `y = 2 + 3x + N(0, 0.1)`, n=100 → slope in `[2.9, 3.1]`, its CI contains 3, `r2 > 0.99`, `f_p < 1e-10`, intercept p < 1e-6, `resid_normality_p > 0.01`.
- compare_dump_ripples (dumps written to `tmp_path` via `write_engine_ply`; joint passed as `"5"` or via a `bone_map_path` json of STUB_BONE_MAP): smooth `make_cylinder(noise=2e-4)` vs corrugated `make_cylinder(corrugation_amp=0.002, corrugation_freq_cpm=120, noise=2e-4)` → `ratio_b_over_a > 3`, bootstrap `significant`, sector `p < 0.05`, `verdict == "b_ripplier"`, `f_lo_cpm < 120 < f_hi_cpm`; two same-seed smooth dumps → `verdict == "no_significant_difference"`; bad joint name → ToolError whose hint contains `"histogram"`.

### (e) llms.txt rule additions (stats)

```
12. stats pack: any tool taking (path, column) accepts .csv/.tsv/.json/.npz/.npy plus .ply with
    column="<joint>[:posed|rest]" (the binned axial radius profile). fit_distribution's KS p is
    optimistic (params fitted on the same data) — rank by AIC, confirm with chi2_p; the chi2 GOF
    is the pack's GLR lane. compare_dump_ripples is significant only when bootstrap CI and the
    paired sector test agree — 'methods_disagree' means collect more poses, not a verdict.
```

---

## 2. ML PACK (ELE489) — `toolsets/ml.py` + `engine/mlkit.py`

New deps in `pyproject.toml`: `scikit-learn>=1.5`, `joblib>=1.4` (joblib ships as a hard sklearn dependency — declared explicitly because we import it directly). Determinism: every stochastic estimator takes `seed: int = 0` → `random_state=seed`; `KMeans(n_init=10)`. Models persist under `data/models/` (new `AppContext.models_dir`, §3; `data/` is already disposable/ignored). Silhouette guard: `n > 20_000` → score on a seeded 10k subsample, `silhouette_subsampled=True`.

### (a) Tool inventory

**1. `feature_engineer_dump(dump: str, joint: str | None = None, channel: str = "posed", features: list[str] | None = None, bone_map_path: str | None = None) -> str`**
Per-vertex feature matrix from an engine PLY for clustering/classifying defect regions; matrix goes to disk, never over MCP. Feature **groups** (`features` selects; default = all applicable): `position` (posed x,y,z), `rest` (restx,y,z), `normal` (nx,ny,nz), `displacement` (disp_x, disp_y, disp_z = posed−rest, disp_mag), `cylinder` (t, r, theta from the joint segment's fitted frame — only when `joint` given; requesting it without a joint → ToolError), `weights` (w_entropy, n_influences, top_weight — only when the dump has j0..j3/w0..w3). Inapplicable requested groups land in `omitted: dict[str, str]` with the reason. `joint=None` → all vertices; `joint` given → masked via `resolve_joint_ids` + `select_segment` (histogram hint on failure).
Writes `data/cache/features_<dumpstem>_<joint|all>.npz` with `X (n,d) float64`, `feature_names (d,) <U`, `dominant_joint (n,) int32`, `vertex_index (n,) int64` (original row indices — how cluster labels map back to mesh vertices).
Returns `FeatureMatrixReport`: `npz_path, dump_path, joint, joint_ids, channel, n_vertices, n_features, feature_names, omitted, feature_stats (dict name → [min, mean, max])`.

**2. `cluster(path: str, columns: list[str] | None = None, key: str | None = None, algorithm: str = "kmeans", n_clusters: int = 3, eps: float = 0.5, min_samples: int = 10, standardize: bool = True, seed: int = 0, save_plot: bool = False) -> str`**
Cluster any numeric table or a `feature_engineer_dump` npz (via `tabular.load_matrix`, which understands the `X`/`feature_names` convention). `algorithm` ∈ `{kmeans, dbscan, agglomerative}` → `KMeans(n_clusters, n_init=10, random_state=seed)`, `DBSCAN(eps, min_samples)`, `AgglomerativeClustering(n_clusters, linkage="ward")`; `standardize` wraps a `StandardScaler`. Irrelevant params ignored (echoed in `params_used`). Metrics: `silhouette_score` (only when `2 <= k_found < n`), `davies_bouldin_score`, kmeans `inertia`, dbscan `n_noise` (label −1 excluded from scores). Labels → `data/cache/clusters_<stem>_<algorithm>.npz` (`labels`, plus `vertex_index` passthrough when input was a feature npz). Plot: seeded 2-component PCA scatter colored by label.
Returns `ClusterReport`: `path, algorithm, params_used (dict[str,float]), standardize, n_samples, n_features, n_clusters_found, cluster_sizes (dict[str,int]), silhouette, silhouette_subsampled, davies_bouldin, inertia, n_noise, labels_path, plot_path` (nullable where inapplicable).

**3. `reduce_dims(path: str, columns: list[str] | None = None, key: str | None = None, method: str = "pca", n_components: int = 2, label_column: str | None = None, standardize: bool = True, seed: int = 0, save_plot: bool = False) -> str`**
`method` ∈ `{pca, lda}` — `PCA(n_components, random_state=seed)`; `LinearDiscriminantAnalysis(n_components=...)` requires `label_column` (via `tabular.load_labels`, excluded from X) and enforces `n_components <= n_classes − 1` with an error stating both numbers. Embedding → `data/cache/reduced_<stem>_<method>.npz` (`Z`, `components`, `feature_names`, `labels` when given). `top_loadings`: per component the 3 largest-|loading| features, `{"pc1": {"disp_mag": 0.82, ...}, ...}`. Plot: 2-D scatter of `Z` (colored by label when available).
Returns `ReduceDimsReport`: `path, method, n_components, n_samples, n_features, explained_variance_ratio (list), cumulative_evr, top_loadings, embedding_path, plot_path`.

**4. `classify_eval(path: str, label_column: str, feature_columns: list[str] | None = None, key: str | None = None, model: str = "knn", cv_folds: int = 5, k: int = 5, max_depth: int | None = None, c: float = 1.0, kernel: str = "rbf", n_estimators: int = 100, standardize: bool = True, seed: int = 0) -> str`**
Fit + k-fold-evaluate one small classifier; persist it for `predict`. `model` ∈ `{knn, naive_bayes, tree, svm, forest, gradient_boost}` → `KNeighborsClassifier(n_neighbors=k)`, `GaussianNB()`, `DecisionTreeClassifier(max_depth, random_state=seed)`, `SVC(C=c, kernel=kernel, random_state=seed)`, `RandomForestClassifier(n_estimators, random_state=seed)` (bagging lane), `GradientBoostingClassifier(n_estimators, random_state=seed)` (boosting lane). Hyperparams not used by the chosen model are ignored and echoed in `params_used`. Pipeline = optional `StandardScaler` + estimator. CV: `StratifiedKFold(cv_folds, shuffle=True, random_state=seed)`; `cross_validate` scoring `accuracy, precision_macro, recall_macro, f1_macro` (mean+std each); confusion matrix from `cross_val_predict` on the same folds. Guards: `n_classes > 20` → refuse (hint: "this is a calculator, not a label encoder"); any class count `< cv_folds` → reduce folds to the min class count and report it. Then refit on all rows → `data/models/<model>_<stem>_<label>.joblib` + sidecar `<same>.meta.json` `{model, params_used, feature_columns, label_column, classes, n_samples, created_utc, cv_metrics}`.
Returns `ClassifyEvalReport`: `path, model, params_used, standardize, n_samples, n_features, feature_columns, label_column, classes (list[str]), class_counts (dict[str,int]), cv_folds_used, metrics (CvMetricsSchema: accuracy_mean/std, precision_macro_mean/std, recall_macro_mean/std, f1_macro_mean/std), confusion_matrix (list[list[int]], rows = true classes in `classes` order), model_path, meta_path`.

**5. `predict(model_path: str, path: str, columns: list[str] | None = None, key: str | None = None) -> str`**
Load a persisted pipeline and predict rows from any supported table. `columns=None` → use `feature_columns` from the model's `.meta.json` sidecar (predicting a csv with the training column names Just Works); explicit `columns` must match the pipeline's `n_features_in_` (error states expected count **and** the sidecar's names). Missing model file → ToolError whose hint lists `data/models/` contents (the models-dir analog of the joint-histogram hint). If the input table also contains the sidecar's `label_column`, accuracy against it is computed → `accuracy_vs_column`. Predictions → `data/cache/predictions_<stem>.npz` (`y_pred`, `y_proba` when the estimator exposes `predict_proba`).
Returns `PredictReport`: `model_path, model, path, n_rows, predicted_class_counts (dict[str,int]), preview (first ≤10 predicted labels as list[str]), accuracy_vs_column (float | None), predictions_path`.

### (b) `engine/mlkit.py` (sklearn wrappers, numpy in/out, no I/O; joblib I/O stays in the toolset)

```python
@dataclass class FeatureMatrix: x: np.ndarray; names: list[str]; vertex_index: np.ndarray
def dump_features(dump: EngineDump, joint_ids: list[int] | None, channel: str,
                  groups: list[str]) -> tuple[FeatureMatrix, dict[str, str]]   # (matrix, omitted)
@dataclass class ClusterResult: labels; n_clusters_found; cluster_sizes: dict[int, int];
                                silhouette: float | None; subsampled: bool;
                                davies_bouldin: float | None; inertia: float | None; n_noise: int | None
def run_cluster(x, algorithm: str, n_clusters: int, eps: float, min_samples: int,
                standardize: bool, seed: int) -> ClusterResult
@dataclass class ReduceResult: z: np.ndarray; evr: list[float]; components: np.ndarray;
                               top_loadings: dict[str, dict[str, float]]
def run_reduce(x, names: list[str], method: str, n_components: int,
               labels: np.ndarray | None, standardize: bool, seed: int) -> ReduceResult
def build_classifier(model: str, k: int, max_depth: int | None, c: float, kernel: str,
                     n_estimators: int, standardize: bool, seed: int) -> Pipeline
@dataclass class CvMetrics: means: dict[str, float]; stds: dict[str, float];
                            confusion: np.ndarray; classes: list[str]; folds_used: int
def crossval_metrics(pipe: Pipeline, x, y, folds: int, seed: int) -> CvMetrics
```

### (c) Pydantic schemas (append to `schemas.py`)

`FeatureMatrixReport`, `ClusterReport`, `ReduceDimsReport`, `CvMetricsSchema`, `ClassifyEvalReport`, `PredictReport` — fields exactly as in (a).

### (d) Test plan — `tests/test_ml.py`

Hand-rolled blobs (no `sklearn.datasets`, keeps golden numbers version-stable): `default_rng(0)`, 3 Gaussians at (0,0), (10,0), (0,10), σ=1, n=100 each.
- cluster kmeans k=3: `n_clusters_found == 3`, `silhouette > 0.7`, every cluster size in [90, 110]; labels npz exists; two runs same seed → identical `labels` arrays (determinism).
- cluster dbscan on blobs + 10 outliers at radius 50: `n_noise >= 8`.
- reduce_dims pca on 3 columns where col0 = N(0,5), col1 = 0.1·col0 + N(0,0.1), col2 = N(0,0.1), `standardize=False`: `evr[0] > 0.9`, `top_loadings["pc1"]` top feature is `col0`.
- reduce_dims lda, 2 separable classes: `n_components=1` works; `n_components=2` → ToolError stating `n_classes − 1 = 1`.
- classify_eval knn on blobs written to a tmp csv with string labels a/b/c: `accuracy_mean > 0.95`, `classes == ["a","b","c"]`, confusion trace ≥ 285/300, model file + meta sidecar exist and sidecar `feature_columns` round-trips; `model="forest"` also `> 0.95` (covers the bagging lane).
- predict round-trip on the same csv with `columns=None` (sidecar path): `accuracy_vs_column > 0.95`, `predicted_class_counts` sums to 300; missing model path → ToolError hint contains the models-dir listing.
- feature_engineer_dump on `make_cylinder(corrugation_amp=0.002, corrugation_freq_cpm=120, with_weights=True)` written to tmp, `joint="5"` + STUB_BONE_MAP `bone_map_path`: names include `disp_mag`, `t`, `w_entropy`; `X.shape[1] == 17` (3+3+3+4+3+1... exact count = position 3 + rest 3 + normal 3 + displacement 4 + cylinder 3 + weights 3 = 19 — implementer asserts the computed `feature_names` length instead of a magic number); mean(disp_mag) ≈ 2/π·0.002 ≈ 0.00127 within ±25%; `joint=None` omits `cylinder` with a reason in `omitted`.
- test_server.py `EXPECTED_TOOLS` grows by the 10 new names; `DSP_TOOLSETS="stats"` registers exactly the 5 stats tools.

### (e) llms.txt rule additions (ml)

```
13. ml pack: matrices never cross MCP — feature_engineer_dump/cluster/reduce_dims write npz under
    data/cache/ (feature npz convention: X + feature_names + vertex_index maps labels back to mesh
    vertices); classify_eval persists a joblib pipeline + .meta.json under data/models/ and predict
    reuses the sidecar's feature_columns. All estimators are seeded (random_state) — identical
    calls must produce identical labels/metrics. Load only models this server wrote (joblib
    deserialization executes code).
```

---

## 3. Shared with other packs / cross-cutting changes

1. **`tabular.py`** (§0) is used by stats + ml here and is the loader engmath/systems/netqueue designers were told to assume. `load_matrix`/`load_labels` are additions beyond the locked two-function surface — same module, needed by ml; other designers may use them too.
2. **Promote `geometry._bone_map_of` → `ply.bone_map_for(dump_path: Path) -> dict[int, str]`** (meta.json first, palette sidecar fallback — logic moves verbatim; geometry.py re-exports/calls it). Needed by tabular's `.ply` lane and `compare_dump_ripples`/`feature_engineer_dump` without importing a toolset from `engine/`.
3. **`_SchemaBase._round_floats`**: extend to also round `dict` **values** that are floats (new schemas carry `dict[str, float]` — params, quantiles, top_loadings, feature_stats; current validator only handles float and list). Backward compatible; existing float-free dicts unaffected.
4. **AppContext**: add `models_dir: Path` + `config.models_dir()` (= `data_dir / "models"`, created on demand like the other dirs); `data/` disposability covers it.
5. **`plots.py` additions** (Agg, Path-returning, same style):
   `save_hist_fit_plot(y, fitted: list[tuple[name, xs, pdf]], path, title)` — density histogram + fitted-pdf overlays;
   `save_regression_plot(x_or_fitted, y, y_hat, resid, path, title, xlabel)` — fit panel over residual panel;
   `save_scatter_plot(z2d, labels, path, title, legend_name)` — embedding/cluster scatter.
6. **Registry**: two lines in `toolsets/__init__.py` (`"stats": _register_stats`, `"ml": _register_ml`), lazy imports as today. sklearn's ~1 s import cost stays inside the lazy loader.
7. **pyproject**: `scikit-learn>=1.5`, `joblib>=1.4`. Pure wheels on bookworm-slim, no GPU — cloud-safe.
8. **Docstring discipline**: each of the 10 tools gets a one-paragraph contract docstring in the `register()` closures, mirroring geometry.py's density (units, defaults, failure hints named).

Key files for the implementer (repo-relative): `src/dsp_server/toolsets/geometry.py` (pattern), `src/dsp_server/schemas.py`, `src/dsp_server/engine/transforms.py`, `src/dsp_server/engine/filters.py`, `src/dsp_server/engine/ply.py`, `tests/synth.py`, `docs/DEVELOPMENT.md`.
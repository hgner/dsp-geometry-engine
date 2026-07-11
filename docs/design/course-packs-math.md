I have all the grounding I need (registry, geometry/imaging tool pattern, schemas base, filters/transforms reuse surface, plots, synth golden cylinder, DEVELOPMENT.md checklist, pyproject). Here is the complete design.

---

# MATH packs design: `engmath` (MAT235+MAT236) and `systems` (ELE301)

Both packs follow the frozen architecture exactly: one pure math module under `src/dsp_server/engine/`, one `toolsets/<name>.py` with `register(mcp, ctx)` + module-level `_impl`s, pydantic models in `schemas.py` (inherit `_SchemaBase`: `extra="forbid"`, 6-sig-fig rounding), every tool returns `model_dump_json()`, failures return `ToolError` JSON, arrays go to disk, plots via `plots.py` (Agg).

## 0. Cross-cutting decisions (both packs)

**New dependency (engmath only):** `pyproject.toml` `dependencies` gains `"sympy>=1.13"`. Pure-python wheel — no Docker change beyond `uv lock` refresh; bookworm-slim unaffected. `sympy` is imported at the top of `engine/symbolic.py`, which is only imported when the `engmath` pack registers (lazy loader in `TOOLSETS`), so `DSP_TOOLSETS` without `engmath` pays zero import cost. With `engmath` on, one-time ~0.5 s at server start — acceptable (server spawns once).

**Array output convention (NEW, shared with stats/ml):** numeric trajectories/responses are saved as `.npz` under `ctx.data_dir / "series"` (created on demand by the toolset — `AppContext` stays frozen). Naming: `<tool>_<safe-stem-or-md5-8-of-expression>.npz`, keys documented per tool below. Propose helper `save_series(path: Path, **arrays: np.ndarray) -> Path` in `engine/tabular.py` (stats designer owns the file — this is a requested addition, see §3 Shared).

**Expression-string security (owned by this design, module `engine/symbolic.py`):**
- `class ExpressionError(ValueError)` — every rejection raises this; toolsets catch it into `ToolError(error="ExpressionError: <reason>", hint=_EXPR_HINT)` where `_EXPR_HINT` = "expressions are mathematical strings parsed by a restricted SymPy whitelist — allowed functions: sin cos tan asin acos atan sinh cosh tanh exp log sqrt Abs sign floor ceiling Heaviside DiracDelta gamma erf Min Max; constants pi, E, I, oo; use ^ or ** for powers; attribute access, `__`, quotes, lambda, semicolons and statements are rejected. This is never Python code."
- `parse_expr_safe(text, symbols=(), allow_primes=False)` pipeline:
  1. length gate: `len(text) > 800` → reject;
  2. character gate: must match `^[0-9A-Za-z_+\-*/^().,\s=<>\[\]']*$` (the `'` and `=` only pass when `allow_primes=True`, used by `solve_ode`'s preprocessor);
  3. token gate: reject on `__`, `lambda`, `import`, `;`, backslash, and `\.(?![0-9])` (dot not followed by a digit — kills attribute access, allows `0.5`);
  4. `sympy.parsing.sympy_parser.parse_expr(text, local_dict=WHITELIST | {s: Symbol(s) for s in symbols}, global_dict={}, transformations=standard_transformations + (convert_xor, implicit_multiplication_application))`. `global_dict={}` is load-bearing: unknown names auto-symbolize instead of resolving into sympy's (or builtins') namespace. NEVER `eval`/`sympify` on raw strings anywhere in either pack.
- `WHITELIST` (module constant): the functions in the hint above plus `pi, E, I, oo, Rational`.

**Known limitation to document (docstrings + llms.txt):** sympy `dsolve`/`integrate` have no interruptible timeout; the mitigation is the 800-char cap, default hints only, and `mode="numeric"` as the escape hatch. ToolError hints must name that escape hatch.

---

## 1. Pack `engmath` — MAT235 + MAT236 (sympy + scipy)

Registry line: `"engmath": _register_engmath` in `toolsets/__init__.py`. File: `toolsets/engmath.py`. Non-goals (explicit): series solutions of ODEs, vector calculus, potential theory, PDEs — out of tool-count budget; revisit only on demand.

### (a) Tool inventory — 5 tools

**1. `solve_ode(equation: str, func: str = "y", var: str = "x", ics: dict[str, float] | None = None, mode: str = "auto", x_end: float = 10.0, n_samples: int = 400, save_plot: bool = False) -> str`**
Contract: solve one ODE given as an equation string with prime notation (`"y'' + 4*y' + 3*y = exp(-x)"`) or explicit `Derivative(y(x),x)`; `ics` keys are `"y(0)"`, `"y'(0)"`, … `mode="symbolic"` (sympy dsolve, error if it fails), `"numeric"` (solve_ivp RK45→Radau fallback; requires full ics), `"auto"` (symbolic first, numeric fallback; when both succeed, cross-checks them). Numeric trajectory saved to `data/series/ode_<md5-8>.npz` (keys `x`, `y`, and `y1..y{order-1}` for derivatives).
Returns `OdeReport`: `equation, order, mode_used ("symbolic"|"numeric"|"both"), classification (list[str], first 3 dsolve hints), method (dsolve hint or "RK45"/"Radau"), solution_str, solution_latex (None when numeric-only), free_constants (list[str], nonempty when ics underdetermine), ics_applied (dict), crosscheck_max_abs_err (float|None), samples_path (str|None), plot_path (str|None)`.

**2. `laplace_transform(expression: str, direction: str = "forward", expression_b: str | None = None, var_t: str = "t", var_s: str = "s") -> str`**
Contract: `direction="forward"` computes L{f(t)} with the convergence condition; `"inverse"` computes L⁻¹{F(s)}; `"convolve"` takes two time-domain expressions, returns (f∗g)(t) via the convolution integral AND verifies the convolution theorem by checking simplify(L{f∗g} − F·G) == 0. Rational F(s) inputs additionally get a pole list (for ROC/inverse-by-partial-fractions teaching).
Returns `LaplaceReport`: `direction, input_str, input_b_str (None unless convolve), result_str, result_latex, convergence_condition (str|None), poles (list[PoleResidueSchema]|None), theorem_verified (bool|None)`.

**3. `linear_algebra(op: str = "solve", matrix: list[list[float]] | None = None, matrix_path: str | None = None, matrix_key: str | None = None, rhs: list[float] | None = None, exact: bool = False) -> str`**
Contract: one matrix tool, five ops. Matrix comes inline as JSON rows or from a file via `tabular.load_matrix(path, key)` (.csv headerless / .npz key / .json list-of-lists). `op="solve"` (AX=B: LU for square, lstsq for over/under-determined, residual norm; `exact=True` & n≤8 adds a sympy-Rational Cramer cross-check), `"analyze"` (det, rank, cond₂, trace, symmetric, orthogonal — the orthogonal-matrix syllabus item), `"eig"`, `"diagonalize"` (P, D saved; orthogonal diagonalization flagged for symmetric A), `"inverse"`. Eigenvalues inline as `[re, im]` pairs for n≤12, else npz only; eigenvectors/P/D always npz (`data/series/linalg_<op>_<md5-8>.npz`, keys `values`, `vectors`/`p`, `d`, `x`, `inv` as applicable).
Returns `LinearAlgebraReport`: `op, shape (list[int]), source (str), det, rank, cond_2, trace, symmetric, orthogonal (all float/bool|None per op), solution (list[float]|None), residual_norm, cramer_verified (bool|None), eigenvalues (list[list[float]]|None), exact_eigenvalues (list[str]|None, exact & n≤8), diagonalizable, orthogonally_diagonalizable, recon_max_abs_err, arrays_path (str|None)`.

**4. `residues_and_integrals(expression: str, var: str = "x", mode: str = "residues") -> str`**
Contract: `mode="residues"` — poles of a rational/meromorphic expression with order and residue at each, plus zeros; `"real_integral"` — evaluate ∫₋∞^∞ f(x)dx by summing upper-half-plane residues (validity gated: rational or rational×{exp(I a x), cos, sin}, no real poles, degree/decay condition — otherwise `ToolError` naming the violated rule), always cross-checked numerically with `scipy.integrate.quad`; `"partial_fractions"` — `apart()` decomposition (the inverse-Laplace workhorse).
Returns `ResidueReport`: `mode, expression, poles (list[PoleResidueSchema]), zeros (list[str]), value_str, value_latex, value_numeric (float|None), quad_value (float|None), quad_abs_err (float|None), method ("residue-uhp"|"apart"|None), result_str (partial fractions)`.

**5. `fourier_series(expression: str | None = None, series_path: str | None = None, column: str | None = None, key: str | None = None, values: list[float] | None = None, interval: list[str] | None = None, period: float | None = None, var: str = "x", n_terms: int = 8, save_plot: bool = False) -> str`**
Contract: exactly one source. Symbolic: `expression` on `interval` (strings parsed safely so `["-pi","pi"]` stays exact; default that) → sympy trig coefficients a₀, aₙ, bₙ and the partial sum. Numeric: a sampled signal (file via `tabular.load_series`, or inline `values` ≤4096) assumed to span exactly ONE period (`period` = its duration, needed only for labeling) → FFT-projected trig coefficients, reconstruction RMS error, and the fraction of Parseval power captured by n_terms. Reconstruction saved to `data/series/fourier_<md5-8>.npz` (keys `t, y, y_recon`); plot overlays signal vs partial sum.
Returns `FourierSeriesReport`: `mode ("symbolic"|"numeric"), n_terms, interval (list[str]|None), period (float|None), a0 (float|None), an, bn (list[float]|None), a0_str, an_str, bn_str (symbolic side, list[str]|None), partial_sum_str, partial_sum_latex (None numeric), dominant_harmonic_n (int), dominant_harmonic_amp (float), recon_rms_err (float|None), parseval_fraction (float|None), samples_path, plot_path`.

### (b) Backing math modules

`src/dsp_server/engine/symbolic.py` (NEW — all sympy, pure, no I/O):
```python
class ExpressionError(ValueError): ...
WHITELIST: dict[str, Any]                          # module constant, see §0
parse_expr_safe(text: str, symbols: Sequence[str] = (), allow_primes: bool = False) -> sp.Expr
parse_ode(equation: str, func: str = "y", var: str = "x") -> tuple[sp.Eq, sp.Function, sp.Symbol, int]   # prime-notation rewrite y''-> Derivative(y(x),(x,2)); returns order
solve_ode_symbolic(eq: sp.Eq, y, x, ics: dict[str, float] | None) -> SymbolicOde   # dataclass: expr, method, classification list, free_constants
ode_rhs_lambda(eq: sp.Eq, y, x) -> Callable[[float, np.ndarray], np.ndarray]        # solve for highest derivative, lambdify state-space RHS; raises ExpressionError if unsolvable
solve_ode_numeric(rhs, order: int, ics_vec: Sequence[float], x0: float, x_end: float, n_samples: int) -> tuple[np.ndarray, np.ndarray]   # RK45, auto-retry Radau; scipy.solve_ivp dense output
laplace_forward(expr, t, s) -> tuple[sp.Expr, sp.Expr | None]                        # (F, convergence cond)
laplace_inverse(expr, s, t) -> sp.Expr
convolve_expr(f, g, t) -> sp.Expr                                                    # Integral(f(tau)g(t-tau),(tau,0,t)).doit()
rational_pole_data(expr, var) -> list[PoleData]                                      # dataclass: point (sp.Expr), order (int), residue (sp.Expr)
real_line_integral_by_residues(expr, var) -> IntegralData                            # dataclass: value, poles_used, method; raises ExpressionError with the violated rule
partial_fractions(expr, var) -> sp.Expr
fourier_series_symbolic(expr, var, a: sp.Expr, b: sp.Expr, n_terms: int) -> SymbolicFourier  # dataclass: a0, an list, bn list, partial_sum
latex_and_str(expr) -> tuple[str, str]                                               # (sp.latex, sp.sstr)
```

`src/dsp_server/engine/linalg.py` (NEW — numpy, optional sympy for exact paths):
```python
analyze_matrix(a: np.ndarray) -> MatrixAnalysis            # det|None(nonsquare), rank, cond_2, trace|None, symmetric, orthogonal
solve_system(a: np.ndarray, b: np.ndarray) -> LinSolve     # x, residual_norm, method ("lu"|"lstsq")
cramer_exact(a: np.ndarray, b: np.ndarray) -> np.ndarray   # sympy Rational, n<=8 only (ValueError otherwise)
eig_decomposition(a: np.ndarray) -> EigResult              # values (complex), vectors, diagonalizable, orthogonally_diagonalizable
diagonalize(a: np.ndarray) -> DiagResult                   # p, d, recon_max_abs_err  (raises ValueError when defective)
exact_eigenvalues(a: np.ndarray) -> list[str]              # sympy path, n<=8
matrix_inverse(a: np.ndarray) -> tuple[np.ndarray, float]  # (inv, max|A@inv - I|)
```

Addition to existing `src/dsp_server/engine/filters.py` (shared with systems pack — do NOT duplicate):
```python
fourier_coefficients(y: np.ndarray, n_terms: int) -> tuple[float, np.ndarray, np.ndarray]
# assumes y spans exactly one period, uniform sampling; rfft projection:
# a0 = mean(y); a_n = 2*Re(X_n)/N; b_n = -2*Im(X_n)/N
```

### (c) Pydantic schemas (schemas.py, all on `_SchemaBase`)
- `PoleResidueSchema`: `location_str: str, re: float, im: float, order: int, residue_str: str | None, residue_re: float | None, residue_im: float | None`
- `OdeReport`, `LaplaceReport`, `LinearAlgebraReport`, `ResidueReport`, `FourierSeriesReport` — fields exactly as listed in (a).

### (d) Test plan — `tests/test_engmath.py` (no engine, `_impl` calls with a tmp AppContext as in existing tests)
1. `solve_ode` symbolic: `"y'' + y = 0"`, ics `{"y(0)":0,"y'(0)":1}` → `simplify(sol - sin(x)) == 0`; `mode_used=="both"`, `crosscheck_max_abs_err < 1e-6`. First-order: `"y' = -2*y"`, `y(0)=3` → `3*exp(-2*x)`. Undetermined-coeffs shape: `"y'' + 3*y' + 2*y = exp(-3*x)"` particular term `exp(-3x)/2` present after simplify.
2. `solve_ode` numeric-only: `mode="numeric"` on `"y' = -2*y"` → max|y_num − 3e^{−2x}| < 1e−6 on [0,10]; npz exists with keys `x,y`.
3. `laplace_transform`: forward `t*exp(-2*t)` → `1/(s+2)**2`, convergence contains `-2`; inverse `1/(s**2+1)` → `sin(t)`; convolve `exp(-t)` with `1` → `1 - exp(-t)` and `theorem_verified is True`.
4. `linear_algebra`: solve `[[2,1],[1,3]]`, rhs `[5,10]` → `[1.0, 3.0]`, residual < 1e−12, `exact=True` → `cramer_verified is True`; eig `[[4,1],[2,3]]` → {5, 2}; analyze on a rotation matrix → `orthogonal is True`, det ≈ 1, cond_2 ≈ 1; diagonalize defective `[[1,1],[0,1]]` → `diagonalizable is False` (report, not ToolError).
5. `residues_and_integrals`: `1/(x**2+1)` residues → poles ±i order 1, residue at +i = −i/2; real_integral → π (`value_numeric ≈ 3.14159`, `quad_abs_err < 1e-8`); `1/(x**2+1)**2` → π/2; `1/(x-1)` real_integral → ToolError naming the real-pole rule; partial_fractions `1/(s**2+3*s+2)` → `1/(s+1) - 1/(s+2)`.
6. `fourier_series` symbolic: `x` on `["-pi","pi"]`, n_terms 3 → bn_str ≃ [2, −1, 2/3], a's zero. Numeric: ±1 square wave, 1024 samples, one period → `b1 ≈ 4/π = 1.27324` (rel 1e−3), `b2 ≈ 0` (abs < 1e−6), `b3 ≈ 0.424413`, `recon_rms_err` decreases from n_terms=1→8, `parseval_fraction > 0.9` at n_terms=8.
7. Security: each of `"__import__('os').system('x')"`, `"().__class__"`, `"lambda x: x"`, `"open('f')"` (dot-gate + no `open` in whitelist), 900-char string → JSON parses to `ToolError` with `error` starting `"ExpressionError:"` and the whitelist hint. Also: unknown bare name `"foo(3)"` parses harmlessly as a Symbol-application, not a crash.

### (e) llms.txt rule addition (append as the next numbered rule)
```
12. engmath pack (MAT235/236): solve_ode, laplace_transform, linear_algebra, residues_and_integrals,
    fourier_series. Expression strings are parsed by a restricted SymPy whitelist (never eval; `__`,
    attribute access, quotes, lambda -> ToolError naming the allowed functions). Symbolic results carry
    latex + plain strings; numeric fallbacks/trajectories live under data/series/*.npz (path in the JSON).
    Symbolic solves can be slow on hard inputs — mode="numeric" is the documented escape hatch.
```

### (f) Shared surface
- Requests to `tabular.py` (stats designer owns): `load_matrix(path: str, key: str | None = None) -> np.ndarray` and `save_series(path: Path, **arrays) -> Path`; engmath consumes `load_series` for `fourier_series` file input.
- `filters.fourier_coefficients` shared with systems; `parse_expr_safe` available to any future symbolic tool.
- New generic plot helper (also used by systems/stats/ml): `plots.save_xy_plot(curves: list[tuple[np.ndarray, np.ndarray, str]], path, title, xlabel, ylabel) -> Path`.

---

## 2. Pack `systems` — ELE301 (scipy.signal)

Registry line: `"systems": _register_systems`. File: `toolsets/systems.py`. Math module: `engine/ltisys.py`. Hard reuse rule: NO new PSD/peak/resample code — `sampling_check` is built on `transforms.spectrum` + `filters.band_energy` + `filters.spectral_peaks`; resampling uses `filters.resample_profile`.

System input convention (tools 1–3): either `num`+`den` (descending powers of s, or of z with DT convention `den=[1,-0.5]` ≡ 1−0.5z⁻¹) or `zeros`/`poles` as `[re, im]` pairs + `gain` (JSON has no complex type). `domain: "s"|"z"`; `"z"` requires `dt`. Exactly one form, else ToolError with a hint showing both forms.

### (a) Tool inventory — 5 tools

**1. `lti_response(num: list[float] | None = None, den: list[float] | None = None, zeros: list[list[float]] | None = None, poles: list[list[float]] | None = None, gain: float | None = None, domain: str = "s", dt: float | None = None, input_kind: str = "step", input_path: str | None = None, input_key: str | None = None, t_end: float | None = None, n_samples: int = 500, save_plot: bool = False) -> str`**
Contract: time response of a CT or DT LTI system. `input_kind`: `"step" | "impulse" | "ramp" | "custom"` (custom loads u via `tabular.load_series`, dt from the file or the system's dt). `t_end` auto-defaults to ≈7 slowest time constants for stable systems (n_samples·dt for DT). Step responses get the full metrics block. Arrays saved to `data/series/lti_<md5-8>.npz` (keys `t, u, y`).
Returns `LtiResponseReport`: `domain, dt, input_kind, stable, poles (list[list[float]]), dc_gain (float|None), step (StepMetricsSchema|None), y_min, y_max, t_at_peak, response_path, plot_path`.

**2. `pole_zero(num, den, zeros, poles, gain, domain="s", dt=None, save_plot=False) -> str`**
Contract: pole-zero map + stability/dynamics reading. Per pole: damping ratio ζ, natural frequency ωₙ (rad/s), time constant (CT: from Re; DT: via s = ln(z)/dt when dt given), magnitude (DT). Reports the causal ROC as text ("Re(s) > −1" / "|z| > 0.9"), minimum-phase and proper flags. Plot: s-plane, or z-plane with unit circle. This plus `bode(domain="z")` covers geometric DTFT evaluation from pole-zero plots.
Returns `PoleZeroReport`: `domain, dt, gain, poles (list[PoleDynamicsSchema]), zeros (list[list[float]]), stable, marginally_stable, causal_roc (str), minimum_phase, proper, plot_path`.

**3. `bode(num, den, zeros, poles, gain, domain="s", dt=None, w_min: float | None = None, w_max: float | None = None, n_points: int = 400, save_plot: bool = False) -> str`**
Contract: magnitude/phase over a log grid (auto-ranged 2 decades beyond the pole/zero spread; DT capped at π/dt) with stability margins computed by interpolated crossover search, treating the given H as the OPEN-loop transfer function (documented in the docstring). Also −3 dB bandwidth, resonant peak, and the high-frequency slope in dB/decade (≈ −20·(#poles−#zeros)). Arrays to `data/series/bode_<md5-8>.npz` (keys `w, mag_db, phase_deg`).
Returns `BodeReport`: `domain, dt, dc_gain_db (float|None — None when a pole sits at the origin), gain_margin_db, gain_margin_freq, phase_margin_deg, phase_margin_freq (all None-able), bandwidth_3db, resonant_peak_db, resonant_freq, hf_slope_db_per_decade, arrays_path, plot_path`.

**4. `sampling_check(fs: float, series_path: str | None = None, column: str | None = None, key: str | None = None, values: list[float] | None = None, dt: float | None = None, dump: str | None = None, joint: str = "armLowerL", channel: str = "posed", bone_map_path: str | None = None, save_plot: bool = False) -> str`**
Contract: the aliasing verdict — and the bridge tool onto the ELE407 lane. Exactly one source: any series file (`tabular.load_series`; dt from the file or the `dt` param), an inline `values` list (≤4096, requires `dt`), or an engine PLY `dump`+`joint`+`channel` whose forearm axial profile becomes the signal (then frequencies are cycles/meter and `fs` is samples/meter — same joint resolution and error-histogram hint as the geometry tools). Spectrum via `transforms.spectrum` on a `Signal1D`; energy above the proposed Nyquist via `filters.band_energy`; top peaks via `filters.spectral_peaks` with each peak's folded alias frequency. Verdict thresholds: `safe` < 1% energy above Nyquist, `marginal` < 5%, else `aliased`. `fs_suggested = 2.5 × bandwidth_99pct`. Plot: PSD with the Nyquist line and folded-peak arrows.
Returns `SamplingCheckReport`: `source (str), n_samples, dt, freq_unit ("Hz"|"cycles/m"), fs_proposed, nyquist, energy_fraction_above_nyquist, bandwidth_99pct, verdict, folded_peaks (list[FoldedPeakSchema]), fs_suggested, plot_path`.

**5. `convolve_signals(path_a: str | None = None, path_b: str | None = None, key_a: str | None = None, key_b: str | None = None, column_a: str | None = None, column_b: str | None = None, values_a: list[float] | None = None, values_b: list[float] | None = None, dt: float | None = None, mode: str = "convolve", scale: str = "continuous", normalize: bool = False, save_plot: bool = False) -> str`**
Contract: convolution or cross-correlation of two series (files via `tabular.load_series` or inline lists ≤4096) with honest dt handling: `scale="continuous"` multiplies the discrete sum by dt (approximates the CT integral — rect∗rect gives a unit triangle, not N), `"discrete"` is the raw DT sum. If the two files carry different dts, b is resampled onto a's grid via `filters.resample_profile` (rational approx, `Fraction(dt_a/dt_b).limit_denominator(1000)`) and `resampled_b=true` is reported. Correlation with `normalize=true` reports the normalized cross-correlation peak and its lag. Result to `data/series/conv_<md5-8>.npz` (keys `t` (or `lag`), `y`).
Returns `ConvolutionReport`: `mode, scale, n_a, n_b, dt, n_out, y_peak, t_at_peak (convolve) | lag_at_peak, peak_corrcoef (correlate+normalize, else None), energy_out, resampled_b, result_path, plot_path`.

### (b) Backing math module — `src/dsp_server/engine/ltisys.py` (NEW, pure numpy/scipy.signal)
```python
build_system(num, den, zeros, poles, gain, domain: str, dt: float | None) -> sps.lti | sps.dlti   # validation + [re,im]->complex
zpk_of(sys) -> tuple[np.ndarray, np.ndarray, float]
stability_of(poles: np.ndarray, domain: str) -> tuple[bool, bool]                # (stable, marginally_stable)
roc_text(poles: np.ndarray, domain: str) -> str
pole_dynamics(poles: np.ndarray, domain: str, dt: float | None) -> list[PoleDyn] # dataclass: re, im, magnitude, damping_ratio, natural_freq_rad_s, time_constant_s (None-able)
default_t_end(poles: np.ndarray, domain: str, dt: float | None) -> float
time_response(sys, kind: str, t: np.ndarray, u: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]  # (t,u,y): step/impulse/lsim | dstep/dimpulse/dlsim
step_metrics(t: np.ndarray, y: np.ndarray) -> StepMetrics    # dataclass: final_value, rise_time_10_90, settling_time_2pct, overshoot_pct, peak_time (None-able when non-convergent)
default_w_grid(zeros, poles, domain, dt, n_points) -> np.ndarray
freq_response(sys, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]   # (w, mag_db, unwrapped phase_deg) via sps.bode/dbode
stability_margins(w, mag_db, phase_deg) -> Margins           # dataclass: gm_db, w_gain_cross, pm_deg, w_phase_cross (None-able; linear interp of crossings — scipy has no margin())
bandwidth_3db(w, mag_db) -> float | None
hf_slope(w, mag_db) -> float                                 # LSQ slope over the top decade
alias_fold(f: np.ndarray, fs: float) -> np.ndarray           # fold into [0, fs/2]
sampling_metrics(freqs, psd, fs) -> SamplingMetrics          # dataclass: nyquist, frac_above, bw_99 (uses filters.band_energy internally? NO — takes precomputed psd; pure)
convolve_uniform(a, b, dt, mode: str, scale: str, normalize: bool) -> tuple[np.ndarray, np.ndarray]  # (t_or_lag, y); sps.correlate + correlation_lags
```

### (c) Pydantic schemas
- `StepMetricsSchema`: `final_value, rise_time_10_90, settling_time_2pct, overshoot_pct, peak_time` (all `float | None`)
- `PoleDynamicsSchema`: `re, im, magnitude (float|None), damping_ratio (float|None), natural_freq_rad_s (float|None), time_constant_s (float|None)`
- `FoldedPeakSchema`: `f_true: float, f_alias: float, prominence_db: float, aliased: bool`
- `LtiResponseReport`, `PoleZeroReport`, `BodeReport`, `SamplingCheckReport`, `ConvolutionReport` — fields as in (a).

### (d) Test plan — `tests/test_systems.py`
1. `lti_response` H(s)=1/(s+1) step: `final_value ≈ 1`, `rise_time_10_90 ≈ 2.19722` (ln 9, rel 2%), `settling_time_2pct ≈ 3.912`, `overshoot_pct < 0.5`; npz `t,u,y` exists. Second-order ωₙ=1, ζ=0.5 (`den=[1,1,1]`): `overshoot_pct ≈ 16.30` (rel 3%), `peak_time ≈ 3.6276` (rel 3%).
2. DT: `num=[1], den=[1,-0.5], domain="z", dt=0.1`, impulse → y[n]=0.5ⁿ (first 5 samples exact rel 1e−9), `stable is True`; pole at `[1.1, 0]` → `stable is False`.
3. `pole_zero` `den=[1,2,5]` → poles −1±2i, `damping_ratio ≈ 0.44721`, `natural_freq_rad_s ≈ 2.23607`, `causal_roc == "Re(s) > -1"`; z-domain pole 0.9 with dt → `causal_roc == "|z| > 0.9"`, time_constant ≈ −dt/ln 0.9.
4. `bode` 1/(s+1): mag at ω=1 ≈ −3.0103 dB (interp, abs 0.05), phase ≈ −45°, `hf_slope ≈ −20` (abs 1.5); margins on 1/(s(s+1)(s+2)): `gain_margin_db ≈ 15.563` (abs 0.1) at ω=√2, `phase_margin_deg ≈ 53.4` (abs 0.5) at ω≈0.4457; 1/(s+1) → both margins None (no crossings).
5. `sampling_check` file lane: 80 Hz tone sampled at dt=0.001 written to a tmp npz → `fs=100`: `verdict=="aliased"`, folded_peaks contains `{f_true≈80, f_alias≈20, aliased: true}`, `energy_fraction_above_nyquist > 0.3`; `fs=200`: `verdict=="safe"`. Dump lane: `make_cylinder(corrugation_amp=0.002, corrugation_freq_cpm=60)` written via `write_engine_ply`, `fs=100` samples/m → `freq_unit=="cycles/m"`, top folded peak `f_true≈60 → f_alias≈40`, verdict aliased; bad joint → ToolError whose hint carries the per-joint histogram (same helper as geometry).
6. `convolve_signals`: two unit rects of duration 1 s (dt=0.01, inline values), continuous scale → `y_peak ≈ 1.0` (rel 1%), `t_at_peak ≈ 1.0`, support 2 s; discrete scale → peak == 100; correlate sin vs same sin shifted by 0.25 s, normalize → `lag_at_peak ≈ −0.25` (sign convention documented: positive lag = b lags a), `peak_corrcoef > 0.99`; mixed dts (0.01 vs 0.02 files) → `resampled_b is True` and peak within 2% of the equal-dt case.

### (e) llms.txt rule addition
```
13. systems pack (ELE301): lti_response, pole_zero, bode, sampling_check, convolve_signals. CT vs DT is
    the domain param ('s'|'z'; 'z' requires dt; DT den convention [1,-0.5] = 1-0.5z^-1). bode margins
    treat H as the OPEN-loop L(s). sampling_check is the aliasing bridge: it takes any series file OR an
    engine dump+joint (units flip to cycles/meter, fs = samples/meter) — run it on forearm profiles.
```

### (f) Shared surface
- Consumes `tabular.load_series` (stats-owned) everywhere a file enters; consumes `transforms.spectrum`, `filters.band_energy/spectral_peaks/resample_profile`, `transforms.extract_forearm_signal` + the geometry pack's `_bone_map_of`-style histogram hint (extract that helper into `toolsets/_shared.py` or import from geometry — implementer's choice, but do not copy-paste).
- New plot helpers in `plots.py` used across packs: `save_xy_plot` (see engmath §f), `save_pole_zero_plot(poles, zeros, domain, path, title)`, `save_bode_plot(w, mag_db, phase_deg, margins, path, title)`, `save_spectrum_nyquist_plot(freqs, psd, nyquist, folded_peaks, path, title)`.
- `data/series/` npz convention shared with engmath/stats/ml.

---

## 3. Checklist deltas outside the packs (docs/DEVELOPMENT.md step 3–5 artifacts)
- `toolsets/__init__.py`: two registry lines (`engmath`, `systems`).
- `pyproject.toml`: `sympy>=1.13` (engmath); `uv lock`; Dockerfile unchanged (pure wheel).
- README tool table + `DSP_TOOLSETS` docs mention the two packs; llms.txt rules 12–13 above (renumber against whatever the stats/imaging/ml designers append — coordinate final numbering at implementation time).
- Coordination requests to the stats designer (tabular.py owner): `load_matrix(path, key=None) -> np.ndarray`, `save_series(path, **arrays) -> Path`, and dt discovery convention for `.npz` (reserved key `dt`) — sampling_check/convolve/fourier_series depend on those three.

Key file paths: `c:\Users\hgner\hakantest\proje10\src\dsp_server\engine\symbolic.py`, `...\engine\linalg.py`, `...\engine\ltisys.py` (new); `...\toolsets\engmath.py`, `...\toolsets\systems.py` (new); `...\engine\filters.py` (+`fourier_coefficients`), `...\src\dsp_server\plots.py` (+4 helpers), `...\src\dsp_server\schemas.py` (+12 models), `...\tests\test_engmath.py`, `...\tests\test_systems.py`, `...\pyproject.toml`, `...\llms.txt`.
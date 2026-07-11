"""Engmath tool pack (MAT235 + MAT236): ODEs, Laplace, linear algebra, residues,
Fourier series — sympy/scipy behind the same contract as every other pack.

Five tools: solve_ode, laplace_transform, linear_algebra, residues_and_integrals,
fourier_series. Every tool returns a pydantic model's ``model_dump_json()``; every
failure returns :class:`~dsp_server.schemas.ToolError` JSON (TabularError messages
go into the hint verbatim, expression rejections carry the whitelist hint). All
symbolic outputs carry {latex, plain} string pairs; numeric trajectories and any
array too large to inline live under ``data/series/*.npz`` — arrays never cross
the MCP boundary.

Security: every expression string is parsed by
:func:`dsp_server.engine.symmath.parse_expr_safe` (restricted SymPy whitelist,
never Python ``eval``). Symbolic solves have no interruptible timeout — the parse
gates cap the blast radius and ``solve_ode(mode="numeric")`` is the documented
escape hatch for hard symbolic inputs.

Testability: module-level ``_impl`` functions take the :class:`AppContext`
explicitly; the registered closures delegate 1:1.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import sympy as sp
from mcp.server.fastmcp import FastMCP

from dsp_server import plots
from dsp_server.engine import symmath, tabular, transforms
from dsp_server.engine.transforms import Signal1D
from dsp_server.schemas import SchemaBase, ToolError
from dsp_server.toolsets import AppContext

MAX_INLINE_VALUES = 4096  # inline sample lists (fourier_series values=)
MAX_INLINE_SOLUTION = 64  # critique #12: larger solutions go npz-only
MAX_INLINE_EIGENVALUES = 12
MAX_N_TERMS = 64

_EXPR_HINT = (
    "expressions are mathematical strings parsed by a restricted SymPy whitelist — "
    f"allowed functions: {symmath.ALLOWED_FUNCTIONS}; constants: {symmath.ALLOWED_CONSTANTS}; "
    "use ** or ^ for powers and explicit '*' for products; attribute access, __, quotes, "
    "lambda, semicolons, huge integer literals and power towers are rejected. This is never "
    "Python code. Symbolic solves can be slow on hard inputs — solve_ode mode='numeric' is "
    "the escape hatch."
)
_MATRIX_HINT = (
    "pass the matrix inline as JSON rows (matrix=[[2,1],[1,3]]) OR as matrix_path= pointing at "
    "a .csv/.tsv/.npz/.json file (.json may be a list of equal-length numeric lists; "
    "matrix_key= selects an .npz member) — exactly one of the two"
)
_FOURIER_HINT = (
    "pass exactly one source: expression= (symbolic, with interval=[lo,hi] strings, default "
    "['-pi','pi']), series_path= (any tabular/npz/ply series file, column=/key= select), or "
    f"values= (inline list, <= {MAX_INLINE_VALUES} samples). Numeric sources are assumed to "
    "span exactly ONE period."
)


# --------------------------------------------------------------------------- #
# Response models (local to the pack, on the shared SchemaBase contract)


class PoleResidueSchema(SchemaBase):
    location_str: str
    re: float
    im: float
    order: int
    residue_str: str | None = None
    residue_re: float | None = None
    residue_im: float | None = None


class OdeReport(SchemaBase):
    equation: str
    order: int
    mode_used: str
    classification: list[str]
    method: str | None = None
    solution_str: str | None = None
    solution_latex: str | None = None
    free_constants: list[str]
    ics_applied: dict[str, float]
    crosscheck_max_abs_err: float | None = None
    samples_path: str | None = None
    plot_path: str | None = None


class LaplaceReport(SchemaBase):
    direction: str
    input_str: str
    input_b_str: str | None = None
    result_str: str
    result_latex: str
    convergence_condition: str | None = None
    poles: list[PoleResidueSchema] | None = None
    theorem_verified: bool | None = None


class LinearAlgebraReport(SchemaBase):
    op: str
    shape: list[int]
    source: str
    det: float | None = None
    rank: int | None = None
    cond_2: float | None = None
    trace: float | None = None
    symmetric: bool | None = None
    orthogonal: bool | None = None
    solution: list[float] | None = None
    residual_norm: float | None = None
    cramer_verified: bool | None = None
    eigenvalues: list[list[float]] | None = None
    exact_eigenvalues: list[str] | None = None
    diagonalizable: bool | None = None
    orthogonally_diagonalizable: bool | None = None
    recon_max_abs_err: float | None = None
    arrays_path: str | None = None


class ResidueReport(SchemaBase):
    mode: str
    expression: str
    poles: list[PoleResidueSchema] = []
    zeros: list[str] = []
    value_str: str | None = None
    value_latex: str | None = None
    value_numeric: float | None = None
    quad_value: float | None = None
    quad_abs_err: float | None = None
    method: str | None = None
    result_str: str | None = None


class FourierSeriesReport(SchemaBase):
    mode: str
    n_terms: int
    interval: list[str] | None = None
    period: float | None = None
    a0: float | None = None
    an: list[float] | None = None
    bn: list[float] | None = None
    a0_str: str | None = None
    an_str: list[str] | None = None
    bn_str: list[str] | None = None
    partial_sum_str: str | None = None
    partial_sum_latex: str | None = None
    dominant_harmonic_n: int
    dominant_harmonic_amp: float
    recon_rms_err: float | None = None
    parseval_fraction: float | None = None
    samples_path: str | None = None
    plot_path: str | None = None


# --------------------------------------------------------------------------- #
# Shared helpers


def _series_dir(ctx: AppContext) -> Path:
    d = ctx.data_dir / "series"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _md5_8(*parts: bytes | str) -> str:
    h = hashlib.md5()
    for p in parts:
        h.update(p.encode("utf-8") if isinstance(p, str) else p)
    return h.hexdigest()[:8]


def _error(exc: Exception, hint: str | None = None) -> str:
    if isinstance(exc, tabular.TabularError):
        hint = str(exc)  # hint-ready by contract
    elif hint is None and isinstance(exc, symmath.ExpressionError):
        hint = _EXPR_HINT
    return ToolError(error=f"{type(exc).__name__}: {exc}", hint=hint).model_dump_json()


def _to_float(value: sp.Expr) -> float:
    return float(sp.re(value).evalf())


def _pole_schema(pole: symmath.PoleData) -> PoleResidueSchema:
    loc = complex(pole.point.evalf())
    residue_str = residue_re = residue_im = None
    if pole.residue is not None:
        residue_str = sp.sstr(pole.residue)
        if pole.residue.is_number:
            rc = complex(pole.residue.evalf())
            residue_re, residue_im = rc.real, rc.imag
    return PoleResidueSchema(
        location_str=sp.sstr(pole.point),
        re=loc.real,
        im=loc.imag,
        order=pole.order,
        residue_str=residue_str,
        residue_re=residue_re,
        residue_im=residue_im,
    )


def _dominant_harmonic(an: np.ndarray, bn: np.ndarray) -> tuple[int, float]:
    amps = np.sqrt(np.asarray(an, dtype=np.float64) ** 2 + np.asarray(bn, dtype=np.float64) ** 2)
    if amps.size == 0 or float(amps.max()) == 0.0:
        return 0, 0.0
    i = int(np.argmax(amps))
    return i + 1, float(amps[i])


# --------------------------------------------------------------------------- #
# Tool 1: solve_ode


def _solve_ode(
    ctx: AppContext,
    equation: str,
    func: str = "y",
    var: str = "x",
    ics: dict[str, float] | None = None,
    mode: str = "auto",
    x_end: float = 10.0,
    n_samples: int = 400,
    save_plot: bool = False,
) -> str:
    try:
        if mode not in ("auto", "symbolic", "numeric"):
            return ToolError(
                error=f"unknown mode '{mode}'", hint="mode must be auto|symbolic|numeric"
            ).model_dump_json()
        eq, f, x, order = symmath.parse_ode(equation, func, var)
        sym_ics, ics_vec, x0 = symmath.parse_ics(ics, func, var, order)

        if mode == "numeric" and ics_vec is None:
            return ToolError(
                error="numeric mode requires a complete set of initial conditions",
                hint=(
                    f"pass ics with keys {func}(x0), {func}'(x0), ... for every order "
                    f"0..{order - 1}, all at the same point x0"
                ),
            ).model_dump_json()

        sob: symmath.SymbolicOde | None = None
        symbolic_exc: Exception | None = None
        if mode in ("auto", "symbolic"):
            try:
                sob = symmath.solve_ode_symbolic(eq, f, x, sym_ics)
            except Exception as exc:
                if mode == "symbolic":
                    return _error(
                        exc,
                        hint=(
                            "symbolic dsolve failed on this equation — mode='numeric' is the "
                            "escape hatch (requires a full ics set)"
                        ),
                    )
                symbolic_exc = exc

        xs: np.ndarray | None = None
        ys: np.ndarray | None = None
        num_method: str | None = None
        if mode == "numeric" or (mode == "auto" and ics_vec is not None):
            try:
                rhs = symmath.ode_rhs_lambda(eq, f, x, order)
                xs, ys, num_method = symmath.solve_ode_numeric(rhs, order, ics_vec, x0, x_end, n_samples)
            except Exception:
                if mode == "numeric" or sob is None:
                    raise
                xs = ys = None  # auto with a symbolic solution in hand: report that

        if sob is None and xs is None:
            return _error(
                symbolic_exc or symmath.ExpressionError("no solution path succeeded for this equation"),
                hint=(
                    "symbolic dsolve failed and numeric integration needs a full ics set — "
                    "pass ics for every order at one point and mode='numeric'"
                ),
            )

        mode_used = (
            "both" if (sob is not None and xs is not None) else ("symbolic" if sob is not None else "numeric")
        )

        crosscheck: float | None = None
        if sob is not None and xs is not None and ys is not None:
            try:
                fn = sp.lambdify(x, sob.expr, modules=["numpy"])
                y_sym = np.real(np.asarray(fn(xs), dtype=np.complex128))
                y_sym = np.broadcast_to(y_sym, xs.shape)
                crosscheck = float(np.max(np.abs(y_sym - ys[0])))
            except Exception:
                crosscheck = None

        samples_path: str | None = None
        plot_path: str | None = None
        if xs is not None and ys is not None:
            stem = _md5_8(equation, str(ics), f"{x_end}:{n_samples}")
            arrays: dict[str, np.ndarray] = {"x": xs, "y": ys[0]}
            for k in range(1, order):
                arrays[f"y{k}"] = ys[k]
            out = _series_dir(ctx) / f"ode_{stem}.npz"
            np.savez(out, **arrays)
            samples_path = str(out)
            if save_plot:
                dt = float(xs[-1] - xs[0]) / max(xs.size - 1, 1)
                sig = Signal1D(y=ys[0], dt=dt, t0=float(xs[0]))
                freqs, psd = transforms.spectrum(sig)
                plot_path = str(
                    plots.save_signal_plot(
                        sig, freqs, psd, ctx.plots_dir / f"ode_{stem}.png", f"solve_ode: {equation}"
                    )
                )

        solution_str = solution_latex = None
        if sob is not None:
            solution_latex, solution_str = symmath.latex_and_str(sob.expr)

        return OdeReport(
            equation=equation,
            order=order,
            mode_used=mode_used,
            classification=sob.classification if sob is not None else [],
            method=sob.method if sob is not None else num_method,
            solution_str=solution_str,
            solution_latex=solution_latex,
            free_constants=sob.free_constants if sob is not None else [],
            ics_applied={str(k): float(v) for k, v in (ics or {}).items()},
            crosscheck_max_abs_err=crosscheck,
            samples_path=samples_path,
            plot_path=plot_path,
        ).model_dump_json()
    except Exception as exc:  # the MCP boundary must never see a raise
        return _error(exc)


# --------------------------------------------------------------------------- #
# Tool 2: laplace_transform


def _laplace_transform(
    ctx: AppContext,
    expression: str,
    direction: str = "forward",
    expression_b: str | None = None,
    var_t: str = "t",
    var_s: str = "s",
) -> str:
    try:
        _ = ctx  # laplace is pure — kept for the uniform _impl signature
        if direction not in ("forward", "inverse", "convolve"):
            return ToolError(
                error=f"unknown direction '{direction}'",
                hint="direction must be forward|inverse|convolve",
            ).model_dump_json()
        t = sp.Symbol(var_t)
        s = sp.Symbol(var_s)

        if direction == "forward":
            f = symmath.parse_expr_safe(expression, symbols=(var_t,))
            big_f, conv = symmath.laplace_forward(f, t, s)
            result = sp.simplify(big_f)
            latex, plain = symmath.latex_and_str(result)
            return LaplaceReport(
                direction=direction,
                input_str=sp.sstr(f),
                result_str=plain,
                result_latex=latex,
                convergence_condition=conv,
            ).model_dump_json()

        if direction == "inverse":
            big_f = symmath.parse_expr_safe(expression, symbols=(var_s,))
            result = symmath.laplace_inverse(big_f, s, t)
            poles = None
            if big_f.is_rational_function(s):
                poles = [_pole_schema(p) for p in symmath.rational_pole_data(big_f, s)]
            latex, plain = symmath.latex_and_str(result)
            return LaplaceReport(
                direction=direction,
                input_str=sp.sstr(big_f),
                result_str=plain,
                result_latex=latex,
                poles=poles,
            ).model_dump_json()

        # convolve
        if expression_b is None:
            return ToolError(
                error="direction='convolve' needs expression_b",
                hint="pass two time-domain expressions f(t) and g(t)",
            ).model_dump_json()
        f = symmath.parse_expr_safe(expression, symbols=(var_t,))
        g = symmath.parse_expr_safe(expression_b, symbols=(var_t,))
        conv_expr = symmath.convolve_expr(f, g, t)

        # Convolution-theorem check (critique #23): True on exact zero, None when
        # symbolically unresolved but not refuted, False only on numeric refutation.
        theorem: bool | None
        try:
            lhs = sp.laplace_transform(conv_expr, t, s, noconds=True)
            rhs = sp.laplace_transform(f, t, s, noconds=True) * sp.laplace_transform(g, t, s, noconds=True)
            diff = sp.simplify(lhs - rhs)
            if diff == 0:
                theorem = True
            else:
                theorem = None
                for s_val in (1.7, 2.6, 3.9):
                    try:
                        v = complex(diff.subs(s, s_val).evalf())
                    except (TypeError, ValueError):
                        continue
                    if abs(v) > 1e-6:
                        theorem = False
                        break
        except Exception:
            theorem = None

        latex, plain = symmath.latex_and_str(conv_expr)
        return LaplaceReport(
            direction=direction,
            input_str=sp.sstr(f),
            input_b_str=sp.sstr(g),
            result_str=plain,
            result_latex=latex,
            theorem_verified=theorem,
        ).model_dump_json()
    except Exception as exc:
        return _error(exc)


# --------------------------------------------------------------------------- #
# Tool 3: linear_algebra


def _load_matrix_input(
    matrix: list[list[float]] | None, matrix_path: str | None, matrix_key: str | None
) -> tuple[np.ndarray, str]:
    if (matrix is None) == (matrix_path is None):
        raise ValueError("pass exactly one of matrix= (inline rows) or matrix_path= (file)")
    if matrix is not None:
        a = np.asarray(matrix, dtype=np.float64)
        if a.ndim != 2:
            raise ValueError(f"inline matrix must be a list of equal-length rows, got shape {a.shape}")
        return a, "inline"
    x, _names = tabular.load_matrix(matrix_path, key=matrix_key)  # critique #1: unpack the tuple
    return x, str(matrix_path)


def _linear_algebra(
    ctx: AppContext,
    op: str = "solve",
    matrix: list[list[float]] | None = None,
    matrix_path: str | None = None,
    matrix_key: str | None = None,
    rhs: list[float] | None = None,
    exact: bool = False,
) -> str:
    try:
        if op not in ("solve", "analyze", "eig", "diagonalize", "inverse"):
            return ToolError(
                error=f"unknown op '{op}'",
                hint="op must be solve|analyze|eig|diagonalize|inverse",
            ).model_dump_json()
        try:
            a, source = _load_matrix_input(matrix, matrix_path, matrix_key)
        except tabular.TabularError as exc:
            return _error(exc)  # TabularError messages are hint-ready verbatim
        except ValueError as exc:
            return ToolError(error=str(exc), hint=_MATRIX_HINT).model_dump_json()

        analysis = symmath.analyze_matrix(a)
        report: dict[str, object] = {
            "op": op,
            "shape": [int(a.shape[0]), int(a.shape[1])],
            "source": source,
            "det": analysis.det,
            "rank": analysis.rank,
            "cond_2": analysis.cond_2,
            "trace": analysis.trace,
            "symmetric": analysis.symmetric,
            "orthogonal": analysis.orthogonal,
        }
        n = a.shape[0]
        square = a.shape[0] == a.shape[1]
        stem = _md5_8(a.tobytes(), op, repr(rhs))
        npz_path = _series_dir(ctx) / f"linalg_{op}_{stem}.npz"

        if op == "analyze":
            return LinearAlgebraReport(**report).model_dump_json()

        if op == "solve":
            if rhs is None:
                return ToolError(
                    error="op='solve' needs rhs=",
                    hint=f"pass rhs= a list of {a.shape[0]} numbers (one per matrix row)",
                ).model_dump_json()
            sol = symmath.solve_system(a, np.asarray(rhs, dtype=np.float64))
            report["residual_norm"] = sol.residual_norm
            if exact and square and n <= 8:
                try:
                    x_exact = symmath.cramer_exact(a, np.asarray(rhs, dtype=np.float64))
                    report["cramer_verified"] = bool(np.allclose(sol.x, x_exact, rtol=1e-8, atol=1e-10))
                except ValueError:
                    report["cramer_verified"] = None
            if sol.x.size <= MAX_INLINE_SOLUTION:
                report["solution"] = [float(v) for v in sol.x]
            else:  # critique #12: arrays stay on disk
                np.savez(npz_path, x=sol.x)
                report["arrays_path"] = str(npz_path)
            return LinearAlgebraReport(**report).model_dump_json()

        if not square:
            return ToolError(
                error=f"op='{op}' needs a square matrix, got shape {a.shape[0]}x{a.shape[1]}",
                hint="eig/diagonalize/inverse are defined for square matrices only",
            ).model_dump_json()

        if op == "inverse":
            try:
                inv, err = symmath.matrix_inverse(a)
            except np.linalg.LinAlgError as exc:
                return ToolError(
                    error=f"matrix is singular — {exc}",
                    hint="det is ~0; use op='analyze' to inspect rank and conditioning",
                ).model_dump_json()
            np.savez(npz_path, inv=inv)
            report["recon_max_abs_err"] = err
            report["arrays_path"] = str(npz_path)
            return LinearAlgebraReport(**report).model_dump_json()

        eig = symmath.eig_decomposition(a)
        report["diagonalizable"] = eig.diagonalizable
        report["orthogonally_diagonalizable"] = eig.orthogonally_diagonalizable
        if n <= MAX_INLINE_EIGENVALUES:
            report["eigenvalues"] = [[float(v.real), float(v.imag)] for v in eig.values]
        if exact and n <= 8:
            try:
                report["exact_eigenvalues"] = symmath.exact_eigenvalues(a)
            except ValueError:
                pass

        if op == "eig":
            np.savez(npz_path, values=eig.values, vectors=eig.vectors)
            report["arrays_path"] = str(npz_path)
            return LinearAlgebraReport(**report).model_dump_json()

        # op == "diagonalize" — a defective matrix is a REPORT, not a ToolError
        if eig.diagonalizable:
            p, d, err = symmath.diagonalize(a)
            np.savez(npz_path, p=p, d=d)
            report["recon_max_abs_err"] = err
            report["arrays_path"] = str(npz_path)
        return LinearAlgebraReport(**report).model_dump_json()
    except Exception as exc:
        return _error(exc)


# --------------------------------------------------------------------------- #
# Tool 4: residues_and_integrals


def _residues_and_integrals(
    ctx: AppContext,
    expression: str,
    var: str = "x",
    mode: str = "residues",
) -> str:
    try:
        _ = ctx  # pure symbolic tool — uniform signature
        if mode not in ("residues", "real_integral", "partial_fractions"):
            return ToolError(
                error=f"unknown mode '{mode}'",
                hint="mode must be residues|real_integral|partial_fractions",
            ).model_dump_json()
        x = sp.Symbol(var)
        expr = symmath.parse_expr_safe(expression, symbols=(var,))

        if mode == "residues":
            poles = symmath.rational_pole_data(expr, x)
            zeros = [
                sp.sstr(z) if m == 1 else f"{sp.sstr(z)} (multiplicity {m})"
                for z, m in symmath.rational_zero_data(expr, x)
            ]
            return ResidueReport(
                mode=mode,
                expression=sp.sstr(expr),
                poles=[_pole_schema(p) for p in poles],
                zeros=zeros,
            ).model_dump_json()

        if mode == "real_integral":
            data = symmath.real_line_integral_by_residues(expr, x)
            latex, plain = symmath.latex_and_str(data.value)
            value_numeric = float(sp.re(data.value).evalf())
            quad_value = quad_abs_err = None
            checked = symmath.quad_crosscheck(expr, x)
            if checked is not None:
                quad_value = checked[0]
                quad_abs_err = abs(checked[0] - value_numeric)
            return ResidueReport(
                mode=mode,
                expression=sp.sstr(expr),
                poles=[_pole_schema(p) for p in data.poles_used],
                value_str=plain,
                value_latex=latex,
                value_numeric=value_numeric,
                quad_value=quad_value,
                quad_abs_err=quad_abs_err,
                method=data.method,
            ).model_dump_json()

        # partial_fractions
        decomposed = symmath.partial_fractions(expr, x)
        latex, plain = symmath.latex_and_str(decomposed)
        return ResidueReport(
            mode=mode,
            expression=sp.sstr(expr),
            result_str=plain,
            value_latex=latex,
            method="apart",
        ).model_dump_json()
    except Exception as exc:
        return _error(exc)


# --------------------------------------------------------------------------- #
# Tool 5: fourier_series


def _fourier_series(
    ctx: AppContext,
    expression: str | None = None,
    series_path: str | None = None,
    column: str | None = None,
    key: str | None = None,
    values: list[float] | None = None,
    interval: list[str] | None = None,
    period: float | None = None,
    var: str = "x",
    n_terms: int = 8,
    save_plot: bool = False,
) -> str:
    try:
        sources = sum(v is not None for v in (expression, series_path, values))
        if sources != 1:
            return ToolError(
                error=f"got {sources} signal sources — exactly one is required", hint=_FOURIER_HINT
            ).model_dump_json()
        if not 1 <= n_terms <= MAX_N_TERMS:
            return ToolError(
                error=f"n_terms must be in 1..{MAX_N_TERMS}, got {n_terms}",
                hint="coefficient lists are inline in the JSON — the count is capped",
            ).model_dump_json()

        if expression is not None:
            return _fourier_symbolic(ctx, expression, interval, var, n_terms, save_plot)

        if values is not None:
            if len(values) > MAX_INLINE_VALUES:
                return ToolError(
                    error=f"inline values has {len(values)} samples — limit {MAX_INLINE_VALUES}",
                    hint="write longer signals to a file and pass series_path=",
                ).model_dump_json()
            y = np.asarray(values, dtype=np.float64)
            source_bits = y.tobytes()
        else:
            loaded = tabular.load_series(series_path, column=column, key=key)
            y = loaded.y
            if period is None and loaded.dt is not None:
                period = float(loaded.dt) * y.size
            source_bits = y.tobytes()
        return _fourier_numeric(ctx, y, period, n_terms, save_plot, source_bits)
    except (symmath.ExpressionError, tabular.TabularError) as exc:
        return _error(exc)  # whitelist hint / verbatim loader message respectively
    except Exception as exc:
        return _error(exc, hint=_FOURIER_HINT if isinstance(exc, ValueError) else None)


def _fourier_symbolic(
    ctx: AppContext,
    expression: str,
    interval: list[str] | None,
    var: str,
    n_terms: int,
    save_plot: bool,
) -> str:
    x = sp.Symbol(var)
    expr = symmath.parse_expr_safe(expression, symbols=(var,))
    interval = interval if interval is not None else ["-pi", "pi"]
    if len(interval) != 2:
        return ToolError(
            error=f"interval must be [lo, hi], got {interval}",
            hint="interval bounds are expression strings so ['-pi','pi'] stays exact",
        ).model_dump_json()
    lo = symmath.parse_expr_safe(str(interval[0]))
    hi = symmath.parse_expr_safe(str(interval[1]))
    if not (lo.is_number and hi.is_number):
        return ToolError(
            error=f"interval bounds must be numeric expressions, got {interval}",
            hint="e.g. ['-pi','pi'], ['0','2*pi'], ['-1','1']",
        ).model_dump_json()

    sf = symmath.fourier_series_symbolic(expr, x, lo, hi, n_terms)
    a0_f = _to_float(sf.a0)
    an_f = np.array([_to_float(v) for v in sf.an])
    bn_f = np.array([_to_float(v) for v in sf.bn])
    dom_n, dom_amp = _dominant_harmonic(an_f, bn_f)
    partial_latex, partial_str = symmath.latex_and_str(sf.partial_sum)
    period_f = float((hi - lo).evalf())

    samples_path = plot_path = None
    recon_rms = parseval = None
    stem = _md5_8(expression, str(interval), str(n_terms))
    try:
        n_grid = 512
        t = np.linspace(float(lo.evalf()), float(hi.evalf()), n_grid, endpoint=False)
        fn = sp.lambdify(x, expr, modules=["numpy"])
        fs = sp.lambdify(x, sf.partial_sum, modules=["numpy"])
        y = np.broadcast_to(np.real(np.asarray(fn(t), dtype=np.complex128)), t.shape).astype(np.float64)
        y_recon = np.broadcast_to(np.real(np.asarray(fs(t), dtype=np.complex128)), t.shape).astype(np.float64)
        recon_rms = float(np.sqrt(np.mean((y - y_recon) ** 2)))
        power = float(np.mean(y**2))
        if power > 0.0:
            parseval = float((a0_f**2 + 0.5 * float(np.sum(an_f**2 + bn_f**2))) / power)
        out = _series_dir(ctx) / f"fourier_{stem}.npz"
        np.savez(out, t=t, y=y, y_recon=y_recon)
        samples_path = str(out)
        if save_plot:
            plot_path = _fourier_plot(ctx, t, y, y_recon, n_terms, stem, f"fourier_series: {expression}")
    except Exception:
        pass  # non-lambdifiable expressions still get their symbolic coefficients

    return FourierSeriesReport(
        mode="symbolic",
        n_terms=n_terms,
        interval=[str(interval[0]), str(interval[1])],
        period=period_f,
        a0=a0_f,
        an=[float(v) for v in an_f],
        bn=[float(v) for v in bn_f],
        a0_str=sp.sstr(sf.a0),
        an_str=[sp.sstr(v) for v in sf.an],
        bn_str=[sp.sstr(v) for v in sf.bn],
        partial_sum_str=partial_str,
        partial_sum_latex=partial_latex,
        dominant_harmonic_n=dom_n,
        dominant_harmonic_amp=dom_amp,
        recon_rms_err=recon_rms,
        parseval_fraction=parseval,
        samples_path=samples_path,
        plot_path=plot_path,
    ).model_dump_json()


def _fourier_numeric(
    ctx: AppContext,
    y: np.ndarray,
    period: float | None,
    n_terms: int,
    save_plot: bool,
    source_bits: bytes,
) -> str:
    a0, an, bn = symmath.fourier_coefficients(y, n_terms)
    y_recon = symmath.fourier_reconstruction(a0, an, bn, y.size)
    recon_rms = float(np.sqrt(np.mean((y - y_recon) ** 2)))
    power = float(np.mean(y**2))
    parseval = None
    if power > 0.0:
        parseval = float((a0**2 + 0.5 * float(np.sum(an**2 + bn**2))) / power)
    dom_n, dom_amp = _dominant_harmonic(an, bn)

    dt_lab = (period / y.size) if period else 1.0 / y.size
    t = np.arange(y.size, dtype=np.float64) * dt_lab
    stem = _md5_8(source_bits, str(n_terms))
    out = _series_dir(ctx) / f"fourier_{stem}.npz"
    np.savez(out, t=t, y=y, y_recon=y_recon)
    plot_path = None
    if save_plot:
        plot_path = _fourier_plot(ctx, t, y, y_recon, n_terms, stem, "fourier_series (numeric)")

    return FourierSeriesReport(
        mode="numeric",
        n_terms=n_terms,
        period=period,
        a0=a0,
        an=[float(v) for v in an],
        bn=[float(v) for v in bn],
        dominant_harmonic_n=dom_n,
        dominant_harmonic_amp=dom_amp,
        recon_rms_err=recon_rms,
        parseval_fraction=parseval,
        samples_path=str(out),
        plot_path=plot_path,
    ).model_dump_json()


def _fourier_plot(
    ctx: AppContext,
    t: np.ndarray,
    y: np.ndarray,
    y_recon: np.ndarray,
    n_terms: int,
    stem: str,
    title: str,
) -> str:
    dt = float(t[1] - t[0]) if t.size > 1 else 1.0
    sig = Signal1D(y=y, dt=dt, t0=float(t[0]))
    sig_r = Signal1D(y=y_recon, dt=dt, t0=float(t[0]))
    freqs, psd = transforms.spectrum(sig)
    freqs_r, psd_r = transforms.spectrum(sig_r)
    return str(
        plots.save_comparison_plot(
            sig,
            "signal",
            freqs,
            psd,
            sig_r,
            f"partial sum (n={n_terms})",
            freqs_r,
            psd_r,
            ctx.plots_dir / f"fourier_{stem}.png",
            title,
        )
    )


# --------------------------------------------------------------------------- #
# Registration


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the five engmath tools on ``mcp``, bound to ``ctx``."""

    @mcp.tool()
    def solve_ode(
        equation: str,
        func: str = "y",
        var: str = "x",
        ics: dict[str, float] | None = None,
        mode: str = "auto",
        x_end: float = 10.0,
        n_samples: int = 400,
        save_plot: bool = False,
    ) -> str:
        """Solve one ordinary differential equation given in prime notation
        ("y'' + 4*y' + 3*y = exp(-x)") or explicit Derivative form. ics keys look like
        "y(0)", "y'(0)". mode='symbolic' (sympy dsolve, ToolError if it fails), 'numeric'
        (solve_ivp RK45 with Radau fallback on [x0, x_end]; requires a full ics set), or
        'auto' (symbolic first, numeric fallback; when both succeed they are cross-checked
        and mode_used='both'). Returns OdeReport JSON: order, dsolve classification hints,
        solution as plain + LaTeX strings, remaining free constants when ics underdetermine,
        the max |symbolic - numeric| cross-check error, and — whenever a trajectory was
        integrated — its npz path under data/series/ (keys x, y, y1..) plus an optional plot.
        Numeric mode is the documented escape hatch for symbolically hard equations."""
        return _solve_ode(ctx, equation, func, var, ics, mode, x_end, n_samples, save_plot)

    @mcp.tool()
    def laplace_transform(
        expression: str,
        direction: str = "forward",
        expression_b: str | None = None,
        var_t: str = "t",
        var_s: str = "s",
    ) -> str:
        """Laplace-transform toolkit. direction='forward' computes L{f(t)} with its
        convergence condition ("Re(s) > a"); 'inverse' computes the causal inverse of F(s)
        (implied Heaviside(t) stripped) and, for rational F, lists every pole with order and
        residue (the inverse-by-partial-fractions view); 'convolve' takes two time-domain
        expressions, returns (f*g)(t) via the convolution integral and checks the convolution
        theorem — theorem_verified is true on an exact symbolic zero, null when sympy cannot
        settle it (unrefuted), false only when numeric sampling refutes it. Returns
        LaplaceReport JSON with plain + LaTeX result strings. Expression strings go through
        the restricted SymPy whitelist (never Python eval)."""
        return _laplace_transform(ctx, expression, direction, expression_b, var_t, var_s)

    @mcp.tool()
    def linear_algebra(
        op: str = "solve",
        matrix: list[list[float]] | None = None,
        matrix_path: str | None = None,
        matrix_key: str | None = None,
        rhs: list[float] | None = None,
        exact: bool = False,
    ) -> str:
        """One matrix tool, five ops, over an inline matrix (JSON rows) OR a file
        (matrix_path=.csv/.tsv/.npz/.json — .json may be a list of equal-length numeric
        lists; matrix_key selects an .npz member). op='solve' (LU for square, least-squares
        otherwise, residual norm; exact=true on n<=8 adds a sympy-Rational Cramer
        cross-check), 'analyze' (det, rank, 2-norm condition, trace, symmetric/orthogonal
        flags), 'eig', 'diagonalize' (P, D saved to npz; defective matrices report
        diagonalizable=false instead of erroring), 'inverse'. Returns LinearAlgebraReport
        JSON; solutions inline only up to 64 entries and eigenvalues (as [re, im] pairs) up
        to n=12 — anything larger lives in data/series/*.npz behind arrays_path."""
        return _linear_algebra(ctx, op, matrix, matrix_path, matrix_key, rhs, exact)

    @mcp.tool()
    def residues_and_integrals(
        expression: str,
        var: str = "x",
        mode: str = "residues",
    ) -> str:
        """Complex-analysis workhorse for rational functions of one variable.
        mode='residues': every pole with location, order and residue, plus the zeros.
        mode='real_integral': evaluate the integral over the whole real line by summing
        upper-half-plane residues — validity is gated (rational, or rational times one
        exp(I*a*x)/cos(a*x)/sin(a*x) factor; no real-axis poles; sufficient decay) and a
        violated rule is named in the ToolError; the closed form is always cross-checked
        against scipy.integrate.quad (quad_value, quad_abs_err). mode='partial_fractions':
        apart() decomposition — the inverse-Laplace workhorse. Returns ResidueReport JSON
        with plain + LaTeX value strings."""
        return _residues_and_integrals(ctx, expression, var, mode)

    @mcp.tool()
    def fourier_series(
        expression: str | None = None,
        series_path: str | None = None,
        column: str | None = None,
        key: str | None = None,
        values: list[float] | None = None,
        interval: list[str] | None = None,
        period: float | None = None,
        var: str = "x",
        n_terms: int = 8,
        save_plot: bool = False,
    ) -> str:
        """Trig Fourier series from exactly ONE source. Symbolic: expression= on
        interval= (bound strings, default ['-pi','pi'] so pi stays exact) -> exact a0/an/bn
        plus the partial sum (plain + LaTeX). Numeric: a sampled signal assumed to span
        exactly one period (series_path= any tabular/npz/ply series with column=/key=, or
        values= inline, <=4096 samples; period= only labels the time axis) -> FFT-projected
        coefficients, reconstruction RMS error and the Parseval power fraction captured by
        n_terms. Reconstruction goes to data/series/fourier_*.npz (keys t, y, y_recon);
        save_plot overlays signal vs partial sum. For an aliasing verdict use sampling_check
        (systems pack); for a mesh corrugation verdict use analyze_corrugation."""
        return _fourier_series(
            ctx,
            expression,
            series_path,
            column,
            key,
            values,
            interval,
            period,
            var,
            n_terms,
            save_plot,
        )

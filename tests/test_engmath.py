"""engmath pack tests: module-level ``_impl`` calls against a tmp AppContext —
no MCP client, no engine binary. Golden numbers follow the design as corrected by
the critique (residues, square-wave Fourier coefficients), plus the hostile
expression suite: every attack string must come back as fast ToolError JSON."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from dsp_server.engine import symmath
from dsp_server.toolsets import AppContext, engmath

X = sp.Symbol("x")
T = sp.Symbol("t")
S = sp.Symbol("s")


@pytest.fixture
def ctx(tmp_path: Path) -> AppContext:
    data = tmp_path / "data"
    return AppContext(
        data_dir=data,
        dumps_dir=data / "dumps",
        cache_dir=data / "cache",
        plots_dir=data / "plots",
        logs_dir=data / "logs",
        engine_root=tmp_path / "engine",
    )


# --------------------------------------------------------------------------- #
# solve_ode


def test_solve_ode_symbolic_harmonic(ctx):
    r = json.loads(engmath._solve_ode(ctx, "y'' + y = 0", ics={"y(0)": 0, "y'(0)": 1}))
    assert "error" not in r
    assert r["order"] == 2
    assert r["mode_used"] == "both"
    sol = sp.sympify(r["solution_str"])
    assert sp.simplify(sol - sp.sin(X)) == 0
    assert r["solution_latex"]
    assert r["free_constants"] == []
    assert r["crosscheck_max_abs_err"] < 1e-6
    assert r["samples_path"] and Path(r["samples_path"]).is_file()


def test_solve_ode_first_order(ctx):
    r = json.loads(engmath._solve_ode(ctx, "y' = -2*y", ics={"y(0)": 3}))
    assert "error" not in r
    sol = sp.sympify(r["solution_str"])
    assert sp.simplify(sol - 3 * sp.exp(-2 * X)) == 0
    assert r["mode_used"] == "both"
    assert r["crosscheck_max_abs_err"] < 1e-6


def test_solve_ode_undetermined_constants_and_particular_term(ctx):
    r = json.loads(engmath._solve_ode(ctx, "y'' + 3*y' + 2*y = exp(-3*x)"))
    assert "error" not in r
    assert r["mode_used"] == "symbolic"  # no full ics -> no numeric lane
    assert set(r["free_constants"]) == {"C1", "C2"}
    assert r["samples_path"] is None
    sol = sp.sympify(r["solution_str"])
    particular = sol.subs({sp.Symbol(c): 0 for c in r["free_constants"]})
    assert sp.simplify(particular - sp.exp(-3 * X) / 2) == 0


def test_solve_ode_numeric_only(ctx):
    r = json.loads(engmath._solve_ode(ctx, "y' = -2*y", ics={"y(0)": 3}, mode="numeric"))
    assert "error" not in r
    assert r["mode_used"] == "numeric"
    assert r["solution_str"] is None and r["solution_latex"] is None
    assert r["method"] in ("RK45", "Radau")
    data = np.load(r["samples_path"])
    assert set(data.files) == {"x", "y"}
    err = float(np.max(np.abs(data["y"] - 3.0 * np.exp(-2.0 * data["x"]))))
    assert err < 1e-6


def test_solve_ode_numeric_requires_full_ics(ctx):
    r = json.loads(engmath._solve_ode(ctx, "y'' + y = 0", ics={"y(0)": 1}, mode="numeric"))
    assert "error" in r
    assert "initial conditions" in r["error"]


# --------------------------------------------------------------------------- #
# laplace_transform


def test_laplace_forward(ctx):
    r = json.loads(engmath._laplace_transform(ctx, "t*exp(-2*t)"))
    assert "error" not in r
    res = sp.sympify(r["result_str"])
    assert sp.simplify(res - 1 / (S + 2) ** 2) == 0
    assert "-2" in r["convergence_condition"]
    assert r["result_latex"]


def test_laplace_inverse_with_poles(ctx):
    r = json.loads(engmath._laplace_transform(ctx, "1/(s**2+1)", direction="inverse"))
    assert "error" not in r
    res = sp.sympify(r["result_str"])
    assert sp.simplify(res - sp.sin(T)) == 0
    assert r["poles"] is not None and len(r["poles"]) == 2
    assert sorted(p["im"] for p in r["poles"]) == [-1.0, 1.0]
    assert all(p["order"] == 1 for p in r["poles"])


def test_laplace_convolution_theorem(ctx):
    r = json.loads(engmath._laplace_transform(ctx, "exp(-t)", direction="convolve", expression_b="1"))
    assert "error" not in r
    res = sp.sympify(r["result_str"])
    assert sp.simplify(res - (1 - sp.exp(-T))) == 0
    assert r["theorem_verified"] is True
    assert r["input_b_str"] == "1"


# --------------------------------------------------------------------------- #
# linear_algebra


def test_linear_algebra_solve_exact(ctx):
    r = json.loads(engmath._linear_algebra(ctx, op="solve", matrix=[[2, 1], [1, 3]], rhs=[5, 10], exact=True))
    assert "error" not in r
    assert np.allclose(r["solution"], [1.0, 3.0], atol=1e-9)
    assert r["residual_norm"] < 1e-12
    assert r["cramer_verified"] is True
    assert r["shape"] == [2, 2]


def test_linear_algebra_matrix_path_json(ctx, tmp_path):
    p = tmp_path / "m.json"
    p.write_text("[[2, 1], [1, 3]]", encoding="utf-8")
    r = json.loads(engmath._linear_algebra(ctx, op="solve", matrix_path=str(p), rhs=[5, 10]))
    assert "error" not in r
    assert np.allclose(r["solution"], [1.0, 3.0], atol=1e-9)
    assert r["source"] == str(p)


def test_linear_algebra_eig(ctx):
    r = json.loads(engmath._linear_algebra(ctx, op="eig", matrix=[[4, 1], [2, 3]], exact=True))
    assert "error" not in r
    vals = sorted(v[0] for v in r["eigenvalues"])
    assert vals == pytest.approx([2.0, 5.0], abs=1e-9)
    assert all(abs(v[1]) < 1e-12 for v in r["eigenvalues"])
    assert r["diagonalizable"] is True
    assert set(r["exact_eigenvalues"]) == {"2", "5"}
    data = np.load(r["arrays_path"])
    assert {"values", "vectors"} <= set(data.files)


def test_linear_algebra_analyze_rotation_is_orthogonal(ctx):
    th = np.deg2rad(30.0)
    m = [[float(np.cos(th)), float(-np.sin(th))], [float(np.sin(th)), float(np.cos(th))]]
    r = json.loads(engmath._linear_algebra(ctx, op="analyze", matrix=m))
    assert "error" not in r
    assert r["orthogonal"] is True
    assert r["det"] == pytest.approx(1.0, abs=1e-9)
    assert r["cond_2"] == pytest.approx(1.0, abs=1e-6)


def test_linear_algebra_defective_is_report_not_error(ctx):
    r = json.loads(engmath._linear_algebra(ctx, op="diagonalize", matrix=[[1, 1], [0, 1]]))
    assert "error" not in r
    assert r["diagonalizable"] is False
    assert r["recon_max_abs_err"] is None
    assert r["arrays_path"] is None


def test_linear_algebra_large_solution_goes_to_npz(ctx):
    rng = np.random.default_rng(7)
    n = 70  # > MAX_INLINE_SOLUTION (critique #12)
    a = rng.normal(size=(n, n)) + n * np.eye(n)
    b = rng.normal(size=n)
    r = json.loads(engmath._linear_algebra(ctx, op="solve", matrix=a.tolist(), rhs=b.tolist()))
    assert "error" not in r
    assert r["solution"] is None
    assert r["arrays_path"] and Path(r["arrays_path"]).is_file()
    x = np.load(r["arrays_path"])["x"]
    assert np.allclose(a @ x, b, atol=1e-8)


def test_linear_algebra_input_validation(ctx):
    r = json.loads(engmath._linear_algebra(ctx, op="solve", rhs=[1.0]))
    assert "error" in r and "matrix" in r["hint"]
    r2 = json.loads(engmath._linear_algebra(ctx, op="solve", matrix_path="nope.parquet", rhs=[1]))
    assert "error" in r2
    assert "unsupported table format" in r2["hint"]  # TabularError message verbatim


# --------------------------------------------------------------------------- #
# residues_and_integrals


def test_residues_of_lorentzian(ctx):
    r = json.loads(engmath._residues_and_integrals(ctx, "1/(x**2+1)", mode="residues"))
    assert "error" not in r
    poles = {p["location_str"]: p for p in r["poles"]}
    assert set(poles) == {"I", "-I"}
    top = poles["I"]
    assert top["order"] == 1
    assert top["residue_re"] == pytest.approx(0.0, abs=1e-12)
    assert top["residue_im"] == pytest.approx(-0.5, abs=1e-12)  # residue at +I is -I/2
    assert r["zeros"] == []


def test_real_integral_lorentzian_is_pi(ctx):
    r = json.loads(engmath._residues_and_integrals(ctx, "1/(x**2+1)", mode="real_integral"))
    assert "error" not in r
    assert r["value_numeric"] == pytest.approx(np.pi, rel=1e-4)
    assert r["quad_abs_err"] < 1e-8
    assert r["method"] == "residue-uhp"
    assert sp.simplify(sp.sympify(r["value_str"]) - sp.pi) == 0


def test_real_integral_squared_lorentzian_is_half_pi(ctx):
    r = json.loads(engmath._residues_and_integrals(ctx, "1/(x**2+1)**2", mode="real_integral"))
    assert "error" not in r
    assert r["value_numeric"] == pytest.approx(np.pi / 2.0, rel=1e-4)
    assert r["quad_abs_err"] < 1e-8


def test_real_integral_rejects_real_pole(ctx):
    r = json.loads(engmath._residues_and_integrals(ctx, "1/(x-1)", mode="real_integral"))
    assert "error" in r
    assert "real root" in r["error"]  # the violated rule is named


def test_partial_fractions(ctx):
    r = json.loads(engmath._residues_and_integrals(ctx, "1/(s**2+3*s+2)", var="s", mode="partial_fractions"))
    assert "error" not in r
    assert r["method"] == "apart"
    res = sp.sympify(r["result_str"])
    assert sp.simplify(res - (1 / (S + 1) - 1 / (S + 2))) == 0


# --------------------------------------------------------------------------- #
# fourier_series


def test_fourier_symbolic_sawtooth(ctx):
    r = json.loads(engmath._fourier_series(ctx, expression="x", n_terms=3))
    assert "error" not in r
    assert r["mode"] == "symbolic"
    assert r["interval"] == ["-pi", "pi"]
    assert r["bn"] == pytest.approx([2.0, -1.0, 2.0 / 3.0], rel=1e-5)
    assert max(abs(v) for v in r["an"]) < 1e-12
    assert abs(r["a0"]) < 1e-12
    assert r["bn_str"] == ["2", "-1", "2/3"]
    assert r["dominant_harmonic_n"] == 1
    assert r["partial_sum_str"] and r["partial_sum_latex"]


def test_fourier_numeric_square_wave(ctx):
    n = 1024
    y = np.ones(n)
    y[n // 2 :] = -1.0
    r8 = json.loads(engmath._fourier_series(ctx, values=y.tolist(), n_terms=8))
    assert "error" not in r8
    assert r8["mode"] == "numeric"
    bn = r8["bn"]
    assert bn[0] == pytest.approx(4.0 / np.pi, rel=1e-3)  # 1.27324
    assert abs(bn[1]) < 1e-6
    assert bn[2] == pytest.approx(0.424413, rel=1e-3)
    assert r8["parseval_fraction"] > 0.9
    assert r8["dominant_harmonic_n"] == 1
    data = np.load(r8["samples_path"])
    assert set(data.files) == {"t", "y", "y_recon"}

    r1 = json.loads(engmath._fourier_series(ctx, values=y.tolist(), n_terms=1))
    assert r8["recon_rms_err"] < r1["recon_rms_err"]


def test_fourier_series_file_carries_dt(ctx, tmp_path):
    n = 256
    y = np.sin(2.0 * np.pi * np.arange(n) / n)
    p = tmp_path / "sig.npz"
    np.savez(p, y=y, dt=np.float64(0.001))
    r = json.loads(engmath._fourier_series(ctx, series_path=str(p), n_terms=3))
    assert "error" not in r
    assert r["period"] == pytest.approx(0.256, rel=1e-9)
    assert r["bn"][0] == pytest.approx(1.0, rel=1e-6)
    assert r["dominant_harmonic_n"] == 1


def test_fourier_series_requires_exactly_one_source(ctx):
    r = json.loads(engmath._fourier_series(ctx))
    assert "error" in r
    assert "exactly one" in r["error"]


# --------------------------------------------------------------------------- #
# Security: hostile expressions must fail fast as ToolError, never hang

HOSTILE = [
    "9**9**9**9",  # power tower (would OOM at parse-eval time)
    "gamma(10**7)",  # exact-gamma blowup (guarded wrapper)
    "__import__('os')",  # quotes + dunder
    "x" * 900,  # length gate
    "().__class__",  # attribute access / dunder
    "lambda x: x",  # statement-ish token
    "open('f')",  # quotes; open is not whitelisted anyway
    "9999999 + 1",  # bare integer literal > 1e6
]


@pytest.mark.parametrize("expr", HOSTILE)
def test_hostile_expressions_fail_fast(ctx, expr):
    t0 = time.monotonic()
    r = json.loads(engmath._residues_and_integrals(ctx, expr))
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"gate took {elapsed:.2f}s on {expr[:40]!r}"
    assert "error" in r
    assert r["error"].startswith("ExpressionError:")
    assert "whitelist" in (r["hint"] or "")


def test_hostile_expressions_gated_in_ode_lane_too(ctx):
    t0 = time.monotonic()
    r = json.loads(engmath._solve_ode(ctx, "y' = __import__('os')"))
    assert time.monotonic() - t0 < 2.0
    assert "error" in r and r["error"].startswith("ExpressionError:")


def test_parse_gates_kernel_level():
    with pytest.raises(symmath.ExpressionError):
        symmath.parse_expr_safe("x**70")  # exponent > 64
    with pytest.raises(symmath.ExpressionError):
        symmath.parse_expr_safe("2**(3**4)")  # parenthesized tower
    with pytest.raises(symmath.ExpressionError):
        symmath.parse_expr_safe("sin(x).diff(x)")  # attribute access
    assert symmath.parse_expr_safe("x**64") is not None  # boundary passes
    expr = symmath.parse_expr_safe("0.5*x + 2^3")  # decimals + caret alias
    assert sp.simplify(expr - (sp.Rational(1, 2) * X + 8)) == 0


def test_unknown_name_parses_harmlessly(ctx):
    expr = symmath.parse_expr_safe("foo(3) + x", symbols=("x",))
    assert expr.has(sp.Function("foo")(3))
    # ...and at the tool boundary it is a report or a ToolError, never a raise.
    out = json.loads(engmath._residues_and_integrals(ctx, "foo(3)"))
    assert isinstance(out, dict)

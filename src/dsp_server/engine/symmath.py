"""Symbolic + numeric math kernels for the engmath pack (MAT235/MAT236).

Two halves, both pure (no I/O, no MCP types):

* a **restricted SymPy parsing pipeline** — every user expression string goes
  through :func:`parse_expr_safe`, which layers textual gates (length, character
  set, token blacklist, integer-literal and power-tower limits) in front of
  ``sympy.parsing.sympy_parser.parse_expr`` with a whitelist ``local_dict`` and a
  minimal constructor-only ``global_dict``. Raw ``eval``/``sympify`` are never
  used. Every rejection raises :class:`ExpressionError` with a hint-ready message.
* ODE / Laplace / residue / Fourier / linear-algebra helpers built on sympy,
  numpy and scipy, consumed by ``toolsets/engmath.py``.

Known limitation (documented, not fixable here): sympy's ``dsolve``/``integrate``
have no interruptible timeout — the mitigation is the 800-char cap, the power and
factorial gates below, and ``mode="numeric"`` as the solve_ode escape hatch.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.integrate import quad, solve_ivp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

MAX_EXPR_LEN = 800
MAX_BARE_INT = 1_000_000
MAX_EXPONENT = 64
MAX_GAMMA_ARG = 170

ALLOWED_FUNCTIONS = (
    "sin cos tan asin acos atan sinh cosh tanh exp log sqrt Abs Heaviside DiracDelta gamma factorial"
)
ALLOWED_CONSTANTS = "pi E I oo"


class ExpressionError(ValueError):
    """Every parse/validation rejection raises this; the message is hint-ready."""


def _guarded_gamma_like(func: Callable[..., sp.Expr], name: str) -> Callable[..., sp.Expr]:
    """gamma/factorial wrapper: numeric arguments with |arg| > 170 are rejected
    BEFORE sympy auto-evaluates them into million-digit integers (DoS gate #8).
    Symbolic arguments pass through untouched."""

    def wrapped(*args: sp.Expr) -> sp.Expr:
        for arg in args:
            a = sp.sympify(arg)
            if a.is_number:
                try:
                    mag = abs(complex(a.evalf()))
                except (TypeError, ValueError) as exc:
                    raise ExpressionError(f"{name} argument {a} is not evaluable") from exc
                if mag > MAX_GAMMA_ARG:
                    raise ExpressionError(
                        f"{name}({a}) rejected — numeric arguments are capped at "
                        f"|arg| <= {MAX_GAMMA_ARG} (exact evaluation would blow up); "
                        "use a float argument below the cap"
                    )
        return func(*args)

    return wrapped


WHITELIST: dict[str, object] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
    "Heaviside": sp.Heaviside,
    "DiracDelta": sp.DiracDelta,
    "gamma": _guarded_gamma_like(sp.gamma, "gamma"),
    "factorial": _guarded_gamma_like(sp.factorial, "factorial"),
    "pi": sp.pi,
    "E": sp.E,
    "I": sp.I,
    "oo": sp.oo,
    "Rational": sp.Rational,
}

# Constructor namespace for sympy's auto_symbol/auto_number token rewrites: the
# generated code references Symbol/Integer/Float/Rational/Function by name, so
# they MUST be resolvable (critique #3 — global_dict={} breaks parsing of "2*x").
# __builtins__ is pinned empty so eval() cannot inject the real builtins module.
_GLOBAL_DICT: dict[str, object] = {
    "Symbol": sp.Symbol,
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Function": sp.Function,
    "__builtins__": {},
}

# The evaluate=False parse (DoS pre-check) emits functional form — Pow/Mul/Add must
# resolve. Harmless expression constructors; __builtins__ stays empty either way.
_GLOBAL_DICT_NOEVAL: dict[str, object] = {**_GLOBAL_DICT, "Add": sp.Add, "Mul": sp.Mul, "Pow": sp.Pow}

_CHAR_RE = re.compile(r"^[0-9A-Za-z_+\-*/^().,\s<>\[\]]*$")
_CHAR_RE_PRIMES = re.compile(r"^[0-9A-Za-z_+\-*/^().,\s<>\[\]'=]*$")
_ATTR_DOT_RE = re.compile(r"\.(?![0-9])")
_BARE_INT_RE = re.compile(r"(?<![\w.])(\d+)(?![\w.])")
_BAD_TOKENS = ("__", "lambda", "import", ";", "\\")


def _gate_powers(text: str) -> None:
    """Reject power towers (``a**b**c``, ``a**(b**c)``) and integer exponents > 64
    (critique #8 — ``9**9**9**9`` passes every other gate and OOMs at eval time)."""
    n = len(text)
    i = text.find("**")
    while i >= 0:
        j = i + 2
        while j < n and text[j] in " \t+-":
            j += 1
        if j < n and text[j] == "(":
            depth, k = 1, j + 1
            while k < n and depth:
                if text[k] == "(":
                    depth += 1
                elif text[k] == ")":
                    depth -= 1
                k += 1
            if "**" in text[j:k]:
                raise ExpressionError("nested power tower rejected (a**(b**c) form)")
            end = k
        else:
            k = j
            while k < n and (text[k].isalnum() or text[k] in "._"):
                k += 1
            token = text[j:k]
            digits = token.split(".")[0].split("e")[0].split("E")[0]
            if digits.isdigit() and int(digits) > MAX_EXPONENT:
                raise ExpressionError(f"exponent {digits} exceeds the limit of {MAX_EXPONENT}")
            end = k
        if text[end:].lstrip().startswith("**"):
            raise ExpressionError("chained power tower rejected (a**b**c form)")
        i = text.find("**", i + 2)


def _gate_text(text: str, allow_primes: bool = False) -> str:
    """All pre-parse gates; returns the normalized (``^`` -> ``**``) text."""
    if not isinstance(text, str) or not text.strip():
        raise ExpressionError("expression must be a non-empty string")
    if len(text) > MAX_EXPR_LEN:
        raise ExpressionError(f"expression is {len(text)} chars — limit {MAX_EXPR_LEN}")
    char_re = _CHAR_RE_PRIMES if allow_primes else _CHAR_RE
    if not char_re.match(text):
        bad = sorted({c for c in text if not char_re.match(c)})
        raise ExpressionError(f"disallowed characters {bad} — expressions are math strings, not code")
    lowered = text.lower()
    for tok in _BAD_TOKENS:
        if tok in lowered:
            raise ExpressionError(f"disallowed token {tok!r} — expressions are math strings, not code")
    if _ATTR_DOT_RE.search(text):
        raise ExpressionError("'.' is only allowed inside decimal literals (attribute access rejected)")
    normalized = text.replace("^", "**")
    for m in _BARE_INT_RE.finditer(normalized):
        if int(m.group(1)) > MAX_BARE_INT:
            raise ExpressionError(f"integer literal {m.group(1)} exceeds the limit of {MAX_BARE_INT}")
    _gate_powers(normalized)
    return normalized


def _reject_huge_powers(expr: sp.Basic) -> None:
    """Reject any Pow with a purely-numeric exponent of magnitude > MAX_EXPONENT.

    The textual ``_gate_powers`` bounds bare exponents but a *parenthesized*
    exponent (``2**(900*900*900)``, ``2**(999999*999999)``) slips past it; if it
    reached an evaluating parse the base power would materialize a multi-gigabyte
    bignum and hang the process. This runs on the ``evaluate=False`` parse, so the
    exponent product (bounded literals, no nested ``**``) is cheap to ``doit`` and
    the base is never raised.
    """
    for pw in expr.atoms(sp.Pow):
        exponent = pw.exp
        if exponent.free_symbols:
            continue  # symbolic exponent never auto-evaluates to a bignum
        try:
            value = exponent.doit()
            magnitude = abs(float(value)) if value.is_number else 0.0
        except (TypeError, ValueError, OverflowError):
            continue
        if magnitude > MAX_EXPONENT:
            raise ExpressionError(f"power exponent {value} exceeds the limit of {MAX_EXPONENT}")


def parse_expr_safe(
    text: str,
    symbols: Sequence[str] = (),
    extra_locals: dict[str, object] | None = None,
) -> sp.Expr:
    """Parse one expression through the gates + whitelist. Unknown bare names
    auto-symbolize (``foo(3)`` becomes an applied undefined function — harmless).
    Explicit multiplication only; ``^`` is accepted as a power alias."""
    normalized = _gate_text(text, allow_primes=False)
    local_dict: dict[str, object] = dict(WHITELIST)
    for name in symbols:
        local_dict[name] = sp.Symbol(name)
    if extra_locals:
        local_dict.update(extra_locals)
    parse_kwargs = {
        "local_dict": local_dict,
        "global_dict": _GLOBAL_DICT,
        "transformations": standard_transformations,
    }
    try:
        # First parse WITHOUT evaluation so no bignum base power is materialized,
        # then reject oversized numeric exponents before the real evaluating parse.
        noeval_kwargs = {**parse_kwargs, "global_dict": _GLOBAL_DICT_NOEVAL}
        unevaluated = parse_expr(normalized, evaluate=False, **noeval_kwargs)
        _reject_huge_powers(unevaluated)
        expr = parse_expr(normalized, **parse_kwargs)
    except ExpressionError:
        raise
    except Exception as exc:
        raise ExpressionError(f"could not parse {text!r}: {exc}") from exc
    if not isinstance(expr, sp.Basic):
        raise ExpressionError(f"{text!r} did not parse to a sympy expression")
    return expr


def latex_and_str(expr: sp.Expr) -> tuple[str, str]:
    """(latex, plain) rendering pair — every symbolic output carries both."""
    return sp.latex(expr), sp.sstr(expr)


# --------------------------------------------------------------------------- #
# ODEs


@dataclass
class SymbolicOde:
    expr: sp.Expr
    method: str
    classification: list[str]
    free_constants: list[str]


def parse_ode(equation: str, func: str = "y", var: str = "x") -> tuple[sp.Eq, sp.Function, sp.Symbol, int]:
    """Prime-notation rewrite: ``y'' + 4*y' + 3*y = exp(-x)`` becomes a sympy Eq
    in ``Derivative(y(x), (x, k))``. Returns (eq, y (undefined Function), x, order)."""
    raw = _gate_text(equation, allow_primes=True)
    if raw.count("=") > 1:
        raise ExpressionError("equation must contain at most one '='")
    x = sp.Symbol(var)
    f = sp.Function(func)

    def _prime_repl(m: re.Match[str]) -> str:
        return f"Derivative({func}({var}),({var},{len(m.group(1))}))"

    text = re.sub(rf"\b{re.escape(func)}('+)", _prime_repl, raw)
    text = re.sub(rf"\b{re.escape(func)}\b(?!\()", f"{func}({var})", text)
    extra = {func: f, "Derivative": sp.Derivative}
    if "=" in text:
        lhs_s, rhs_s = text.split("=", 1)
        lhs = parse_expr_safe(lhs_s, symbols=(var,), extra_locals=extra)
        rhs = parse_expr_safe(rhs_s, symbols=(var,), extra_locals=extra)
    else:
        lhs = parse_expr_safe(text, symbols=(var,), extra_locals=extra)
        rhs = sp.S.Zero
    eq = sp.Eq(lhs, rhs)
    try:
        order = int(sp.ode_order(eq, f(x)))
    except Exception as exc:
        raise ExpressionError(f"could not determine ODE order of {equation!r}: {exc}") from exc
    if order < 1:
        raise ExpressionError(f"{equation!r} contains no derivative of {func}({var})")
    return eq, f, x, order


def parse_ics(
    ics: dict[str, float] | None, func: str, var: str, order: int
) -> tuple[dict[sp.Expr, float], list[float] | None, float]:
    """Initial conditions ``{"y(0)": 0, "y'(0)": 1}`` -> (sympy ics dict for
    dsolve, dense [y, y', ...] vector when orders 0..order-1 share one point
    else None, the evaluation point x0)."""
    if not ics:
        return {}, None, 0.0
    x = sp.Symbol(var)
    f = sp.Function(func)
    pat = re.compile(rf"^\s*{re.escape(func)}('*)\((.+)\)\s*$")
    by_order: dict[int, tuple[float, float]] = {}
    sym_ics: dict[sp.Expr, float] = {}
    for key, value in ics.items():
        m = pat.match(str(key))
        if not m:
            raise ExpressionError(f'bad ics key {key!r} — expected forms like "{func}(0)" or "{func}\'(0)"')
        k = len(m.group(1))
        point = parse_expr_safe(m.group(2))
        if not point.is_number:
            raise ExpressionError(f"ics point {m.group(2)!r} must be numeric")
        x0 = float(point.evalf())
        v = float(value)
        target = f(x).subs(x, point) if k == 0 else sp.Derivative(f(x), (x, k)).subs(x, point)
        sym_ics[target] = v
        by_order[k] = (x0, v)
    points = {p for p, _ in by_order.values()}
    complete = set(by_order) == set(range(order)) and len(points) == 1
    x0 = points.pop() if len(points) == 1 else 0.0
    vec = [by_order[k][1] for k in range(order)] if complete else None
    return sym_ics, vec, x0


def solve_ode_symbolic(
    eq: sp.Eq, f: sp.Function, x: sp.Symbol, ics: dict[sp.Expr, float] | None
) -> SymbolicOde:
    """dsolve wrapper: classification hints, the solution rhs, and any free C_k."""
    try:
        classification = list(sp.classify_ode(eq, f(x)))[:3]
    except Exception:
        classification = []
    sol = sp.dsolve(eq, f(x), ics=ics or None)
    if isinstance(sol, list):
        sol = sol[0]
    expr = sol.rhs if isinstance(sol, sp.Eq) else sol
    free = sorted(str(s) for s in expr.free_symbols if str(s).startswith("C") and str(s)[1:].isdigit())
    method = classification[0] if classification else "dsolve"
    return SymbolicOde(expr=expr, method=method, classification=classification, free_constants=free)


def ode_rhs_lambda(
    eq: sp.Eq, f: sp.Function, x: sp.Symbol, order: int
) -> Callable[[float, np.ndarray], np.ndarray]:
    """State-space RHS for solve_ivp: solve for the highest derivative, then
    lambdify over (x, y, y', ..., y^(order-1))."""
    top = sp.Derivative(f(x), (x, order))
    try:
        solved = sp.solve(eq, top)
    except Exception as exc:
        raise ExpressionError(f"cannot solve for the highest derivative: {exc}") from exc
    if not solved:
        raise ExpressionError(
            f"cannot solve the equation for {sp.sstr(top)} — numeric mode needs an "
            "equation explicit in its highest derivative"
        )
    rhs = solved[0]
    states = [sp.Dummy(f"s{k}") for k in range(order)]
    for k in range(order - 1, 0, -1):
        rhs = rhs.subs(sp.Derivative(f(x), (x, k)), states[k])
    rhs = rhs.subs(f(x), states[0])
    extra = rhs.free_symbols - {x, *states}
    if extra:
        raise ExpressionError(
            f"equation contains free parameters {sorted(map(str, extra))} — "
            "numeric mode needs fully numeric coefficients"
        )
    fn = sp.lambdify((x, *states), rhs, modules=["numpy"])

    def rhs_state(t: float, yv: np.ndarray) -> np.ndarray:
        out = np.empty(order, dtype=np.float64)
        out[:-1] = yv[1:]
        out[-1] = fn(t, *yv)
        return out

    return rhs_state


def solve_ode_numeric(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    order: int,
    ics_vec: Sequence[float],
    x0: float,
    x_end: float,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """(x grid, state matrix (order, n), method) — RK45 with a Radau retry for
    stiff systems; tolerances tight enough for a 1e-6 symbolic cross-check."""
    if x_end <= x0:
        raise ExpressionError(f"x_end ({x_end}) must exceed the ics point ({x0})")
    n_samples = int(max(8, min(n_samples, 20_000)))
    t_eval = np.linspace(x0, x_end, n_samples)
    last_message = ""
    for method in ("RK45", "Radau"):
        res = solve_ivp(
            rhs,
            (x0, x_end),
            np.asarray(ics_vec, dtype=np.float64),
            method=method,
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-12,
        )
        if res.success:
            return res.t, res.y, method
        last_message = str(res.message)
    raise ExpressionError(f"numeric integration failed (RK45 then Radau): {last_message}")


# --------------------------------------------------------------------------- #
# Laplace


def laplace_forward(expr: sp.Expr, t: sp.Symbol, s: sp.Symbol) -> tuple[sp.Expr, str | None]:
    """(F(s), convergence condition string like 'Re(s) > -2' or None)."""
    res = sp.laplace_transform(expr, t, s)
    if isinstance(res, tuple):
        f_expr, a, _cond = res
        conv = None if a in (sp.S.NegativeInfinity, None) else f"Re({s}) > {sp.sstr(a)}"
        return f_expr, conv
    return res, None


def laplace_inverse(expr: sp.Expr, s: sp.Symbol, t: sp.Symbol) -> sp.Expr:
    """Inverse transform with the causal Heaviside(t) factor stripped (it is
    implied — all engmath time-domain work is one-sided); Heaviside(t-a) stays."""
    out = sp.inverse_laplace_transform(expr, s, t)
    return sp.simplify(out.subs(sp.Heaviside(t), sp.S.One))


def convolve_expr(f: sp.Expr, g: sp.Expr, t: sp.Symbol) -> sp.Expr:
    """(f * g)(t) via the one-sided convolution integral, evaluated."""
    tau = sp.Dummy("tau", positive=True)
    conv = sp.integrate(f.subs(t, tau) * g.subs(t, t - tau), (tau, 0, t))
    return sp.simplify(conv)


# --------------------------------------------------------------------------- #
# Residues / integrals / partial fractions


@dataclass
class PoleData:
    point: sp.Expr
    order: int
    residue: sp.Expr | None


@dataclass
class IntegralData:
    value: sp.Expr
    poles_used: list[PoleData] = field(default_factory=list)
    method: str = "residue-uhp"


def rational_pole_data(expr: sp.Expr, x: sp.Symbol, with_residues: bool = True) -> list[PoleData]:
    """Poles (with order and residue) of a rational function of ``x``."""
    if not expr.is_rational_function(x):
        raise ExpressionError(
            f"{sp.sstr(expr)} is not a rational function of {x} — pole/residue "
            "extraction supports rational functions only"
        )
    den = sp.fraction(sp.cancel(sp.together(expr)))[1]
    den_poly = sp.Poly(den, x)
    if den_poly.degree() == 0:
        return []
    root_map = sp.roots(den_poly)
    if sum(root_map.values()) < den_poly.degree():
        raise ExpressionError(f"could not find all denominator roots of {sp.sstr(den)} in closed form")
    out: list[PoleData] = []
    for point, mult in sorted(root_map.items(), key=lambda kv: sp.default_sort_key(kv[0])):
        res = None
        if with_residues:
            res = sp.simplify(sp.residue(expr, x, point))
        out.append(PoleData(point=point, order=int(mult), residue=res))
    return out


def rational_zero_data(expr: sp.Expr, x: sp.Symbol) -> list[tuple[sp.Expr, int]]:
    """Zeros (root, multiplicity) of a rational function's numerator."""
    num, _den = sp.fraction(sp.cancel(sp.together(expr)))
    num_poly = sp.Poly(num, x)
    if num_poly.degree() < 1:
        return []
    return sorted(sp.roots(num_poly).items(), key=lambda kv: sp.default_sort_key(kv[0]))


def _split_oscillatory(expr: sp.Expr, x: sp.Symbol) -> tuple[sp.Expr, str | None, sp.Expr | None]:
    """expr = R(x) * osc with osc in {1, exp(I*a*x), cos(a*x), sin(a*x)}."""
    if expr.is_rational_function(x):
        return expr, None, None
    for kind, cls in (("cos", sp.cos), ("sin", sp.sin)):
        atoms = expr.atoms(cls)
        if len(atoms) == 1:
            trig = next(iter(atoms))
            a = sp.simplify(trig.args[0] / x)
            if a.is_number and sp.simplify(trig.args[0] - a * x) == 0:
                rational = sp.simplify(expr / trig)
                if rational.is_rational_function(x):
                    return rational, kind, a
    for e in expr.atoms(sp.exp):
        a = sp.simplify(e.args[0] / (sp.I * x))
        if a.is_number and sp.simplify(e.args[0] - sp.I * a * x) == 0:
            rational = sp.simplify(expr / e)
            if rational.is_rational_function(x):
                return rational, "exp", a
    raise ExpressionError(
        "real_integral supports rational functions optionally multiplied by one "
        "exp(I*a*x), cos(a*x) or sin(a*x) factor — got: " + sp.sstr(expr)
    )


def real_line_integral_by_residues(expr: sp.Expr, x: sp.Symbol) -> IntegralData:
    """Evaluate the principal-value-free integral of ``expr`` over the whole real
    line by summing residues in one half plane. Raises :class:`ExpressionError`
    naming the violated validity rule."""
    rational, kind, a = _split_oscillatory(expr, x)
    num, den = sp.fraction(sp.cancel(sp.together(rational)))
    deg_gap = sp.Poly(den, x).degree() - sp.Poly(num, x).degree()
    pole_list = rational_pole_data(rational, x, with_residues=False)
    for p in pole_list:
        if complex(p.point.evalf()).imag == 0.0:
            raise ExpressionError(
                f"denominator has a real root at {x} = {sp.sstr(p.point)} — the residue "
                "method requires no poles on the real axis"
            )
    if kind is None and deg_gap < 2:
        raise ExpressionError(
            f"decay rule violated: deg(denominator) must exceed deg(numerator) by >= 2 "
            f"for a pure rational integrand (gap here is {deg_gap})"
        )
    if kind is not None and deg_gap < 1:
        raise ExpressionError(
            "Jordan's lemma needs deg(denominator) >= deg(numerator) + 1 for oscillatory integrands"
        )
    a_val = 0.0 if a is None else float(sp.re(a).evalf())
    if kind is not None and a_val == 0.0:
        raise ExpressionError("oscillatory factor must have a nonzero frequency a")
    if kind in ("cos", "sin"):
        # cos/sin are even/odd in a: close the UHP with exp(+i|a|x); the sin sign
        # correction below restores the odd symmetry. abs() is valid ONLY here.
        upper = True
        g = rational * sp.exp(sp.I * abs(a_val) * x)
    elif kind == "exp":
        # A bare complex exponential exp(i*a*x) decays in the UHP iff a > 0, in the
        # LHP iff a < 0 — the closing exponential must keep a's true sign.
        upper = a_val > 0.0
        g = rational * sp.exp(sp.I * a_val * x)
    else:  # pure rational integrand
        upper = True
        g = rational
    used: list[PoleData] = []
    total = sp.S.Zero
    for p in pole_list:
        im = complex(p.point.evalf()).imag
        if (im > 0.0) == upper and im != 0.0:
            res = sp.simplify(sp.residue(g, x, p.point))
            used.append(PoleData(point=p.point, order=p.order, residue=res))
            total += res
    value = 2 * sp.pi * sp.I * total if upper else -2 * sp.pi * sp.I * total
    if kind == "cos":
        value = sp.re(value)
    elif kind == "sin":
        value = sp.im(value) if a_val > 0.0 else -sp.im(value)
    value = sp.simplify(value)
    return IntegralData(value=value, poles_used=used, method="residue-uhp")


def quad_crosscheck(expr: sp.Expr, x: sp.Symbol) -> tuple[float, float] | None:
    """(quad value, quad's own abs-error estimate) over the real line, or None
    when the integrand cannot be evaluated numerically."""
    try:
        integrand = sp.re(expr) if expr.has(sp.I) else expr
        fn = sp.lambdify(x, integrand, modules=["numpy"])
        val, err = quad(lambda v: float(np.real(fn(v))), -np.inf, np.inf, limit=200)
        return float(val), float(err)
    except Exception:
        return None


def partial_fractions(expr: sp.Expr, x: sp.Symbol) -> sp.Expr:
    if not expr.is_rational_function(x):
        raise ExpressionError(f"partial fractions require a rational function of {x} — got {sp.sstr(expr)}")
    return sp.apart(expr, x)


# --------------------------------------------------------------------------- #
# Fourier series


@dataclass
class SymbolicFourier:
    a0: sp.Expr
    an: list[sp.Expr]
    bn: list[sp.Expr]
    partial_sum: sp.Expr


def fourier_series_symbolic(
    expr: sp.Expr, x: sp.Symbol, a: sp.Expr, b: sp.Expr, n_terms: int
) -> SymbolicFourier:
    """Trig Fourier coefficients on [a, b] with the a0-as-mean convention:
    partial sum = a0 + sum an*cos(n*w*x) + bn*sin(n*w*x), w = 2*pi/(b-a)."""
    length = sp.simplify(b - a)
    if not length.is_positive:
        raise ExpressionError(f"interval upper bound must exceed the lower ({sp.sstr(a)} .. {sp.sstr(b)})")
    w = 2 * sp.pi / length
    a0 = sp.simplify(sp.integrate(expr, (x, a, b)) / length)
    an: list[sp.Expr] = []
    bn: list[sp.Expr] = []
    partial = a0
    for n in range(1, n_terms + 1):
        an_n = sp.simplify(2 / length * sp.integrate(expr * sp.cos(n * w * x), (x, a, b)))
        bn_n = sp.simplify(2 / length * sp.integrate(expr * sp.sin(n * w * x), (x, a, b)))
        an.append(an_n)
        bn.append(bn_n)
        partial = partial + an_n * sp.cos(n * w * x) + bn_n * sp.sin(n * w * x)
    return SymbolicFourier(a0=a0, an=an, bn=bn, partial_sum=partial)


def fourier_coefficients(y: np.ndarray, n_terms: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Trig coefficients of a signal assumed to span exactly ONE period, uniform
    sampling, via rfft projection: a0 = mean(y); a_n = 2*Re(X_n)/N; b_n = -2*Im(X_n)/N."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 8:
        raise ExpressionError(f"need >= 8 samples for a Fourier projection, got {n}")
    if n_terms >= n // 2:
        raise ExpressionError(
            f"n_terms ({n_terms}) must be below N/2 ({n // 2}) for {n} samples — "
            "harmonics beyond Nyquist do not exist in the data"
        )
    spec = np.fft.rfft(y)
    a0 = float(y.mean())
    an = 2.0 * np.real(spec[1 : n_terms + 1]) / n
    bn = -2.0 * np.imag(spec[1 : n_terms + 1]) / n
    return a0, an, bn


def fourier_reconstruction(a0: float, an: np.ndarray, bn: np.ndarray, n: int) -> np.ndarray:
    """Partial sum evaluated back on the original sample grid."""
    k = np.arange(n, dtype=np.float64)
    y = np.full(n, a0, dtype=np.float64)
    for i in range(an.size):
        phase = 2.0 * np.pi * (i + 1) * k / n
        y += an[i] * np.cos(phase) + bn[i] * np.sin(phase)
    return y


# --------------------------------------------------------------------------- #
# Linear algebra (numpy first, sympy for the exact n<=8 paths)


@dataclass
class MatrixAnalysis:
    det: float | None
    rank: int
    cond_2: float
    trace: float | None
    symmetric: bool | None
    orthogonal: bool | None


@dataclass
class LinSolve:
    x: np.ndarray
    residual_norm: float
    method: str


@dataclass
class EigResult:
    values: np.ndarray  # complex (n,)
    vectors: np.ndarray  # complex (n, n)
    diagonalizable: bool
    orthogonally_diagonalizable: bool


def _as_matrix(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.size == 0:
        raise ValueError(f"matrix must be 2-D and non-empty, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("matrix contains NaN/inf entries")
    return a


def analyze_matrix(a: np.ndarray) -> MatrixAnalysis:
    a = _as_matrix(a)
    square = a.shape[0] == a.shape[1]
    det = float(np.linalg.det(a)) if square else None
    trace = float(np.trace(a)) if square else None
    symmetric = bool(np.allclose(a, a.T, rtol=1e-9, atol=1e-12)) if square else None
    orthogonal = bool(np.allclose(a @ a.T, np.eye(a.shape[0]), rtol=1e-8, atol=1e-10)) if square else None
    return MatrixAnalysis(
        det=det,
        rank=int(np.linalg.matrix_rank(a)),
        cond_2=float(np.linalg.cond(a, 2)),
        trace=trace,
        symmetric=symmetric,
        orthogonal=orthogonal,
    )


def solve_system(a: np.ndarray, b: np.ndarray) -> LinSolve:
    a = _as_matrix(a)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if b.size != a.shape[0]:
        raise ValueError(f"rhs has {b.size} entries but the matrix has {a.shape[0]} rows")
    if a.shape[0] == a.shape[1]:
        try:
            x = np.linalg.solve(a, b)
            method = "lu"
        except np.linalg.LinAlgError:
            x, *_ = np.linalg.lstsq(a, b, rcond=None)
            method = "lstsq"
    else:
        x, *_ = np.linalg.lstsq(a, b, rcond=None)
        method = "lstsq"
    residual = float(np.linalg.norm(a @ x - b))
    return LinSolve(x=np.asarray(x, dtype=np.float64), residual_norm=residual, method=method)


def _exact_matrix(a: np.ndarray) -> sp.Matrix:
    return sp.Matrix([[sp.Rational(v).limit_denominator(10**12) for v in row] for row in a.tolist()])


def cramer_exact(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cramer's rule over sympy Rationals — verification path, n <= 8 only."""
    a = _as_matrix(a)
    n = a.shape[0]
    if a.shape[0] != a.shape[1] or n > 8:
        raise ValueError(f"cramer_exact needs a square matrix with n <= 8, got {a.shape}")
    am = _exact_matrix(a)
    bm = sp.Matrix([sp.Rational(v).limit_denominator(10**12) for v in np.asarray(b).reshape(-1)])
    det = am.det()
    if det == 0:
        raise ValueError("matrix is exactly singular — Cramer's rule does not apply")
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        mi = am.copy()
        mi[:, i] = bm
        out[i] = float((mi.det() / det).evalf())
    return out


def eig_decomposition(a: np.ndarray) -> EigResult:
    a = _as_matrix(a)
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"eigendecomposition needs a square matrix, got {a.shape}")
    values, vectors = np.linalg.eig(a)
    diagonalizable = bool(np.linalg.matrix_rank(vectors) == a.shape[0])
    symmetric = bool(np.allclose(a, a.T, rtol=1e-9, atol=1e-12))
    return EigResult(
        values=values,
        vectors=vectors,
        diagonalizable=diagonalizable,
        orthogonally_diagonalizable=symmetric,
    )


def diagonalize(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(P, D, max|P D P^-1 - A|); raises ValueError when the matrix is defective."""
    res = eig_decomposition(a)
    if not res.diagonalizable:
        raise ValueError("matrix is defective (eigenvectors do not span) — not diagonalizable")
    p = res.vectors
    d = np.diag(res.values)
    recon = p @ d @ np.linalg.inv(p)
    err = float(np.max(np.abs(recon - np.asarray(a, dtype=np.float64))))
    return p, d, err


def exact_eigenvalues(a: np.ndarray) -> list[str]:
    """Exact (sympy) eigenvalues, one string per root with multiplicity noted."""
    a = _as_matrix(a)
    if a.shape[0] != a.shape[1] or a.shape[0] > 8:
        raise ValueError(f"exact eigenvalues need a square matrix with n <= 8, got {a.shape}")
    vals = _exact_matrix(a).eigenvals()
    out: list[str] = []
    for val, mult in sorted(vals.items(), key=lambda kv: sp.default_sort_key(kv[0])):
        text = sp.sstr(sp.nsimplify(val, rational=False))
        out.append(text if mult == 1 else f"{text} (multiplicity {mult})")
    return out


def matrix_inverse(a: np.ndarray) -> tuple[np.ndarray, float]:
    a = _as_matrix(a)
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"inverse needs a square matrix, got {a.shape}")
    inv = np.linalg.inv(a)
    err = float(np.max(np.abs(a @ inv - np.eye(a.shape[0]))))
    return inv, err

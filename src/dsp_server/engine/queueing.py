"""Closed-form queueing theory for the netqueue pack (ELE412): M/M/1, M/M/m,
M/G/1 via the exact Pollaczek-Khinchine formula, Erlang-B/C, and Little's law.

Pure functions, no I/O, no numpy — scalars in, scalars out. Every failure raises
:class:`ValueError` with a hint-ready message. Instability (rho >= 1) raises
:class:`UnstableQueueError`, whose ``str()`` is EXACTLY
``"unstable queue: rho=<value> >= 1"`` and whose ``remedy`` attribute carries the
fix (minimum per-server service rate / minimum stable server count) — the toolset
splits those into ``ToolError.error`` / ``ToolError.hint``.

Time units are whatever lambda/mu are given in; W/Wq inherit them. All models
assume Poisson arrivals; the service rate is per server.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

QUEUE_MODELS = ("mm1", "mmm", "mg1")
BLOCKING_MODELS = ("b", "c")


class UnstableQueueError(ValueError):
    """rho >= 1: the queue has no steady state.

    ``str(exc)`` is the exact ToolError.error contract string
    (``"unstable queue: rho=<value> >= 1"``); ``remedy`` is the ToolError.hint.
    """

    def __init__(self, rho: float, remedy: str) -> None:
        super().__init__(f"unstable queue: rho={rho:.6g} >= 1")
        self.rho = rho
        self.remedy = remedy


@dataclass(frozen=True)
class QueueMetrics:
    """Steady-state metrics of one queueing model (all closed-form, all exact)."""

    model: str
    lam: float
    mu: float
    servers: int
    rho: float
    offered_load: float  # a = lam/mu, in erlangs
    p0: float
    l_queue: float
    l_system: float
    w_queue: float
    w_system: float
    p_wait: float
    erlang_c: float | None = None  # mmm only (== p_wait there)
    scv: float | None = None  # mg1 only: squared coefficient of variation of service time


def _check_rates(lam: float, mu: float) -> None:
    if not math.isfinite(lam) or lam <= 0.0:
        raise ValueError(f"arrival_rate must be a finite value > 0, got {lam:g}")
    if not math.isfinite(mu) or mu <= 0.0:
        raise ValueError(f"service_rate must be a finite value > 0, got {mu:g}")


def _check_servers(m: int) -> None:
    if m < 1:
        raise ValueError(f"servers must be an integer >= 1, got {m}")


def min_stable_servers(lam: float, mu: float) -> int:
    """Smallest integer m with lam/(m*mu) < 1 strictly (float-edge safe)."""
    m = max(math.floor(lam / mu) + 1, 1)
    while lam / (m * mu) >= 1.0:  # guards float undershoot of the floor at exact ratios
        m += 1
    return m


def _min_servers_for_load(a: float) -> int:
    """Smallest integer m with a/m < 1 strictly (float-edge safe)."""
    m = max(math.floor(a) + 1, 1)
    while a / m >= 1.0:
        m += 1
    return m


def _check_stable(lam: float, mu: float, m: int) -> float:
    rho = lam / (m * mu)
    if rho >= 1.0:
        remedy = f"need service_rate > {lam / m:.6g} per server or servers >= {min_stable_servers(lam, mu)}"
        raise UnstableQueueError(rho, remedy)
    return rho


def mm1_metrics(lam: float, mu: float) -> QueueMetrics:
    """M/M/1: Poisson arrivals, exponential service, one server.

    P0 = 1-rho, L = rho/(1-rho), Lq = rho^2/(1-rho), W = 1/(mu-lam),
    Wq = rho/(mu-lam), p_wait = rho (PASTA: P(server busy) = rho).
    """
    _check_rates(lam, mu)
    rho = _check_stable(lam, mu, 1)
    return QueueMetrics(
        model="mm1",
        lam=lam,
        mu=mu,
        servers=1,
        rho=rho,
        offered_load=lam / mu,
        p0=1.0 - rho,
        l_queue=rho * rho / (1.0 - rho),
        l_system=rho / (1.0 - rho),
        w_queue=rho / (mu - lam),
        w_system=1.0 / (mu - lam),
        p_wait=rho,
    )


def mg1_metrics(lam: float, mu: float, scv: float = 1.0) -> QueueMetrics:
    """M/G/1 via the Pollaczek-Khinchine formula: Lq = rho^2 (1+scv) / (2 (1-rho)).

    ``scv`` is the squared coefficient of variation of the SERVICE time
    (Var[S]/E[S]^2): scv=0 -> deterministic service (M/D/1), scv=1 ->
    exponential service (M/M/1). Exact, not an approximation. p_wait = rho
    (PASTA holds in any M/G/1).
    """
    _check_rates(lam, mu)
    if not math.isfinite(scv) or scv < 0.0:
        raise ValueError(
            f"scv must be a finite value >= 0, got {scv:g} "
            "(0 = deterministic service M/D/1, 1 = exponential service M/M/1)"
        )
    rho = _check_stable(lam, mu, 1)
    l_queue = rho * rho * (1.0 + scv) / (2.0 * (1.0 - rho))
    w_queue = l_queue / lam
    w_system = w_queue + 1.0 / mu
    return QueueMetrics(
        model="mg1",
        lam=lam,
        mu=mu,
        servers=1,
        rho=rho,
        offered_load=lam / mu,
        p0=1.0 - rho,
        l_queue=l_queue,
        l_system=lam * w_system,
        w_queue=w_queue,
        w_system=w_system,
        p_wait=rho,
        scv=scv,
    )


def mmm_metrics(lam: float, mu: float, m: int) -> QueueMetrics:
    """M/M/m: Poisson arrivals, m identical exponential servers (mu per server).

    P0 = [sum_{k<m} a^k/k! + a^m/(m!(1-rho))]^-1, p_wait = Erlang-C,
    Lq = C rho/(1-rho), Wq = Lq/lam, W = Wq + 1/mu, L = lam W.
    """
    _check_rates(lam, mu)
    _check_servers(m)
    rho = _check_stable(lam, mu, m)
    a = lam / mu
    # a^k/k! computed with a running term: no factorial overflow for m <= ~500.
    term = 1.0
    partial = 1.0
    for k in range(1, m):
        term *= a / k
        partial += term
    tail = term * (a / m) / (1.0 - rho)  # a^m / (m! (1-rho))
    p0 = 1.0 / (partial + tail)
    c = erlang_c(a, m)
    l_queue = c * rho / (1.0 - rho)
    w_queue = l_queue / lam
    w_system = w_queue + 1.0 / mu
    return QueueMetrics(
        model="mmm",
        lam=lam,
        mu=mu,
        servers=m,
        rho=rho,
        offered_load=a,
        p0=p0,
        l_queue=l_queue,
        l_system=lam * w_system,
        w_queue=w_queue,
        w_system=w_system,
        p_wait=c,
        erlang_c=c,
    )


def erlang_b(a: float, m: int) -> float:
    """Erlang-B blocking probability of an M/M/m/m loss system at ``a`` erlangs.

    Numerically stable recursion B(k) = a B(k-1) / (k + a B(k-1)), B(0) = 1 —
    defined for ANY offered load (a loss system is always stable).
    """
    _check_servers(m)
    if not math.isfinite(a) or a < 0.0:
        raise ValueError(f"offered_load_erlangs must be a finite value >= 0, got {a:g}")
    b = 1.0
    for k in range(1, m + 1):
        ab = a * b
        b = ab / (k + ab)
    return b


def erlang_c(a: float, m: int) -> float:
    """Erlang-C probability an arrival must wait in an M/M/m delay system.

    Derived from Erlang-B: C = m B / (m - a (1 - B)). Requires a < m — at
    a >= m the delay system has no steady state and this raises
    :class:`UnstableQueueError`.
    """
    b = erlang_b(a, m)  # validates m and a
    rho = a / m
    if rho >= 1.0:
        raise UnstableQueueError(
            rho,
            f"Erlang-C requires offered_load < servers — need servers >= {_min_servers_for_load(a)} "
            f"at {a:.6g} erlangs offered (or use model='b': Erlang-B stays defined for any load)",
        )
    return m * b / (m - a * (1.0 - b))


def solve_little(l_avg: float | None, lam: float | None, w: float | None) -> tuple[float, float, float, str]:
    """Little's theorem L = lam * W: give exactly two, get (L, lam, W, solved_for).

    lam must be > 0 when given; L and W may be 0 (empty system is legal). The
    only division-by-zero slot is solving lam = L/W with W = 0 — that raises a
    ValueError naming w_avg. Applies unchanged to queue-only quantities
    (Lq = lam * Wq in the same slots).
    """
    given = {"l_avg": l_avg, "arrival_rate": lam, "w_avg": w}
    provided = [k for k, v in given.items() if v is not None]
    if len(provided) != 2:
        received = ", ".join(f"{k}={given[k]:g}" for k in provided) or "none"
        raise ValueError(
            f"give exactly two of l_avg, arrival_rate, w_avg — received {len(provided)} ({received})"
        )
    if lam is not None and (not math.isfinite(lam) or lam <= 0.0):
        raise ValueError(f"arrival_rate must be a finite value > 0, got {lam:g}")
    if l_avg is not None and (not math.isfinite(l_avg) or l_avg < 0.0):
        raise ValueError(f"l_avg must be a finite value >= 0, got {l_avg:g}")
    if w is not None and (not math.isfinite(w) or w < 0.0):
        raise ValueError(f"w_avg must be a finite value >= 0, got {w:g}")
    if l_avg is None:
        assert lam is not None and w is not None
        return lam * w, lam, w, "l_avg"
    if w is None:
        assert lam is not None
        return l_avg, lam, l_avg / lam, "w_avg"
    if w == 0.0:
        raise ValueError("cannot solve arrival_rate = l_avg / w_avg: w_avg=0 divides by zero")
    return l_avg, l_avg / w, w, "arrival_rate"

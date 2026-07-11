"""Netqueue tool pack (ELE412 queueing mini): closed-form queueing theory.

Three tools: queueing_calc (M/M/1, M/M/m, M/G/1 via the exact Pollaczek-Khinchine
formula), little_law (L = lambda * W, solve the third from any two), and
erlang_blocking (Erlang-B loss / Erlang-C wait probability). Pure closed-form
math — no files written, no plots, no arrays; scalars in, compact pydantic JSON
out. Same contract as every pack: tools return a model's ``model_dump_json()``;
every failure returns ToolError JSON. Instability follows the pinned split:
error EXACTLY ``"unstable queue: rho=<value> >= 1"``, remedy (needed service
rate / server count) in the hint.

Testability: each registered tool is a thin closure over :class:`AppContext`
delegating to a module-level ``_impl`` function that tests call directly. The
pack writes nothing to disk, so ``ctx`` is unused — kept for the uniform
``_impl(ctx, ...)`` pattern shared by every pack.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from dsp_server.engine import queueing
from dsp_server.engine.queueing import UnstableQueueError
from dsp_server.schemas import SchemaBase, ToolError
from dsp_server.toolsets import AppContext

# mmm's P0 sum uses a running a^k/k! term — overflow-free headroom up to ~500 servers.
_MAX_SERVERS = 512
# Erlang-B recursion is O(servers) and numerically stable — cap only against abuse.
_MAX_BLOCKING_SERVERS = 100_000

_MODEL_HINT = (
    f"expected one of {list(queueing.QUEUE_MODELS)}: 'mm1' (M/M/1), 'mmm' (M/M/m, uses servers), "
    "'mg1' (M/G/1 via Pollaczek-Khinchine, uses scv; scv=0 is M/D/1, scv=1 is M/M/1)"
)
_BLOCKING_MODEL_HINT = (
    f"expected one of {list(queueing.BLOCKING_MODELS)}: 'b' = Erlang-B blocking probability "
    "(loss system, no queue), 'c' = Erlang-C wait probability (delay system, needs offered_load < servers)"
)
_RATES_HINT = (
    "arrival_rate and service_rate must be finite values > 0 (service_rate is per server); "
    "time units are whatever lambda/mu are given in — W/Wq inherit them"
)
_LITTLE_HINT = (
    "give exactly two of l_avg, arrival_rate, w_avg and the third is solved from L = lambda * W; "
    "arrival_rate must be > 0, l_avg/w_avg >= 0; queue-only quantities work in the same slots "
    "(Lq = lambda * Wq)"
)
_BLOCKING_HINT = (
    "servers is a positive integer, offered_load_erlangs >= 0 (a = lambda/mu); "
    "model 'b' is defined for any load, model 'c' requires offered_load < servers"
)


# --------------------------------------------------------------------------- #
# Response models (pack-local, on the shared SchemaBase contract)


class QueueingReport(SchemaBase):
    """queueing_calc: closed-form steady-state metrics of one queueing model."""

    model: str
    arrival_rate: float
    service_rate: float
    servers: int
    scv: float | None = None  # mg1 only
    rho: float
    offered_load_erlangs: float
    p0: float
    l_system: float
    l_queue: float
    w_system: float
    w_queue: float
    p_wait: float
    erlang_c: float | None = None  # mmm only (== p_wait there)
    notes: str | None = None


class LittleLawReport(SchemaBase):
    """little_law: L = lambda * W with the missing quantity solved."""

    l_avg: float
    arrival_rate: float
    w_avg: float
    solved_for: str  # "l_avg" | "arrival_rate" | "w_avg"


class ErlangBlockingReport(SchemaBase):
    """erlang_blocking: Erlang-B loss probability or Erlang-C wait probability."""

    model: str  # "b" | "c"
    servers: int
    offered_load_erlangs: float
    rho: float  # a / servers
    probability: float  # B for model="b", C for model="c"
    erlang_b: float  # always reported (C is derived from B)
    erlang_c: float | None = None  # model="c" only (== probability)
    carried_load_erlangs: float | None = None  # model="b" only: a (1 - B)
    blocked_load_erlangs: float | None = None  # model="b" only: a B
    notes: str | None = None


# --------------------------------------------------------------------------- #
# Shared helpers


def _error(exc: Exception, hint: str | None = None) -> str:
    return ToolError(error=f"{type(exc).__name__}: {exc}", hint=hint).model_dump_json()


def _unstable(exc: UnstableQueueError) -> str:
    # Pinned split: error is EXACTLY "unstable queue: rho=<value> >= 1", remedy in hint.
    return ToolError(error=str(exc), hint=exc.remedy).model_dump_json()


# --------------------------------------------------------------------------- #
# Tool 1: queueing_calc


def _queueing_calc(
    ctx: AppContext,
    arrival_rate: float,
    service_rate: float,
    model: str = "mm1",
    servers: int = 1,
    scv: float = 1.0,
) -> str:
    try:
        if model not in queueing.QUEUE_MODELS:
            return ToolError(error=f"unknown model '{model}'", hint=_MODEL_HINT).model_dump_json()
        m = int(servers)
        if m < 1 or m > _MAX_SERVERS:
            return ToolError(
                error=f"servers must be between 1 and {_MAX_SERVERS}, got {servers}",
                hint="service_rate is per server; model='mmm' is the multi-server lane",
            ).model_dump_json()
        if model in ("mm1", "mg1") and m != 1:
            return ToolError(
                error=f"model '{model}' is single-server but servers={m}",
                hint="use model='mmm' for multi-server queues (service_rate stays per server)",
            ).model_dump_json()
        if model == "mm1":
            metrics = queueing.mm1_metrics(arrival_rate, service_rate)
            notes = None
        elif model == "mmm":
            metrics = queueing.mmm_metrics(arrival_rate, service_rate, m)
            notes = None
        else:
            metrics = queueing.mg1_metrics(arrival_rate, service_rate, scv=scv)
            notes = (
                "M/G/1: Lq/Wq exact via Pollaczek-Khinchine — scv=0 reproduces M/D/1, scv=1 reproduces M/M/1"
            )
        return QueueingReport(
            model=metrics.model,
            arrival_rate=metrics.lam,
            service_rate=metrics.mu,
            servers=metrics.servers,
            scv=metrics.scv,
            rho=metrics.rho,
            offered_load_erlangs=metrics.offered_load,
            p0=metrics.p0,
            l_system=metrics.l_system,
            l_queue=metrics.l_queue,
            w_system=metrics.w_system,
            w_queue=metrics.w_queue,
            p_wait=metrics.p_wait,
            erlang_c=metrics.erlang_c,
            notes=notes,
        ).model_dump_json()
    except UnstableQueueError as exc:
        return _unstable(exc)
    except Exception as exc:  # the MCP boundary must never see a raise
        return _error(exc, hint=_RATES_HINT)


# --------------------------------------------------------------------------- #
# Tool 2: little_law


def _little_law(
    ctx: AppContext,
    l_avg: float | None = None,
    arrival_rate: float | None = None,
    w_avg: float | None = None,
) -> str:
    try:
        l_out, lam_out, w_out, solved_for = queueing.solve_little(l_avg, arrival_rate, w_avg)
        return LittleLawReport(
            l_avg=l_out, arrival_rate=lam_out, w_avg=w_out, solved_for=solved_for
        ).model_dump_json()
    except Exception as exc:
        return _error(exc, hint=_LITTLE_HINT)


# --------------------------------------------------------------------------- #
# Tool 3: erlang_blocking


def _erlang_blocking(
    ctx: AppContext,
    servers: int,
    offered_load_erlangs: float,
    model: str = "b",
) -> str:
    try:
        if model not in queueing.BLOCKING_MODELS:
            return ToolError(error=f"unknown model '{model}'", hint=_BLOCKING_MODEL_HINT).model_dump_json()
        m = int(servers)
        if m > _MAX_BLOCKING_SERVERS:
            return ToolError(
                error=f"servers must be <= {_MAX_BLOCKING_SERVERS}, got {servers}",
                hint=_BLOCKING_HINT,
            ).model_dump_json()
        a = float(offered_load_erlangs)
        b = queueing.erlang_b(a, m)  # validates m >= 1 and a >= 0
        if model == "b":
            return ErlangBlockingReport(
                model="b",
                servers=m,
                offered_load_erlangs=a,
                rho=a / m,
                probability=b,
                erlang_b=b,
                carried_load_erlangs=a * (1.0 - b),
                blocked_load_erlangs=a * b,
                notes="Erlang-B: blocked calls cleared (no queue); probability = P(all servers busy)",
            ).model_dump_json()
        c = queueing.erlang_c(a, m)  # raises UnstableQueueError when a >= m
        return ErlangBlockingReport(
            model="c",
            servers=m,
            offered_load_erlangs=a,
            rho=a / m,
            probability=c,
            erlang_b=b,
            erlang_c=c,
            notes="Erlang-C: blocked calls wait (infinite queue); probability = P(arrival must wait)",
        ).model_dump_json()
    except UnstableQueueError as exc:
        return _unstable(exc)
    except Exception as exc:
        return _error(exc, hint=_BLOCKING_HINT)


# --------------------------------------------------------------------------- #
# Registration


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the three netqueue tools on ``mcp``, bound to ``ctx``."""

    @mcp.tool()
    def queueing_calc(
        arrival_rate: float,
        service_rate: float,
        model: str = "mm1",
        servers: int = 1,
        scv: float = 1.0,
    ) -> str:
        """Closed-form steady-state metrics for one queue. model='mm1' (M/M/1), 'mmm' (M/M/m
        with `servers` identical servers), or 'mg1' (M/G/1 via the EXACT Pollaczek-Khinchine
        formula — `scv` is the squared coefficient of variation of the service time: scv=0 is
        deterministic service M/D/1, scv=1 is exponential service M/M/1; scv is ignored by the
        other models). service_rate is PER SERVER; time units are whatever lambda/mu are given
        in and W/Wq inherit them. Returns QueueingReport JSON: rho, offered load in erlangs,
        P0, L/Lq (mean number in system/queue), W/Wq (mean time in system/queue), p_wait
        (probability an arrival must wait: Erlang-C for mmm, rho for mm1/mg1 by PASTA), and
        erlang_c (mmm only). rho >= 1 returns ToolError 'unstable queue: rho=<value> >= 1'
        with the minimum stable per-server service rate and server count in the hint."""
        return _queueing_calc(ctx, arrival_rate, service_rate, model, servers, scv)

    @mcp.tool()
    def little_law(
        l_avg: float | None = None,
        arrival_rate: float | None = None,
        w_avg: float | None = None,
    ) -> str:
        """Little's theorem L = lambda * W: give exactly two of l_avg (mean number in system),
        arrival_rate (lambda), and w_avg (mean time in system) — the third is solved. Applies
        unchanged to queue-only quantities: pass Lq/Wq in the same slots (Lq = lambda * Wq).
        arrival_rate must be > 0 when given; l_avg and w_avg may be 0 (an empty system is
        legal input). Returns LittleLawReport JSON with all three values and solved_for naming
        the computed one. The only rejected zero is the division-by-zero slot: solving
        arrival_rate = l_avg / w_avg with w_avg=0 returns ToolError naming w_avg."""
        return _little_law(ctx, l_avg, arrival_rate, w_avg)

    @mcp.tool()
    def erlang_blocking(
        servers: int,
        offered_load_erlangs: float,
        model: str = "b",
    ) -> str:
        """Erlang blocking/wait probability for `servers` identical servers at
        offered_load_erlangs (a = lambda/mu, in erlangs). model='b': Erlang-B blocking
        probability of the M/M/m/m LOSS system (blocked calls cleared, no queue) — defined
        for any load; also reports carried/blocked load in erlangs. model='c': Erlang-C
        probability an arrival waits in the M/M/m DELAY system (infinite queue) — requires
        offered_load < servers, otherwise ToolError 'unstable queue: rho=<value> >= 1' with
        the minimum server count in the hint. Computed via the numerically stable Erlang-B
        recursion; C is derived from B and both are reported. Returns ErlangBlockingReport
        JSON."""
        return _erlang_blocking(ctx, servers, offered_load_erlangs, model)

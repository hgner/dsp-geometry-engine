"""OS tool pack (Tanenbaum mini): CPU scheduling, page replacement, and the
Banker's algorithm — three deterministic simulators over pure stdlib math in
:mod:`dsp_server.engine.osalgo`.

Module is named ``os_sim`` (never ``os.py`` — stdlib shadow hazard); the
registry key is ``"os"``. Same contract as the geometry pack: every tool
returns a pydantic model's ``model_dump_json()``; every failure returns a
:class:`ToolError` whose hint restates the tool's input contract. Tables are
hard-capped with explicit truncation flags (60 Gantt segments, first 50 paging
steps, 32x16 Banker matrices).

Testability: each registered tool is a thin closure over :class:`AppContext`
delegating to a module-level ``_impl`` function that tests call directly.
"""

from __future__ import annotations

from dataclasses import asdict

from mcp.server.fastmcp import FastMCP

from dsp_server.engine import osalgo
from dsp_server.schemas import SchemaBase, ToolError
from dsp_server.toolsets import AppContext

_MAX_STEP_ROWS = 50
_PROC_KEYS = frozenset({"name", "arrival", "burst", "priority"})

_SCHEDULE_HINT = (
    "processes = [{name, arrival >= 0, burst > 0, priority?}] with 1..50 unique names; "
    "algorithms: fcfs, sjf, srtf, rr (quantum > 0), priority (LOWER number = higher priority)"
)
_PAGING_HINT = "reference = 1..2000 page numbers >= 0, frames 1..64; algorithms: fifo, lru, opt, clock"
_BANKERS_HINT = (
    "available: m ints; allocation and max_demand: n x m ints with max_demand >= allocation "
    "(n <= 32 processes, m <= 16 resource types); optional request_process (0-based) + request "
    "(m ints) must be given together"
)


# --------------------------------------------------------------------------- #
# Response models (pack-local, on the shared SchemaBase contract)


class ProcStatsSchema(SchemaBase):
    """Per-process outcome: wait = turnaround - burst, turnaround = completion -
    arrival, response = first-time-on-CPU - arrival."""

    name: str
    arrival: float
    burst: float
    priority: int
    start: float
    completion: float
    wait: float
    turnaround: float
    response: float


class ScheduleReport(SchemaBase):
    """schedule_sim: per-process stats (input order) + averages + text Gantt."""

    algorithm: str
    quantum: float | None = None
    n_processes: int
    per_process: list[ProcStatsSchema]
    avg_wait: float
    avg_turnaround: float
    avg_response: float
    makespan: float
    cpu_idle: float
    gantt: str
    gantt_segments: int
    gantt_truncated: bool


class PageStepSchema(SchemaBase):
    """One reference-string step; frames in slot order; evicted None on hits
    and free-slot fills."""

    step: int
    page: int
    fault: bool
    evicted: int | None = None
    frames: list[int]


class PagingReport(SchemaBase):
    """page_replacement_sim: fault counts + first-50-steps table + OPT baseline."""

    algorithm: str
    n_frames: int
    ref_length: int
    faults: int
    hits: int
    fault_rate: float
    opt_faults: int
    vs_opt: str
    steps: list[PageStepSchema]
    steps_truncated: bool


class BankersReport(SchemaBase):
    """bankers_check: safety verdict (of the evaluated state) + optional
    request outcome. safe_sequence is empty when unsafe; stuck when safe."""

    n_processes: int
    n_resources: int
    safe: bool
    safe_sequence: list[str]
    stuck: list[str]
    need: list[list[int]]
    explanation: str
    request_granted: bool | None = None
    request_reason: str | None = None


# --------------------------------------------------------------------------- #
# Shared helpers


def _error(exc: Exception, hint: str | None = None) -> str:
    return ToolError(error=f"{type(exc).__name__}: {exc}", hint=hint).model_dump_json()


def _parse_processes(processes: list[dict]) -> list[osalgo.Proc]:
    if not isinstance(processes, list) or not processes:
        raise ValueError("processes must be a non-empty list of process objects")
    procs: list[osalgo.Proc] = []
    for k, entry in enumerate(processes):
        if not isinstance(entry, dict):
            raise ValueError(f"process #{k} is not an object: {entry!r}")
        unknown = sorted(set(entry) - _PROC_KEYS)
        if unknown:
            raise ValueError(
                f"process #{k} has unknown key(s) {unknown} — allowed: name, arrival, burst, priority"
            )
        missing = [key for key in ("name", "arrival", "burst") if key not in entry]
        if missing:
            raise ValueError(
                f"process #{k} is missing required key(s) {missing} — required: name, arrival, burst"
            )
        name = entry["name"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"process #{k}: name must be a non-empty string, got {name!r}")
        try:
            arrival = float(entry["arrival"])
            burst = float(entry["burst"])
        except (TypeError, ValueError) as inner:
            raise ValueError(f"process #{k} ('{name}'): arrival and burst must be numbers") from inner
        pr_raw = entry.get("priority", 0)
        if isinstance(pr_raw, bool) or not (
            isinstance(pr_raw, int) or (isinstance(pr_raw, float) and pr_raw.is_integer())
        ):
            raise ValueError(f"process #{k} ('{name}'): priority must be an integer, got {pr_raw!r}")
        procs.append(osalgo.Proc(name=name, arrival=arrival, burst=burst, priority=int(pr_raw)))
    return procs


# --------------------------------------------------------------------------- #
# Tool 1: schedule_sim


def _schedule_sim(
    ctx: AppContext,
    processes: list[dict],
    algorithm: str = "fcfs",
    quantum: float = 2.0,
) -> str:
    try:
        if algorithm not in osalgo.SCHEDULE_ALGORITHMS:
            return ToolError(
                error=f"unknown algorithm '{algorithm}'",
                hint=f"expected one of {list(osalgo.SCHEDULE_ALGORITHMS)}",
            ).model_dump_json()
        procs = _parse_processes(processes)
        result = osalgo.simulate_schedule(procs, algorithm, quantum=quantum)
        gantt, n_segments, truncated = osalgo.gantt_text(result.segments)
        n = len(result.per_process)
        return ScheduleReport(
            algorithm=algorithm,
            quantum=quantum if algorithm == "rr" else None,
            n_processes=n,
            per_process=[ProcStatsSchema(**asdict(s)) for s in result.per_process],
            avg_wait=sum(s.wait for s in result.per_process) / n,
            avg_turnaround=sum(s.turnaround for s in result.per_process) / n,
            avg_response=sum(s.response for s in result.per_process) / n,
            makespan=result.makespan,
            cpu_idle=result.idle_time,
            gantt=gantt,
            gantt_segments=n_segments,
            gantt_truncated=truncated,
        ).model_dump_json()
    except Exception as exc:  # the MCP boundary must never see a raise
        return _error(exc, hint=_SCHEDULE_HINT)


# --------------------------------------------------------------------------- #
# Tool 2: page_replacement_sim


def _page_replacement_sim(
    ctx: AppContext,
    reference: list[int],
    frames: int,
    algorithm: str = "lru",
) -> str:
    try:
        if algorithm not in osalgo.PAGING_ALGORITHMS:
            return ToolError(
                error=f"unknown algorithm '{algorithm}'",
                hint=f"expected one of {list(osalgo.PAGING_ALGORITHMS)}",
            ).model_dump_json()
        steps, faults = osalgo.simulate_paging(reference, frames, algorithm)
        if algorithm == "opt":
            opt_faults = faults
            vs_opt = "opt is the offline lower bound"
        else:
            _opt_steps, opt_faults = osalgo.simulate_paging(reference, frames, "opt")
            diff = faults - opt_faults
            pct = 100.0 * diff / opt_faults
            vs_opt = (
                f"{algorithm}: {faults} faults vs OPT {opt_faults} "
                f"(+{diff}, {pct:.1f}% above the offline optimum)"
            )
        ref_length = len(steps)
        return PagingReport(
            algorithm=algorithm,
            n_frames=frames,
            ref_length=ref_length,
            faults=faults,
            hits=ref_length - faults,
            fault_rate=faults / ref_length,
            opt_faults=opt_faults,
            vs_opt=vs_opt,
            steps=[
                PageStepSchema(
                    step=s.step, page=s.page, fault=s.fault, evicted=s.evicted, frames=list(s.frames)
                )
                for s in steps[:_MAX_STEP_ROWS]
            ],
            steps_truncated=len(steps) > _MAX_STEP_ROWS,
        ).model_dump_json()
    except Exception as exc:
        return _error(exc, hint=_PAGING_HINT)


# --------------------------------------------------------------------------- #
# Tool 3: bankers_check


def _bankers_check(
    ctx: AppContext,
    available: list[int],
    allocation: list[list[int]],
    max_demand: list[list[int]],
    process_names: list[str] | None = None,
    request_process: int | None = None,
    request: list[int] | None = None,
) -> str:
    try:
        if (request_process is None) != (request is None):
            return ToolError(
                error="request_process and request must be given together",
                hint=_BANKERS_HINT,
            ).model_dump_json()
        names = process_names
        if request is None or request_process is None:
            outcome = osalgo.bankers_safety(available, allocation, max_demand, names=names)
            granted: bool | None = None
            reason: str | None = None
        else:
            granted, reason, outcome = osalgo.bankers_request(
                available, allocation, max_demand, request_process, request, names=names
            )
        resolved = names if names is not None else [f"P{i}" for i in range(len(allocation))]
        return BankersReport(
            n_processes=len(allocation),
            n_resources=len(available),
            safe=outcome.safe,
            safe_sequence=[resolved[i] for i in outcome.sequence] if outcome.safe else [],
            stuck=[resolved[i] for i in outcome.stuck],
            need=outcome.need,
            explanation=outcome.explanation,
            request_granted=granted,
            request_reason=reason,
        ).model_dump_json()
    except Exception as exc:
        return _error(exc, hint=_BANKERS_HINT)


# --------------------------------------------------------------------------- #
# Registration


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the three os tools on ``mcp``, bound to ``ctx``."""

    @mcp.tool()
    def schedule_sim(processes: list[dict], algorithm: str = "fcfs", quantum: float = 2.0) -> str:
        """Single-CPU scheduling simulation over 1..50 processes, each
        {name, arrival >= 0, burst > 0, priority?} (priority defaults to 0; LOWER number =
        higher priority). Algorithms: fcfs, sjf (non-preemptive shortest job first), srtf
        (preemptive SJF), rr (round-robin with `quantum` > 0; same-instant arrivals enqueue
        BEFORE the preempted process), priority (non-preemptive). Fully deterministic: all
        ties break by (arrival time, input order); zero context-switch cost; the CPU idles
        forward to the next arrival (idle[a-b) Gantt gaps; the timeline starts at t=0).
        Returns ScheduleReport JSON: per-process start/completion/wait/turnaround/response
        (wait = turnaround - burst; response = first-start - arrival) in input order plus
        averages, makespan, total idle time, and a text Gantt chart of MERGED consecutive
        same-process slices like "P1[0-4) | P2[4-7)", capped at 60 segments with
        gantt_truncated. quantum is ignored (and reported null) for non-rr algorithms."""
        return _schedule_sim(ctx, processes, algorithm, quantum)

    @mcp.tool()
    def page_replacement_sim(reference: list[int], frames: int, algorithm: str = "lru") -> str:
        """Page-replacement simulation over an integer reference string (1..2000 page
        numbers >= 0) with 1..64 frames. Algorithms: fifo, lru, opt (Belady offline optimal;
        among pages never used again the one in the LOWEST frame slot is evicted), clock
        (second chance; the reference bit is set on load AND on every hit, and the hand
        clears set bits as it sweeps). Returns PagingReport JSON: faults/hits/fault_rate,
        a per-step table (page, fault, evicted page or null, frames in slot order) trimmed
        to the FIRST 50 steps with steps_truncated (faults always count the full string),
        and — on every run — opt_faults from co-running OPT on the same input, with vs_opt
        comparing this algorithm to the offline lower bound."""
        return _page_replacement_sim(ctx, reference, frames, algorithm)

    @mcp.tool()
    def bankers_check(
        available: list[int],
        allocation: list[list[int]],
        max_demand: list[list[int]],
        process_names: list[str] | None = None,
        request_process: int | None = None,
        request: list[int] | None = None,
    ) -> str:
        """Banker's algorithm: safety check plus optional resource-request evaluation, for
        up to 32 processes x 16 resource types (non-negative integers; max_demand >=
        allocation elementwise — the error names the violating cell like 'P2 resource 1').
        The safety scan is deterministic PASS-CONTINUATION: sweep P0..Pn-1 granting and
        continuing within the pass, repeating until a full pass grants nothing. With
        request_process (0-based) + request (given together): deny 'exceeds need' /
        'exceeds available' without state change, else pretend-grant and re-run safety —
        granted iff the resulting state is safe. Returns BankersReport JSON: safe,
        safe_sequence (names; empty when unsafe), stuck (names; empty when safe), the need
        matrix, an explanation (unsafe: each stuck process's unmet need vs what can ever
        become available), and request_granted/request_reason. When a request passed the
        bounds checks the safety fields describe the POST-pretend-grant state; bounds
        denials report the unchanged original state. Names default to P0..Pn-1
        (process_names overrides everywhere, including the explanation)."""
        return _bankers_check(ctx, available, allocation, max_demand, process_names, request_process, request)

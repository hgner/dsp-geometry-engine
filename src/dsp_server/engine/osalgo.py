"""Pure OS-course algorithms (Tanenbaum/Silberschatz): single-CPU scheduling,
page replacement, and the Banker's algorithm.

Deterministic by contract:

- Scheduling ties break by ``(arrival time, input order)`` and a LOWER priority
  number means HIGHER priority. Context switches are free; when nothing is ready
  the CPU idles forward to the next arrival as an ``("idle", a, b)`` segment; the
  timeline starts at t=0 even when the first arrival is later.
- Round-robin: arrivals at the preemption instant enqueue BEFORE the preempted
  process re-enters the ready queue.
- clock sets the reference bit on load AND on every hit; the hand starts at
  slot 0 and fills free slots in hand order.
- OPT (Belady) evicts the page whose next use is farthest away; among pages
  never used again the one in the LOWEST frame slot loses.
- Banker's safety scan is PASS-CONTINUATION: sweep P0..Pn-1 granting and
  CONTINUING within the pass; repeat passes until a full pass grants nothing.
  This reproduces the textbook sequence [P1, P3, P4, P0, P2] on the canonical
  Silberschatz instance (a restart-after-grant scan would give [P1,P3,P0,P2,P4]).

No I/O, stdlib only. Every failure raises :class:`ValueError` with a hint-ready
message — toolset modules surface it verbatim inside ``ToolError``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

SCHEDULE_ALGORITHMS = ("fcfs", "sjf", "srtf", "rr", "priority")
PAGING_ALGORITHMS = ("fifo", "lru", "opt", "clock")

MAX_PROCESSES = 50
MAX_REFERENCE_LENGTH = 2000
MAX_FRAMES = 64
MAX_BANKER_PROCESSES = 32
MAX_BANKER_RESOURCES = 16

_EPS = 1e-9

# A schedule segment: (process name or "idle", start time, end time).
Segment = tuple[str, float, float]


# --------------------------------------------------------------------------- #
# CPU scheduling


@dataclass(frozen=True)
class Proc:
    """One process: lower ``priority`` number = higher priority (Silberschatz)."""

    name: str
    arrival: float
    burst: float
    priority: int = 0


@dataclass(frozen=True)
class ProcStats:
    """Per-process outcome: wait = turnaround - burst; turnaround = completion -
    arrival; response = first-time-on-CPU - arrival."""

    name: str
    arrival: float
    burst: float
    priority: int
    start: float
    completion: float
    wait: float
    turnaround: float
    response: float


@dataclass(frozen=True)
class ScheduleResult:
    """Segments are merged (consecutive same-name slices coalesce); per_process
    is in input order."""

    segments: list[Segment]
    per_process: list[ProcStats]
    makespan: float
    idle_time: float


def _validate_procs(procs: list[Proc]) -> None:
    if not 1 <= len(procs) <= MAX_PROCESSES:
        raise ValueError(f"expected 1..{MAX_PROCESSES} processes, got {len(procs)}")
    seen: set[str] = set()
    for k, p in enumerate(procs):
        if not p.name:
            raise ValueError(f"process #{k} has an empty name")
        if p.name == "idle":
            raise ValueError("'idle' is reserved for CPU gaps in the Gantt chart — rename the process")
        if p.name in seen:
            raise ValueError(f"duplicate process name '{p.name}' — names must be unique")
        seen.add(p.name)
        if not (math.isfinite(p.arrival) and p.arrival >= 0.0):
            raise ValueError(f"process '{p.name}': arrival must be finite and >= 0, got {p.arrival}")
        if not (math.isfinite(p.burst) and p.burst > 0.0):
            raise ValueError(f"process '{p.name}': burst must be finite and > 0, got {p.burst}")


def _run_nonpreemptive(procs: list[Proc], algorithm: str) -> tuple[list[Segment], list[float], list[float]]:
    n = len(procs)
    if algorithm == "fcfs":

        def key(idx: int) -> tuple[float, float, int]:
            return (procs[idx].arrival, 0.0, idx)

    elif algorithm == "sjf":

        def key(idx: int) -> tuple[float, float, int]:
            return (procs[idx].burst, procs[idx].arrival, idx)

    else:  # priority — lower number first, (arrival, input order) tiebreak

        def key(idx: int) -> tuple[float, float, int]:
            return (float(procs[idx].priority), procs[idx].arrival, idx)

    t = 0.0
    segments: list[Segment] = []
    done = [False] * n
    start = [0.0] * n
    completion = [0.0] * n
    for _ in range(n):
        pending = [i for i in range(n) if not done[i]]
        ready = [i for i in pending if procs[i].arrival <= t]
        if not ready:
            t_next = min(procs[i].arrival for i in pending)
            segments.append(("idle", t, t_next))
            t = t_next
            ready = [i for i in pending if procs[i].arrival <= t]
        i = min(ready, key=key)
        start[i] = t
        t += procs[i].burst
        completion[i] = t
        segments.append((procs[i].name, start[i], t))
        done[i] = True
    return segments, start, completion


def _run_srtf(procs: list[Proc]) -> tuple[list[Segment], list[float], list[float]]:
    n = len(procs)
    remaining = [p.burst for p in procs]
    started = [False] * n
    start = [0.0] * n
    completion = [0.0] * n
    segments: list[Segment] = []
    t = 0.0
    n_done = 0
    while n_done < n:
        pending = [i for i in range(n) if remaining[i] > 0.0]
        ready = [i for i in pending if procs[i].arrival <= t]
        if not ready:
            t_next = min(procs[i].arrival for i in pending)
            segments.append(("idle", t, t_next))
            t = t_next
            continue
        i = min(ready, key=lambda j: (remaining[j], procs[j].arrival, j))
        if not started[i]:
            started[i] = True
            start[i] = t
        future = [procs[j].arrival for j in pending if procs[j].arrival > t]
        next_arr = min(future) if future else math.inf
        if t + remaining[i] <= next_arr + _EPS:  # completes before the next arrival
            run = remaining[i]
            remaining[i] = 0.0
        else:  # run to the arrival, then re-decide (possible preemption)
            run = next_arr - t
            remaining[i] -= run
        end = t + run
        segments.append((procs[i].name, t, end))
        t = end
        if remaining[i] == 0.0:
            completion[i] = t
            n_done += 1
    return segments, start, completion


def _run_rr(procs: list[Proc], quantum: float) -> tuple[list[Segment], list[float], list[float]]:
    n = len(procs)
    order = sorted(range(n), key=lambda i: (procs[i].arrival, i))
    remaining = [p.burst for p in procs]
    started = [False] * n
    start = [0.0] * n
    completion = [0.0] * n
    segments: list[Segment] = []
    queue: deque[int] = deque()
    next_new = 0
    t = 0.0

    def admit(now: float) -> None:
        nonlocal next_new
        while next_new < n and procs[order[next_new]].arrival <= now:
            queue.append(order[next_new])
            next_new += 1

    admit(t)
    n_done = 0
    while n_done < n:
        if not queue:
            t_next = procs[order[next_new]].arrival
            segments.append(("idle", t, t_next))
            t = t_next
            admit(t)
            continue
        i = queue.popleft()
        if not started[i]:
            started[i] = True
            start[i] = t
        if remaining[i] <= quantum + _EPS:
            run = remaining[i]
            remaining[i] = 0.0
        else:
            run = quantum
            remaining[i] -= run
        end = t + run
        segments.append((procs[i].name, t, end))
        t = end
        admit(t)  # same-instant arrivals enqueue BEFORE the preempted process
        if remaining[i] > 0.0:
            queue.append(i)
        else:
            completion[i] = t
            n_done += 1
    return segments, start, completion


def _merge_segments(segments: list[Segment]) -> list[Segment]:
    """Coalesce consecutive same-name slices (they are time-contiguous by
    construction) — a solo RR process shows ONE segment, not one per quantum."""
    merged: list[Segment] = []
    for name, s, e in segments:
        if merged and merged[-1][0] == name:
            merged[-1] = (name, merged[-1][1], e)
        else:
            merged.append((name, s, e))
    return merged


def simulate_schedule(procs: list[Proc], algorithm: str, quantum: float = 2.0) -> ScheduleResult:
    """Simulate one of :data:`SCHEDULE_ALGORITHMS` on a single CPU (zero
    context-switch cost, timeline from t=0). See the module docstring for the
    determinism contract."""
    if algorithm not in SCHEDULE_ALGORITHMS:
        raise ValueError(f"unknown algorithm '{algorithm}' — expected one of {list(SCHEDULE_ALGORITHMS)}")
    _validate_procs(procs)
    if algorithm == "rr":
        if not (math.isfinite(quantum) and quantum > 0.0):
            raise ValueError(f"rr needs quantum > 0, got {quantum}")
        raw = _run_rr(procs, quantum)
    elif algorithm == "srtf":
        raw = _run_srtf(procs)
    else:
        raw = _run_nonpreemptive(procs, algorithm)
    segments, start, completion = raw
    segments = _merge_segments(segments)
    per_process: list[ProcStats] = []
    for i, p in enumerate(procs):
        turnaround = completion[i] - p.arrival
        per_process.append(
            ProcStats(
                name=p.name,
                arrival=p.arrival,
                burst=p.burst,
                priority=p.priority,
                start=start[i],
                completion=completion[i],
                wait=turnaround - p.burst,
                turnaround=turnaround,
                response=start[i] - p.arrival,
            )
        )
    makespan = segments[-1][2] if segments else 0.0
    idle_time = sum(e - s for name, s, e in segments if name == "idle")
    return ScheduleResult(segments=segments, per_process=per_process, makespan=makespan, idle_time=idle_time)


def gantt_text(segments: list[Segment], max_segments: int = 60) -> tuple[str, int, bool]:
    """Render segments as ``"P1[0-24) | P2[24-27) | ..."`` (%g numbers), capped
    at ``max_segments`` then ``" … (+N more)"``. Returns (text, total, truncated)."""
    parts = [f"{name}[{s:g}-{e:g})" for name, s, e in segments[:max_segments]]
    text = " | ".join(parts)
    truncated = len(segments) > max_segments
    if truncated:
        text += f" … (+{len(segments) - max_segments} more)"
    return text, len(segments), truncated


# --------------------------------------------------------------------------- #
# Page replacement


@dataclass(frozen=True)
class PageStep:
    """One reference-string step; ``frames`` lists the occupied slots in slot
    order after the step; ``evicted`` is None on hits and free-slot fills."""

    step: int
    page: int
    fault: bool
    evicted: int | None
    frames: tuple[int, ...]


def _page_int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise ValueError(f"{what} = {value!r} is not an integer page number")
    return int(value)


def _victim_slot(
    algorithm: str,
    slots: list[int | None],
    loaded_at: list[int],
    last_used: list[int],
    pages: list[int],
    idx: int,
) -> int:
    n_frames = len(slots)
    if algorithm == "fifo":
        return min(range(n_frames), key=lambda s: (loaded_at[s], s))
    if algorithm == "lru":
        return min(range(n_frames), key=lambda s: (last_used[s], s))
    # opt: farthest next use wins; ties (never used again) -> lowest slot, via
    # the strict > comparison over slots scanned in ascending order.
    n = len(pages)
    best_slot = 0
    best_next = -1
    for s in range(n_frames):
        page = slots[s]
        nxt = n  # sentinel: never used again
        for j in range(idx + 1, n):
            if pages[j] == page:
                nxt = j
                break
        if nxt > best_next:
            best_next = nxt
            best_slot = s
    return best_slot


def simulate_paging(reference: list[int], n_frames: int, algorithm: str) -> tuple[list[PageStep], int]:
    """Simulate one of :data:`PAGING_ALGORITHMS`; returns (all steps, faults).
    Clock convention: reference bit set on load AND on every hit; the hand
    clears set bits as it sweeps and stops on the first clear one."""
    if algorithm not in PAGING_ALGORITHMS:
        raise ValueError(f"unknown algorithm '{algorithm}' — expected one of {list(PAGING_ALGORITHMS)}")
    if not 1 <= len(reference) <= MAX_REFERENCE_LENGTH:
        raise ValueError(f"reference string length must be 1..{MAX_REFERENCE_LENGTH}, got {len(reference)}")
    if not 1 <= n_frames <= MAX_FRAMES:
        raise ValueError(f"frames must be 1..{MAX_FRAMES}, got {n_frames}")
    pages: list[int] = []
    for k, raw in enumerate(reference):
        page = _page_int(raw, f"reference[{k}]")
        if page < 0:
            raise ValueError(f"reference[{k}] = {page} — page numbers must be >= 0")
        pages.append(page)

    slots: list[int | None] = [None] * n_frames
    loaded_at = [0] * n_frames  # fifo
    last_used = [0] * n_frames  # lru
    ref_bit = [False] * n_frames  # clock
    hand = 0
    steps: list[PageStep] = []
    faults = 0

    for idx, page in enumerate(pages):
        step = idx + 1
        evicted: int | None = None
        if page in slots:
            fault = False
            s = slots.index(page)
            if algorithm == "lru":
                last_used[s] = step
            elif algorithm == "clock":
                ref_bit[s] = True
        else:
            fault = True
            faults += 1
            if algorithm == "clock":
                while True:  # terminates within 2*n_frames iterations (bits get cleared)
                    if slots[hand] is None or not ref_bit[hand]:
                        s = hand
                        break
                    ref_bit[hand] = False
                    hand = (hand + 1) % n_frames
                if slots[s] is not None:
                    evicted = slots[s]
                slots[s] = page
                ref_bit[s] = True
                hand = (s + 1) % n_frames
            else:
                if None in slots:
                    s = slots.index(None)  # lowest free slot
                else:
                    s = _victim_slot(algorithm, slots, loaded_at, last_used, pages, idx)
                    evicted = slots[s]
                slots[s] = page
                loaded_at[s] = step
                last_used[s] = step
        steps.append(
            PageStep(
                step=step,
                page=page,
                fault=fault,
                evicted=evicted,
                frames=tuple(p for p in slots if p is not None),
            )
        )
    return steps, faults


# --------------------------------------------------------------------------- #
# Banker's algorithm


@dataclass(frozen=True)
class BankersOutcome:
    """``sequence``/``stuck`` are process INDICES; ``need`` describes the
    evaluated state; ``explanation`` is human-readable and uses the caller's
    process names."""

    safe: bool
    sequence: list[int]
    stuck: list[int]
    need: list[list[int]]
    explanation: str


def _vec(values: list[int]) -> str:
    return "[" + ",".join(str(int(v)) for v in values) + "]"


def _nonneg_int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        else:
            raise ValueError(f"{what} = {value!r} is not an integer")
    if value < 0:
        raise ValueError(f"{what} = {value} must be >= 0")
    return int(value)


def _validate_banker(
    available: list[int],
    allocation: list[list[int]],
    max_demand: list[list[int]],
    names: list[str] | None,
) -> tuple[list[int], list[list[int]], list[list[int]], list[list[int]], list[str]]:
    n = len(allocation)
    if not 1 <= n <= MAX_BANKER_PROCESSES:
        raise ValueError(f"expected 1..{MAX_BANKER_PROCESSES} processes, got {n} allocation rows")
    m = len(available)
    if not 1 <= m <= MAX_BANKER_RESOURCES:
        raise ValueError(f"expected 1..{MAX_BANKER_RESOURCES} resource types, got {m}")
    if len(max_demand) != n:
        raise ValueError(f"allocation has {n} rows but max_demand has {len(max_demand)}")
    if names is None:
        names = [f"P{i}" for i in range(n)]
    elif len(names) != n:
        raise ValueError(f"process_names has {len(names)} entries but allocation has {n} rows")
    avail = [_nonneg_int(v, f"available[{j}]") for j, v in enumerate(available)]
    alloc: list[list[int]] = []
    maxd: list[list[int]] = []
    for i in range(n):
        if len(allocation[i]) != m or len(max_demand[i]) != m:
            raise ValueError(
                f"row {names[i]}: allocation has {len(allocation[i])} and max_demand has "
                f"{len(max_demand[i])} entries — both must have {m} (one per resource type)"
            )
        alloc.append([_nonneg_int(v, f"allocation[{i}][{j}]") for j, v in enumerate(allocation[i])])
        maxd.append([_nonneg_int(v, f"max_demand[{i}][{j}]") for j, v in enumerate(max_demand[i])])
    need: list[list[int]] = []
    for i in range(n):
        row: list[int] = []
        for j in range(m):
            diff = maxd[i][j] - alloc[i][j]
            if diff < 0:
                raise ValueError(
                    f"max_demand < allocation at {names[i]} resource {j} "
                    f"(max={maxd[i][j]}, allocated={alloc[i][j]})"
                )
            row.append(diff)
        need.append(row)
    return avail, alloc, maxd, need, names


def bankers_safety(
    available: list[int],
    allocation: list[list[int]],
    max_demand: list[list[int]],
    names: list[str] | None = None,
) -> BankersOutcome:
    """Deterministic pass-continuation safety scan (see module docstring)."""
    avail, alloc, _maxd, need, names = _validate_banker(available, allocation, max_demand, names)
    n = len(alloc)
    m = len(avail)
    work = list(avail)
    finished = [False] * n
    sequence: list[int] = []
    granted_any = True
    while granted_any:
        granted_any = False
        for i in range(n):  # grant and CONTINUE within the pass (critique fix #5)
            if finished[i]:
                continue
            if all(need[i][j] <= work[j] for j in range(m)):
                for j in range(m):
                    work[j] += alloc[i][j]
                finished[i] = True
                sequence.append(i)
                granted_any = True
    stuck = [i for i in range(n) if not finished[i]]
    if stuck:
        clauses = [
            f"{names[i]} needs {_vec(need[i])} but only {_vec(work)} can ever become available" for i in stuck
        ]
        explanation = "unsafe: " + "; ".join(clauses)
    else:
        explanation = "safe: sequence " + " -> ".join(names[i] for i in sequence)
    return BankersOutcome(safe=not stuck, sequence=sequence, stuck=stuck, need=need, explanation=explanation)


def bankers_request(
    available: list[int],
    allocation: list[list[int]],
    max_demand: list[list[int]],
    pid: int,
    request: list[int],
    names: list[str] | None = None,
) -> tuple[bool, str, BankersOutcome]:
    """Resource-request algorithm: deny 'exceeds need'/'exceeds available'
    without touching the state (the returned outcome describes the ORIGINAL
    state), else pretend-grant and run the safety scan on the result — granted
    iff safe (the outcome then describes the POST-grant state)."""
    avail, alloc, maxd, need, names = _validate_banker(available, allocation, max_demand, names)
    n = len(alloc)
    m = len(avail)
    pid = _nonneg_int(pid, "request_process")
    if pid >= n:
        raise ValueError(f"request_process must be 0..{n - 1}, got {pid}")
    if len(request) != m:
        raise ValueError(f"request has {len(request)} entries but there are {m} resource types")
    req = [_nonneg_int(v, f"request[{j}]") for j, v in enumerate(request)]
    who = names[pid]
    if any(req[j] > need[pid][j] for j in range(m)):
        outcome = bankers_safety(avail, alloc, maxd, names=names)
        reason = (
            f"denied: exceeds need — {who} requested {_vec(req)} but its remaining need is {_vec(need[pid])}"
        )
        return False, reason, outcome
    if any(req[j] > avail[j] for j in range(m)):
        outcome = bankers_safety(avail, alloc, maxd, names=names)
        reason = f"denied: exceeds available — {who} requested {_vec(req)} but available is {_vec(avail)}"
        return False, reason, outcome
    new_avail = [avail[j] - req[j] for j in range(m)]
    new_alloc = [row[:] for row in alloc]
    for j in range(m):
        new_alloc[pid][j] += req[j]
    outcome = bankers_safety(new_avail, new_alloc, maxd, names=names)
    if outcome.safe:
        seq = " -> ".join(names[i] for i in outcome.sequence)
        return True, f"granted: post-grant state is safe (sequence {seq})", outcome
    stuck_names = ", ".join(names[i] for i in outcome.stuck)
    return False, f"denied: pretend-grant leaves an unsafe state (stuck: {stuck_names})", outcome

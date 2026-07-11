"""OS pack (Tanenbaum mini) tests: textbook-golden scheduling, paging, Banker's.

No engine, no MCP client — the tools are exercised through their module-level
``_impl`` functions (the registered closures delegate 1:1). Every golden number
is hand-verifiable from Silberschatz/Tanenbaum; critique fixes #5 (Banker's
pass-continuation scan -> [P1,P3,P4,P0,P2]) and #10 (merged Gantt segments ->
RR q=4 gives P1 exactly 2 slices) are asserted below.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dsp_server.engine import osalgo
from dsp_server.toolsets import AppContext, os_sim


def _ctx(tmp_path: Path) -> AppContext:
    data = tmp_path / "data"
    return AppContext(
        data_dir=data,
        dumps_dir=data / "dumps",
        cache_dir=data / "cache",
        plots_dir=data / "plots",
        logs_dir=data / "logs",
        engine_root=tmp_path / "engine",
    )


def _p(name: str, arrival: float, burst: float, priority: int | None = None) -> dict:
    entry: dict = {"name": name, "arrival": arrival, "burst": burst}
    if priority is not None:
        entry["priority"] = priority
    return entry


CLASSIC = [_p("P1", 0, 24), _p("P2", 0, 3), _p("P3", 0, 3)]  # Silberschatz FCFS/SJF/RR set

# Banker's canonical 5x3 instance (Silberschatz).
AVAILABLE = [3, 3, 2]
ALLOC = [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]]
MAXD = [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]]

# The same state after granting (1,0,2) to P1 (used by the denial pair below).
AVAILABLE2 = [2, 3, 0]
ALLOC2 = [[0, 1, 0], [3, 0, 2], [3, 0, 2], [2, 1, 1], [0, 0, 2]]


# --------------------------------------------------------------------------- #
# schedule_sim


def test_fcfs_classic(tmp_path: Path):
    rep = json.loads(os_sim._schedule_sim(_ctx(tmp_path), CLASSIC, "fcfs"))
    assert "error" not in rep
    assert [s["wait"] for s in rep["per_process"]] == [0, 24, 27]
    assert rep["avg_wait"] == pytest.approx(17.0)
    assert rep["avg_turnaround"] == pytest.approx(27.0)
    assert rep["makespan"] == 30
    assert rep["cpu_idle"] == 0
    assert rep["gantt"] == "P1[0-24) | P2[24-27) | P3[27-30)"
    assert rep["gantt_segments"] == 3
    assert rep["gantt_truncated"] is False
    assert rep["quantum"] is None  # not rr
    # FCFS is non-preemptive: response == wait for every process
    assert all(s["response"] == s["wait"] for s in rep["per_process"])


def test_sjf_classic(tmp_path: Path):
    rep = json.loads(os_sim._schedule_sim(_ctx(tmp_path), CLASSIC, "sjf"))
    assert "error" not in rep
    assert [s["wait"] for s in rep["per_process"]] == [6, 0, 3]  # order P2, P3, P1
    assert rep["avg_wait"] == pytest.approx(3.0)
    assert rep["gantt"] == "P2[0-3) | P3[3-6) | P1[6-30)"


def test_rr_q4_classic(tmp_path: Path):
    """Critique #10: consecutive same-process slices merge — P1 has exactly 2
    Gantt segments (after t=10 it is alone; quantum expiries re-enqueue only it)."""
    rep = json.loads(os_sim._schedule_sim(_ctx(tmp_path), CLASSIC, "rr", 4.0))
    assert "error" not in rep
    waits = {s["name"]: s["wait"] for s in rep["per_process"]}
    assert waits == {"P1": 6, "P2": 4, "P3": 7}
    assert rep["avg_wait"] == pytest.approx(17 / 3, rel=1e-5)
    responses = {s["name"]: s["response"] for s in rep["per_process"]}
    assert responses == {"P1": 0, "P2": 4, "P3": 7}
    assert rep["avg_response"] == pytest.approx(11 / 3, rel=1e-5)
    assert rep["avg_response"] != rep["avg_wait"]  # response != wait under RR
    assert rep["gantt"] == "P1[0-4) | P2[4-7) | P3[7-10) | P1[10-30)"
    assert rep["gantt"].count("P1[") == 2
    assert rep["gantt_segments"] == 4
    assert rep["quantum"] == 4


def test_srtf_golden(tmp_path: Path):
    procs = [_p("P1", 0, 8), _p("P2", 1, 4), _p("P3", 2, 9), _p("P4", 3, 5)]
    rep = json.loads(os_sim._schedule_sim(_ctx(tmp_path), procs, "srtf"))
    assert "error" not in rep
    assert rep["gantt"] == "P1[0-1) | P2[1-5) | P4[5-10) | P1[10-17) | P3[17-26)"
    assert [s["wait"] for s in rep["per_process"]] == [9, 0, 15, 2]
    assert rep["avg_wait"] == pytest.approx(6.5)


def test_priority_golden(tmp_path: Path):
    procs = [
        _p("P1", 0, 10, 3),
        _p("P2", 0, 1, 1),
        _p("P3", 0, 2, 4),
        _p("P4", 0, 1, 5),
        _p("P5", 0, 5, 2),
    ]
    rep = json.loads(os_sim._schedule_sim(_ctx(tmp_path), procs, "priority"))
    assert "error" not in rep
    assert rep["gantt"] == "P2[0-1) | P5[1-6) | P1[6-16) | P3[16-18) | P4[18-19)"
    assert [s["wait"] for s in rep["per_process"]] == [6, 0, 16, 18, 1]
    assert rep["avg_wait"] == pytest.approx(8.2)


def test_idle_gap(tmp_path: Path):
    rep = json.loads(os_sim._schedule_sim(_ctx(tmp_path), [_p("P1", 0, 2), _p("P2", 5, 2)], "fcfs"))
    assert "error" not in rep
    assert "idle[2-5)" in rep["gantt"]
    assert rep["cpu_idle"] == 3
    assert rep["makespan"] == 7


def test_gantt_truncation(tmp_path: Path):
    """Two RR q=1 processes alternate: 200 unmergeable slices -> capped at 60."""
    procs = [_p("P1", 0, 100), _p("P2", 0, 100)]
    rep = json.loads(os_sim._schedule_sim(_ctx(tmp_path), procs, "rr", 1.0))
    assert "error" not in rep
    assert rep["gantt_segments"] == 200
    assert rep["gantt_truncated"] is True
    assert rep["gantt"].endswith("(+140 more)")


def test_schedule_validation(tmp_path: Path):
    ctx = _ctx(tmp_path)
    dup = json.loads(os_sim._schedule_sim(ctx, [_p("P1", 0, 1), _p("P1", 0, 2)], "fcfs"))
    assert "error" in dup and "duplicate" in dup["error"]
    bad_burst = json.loads(os_sim._schedule_sim(ctx, [_p("P1", 0, 0)], "fcfs"))
    assert "error" in bad_burst and "burst" in bad_burst["error"]
    too_many = json.loads(os_sim._schedule_sim(ctx, [_p(f"P{i}", 0, 1) for i in range(51)], "fcfs"))
    assert "error" in too_many and "50" in too_many["error"]
    bad_quantum = json.loads(os_sim._schedule_sim(ctx, [_p("P1", 0, 1)], "rr", 0.0))
    assert "error" in bad_quantum and "quantum" in bad_quantum["error"]
    unknown = json.loads(os_sim._schedule_sim(ctx, [_p("P1", 0, 1)], "hrrn"))
    assert "error" in unknown
    assert "fcfs" in unknown["hint"] and "srtf" in unknown["hint"]
    missing = json.loads(os_sim._schedule_sim(ctx, [{"name": "P1", "arrival": 0}], "fcfs"))
    assert "error" in missing and "burst" in missing["error"]
    reserved = json.loads(os_sim._schedule_sim(ctx, [_p("idle", 0, 1)], "fcfs"))
    assert "error" in reserved


# --------------------------------------------------------------------------- #
# page_replacement_sim

REF20 = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]  # Silberschatz


@pytest.mark.parametrize(("algorithm", "expected"), [("fifo", 15), ("lru", 12), ("opt", 9)])
def test_silberschatz_fault_counts(tmp_path: Path, algorithm: str, expected: int):
    rep = json.loads(os_sim._page_replacement_sim(_ctx(tmp_path), REF20, 3, algorithm))
    assert "error" not in rep
    assert rep["faults"] == expected
    assert rep["opt_faults"] == 9  # OPT co-runs on every input
    assert rep["hits"] == 20 - expected
    assert rep["fault_rate"] == pytest.approx(expected / 20, rel=1e-5)
    assert rep["ref_length"] == 20
    assert len(rep["steps"]) == 20
    assert rep["steps_truncated"] is False


def test_vs_opt_strings(tmp_path: Path):
    ctx = _ctx(tmp_path)
    lru = json.loads(os_sim._page_replacement_sim(ctx, REF20, 3, "lru"))
    assert "12" in lru["vs_opt"] and "9" in lru["vs_opt"]
    assert "above the offline optimum" in lru["vs_opt"]
    opt = json.loads(os_sim._page_replacement_sim(ctx, REF20, 3, "opt"))
    assert opt["vs_opt"] == "opt is the offline lower bound"


def test_belady_anomaly(tmp_path: Path):
    ctx = _ctx(tmp_path)
    ref = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    r3 = json.loads(os_sim._page_replacement_sim(ctx, ref, 3, "fifo"))
    r4 = json.loads(os_sim._page_replacement_sim(ctx, ref, 4, "fifo"))
    assert (r3["faults"], r4["faults"]) == (9, 10)  # more frames, MORE faults


def test_clock_hand_verified(tmp_path: Path):
    """Bit set on load AND hit: at step 5 the sweep clears 1,2,3 then evicts 1."""
    rep = json.loads(os_sim._page_replacement_sim(_ctx(tmp_path), [1, 2, 3, 1, 4], 3, "clock"))
    assert "error" not in rep
    assert rep["faults"] == 4
    steps = rep["steps"]
    assert steps[3]["fault"] is False  # step 4: page 1 hit
    assert steps[4]["fault"] is True
    assert steps[4]["evicted"] == 1
    assert steps[4]["frames"] == [4, 2, 3]
    assert rep["opt_faults"] == 4  # 4 distinct pages -> 4 compulsory faults


def test_steps_truncation(tmp_path: Path):
    rng = np.random.default_rng(3)
    ref = [int(p) for p in rng.integers(0, 12, size=120)]
    rep = json.loads(os_sim._page_replacement_sim(_ctx(tmp_path), ref, 4, "lru"))
    assert "error" not in rep
    assert len(rep["steps"]) == 50
    assert rep["steps_truncated"] is True
    assert rep["ref_length"] == 120
    full_steps, full_faults = osalgo.simulate_paging(ref, 4, "lru")
    assert len(full_steps) == 120
    assert rep["faults"] == full_faults  # faults counted over ALL 120 steps


def test_paging_validation(tmp_path: Path):
    ctx = _ctx(tmp_path)
    unknown = json.loads(os_sim._page_replacement_sim(ctx, [1, 2], 2, "mru"))
    assert "error" in unknown and "clock" in unknown["hint"]
    assert "error" in json.loads(os_sim._page_replacement_sim(ctx, [1, 2], 0, "lru"))
    assert "error" in json.loads(os_sim._page_replacement_sim(ctx, [1, 2], 65, "lru"))
    assert "error" in json.loads(os_sim._page_replacement_sim(ctx, [], 3, "lru"))
    assert "error" in json.loads(os_sim._page_replacement_sim(ctx, [0] * 2001, 3, "lru"))
    negative = json.loads(os_sim._page_replacement_sim(ctx, [1, -2], 3, "lru"))
    assert "error" in negative and ">= 0" in negative["error"]


# --------------------------------------------------------------------------- #
# bankers_check


def test_bankers_safety_canonical(tmp_path: Path):
    """Critique #5: pass-continuation scan gives the textbook [P1,P3,P4,P0,P2]."""
    rep = json.loads(os_sim._bankers_check(_ctx(tmp_path), AVAILABLE, ALLOC, MAXD))
    assert "error" not in rep
    assert rep["safe"] is True
    assert rep["safe_sequence"] == ["P1", "P3", "P4", "P0", "P2"]
    assert rep["stuck"] == []
    assert rep["need"][0] == [7, 4, 3]
    assert rep["explanation"] == "safe: sequence P1 -> P3 -> P4 -> P0 -> P2"
    assert rep["request_granted"] is None
    assert rep["request_reason"] is None
    assert rep["n_processes"] == 5 and rep["n_resources"] == 3


def test_bankers_custom_names(tmp_path: Path):
    names = ["A", "B", "C", "D", "E"]
    rep = json.loads(os_sim._bankers_check(_ctx(tmp_path), AVAILABLE, ALLOC, MAXD, names))
    assert "error" not in rep
    assert rep["safe_sequence"] == ["B", "D", "E", "A", "C"]
    assert rep["explanation"] == "safe: sequence B -> D -> E -> A -> C"


def test_bankers_request_granted(tmp_path: Path):
    rep = json.loads(os_sim._bankers_check(_ctx(tmp_path), AVAILABLE, ALLOC, MAXD, None, 1, [1, 0, 2]))
    assert "error" not in rep
    assert rep["request_granted"] is True
    assert "granted" in rep["request_reason"]
    assert rep["safe"] is True  # POST-pretend-grant state
    assert rep["safe_sequence"] == ["P1", "P3", "P4", "P0", "P2"]
    assert rep["need"][1] == [0, 2, 0]  # post-grant need of P1


def test_bankers_request_denied_exceeds_available(tmp_path: Path):
    rep = json.loads(os_sim._bankers_check(_ctx(tmp_path), AVAILABLE2, ALLOC2, MAXD, None, 4, [3, 3, 0]))
    assert "error" not in rep
    assert rep["request_granted"] is False
    assert "exceeds available" in rep["request_reason"]
    assert rep["safe"] is True  # bounds denial: the UNCHANGED state is reported
    assert rep["safe_sequence"] == ["P1", "P3", "P4", "P0", "P2"]


def test_bankers_request_denied_unsafe(tmp_path: Path):
    """Textbook: granting (0,2,0) to P0 from the post-grant state strands all 5."""
    rep = json.loads(os_sim._bankers_check(_ctx(tmp_path), AVAILABLE2, ALLOC2, MAXD, None, 0, [0, 2, 0]))
    assert "error" not in rep
    assert rep["request_granted"] is False
    assert "unsafe" in rep["request_reason"]
    assert rep["safe"] is False  # POST-pretend-grant state is reported
    assert rep["safe_sequence"] == []
    assert rep["stuck"] == ["P0", "P1", "P2", "P3", "P4"]
    assert rep["explanation"].startswith("unsafe: ")
    assert "P0 needs [7,2,3] but only [2,1,0] can ever become available" in rep["explanation"]


def test_bankers_validation(tmp_path: Path):
    ctx = _ctx(tmp_path)
    alloc_bad = [row[:] for row in ALLOC]
    alloc_bad[2][1] = 1  # max_demand[2][1] == 0 < allocation -> named cell
    cell = json.loads(os_sim._bankers_check(ctx, AVAILABLE, alloc_bad, MAXD))
    assert "error" in cell and "P2 resource 1" in cell["error"]
    too_many = json.loads(os_sim._bankers_check(ctx, [1], [[0]] * 33, [[1]] * 33))
    assert "error" in too_many and "32" in too_many["error"]
    lonely = json.loads(os_sim._bankers_check(ctx, AVAILABLE, ALLOC, MAXD, None, 1, None))
    assert "error" in lonely and "together" in lonely["error"]
    bad_names = json.loads(os_sim._bankers_check(ctx, AVAILABLE, ALLOC, MAXD, ["A", "B"]))
    assert "error" in bad_names and "process_names" in bad_names["error"]
    negative = json.loads(os_sim._bankers_check(ctx, [3, -3, 2], ALLOC, MAXD))
    assert "error" in negative
    bad_pid = json.loads(os_sim._bankers_check(ctx, AVAILABLE, ALLOC, MAXD, None, 9, [0, 0, 0]))
    assert "error" in bad_pid
    exceeds_need = json.loads(os_sim._bankers_check(ctx, AVAILABLE, ALLOC, MAXD, None, 3, [1, 1, 1]))
    assert "error" not in exceeds_need  # a lawful denial, not a ToolError
    assert exceeds_need["request_granted"] is False
    assert "exceeds need" in exceeds_need["request_reason"]


# --------------------------------------------------------------------------- #
# engine-level details


def test_gantt_text_formatting():
    text, total, truncated = osalgo.gantt_text([("P1", 0.0, 2.5), ("idle", 2.5, 4.0)])
    assert text == "P1[0-2.5) | idle[2.5-4)"
    assert total == 2 and truncated is False


def test_merge_is_time_contiguous():
    result = osalgo.simulate_schedule(
        [osalgo.Proc("P1", 0.0, 9.0), osalgo.Proc("P2", 0.0, 2.0)], "rr", quantum=4.0
    )
    # Raw slices: P1[0-4) P2[4-6) P1[6-10) P1[10-11) — the last two are
    # contiguous same-process slices and must merge into P1[6-11),
    # leaving exactly 3 segments.
    assert [s[0] for s in result.segments] == ["P1", "P2", "P1"]
    assert result.segments[-1] == ("P1", 6.0, 11.0)
    assert result.makespan == 11.0
    assert result.idle_time == 0.0

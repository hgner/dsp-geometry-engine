"""Netqueue pack (ELE412) tests: closed-form goldens, all hand-verifiable.

No engine, no MCP client, no files — the tools are exercised through their
module-level ``_impl`` functions (the registered closures delegate 1:1).
Golden numbers: M/M/1 textbook set (lam=2, mu=3), the M/M/4 set (lam=3, mu=1:
P0 = 1/26.5, Erlang-C = 27/53), Erlang-B table values (B(3,4) = 27/131,
B(2,2) = 0.4), and the classic M/D/1 teaching point (Lq exactly half of
M/M/1's Lq), reached here as M/G/1 with scv=0. JSON assertions use
rel=1e-4 because SchemaBase rounds every float to 6 significant figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dsp_server.engine import queueing
from dsp_server.toolsets import AppContext, netqueue


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


def _ok(raw: str) -> dict:
    rep = json.loads(raw)
    assert "error" not in rep, rep
    return rep


def _err(raw: str) -> dict:
    rep = json.loads(raw)
    assert "error" in rep, rep
    return rep


# --------------------------------------------------------------------------- #
# queueing_calc: M/M/1


def test_mm1_goldens(tmp_path: Path):
    ctx = _ctx(tmp_path)
    rep = _ok(netqueue._queueing_calc(ctx, arrival_rate=2.0, service_rate=3.0, model="mm1"))
    assert rep["model"] == "mm1"
    assert rep["servers"] == 1
    assert rep["rho"] == pytest.approx(2 / 3, rel=1e-4)
    assert rep["offered_load_erlangs"] == pytest.approx(2 / 3, rel=1e-4)
    assert rep["p0"] == pytest.approx(1 / 3, rel=1e-4)
    assert rep["l_system"] == pytest.approx(2.0, rel=1e-4)
    assert rep["l_queue"] == pytest.approx(4 / 3, rel=1e-4)  # 1.33333
    assert rep["w_system"] == pytest.approx(1.0, rel=1e-4)
    assert rep["w_queue"] == pytest.approx(2 / 3, rel=1e-4)
    assert rep["p_wait"] == pytest.approx(2 / 3, rel=1e-4)
    assert rep["erlang_c"] is None
    assert rep["scv"] is None
    assert rep["notes"] is None


# --------------------------------------------------------------------------- #
# queueing_calc: M/M/4 (lam=3, mu=1 -> a=3, rho=0.75)


def test_mm4_goldens(tmp_path: Path):
    ctx = _ctx(tmp_path)
    rep = _ok(netqueue._queueing_calc(ctx, arrival_rate=3.0, service_rate=1.0, model="mmm", servers=4))
    assert rep["model"] == "mmm"
    assert rep["servers"] == 4
    assert rep["rho"] == pytest.approx(0.75, rel=1e-4)
    assert rep["offered_load_erlangs"] == pytest.approx(3.0, rel=1e-4)
    assert rep["p0"] == pytest.approx(1 / 26.5, rel=1e-4)  # 0.0377358
    assert rep["erlang_c"] == pytest.approx(27 / 53, rel=1e-4)  # 0.509434
    assert rep["p_wait"] == rep["erlang_c"]
    assert rep["l_queue"] == pytest.approx(81 / 53, rel=1e-4)  # 1.5283
    assert rep["w_queue"] == pytest.approx(27 / 53, rel=1e-4)  # 0.509434
    assert rep["w_system"] == pytest.approx(80 / 53, rel=1e-4)  # 1.50943
    assert rep["l_system"] == pytest.approx(240 / 53, rel=1e-4)  # 4.5283


def test_erlang_b_table_values():
    assert queueing.erlang_b(3.0, 4) == pytest.approx(27 / 131, rel=1e-9)  # 0.206107
    assert queueing.erlang_b(2.0, 2) == pytest.approx(0.4, rel=1e-9)
    assert queueing.erlang_b(0.0, 5) == 0.0
    assert queueing.erlang_c(3.0, 4) == pytest.approx(27 / 53, rel=1e-9)  # 0.509434


# --------------------------------------------------------------------------- #
# queueing_calc: M/G/1 Pollaczek-Khinchine (critique #9: scv=0 -> M/D/1, scv=1 -> M/M/1)


def test_mg1_scv0_reproduces_md1(tmp_path: Path):
    ctx = _ctx(tmp_path)
    mm1 = _ok(netqueue._queueing_calc(ctx, arrival_rate=2.0, service_rate=3.0, model="mm1"))
    md1 = _ok(netqueue._queueing_calc(ctx, arrival_rate=2.0, service_rate=3.0, model="mg1", scv=0.0))
    assert md1["model"] == "mg1"
    assert md1["scv"] == 0.0
    # THE teaching point: deterministic service halves the M/M/1 queue.
    # Exact at engine level (unrounded); rel=1e-4 on the round6'd JSON values.
    met_md1 = queueing.mg1_metrics(2.0, 3.0, scv=0.0)
    met_mm1 = queueing.mm1_metrics(2.0, 3.0)
    assert met_md1.l_queue == pytest.approx(met_mm1.l_queue / 2, rel=1e-12)
    assert md1["l_queue"] == pytest.approx(mm1["l_queue"] / 2, rel=1e-4)
    assert md1["l_queue"] == pytest.approx(2 / 3, rel=1e-4)  # 0.666667
    assert md1["w_queue"] == pytest.approx(1 / 3, rel=1e-4)  # 0.333333
    assert md1["w_system"] == pytest.approx(2 / 3, rel=1e-4)  # 0.666667
    assert md1["l_system"] == pytest.approx(4 / 3, rel=1e-4)  # 1.33333
    assert md1["p_wait"] == pytest.approx(2 / 3, rel=1e-4)
    assert md1["erlang_c"] is None
    assert "Pollaczek" in md1["notes"]


def test_mg1_scv1_reproduces_mm1(tmp_path: Path):
    ctx = _ctx(tmp_path)
    mm1 = _ok(netqueue._queueing_calc(ctx, arrival_rate=2.0, service_rate=3.0, model="mm1"))
    mg1 = _ok(netqueue._queueing_calc(ctx, arrival_rate=2.0, service_rate=3.0, model="mg1", scv=1.0))
    for field in (
        "rho",
        "offered_load_erlangs",
        "p0",
        "l_system",
        "l_queue",
        "w_system",
        "w_queue",
        "p_wait",
    ):
        assert mg1[field] == pytest.approx(mm1[field], rel=1e-6), field


# --------------------------------------------------------------------------- #
# Little's law as an internal invariant of every model (unrounded engine metrics)


def test_little_invariant_all_models():
    for metrics in (
        queueing.mm1_metrics(2.0, 3.0),
        queueing.mmm_metrics(3.0, 1.0, 4),
        queueing.mg1_metrics(2.0, 3.0, scv=0.7),
    ):
        assert metrics.l_system == pytest.approx(metrics.lam * metrics.w_system, rel=1e-9)
        assert metrics.l_queue == pytest.approx(metrics.lam * metrics.w_queue, rel=1e-9, abs=1e-12)


def test_mmm_servers1_equals_mm1(tmp_path: Path):
    ctx = _ctx(tmp_path)
    mm1 = _ok(netqueue._queueing_calc(ctx, arrival_rate=2.0, service_rate=3.0, model="mm1"))
    mm_m = _ok(netqueue._queueing_calc(ctx, arrival_rate=2.0, service_rate=3.0, model="mmm", servers=1))
    for field in (
        "rho",
        "offered_load_erlangs",
        "p0",
        "l_system",
        "l_queue",
        "w_system",
        "w_queue",
        "p_wait",
    ):
        assert mm_m[field] == pytest.approx(mm1[field], rel=1e-6), field
    assert mm_m["erlang_c"] == pytest.approx(mm_m["p_wait"], rel=1e-6)
    assert mm1["erlang_c"] is None


# --------------------------------------------------------------------------- #
# Instability contract (critique #31: pinned error/hint split)


def test_unstable_mm1_exact_error_and_remedy(tmp_path: Path):
    ctx = _ctx(tmp_path)
    rep = _err(netqueue._queueing_calc(ctx, arrival_rate=5.0, service_rate=4.0, model="mm1"))
    assert rep["error"] == "unstable queue: rho=1.25 >= 1"
    assert "service_rate > 5" in rep["hint"]
    assert "servers >= 2" in rep["hint"]


def test_unstable_rho_exactly_one(tmp_path: Path):
    ctx = _ctx(tmp_path)
    rep = _err(netqueue._queueing_calc(ctx, arrival_rate=4.0, service_rate=4.0, model="mm1"))
    assert rep["error"] == "unstable queue: rho=1 >= 1"
    assert "servers >= 2" in rep["hint"]


def test_unstable_mmm(tmp_path: Path):
    ctx = _ctx(tmp_path)
    rep = _err(netqueue._queueing_calc(ctx, arrival_rate=9.0, service_rate=1.0, model="mmm", servers=4))
    assert rep["error"] == "unstable queue: rho=2.25 >= 1"
    assert "servers >= 10" in rep["hint"]


def test_min_stable_servers():
    assert queueing.min_stable_servers(5.0, 4.0) == 2
    assert queueing.min_stable_servers(4.0, 4.0) == 2
    assert queueing.min_stable_servers(8.0, 2.0) == 5  # rho must be STRICTLY < 1


# --------------------------------------------------------------------------- #
# queueing_calc validation


def test_queueing_calc_unknown_model_hint(tmp_path: Path):
    ctx = _ctx(tmp_path)
    rep = _err(netqueue._queueing_calc(ctx, arrival_rate=1.0, service_rate=2.0, model="md1"))
    assert "md1" in rep["error"]
    for allowed in ("mm1", "mmm", "mg1"):
        assert allowed in rep["hint"]


def test_queueing_calc_bad_inputs(tmp_path: Path):
    ctx = _ctx(tmp_path)
    rep = _err(netqueue._queueing_calc(ctx, arrival_rate=0.0, service_rate=2.0))
    assert "arrival_rate" in rep["error"]
    rep = _err(netqueue._queueing_calc(ctx, arrival_rate=1.0, service_rate=-2.0))
    assert "service_rate" in rep["error"]
    rep = _err(netqueue._queueing_calc(ctx, arrival_rate=1.0, service_rate=2.0, servers=0))
    assert "servers" in rep["error"]
    rep = _err(netqueue._queueing_calc(ctx, arrival_rate=1.0, service_rate=2.0, model="mm1", servers=2))
    assert "single-server" in rep["error"]
    assert "mmm" in rep["hint"]
    rep = _err(netqueue._queueing_calc(ctx, arrival_rate=1.0, service_rate=2.0, model="mg1", scv=-0.5))
    assert "scv" in rep["error"]


# --------------------------------------------------------------------------- #
# little_law


def test_little_law_solves_each_slot(tmp_path: Path):
    ctx = _ctx(tmp_path)
    rep = _ok(netqueue._little_law(ctx, l_avg=240 / 53, arrival_rate=3.0))
    assert rep["solved_for"] == "w_avg"
    assert rep["w_avg"] == pytest.approx(80 / 53, rel=1e-4)  # 1.50943
    assert rep["l_avg"] == pytest.approx(240 / 53, rel=1e-4)
    assert rep["arrival_rate"] == pytest.approx(3.0, rel=1e-4)

    rep = _ok(netqueue._little_law(ctx, arrival_rate=2.0, w_avg=1.0))
    assert rep["solved_for"] == "l_avg"
    assert rep["l_avg"] == pytest.approx(2.0, rel=1e-4)

    rep = _ok(netqueue._little_law(ctx, l_avg=2.0, w_avg=1.0))
    assert rep["solved_for"] == "arrival_rate"
    assert rep["arrival_rate"] == pytest.approx(2.0, rel=1e-4)


def test_little_law_zeros_are_legal(tmp_path: Path):
    """critique #30: L/W >= 0 is legal input; an empty system is a real system."""
    ctx = _ctx(tmp_path)
    rep = _ok(netqueue._little_law(ctx, l_avg=0.0, w_avg=2.5))
    assert rep["solved_for"] == "arrival_rate"
    assert rep["arrival_rate"] == 0.0

    rep = _ok(netqueue._little_law(ctx, arrival_rate=3.0, w_avg=0.0))
    assert rep["solved_for"] == "l_avg"
    assert rep["l_avg"] == 0.0


def test_little_law_argument_count_errors(tmp_path: Path):
    ctx = _ctx(tmp_path)
    rep = _err(netqueue._little_law(ctx, l_avg=2.0))
    assert "exactly two" in rep["error"]
    rep = _err(netqueue._little_law(ctx))
    assert "exactly two" in rep["error"]
    rep = _err(netqueue._little_law(ctx, l_avg=2.0, arrival_rate=2.0, w_avg=1.0))
    assert "exactly two" in rep["error"]
    assert "received 3" in rep["error"]


def test_little_law_value_errors(tmp_path: Path):
    ctx = _ctx(tmp_path)
    rep = _err(netqueue._little_law(ctx, l_avg=-1.0, arrival_rate=2.0))
    assert "l_avg" in rep["error"]
    rep = _err(netqueue._little_law(ctx, arrival_rate=0.0, w_avg=1.0))
    assert "arrival_rate must be" in rep["error"]
    # critique #30: the ONLY rejected zero is the division-by-zero slot, named.
    rep = _err(netqueue._little_law(ctx, l_avg=2.0, w_avg=0.0))
    assert "w_avg=0" in rep["error"]


# --------------------------------------------------------------------------- #
# erlang_blocking


def test_erlang_blocking_b(tmp_path: Path):
    ctx = _ctx(tmp_path)
    rep = _ok(netqueue._erlang_blocking(ctx, servers=4, offered_load_erlangs=3.0, model="b"))
    assert rep["model"] == "b"
    assert rep["probability"] == pytest.approx(27 / 131, rel=1e-4)  # 0.206107
    assert rep["erlang_b"] == rep["probability"]
    assert rep["erlang_c"] is None
    assert rep["rho"] == pytest.approx(0.75, rel=1e-4)
    assert rep["carried_load_erlangs"] == pytest.approx(3 * (1 - 27 / 131), rel=1e-4)
    assert rep["blocked_load_erlangs"] == pytest.approx(3 * 27 / 131, rel=1e-4)
    assert rep["carried_load_erlangs"] + rep["blocked_load_erlangs"] == pytest.approx(3.0, rel=1e-4)


def test_erlang_blocking_c(tmp_path: Path):
    ctx = _ctx(tmp_path)
    rep = _ok(netqueue._erlang_blocking(ctx, servers=4, offered_load_erlangs=3.0, model="c"))
    assert rep["model"] == "c"
    assert rep["probability"] == pytest.approx(27 / 53, rel=1e-4)  # 0.509434
    assert rep["erlang_c"] == rep["probability"]
    assert rep["erlang_b"] == pytest.approx(27 / 131, rel=1e-4)
    assert rep["carried_load_erlangs"] is None


def test_erlang_blocking_zero_load(tmp_path: Path):
    ctx = _ctx(tmp_path)
    rep = _ok(netqueue._erlang_blocking(ctx, servers=5, offered_load_erlangs=0.0, model="b"))
    assert rep["probability"] == 0.0
    assert rep["carried_load_erlangs"] == 0.0


def test_erlang_blocking_c_overload_unstable(tmp_path: Path):
    ctx = _ctx(tmp_path)
    rep = _err(netqueue._erlang_blocking(ctx, servers=2, offered_load_erlangs=2.0, model="c"))
    assert rep["error"] == "unstable queue: rho=1 >= 1"
    assert "servers >= 3" in rep["hint"]
    # Erlang-B stays defined at the same load: a loss system is always stable.
    rep = _ok(netqueue._erlang_blocking(ctx, servers=2, offered_load_erlangs=2.0, model="b"))
    assert rep["probability"] == pytest.approx(0.4, rel=1e-4)


def test_erlang_blocking_validation(tmp_path: Path):
    ctx = _ctx(tmp_path)
    rep = _err(netqueue._erlang_blocking(ctx, servers=4, offered_load_erlangs=3.0, model="x"))
    assert "'b'" in rep["hint"] and "'c'" in rep["hint"]
    rep = _err(netqueue._erlang_blocking(ctx, servers=0, offered_load_erlangs=3.0))
    assert "servers" in rep["error"]
    rep = _err(netqueue._erlang_blocking(ctx, servers=4, offered_load_erlangs=-1.0))
    assert "offered_load_erlangs" in rep["error"]

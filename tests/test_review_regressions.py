"""Regression tests for the adversarial-review findings on the course packs.

Each test pins a confirmed bug so it cannot silently return. Numbered to the
review: engmath DoS gate + residue sign, ml all-NaN column, tabular
TabularError contract, systems peak_time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from dsp_server.engine import ltisys, symmath, tabular
from dsp_server.toolsets import AppContext, ml


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


# --- engmath blocker: parenthesized-exponent DoS gate ---------------------- #


@pytest.mark.parametrize("hostile", ["2**(900*900*900)", "2**(999999*999999)", "2**(999999)"])
def test_huge_parenthesized_exponent_rejected(hostile):
    with pytest.raises(symmath.ExpressionError, match="exponent"):
        symmath.parse_expr_safe(hostile)


def test_legit_powers_still_parse():
    assert symmath.parse_expr_safe("2**10") == sp.Integer(1024)
    # a symbolic exponent is never a DoS risk and must still work
    assert symmath.parse_expr_safe("x**(n+1)").free_symbols == {sp.Symbol("x"), sp.Symbol("n")}


# --- engmath major: residue closing-exponential sign ----------------------- #


def test_residue_negative_frequency_exponential():
    x = sp.Symbol("x")
    for expr_str in ("exp(-I*x)/(x**2+1)", "exp(I*x)/(x**2+1)"):
        res = symmath.real_line_integral_by_residues(symmath.parse_expr_safe(expr_str), x)
        assert float(sp.N(res.value)) == pytest.approx(float(np.pi / np.e), rel=1e-6)


# --- ml major: all-NaN (string) column no longer refuses labeled data ------ #


def _write_labeled_csv(path: Path) -> Path:
    rng = np.random.default_rng(0)
    lines = ["f0,f1,label"]
    for cls, (cx, cy) in enumerate([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]):
        for _ in range(100):
            p = rng.normal([cx, cy], 1.0)
            lines.append(f"{p[0]},{p[1]},{'abc'[cls]}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_cluster_ignores_string_label_column(ctx, tmp_path):
    csv = _write_labeled_csv(tmp_path / "blobs.csv")
    out = json.loads(ml._cluster(ctx, str(csv), algorithm="kmeans", n_clusters=3))
    assert "error" not in out, out
    assert out["n_clusters_found"] == 3


def test_reduce_dims_ignores_string_label_column(ctx, tmp_path):
    csv = _write_labeled_csv(tmp_path / "blobs.csv")
    out = json.loads(ml._reduce_dims(ctx, str(csv), method="pca", n_components=2))
    assert "error" not in out, out
    assert out["explained_variance_ratio"][0] > 0.5


# --- tabular minors: every failure is a TabularError (hint-ready) ----------- #


def test_npz_2d_string_column_is_tabular_error(tmp_path):
    npz = tmp_path / "m.npz"
    np.savez(npz, arr=np.arange(20.0).reshape(10, 2))
    with pytest.raises(tabular.TabularError, match="shape"):
        tabular.load_series(npz, column="temp")


def test_json_list_of_lists_bad_cell_is_tabular_error(tmp_path):
    mj = tmp_path / "m.json"
    mj.write_text('[[1, 2], ["x", 3]]', encoding="utf-8")
    with pytest.raises(tabular.TabularError, match="non-numeric"):
        tabular.load_matrix(mj)


# --- systems minor: peak_time is None for a non-overshooting response ------- #


@pytest.mark.parametrize("t_end", [15.0, 30.0])
def test_first_order_step_has_no_peak_time(t_end):
    t = np.linspace(0.0, t_end, 500)
    y = 1.0 - np.exp(-t)  # first-order lag: monotonic, no overshoot
    metrics = ltisys.step_metrics(t, y)
    assert metrics.overshoot_pct == pytest.approx(0.0, abs=1e-6)
    assert metrics.peak_time is None

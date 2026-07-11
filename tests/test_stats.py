"""Stats pack (ELE320) tests: synthetic csv series + synthetic cylinder dumps.

No engine, no MCP client — the tools are exercised through their module-level
``_impl`` functions (the registered closures delegate 1:1). All seeds are fixed;
every p-threshold assertion below was verified against the pinned seeds at
implementation time (scipy 1.18 goldens).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dsp_server.engine.ply import write_engine_ply
from dsp_server.toolsets import AppContext, stats
from synth import STUB_BONE_MAP, make_cylinder

CORR_FREQ = 120.0  # cycles/meter injected into the ripplier cylinder
CORR_AMP = 0.002  # meters


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


def _write_csv(path: Path, header: list[str], columns: list[np.ndarray]) -> Path:
    rows = np.column_stack(columns)
    lines = [",".join(header)]
    lines += [",".join(f"{v:.10g}" for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _bone_map_json(tmp_path: Path) -> Path:
    path = tmp_path / "bones.json"
    path.write_text(json.dumps({str(k): v for k, v in STUB_BONE_MAP.items()}), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# describe


def test_describe_normal_sample(tmp_path: Path):
    ctx = _ctx(tmp_path)
    y = np.random.default_rng(42).normal(5.0, 2.0, 400)
    path = _write_csv(tmp_path / "series.csv", ["v"], [y])
    rep = json.loads(stats._describe(ctx, str(path)))
    assert "error" not in rep
    assert rep["n"] == 400
    assert rep["n_dropped_nan"] == 0
    assert rep["column"] == "v"
    assert abs(rep["mean"] - 5.0) < 0.25
    assert abs(rep["std"] - 2.0) < 0.2
    assert rep["ci_mean_lo"] < 5.0 < rep["ci_mean_hi"]
    assert rep["ci_std_lo"] < rep["std"] < rep["ci_std_hi"]
    assert rep["normality_test"] == "shapiro"
    assert rep["normality_p"] > 0.05  # verified: 0.730 for this seed
    assert rep["quantiles"]["p50"] == pytest.approx(rep["median"], rel=1e-6)
    assert rep["source_note"] is None


def test_describe_on_dump_profile(tmp_path: Path):
    """critique #20: on .ply the report must say n counts profile samples."""
    ctx = _ctx(tmp_path)
    dump_path = tmp_path / "smooth.ply"
    write_engine_ply(dump_path, make_cylinder(noise=2e-4, seed=0))
    rep = json.loads(stats._describe(ctx, str(dump_path), column="5:posed"))
    assert "error" not in rep
    assert rep["n"] == 256  # axial profile samples, NOT vertices
    assert rep["source_note"] == "n = axial profile samples, not vertices"
    assert rep["column"] == "5:posed"
    assert rep["mean"] == pytest.approx(0.04, rel=0.05)


def test_describe_missing_column_hint(tmp_path: Path):
    ctx = _ctx(tmp_path)
    path = _write_csv(tmp_path / "two.csv", ["t", "r"], [np.arange(10.0), np.arange(10.0) * 2.0])
    rep = json.loads(stats._describe(ctx, str(path)))
    assert "error" in rep
    assert "pass column=" in rep["hint"]
    assert "'t'" in rep["hint"] and "'r'" in rep["hint"]


# --------------------------------------------------------------------------- #
# fit_distribution


def test_fit_distribution_exponential(tmp_path: Path):
    ctx = _ctx(tmp_path)
    y = np.random.default_rng(1).exponential(2.0, 500)
    path = _write_csv(tmp_path / "expon.csv", ["v"], [y])
    rep = json.loads(stats._fit_distribution(ctx, str(path), save_plot=True))
    assert "error" not in rep
    assert rep["best_dist"] == "expon"
    assert rep["skipped"] == {}
    by_name = {c["name"]: c for c in rep["candidates"]}
    assert by_name["expon"]["aic"] < by_name["norm"]["aic"]
    assert by_name["expon"]["loc_fixed"] is True
    assert 1.75 <= by_name["expon"]["params"]["scale"] <= 2.25  # verified: 1.946
    assert by_name["expon"]["chi2_p"] is not None and by_name["expon"]["chi2_p"] > 0.05
    assert by_name["expon"]["k_params"] == 1  # loc pinned -> only scale is free
    aics = [c["aic"] for c in rep["candidates"]]
    assert aics == sorted(aics)  # AIC-ascending contract
    assert rep["plot_path"] is not None and Path(rep["plot_path"]).is_file()


def test_fit_distribution_normal(tmp_path: Path):
    ctx = _ctx(tmp_path)
    y = np.random.default_rng(7).normal(10.0, 2.0, 400)  # strictly positive sample
    path = _write_csv(tmp_path / "norm10.csv", ["v"], [y])
    rep = json.loads(stats._fit_distribution(ctx, str(path)))
    assert "error" not in rep
    assert rep["best_dist"] == "norm"
    best = rep["candidates"][0]
    assert best["chi2_p"] > 0.01  # verified: 0.290 for this seed
    assert best["loc_fixed"] is False
    assert best["params"]["loc"] == pytest.approx(10.0, abs=0.3)


# --------------------------------------------------------------------------- #
# hypothesis_test


def test_hypothesis_welch_detects_shift(tmp_path: Path):
    ctx = _ctx(tmp_path)
    a = np.random.default_rng(0).normal(0.0, 1.0, 200)
    b = np.random.default_rng(1).normal(0.5, 1.0, 200)
    pa = _write_csv(tmp_path / "a.csv", ["v"], [a])
    pb = _write_csv(tmp_path / "b.csv", ["v"], [b])
    rep = json.loads(stats._hypothesis_test(ctx, str(pa), str(pb)))
    assert "error" not in rep
    assert rep["test_used"] == "welch"  # auto's two-sample default
    assert rep["p_value"] < 1e-3  # verified: 1.75e-5
    assert rep["reject_h0"] is True
    assert 0.3 < abs(rep["effect_size"]) < 0.7  # verified: -0.435 (a minus b)
    assert rep["effect_size_kind"] == "cohen_d"
    assert rep["ci_diff_hi"] < 0.0  # CI on mean_a - mean_b excludes 0
    assert rep["df"] is not None
    assert rep["n_a"] == 200 and rep["n_b"] == 200
    assert rep["n_dropped_a"] == 0 and rep["n_dropped_b"] == 0


def test_hypothesis_same_distribution_quiet(tmp_path: Path):
    ctx = _ctx(tmp_path)
    a = np.random.default_rng(3).normal(0.0, 1.0, 200)
    b = np.random.default_rng(4).normal(0.0, 1.0, 200)
    pa = _write_csv(tmp_path / "a.csv", ["v"], [a])
    pb = _write_csv(tmp_path / "b.csv", ["v"], [b])
    rep = json.loads(stats._hypothesis_test(ctx, str(pa), str(pb)))
    assert rep["p_value"] > 0.05  # verified: 0.997 for these seeds
    assert rep["reject_h0"] is False


def test_hypothesis_f_var(tmp_path: Path):
    ctx = _ctx(tmp_path)
    a = np.random.default_rng(0).normal(0.0, 1.0, 200)
    b = np.random.default_rng(1).normal(0.0, 3.0, 200)
    pa = _write_csv(tmp_path / "a.csv", ["v"], [a])
    pb = _write_csv(tmp_path / "b.csv", ["v"], [b])
    rep = json.loads(stats._hypothesis_test(ctx, str(pa), str(pb), test="f_var"))
    assert rep["test_used"] == "f_var"
    assert rep["p_value"] < 1e-6  # verified: 2.4e-43
    assert rep["df"] is None  # F has a dof PAIR (n_a-1, n_b-1) — recoverable from n_a/n_b
    assert rep["effect_size"] is None


def test_hypothesis_auto_one_sample(tmp_path: Path):
    ctx = _ctx(tmp_path)
    a = np.random.default_rng(0).normal(0.0, 1.0, 200)
    pa = _write_csv(tmp_path / "a.csv", ["v"], [a])
    rep = json.loads(stats._hypothesis_test(ctx, str(pa), test="auto"))
    assert rep["test_used"] == "t_one_sample"
    assert rep["n_b"] is None and rep["mean_b"] is None
    assert rep["reject_h0"] is False  # true mean is mu0=0; verified p=0.823


def test_hypothesis_paired_unequal_lengths(tmp_path: Path):
    ctx = _ctx(tmp_path)
    pa = _write_csv(tmp_path / "a.csv", ["v"], [np.random.default_rng(0).normal(0, 1, 200)])
    pb = _write_csv(tmp_path / "b.csv", ["v"], [np.arange(150.0)])
    rep = json.loads(stats._hypothesis_test(ctx, str(pa), str(pb), test="paired_t"))
    assert "error" in rep
    assert "200" in rep["error"] and "150" in rep["error"]  # names BOTH lengths


def test_hypothesis_levene_forces_two_sided(tmp_path: Path):
    """critique #21: scipy's levene takes no alternative — the report must echo
    'two-sided (forced)' instead of silently pretending the request was honored."""
    ctx = _ctx(tmp_path)
    pa = _write_csv(tmp_path / "a.csv", ["v"], [np.random.default_rng(0).normal(0, 1, 200)])
    pb = _write_csv(tmp_path / "b.csv", ["v"], [np.random.default_rng(1).normal(0.5, 1, 200)])
    rep = json.loads(stats._hypothesis_test(ctx, str(pa), str(pb), test="levene", alternative="greater"))
    assert "error" not in rep
    assert rep["test_used"] == "levene"
    assert rep["alternative"] == "two-sided (forced)"


def test_hypothesis_unknown_test(tmp_path: Path):
    ctx = _ctx(tmp_path)
    pa = _write_csv(tmp_path / "a.csv", ["v"], [np.arange(10.0)])
    rep = json.loads(stats._hypothesis_test(ctx, str(pa), test="anova"))
    assert "error" in rep
    assert "f_var" in rep["error"]  # error lists the valid test names


# --------------------------------------------------------------------------- #
# regression_fit


def test_regression_fit_line(tmp_path: Path):
    ctx = _ctx(tmp_path)
    rng = np.random.default_rng(5)
    x = rng.uniform(0.0, 10.0, 100)
    y = 2.0 + 3.0 * x + rng.normal(0.0, 0.1, 100)
    # one NaN row exercises the joint drop + count
    path = _write_csv(tmp_path / "reg.csv", ["x", "y"], [np.append(x, np.nan), np.append(y, 5.0)])
    rep = json.loads(stats._regression_fit(ctx, str(path), y_column="y", save_plot=True))
    assert "error" not in rep
    assert rep["n"] == 100
    assert rep["n_dropped_nan"] == 1
    assert rep["x_columns"] == ["x"]
    intercept, slope = rep["coefficients"]
    assert intercept["name"] == "intercept"
    assert slope["name"] == "x"
    assert 2.9 < slope["estimate"] < 3.1  # verified: 2.995
    assert slope["ci_lo"] < 3.0 < slope["ci_hi"]
    assert intercept["p_value"] < 1e-6
    assert rep["r2"] > 0.99  # verified: 0.99987
    assert rep["f_p"] < 1e-10
    assert rep["resid_normality_p"] > 0.01  # verified: 0.574
    assert 1.0 < rep["durbin_watson"] < 3.0
    assert rep["plot_path"] is not None and Path(rep["plot_path"]).is_file()


def test_regression_bad_column(tmp_path: Path):
    ctx = _ctx(tmp_path)
    path = _write_csv(tmp_path / "reg.csv", ["x", "y"], [np.arange(10.0), np.arange(10.0)])
    rep = json.loads(stats._regression_fit(ctx, str(path), y_column="z"))
    assert "error" in rep
    assert "columns present" in rep["hint"]


# --------------------------------------------------------------------------- #
# compare_dump_ripples


def test_compare_dump_ripples_detects_corrugation(tmp_path: Path):
    ctx = _ctx(tmp_path)
    path_a = tmp_path / "smooth.ply"
    path_b = tmp_path / "corr.ply"
    write_engine_ply(path_a, make_cylinder(noise=2e-4, seed=0))
    write_engine_ply(
        path_b,
        make_cylinder(corrugation_amp=CORR_AMP, corrugation_freq_cpm=CORR_FREQ, noise=2e-4, seed=1),
    )
    rep = json.loads(
        stats._compare_dump_ripples(
            ctx,
            str(path_a),
            str(path_b),
            joint="armLowerL",
            bone_map_path=str(_bone_map_json(tmp_path)),
            n_boot=200,
        )
    )
    assert "error" not in rep
    assert rep["band_source"] == "dominant_peak_b"
    assert rep["f_lo_cpm"] < CORR_FREQ < rep["f_hi_cpm"]  # verified: [83.4, 154.9]
    assert rep["ratio_b_over_a"] > 3.0  # verified: 36.5
    assert rep["bootstrap"]["significant"] is True
    assert rep["bootstrap"]["ci_ratio_lo"] > 1.0
    assert rep["bootstrap"]["n_boot"] == 200
    assert rep["sector"]["test_name"] == "mannwhitneyu"
    assert rep["sector"]["n_sectors_used"] == 8  # critique #28: default n_sectors=8
    assert rep["sector"]["p_value"] < 0.05  # verified: 1.55e-4
    assert rep["verdict"] == "b_ripplier"
    assert rep["vertex_count_a"] == 62 * 16  # cap rings belong to elbow/wrist joints
    assert rep["joint_ids"] == [5]
    assert any("UNPAIRED" in note for note in rep["assumptions"])  # critique #6 rationale


def test_compare_dump_ripples_identical_dumps(tmp_path: Path):
    ctx = _ctx(tmp_path)
    path_a = tmp_path / "smooth_a.ply"
    path_b = tmp_path / "smooth_b.ply"
    write_engine_ply(path_a, make_cylinder(noise=2e-4, seed=0))
    write_engine_ply(path_b, make_cylinder(noise=2e-4, seed=0))
    rep = json.loads(stats._compare_dump_ripples(ctx, str(path_a), str(path_b), joint="5", n_boot=100))
    assert "error" not in rep
    assert rep["ratio_b_over_a"] == pytest.approx(1.0, rel=1e-6)
    assert rep["bootstrap"]["significant"] is False  # CI straddles 1 (verified [0.56, 1.86])
    assert rep["sector"]["significant"] is False  # verified p = 1.0
    assert rep["verdict"] == "no_significant_difference"


def test_compare_dump_ripples_bad_joint_histogram_hint(tmp_path: Path):
    ctx = _ctx(tmp_path)
    path_a = tmp_path / "smooth.ply"
    write_engine_ply(path_a, make_cylinder(noise=2e-4, seed=0))
    rep = json.loads(
        stats._compare_dump_ripples(
            ctx, str(path_a), str(path_a), joint="noSuchBone", bone_map_path=str(_bone_map_json(tmp_path))
        )
    )
    assert "error" in rep
    assert "noSuchBone" in rep["error"]
    assert "histogram" in rep["hint"]


def test_compare_dump_ripples_arg_validation(tmp_path: Path):
    ctx = _ctx(tmp_path)
    path_a = tmp_path / "smooth.ply"
    write_engine_ply(path_a, make_cylinder(noise=2e-4, seed=0))
    rep = json.loads(stats._compare_dump_ripples(ctx, str(path_a), str(path_a), method="jackknife"))
    assert "error" in rep and "bootstrap|sector|both" in rep["hint"]
    rep = json.loads(stats._compare_dump_ripples(ctx, str(path_a), str(path_a), f_lo_cpm=50.0))
    assert "error" in rep and "BOTH" in rep["error"]

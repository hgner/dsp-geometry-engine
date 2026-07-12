"""Hardening regression tests — locks the fixes from the full-surface audit and the
security/robustness invariants they protect (DoS gates, non-finite -> JSON, registry
resilience, contract-honoring degenerate paths).
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from dsp_server import config
from dsp_server.engine import ltisys, scoring, statskit, symmath, tabular, wavelets
from dsp_server.schemas import SchemaBase, _round_nested, round6
from dsp_server.toolsets import AppContext, rendering


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
# symmath: power-tower DoS gate (nested-base compounding)


@pytest.mark.parametrize(
    "expr",
    [
        "(((2**64)**64)**64)**64",  # nested base: effective 2**(64**4)
        "((((2**64)**64)**64)**64)**64",
        "2**(900*900*900)",  # parenthesized exponent
        "9**9**9**9",  # chained
        "x**900",  # bare oversize
    ],
)
def test_symmath_power_bombs_rejected_fast(expr: str):
    with pytest.raises(symmath.ExpressionError):
        symmath.parse_expr_safe(expr, symbols=["x"])


@pytest.mark.parametrize("expr", ["x**2 + 3*x + 1", "(x**2)**3", "2**10", "sin(x)**2", "x**(1/2)"])
def test_symmath_legit_powers_accepted(expr: str):
    symmath.parse_expr_safe(expr, symbols=["x"])  # must not raise


# --------------------------------------------------------------------------- #
# tabular: size / decompression guards


def test_tabular_npz_zip_bomb_guarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    p = tmp_path / "bomb.npz"
    np.savez_compressed(p, X=np.zeros(20_000, dtype=np.float64))  # tiny compressed, 160 KB uncompressed
    monkeypatch.setattr(tabular, "_NPZ_MEMBER_BYTE_BUDGET", 1024)  # 1 KB budget
    with pytest.raises(tabular.TabularError, match="decompress"):
        tabular.load_series(str(p))


def test_tabular_ply_respects_file_size_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ply = tmp_path / "tiny.ply"
    ply.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(tabular, "MAX_FILE_BYTES", 10)  # smaller than the file
    with pytest.raises(tabular.TabularError, match="MB"):
        tabular.load_series(str(ply))


def test_tabular_list_of_dicts_distinct_keys_is_bounded(tmp_path: Path):
    # Each record has a distinct key -> n x n cells; the cell guard must fire quickly
    # (the old O(n^2) dedup would hang), not spin.
    p = tmp_path / "records.json"
    p.write_text(json.dumps([{f"k{i}": 1.0} for i in range(4000)]), encoding="utf-8")
    with pytest.raises(tabular.TabularError, match="cells"):
        tabular.load_table(str(p))


# --------------------------------------------------------------------------- #
# schema: non-finite -> valid JSON null; nested rounding; bool preserved


def test_round_nested_recurses_and_preserves_bool():
    out = _round_nested({"a": [1.23456789, 2.0], "b": True, "c": {"d": 3.14159265}})
    assert out["a"][0] == 1.23457  # 6 sig figs inside a list
    assert out["c"]["d"] == 3.14159  # inside a nested dict
    assert out["b"] is True  # a bool is NOT coerced to 1.0


def test_model_non_finite_serializes_to_json_null():
    class _M(SchemaBase):
        x: float

    for bad in (float("inf"), float("-inf"), float("nan")):
        assert not math.isfinite(round6(bad))  # round6 passes it through unchanged
        out = json.loads(_M(x=bad).model_dump_json())  # valid JSON (null), not Infinity/NaN token
        assert out["x"] is None


# --------------------------------------------------------------------------- #
# registry resilience: a broken pack must not sink the server


def test_registry_broken_pack_is_skipped(stub_engine_env, monkeypatch: pytest.MonkeyPatch):
    from dsp_server import toolsets
    from dsp_server.server import create_server

    def _boom(mcp, _ctx):
        raise RuntimeError("simulated broken pack")

    monkeypatch.setitem(toolsets.TOOLSETS, "netqueue", _boom)
    names = {t.name for t in asyncio.run(create_server().list_tools())}
    assert "queueing_calc" not in names  # the broken pack's tools are absent
    assert "describe" in names and "extract_mesh_telemetry" in names  # the rest still registered


def test_toolsets_whitespace_env_registers_all(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DSP_TOOLSETS", "   ")  # whitespace-only used to register ZERO packs
    assert config.enabled_toolsets() is None
    monkeypatch.setenv("DSP_TOOLSETS", "")
    assert config.enabled_toolsets() is None
    monkeypatch.setenv("DSP_TOOLSETS", "netqueue, os")
    assert config.enabled_toolsets() == ["netqueue", "os"]


# --------------------------------------------------------------------------- #
# correctness / contract-honoring degenerate paths


def test_match_shapes_i2_is_stable_near_unit_moment():
    # A log-Hu moment near 0 (|h| ~ 1) is where OpenCV's I1 reciprocal blows up; I2 stays
    # bounded. |0.02-0.90| + |−1.0−−1.1| = 0.98 (not the ~48.9 the reciprocal would give).
    a = np.array([0.02, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    b = np.array([0.90, -1.1, -2.0, -3.0, -4.0, -5.0, -6.0])
    assert scoring.match_shapes_i2(a, b) == pytest.approx(0.98, abs=0.01)


def test_wavelet_flags_excess_detail_not_parity():
    rng = np.random.default_rng(0)
    ref = ndimage.gaussian_filter(rng.random((64, 64)), sigma=3.0, mode="reflect")  # smooth
    gen = np.clip(ref + 0.3 * rng.random((64, 64)), 0.0, 1.0)  # add hallucinated fine detail
    cmp = wavelets.compare_wavelets(ref, gen, wavelet="db2", levels=3)
    assert cmp.interpretation == "excess detail / hallucination"
    assert cmp.micro_parity > 1.5


def test_wavelet_constant_reference_no_blowup():
    ref = np.full((64, 64), 0.5, dtype=np.float64)  # flat: detail bands are float roundoff
    gen = ndimage.gaussian_filter(np.random.default_rng(0).random((64, 64)), sigma=1.0)
    cmp = wavelets.compare_wavelets(ref, gen, wavelet="db2", levels=3)
    assert all(p <= wavelets._PARITY_CAP for p in cmp.detail_parity_by_scale)  # not ~1e30
    assert cmp.interpretation != "parity"  # flat ref vs textured gen is genuinely not parity


def test_ltisys_zero_numerator_rejected():
    with pytest.raises(ValueError, match="nonzero"):
        ltisys.build_system(num=[0.0], den=[1.0, -0.5], domain="z", dt=1.0)


def test_describe_constant_sample_no_nan():
    r = statskit.describe_series(np.full(50, 3.0))
    assert r.skewness == 0.0 and math.isfinite(r.kurtosis_excess)
    assert r.normality_test == "degenerate" and math.isfinite(r.normality_p)


def test_brdf_mc_samples_one_does_not_sink_report(ctx: AppContext):
    rep = json.loads(rendering._verify_brdf_energy(ctx, model="ggx", roughness=0.5, mc_samples=1))
    assert "error" not in rep, rep
    assert rep["variance_reduction"] is None  # MC pass skipped (needs >= 2), report intact
    assert "energy_conserving" in rep

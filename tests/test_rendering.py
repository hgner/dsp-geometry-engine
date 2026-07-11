"""Rendering pack tests: the white-furnace BRDF energy audit.

All goldens are deterministic. The Lambertian directional albedo is exact by
construction (Gauss-Legendre in cos_theta integrates the cos-weighted constant
integrand exactly), so it is pinned to the input albedo. GGX energy conservation,
the single-scatter loss trend, the d_gain leak detector, and the Monte-Carlo
variance-reduction claim are all checked through the module-level ``_impl`` /
:mod:`dsp_server.engine.brdf` entry points (no MCP boundary in the loop).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dsp_server.engine import brdf
from dsp_server.toolsets import AppContext
from dsp_server.toolsets.rendering import _verify_brdf_energy


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
# Lambertian: directional albedo == albedo, exactly, at every angle


def test_lambertian_albedo_is_exactly_the_input_albedo(ctx: AppContext):
    # Pure-math self-check: rho == albedo (0.8) at every wo angle, rel 1e-3.
    for deg in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0):
        rho = brdf.hemisphere_albedo("lambertian", np.deg2rad(deg), params={"albedo": 0.8})
        assert rho == pytest.approx(0.8, rel=1e-3)  # pinned golden: Lambertian rho == 0.8

    rep = json.loads(_verify_brdf_energy(ctx, model="lambertian", albedo=0.8, save_plot=True))
    assert "error" not in rep
    assert rep["energy_conserving"] is True
    assert rep["single_scatter_loss"] is None  # loss is a ggx-only diagnostic
    for v in rep["albedo_by_angle_deg"].values():
        assert v == pytest.approx(0.8, rel=1e-3)
    assert rep["max_albedo"] == pytest.approx(0.8, rel=1e-3)
    assert rep["min_albedo"] == pytest.approx(0.8, rel=1e-3)

    curve = Path(rep["albedo_curve_path"])
    assert curve.parent == ctx.data_dir / "brdf"
    with np.load(curve, allow_pickle=False) as z:
        assert z["rho"].shape == (6,)
        np.testing.assert_allclose(z["rho"], 0.8, rtol=1e-3)
    assert Path(rep["plot_path"]).is_file()


# --------------------------------------------------------------------------- #
# GGX: energy-conserving, and single-scatter loss grows with roughness


def test_ggx_energy_conserving_at_low_roughness(ctx: AppContext):
    rep = json.loads(_verify_brdf_energy(ctx, model="ggx", roughness=0.3, f0=0.04))
    assert "error" not in rep
    assert rep["energy_conserving"] is True
    vals = list(rep["albedo_by_angle_deg"].values())
    assert all(0.0 < v <= 1.0 + 1e-6 for v in vals)  # 0 < rho <= 1 at every angle
    assert rep["max_albedo"] <= 1.0 + 1e-3
    assert rep["single_scatter_loss"] is not None


def test_ggx_single_scatter_loss_grows_with_roughness(ctx: AppContext):
    loss = {}
    for r in (0.2, 0.5, 0.9):
        rep = json.loads(_verify_brdf_energy(ctx, model="ggx", roughness=r, f0=0.04))
        assert "error" not in rep
        assert rep["energy_conserving"] is True
        loss[r] = rep["single_scatter_loss"]
    # More single-scatter darkening at higher roughness (Smith masking loses energy).
    assert loss[0.9] > loss[0.5] > loss[0.2]


# --------------------------------------------------------------------------- #
# Leak detector: a mis-normalized D (d_gain=2.0) manufactures energy


def test_broken_brdf_d_gain_leak_is_detected(ctx: AppContext):
    rep = json.loads(_verify_brdf_energy(ctx, model="ggx", roughness=0.5, f0=1.0, d_gain=2.0))
    assert "error" not in rep
    # Pinned leak-detection golden: doubled D-normalization pushes rho past 1.
    assert rep["max_albedo"] > 1.0
    assert rep["energy_conserving"] is False

    # ...and the physical (d_gain=1.0) counterpart stays conserving.
    ok = json.loads(_verify_brdf_energy(ctx, model="ggx", roughness=0.5, f0=1.0, d_gain=1.0))
    assert ok["energy_conserving"] is True
    assert ok["max_albedo"] <= 1.0 + 1e-3


# --------------------------------------------------------------------------- #
# Monte-Carlo: cosine importance sampling beats uniform (variance_reduction > 1)


def test_mc_cosine_importance_sampling_reduces_variance(ctx: AppContext):
    params = {"roughness": 0.4, "f0": 0.04, "d_gain": 1.0}
    var_u = brdf.mc_variance("ggx", 0.0, 40000, seed=0, sampling="uniform", params=params)
    var_c = brdf.mc_variance("ggx", 0.0, 40000, seed=0, sampling="cosine", params=params)
    assert var_c < var_u
    assert var_u / var_c > 1.0

    rep = json.loads(_verify_brdf_energy(ctx, model="ggx", roughness=0.4, f0=0.04, mc_samples=40000, seed=0))
    assert "error" not in rep
    assert rep["mc_variance_uniform"] is not None
    assert rep["mc_variance_cosine"] is not None
    assert rep["variance_reduction"] > 1.0

    # Determinism: same seed -> identical variance estimate.
    var_u2 = brdf.mc_variance("ggx", 0.0, 40000, seed=0, sampling="uniform", params=params)
    assert var_u2 == var_u


# --------------------------------------------------------------------------- #
# Failure envelope: unknown model returns ToolError JSON, never raises


def test_unknown_model_is_tool_error(ctx: AppContext):
    rep = json.loads(_verify_brdf_energy(ctx, model="phong"))
    assert "error" in rep
    assert "phong" in rep["error"]
    assert "ggx" in rep["hint"]

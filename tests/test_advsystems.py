"""Advanced systems-pack tests (ELE301): group delay (LTI digital/analog + the
cross-spectrum LAG lane) and state-space controllability/observability.

No engine, no MCP client. Tool bodies are exercised through their module-level
``_impl`` functions with a tmp AppContext (the registered closures delegate 1:1).
Goldens are pinned with tight tolerances; the two backing engine modules
(ltisys.group_delay_*, statespace.*) are also spot-checked directly.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import firwin

from dsp_server.engine import ltisys, statespace
from dsp_server.toolsets import AppContext, systems


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
# group_delay — LTI digital: linear-phase FIR is a flat (N-1)/2 samples


def test_group_delay_linear_phase_fir(ctx: AppContext):
    b = firwin(21, 0.3).tolist()  # Type-I symmetric FIR -> group delay = (21-1)/2 = 10 samples
    res = json.loads(systems._group_delay(ctx, num=b, den=[1.0], domain="z", dt=1.0, save_plot=True))
    assert "error" not in res
    assert res["mode"] == "lti"
    assert res["domain"] == "z"
    assert res["tau_unit"] == "samples"
    assert res["dt"] == 1.0
    assert abs(res["tau_at_dc"] - 10.0) < 1e-6
    assert abs(res["tau_mean"] - 10.0) < 0.5
    assert Path(res["plot_path"]).is_file()

    with np.load(res["group_delay_path"]) as z:
        assert set(z.files) == {"w", "tau", "tau_seconds"}
        w, tau = z["w"], z["tau"]
        assert np.allclose(z["tau_seconds"], tau, atol=1e-9)  # dt = 1 sample -> seconds == samples

    # Tight golden: variance ~ 0 over the passband (below the 0.3*Nyquist cutoff).
    passband = w < 0.25 * np.pi  # Nyquist = pi/dt = pi rad/s
    assert int(passband.sum()) > 10
    assert np.allclose(tau[passband], 10.0, atol=1e-6)
    assert float(np.var(tau[passband])) < 1e-6


# --------------------------------------------------------------------------- #
# group_delay — LTI analog: H(s)=1/(s+1) has tau = 1/(1+w^2) -> 1 s at DC


def test_group_delay_analog_first_order(ctx: AppContext):
    res = json.loads(systems._group_delay(ctx, num=[1.0], den=[1.0, 1.0], domain="s"))
    assert "error" not in res
    assert res["mode"] == "lti"
    assert res["domain"] == "s"
    assert res["tau_unit"] == "seconds"
    assert res["dt"] is None
    assert abs(res["tau_at_dc"] - 1.0) < 0.02
    assert abs(res["tau_max"] - 1.0) < 0.02  # tau decreases with w, max at the low end
    assert abs(res["tau_mean_s"] - res["tau_mean"]) < 1e-9  # already seconds


def test_group_delay_lti_z_requires_dt(ctx: AppContext):
    res = json.loads(systems._group_delay(ctx, num=[1.0], den=[1.0, -0.5], domain="z"))
    assert "error" in res
    assert "dt" in res["error"]


# --------------------------------------------------------------------------- #
# group_delay — cross-spectrum: a pure k-sample delay gives a flat tau = k*dt


def _delayed_pair(tmp_path: Path, n: int, k: int, dt: float, seed: int) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n)
    a = np.roll(base, k)  # a[n] = base[n-k]: series_a LAGS series_b by k samples
    pa, pb = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(pa, y=a, dt=np.asarray(dt))
    np.savez(pb, y=base, dt=np.asarray(dt))
    return pa, pb


def test_group_delay_cross_spectrum_pure_delay(ctx: AppContext, tmp_path: Path):
    n, dt, k = 256, 0.01, 5
    pa, pb = _delayed_pair(tmp_path, n, k, dt, seed=0)

    res = json.loads(systems._group_delay(ctx, series_a=str(pa), series_b=str(pb)))
    assert "error" not in res
    assert res["mode"] == "cross_spectrum"
    assert res["domain"] == "cross-spectrum"
    assert res["tau_unit"] == "seconds"
    assert res["dt"] == dt
    assert res["n_points"] == n // 2 + 1
    assert abs(res["tau_at_dc"] - k * dt) < 1e-6  # positive tau = a lags b

    with np.load(res["group_delay_path"]) as z:
        tau = z["tau"]
    base = np.random.default_rng(0).standard_normal(n)  # series_b, to mask the coherent band
    power = np.abs(np.fft.rfft(base)) ** 2
    coherent = power > 0.01 * power.max()
    assert np.allclose(tau[coherent], k * dt, atol=1e-6)


def test_group_delay_cross_spectrum_requires_dt(ctx: AppContext, tmp_path: Path):
    rng = np.random.default_rng(1)
    base = rng.standard_normal(128)
    pa, pb = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(pa, np.roll(base, 3))
    np.save(pb, base)

    res = json.loads(systems._group_delay(ctx, series_a=str(pa), series_b=str(pb)))
    assert "error" in res
    assert "pass dt=" in res["error"]

    ok = json.loads(systems._group_delay(ctx, series_a=str(pa), series_b=str(pb), dt=0.02))
    assert "error" not in ok
    assert abs(ok["tau_at_dc"] - 3 * 0.02) < 1e-6


def test_group_delay_cross_spectrum_needs_both_and_equal_length(ctx: AppContext, tmp_path: Path):
    pa, pb = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(pa, y=np.ones(64), dt=np.asarray(0.01))
    np.savez(pb, y=np.ones(32), dt=np.asarray(0.01))

    only_a = json.loads(systems._group_delay(ctx, series_a=str(pa)))
    assert "error" in only_a
    assert "series_b" in only_a["error"]

    mismatch = json.loads(systems._group_delay(ctx, series_a=str(pa), series_b=str(pb)))
    assert "error" in mismatch
    assert "equal-length" in mismatch["error"]


def test_group_delay_xspec_helper_sign_and_flatness():
    rng = np.random.default_rng(2)
    base = rng.standard_normal(200)
    w, tau = ltisys.group_delay_xspec(np.roll(base, 4), base, 0.005)
    assert w[0] == 0.0
    power = np.abs(np.fft.rfft(base)) ** 2
    coherent = power > 0.01 * power.max()
    assert np.allclose(tau[coherent], 4 * 0.005, atol=1e-6)


# --------------------------------------------------------------------------- #
# state_space_analysis — controllability


def test_state_space_controllable(ctx: AppContext):
    res = json.loads(systems._state_space_analysis(ctx, a=[[0, 1], [0, 0]], b=[[0], [1]]))
    assert "error" not in res
    assert res["n_states"] == 2 and res["n_inputs"] == 1
    assert res["ctrb_rank"] == 2
    assert res["controllable"] is True
    assert res["uncontrollable_dim"] == 0
    assert res["ctrb_min_singular"] > 0.5  # both singular values = 1
    assert res["n_outputs"] is None and res["observable"] is None
    assert "controllable" in res["note"]


def test_state_space_uncontrollable(ctx: AppContext):
    res = json.loads(systems._state_space_analysis(ctx, a=[[1, 0], [0, 2]], b=[[1], [0]]))
    assert "error" not in res
    assert res["ctrb_rank"] == 1
    assert res["controllable"] is False
    assert res["uncontrollable_dim"] == 1
    assert res["ctrb_min_singular"] < 1e-9
    assert "NOT controllable" in res["note"]
    assert "1 uncontrollable" in res["note"]


# --------------------------------------------------------------------------- #
# state_space_analysis — observability


def test_state_space_observable(ctx: AppContext):
    res = json.loads(systems._state_space_analysis(ctx, a=[[0, 1], [0, 0]], b=[[0], [1]], c=[[1, 0]]))
    assert "error" not in res
    assert res["n_outputs"] == 1
    assert res["obsv_rank"] == 2
    assert res["observable"] is True
    assert res["unobservable_dim"] == 0
    assert res["obsv_min_singular"] > 0.5
    assert "observable" in res["note"]


def test_state_space_unobservable(ctx: AppContext):
    res = json.loads(systems._state_space_analysis(ctx, a=[[1, 0], [0, 2]], b=[[1], [1]], c=[[1, 0]]))
    assert "error" not in res
    assert res["obsv_rank"] == 1
    assert res["observable"] is False
    assert res["unobservable_dim"] == 1
    assert res["obsv_min_singular"] < 1e-9
    assert "NOT observable" in res["note"]


# --------------------------------------------------------------------------- #
# state_space_analysis — path inputs and validation


def test_state_space_matrix_from_path(ctx: AppContext, tmp_path: Path):
    a_path, b_path = tmp_path / "A.npz", tmp_path / "B.npz"
    np.savez(a_path, np.array([[0.0, 1.0], [0.0, 0.0]]))
    np.savez(b_path, np.array([[0.0], [1.0]]))

    res = json.loads(systems._state_space_analysis(ctx, a_path=str(a_path), b_path=str(b_path)))
    assert "error" not in res
    assert res["ctrb_rank"] == 2 and res["controllable"] is True


def test_state_space_shape_validation(ctx: AppContext):
    non_square = json.loads(systems._state_space_analysis(ctx, a=[[1, 0, 0], [0, 1, 0]], b=[[1], [0]]))
    assert "error" in non_square and "square" in non_square["error"]

    bad_b = json.loads(systems._state_space_analysis(ctx, a=[[0, 1], [0, 0]], b=[[1], [0], [0]]))
    assert "error" in bad_b and "rows" in bad_b["error"]

    bad_c = json.loads(systems._state_space_analysis(ctx, a=[[0, 1], [0, 0]], b=[[0], [1]], c=[[1, 0, 0]]))
    assert "error" in bad_c and "columns" in bad_c["error"]

    missing_b = json.loads(systems._state_space_analysis(ctx, a=[[0, 1], [0, 0]]))
    assert "error" in missing_b and "B" in missing_b["error"]


def test_statespace_helper_matrices():
    ctrb = statespace.controllability_matrix([[0, 1], [0, 0]], [[0], [1]])
    assert ctrb.shape == (2, 2)
    obsv = statespace.observability_matrix([[0, 1], [0, 0]], [[1, 0]])
    assert obsv.shape == (2, 2)
    analysis = statespace.analyze([[1, 0], [0, 2]], [[1], [0]])
    assert analysis.ctrb_rank == 1 and analysis.controllable is False


# --------------------------------------------------------------------------- #
# Registration surface


def test_register_exposes_advanced_tools(ctx: AppContext):
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("advsystems-test")
    systems.register(mcp, ctx)
    names = {tool.name for tool in asyncio.run(mcp.list_tools())}
    assert {"group_delay", "state_space_analysis"} <= names
    # The five original systems tools remain registered alongside the two new ones.
    assert {"lti_response", "pole_zero", "bode", "sampling_check", "convolve_signals"} <= names

"""Systems-pack tests (ELE301): golden step metrics, Bode margins, aliasing
verdicts, and convolution identities — no engine, no MCP client. Tool bodies are
exercised through their module-level ``_impl`` functions with a tmp AppContext
(the registered closures delegate to them 1:1).
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import numpy as np
import pytest

from dsp_server.engine.ply import write_engine_ply
from dsp_server.toolsets import AppContext, systems
from synth import make_cylinder

RISE_LN9 = math.log(9.0)  # exact 10-90% rise of a first-order lag, in time constants
SETTLE_2PCT = math.log(50.0)  # ~3.912
OVERSHOOT_Z05 = 100.0 * math.exp(-math.pi * 0.5 / math.sqrt(0.75))  # 16.303%
PEAK_TIME_Z05 = math.pi / math.sqrt(0.75)  # 3.6276 s for wn=1, zeta=0.5


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
# (1) lti_response — CT step metrics


def test_lti_response_first_order_step(ctx: AppContext):
    res = json.loads(systems._lti_response(ctx, num=[1.0], den=[1.0, 1.0]))
    assert "error" not in res
    assert res["stable"] is True
    assert abs(res["dc_gain"] - 1.0) < 1e-9
    step = res["step"]
    assert step is not None
    assert abs(step["final_value"] - 1.0) < 0.01
    assert abs(step["rise_time_10_90"] - RISE_LN9) / RISE_LN9 < 0.02
    assert abs(step["settling_time_2pct"] - SETTLE_2PCT) / SETTLE_2PCT < 0.03
    assert step["overshoot_pct"] < 0.5
    with np.load(res["response_path"]) as z:
        assert set(z.files) == {"t", "u", "y"}
        assert z["t"].size == 500 and z["y"].size == 500


def test_lti_response_second_order_step(ctx: AppContext):
    res = json.loads(systems._lti_response(ctx, num=[1.0], den=[1.0, 1.0, 1.0]))
    assert "error" not in res
    step = res["step"]
    assert abs(step["overshoot_pct"] - OVERSHOOT_Z05) / OVERSHOOT_Z05 < 0.03
    assert abs(step["peak_time"] - PEAK_TIME_Z05) / PEAK_TIME_Z05 < 0.03


# --------------------------------------------------------------------------- #
# (2) lti_response — DT impulse and instability


def test_lti_response_dt_impulse_geometric(ctx: AppContext):
    res = json.loads(
        systems._lti_response(ctx, num=[1.0], den=[1.0, -0.5], domain="z", dt=0.1, input_kind="impulse")
    )
    assert "error" not in res
    assert res["stable"] is True
    assert res["dt"] == 0.1
    with np.load(res["response_path"]) as z:
        y = z["y"]
    assert np.allclose(y[:5], 0.5 ** np.arange(5), rtol=1e-9, atol=0.0)


def test_lti_response_dt_unstable_pole(ctx: AppContext):
    res = json.loads(
        systems._lti_response(ctx, poles=[[1.1, 0.0]], gain=1.0, domain="z", dt=0.1, input_kind="impulse")
    )
    assert "error" not in res
    assert res["stable"] is False


def test_lti_response_rejects_ambiguous_form(ctx: AppContext):
    res = json.loads(systems._lti_response(ctx))
    assert "error" in res
    assert "num" in res["hint"] and "zeros" in res["hint"]


# --------------------------------------------------------------------------- #
# (3) pole_zero — CT dynamics and DT time constants


def test_pole_zero_ct_second_order(ctx: AppContext):
    res = json.loads(systems._pole_zero(ctx, num=[1.0], den=[1.0, 2.0, 5.0]))
    assert "error" not in res
    assert res["stable"] is True
    assert res["causal_roc"] == "Re(s) > -1"
    upper = next(p for p in res["poles"] if p["im"] > 0)
    assert abs(upper["re"] + 1.0) < 1e-6 and abs(upper["im"] - 2.0) < 1e-6
    assert abs(upper["damping_ratio"] - 1.0 / math.sqrt(5.0)) < 1e-3
    assert abs(upper["natural_freq_rad_s"] - math.sqrt(5.0)) < 1e-3
    assert res["proper"] is True and res["minimum_phase"] is True


def test_pole_zero_dt_pole(ctx: AppContext):
    res = json.loads(systems._pole_zero(ctx, poles=[[0.9, 0.0]], gain=1.0, domain="z", dt=0.1))
    assert "error" not in res
    assert res["causal_roc"] == "|z| > 0.9"
    pole = res["poles"][0]
    assert abs(pole["magnitude"] - 0.9) < 1e-9
    assert abs(pole["time_constant_s"] - (-0.1 / math.log(0.9))) < 1e-3
    assert res["stable"] is True


# --------------------------------------------------------------------------- #
# (4) bode — response goldens and stability margins


def test_bode_first_order(ctx: AppContext):
    res = json.loads(systems._bode(ctx, num=[1.0], den=[1.0, 1.0]))
    assert "error" not in res
    with np.load(res["arrays_path"]) as z:
        w, mag, ph = z["w"], z["mag_db"], z["phase_deg"]
    mag_at_1 = float(np.interp(0.0, np.log10(w), mag))  # log10(1) = 0
    ph_at_1 = float(np.interp(0.0, np.log10(w), ph))
    assert abs(mag_at_1 + 3.0103) < 0.05
    assert abs(ph_at_1 + 45.0) < 0.5
    assert abs(res["hf_slope_db_per_decade"] + 20.0) < 1.5
    assert res["gain_margin_db"] is None and res["phase_margin_deg"] is None
    assert abs(res["dc_gain_db"]) < 0.01
    assert abs(res["bandwidth_3db"] - 1.0) < 0.05


def test_bode_margins_golden(ctx: AppContext):
    # open loop L(s) = 1/(s(s+1)(s+2)): GM = 20*log10(6) = 15.563 dB at w = sqrt(2),
    # PM = 53.4 deg at w = 0.4457 rad/s.
    res = json.loads(systems._bode(ctx, num=[1.0], den=[1.0, 3.0, 2.0, 0.0]))
    assert "error" not in res
    assert abs(res["gain_margin_db"] - 15.563) < 0.1
    assert abs(res["gain_margin_freq"] - math.sqrt(2.0)) < 0.02
    assert abs(res["phase_margin_deg"] - 53.4) < 0.5
    assert abs(res["phase_margin_freq"] - 0.4457) < 0.01
    assert res["dc_gain_db"] is None  # pole at the origin


# --------------------------------------------------------------------------- #
# (5) sampling_check — file lane, guards, dump lane


def _tone_npz(tmp_path: Path, f_hz: float = 80.0, dt: float = 0.001, n: int = 2000) -> Path:
    t = np.arange(n) * dt
    path = tmp_path / "tone.npz"
    np.savez(path, y=np.sin(2.0 * np.pi * f_hz * t), dt=np.asarray(dt))
    return path


def test_sampling_check_aliased_then_safe(ctx: AppContext, tmp_path: Path):
    tone = _tone_npz(tmp_path)

    res = json.loads(systems._sampling_check(ctx, fs=100.0, series_path=str(tone)))
    assert "error" not in res
    assert res["freq_unit"] == "Hz" and res["dt"] == 0.001
    assert res["verdict"] == "aliased"
    assert res["energy_fraction_above_nyquist"] > 0.3
    top = res["folded_peaks"][0]
    assert abs(top["f_true"] - 80.0) < 1.0
    assert abs(top["f_alias"] - 20.0) < 1.0
    assert top["aliased"] is True
    assert 75.0 < res["bandwidth_99pct"] < 85.0

    res2 = json.loads(systems._sampling_check(ctx, fs=200.0, series_path=str(tone)))
    assert "error" not in res2
    assert res2["verdict"] == "safe"


def test_sampling_check_insufficient_resolution_guard(ctx: AppContext, tmp_path: Path):
    tone = _tone_npz(tmp_path)  # reference Nyquist = 500 Hz
    res = json.loads(systems._sampling_check(ctx, fs=950.0, series_path=str(tone)))
    assert "error" in res
    assert "insufficient_resolution" in res["error"]


def test_sampling_check_dt_missing_toolerror(ctx: AppContext, tmp_path: Path):
    t = np.arange(512) * 0.001
    csv = tmp_path / "tone.csv"
    np.savetxt(csv, np.sin(2.0 * np.pi * 80.0 * t))

    res = json.loads(systems._sampling_check(ctx, fs=100.0, series_path=str(csv)))
    assert "error" in res
    assert "pass dt=" in res["error"]

    ok = json.loads(systems._sampling_check(ctx, fs=100.0, series_path=str(csv), dt=0.001))
    assert "error" not in ok
    assert ok["verdict"] == "aliased"


def test_sampling_check_dump_lane(ctx: AppContext, tmp_path: Path):
    dump = make_cylinder(corrugation_amp=0.002, corrugation_freq_cpm=60.0)
    ply_path = tmp_path / "cyl.ply"
    write_engine_ply(ply_path, dump)

    res = json.loads(systems._sampling_check(ctx, fs=100.0, dump=str(ply_path), joint="5"))
    assert "error" not in res
    assert res["freq_unit"] == "cycles/m"
    assert res["n_samples"] == 256
    assert res["verdict"] == "aliased"
    top = res["folded_peaks"][0]
    assert 50.0 < top["f_true"] < 70.0
    assert abs(top["f_alias"] - (100.0 - top["f_true"])) < 1e-3
    assert top["aliased"] is True

    bad = json.loads(systems._sampling_check(ctx, fs=100.0, dump=str(ply_path), joint="noSuchBone"))
    assert "error" in bad
    assert "noSuchBone" in bad["error"]
    assert "histogram" in bad["hint"]


# --------------------------------------------------------------------------- #
# (6) convolve_signals — CT/DT scaling, correlation lag, mixed dt


def test_convolve_rects_unit_triangle(ctx: AppContext):
    rect = [1.0] * 100  # 1 s at dt = 0.01

    res = json.loads(systems._convolve_signals(ctx, values_a=rect, values_b=rect, dt=0.01))
    assert "error" not in res
    assert abs(res["y_peak"] - 1.0) < 0.01  # CT rect*rect -> unit triangle
    assert abs(res["t_at_peak"] - 1.0) <= 0.02
    assert res["n_out"] == 199  # support ~2 s
    with np.load(res["result_path"]) as z:
        assert set(z.files) == {"t", "y"}

    res_d = json.loads(
        systems._convolve_signals(ctx, values_a=rect, values_b=rect, dt=0.01, scale="discrete")
    )
    assert res_d["y_peak"] == 100.0  # raw DT sum


def test_correlate_shifted_sine(ctx: AppContext):
    t = np.arange(0.0, 2.0, 0.01)  # 200 samples, 1 Hz sine
    a = np.sin(2.0 * np.pi * t)
    b = np.sin(2.0 * np.pi * (t - 0.25))  # b lags a by 0.25 s
    res = json.loads(
        systems._convolve_signals(
            ctx,
            values_a=a.tolist(),
            values_b=b.tolist(),
            dt=0.01,
            mode="correlate",
            normalize=True,
        )
    )
    assert "error" not in res
    # scipy convention c[k] = sum a[n]*b[n-k]: a delayed copy in b peaks at negative lag
    assert abs(res["lag_at_peak"] - (-0.25)) < 0.011
    assert res["peak_corrcoef"] > 0.99
    with np.load(res["result_path"]) as z:
        assert set(z.files) == {"lag", "y"}


def test_convolve_mixed_dt_resamples_b(ctx: AppContext, tmp_path: Path):
    path_a = tmp_path / "a.npz"
    path_b = tmp_path / "b.npz"
    np.savez(path_a, y=np.ones(100), dt=np.asarray(0.01))  # 1 s rect at dt=0.01
    np.savez(path_b, y=np.ones(50), dt=np.asarray(0.02))  # 1 s rect at dt=0.02

    res = json.loads(systems._convolve_signals(ctx, path_a=str(path_a), path_b=str(path_b)))
    assert "error" not in res
    assert res["resampled_b"] is True
    assert res["dt"] == 0.01

    base = json.loads(systems._convolve_signals(ctx, values_a=[1.0] * 100, values_b=[1.0] * 100, dt=0.01))
    assert abs(res["y_peak"] - base["y_peak"]) <= 0.02 * base["y_peak"]


def test_convolve_dt_missing_toolerror(ctx: AppContext):
    res = json.loads(systems._convolve_signals(ctx, values_a=[1.0] * 16, values_b=[1.0] * 16))
    assert "error" in res
    assert "pass dt=" in res["error"]


# --------------------------------------------------------------------------- #
# Registration surface


def test_register_exposes_all_tools(ctx: AppContext):
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("systems-test")
    systems.register(mcp, ctx)
    names = {tool.name for tool in asyncio.run(mcp.list_tools())}
    assert names == {
        "lti_response",
        "pole_zero",
        "bode",
        "sampling_check",
        "convolve_signals",
        "group_delay",
        "state_space_analysis",
    }

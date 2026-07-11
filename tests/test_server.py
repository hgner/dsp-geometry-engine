"""MCP-layer tests: toolset registration + the tool bodies end-to-end against the
stub engine. No MCP client is needed — registration is asserted via the server's
async ``list_tools`` and the tool bodies are exercised through their module-level
``_impl`` functions (the registered closures delegate to them 1:1).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from dsp_server.server import create_server
from dsp_server.toolsets import AppContext, geometry

# Full 41-tool inventory across the nine packs (curriculum order + rendering).
EXPECTED_BY_PACK = {
    "geometry": {
        "extract_mesh_telemetry",
        "analyze_corrugation",
        "compare_geometry_signals",
        "localize_defect",
        "lbs_differential",
        "score_bake",
        "analyze_mesh_topology",
    },
    "imaging": {"compare_depth_renders", "enhance_image", "filter_image", "segment_image", "restore_image"},
    "stats": {"describe", "fit_distribution", "hypothesis_test", "regression_fit", "compare_dump_ripples"},
    "engmath": {
        "solve_ode",
        "laplace_transform",
        "linear_algebra",
        "residues_and_integrals",
        "fourier_series",
    },
    "systems": {
        "lti_response",
        "pole_zero",
        "bode",
        "sampling_check",
        "convolve_signals",
        "group_delay",
        "state_space_analysis",
    },
    "ml": {"feature_engineer_dump", "cluster", "reduce_dims", "classify_eval", "predict"},
    "netqueue": {"queueing_calc", "little_law", "erlang_blocking"},
    "os": {"schedule_sim", "page_replacement_sim", "bankers_check"},
    "rendering": {"verify_brdf_energy"},
}
EXPECTED_TOOLS = set().union(*EXPECTED_BY_PACK.values())


def _tool_names(mcp) -> set[str]:
    return {tool.name for tool in asyncio.run(mcp.list_tools())}


def test_all_toolsets_register_expected_tools(stub_engine_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DSP_TOOLSETS", raising=False)
    names = _tool_names(create_server())
    assert names == EXPECTED_TOOLS
    assert len(names) == 41  # no duplicate names across packs


def test_toolsets_env_filters_packs(stub_engine_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DSP_TOOLSETS", "imaging")
    assert _tool_names(create_server()) == EXPECTED_BY_PACK["imaging"]
    monkeypatch.setenv("DSP_TOOLSETS", "netqueue,os")
    assert _tool_names(create_server()) == EXPECTED_BY_PACK["netqueue"] | EXPECTED_BY_PACK["os"]


def test_geometry_pipeline_against_stub(stub_engine_env):
    ctx = AppContext.from_config()

    # --- tool 1: extract_mesh_telemetry -------------------------------------
    telemetry = json.loads(geometry._extract_mesh_telemetry(ctx, pose_clip="clip-cin-walktalk"))
    assert "error" not in telemetry
    assert telemetry["vertex_count"] == 64 * 16
    assert telemetry["face_count"] > 0
    assert telemetry["clip_id"] == "clip-cin-walktalk"
    assert telemetry["bone_map"]["5"] == "armLowerL"
    assert "engine_stale" in telemetry
    assert any("armLowerL" in key for key in telemetry["arm_chain_histogram"])
    ply_path = telemetry["ply_path"]
    assert Path(ply_path).is_file()
    assert telemetry["palette_path"] is not None and Path(telemetry["palette_path"]).is_file()

    # --- tool 2: analyze_corrugation (stub injects 60 cy/m on armLowerL) ----
    report = json.loads(geometry._analyze_corrugation(ctx, ply_path))
    assert "error" not in report
    assert report["roughness"]["corrugated"] is True
    one_bin_cpm = 1.0 / report["t_span_m"]
    assert abs(report["roughness"]["dominant_freq_cpm"] - 60.0) <= 2.0 * one_bin_cpm
    assert report["extent_band_energy_fraction"] > 0.0

    # --- tool 3: compare_geometry_signals, rest vs posed of the same dump ---
    comparison = json.loads(geometry._compare_geometry_signals(ctx, ply_path, "rest", ply_path, "posed"))
    assert "error" not in comparison
    rest_ripple = comparison["roughness_a"]["rel_ripple"]
    posed_ripple = comparison["roughness_b"]["rel_ripple"]
    assert posed_ripple > rest_ripple * 5.0
    assert comparison["displacement"] is not None
    assert comparison["displacement"]["max_m"] > 0.0

    # --- tool 4: localize_defect --------------------------------------------
    localized = json.loads(geometry._localize_defect(ctx, ply_path))
    assert "error" not in localized
    assert localized["top"], "expected at least one ranked segment"
    assert localized["top"][0]["joint_name"] == "armLowerL"

    # --- tool 5: lbs_differential (identity palette -> pure LBS == rest) ----
    differential = json.loads(geometry._lbs_differential(ctx, ply_path))
    assert "error" not in differential
    assert differential["verdict"] == "deformer-band"
    assert differential["contributing_bones"] == ["armLowerL"]
    assert differential["roughness_engine"]["corrugated"] is True
    assert differential["roughness_pure_lbs"]["corrugated"] is False

    # --- ToolError path: unknown joint -> hint carries the joint histogram --
    failure = json.loads(geometry._analyze_corrugation(ctx, ply_path, joint="noSuchBone"))
    assert "error" in failure
    assert "noSuchBone" in failure["error"]
    assert "histogram" in failure["hint"]
    assert "armLowerL" in failure["hint"]

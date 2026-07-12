"""``dsp-tool``: run ONE registered gate tool from the command line and print its JSON.

A thin CLI over the tools' module-level ``_impl`` functions — the exact callables the
MCP server registers and the tests exercise 1:1 — so a NON-MCP consumer (e.g. proje8,
which spawns helpers and parses JSON rather than speaking MCP) can invoke a single
comparison-gate tool without an MCP client:

    uv run dsp-tool <tool_name> --args-json '<json-object>'

prints the tool's JSON envelope on stdout (exit 0), or a ``{"error": ...}`` object and
a non-zero exit on an unknown tool / malformed args. Deterministic, no network — the
same contract the MCP tools honor. Exposes the comparison-gate tools (see
docs/COMPARISON-GATE.md); extend :func:`_dispatch` to surface more.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable

from dsp_server.toolsets import AppContext


def _dispatch() -> dict[str, Callable[..., str]]:
    """Map tool name -> its ``_impl(ctx, **args) -> json_str``. Lazy-imported to keep
    startup cheap; these are the same functions the MCP closures delegate to."""
    from dsp_server.toolsets import geometry, imaging, perceptual, video

    return {
        # video pack — temporal + geometric gates
        "evaluate_spatiotemporal_frequencies": video._evaluate_spatiotemporal_frequencies,
        "verify_motion_consistency": video._verify_motion_consistency,
        "verify_camera_projection": video._verify_camera_projection,
        "analyze_photometric_consistency": video._analyze_photometric_consistency,
        "evaluate_occlusion_boundaries": video._evaluate_occlusion_boundaries,
        # imaging pack — spatial multiresolution
        "compare_wavelet_signatures": imaging._compare_wavelet_signatures,
        # perceptual pack — FR-VQA
        "evaluate_perceptual_similarity": perceptual._evaluate_perceptual_similarity,
        "verify_identity_coherence": perceptual._verify_identity_coherence,
        # geometry pack — structure
        "score_bake": geometry._score_bake,
    }


def run_tool(tool: str, args: dict) -> tuple[str, int]:
    """``(json_text, exit_code)``. Never raises — mirrors the tools' no-raise contract.

    A known tool returns its JSON envelope with exit 0 (even a tool-level ToolError is
    valid JSON on stdout, exit 0 — the caller inspects the ``error`` field). An unknown
    tool or bad argument names return an ``{"error": ...}`` object with a non-zero exit.
    """
    dispatch = _dispatch()
    fn = dispatch.get(tool)
    if fn is None:
        return json.dumps({"error": f"unknown tool '{tool}'", "hint": f"available: {sorted(dispatch)}"}), 2
    ctx = AppContext.from_config()
    try:
        return fn(ctx, **args), 0
    except TypeError as exc:  # wrong/missing kwarg names for this tool (bound before the _impl's own try)
        return json.dumps({"error": f"bad args for '{tool}': {exc}"}), 2
    except Exception as exc:  # defensive: the _impls catch their own; the CLI must never crash
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}), 1


def _emit(text: str) -> None:
    # stdout IS the interface for this CLI (unlike the library modules, which the T20
    # lint keeps off stdout); write directly rather than via the banned print().
    sys.stdout.write(text + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dsp-tool", description="Run one DSP comparison-gate tool and print its JSON."
    )
    parser.add_argument("tool", help="tool name (e.g. verify_motion_consistency)")
    parser.add_argument("--args-json", default="{}", help="JSON object of the tool's arguments")
    ns = parser.parse_args(argv)
    try:
        args = json.loads(ns.args_json)
    except json.JSONDecodeError as exc:
        _emit(json.dumps({"error": f"--args-json is not valid JSON: {exc}"}))
        return 2
    if not isinstance(args, dict):
        _emit(json.dumps({"error": "--args-json must be a JSON object of the tool's arguments"}))
        return 2
    text, code = run_tool(ns.tool, args)
    _emit(text)
    return code


if __name__ == "__main__":
    sys.exit(main())

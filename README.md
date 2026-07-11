# DSP-Geometry-Engine

An MCP "DSP microscope" for engine geometry: it turns 3D vertex dumps from the proje7-engine C++
engine into 1D signals, runs FFT/roughness analysis on them, and returns compact JSON summaries to
Claude over MCP (stdio locally, streamable HTTP in the cloud). Built to root-cause the forearm
corrugation defect on the imported `cc0_male_rigged3` character.

## Quickstart

```powershell
uv sync            # install (Python 3.12, locked deps)
uv run pytest      # full test suite — runs against a bundled stub engine, no C++ needed
```

MCP registration is automatic in Claude Code: the committed `.mcp.json` starts the server via
`uv run dsp-server` when you open this project. For Claude Desktop, run
`scripts/register-claude-desktop.ps1`.

## Tools

| Tool | One-liner |
| --- | --- |
| `extract_mesh_telemetry` | Run the engine dump CLI on a character/clip; return paths, counts, bone map, deformer push stats. |
| `analyze_corrugation` | FFT/roughness report for one joint segment (cycles/meter, wavelength, ridge count, verdict). |
| `compare_geometry_signals` | Compare two channels/dumps (e.g. rest vs posed) — spectra plus per-vertex displacement stats. |
| `localize_defect` | Per-joint roughness ranking across the whole mesh: forearm-only or systemic, in one call. |
| `lbs_differential` | Pure-numpy LBS vs engine dump differential — isolates weight bugs from deformer bugs. |
| `compare_depth_renders` | 2D spectral/SSIM comparison of depth/AOV PNGs (render-space cross-validation). |

All tools return small JSON summaries; arrays stay on disk under `data/`.

## Docs

- `docs/ARCHITECTURE.md` — data flow, two-repo layout, tool surface, PLY/stderr contract, isolation logic.
- `docs/DEVELOPMENT.md` — setup, lint/test, engine build lane, MCP registration, adding a course pack.
- `docs/DEPLOYMENT.md` — container image, AWS (App Runner / ECS Fargate), env var reference, security.
- `llms.txt` — canonical rules for LLM agents working in this repo (imported by `CLAUDE.md`).

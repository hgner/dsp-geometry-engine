# Development

## Prerequisites

- uv (0.11+). uv manages the interpreter — the project pins Python 3.12 via `.python-version`
  (system Python version is irrelevant).
- Windows only for the real-engine lane: an MSVC/vcpkg toolchain and a proje7-engine checkout, whose
  location comes from `DSP_ENGINE_ROOT` (a single exe can be pinned with `DSP_ENGINE_CLI` instead).
  The engine is a separate private repository and is not required to work on this one.
- No engine is needed for development: 48 of the 49 DSP tools are pure Python over files on disk, and
  the one that is not (`extract_mesh_telemetry`) runs against `tests/stub_engine.py` — which is what
  the whole test suite and CI use (conftest sets `DSP_ENGINE_CLI=tests/stub_engine.py`). Likewise no
  Blender: the body-mesh tests use `tests/stub_blender.py` + `tests/stub_character_bake.py`.

## Setup

```powershell
uv sync          # creates .venv from uv.lock (locked, reproducible)
```

## Lint and test

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Rules that ruff enforces by construction: line length 110, sorted imports, modern typing,
and T20 — no `print()` under `src/` (stdout is MCP JSON-RPC; tests are exempt).

## Engine lane (local-only)

- `scripts/build-engine.ps1` — thin delegator to `proje8\scripts\build-engine-cli.ps1` with
  `-Preset windows-msvc-static-md-release -Target layered_field_dump_cli` (~1-3 min, tool TU only).
  Stale policy: when the rtx preset is chosen, the `-Target` argument is omitted so ALL rtx tools
  relink together (FEAT-CG-2 stale-sibling policy) — never relink a single rtx tool in isolation.
- `scripts/verify-engine.ps1` — smoke against the real exe (600 s timeouts, prints
  `VERIFY_OK`/`VERIFY_FAIL`): rest + two posed procedural dumps, exit 0, `bone-map:` present on
  stderr, and posed-vs-posed vertex counts equal (the rest iso-surface is a different pipeline —
  never compare its count against a posed dump). Post-patch it adds a rigged3 posed dump + palette
  sidecar check (boneCount == bone-map entries), skipped with a warning when the baked character
  asset the script points at is absent — it lives in the operator's local character library, not in
  this repository.
- The DSP MCP server itself never builds the engine: `engine_stale` in tool responses is a warning only.
  Rebuild explicitly with the script above.

## MCP registration

The project registers two independent child processes. A shared config file does not make either
server a plugin, toolset, or subprocess of the other.

| Registered name | Command | Configuration family |
| --- | --- | --- |
| `dsp-geometry-engine` | `uv --directory <repo> run dsp-server` | `DSP_*` |
| `blender-body-mesh` | `uv --directory <repo> run bodymesh-server` | `BODYMESH_*` |

Neither config file is committed — both are gitignored so nobody inherits another machine's absolute
paths. Copy the template once and edit the paths in it:

```powershell
Copy-Item .mcp.json.example .mcp.json                    # Claude Code
Copy-Item .codex/config.toml.example .codex/config.toml  # Codex
```

- Codex: automatic for this trusted project once `.codex/config.toml` exists — it launches
  both commands as separate MCP servers. Restart Codex and open a fresh task after changing the
  registration; verify both names with `/mcp` in Codex or `codex mcp list` from the repository root.
- Blender body-mesh: its local-only `bodymesh-server` invokes
  the direct Blender 4.2 executable in background mode, uses the installed MPFB extension, reads the
  55-bone assets from `BODYMESH_ENGINE_SKELETON_DIR`, and runs `BODYMESH_CHARACTER_BAKE_EXE`. Never
  launch `blender-launcher.exe` from automation and never expose this private-photo/process surface
  over HTTP. See `docs/BODY-MESH-MCP.md`.
- Claude Code: automatic once `.mcp.json` exists — it registers both names and launches two stdio
  processes. Do not also start either command by hand inside a session.
- Claude Desktop: run `scripts/register-claude-desktop.ps1`. It creates `%APPDATA%\Claude` if
  missing, merges both registrations (never clobbers) into `claude_desktop_config.json`, and resolves
  the absolute `uv.exe` path via `(Get-Command uv).Source` because Desktop does not inherit the full
  PATH.
  The registration is inert until Claude Desktop is actually installed — safe to run early.

The client performs cross-server orchestration by passing returned artifact paths into later tool
calls. Neither MCP host launches, discovers, or sends MCP requests to the other. The body-mesh bridge
does reuse `dsp_server.engine.ply` as an ordinary in-process parser; this shared library import does
not start the DSP MCP host or merge the tool registries.

## Adding a course pack (the extensibility contract)

1. new math module under `engine/` (pure functions, numpy in/out, no I/O);
2. `toolsets/<name>.py` with `register(mcp, ctx)` and pydantic schemas — tools return JSON
   summaries only, arrays to `data/`;
3. one registry line in `toolsets/__init__.py`;
4. `tests/test_<name>.py` with synthetic-signal golden tests (no engine dependency);
5. a rules section appended to llms.txt.

New packs inherit the shared kernel for free: the `engine/` math modules, `ply.py`/npz caching,
`Signal1D`, `plots.py`, the `ToolError` envelope, and the engine bridge. Packs are toggled per
client via `DSP_TOOLSETS` (comma list; unset = all).

## The RCA prompt (Phase 5 — concluded 2026-07-11)

The engine patch (`--character`/`--palette-out`/`--weights`) landed and this prompt drove the
root-cause session on the first rigged3 vertex dumps:

> "Run `extract_mesh_telemetry` on cc0_male_rigged3 for clip-cin-stand-attention and
> clip-cin-walktalk. `localize_defect` on the flexion dump; `analyze_corrugation` on armLowerL;
> `compare_geometry_signals` rest-vs-posed; then `lbs_differential` with the palette sidecar.
> Correlate dominant_wavelength_m against the forearm capsule geometry from `--list-sources` to
> name the guilty weld/weight array."

It reached a verdict (llms.txt rule 8): the forearm ripple is 118-123 cy/m (~8.4 mm, the mesh
edge-loop spacing, ~20 dB prominence) and is already present in the bind positions, so engine capsule
welds, flexion deformers, and retargeted weights are exonerated — the defect enters in the upstream
Blender retarget/bake. Keep the prompt as the template for the next import: the dual-telemetry
differential (pure-numpy LBS vs engine dump) is the reusable method, not a one-off.

## CI overview

- GitHub Actions (`.github/workflows/ci.yml`): Python-only, runs on `ubuntu-latest` (fast lane) and
  `windows-latest` (realism lane). Steps: `uv sync --locked`, `uv run ruff check .`,
  `uv run ruff format --check .`, `uv run pytest`. conftest points `DSP_ENGINE_CLI` at
  `tests/stub_engine.py`, so the whole bridge path is exercised without any C++. The ubuntu lane
  also does a `docker build` (build only, no push) to keep the Dockerfile honest.
- Real-engine verification is deliberately NOT in CI: it is the local PowerShell lane
  (`scripts/verify-engine.ps1`, `scripts/verify-local.ps1`).
- Real Blender/MPFB/engine-bake verification is also local-only. CI uses `tests/stub_blender.py` and
  `tests/stub_character_bake.py`; the body-mesh bridge tests still exercise request validation,
  subprocess/result handling, artifact confinement, exact 55-bone validation, skin/tangent gates,
  and DSP-PLY parsing without installing Blender or the C++ tools.

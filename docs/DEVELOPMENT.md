# Development

## Prerequisites

- uv (0.11+). uv manages the interpreter — the project pins Python 3.12 via `.python-version`
  (system Python version is irrelevant).
- Windows only for the real-engine lane: MSVC/vcpkg toolchain via proje7-engine, and the engine
  worktree at `C:\Users\hgner\hakantest\proje7-engine` (or point `DSP_ENGINE_ROOT` elsewhere).
- No engine is needed for development: the test suite runs entirely against `tests/stub_engine.py`.

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
  sidecar check (boneCount == bone-map entries), skipped with a warning if the `D:` asset is absent.
- The MCP server itself never builds the engine: `engine_stale` in tool responses is a warning only.
  Rebuild explicitly with the script above.

## MCP registration

- Codex: automatic for this trusted project. The committed `.codex/config.toml` launches
  `uv --directory <repo> run dsp-server` with `MPLBACKEND=Agg`, `DSP_ENGINE_ROOT`, and
  `DSP_DATA_DIR` set. Restart Codex and open a fresh task after changing the registration; verify
  it with `/mcp` in Codex or `codex mcp list` from the repository root.
- Blender body-mesh: the same config registers the separate local-only `bodymesh-server`. It invokes
  the direct Blender 4.2 executable in background mode and uses the installed MPFB extension. Never
  launch `blender-launcher.exe` from automation and never expose this private-photo/process surface
  over HTTP. See `docs/BODY-MESH-MCP.md`.
- Claude Code: automatic. The committed `.mcp.json` launches `uv --directory <repo> run dsp-server`
  with `MPLBACKEND=Agg`, `DSP_ENGINE_ROOT`, and `DSP_DATA_DIR` set. Do not also start the server by
  hand inside a session.
- Claude Desktop: run `scripts/register-claude-desktop.ps1`. It creates `%APPDATA%\Claude` if
  missing, merges (never clobbers) into `claude_desktop_config.json`, and resolves the absolute
  `uv.exe` path via `(Get-Command uv).Source` because Desktop does not inherit the full PATH.
  The registration is inert until Claude Desktop is actually installed — safe to run early.

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

## The RCA prompt (Phase 5)

Once the engine patch (`--character`/`--palette-out`/`--weights`) has landed, this is the prompt
that drives the actual root-cause session:

> "Run `extract_mesh_telemetry` on cc0_male_rigged3 for clip-cin-stand-attention and
> clip-cin-walktalk. `localize_defect` on the flexion dump; `analyze_corrugation` on armLowerL;
> `compare_geometry_signals` rest-vs-posed; then `lbs_differential` with the palette sidecar.
> Correlate dominant_wavelength_m against the forearm capsule geometry from `--list-sources` to
> name the guilty weld/weight array."

## CI overview

- GitHub Actions (`.github/workflows/ci.yml`): Python-only, runs on `ubuntu-latest` (fast lane) and
  `windows-latest` (realism lane). Steps: `uv sync --locked`, `uv run ruff check .`,
  `uv run ruff format --check .`, `uv run pytest`. conftest points `DSP_ENGINE_CLI` at
  `tests/stub_engine.py`, so the whole bridge path is exercised without any C++. The ubuntu lane
  also does a `docker build` (build only, no push) to keep the Dockerfile honest.
- Real-engine verification is deliberately NOT in CI: it is the local PowerShell lane
  (`scripts/verify-engine.ps1`, `scripts/verify-local.ps1`).
- Real Blender/MPFB verification is also local-only. CI uses `tests/stub_blender.py`; the body-mesh
  bridge tests still exercise request validation, subprocess/result handling, artifact confinement,
  and DSP-PLY parsing without installing Blender.

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

## Release (PyPI)

`.github/workflows/release.yml` publishes the `dsp-geometry-engine` distribution to PyPI using
**Trusted Publishing (OIDC)**. There is no PyPI API token anywhere in this repository, and none
should ever be added: the workflow mints a short-lived token from PyPI at publish time.

Pipeline, in order — `verify` (tag equals `[project].version`) -> `test` (the same lint/format/pytest
matrix as `ci.yml`, on ubuntu **and** windows) -> `build` (`uv build`, a metadata check, plus a smoke
test that installs the wheel into a throwaway environment and asserts every packaged module imports
and all four console scripts are registered) -> `publish` -> GitHub Release. Nothing is built until
the tag has been checked, because **a version published under the wrong number is burnt permanently**:
yanking a PyPI release hides it, it does not free the number for re-upload.

The smoke test is the only check in this repository that sees what a `pip install` actually gets.
Everything else — `ci.yml`, `uv sync --locked`, the pytest run — resolves through `uv.lock`, but a
consumer installing from PyPI has no lockfile and gets the newest version each `Requires-Dist`
specifier allows. So the smoke step deliberately uses `uv run --no-project --with <wheel>`, which
re-resolves the wheel's own dependencies from scratch, and then imports **every** module in the
package tree (skipping `bodymesh_server.blender_scripts.*`, which import Blender's `bpy` and only
ever run inside Blender). Importing just the top-level packages is not enough: they pull in almost
nothing, so a shallow check stays green while every toolset is broken. This is what caught
`mcp[cli]` needing a `<2` ceiling — `uv.lock` pinned 1.x, but an unbounded specifier resolved
`mcp` 2.0.0, which removed `mcp.server.fastmcp` and broke all four console scripts. **Any dependency
whose import path this project reaches into needs an upper bound, and the smoke test is what proves
it.**

The metadata check is `twine check --strict` (a README that stops rendering becomes a release failure
rather than a broken project page) plus an assertion that the built `METADATA`/`PKG-INFO` carries
`License-Expression` and **zero** `Classifier: License ::` lines. That second half is hand-rolled
because nothing else enforces it: `pyproject.toml` declares a PEP 639 SPDX expression, and hatchling,
`packaging`, and `twine check --strict` all accept a distribution carrying an SPDX expression *and* a
license classifier. Warehouse only rejects the legacy `License:` field against `License-Expression`,
so that combination would upload silently and freeze into the published metadata, which is immutable
per version.

### Owner-only setup on pypi.org (must happen BEFORE the first release)

The project does not exist on PyPI yet, so a normal trusted publisher cannot be attached to it. The
owner must first register a **pending publisher**, which creates the project on first successful
publish. Go to <https://pypi.org/manage/account/publishing/> (account sidebar -> "Publishing"), pick
the **GitHub** tab, and fill in exactly:

| Field | Value | Notes |
| --- | --- | --- |
| PyPI Project Name | `dsp-geometry-engine` | required; the project that gets created on first use |
| Owner | `hgner` | required; GitHub user or org that owns the repo |
| Repository name | `hippocampus` | required; repo name only, no owner prefix, no URL |
| Workflow name | `release.yml` | required; **filename only** — must end in `.yml`/`.yaml`, no `.github/workflows/` prefix, no directories |
| Environment name | `pypi` | optional but strongly recommended; must match the workflow's `environment:` |

Then click **Add**. Repeat the whole thing on <https://test.pypi.org/manage/account/publishing/> with
Environment name `testpypi` — TestPyPI is a separate service with its own accounts and its own
publishers; a publisher registered on PyPI does nothing for TestPyPI.

Things that silently make the first release fail:

- **A pending publisher reserves nothing.** If anyone registers the name `dsp-geometry-engine` on
  PyPI before the first publish lands, the pending publisher is invalidated.
- **Repository name is not checked when you save it.** PyPI validates the *owner* live against
  GitHub's API (and stores the canonical login plus the numeric owner ID), but it cannot check the
  repository, because the repository may be private. A typo is accepted at configuration time and
  only surfaces at publish time as an unhelpful "not a valid token" / invalid-publisher error.
- **Renaming this repository invalidates the publisher.** PyPI matches the OIDC `repository` claim
  (`owner/repo`, case-insensitively) against what was configured. If `hippocampus` is renamed to
  `dsp-geometry-engine`, the publisher must be edited (or deleted and re-added) with the new
  repository name — otherwise the next release fails authentication with no hint that a rename is the
  cause. Do the rename *before* configuring the pending publisher, or fix the publisher immediately
  after. Renaming the *GitHub account* is safe: that is pinned by numeric owner ID.
- **The publishing job must stay in `release.yml`.** PyPI also matches the `job_workflow_ref` claim,
  `hgner/hippocampus/.github/workflows/release.yml@<ref>`. Moving the `uv publish` step into a
  reusable workflow called via `workflow_call` changes that claim to the *called* workflow and breaks
  the match. Reusing another workflow for the *test* jobs is fine; the upload job is not.
- **Environment name has to agree on both sides.** It is compared case-insensitively, but "configured
  on PyPI, absent from the job" and "different name" both fail.

### Owner-only setup on github.com

1. Settings -> Environments -> **New environment**, twice: `pypi` and `testpypi`. They can be empty;
   they only need to exist under those exact names, because the environment name is part of what PyPI
   verifies.
2. On `pypi`, add **Required reviewers** (yourself) to get a manual approval gate — the run pauses
   before the upload and waits. This is the recommended configuration and the reason the workflow uses
   environments at all. Optionally also restrict its deployment branches/tags to `v*`.
3. If an org policy restricts `GITHUB_TOKEN`, confirm workflows may still be granted `contents: write`
   — the last job needs it to create the GitHub Release and attach the sdist and wheel.
4. **Make the repository public before the first upload.** Trusted publishing works fine from a
   private repo, but every entry in `[project.urls]` (Homepage, Repository, Issues, Documentation,
   Changelog) points at `github.com/hgner/hippocampus`, and all five 404 for anonymous visitors while
   the repo is private. Those URLs are frozen into the published metadata for that version — PyPI
   metadata is immutable per release, so this cannot be corrected after upload without cutting a new
   version. The same applies to the relative links in `README.md` (`LICENSE`, `NOTICE`,
   `LICENSES/README.md`, `docs/BODY-MESH-MCP.md`, `docs/COMPARISON-GATE.md`): PyPI renders the README
   as the long description but resolves relative links against `pypi.org`, so they 404 there
   regardless of repo visibility. Rewrite them to absolute `https://github.com/...` URLs if the PyPI
   page should be self-contained.

### TestPyPI rehearsal

Actions -> **release** -> "Run workflow" -> pick a branch -> target `testpypi`. This runs the whole
pipeline (verify, tests, build, smoke test) and uploads to TestPyPI only. No tag is required, and a
dispatch run can never reach real PyPI from a branch: the workflow refuses `target: pypi` unless it is
running from a `v*` tag.

Install the rehearsed artifact to confirm it is usable — dependencies must come from real PyPI,
since TestPyPI does not mirror them:

```powershell
uv venv C:\Temp\relcheck
uv pip install --python C:\Temp\relcheck\Scripts\python.exe `
  --index-url https://test.pypi.org/simple/ `
  --extra-index-url https://pypi.org/simple/ `
  dsp-geometry-engine
```

Repeating a rehearsal at the same version uploads nothing: `uv publish --check-url` skips files the
index already has (without it, TestPyPI answers a re-upload with a 400). To rehearse the same code
again for real, bump to a throwaway version such as `0.1.0.dev1` on the branch.

### Real release

1. Land everything on `main` with CI green.
2. Bump `version` in `pyproject.toml`, run `uv lock`, commit.
3. Tag and push — the tag must be exactly `v` + the pyproject version:
   ```powershell
   git tag -a v0.1.0 -m "v0.1.0"
   git push origin v0.1.0
   ```
4. Approve the `pypi` deployment when Actions asks (if required reviewers are configured).
5. The workflow publishes to PyPI and then creates the GitHub Release for the tag, attaching the same
   sdist and wheel that were uploaded.

If the tag disagrees with `pyproject.toml`, the first job fails immediately with the expected tag
name and **nothing is built or uploaded** — delete the tag, fix one of the two, re-tag. A failed
upload can be retried with "Re-run failed jobs", or by dispatching the workflow with the tag selected
as the ref and target `pypi`; re-runs are safe because `--check-url` skips files PyPI already has.

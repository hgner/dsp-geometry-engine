# Architecture

## Purpose

DSP-Geometry-Engine is an MCP server that gives Claude a quantitative instrument for mesh-space
defect hunting. The C++ engine (proje7-engine) can dump a posed, skinned, deformed character surface
as an ASCII PLY; this server drives that CLI, reduces the 3D vertex cloud to 1D axial signals per
bone segment, applies classical DSP (detrending, zero-phase IIR/FIR filtering, PSD/Welch spectra,
peak prominence), and returns compact JSON verdicts. The concrete mission is the corrugation defect
on the forearm of the imported `cc0_male_rigged3` character: periodic radial rippling along the
forearm under flexion, originally observed in depth renders. The server exists so an LLM can measure
the defect (frequency, wavelength, extent, phase coherence) instead of eyeballing it, and can
attribute it (skin weights vs engine deformers vs render-side) via a dual-telemetry differential.

## Data flow

Arrays never cross the MCP boundary. Every tool response is a small pydantic JSON document; vertex
arrays, spectra, and plots live on disk under `data/`.

```
+--------+   MCP (stdio local /            +------------+
| Claude | <--streamable HTTP cloud------> | dsp-server |
+--------+   compact JSON only             +-----+------+
                                                 | subprocess (cxx_bridge/engine_cli.py)
                                                 v
                                  layered_field_dump_cli.exe        (proje7-engine, build/<preset>/)
                                                 |
                                     data/dumps/*.ply  (+ .meta.json, .palette.json; stderr -> data/logs/)
                                                 |
                                       numpy parse (engine/ply.py, .npz cache in data/cache/)
                                                 |
                              segment select -> axis fit -> 1D axial profile (engine/transforms.py)
                                                 |
                                  FFT / roughness / filters (engine/filters.py, scipy)
                                                 |
                                    pydantic JSON summary  --------> back over MCP to Claude
                                    (optional PNG plot saved to data/plots/, path returned)
```

## Two-repo relationship

| Repo | Role | CI |
| --- | --- | --- |
| `proje10` (this repo) | Python MCP wrapper: bridge, DSP math, tools, tests, docs | GitHub Actions (ruff + pytest + stub engine, ubuntu + windows; docker build) |
| `proje7-engine` | C++ engine worktree (branch `engine`); owns `layered_field_dump_cli` | Local-only: PowerShell scripts (`scripts/verify-engine.ps1`) |

proje10 finds the engine via `DSP_ENGINE_ROOT` (default `C:/Users/hgner/hakantest/proje7-engine`)
and looks for exes under `build/<preset>/` — never `build/bin`. `DSP_ENGINE_CLI` overrides discovery
with a single explicit path (a `.py` path runs via the current interpreter — that is how CI swaps in
`tests/stub_engine.py`). CI never touches the C++ repo; the real-engine lane is local PowerShell.

## Components

| Component | Responsibility |
| --- | --- |
| `src/dsp_server/server.py` | FastMCP entrypoint; builds `AppContext`, registers toolset packs, selects transport (`DSP_TRANSPORT`), optional bearer auth on HTTP. |
| `src/dsp_server/toolsets/__init__.py` | The `TOOLSETS` registry; `DSP_TOOLSETS` filtering. |
| `src/dsp_server/toolsets/geometry.py` | ELE407 1-D lane: tools 1-5 (`register(mcp, ctx)`). |
| `src/dsp_server/toolsets/imaging.py` | ELE407 week-14 2-D lane: tool 6. |
| `src/dsp_server/schemas.py` | Shared pydantic v2 response models; 6-sig-fig rounding; `ToolError` envelope. |
| `src/dsp_server/config.py` | All env knobs (`DSP_*`), data-dir layout, transport selection. |
| `src/dsp_server/plots.py` | Agg-only matplotlib; opt-in PNGs to `data/plots/`, path returned. |
| `src/dsp_server/engine/ply.py` | Engine PLY dialect reader/writer, `.npz` cache, stderr metadata (`DumpMeta`). |
| `src/dsp_server/engine/transforms.py` | Segment selection, axis fit (PCA), cylindrical coords, axial/sector/displacement profiles, roughness report, corrugation extent. |
| `src/dsp_server/engine/filters.py` | Detrend, SOS highpass (butter/cheby2, `sosfiltfilt`), FIR alternative, resampling, band energy, spectral peaks, Welch. |
| `src/dsp_server/engine/lbs.py` | Baked-JSON loader, FK/palette recompute, pure-numpy LBS, palette sidecar, weight surgery. |
| `src/dsp_server/engine/image2d.py` | Depth/AOV PNG loading, windowed 2D FFT ROI spectrum, SSIM/difference stats. |
| `src/cxx_bridge/engine_cli.py` | Engine discovery, `run_field_dump` subprocess wrapper, feature detection of patched flags, output naming, stale-binary guard. |
| `tests/stub_engine.py` | CLI-faithful fake engine (synthetic corrugated cylinder) used by CI and local tests. |

## Tool surface

| Tool (pack) | Key inputs | Output schema |
| --- | --- | --- |
| `extract_mesh_telemetry` (geometry) | `character_json=None, sex="male", pose_clip="clip-cin-stand-attention", label=None, want_palette=True` | `TelemetryResult` (paths, vertex/face/bone counts, clip + sampleTime, collisionPush floats, bone map, arm-chain joint histogram, `engine_stale`) or `ToolError` |
| `analyze_corrugation` (geometry) | `dump, joint="armLowerL", channel="posed", save_plot=False` | `CorrugationReport` (embeds `RoughnessSchema` + `SpectralPeak` list) |
| `compare_geometry_signals` (geometry) | `dump_a, channel_a, dump_b, channel_b, joint, save_plot` | `SignalComparison` (per-channel roughness, spectra deltas; equal-count dumps add per-vertex displacement stats + t-location of max) |
| `localize_defect` (geometry) | `dump, channel="posed", min_verts=100, top_k=8` | ranked `SegmentRoughness` list (per-joint) |
| `lbs_differential` (geometry) | `dump, palette=None, baked_json=None, joint="armLowerL", weight_surgery="none"` | `LbsDifferentialReport` (verdict hint: `deformer-band` / `weight-band` / `both` / `neither`) |
| `compare_depth_renders` (imaging) | `image_a, image_b=None, roi=None, save_plot=False` | imaging-pack report (2D spectral ridge freq + orientation, SSIM, difference stats) |

Every tool returns `model_dump_json()` of a pydantic model — scalars, short dicts, and file paths
only. Failures return the structured `ToolError{error, hint, stderr_tail}` instead of raising across
the MCP boundary.

## Dual-telemetry isolation (the core RCA decision)

Two independent telemetry channels exist for the same pose: the engine's own dump (LBS + deformers)
and a pure-numpy LBS reconstruction from rest positions + weights + palette (`engine/lbs.py`, which
reproduces `evaluateSkinnedFrame` to ~1e-4 m). Where the corrugation shows up decides who is guilty:

| Corrugation in numpy pure-LBS | Corrugation in engine dump | Verdict |
| --- | --- | --- |
| yes | yes | Skin weights guilty (defect exists before deformers run) |
| no | yes | Deformers guilty (capsule/collision push introduces it) |
| no | no | Position-space is clean — escalate to render-space / imaging lane (`compare_depth_renders`); check the normal channel first |

## PLY + stderr contract (verbatim)

The engine dialect, verified against `proje7-engine tools/layered-field-dump/main.cpp`:

```
ply
format ascii 1.0
comment posed layered mesh colored by dominant skin joint
element vertex <N>
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
property int sourceIndex
property float restx
property float resty
property float restz
element face <M>
property list uchar int vertex_indices
end_header
<N vertex rows>
<M face rows, literally: 3 i j k>
```

Patched builds invoked with `--weights` append `property int j0..j3` and `property float w0..w3`
after `restz`. ASCII only — binary formats are rejected with a clear error.

Quirk that must never be "fixed": the property NAMED `sourceIndex` carries the dominant skin-joint
index (argmax of the vertex's 4 weights). A future literal `dominantJoint` property wins if present.
One posed dump carries both bind (`restx/resty/restz`) and posed (`x/y/z`) positions on the same
row, so bind-vs-posed needs a single engine invocation.

stderr protocol (the bone table exists only here):

```
bone-map: 0=pelvis 1=spine ...
<sex>-posed-ok clip=<clipId> sampleTime=<sec> verts=<n>
deformers applied (collisionPush=<float> bakedCollisionPush=<float>)
import-ok verts=<v> bones=<b> clips=<c> packClips=<p>      # patched --character runs only
```

The `collisionPush` / `bakedCollisionPush` floats are max deformer push magnitudes in meters, not
counts. Exit codes: unknown argument or missing `--out` -> 2; runtime failure -> 1. Parsed metadata
is persisted as `<dump>.meta.json` next to each PLY (`DumpMeta` in `engine/ply.py`).

## Stale-binary policy

Every bridge response carries `engine_stale`: the exe's mtime compared against the newest source
under `src/core` + `tools/layered-field-dump` + the CMake files (a port of proje8's
`shouldRebuildEngineCli` semantics, with a 30 s in-process cache). It is a warning only — the server
never auto-builds, because 17-minute engine builds would blow MCP timeouts. Rebuilds are explicit
(`scripts/build-engine.ps1`); when the rtx preset is chosen, all rtx tools relink together.

## ELE407 map

Course weeks 2 (resampling), 5 (SOS flow-graph realizations), 8-9 (IIR/FIR design), and 12
(spectral estimation) live in `engine/filters.py`; week 14 (2-D signal processing) lives in
`engine/image2d.py`.

## Cloud split model (M9)

Geometry production stays local (Windows exes, DXR GPU, the `D:\` asset library); analysis runs
anywhere. The container image ships no engine binary, so `extract_mesh_telemetry` degrades to a
structured no-engine error, while the five analysis tools operate on telemetry files synced into the
container's `/data` volume (e.g. `aws s3 sync` from the local box). The transport flips to
`streamable-http` via env; tool code is identical in both deployments. See `docs/DEPLOYMENT.md`.

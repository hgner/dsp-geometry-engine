# Blender Body-Mesh MCP

`blender-body-mesh` is the client registration name. It launches `bodymesh-server` as its own
local-only stdio operating-system process. It is packaged beside `dsp-geometry-engine`, but it is not
a DSP pack, child server, or nested server. It turns bounded MPFB parameters into Blender body-mesh
candidates, writes artifacts that the analysis MCP can measure, and produces a baked character
accepted by the geometry engine. The Codex/Claude client orchestrates the two MCP connections;
neither MCP server calls the other.

## Server identity and boundary

**Direct answer:** `blender-body-mesh` does not run under `dsp-geometry-engine`. Codex or Claude runs
both as sibling MCP connections and orchestrates their tool calls.

Use these names when checking MCP registration or diagnosing missing tools:

| Meaning | Body-mesh server | Analysis server |
| --- | --- | --- |
| Registered client name | `blender-body-mesh` | `dsp-geometry-engine` |
| Executable entrypoint | `bodymesh-server` | `dsp-server` |
| Tool count | 6 | 49 |
| Owns body generation | Yes | No |
| Owns DSP/image/engine analysis | No | Yes |
| Remote HTTP deployment today | No | Optional |

The body-mesh tools are not registered by `dsp-server`, and the DSP tools are not registered by
`bodymesh-server`. When both appear in one Codex or Claude task, that means the client opened two
independent MCP connections. It does not mean one server is running under the other.

Both entrypoints are distributed in the same Python project, and `bodymesh-server` reuses the pure
DSP PLY parser for artifact validation. Shared package code is a library dependency only: it does not
start `dsp-server`, expose DSP tools on the body-mesh connection, or create server-to-server RPC.

The only handoff is through file paths and client orchestration:

1. The client may call DSP `segment_image` to create reference masks.
2. The client passes those mask paths to body-mesh `prepare_bodymesh_job`.
3. The body-mesh server runs Blender/MPFB and returns paths such as `body_dsp.ply` and
   `body_engine.baked.json`.
4. The client may pass one of those paths to DSP `analyze_mesh_topology`,
   `extract_mesh_telemetry`, or another analysis tool.

At no step does `bodymesh-server` send an MCP request to `dsp-server`, or vice versa. Shared files are
not automatically consumed; the client chooses which artifact path crosses into the next tool call.

## What it does — and does not do

The supported backend is the installed MPFB 2.0.x extension for Blender 4.2. A candidate is created
from explicit phenotype macros and optional detailed MPFB targets. Front, side, and back photographs
are retained as job references, and Blender renders matching orthographic silhouettes for iterative
comparison. Every completed candidate also carries the engine's exact 55-bone sex-specific skeleton,
an arms-down bind mesh, four-influence skinning, UV tangents, and `character_bake_cli` JSON.

This is **not** one-shot photo-to-3D reconstruction. Neither MPFB nor MB-LAB contains a photograph,
landmark, or silhouette inference model. A single perspective/clothed image is underdetermined.
Calibrated front and side references, a known height, and DSP-generated masks make the loop useful;
the LLM still has to revise the bounded MPFB controls from comparison results.

MB-LAB 1.8.1 is detected by the runtime tool but is not invoked. It is archived, lacks an equivalent
maintained headless path, and its own data/model license defaults generated meshes to AGPL-3. MPFB's
core assets are CC0 and its documented generated output is unrestricted; third-party MPFB assets may
carry their own terms.

## `blender-body-mesh` tool surface

| Tool | Purpose |
| --- | --- |
| `inspect_bodymesh_runtime` | Resolve direct `blender.exe`, detect MPFB/MB-LAB versions, and report configured roots/warnings without creating a mesh. |
| `list_bodymesh_parameters` | Describe the eleven MPFB macro controls; optionally search installed detailed target files by name. |
| `prepare_bodymesh_job` | Validate/copy front + optional side/back images and masks into an isolated job, recording hashes, known height, and explicit `rig_sex`. |
| `create_body_mesh` | Run MPFB generation, exact engine retarget, GLB export, and `character_bake_cli`. Repeated calls create immutable revisions. |
| `get_bodymesh_job` | Return compact reference/candidate history for a job. |
| `render_identity_set` | Render or reuse the fixed `identity-v1` set for a completed candidate: eight face closeups plus front/three-quarter/side body views. |

All body parameter and target weights are in `[0,1]`. `rig_sex` must be explicitly `male` or `female`
because the continuous MPFB gender control can be neutral while the engine skeleton cannot.
`known_height_m` is constrained to `[0.4,2.5]` and applies a final uniform scale.

## Running one tool without MCP — `bodymesh-tool`

A caller that spawns helper processes and parses JSON, rather than speaking MCP, runs a single
body-mesh tool through the `bodymesh-tool` console script (declared in `[project.scripts]`):

```text
uv run --no-sync bodymesh-tool <tool-name> --args-json '<json-object>'
```

`--no-sync` matters on win32. A bare `uv run` re-syncs the environment before executing, and that
sync tries to rewrite the console-script wrapper `.exe` files — which fails while a registered
`bodymesh-server` (or `dsp-server`) process is running and holding its own wrapper open, because
Windows will not replace a file that is in use. `--no-sync` skips the re-sync and runs against the
environment as it already stands, which is exactly the state you want while the MCP servers are live.

The contract:

- `--args-json` defaults to `{}` and must decode to a JSON object of the tool's keyword arguments.
- On success, stdout is exactly the JSON string the MCP tool returns, plus a trailing newline, and
  the exit code is 0. Nothing else is printed on stdout.
- On failure, stdout carries a `BodyMeshError` object (`error`, optional `hint`) and the exit code is
  nonzero: 2 for a bad command line — unknown tool name, unparseable or non-object `--args-json`,
  arguments the tool does not accept — and 1 when the tool itself raised or returned an error
  payload. A caller must therefore read stdout even on a nonzero exit; the structured failure is
  there, not on stderr.
- All six tools are exposed under their MCP names: `inspect_bodymesh_runtime`,
  `list_bodymesh_parameters`, `prepare_bodymesh_job`, `create_body_mesh`, `get_bodymesh_job`,
  `render_identity_set`. The CLI dispatches to the same implementation functions `bodymesh-server`
  registers, so the two surfaces cannot drift.
- A CLI call is a separate OS process but not a second worker. Blender stages take the same
  cross-process `.blender-worker.lock` under the data directory, so a CLI invocation issued while the
  MCP server is mid-render fails fast with `another Blender body-mesh job is already running` instead
  of starting a competing Blender.

### Wall-clock budgets

Blender dominates the cost, and the two heavy operations differ by more than an order of magnitude,
so a caller-side timeout tuned for one will misbehave on the other. Measured on a Windows workstation
rendering Cycles on a GPU device:

| Operation | Measured wall clock |
| --- | --- |
| `create_body_mesh` — MPFB generation, engine retarget, GLB export, bake; `uv` startup and Blender included | ~6 s |
| `render_identity_set` — eleven Cycles renders at 96 non-adaptive samples | ~92 s per set |

Treat these as calibration points, not guarantees: sample count is fixed but scene complexity, GPU,
and driver are not. The server's own Blender timeout is `BODYMESH_TIMEOUT_S` (default 600 s, accepted
range `[10,3600]`); a caller-side timeout belongs above the operation's measured cost and at or below
that ceiling.

## Reference preparation

The server reads only explicit image paths beneath `BODYMESH_INPUT_ROOTS`. The client registrations
are no longer tracked; both shipped templates (`.mcp.json.example`, `.codex/config.toml.example`) and
`scripts/register-claude-desktop.ps1` pin the allowlist to the job inbox alone:

- `data/bodymesh/inbox` — relative entries resolve against the server's working directory

Widening it is a deliberate local edit. Multiple roots are separated by `os.pathsep` (`;` on Windows),
so adding repository `data/` picks up DSP-generated masks under `data/images/`, and adding a
directory outside the repository grants the Blender boundary read access to everything beneath it —
keep it as narrow as the images you actually feed in.

Accepted inputs are PNG, JPEG, WebP, BMP, and TIFF, with file-size, dimension, decompression, and
path-traversal gates. The server copies and normalizes each image to PNG before Blender sees the job.
URLs, arbitrary Blender files, Python, operators, add-on paths, and caller-selected output paths are
not accepted.

For quantitative fitting, first use the DSP MCP's `segment_image` on each reference and pass its mask
path to `prepare_bodymesh_job`. Mask dimensions must match the corresponding image.

## Candidate artifacts

Each call creates:

```text
data/bodymesh/jobs/<job_id>/
  manifest.json
  references/
    front.png
    front_mask.png              # when supplied
  engine/
    skeleton_<sex>.json         # immutable validated snapshot for this job
  candidates/<candidate_id>/
    request.json
    mpfb_result.json
    result.json
    blender.stdout.log
    blender.stderr.log
    character.blend
    body.obj
    body_dsp.ply
    body_dsp.meta.json
    engine_retarget.request.json
    engine_retarget.result.json
    engine-retarget.stdout.log
    engine-retarget.stderr.log
    character-bake.stderr.log
    body_engine.glb
    body_engine.baked.json
    renders/front_mask.png
    renders/side_mask.png
    comparison/front_reference_mask.png  # normalized to the render canvas when supplied
    identity/identity-v1/                 # created by render_identity_set
      manifest.json
      request.json
      worker-result.json
      blender.stdout.log
      blender.stderr.log
      face/*.png
      body/*.png
```

`body_dsp.ply` remains neutral, unrigged Blender telemetry for topology and descriptive DSP. It is
not an engine dump. `body_engine.glb` is the skinned interchange artifact, and
`body_engine.baked.json` is the validated engine character for `engine --character`/
`extract_mesh_telemetry(character_json=...)`. `mpfb_result.json` records only the first Blender
stage; `result.json` is atomically replaced with the authoritative whole-candidate success or
failure after retarget, bake, and validation.

### `CandidateResult` is a stable output contract

`create_body_mesh` returns `CandidateResult` (`src/bodymesh_server/schemas.py`); for a successful
candidate the same model is persisted as that candidate's `result.json`. A failed candidate's
`result.json` is instead a compact failure record — `candidate_id`, `status: "failed"`,
`candidate_dir`, `duration_s`, `error`, `stderr_tail` — so check `status` before reading the rest.

`CandidateResult` is deliberately frozen so a downstream consumer can vendor fixtures against it and
parse it without defensive lookups. These fields are guaranteed present, under these names and with
their current semantics:

`job_id`, `candidate_id`, `status`, `backend`, `candidate_dir`, `blend_path`, `obj_path`,
`ply_path`, `meta_path`, `engine_glb_path`, `baked_character_path`, `rig_sex`, `bone_count`,
`rest_pose`, `engine_contract`, `render_paths`, `reference_mask_paths`, `vertex_count`,
`face_count`, `duration_s`, `macro_parameters`, `target_parameters`, `stderr_tail`, `warnings`.

Evolution is additive only: new fields may appear, existing ones are not renamed, removed, or given
new semantics. That is why the identity render set is versioned separately — `render_identity_set`
returns its own `IdentityRenderSet` (carrying an explicit `schema_version`) and never extends
`CandidateResult`. Any future change that cannot be expressed additively is a different contract, not
a revision of this one.

## Identity render contract

`render_identity_set(job_id, candidate_id, force=false)` accepts identifiers for a completed
candidate, resolves that candidate's `character.blend` internally, and runs one headless Blender
invocation. Arbitrary Blender inputs, render settings, and output paths are not accepted. The source
blend is hashed before and after rendering and must remain unchanged. The operation returns a
separately versioned `IdentityRenderSet`; it does not add to or change the frozen `CandidateResult`
contract.

The fixed `identity-v1` preset uses Cycles, 96 non-adaptive samples, seed `20260715`, deterministic
camera, expression, and studio-light variants. It produces N=8 closeups at 768x768 and M=3 body
views at 768x1024:

```text
identity/identity-v1/
  manifest.json
  face/
    face-00-neutral-key-left.png
    face-01-neutral-key-right.png
    face-02-smile-soft-front.png
    face-03-smile-left.png
    face-04-attentive-front.png
    face-05-attentive-right.png
    face-06-speaking-left.png
    face-07-speaking-right.png
  body/
    body-front.png
    body-three-quarter.png
    body-side.png
```

These paths and asset ordering are the stable preset contract. Each manifest asset also records its
relative and absolute path, dimensions, view, expression, lighting, and SHA-256. The manifest records
a recipe SHA-256 plus Blender, MPFB, renderer, and device provenance. The recipe covers the normalized
worker source and fixed render contract, so a behavior/contract change cannot silently reuse stale
images. With `force=false`, a valid manifest is reused only when both the source blend and recipe
hashes still match. `force=true` rebuilds the same final path through a staging directory and
recoverable atomic promotion. Fixed composition and seed do not imply byte-identical output across
Blender, Cycles, driver, or hardware versions; consumers should use the manifest and stable semantic
paths rather than assume cross-device PNG identity.

## Engine rig contract

Job preparation validates the authoritative `skeleton_male.json` or `skeleton_female.json` from
`BODYMESH_ENGINE_SKELETON_DIR` and snapshots it with its SHA-256. The retarget stage uses that
immutable copy. It asks MPFB for its topology-specific 53-bone `game_engine` weights,
maps those groups onto a newly built exact 55-bone engine armature, splits the extra MPFB spine weight
between `spine`/`chest`, limits and normalizes every vertex to four influences, fits the A-pose for
weighting, then bakes the inverse fit into the mesh and armature as the new arms-down rest. Because
the target rig is rotated 180 degrees to match MPFB's facing direction, the `_l`/`_r` source groups
are swapped once at that boundary. Both Blender and the final baked-JSON gate verify all 23 paired
bone/weighted-vertex centroids remain on the same physical side.

MPFB's game-engine weights have no nose, jaw, or eye groups. Those four required contract bones are
present with the correct hierarchy but currently have zero influence; body, limb, toe, and all finger
chains are weighted. This does not block body posing but is not facial-animation support.

`character_bake_cli` is not treated as sufficient validation by itself. Before a candidate is marked
complete, the MCP additionally requires exact 55 names/parent relationships, each of the 55 inverse
bind matrices to invert its computed bind-world matrix, schema-valid clips/tracks/keyframes with
known bone references, indexed triangles, finite UV/tangent data, tangent handedness, normalized
four-slot skinning, valid joint ranges, spatially local left/right skinning, and predominantly
downward upper-arm-to-elbow/hand directions in the bind pose.

## Intended fitting loop

1. DSP `segment_image` on calibrated front/side references.
2. Body-mesh `prepare_bodymesh_job` with images, masks, known height, and explicit `rig_sex`.
3. Body-mesh `list_bodymesh_parameters` for relevant controls.
4. Body-mesh `create_body_mesh` for candidate 1.
5. DSP `compare_depth_renders`, `compare_wavelet_signatures`, or
   `evaluate_perceptual_similarity` on each returned `reference_mask_paths[view]` / `render_paths[view]`
   pair; the body-mesh server makes those masks identically sized and centered.
6. Revise macro/detailed target values and call `create_body_mesh` again.
7. DSP `analyze_mesh_topology` on neutral `body_dsp.ply`; use returned
   `baked_character_path` for posed engine telemetry.

The current stage now completes deterministic rig/export/bake, but fitting remains LLM-driven. A
future automatic fitter still needs calibrated cameras/masks, body landmarks, an objective function,
and a bounded optimizer.

## Process and security boundary

The `bodymesh-server` host process is stdio-only. It serializes one validated `request.json`, invokes
the direct Blender binary (never `blender-launcher.exe`) with `--background --python-exit-code 1`,
captures logs, enforces
a timeout and cross-process single-worker lock, and accepts only paths contained by the candidate
directory. It validates mesh counts plus each render's PNG encoding, dimensions, and binary mask
content before marking a candidate complete. It then runs a second confined Blender retarget stage
and the configured `character_bake_cli`; any rig, skin, tangent, bake, or contract failure marks the
candidate failed. Mixed successful/failed attempts give the job a `partial` status. The Blender-side
workers are explicitly GPL-3.0-or-later and communicate only through request/result JSON; their
component license text and notice are shipped under `LICENSES/`. Cloud credentials and token-like
environment variables are not forwarded to Blender. Blender and bake stdout/stderr are captured in
the candidate directory; only compact result JSON and artifact paths return through MCP.

Do not expose this server as a remote HTTP MCP without a separate OS identity, private input staging,
strong authentication, network policy, and resource quotas: it handles private photographs and
executes a desktop DCC application.

## Known issues

**Windows `MAX_PATH` during identity staging — keep candidate labels short.** `render_identity_set`
renders into a staging directory beside the final one and promotes it atomically, so a file's working
path is `identity/.identity-v1.<32 hex>.tmp/face/<name>.png` while its published path is
`identity/identity-v1/face/<name>.png`. The staging directory name is 49 characters against the final
11, so every staged path is 38 characters longer than the published one. On a Windows build without
long paths enabled, a candidate whose published PNG paths would fit inside the 260-character limit can
still overrun it in staging; Blender then fails at the first PNG save with
`Render error (No such file or directory) cannot save`, deterministically across retries.

Most of that budget goes to the job and candidate ids, which embed the labels you pass. Both are
built the same way — `<YYYYmmdd-HHMMSS>-<slug>-<6 hex>`, where the slug is the sanitized label
truncated to 40 characters — so each can reach 63 characters, and the staged path carries both plus
the data root and the asset filename. Measured with the repository a few directories below the drive
root, a 51-character candidate id put the first staged PNG at 263 characters and failed, while a
44-character id landed at 258 and rendered.

Workarounds: keep job and candidate labels short — the timestamp and hex suffix already make ids
unique, so the label only has to be recognizable to you — or point `BODYMESH_DATA_DIR` at a shallow
directory.

## Registration troubleshooting

- If DSP tools work but the six body-mesh tools are absent, `dsp-geometry-engine` is healthy and
  `blender-body-mesh` is missing, stale, or failed independently. Inspect the client MCP list and the
  body-mesh server logs.
- If body-mesh tools work but analysis tools are absent, diagnose `dsp-geometry-engine`; restarting
  Blender will not repair that separate connection.
- After changing `.codex/config.toml`, `.mcp.json`, or Claude Desktop configuration, restart the
  client and open a fresh task so both registrations are reloaded.
- Do not manually start either stdio command inside an already registered session. The extra process
  has no client connection and may appear to hang while waiting for JSON-RPC on stdin.
- A completed body-mesh candidate is not automatically analyzed. Explicitly pass its returned path
  to the desired DSP MCP tool.

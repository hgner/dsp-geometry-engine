# Blender Body-Mesh MCP

`bodymesh-server` is a second, local-only MCP server in this repository. It turns bounded MPFB
parameters into Blender body-mesh candidates, writes artifacts that the existing DSP MCP can
measure, and produces a baked character accepted by the geometry engine. The Codex/Claude client
orchestrates the two MCP servers; neither MCP server calls the other.

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

## MCP tools

| Tool | Purpose |
| --- | --- |
| `inspect_bodymesh_runtime` | Resolve direct `blender.exe`, detect MPFB/MB-LAB versions, and report configured roots/warnings without creating a mesh. |
| `list_bodymesh_parameters` | Describe the eleven MPFB macro controls; optionally search installed detailed target files by name. |
| `prepare_bodymesh_job` | Validate/copy front + optional side/back images and masks into an isolated job, recording hashes, known height, and explicit `rig_sex`. |
| `create_body_mesh` | Run MPFB generation, exact engine retarget, GLB export, and `character_bake_cli`. Repeated calls create immutable revisions. |
| `get_bodymesh_job` | Return compact reference/candidate history for a job. |

All body parameter and target weights are in `[0,1]`. `rig_sex` must be explicitly `male` or `female`
because the continuous MPFB gender control can be neutral while the engine skeleton cannot.
`known_height_m` is constrained to `[0.4,2.5]` and applies a final uniform scale.

## Reference preparation

The server reads only explicit image paths beneath `BODYMESH_INPUT_ROOTS`. The checked-in local
configuration allows:

- repository `data/` (including `data/bodymesh/inbox/` and DSP-generated masks under `data/images/`)
- the current user's Desktop

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
```

`body_dsp.ply` remains neutral, unrigged Blender telemetry for topology and descriptive DSP. It is
not an engine dump. `body_engine.glb` is the skinned interchange artifact, and
`body_engine.baked.json` is the validated engine character for `engine --character`/
`extract_mesh_telemetry(character_json=...)`. `mpfb_result.json` records only the first Blender
stage; `result.json` is atomically replaced with the authoritative whole-candidate success or
failure after retarget, bake, and validation.

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

The outer MCP is stdio-only. It serializes one validated `request.json`, invokes the direct Blender
binary (never `blender-launcher.exe`) with `--background --python-exit-code 1`, captures logs, enforces
a timeout and cross-process single-worker lock, and accepts only paths contained by the candidate
directory. It validates mesh counts plus each render's PNG encoding, dimensions, and binary mask
content before marking a candidate complete. It then runs a second confined Blender retarget stage
and the configured `character_bake_cli`; any rig, skin, tangent, bake, or contract failure marks the
candidate failed. Mixed successful/failed attempts give the job a `partial` status. The Blender-side
workers are explicitly GPL-3.0-or-later and communicate only through request/result JSON; their
component license text and notice are shipped under `LICENSES/`. Cloud credentials and token-like
environment variables are not forwarded to Blender.

Do not expose this server as a remote HTTP MCP without a separate OS identity, private input staging,
strong authentication, network policy, and resource quotas: it handles private photographs and
executes a desktop DCC application.

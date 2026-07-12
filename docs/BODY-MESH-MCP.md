# Blender Body-Mesh MCP

`bodymesh-server` is a second, local-only MCP server in this repository. It turns bounded MPFB
parameters into Blender body-mesh candidates and writes artifacts that the existing DSP MCP can
measure. The Codex/Claude client orchestrates the two servers; neither server calls the other.

## What it does — and does not do

The supported backend is the installed MPFB 2.0.x extension for Blender 4.2. A candidate is created
from explicit phenotype macros and optional detailed MPFB targets. Front, side, and back photographs
are retained as job references, and Blender renders matching orthographic silhouettes for iterative
comparison.

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
| `prepare_bodymesh_job` | Validate/copy front + optional side/back images and masks into an isolated job, recording hashes and known height. |
| `create_body_mesh` | Run one headless Blender/MPFB candidate with bounded parameters and fixed outputs. Repeated calls create immutable revisions. |
| `get_bodymesh_job` | Return compact reference/candidate history for a job. |

All body parameter and target weights are in `[0,1]`. `known_height_m` is constrained to `[0.4,2.5]`
and applies a final uniform scale, so the exported PLY height is physically meaningful.

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
  candidates/<candidate_id>/
    request.json
    result.json
    blender.stdout.log
    blender.stderr.log
    character.blend
    body.obj
    body_dsp.ply
    body_dsp.meta.json
    renders/front_mask.png
    renders/side_mask.png
    comparison/front_reference_mask.png  # normalized to the render canvas when supplied
```

`body_dsp.ply` uses the ASCII property contract already consumed by `dsp_server.engine.ply`. It is a
neutral, unrigged Blender/MPFB surface in meters: posed and rest positions are identical and the
single pseudo-joint is `body`. It is valid for `analyze_mesh_topology` and descriptive surface DSP.
It is **not** a proje7-engine dump, so `lbs_differential` or engine-deformer conclusions are invalid.
To enter the engine lane, export/retarget/bake through the established character pipeline first and
then call `extract_mesh_telemetry(character_json=...)` on the resulting baked JSON.

## Intended fitting loop

1. DSP `segment_image` on calibrated front/side references.
2. Body-mesh `prepare_bodymesh_job` with images, masks, and known height.
3. Body-mesh `list_bodymesh_parameters` for relevant controls.
4. Body-mesh `create_body_mesh` for candidate 1.
5. DSP `compare_depth_renders`, `compare_wavelet_signatures`, or
   `evaluate_perceptual_similarity` on each returned `reference_mask_paths[view]` / `render_paths[view]`
   pair; the body-mesh server makes those masks identically sized and centered.
6. Revise macro/detailed target values and call `create_body_mesh` again.
7. DSP `analyze_mesh_topology` on `body_dsp.ply` before downstream retarget/bake.

The current stage deliberately stops at deterministic candidate generation and comparison artifacts.
A future automatic fitter needs calibrated cameras/masks, body landmarks, an objective function, and
a bounded optimizer; the existing tools do not infer those semantic constraints by themselves.

## Process and security boundary

The outer MCP is stdio-only. It serializes one validated `request.json`, invokes the direct Blender
binary (never `blender-launcher.exe`) with `--background --python-exit-code 1`, captures logs, enforces
a timeout and cross-process single-worker lock, and accepts only paths contained by the candidate
directory. It validates mesh counts plus each render's PNG encoding, dimensions, and binary mask
content before marking a candidate complete. Mixed successful/failed attempts give the job a
`partial` status. The
Blender-side MPFB worker is explicitly GPL-3.0-or-later and communicates only through request/result
JSON; its component license text and notice are shipped under `LICENSES/`. Cloud credentials and
token-like environment variables are not forwarded to Blender.

Do not expose this server as a remote HTTP MCP without a separate OS identity, private input staging,
strong authentication, network policy, and resource quotas: it handles private photographs and
executes a desktop DCC application.

# Engine Debugging Playbook

Advisory guide for using the DSP-Geometry-Engine MCP tools on the proje7-engine C++ codebase — or on
any mesh / render / control / telemetry problem. It answers one question: **you see a symptom in the
engine; which tool do you pick up, what do you feed it, and what does a bad answer look like?**

## Why these tools exist (the LLM-deficit principle)

An LLM cannot, inside its context window, traverse a 75k-vertex adjacency array, unwrap-and-differentiate
a phase spectrum, integrate a BRDF over the hemisphere, or take the rank of a controllability matrix. It
*can* read a one-line verdict. Every tool here does the mechanical math on data that lives on disk and
returns a compact JSON summary — a number, a boolean, a ranked list — that the model can reason about.
Reach for a tool the moment a question requires arithmetic over an array you cannot hold in your head.

Two standing rules when you use them on the engine:
1. **Arrays never cross the MCP boundary.** Results are scalars, short dicts, and file paths. The full
   arrays/plots are under `data/` if you need to point a human at them.
2. **These are instruments and calculators, not ground truth about the shipped binary.** Where the
   engine has its own C++ test (furnace tests, skinning goldens), that test is authoritative; these
   tools cross-check, localize, and explain — they don't replace the in-engine gate.

Trim the visible tool set per session with `DSP_TOOLSETS` (e.g. `geometry,rendering` for a mesh+shader
session) so the model isn't choosing among 41 tools when 8 are relevant.

---

## Symptom → tool index

| You are seeing… | Reach for | Pack |
| --- | --- | --- |
| Periodic ripple / corrugation on a limb | `analyze_corrugation`, then `localize_defect` | geometry |
| "Is the ripple in the source mesh or the skinning?" | `lbs_differential` | geometry |
| "Is the defect worse in pose A vs B — significantly?" | `compare_dump_ripples` (stats), `compare_geometry_signals` (descriptive) | stats / geometry |
| Holes, floating islands, "not watertight", spaghetti | `analyze_mesh_topology` | geometry |
| Tearing / sliding / lag during animation | `group_delay` (cross-spectrum mode) | systems |
| Fireflies, blown-out exposure, too-bright material | `verify_brdf_energy` | rendering |
| Corrugation seen only in depth renders, not verts | `compare_depth_renders`, `segment_image` | imaging |
| A feedback/physics loop won't reach a target state | `state_space_analysis` | systems |
| Filter/controller stability, resonance, margins | `pole_zero`, `bode`, `lti_response` | systems |
| Aliasing after a resample / sampling-rate choice | `sampling_check` | systems |
| "Which of N telemetry runs is the outlier?" | `describe`, `hypothesis_test`, `cluster` | stats / ml |
| Hand-checking ODE / Laplace / eigen / residue math | engmath pack | engmath |
| Queue depth / latency of an async pipeline | `queueing_calc`, `little_law` | netqueue |
| Scheduler / paging / deadlock reasoning | os pack | os |

---

## 1. Mesh geometry defects (the primary lane)

**Symptom: periodic ripple on a limb (the `cc0_male_rigged3` forearm class).**
`analyze_corrugation(dump, joint="armLowerL")` turns the joint's vertices into a radial profile along the
bone axis and reports `dominant_freq_cpm`, `dominant_wavelength_m`, `peak_prominence_db`, and a
`corrugated` verdict. Red flag: prominence > 6 dB with a wavelength matching the mesh edge-loop spacing.
Then `localize_defect(dump)` ranks every joint by spectral-peak prominence so you learn whether it's
forearm-only or systemic (spine/pelvis showed peaks on rigged3 too).

**Symptom: "is this the mesh or the skinning?"** — the load-bearing diagnostic.
`lbs_differential(dump)` reconstructs pure numpy LBS from the rest positions + weights + palette and
diffs it against the engine's posed output. Verdict `weight-band` → the corrugation is in the retargeted
weights; `deformer-band` → it's the capsule/flexion deformers; present in the raw bind mesh → it's the
source bake (this is what exonerated the engine on rigged3 — the ripple was already in `baked.json`).

**Symptom: significance, not just "it looks different".**
`compare_geometry_signals` gives descriptive deltas between two dumps/channels (rest vs posed, clip A vs
B). For a *statistical* verdict — "is pose B genuinely rippled more than A, or is it noise?" — use
`compare_dump_ripples` (stats pack): bootstrap CI on the band-energy ratio + unpaired Mann-Whitney across
angular sectors.

**Symptom: holes, floating islands, non-watertight iso-surface, "4-5 bone spaghetti".**
`analyze_mesh_topology(dump)` reports `n_components` (disconnected islands), `boundary_edge_count`
(holes), `nonmanifold_edge_count`, `is_watertight`, valence histogram + `n_irregular_valence`,
`euler_characteristic` and `estimated_genus`. Red flags for an engine iso-surface bug: `n_components > 1`
(the `IsoSurfaceExtractor` welded into disjoint shells), `nonmanifold_edge_count > 0` (a bad weld), or a
valence spike at the corrugated loops (a subdivision/bake artifact signature). Optional geodesic distance
between two vertices/joints via `src_vertex`/`dst_vertex`. This is whole-mesh QA — the complement to the
per-joint `analyze_corrugation`.

---

## 2. Skinning / animation phase (tearing, sliding, lag)

If an artifact isn't a static ripple but a *tearing or sliding during motion*, it's likely a phase/lag
issue — some vertex transforms lagging others across spatial frequencies. `group_delay` in
**cross-spectrum mode** takes two vertex-correspondent signals (e.g. the engine-posed forearm profile vs
the numpy pure-LBS profile of the same segment, via `.ply` column addressing or exported `.npz` series)
and reports the frequency-dependent lag `tau(w)`. A large `tau_variance` or a `tau` that ramps with
frequency means the blend/filter is delaying high-frequency vertex motion behind low — a phase-alignment
bug in the skinning or a filtered blend. Pair with `lbs_differential` to localize which space introduces
the lag. (Note: group delay of a *single* profile is window-relative and not meaningful — always use two
correspondent signals here.)

---

## 3. Render / shader energy (fireflies, exposure blowout)

The engine is a DXR path tracer; fireflies and blown exposure are usually a **BRDF energy leak**.
`verify_brdf_energy(model="ggx", roughness=…, f0=…)` numerically integrates the BRDF over the hemisphere
(white-furnace test) and returns directional albedo per viewing angle. Red flags: `energy_conserving:
false` / `max_albedo > 1` = a non-physical leak (a mis-normalized NDF, a bad `1/(4·cosθ·cosθ)` term) that
blows out exposure; `single_scatter_loss` large at high roughness = expected single-scatter darkening.
With `mc_samples > 0` it also reports the variance reduction of cosine-importance vs uniform sampling —
high variance is literally the firefly mechanism.

**Critical caveat:** this is a *design calculator*. It integrates the analytic models it ships
(Lambertian, GGX Cook-Torrance / Smith / Schlick). To debug the *actual* engine output you must point it
at the engine's real BRDF parameters and confirm the model matches the shipped HLSL — and the engine's
own C++ furnace tests (see the DXR ADRs: constant-map furnace, energy brackets) remain the authoritative
gate. Use this to catch an energy leak in a *proposed* BRDF before it's baked into a shader.

**Symptom: corrugation/artifact visible only in depth renders, not in vertex dumps.**
That's the render-space escalation path. `compare_depth_renders(image_a, image_b)` does a 2-D spectral +
SSIM comparison of depth/AOV PNGs; `segment_image` (imaging) with morphology gives you a connected-
component defect mask on the render; `filter_image`/`enhance_image` isolate the band.

---

## 4. Control loops & dynamic systems

**Symptom: an automated feedback loop or physics solver won't drive the engine to a target state.**
`state_space_analysis(a=…, b=…, c=…)` builds the controllability matrix `[B AB … Aⁿ⁻¹B]` and reports its
rank. If `controllable: false` (rank < n), it names the `uncontrollable_dim` — there are states your
inputs physically cannot reach, no gain tuning will fix it. With `c=…` it does the observability dual
(can the outputs reconstruct the state). `ctrb_min_singular` near zero warns of *near*-loss (numerically
fragile controllability). Matrices go in inline as `list[list[float]]` or as csv/npz paths.

**Stability, resonance, margins:** `pole_zero` (stability + damping + natural frequencies),
`bode` (gain/phase margins), `lti_response` (step/impulse metrics — overshoot, settling). `group_delay`
in LTI mode reports where a filter delays which frequencies. `sampling_check` gives an aliasing verdict
for a proposed sample rate (and refuses to bless a rate when the reference signal isn't sampled well
above it — no false "safe").

---

## 5. Telemetry, statistics, ML (analyzing engine output at scale)

Engine runs emit a lot of numbers (per-frame timings, per-vertex displacements, bake scores). The stats
and ml packs are general-purpose over any `.csv/.tsv/.json/.npz/.npy` (column-addressed), with `.ply`
dumps addressable as `column="<joint>[:posed|rest]"`:
- `describe` (moments/quantiles/CIs), `fit_distribution` (which distribution + goodness-of-fit),
  `hypothesis_test` (is run A different from run B), `regression_fit` (does timing scale with vertex count).
- `feature_engineer_dump` builds a per-vertex feature matrix from a dump; `cluster` finds defect regions;
  `reduce_dims` (PCA/LDA) collapses them; `classify_eval`/`predict` train + apply a small model (models
  are path-confined to `data/models/` — `predict` only loads sidecar-verified models this server wrote).

## 6. Math calculators & modeling (engmath, netqueue, os)

- **engmath** (`solve_ode`, `laplace_transform`, `linear_algebra`, `residues_and_integrals`,
  `fourier_series`) — hand-check the analysis before you commit it to C++. Expression strings are
  sandboxed (whitelisted functions, literal-size gates); hostile or huge input returns a fast ToolError.
- **netqueue** (`queueing_calc`, `little_law`, `erlang_blocking`) — model an async job queue or a bake
  pipeline: utilization, expected latency, occupancy. Unstable input (ρ ≥ 1) is flagged, not silently NaN.
- **os** (`schedule_sim`, `page_replacement_sim`, `bankers_check`) — reason about a scheduler, a cache/
  paging policy, or deadlock in a resource-acquisition graph. Deterministic textbook simulators.

---

## Red-flag cheat sheet

| Tool | Output that means "there's a bug" |
| --- | --- |
| `analyze_corrugation` | `corrugated: true`, prominence > 6 dB at the edge-loop wavelength |
| `lbs_differential` | verdict names the guilty space (`weight-band` / `deformer-band`) |
| `analyze_mesh_topology` | `is_watertight: false`, `nonmanifold_edge_count > 0`, `n_components > 1` |
| `group_delay` (xspec) | large `tau_variance`; `tau` ramps with frequency |
| `verify_brdf_energy` | `energy_conserving: false`, `max_albedo > 1` (leak) |
| `state_space_analysis` | `controllable: false` (rank < n) — unreachable states |
| `sampling_check` | verdict `aliased`; `insufficient_resolution` (reference undersampled) |
| `compare_dump_ripples` | CI on the band-energy ratio excludes 1.0 (significant) |

---

## For a fresh engine session

Register the server (it's already wired at user + project scope; new sessions see all 41 tools), set
`DSP_TOOLSETS` to the lanes your task needs, and start from the symptom index above. The corrugation RCA
is *closed* (verdict: source Blender bake, not the engine — see `llms.txt` rule 8), so a new mesh
investigation begins with `analyze_mesh_topology` + `lbs_differential` to classify a fresh defect before
spending an engine rebuild. For a shader/exposure investigation, start with `verify_brdf_energy` against
the real material parameters. For a control/stability question, `state_space_analysis` + `pole_zero`.

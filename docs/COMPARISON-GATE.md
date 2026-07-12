# The AI-Video Comparison Gate

How to validate ONE generated clip against the deterministic engine's ground truth using the nine
comparison-gate tools. Companion to `docs/ENGINE-PLAYBOOK.md` §3b — that section indexes the tools;
this doc is the *integration* contract: which symptom picks which tool, how the per-tool verdicts
combine into a single clip pass/fail, and where the tools lie to you. The whole flow is proven
end-to-end in `tests/test_comparison_gate_e2e.py` (each gate on its own scenario + the combiner).

Enable exactly the four packs this gate spans:

```
DSP_TOOLSETS=video,imaging,geometry,perceptual
```

(unset = all packs, but the gate never needs stats/engmath/systems/ml/netqueue/os — trim so the model
chooses among 9 tools, not 49.)

The engine renders EXACT AOV passes a generator can only guess — beauty, per-pixel NORMAL, DEPTH/Z,
and an object ID/MASK. The gate is: does the generated clip agree with those passes? Arrays never cross
the MCP boundary (llms.txt rule 11); every tool returns a scalar/verdict + an on-disk path under
`data/{video,perceptual}/`.

---

## Symptom → tool

| The generated clip… | Reach for | Pack | Consumes |
| --- | --- | --- | --- |
| body shape/pose/proportions wrong | `score_bake` | geometry | DEPTH + MASK (+ ref depth pack) |
| looks soft / lost fine texture | `compare_wavelet_signatures` | imaging | beauty + ref beauty |
| "math flags huge error but looks fine" (a shift) | `evaluate_perceptual_similarity` | perceptual | beauty + ref beauty |
| shimmers / boils / flickers over time | `evaluate_spatiotemporal_frequencies` | video | frame STACK (+ ref stack) |
| a region melts / drifts / hallucinates motion | `verify_motion_consistency` | video | frame STACK (self-ref) |
| camera drifted / panned / FOV warped | `verify_camera_projection` | video | beauty + ref beauty |
| lighting ignores the geometry / baked shading | `analyze_photometric_consistency` | video | beauty + NORMAL (+ ALBEDO) |
| occlusion edges blurred / depth layers bleed | `evaluate_occlusion_boundaries` | video | beauty + DEPTH (+ ref beauty) |
| a tracked object morphs identity across frames | `verify_identity_coherence` | perceptual | RGB stack + MASK |

Six gates are PER-FRAME (score_bake, wavelet, perceptual, camera, photometric, occlusion — run on one
representative keyframe against its aligned passes); two are TEMPORAL (spatiotemporal, motion — take the
whole `(T,H,W)` stack); one is per-object over the sequence (identity).

---

## The six decision axes (and the red-flag verdict per tool)

Route by *axis of failure*, not by appearance — many defects look the same to a VLM and only the math
separates them.

**1. STRUCTURE — is the silhouette/pose right?**
`score_bake(bake_dir, reference_glob)` → silhouette IoU + Hu-moment distance (I2) + edge-orientation,
best-of-mirror, scale/translation/facing-normalized. Red flag: `pose_ok == false` → `verdict ==
"pose-weak"` (`iou_best < 0.50` or `hu_best > 0.30`). The trailing `*` (corrugation advisory,
`corr_db >= 30`) is NON-gating — it never fails the clip. Blind by construction to camera shift/zoom
(`normalize_silhouette` crops-to-bbox and scale-normalizes, absorbing translation + scale) and to any
RGB/lighting defect (reads depth only).

**2. SPATIAL — is the frame's detail intact?**
`compare_wavelet_signatures(ref, gen)` → per-octave band-energy parity. Red flag: `micro_parity <
0.7*macro_parity` with `worst_scale == 1` → `"micro-texture loss / smoothing"`; or `micro_parity > 1.5`
→ `"excess detail / hallucination"` (added shimmer/boil).
`evaluate_perceptual_similarity(ref, gen)` → CW-SSIM (shift-tolerant). Red flag: `cw_ssim < 0.55` →
`"distinct"` (replaced/hallucinated content). Its *raison d'être* is the INVERSE: a high `cw_ssim` with a
low pixel `ssim` and large `shift_tolerant_gap` = a benign few-pixel shift — it OVERRULES a wavelet/pixel
false alarm. Run both; perceptual is the tie-breaker.

**3. TEMPORAL — is the clip stable over time?**
`evaluate_spatiotemporal_frequencies(stack[, ref])` → per-pixel FFT along T. Red flag:
`temporal_hf_energy_fraction > 0.35` (`_FLICKER_ABS_FLOOR`) → `"flicker"`; with a reference,
`delta_hf > 0.1` → `"flicker-vs-ref"`. `dominant_temporal_freq_cpf ≈ 0.5` = a Nyquist alternating-frame
pump; high `boil_energy_fraction` = fine-detail boil (the spatial high-pass separates a uniform flicker
from boil).
`verify_motion_consistency(stack, tau)` → bidirectional Lucas-Kanade `‖f + b(p+f)‖`. Red flag:
`inconsistent_fraction` above control / `max_fb_residual_px > 0.5` / a nonempty `invalid_points` cluster =
a LOCAL geometry-disobeying (melting) region. A globally coherent pan round-trips to ~0 — motion-clean is
the POSITIVE signal that motion is coherent even when the camera is wrong.

**4. PROJECTION — is the camera right?**
`verify_camera_projection(ref, gen)` → Shi-Tomasi corners tracked ref→gen, RANSAC homography + fundamental.
Red flag: `"camera-drift"` (`homography_inlier_fraction >= 0.6`, a single homography explains a nonzero
`camera_shift_px`) or `"geometry-inconsistent"` (`homography_inlier_fraction < 0.6`, no global model fits).
`epipolar_degenerate == true` on a same-view/pure-translation pair — don't read the epipolar distance then.

**5. LIGHTING / OCCLUSION — do shading and edges obey the passes?**
`analyze_photometric_consistency(gen, normal[, albedo])` → fits `S ~ ambient + N·L`. Red flag:
`photometric_r2 < 0.5` → `"inconsistent"` (shading the true normals can't explain); `albedo_shading_leak`
high = a failed intrinsic decomposition.
`evaluate_occlusion_boundaries(gen, depth[, ref])` → variance-of-Laplacian in the DEPTH-pass edge band.
Red flag: `gen_vs_ref_ratio < 0.6` → `"soft-boundaries"` (the generator, having no Z-buffer, bled fg/bg
across the occlusion border).

**6. IDENTITY — does a tracked object keep its material?**
`verify_identity_coherence(stack, mask)` → cosine drift of the masked object's colour+texture descriptor
vs frame 0. Red flag: `first_break_frame >= 0` → `"identity-drift"` (`min_coherence` in [0.5, 0.8)) or
`"morph"` (`< 0.5`). Catches the slow material drift that flow (tracks WHERE) and shape gates miss.

---

## Combining into one clip pass/fail

Run the axes in this order; a clip PASSES iff every gate passes. But because a single physical defect
co-fires several gates, the pipeline must ATTRIBUTE before it fails — count root causes, not flags.

```
1. STRUCTURE   score_bake.pose_ok                       -> else FAIL(structure)
2. PROJECTION  camera.verdict == "camera-consistent"    -> else FAIL(camera)   [see attribution]
3. TEMPORAL    spatiotemporal.flicker_verdict=="stable"
               AND motion.inconsistent_fraction<=ctrl   -> else FAIL(temporal)
4. LIGHTING    photometric.verdict == "consistent"      -> else FAIL(lighting)
5. OCCLUSION   occlusion.verdict == "sharp"             -> else FAIL(occlusion)
6. SPATIAL     wavelet parity AND perceptual!="distinct"-> else FAIL(spatial)
               (perceptual OVERRULES wavelet on a benign shift: high cw_ssim + low ssim)
7. IDENTITY    identity.first_break_frame == -1         -> else FAIL(identity)
```

`test_comparison_gate_e2e.py::test_full_comparison_gate_composes` is exactly this combiner: a clean
bundle passes every axis (overall PASS); a defect bundle flags every axis (overall FAIL).

Attribution (resolve the multi-gate interferers so you report ONE root cause):

- **Camera-drift is the master interferer — BUT it does not touch score_bake.** A global shift/zoom
  misregisters every *pixel-aligned* pass: it collapses `photometric_r2` (each luminance pixel meets the
  WRONG normal), starves `evaluate_occlusion_boundaries` (real edges slide off the depth band), and tanks
  pixel-SSIM — all FALSE consequences. It does NOT tank `score_bake` (silhouette IoU is
  translation/scale-normalized, so it is camera-drift-blind — see §1). If `verify_camera_projection` says
  `camera-drift` AND `verify_motion_consistency` stays clean (motion coherent, only projection wrong),
  attribute to CAMERA and treat the photometric/occlusion/SSIM flags as consequences; re-run those on a
  camera-aligned frame.
- **Perceptual overrules the spatial math.** A benign 2–3 px texture shift is a large wavelet/pixel error
  but `cw_ssim ≈ 1`. High `cw_ssim` + `shift_tolerant_gap > 0.15` ⇒ dismiss the wavelet/SSIM red flag.
- **Temporal defects bleed sideways.** A melting region raises spatiotemporal `boil` AND motion FB
  residual AND wavelet Scale-1 AND perceptual — that cluster is ONE defect (local hallucination), caught
  by motion; a spatially-uniform flicker raises spatiotemporal HF AND motion AND SSIM but leaves
  photometric/occlusion/wavelet CLEAN — that discriminator says "exposure pump," not geometry.
- **Identity is the lonely gate.** A slow material morph that keeps brightness-constancy (motion clean)
  and the silhouette (score_bake clean) fires ONLY identity (+ maybe photometric) — its unique niche.

---

## Honest caveats

- **Pure-scipy, not OpenCV / LPIPS / DINOv2.** The optical flow is a hand-rolled pyramidal Lucas-Kanade
  (`optflow`), not calibrated OpenCV; perceptual is CW-SSIM over a Gabor bank, not a neural LPIPS; identity
  is a colour+texture cosine, not a face/DINOv2 embedding. The server stays a pure-wheel, deterministic,
  cloud/cron-safe product (no torch, no downloaded weights) — honestly not neural-SOTA. A true neural gate
  belongs in the video pipeline (proje8, which already has GPU/torch); these are the deterministic rigid
  checks it calls.
- **Thresholds are synthetic-golden-calibrated.** `_FLICKER_ABS_FLOOR=0.35`, `_FLICKER_REF_MARGIN=0.1`,
  camera `consistent_px=0.75` / homography-inlier `0.60`, photometric `r2_ok=0.5`, occlusion `ref_ok=0.6`,
  wavelet `_MICRO_SMOOTH_RATIO=0.7` / `_GAIN_CEIL=1.5`, perceptual `equivalent_cw=0.85` / `distinct_cw=0.55`,
  identity `break_threshold=0.8`. These were tuned against the synthetic goldens (`tests/test_{video,
  geometry_gate,wavelets,perceptual,scoring}.py`), NOT against a corpus of real generator failures —
  treat verdicts near a boundary as advisory and re-tune per real content. The first real gen-vs-ref clip
  pair is what will calibrate them.
- **The identity tracker uses a FIXED mask.** `verify_identity_coherence` samples the same HxW region every
  frame; if the object translates out of it the descriptor drifts for a benign reason. Pair with
  `verify_motion_consistency` when the object moves a lot, or feed a per-frame mask sequence.
- **Photometric is a linear proxy.** `S ~ ambient + N·L` has no cast shadows, no `max(0,·)` clamp — a
  consistency gate, not a light solver. A uniform +offset is fully absorbed by `ambient`, so it cannot see
  a temporal flicker (that's spatiotemporal's job).
- **score_bake's corrugation is advisory.** The render-side depth FFT doesn't resolve the ~8 mm edge-loop
  ripple as a peak; `mesh_ok` only appends a `*`. The authoritative corrugation detectors remain
  `compare_wavelet_signatures` Scale-1 (spatial) and `analyze_corrugation` on the vertex DUMP (~119 cy/m).
- **Per-frame gates can't see time; temporal gates can't localize geometry.** Each axis is deliberately
  narrow; the value is the COMBINATION + attribution above, not any single scalar. And no shared clip
  satisfies every gate's clean-case at once (the temporal gate wants spatially-smooth content; camera /
  perceptual want broadband texture with corners) — feed each gate its appropriate keyframe/stack, as the
  e2e test does.

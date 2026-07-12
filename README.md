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

## Tool packs

49 tools across 11 packs (the engine's own DSP lane, one pack per relevant EE course, a rendering lane
for the ray tracer, a video lane for the AI-video comparison gate, and a perceptual FR-VQA lane). Each pack registers by default;
trim per client with the `DSP_TOOLSETS` env var (comma-separated names). General data goes in as
`.csv/.tsv/.json/.npz/.npy` paths (column-addressed); `.ply` paths address engine dumps as
`column="<joint>[:posed|rest]"`; video tools take a directory or list of frame images.

| Pack (`DSP_TOOLSETS` name) | Course / lane | Tools |
| --- | --- | --- |
| `geometry` | ELE407 DSP + mesh QA | `extract_mesh_telemetry`, `analyze_corrugation`, `compare_geometry_signals`, `localize_defect`, `lbs_differential`, `score_bake`, `analyze_mesh_topology` |
| `imaging` | ELE490 Image Processing | `compare_depth_renders`, `enhance_image`, `filter_image`, `segment_image`, `restore_image`, `compare_wavelet_signatures` |
| `stats` | ELE320 Probability & Statistics | `describe`, `fit_distribution`, `hypothesis_test`, `regression_fit`, `compare_dump_ripples` |
| `engmath` | MAT235/236 Engineering Math | `solve_ode`, `laplace_transform`, `linear_algebra`, `residues_and_integrals`, `fourier_series` |
| `systems` | ELE301 Signals & Systems + control | `lti_response`, `pole_zero`, `bode`, `sampling_check`, `convolve_signals`, `group_delay`, `state_space_analysis` |
| `ml` | ELE489 Machine Learning | `feature_engineer_dump`, `cluster`, `reduce_dims`, `classify_eval`, `predict` |
| `netqueue` | ELE412 Data Communication | `queueing_calc`, `little_law`, `erlang_blocking` |
| `os` | Operating Systems (Tanenbaum) | `schedule_sim`, `page_replacement_sim`, `bankers_check` |
| `rendering` | PBR / ray-tracer energy | `verify_brdf_energy` |
| `video` | AI-video comparison gate | `evaluate_spatiotemporal_frequencies`, `verify_motion_consistency`, `verify_camera_projection`, `analyze_photometric_consistency`, `evaluate_occlusion_boundaries` |
| `perceptual` | Perceptual / semantic FR-VQA (classical) | `evaluate_perceptual_similarity`, `verify_identity_coherence` |

All tools return small JSON summaries; arrays, plots, models, and images stay on disk under `data/`
(`plots/`, `series/`, `images/`, `features/`, `models/`, `brdf/`, `video/`, `perceptual/`). Preset examples:
`DSP_TOOLSETS=geometry,imaging,stats` for a corrugation-RCA session; `DSP_TOOLSETS=engmath,systems,stats`
for coursework-style calculation; `DSP_TOOLSETS=geometry,rendering` for engine mesh + shader QA;
`DSP_TOOLSETS=video,imaging,geometry,perceptual` for the full AI-video generation comparison gate
(math + perceptual).

## Docs

- `docs/ENGINE-PLAYBOOK.md` — **when/where to reach for each tool in an engine-debugging session** (advisory).
- `docs/COMPARISON-GATE.md` — **the AI-video comparison gate**: symptom→tool, how the per-tool verdicts combine into one clip pass/fail, and the honest caveats (proven end-to-end in `tests/test_comparison_gate_e2e.py`).
- `docs/ARCHITECTURE.md` — data flow, two-repo layout, tool surface, PLY/stderr contract, isolation logic.
- `docs/DEVELOPMENT.md` — setup, lint/test, engine build lane, MCP registration, adding a course pack.
- `docs/DEPLOYMENT.md` — container image, AWS (App Runner / ECS Fargate), env var reference, security.
- `llms.txt` — canonical rules for LLM agents working in this repo (imported by `CLAUDE.md`).

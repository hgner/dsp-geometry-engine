# Pack designs: imaging v2 (ELE490), netqueue (ELE412), os (Tanenbaum)

Grounded against the frozen architecture: `toolsets/__init__.py` (AppContext/TOOLSETS), the `_impl` + closure + `ToolError` pattern in `geometry.py`/`imaging.py`, `_SchemaBase` (extra="forbid", round6) in `schemas.py`, `engine/image2d.py`, `plots.py` (Agg-locked), `tests/synth.py`, and the 5-step checklist in `docs/DEVELOPMENT.md`.

Conventions assumed everywhere below (inherited, not restated per tool): every tool is a thin `@mcp.tool()` closure delegating to a module-level `_impl(ctx, ...)`; returns `<Schema>.model_dump_json()`; catches ALL exceptions and returns `ToolError{error, hint, stderr_tail=None}`; unknown `mode`/`algorithm` values return ToolError whose hint lists the allowed values (the `lbs.WEIGHT_SURGERY_MODES` pattern); floats auto-round to 6 sig figs via `_SchemaBase`; no numpy array ever crosses the MCP boundary.

---

## PACK 1 — imaging v2 (ELE490), extends `src/dsp_server/toolsets/imaging.py`

Pack grows from 1 tool to **5 tools** (within the 3–6 cap). No registry change (imaging already registered). Gonzalez & Woods coverage: intensity transformations (enhance), spatial + frequency-domain filtering (filter), restoration (restore), morphology + segmentation (segment). Compression and color are deliberately out — grayscale depth renders are the domain.

**Output location convention (new):** processed images are written as **16-bit grayscale PNGs** (Pillow mode `I;16`) under `ctx.data_dir / "images"`, created on demand by a pack-local `_images_dir(ctx)` helper (`mkdir(parents=True, exist_ok=True)`). No AppContext change — the architecture stays frozen. `load_image_gray` already reads 16-bit PNGs back, so tool outputs are valid inputs to every other imaging tool (chaining: restore → enhance → segment → compare_depth_renders). Binary masks are the exception: written as 8-bit {0, 255}. Deterministic naming `"{stem}__{mode}{'__'+_safe(out_label) if out_label else ''}.png"` — re-runs overwrite, same as dump labels.

### (a) Tool inventory

**1. `enhance_image(image: str, mode: str = "histogram_eq", gamma: float = 1.0, tiles: int = 8, clip_limit: float = 0.01, out_label: str | None = None, save_plot: bool = False) -> str`**
Contract: intensity transformation of a grayscale/depth image on disk — `histogram_eq` (global), `adaptive_eq` (CLAHE-like: `tiles`×`tiles` grid, per-tile clipped histogram eq, bilinear mapping interpolation; `clip_limit` = histogram ceiling as fraction of tile pixel count, 0 disables clipping), `gamma` (s = r^gamma), `log` (s = log1p(255·r)/log1p(255)) — writes the result PNG and reports before/after histogram statistics. `save_plot` writes a before/after histogram figure.
Returns **ImageEnhanceReport**: `image, mode, params (dict[str,float] — only the params the mode used), output_path, height, width, stats_before (HistStatsSchema), stats_after, delta_entropy_bits, plot_path|null`.

**2. `filter_image(image: str, mode: str = "gaussian", sigma: float = 2.0, size: int = 3, amount: float = 1.0, cutoff_cpp: float = 0.1, order: int = 2, out_label: str | None = None) -> str`**
Contract: spatial or frequency-domain filtering — `gaussian` (sigma), `median` (size), `unsharp` (sigma, amount: out = img + amount·(img − gaussian)), `sobel` (gradient magnitude, normalized to [0,1] by its max), `butterworth_lp` / `butterworth_hp` (frequency-domain H = 1/(1+(D/D0)^(2·order)) on the radial fftfreq grid; `cutoff_cpp` in cycles/pixel, consistent with `Spectrum2DSchema` units). Output clipped to [0,1] before writing. Params not used by the chosen mode are ignored (and omitted from `params`).
Returns **ImageFilterReport**: `image, mode, params, output_path, height, width, stats_before, stats_after, rms_change (RMS of out−in), ssim_vs_input, edge_density|null` (sobel only: fraction of pixels above Otsu of the magnitude image).

**3. `segment_image(image: str, mode: str = "otsu", foreground: str = "bright", block: int = 31, offset: float = 0.02, k: int = 2, morph: str = "open_close", morph_size: int = 3, roi: list[int] | None = None, min_area_px: int = 8, max_components: int = 20, out_label: str | None = None) -> str`**
Contract: THE defect-mask tool for depth renders. Threshold modes: `otsu` (global, threshold reported), `adaptive` (local mean over a `block`×`block` uniform filter ± `offset`; threshold=null), `kmeans` (1-D Lloyd on the 256-bin gray histogram, `k` clusters; foreground = the brightest cluster for `foreground="bright"`, darkest for `"dark"`; sorted centers reported). Then binary morphology cleanup (`morph` in none/open/close/open_close with a `morph_size`² structuring element), then `ndimage.label` connected components. `roi` is the same `[x0,y0,x1,y1]` exclusive-bound crop as `compare_depth_renders`; **component bbox/centroid are ALWAYS reported in full-image pixel coordinates** (roi offset added back). Components with area < `min_area_px` are dropped; the survivors are sorted by area descending and capped at `max_components` with a truncation flag. Writes the mask as an 8-bit {0,255} PNG.
Returns **SegmentationReport**: `image, mode, foreground, threshold|null, kmeans_centers|null, morph, morph_size, roi|null, mask_path, height, width (of analyzed region), foreground_fraction, n_components_total (post min_area filter), components (list[ComponentSchema], capped), components_truncated`.

**4. `restore_image(image: str, mode: str = "wiener", kernel_size: int = 5, noise_power: float | None = None, psf_sigma: float = 2.0, nsr: float = 0.01, n_iter: int = 10, out_label: str | None = None) -> str`**
Contract: honest-scipy restoration — `wiener` (scipy.signal.wiener adaptive local-statistics **denoiser**, params kernel_size/noise_power — the docstring says explicitly it is not deconvolution), `wiener_deconv` (frequency-domain Wiener deconvolution assuming a Gaussian PSF of `psf_sigma` px: F̂ = conj(H)/(|H|²+nsr)·G), `richardson_lucy` (`n_iter` iterations, Gaussian PSF `psf_sigma`, fftconvolve-based). **No blind deconvolution, no NLM** — pure-scipy NLM at render resolution is dishonest (O(HW·S²·P²)); the docstring says so and points at `wiener`. Reports a sharpness delta so the caller can see whether restoration helped.
Returns **RestorationReport**: `image, mode, params, output_path, height, width, stats_before, stats_after, sharpness_before, sharpness_after (variance of ndimage.laplace), rms_change, ssim_vs_input`.

**5. `compare_depth_renders`** — unchanged (existing).

### (b) engine/image2d.py additions (signatures only; pure numpy/scipy/Pillow — image2d already owns image file I/O via `load_image_gray`, so `save_image_gray` lives here too)

```python
def save_image_gray(img: np.ndarray, path: Path | str) -> Path          # clip [0,1] -> uint16 I;16 PNG
@dataclass class HistStats: mean: float; std: float; p01: float; p50: float; p99: float; entropy_bits: float
def hist_stats(img: np.ndarray, bins: int = 256) -> HistStats           # entropy of the 256-bin histogram
def histogram256(img: np.ndarray, bins: int = 256) -> tuple[np.ndarray, np.ndarray]  # (counts, edges) for plots
def equalize_hist(img: np.ndarray, bins: int = 256) -> np.ndarray
def adaptive_equalize(img: np.ndarray, tiles: int = 8, clip_limit: float = 0.01, bins: int = 256) -> np.ndarray
    # reflect-pad to a tile multiple, per-tile clipped-histogram CDF mapping, bilinear interp, crop back
def adjust_gamma(img: np.ndarray, gamma: float) -> np.ndarray
def log_transform(img: np.ndarray, scale: float = 255.0) -> np.ndarray  # log1p(scale*img)/log1p(scale)
def gaussian_smooth(img: np.ndarray, sigma: float = 2.0) -> np.ndarray  # thin ndimage wrappers keep
def median_smooth(img: np.ndarray, size: int = 3) -> np.ndarray         # scipy out of toolsets/
def unsharp_mask(img: np.ndarray, sigma: float = 2.0, amount: float = 1.0) -> np.ndarray
def sobel_magnitude(img: np.ndarray) -> np.ndarray                      # hypot(sobel_y, sobel_x)/max, flat->zeros
def butterworth_filter(img: np.ndarray, cutoff_cpp: float, order: int = 2, highpass: bool = False) -> np.ndarray
def otsu_threshold(img: np.ndarray, bins: int = 256) -> float
def adaptive_threshold_mask(img: np.ndarray, block: int = 31, offset: float = 0.02,
                            foreground: str = "bright") -> np.ndarray   # bool mask
def kmeans_gray(img: np.ndarray, k: int = 2, bins: int = 256, iters: int = 50) -> tuple[np.ndarray, np.ndarray]
    # (label_img int, centers ascending) — histogram-weighted Lloyd, deterministic init at quantiles
def binary_cleanup(mask: np.ndarray, mode: str = "open_close", size: int = 3) -> np.ndarray
@dataclass class ComponentStats: label: int; area_px: int; bbox: Roi; centroid_xy: tuple[float, float]; mean_intensity: float
def component_stats(mask: np.ndarray, img: np.ndarray, offset_xy: tuple[int, int] = (0, 0),
                    min_area_px: int = 1) -> list[ComponentStats]       # offset_xy shifts bbox/centroid to full-image coords
def wiener_denoise(img: np.ndarray, kernel_size: int = 5, noise_power: float | None = None) -> np.ndarray
def gaussian_psf(sigma: float, radius: int | None = None) -> np.ndarray # radius default ceil(4*sigma), normalized
def wiener_deconvolve(img: np.ndarray, psf_sigma: float, nsr: float = 0.01) -> np.ndarray
def richardson_lucy(img: np.ndarray, psf_sigma: float, n_iter: int = 10) -> np.ndarray
def laplacian_sharpness(img: np.ndarray) -> float                       # var(ndimage.laplace(img))
```

New `plots.py` helper: `save_histogram_pair(counts_before, counts_after, bin_edges, path, title) -> Path` (single panel, two step-histograms, same style as existing helpers).

### (c) Pydantic schemas to add (all extend `_SchemaBase`)

```python
class HistStatsSchema(_SchemaBase):
    mean: float; std: float; p01: float; p50: float; p99: float; entropy_bits: float

class ImageEnhanceReport(_SchemaBase):
    image: str; mode: str; params: dict[str, float]
    output_path: str; height: int; width: int
    stats_before: HistStatsSchema; stats_after: HistStatsSchema
    delta_entropy_bits: float; plot_path: str | None = None

class ImageFilterReport(_SchemaBase):
    image: str; mode: str; params: dict[str, float]
    output_path: str; height: int; width: int
    stats_before: HistStatsSchema; stats_after: HistStatsSchema
    rms_change: float; ssim_vs_input: float
    edge_density: float | None = None

class ComponentSchema(_SchemaBase):
    label: int; area_px: int; bbox: list[int]          # [x0, y0, x1, y1], exclusive upper, full-image coords
    centroid_xy: list[float]; mean_intensity: float

class SegmentationReport(_SchemaBase):
    image: str; mode: str; foreground: str
    threshold: float | None = None; kmeans_centers: list[float] | None = None
    morph: str; morph_size: int; roi: list[int] | None = None
    mask_path: str; height: int; width: int
    foreground_fraction: float; n_components_total: int
    components: list[ComponentSchema]; components_truncated: bool

class RestorationReport(_SchemaBase):
    image: str; mode: str; params: dict[str, float]
    output_path: str; height: int; width: int
    stats_before: HistStatsSchema; stats_after: HistStatsSchema
    sharpness_before: float; sharpness_after: float
    rms_change: float; ssim_vs_input: float
```

### (d) Test plan — `tests/test_imaging_v2.py` + new shared factory `tests/synth2d.py`

`tests/synth2d.py` factories (numpy only, seeded):
```python
def make_bimodal(shape=(200, 200), lo=0.2, hi=0.8, noise=0.02, seed=0) -> np.ndarray   # left half lo, right half hi
def make_rects(shape=(128, 128)) -> np.ndarray   # bg 0; rect A x[10,30) y[10,20) val .9; B x[60,80) y[60,100) val .8; C x[100,105) y[5,10) val .7
def make_two_tone(shape=(128, 128), f_lo=0.02, f_hi=0.3, amp=0.2) -> np.ndarray        # 0.5 + amp*sin(2πf_lo·x) + amp*sin(2πf_hi·x)
def make_blocks(shape=(128, 128), block=16, seed=1) -> np.ndarray                       # random 0/1 checker-ish blocks
def make_step(shape=(128, 128), col=64) -> np.ndarray                                   # 0 left of col, 1 from col
```

Golden cases (expected numbers):
1. `adjust_gamma(const 0.25, gamma=0.5)` → every pixel exactly `0.5`; `log_transform` endpoints: 0→0, 1→1 exactly.
2. `equalize_hist` on `np.linspace(0, 0.5, 256*256).reshape(256,256)` (low-contrast ramp): `stats_after.p99 > 0.95`, `delta_entropy_bits >= 0`; on a full-range uniform ramp, output ≈ input within 1/256 (fixed point).
3. `adaptive_equalize(tiles=4)` on a dark-gradient + faint-texture image: std inside the darkest 32×32 tile strictly increases vs input; output within [0,1].
4. `otsu_threshold(make_bimodal(seed=0))` ∈ (0.4, 0.6); segment_image mode=otsu → `foreground_fraction` = 0.5 ± 0.02.
5. `segment_image` on `make_rects()`, otsu, morph="none", min_area_px=8 → exactly `n_components_total == 3`, areas [800, 200, 25] in that order, bbox B == [60, 60, 80, 100], centroid B == [69.5, 79.5]; with `roi=[50, 50, 128, 128]` → 1 component whose bbox is still [60, 60, 80, 100] (full-image coords).
6. `butterworth_filter(make_two_tone(), cutoff_cpp=0.1, order=2)`: FFT-bin amplitude at f=0.3 attenuated by ≥ 8× (theoretical |H| = 1/√82 ≈ 0.11); amplitude at f=0.02 within 5% of input. `butterworth_hp` reverses which tone survives.
7. `sobel_magnitude(make_step(col=64))`: argmax of the column-mean magnitude ∈ {63, 64}; `edge_density` > 0.
8. Restoration: clean = `make_blocks(seed=1)`; blurred = gaussian_smooth(clean, 2.0) + N(0, 0.005) (seed 2). Both `wiener_deconv(psf_sigma=2, nsr=0.01)` and `richardson_lucy(psf_sigma=2, n_iter=10)`: `laplacian_sharpness(restored) > laplacian_sharpness(blurred)` AND `ssim(restored, clean) > ssim(blurred, clean)`.
9. Round-trip: `save_image_gray` then `load_image_gray` → max abs error ≤ 1/65535 + eps.
10. Error paths (call `_impl` directly): unknown mode → ToolError JSON whose hint lists allowed modes; missing file → ToolError; `segment_image` empty roi → ToolError (reuses `_ROI_HINT`).

### (e) llms.txt rule addition

```
12. Imaging pack (5 tools): processed outputs are 16-bit grayscale PNGs under data/images/ (masks
    8-bit 0/255) — paths + compact stats only, and outputs are valid inputs to every imaging tool.
    segment_image is the defect-mask tool: component bbox/centroid are ALWAYS full-image pixel
    coords even with roi. Restoration is honest scipy — adaptive Wiener denoise, Gaussian-PSF
    Wiener deconvolution, Richardson–Lucy; no blind deconvolution, no NLM.
```
(Also: whichever pack lands last updates rule 11's "Six MCP tools" count.)

### (f) Shared with other packs
- `tests/synth2d.py` is the 2-D analogue of `tests/synth.py` — the ml pack (ELE489) can reuse `make_bimodal`/`make_blocks` for clustering/classification fixtures.
- `save_image_gray` + `data/images/` convention available to any pack that emits images.
- `cutoff_cpp` (cycles/pixel) unit is shared with `Spectrum2DSchema` — no new unit introduced.
- No dependency on `tabular.py` (stats designer owns it); imaging inputs are image paths only.

---

## PACK 2 — netqueue (ELE412 queueing mini), new `src/dsp_server/toolsets/netqueue.py`

**2 tools** (mini-pack scope approved at 1–2). Pure closed-form math, zero new deps, no files written, no plots. Registry: add `_register_netqueue` + `"netqueue"` key in `toolsets/__init__.py`.

### (a) Tool inventory

**1. `queueing_calc(model: str = "mm1", arrival_rate: float, service_rate: float, servers: int = 1) -> str`**
(Implementation note: FastMCP needs defaults last — actual signature order `arrival_rate: float, service_rate: float, model: str = "mm1", servers: int = 1`.)
Contract: closed-form steady-state metrics for `mm1` (M/M/1), `mmm` (M/M/m, uses `servers`), `md1` (M/D/1 — Lq/Wq exact via Pollaczek–Khinchine, not an approximation). `service_rate` is per server. Time units are whatever λ/μ are given in; W/Wq inherit them. `p_wait` (probability an arrival must queue): ρ for mm1 and md1 (PASTA — P(server busy) = ρ in any M/G/1), Erlang-C for mmm. **Stability gate:** ρ = λ/(m·μ) ≥ 1 returns ToolError `"unstable queue: rho >= 1"` with hint `"rho=<value> — need service_rate > <λ/m> or servers >= <ceil(λ/μ)+adjust>"` (the minimum integer m with λ/(m·μ) < 1). Nonpositive rates, `servers < 1`, or `servers != 1` for mm1/md1 → ToolError.
Returns **QueueingReport**: `model, arrival_rate, service_rate, servers, rho, offered_load_erlangs (a = λ/μ), p0, l_system (L), l_queue (Lq), w_system (W), w_queue (Wq), p_wait, erlang_c|null (mmm only, == p_wait — kept as an explicitly named field), notes|null` (md1 sets notes="M/D/1: Lq/Wq exact via Pollaczek-Khinchine").

**2. `little_law(l_avg: float | None = None, arrival_rate: float | None = None, w_avg: float | None = None) -> str`**
Contract: Little's theorem L = λW — give exactly two of {l_avg, arrival_rate, w_avg}, get the third. Applies unchanged to queue-only quantities (Lq = λ·Wq) — the docstring says to pass Lq/Wq in the same slots. Not-exactly-two given, or any nonpositive value → ToolError with hint naming which arguments were received.
Returns **LittleLawReport**: `l_avg, arrival_rate, w_avg, solved_for` (one of `"l_avg" | "arrival_rate" | "w_avg"`).

(Field names `l_system`/`l_queue`/`l_avg` deliberately avoid a bare `l` — ruff E741 bans it.)

### (b) engine/queueing.py (new module; pure functions, no I/O, no numpy needed but allowed)

```python
@dataclass(frozen=True)
class QueueMetrics:
    model: str; lam: float; mu: float; servers: int
    rho: float; offered_load: float
    p0: float; lq: float; l: float; wq: float; w: float
    p_wait: float; erlang_c: float | None

def mm1_metrics(lam: float, mu: float) -> QueueMetrics
def mmm_metrics(lam: float, mu: float, m: int) -> QueueMetrics
def md1_metrics(lam: float, mu: float) -> QueueMetrics
def erlang_b(a: float, m: int) -> float   # recursive B(k)=a·B(k-1)/(k+a·B(k-1)) — numerically stable
def erlang_c(a: float, m: int) -> float   # C = m·B/(m − a·(1−B))
def solve_little(l_avg: float | None, lam: float | None, w: float | None) -> tuple[float, float, float, str]
```
All raise `ValueError(f"unstable: rho={rho:.6g} >= 1 (need mu > {lam/m:.6g} per server or servers >= {m_min})")` on instability; the tool converts to ToolError verbatim (utilization lands in the hint per the locked requirement).
Formulas: M/M/1: P0=1−ρ, L=ρ/(1−ρ), Lq=ρ²/(1−ρ), W=1/(μ−λ), Wq=ρ/(μ−λ), Pwait=ρ. M/M/m: P0 = [Σ_{k<m} aᵏ/k! + aᵐ/(m!(1−ρ))]⁻¹, C=Erlang-C, Lq=C·ρ/(1−ρ), Wq=Lq/λ, W=Wq+1/μ, L=λW. M/D/1: Lq=ρ²/(2(1−ρ)), Wq=Lq/λ, W=Wq+1/μ, L=λW, P0=1−ρ, Pwait=ρ.

### (c) Pydantic schemas

```python
class QueueingReport(_SchemaBase):
    model: str; arrival_rate: float; service_rate: float; servers: int
    rho: float; offered_load_erlangs: float
    p0: float; l_system: float; l_queue: float; w_system: float; w_queue: float
    p_wait: float; erlang_c: float | None = None
    notes: str | None = None

class LittleLawReport(_SchemaBase):
    l_avg: float; arrival_rate: float; w_avg: float; solved_for: str
```

### (d) Test plan — `tests/test_netqueue.py` (all golden, closed-form)

1. M/M/1 λ=2, μ=3: rho=0.666667, L=2.0, Lq=1.33333, W=1.0, Wq=0.666667, P0=0.333333, p_wait=0.666667.
2. M/M/4 λ=3, μ=1 (a=3, ρ=0.75): P0=0.037736 (=1/26.5), erlang_c=p_wait=0.509434, Lq=1.5283, Wq=0.509434, W=1.50943, L=4.5283. Also `erlang_b(3,4)=0.206107`.
3. M/D/1 λ=2, μ=3: Lq=0.666667 — exactly half of M/M/1's Lq (the classic teaching point, asserted as `md1.lq == pytest.approx(mm1.lq / 2)`), Wq=0.333333, W=0.666667, L=1.33333, p_wait=0.666667, erlang_c is null, notes mentions "Pollaczek".
4. Consistency: for every model, L == λ·W and Lq == λ·Wq to 1e-9 (Little's law as an internal invariant).
5. Instability: λ=5, μ=4, mm1 → ToolError; hint contains `"rho=1.25"` and `"servers >= 2"`. λ=4, μ=4 (ρ=1 exactly) → also ToolError.
6. `little_law(l_avg=4.5283, arrival_rate=3)` → w_avg=1.50943, solved_for="w_avg"; one arg only or all three → ToolError; negative value → ToolError.
7. Degenerate mmm with servers=1 == mm1 field-for-field.

### (e) llms.txt rule addition

```
13. netqueue is closed-form only (M/M/1, M/M/m, M/D/1 — the M/D/1 Lq/Wq are EXACT
    Pollaczek-Khinchine, not simulation). rho >= 1 returns ToolError with the utilization and the
    minimum stable server count in the hint. Time units are whatever lambda/mu were given in;
    p_wait is Erlang-C for mmm and rho for mm1/md1.
```

### (f) Shared
- Nothing shared outward; consumes only `ToolError`/`_SchemaBase`. `erlang_b`/`erlang_c` in `engine/queueing.py` are available to the stats pack should it ever want traffic examples. No `tabular.py` dependency — inputs are scalars.

---

## PACK 3 — os (Tanenbaum mini), new `src/dsp_server/toolsets/os_sim.py`

**3 tools** (mini scope approved at 2–3). Deterministic, stdlib + numpy only. **Module is named `os_sim.py`** (never `os.py` — stdlib shadow hazard for tooling), registry key is `"os"`: add `_register_os` importing `dsp_server.toolsets.os_sim` + `"os"` key in TOOLSETS. Math module: `engine/osalgo.py`. All three tools are table-heavy — every table is capped with an explicit truncation signal.

### (a) Tool inventory

**1. `schedule_sim(processes: list[dict], algorithm: str = "fcfs", quantum: float = 2.0) -> str`**
Contract: single-CPU scheduling simulation. `processes` = list of `{name: str, arrival: float >= 0, burst: float > 0, priority?: int}` (priority defaults 0; **lower number = higher priority**), 1–50 processes, names must be unique. Algorithms: `fcfs`, `sjf` (non-preemptive shortest job first among arrived), `srtf` (preemptive SJF), `rr` (round-robin, `quantum > 0`, ready-queue order: arrivals enqueue before the preempted process at the same instant), `priority` (non-preemptive). **All ties break by (arrival time, input order)** — fully deterministic. Zero context-switch cost. CPU idles forward to the next arrival when no process is ready. Definitions: wait = turnaround − burst, turnaround = completion − arrival, response = start − arrival (first time on CPU). Gantt is a text string of `name[start-end)` segments joined by `" | "` (idle gaps as `idle[a-b)`), capped at 60 segments then `" … (+N more)"`.
Returns **ScheduleReport**: `algorithm, quantum|null (rr only), n_processes, per_process (list[ProcStatsSchema]), avg_wait, avg_turnaround, avg_response, makespan, cpu_idle, gantt, gantt_segments, gantt_truncated`.

**2. `page_replacement_sim(reference: list[int], frames: int, algorithm: str = "lru") -> str`**
Contract: page-replacement simulation over a reference string (ints ≥ 0, length 1–2000) with `frames` 1–64. Algorithms: `fifo`, `lru`, `opt` (Belady offline optimal; tie among pages never used again → evict the one in the lowest frame slot), `clock` (second chance; **reference bit set on load AND on every hit**; victim = first frame with clear bit as the hand sweeps, clearing set bits as it passes). Per-step table trimmed to the **first 50 steps** with `steps_truncated`. The tool **always additionally runs OPT** on the same input and reports it as the offline lower bound with a comparative hint string, e.g. `"lru: 12 faults vs OPT 9 (+3, 33.3% above the offline optimum)"` (or `"opt is the offline lower bound"` when algorithm=opt).
Returns **PagingReport**: `algorithm, n_frames, ref_length, faults, hits, fault_rate, opt_faults, vs_opt, steps (list[PageStepSchema], ≤50), steps_truncated`.

**3. `bankers_check(available: list[int], allocation: list[list[int]], max_demand: list[list[int]], process_names: list[str] | None = None, request_process: int | None = None, request: list[int] | None = None) -> str`**
Contract: Banker's algorithm. Validates: ≤32 processes × ≤16 resource types, consistent dimensions, non-negative integers, `max_demand >= allocation` elementwise (ToolError names the violating process/resource cell). Safety scan is **deterministic lowest-index-first** (repeatedly sweep P0..Pn-1, run the first process whose need ≤ work) → `safe` + the produced `safe_sequence`, or the list of `stuck` processes with an `explanation` naming each stuck process's unmet need vs available (e.g. `"unsafe: P0 needs [7,2,3] but only [2,1,0] can ever become available; ..."`). With `request_process` + `request`: resource-request algorithm — deny with reason `"exceeds need"` / `"exceeds available"` without state change, else pretend-grant and run safety on the resulting state: granted iff safe (reason reports the post-grant sequence or the stuck set). Names default `P0..P{n-1}`.
Returns **BankersReport**: `n_processes, n_resources, safe, safe_sequence (names; empty when unsafe), stuck (names; empty when safe), need (list[list[int]] — small by the 32×16 cap), explanation, request_granted|null, request_reason|null`. When a request is evaluated, `safe`/`safe_sequence`/`stuck`/`explanation` describe the **post-pretend-grant** state (denied-for-bounds requests report the unchanged original state, with `request_reason` saying why).

### (b) engine/osalgo.py (new module; pure functions, no I/O)

```python
@dataclass(frozen=True)
class Proc: name: str; arrival: float; burst: float; priority: int = 0

@dataclass(frozen=True)
class ProcStats:
    name: str; arrival: float; burst: float; priority: int
    start: float; completion: float; wait: float; turnaround: float; response: float

@dataclass(frozen=True)
class ScheduleResult:
    segments: list[tuple[str, float, float]]   # (name-or-"idle", start, end), merged consecutive same-name
    per_process: list[ProcStats]               # input order
    makespan: float; idle_time: float

SCHEDULE_ALGORITHMS = ("fcfs", "sjf", "srtf", "rr", "priority")
def simulate_schedule(procs: list[Proc], algorithm: str, quantum: float = 2.0) -> ScheduleResult
def gantt_text(segments: list[tuple[str, float, float]], max_segments: int = 60) -> tuple[str, int, bool]
    # (text, n_segments_total, truncated); numbers formatted %g

@dataclass(frozen=True)
class PageStep: step: int; page: int; fault: bool; evicted: int | None; frames: tuple[int, ...]

PAGING_ALGORITHMS = ("fifo", "lru", "opt", "clock")
def simulate_paging(reference: list[int], n_frames: int, algorithm: str) -> tuple[list[PageStep], int]  # (steps, faults)

@dataclass(frozen=True)
class BankersOutcome:
    safe: bool; sequence: list[int]; stuck: list[int]            # process indices
    need: list[list[int]]; explanation: str

def bankers_safety(available: list[int], allocation: list[list[int]],
                   max_demand: list[list[int]]) -> BankersOutcome
def bankers_request(available: list[int], allocation: list[list[int]], max_demand: list[list[int]],
                    pid: int, request: list[int]) -> tuple[bool, str, BankersOutcome]
    # (granted, reason, outcome-of-evaluated-state)
```

### (c) Pydantic schemas

```python
class ProcStatsSchema(_SchemaBase):
    name: str; arrival: float; burst: float; priority: int
    start: float; completion: float; wait: float; turnaround: float; response: float

class ScheduleReport(_SchemaBase):
    algorithm: str; quantum: float | None = None
    n_processes: int; per_process: list[ProcStatsSchema]
    avg_wait: float; avg_turnaround: float; avg_response: float
    makespan: float; cpu_idle: float
    gantt: str; gantt_segments: int; gantt_truncated: bool

class PageStepSchema(_SchemaBase):
    step: int; page: int; fault: bool
    evicted: int | None = None; frames: list[int]

class PagingReport(_SchemaBase):
    algorithm: str; n_frames: int; ref_length: int
    faults: int; hits: int; fault_rate: float
    opt_faults: int; vs_opt: str
    steps: list[PageStepSchema]; steps_truncated: bool

class BankersReport(_SchemaBase):
    n_processes: int; n_resources: int
    safe: bool; safe_sequence: list[str]; stuck: list[str]
    need: list[list[int]]; explanation: str
    request_granted: bool | None = None; request_reason: str | None = None
```

### (d) Test plan — `tests/test_os_sim.py` (textbook-golden, all hand-verifiable)

Scheduling (classic Silberschatz/Tanenbaum sets, P1(arrival,burst[,prio])):
1. FCFS, P1(0,24) P2(0,3) P3(0,3): waits [0,24,27], avg_wait=17.0, makespan=30, gantt `"P1[0-24) | P2[24-27) | P3[27-30)"`.
2. SJF, same set: order P2,P3,P1 → waits [6,0,3], avg_wait=3.0.
3. RR q=4, same set: avg_wait=5.66667 (waits P1=6, P2=4, P3=7), avg_response=3.66667 (responses 0,4,7) — asserts response ≠ wait under RR; P1's gantt shows 6 slices.
4. SRTF, P1(0,8) P2(1,4) P3(2,9) P4(3,5): segments P1[0-1) P2[1-5) P4[5-10) P1[10-17) P3[17-26); waits [9,0,15,2], avg_wait=6.5.
5. Priority, P1(0,10,3) P2(0,1,1) P3(0,2,4) P4(0,1,5) P5(0,5,2): order P2,P5,P1,P3,P4, avg_wait=8.2.
6. Idle gap: P1(0,2) P2(5,2) FCFS → gantt contains `idle[2-5)`, cpu_idle=3, makespan=7.
7. Validation: duplicate names / burst≤0 / 51 processes / rr with quantum≤0 → ToolError.

Paging:
8. Classic 20-ref string `[7,0,1,2,0,3,0,4,2,3,0,3,2,1,2,0,1,7,0,1]`, 3 frames: FIFO=15 faults, LRU=12, OPT=9 (Silberschatz's exact numbers); `opt_faults=9` on every run; `vs_opt` for LRU contains `"12"` and `"9"`.
9. Belady's anomaly: ref `[1,2,3,4,1,2,5,1,2,3,4,5]` FIFO: 3 frames → 9 faults, 4 frames → 10 faults (asserted as a pair).
10. Clock hand-verified: ref `[1,2,3,1,4]`, 3 frames → 4 faults; step 5 evicts page **1** (bit set on load+hit convention: sweep clears 1,2,3 then evicts 1), frames after = [4,2,3].
11. Truncation: 120-ref random string (seed 3) → len(steps)==50, steps_truncated=True, faults counted over all 120.

Banker's (Silberschatz's canonical 5×3 instance): available=[3,3,2], alloc=[[0,1,0],[2,0,0],[3,0,2],[2,1,1],[0,0,2]], max=[[7,5,3],[3,2,2],[9,0,2],[2,2,2],[4,3,3]]:
12. Safety → safe=True, safe_sequence == [P1,P3,P4,P0,P2] (the deterministic lowest-index scan), need row 0 == [7,4,3].
13. Request (1,0,2) by P1 → granted=True, post-state safe.
14. After granting 13 (available=[2,3,0], P1 alloc=[3,0,2]): request (3,3,0) by P4 → denied "exceeds available"; request (0,2,0) by P0 → denied, post-pretend state unsafe with all 5 processes stuck (textbook result).
15. Validation: max < alloc at a named cell → ToolError naming `"P2 resource 1"`; 33 processes → ToolError.

### (e) llms.txt rule addition

```
14. os pack is deterministic by contract: scheduling ties break by (arrival, input order) and
    LOWER priority number = higher priority; clock sets the reference bit on load AND on hit;
    page_replacement_sim always co-runs OPT as the offline lower bound. Tables are hard-capped
    (50 processes, first 50 paging steps, 32x16 banker) with explicit *_truncated flags.
```

### (f) Shared
- Nothing consumed beyond `ToolError`/`_SchemaBase`; no numpy arrays to disk, no plots, no `tabular.py`. `gantt_text`'s cap-then-`"… (+N more)"` convention is the recommended pattern for any future pack emitting text tables.

---

## Cross-pack integration notes (for the implementer)

- **Registry diff** (`toolsets/__init__.py`): two new loader functions (`_register_netqueue` → `toolsets.netqueue`, `_register_os` → `toolsets.os_sim`) + two TOOLSETS entries `"netqueue"`, `"os"`. imaging: no registry change.
- **Tool totals**: geometry 5 + imaging 5 + netqueue 2 + os 3 = 15 from these packs; llms.txt rule 11's "Six MCP tools" sentence must be rewritten by whichever pack implementation lands last (coordinate with the stats/engmath/systems/ml designers).
- **New deps**: none for all three packs (numpy/scipy/Pillow/matplotlib already present) — cloud-safe by construction.
- **Files created per pack**: imaging → edits `toolsets/imaging.py`, `engine/image2d.py`, `plots.py`, `schemas.py`, adds `tests/synth2d.py` + `tests/test_imaging_v2.py`; netqueue → adds `toolsets/netqueue.py`, `engine/queueing.py`, `tests/test_netqueue.py`, edits `schemas.py` + registry; os → adds `toolsets/os_sim.py`, `engine/osalgo.py`, `tests/test_os_sim.py`, edits `schemas.py` + registry.
- **Only genuinely shared new surface**: `tests/synth2d.py` (imaging ↔ future ml pack) and the `data/images/` + 16-bit-PNG output convention (any future image-emitting tool). Neither mini-pack touches `tabular.py`, `Signal1D`, or the engine bridge.
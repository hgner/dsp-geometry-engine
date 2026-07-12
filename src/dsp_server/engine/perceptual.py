"""Classical perceptual / semantic gates for the video comparison suite (pure scipy).

The DSP/geometry gates measure MATHEMATICAL parity; a generative model that shifts a
wood-grain texture by 3 px is a huge FFT/wavelet error yet perceptually identical.
This module closes that gap WITHOUT a neural net (no torch, no downloaded weights —
the server stays a pure-wheel, deterministic, cloud/cron-safe measurement product):

* :func:`cw_ssim` — Complex-Wavelet SSIM (Sampat et al. 2009) over a complex Gabor
  bank. A small spatial shift only PHASE-shifts the complex coefficients, and CW-SSIM
  compares the MAGNITUDE of their correlation, so it is insensitive to a few-pixel
  translation — the classical answer to "shifted but looks identical". This is the
  headline perceptual metric; plain SSIM (pixel-structural) is reported alongside so
  the caller can see the gap (low SSIM + high CW-SSIM = a shift-only difference).
* :func:`ms_ssim` — multi-scale SSIM (per-scale SSIM, geometric mean): scale-aware
  perceptual similarity.
* :func:`region_descriptor` / :func:`identity_coherence` — a per-frame colour +
  texture signature of a masked object, and its cosine drift across a clip: catches
  an object that MORPHS (colour/texture identity change) even when optical flow says
  the pixels moved coherently. A classical stand-in for a DINOv2 semantic tracker —
  it flags gross colour/texture/shape morphing, not fine semantic identity.

Pure numpy + scipy (FFT convolution, uniform/gaussian filters). Deterministic. Frame
I/O is owned by :mod:`dsp_server.engine.image2d` / :mod:`dsp_server.engine.video`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve

__all__ = [
    "IdentityCoherenceResult",
    "PerceptualResult",
    "analyze_perceptual_similarity",
    "cw_ssim",
    "gabor_bank",
    "identity_coherence",
    "ms_ssim",
    "region_descriptor",
]

_TINY = 1e-12
# Complex Gabor bank: a few octave-spaced radial frequencies x evenly spaced orientations.
_FREQS_CPP = (0.12, 0.25)  # cycles/pixel
_N_ORIENT = 6


@dataclass
class PerceptualResult:
    """Perceptual similarity of a generated frame vs a reference (higher = closer).

    ``cw_ssim`` is shift-tolerant (the headline: a few-pixel texture shift stays ~1);
    ``ms_ssim`` is multi-scale structural; ``ssim`` is single-scale pixel-structural.
    ``shift_tolerant_gap = cw_ssim - ssim`` is large when the frames differ mainly by
    a small spatial shift (mathematically divergent, perceptually equivalent).
    ``verdict``: ``perceptually-equivalent`` / ``perceptually-degraded`` / ``distinct``.
    """

    height: int
    width: int
    cw_ssim: float
    ms_ssim: float
    ssim: float
    shift_tolerant_gap: float
    verdict: str
    verdict_reason: str


@dataclass
class IdentityCoherenceResult:
    """Semantic-identity coherence of a masked object across a clip.

    ``coherence_by_frame`` is the cosine similarity of each frame's colour+texture
    descriptor to frame 0 (index 0 is 1.0 by construction); it lives on disk, not
    inline. ``min_coherence`` at ``min_coherence_frame`` is the worst drift;
    ``first_break_frame`` is the first frame below ``break_threshold`` (or -1).
    ``verdict``: ``coherent`` / ``identity-drift`` / ``morph``.
    """

    n_frames: int
    mask_pixel_count: int
    break_threshold: float
    coherence_by_frame: np.ndarray
    mean_coherence: float
    min_coherence: float
    min_coherence_frame: int
    first_break_frame: int
    verdict: str
    verdict_reason: str


# --------------------------------------------------------------------------- #
# Complex Gabor bank


def _gabor_kernel(freq: float, theta: float) -> np.ndarray:
    """One zero-DC complex Gabor kernel at radial frequency ``freq`` (cyc/px) and
    orientation ``theta`` (rad). Sigma is tied to the frequency (~1-octave band)."""
    sigma = 0.5 / float(freq)
    half = int(np.ceil(2.5 * sigma))
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float64)
    xr = xx * np.cos(theta) + yy * np.sin(theta)
    yr = -xx * np.sin(theta) + yy * np.cos(theta)
    envelope = np.exp(-(xr * xr + yr * yr) / (2.0 * sigma * sigma))
    kernel = envelope * np.exp(1j * 2.0 * np.pi * float(freq) * xr)
    return kernel - kernel.mean()  # zero DC: flat regions give ~0 response


def gabor_bank(freqs: tuple[float, ...] = _FREQS_CPP, n_orient: int = _N_ORIENT) -> list[np.ndarray]:
    """Complex Gabor filterbank: ``len(freqs) * n_orient`` zero-DC kernels."""
    thetas = np.pi * np.arange(n_orient) / n_orient
    return [_gabor_kernel(f, float(t)) for f in freqs for t in thetas]


def _responses(img: np.ndarray, bank: list[np.ndarray]) -> list[np.ndarray]:
    """Complex Gabor responses of ``img`` (FFT convolution, same size)."""
    arr = np.asarray(img, dtype=np.float64)
    return [fftconvolve(arr, k, mode="same") for k in bank]


# --------------------------------------------------------------------------- #
# CW-SSIM (shift-tolerant) + MS-SSIM + single-scale SSIM


def cw_ssim(ref: np.ndarray, gen: np.ndarray, bank: list[np.ndarray] | None = None) -> float:
    """Complex-Wavelet SSIM in [0, 1] over a Gabor bank — insensitive to small shifts.

    Per subband and per local window (scaled to the subband's kernel):
    ``(2|<c1, c2*>| + K) / (<|c1|^2> + <|c2|^2> + K)``. A pure phase shift (a small
    translation) leaves this at 1; uncorrelated content drives it toward 0. The
    per-window values are energy-weighted (flat windows are uninformative) and
    averaged over subbands.
    """
    a = np.asarray(ref, dtype=np.float64)
    b = np.asarray(gen, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"cw_ssim needs same-shape frames, got {a.shape} vs {b.shape}")
    if a.ndim != 2:
        raise ValueError(f"expected 2-D grayscale frames, got shape {a.shape}")
    if bank is None:
        bank = gabor_bank()
    # Subtract each frame's mean before convolution: fftconvolve zero-pads, so a
    # zero-DC Gabor kernel hanging off the edge sees the frame's DC level and emits a
    # large border ring whose energy scales with brightness — on a flat frame that ring
    # would dominate the energy-weighted score (a false "degraded"). Centering makes the
    # off-edge region ~0, killing the artifact; the interior structural response (the
    # kernel already has zero DC) is unchanged.
    a = a - a.mean()
    b = b - b.mean()
    resp_a = _responses(a, bank)
    resp_b = _responses(b, bank)
    scores: list[float] = []
    weights: list[float] = []
    for c1, c2, kern in zip(resp_a, resp_b, bank, strict=True):
        # The averaging window must span several cycles of the subband, else a
        # low-frequency band (whose response is correlated over its kernel width) has
        # ~1 independent sample per window and |<c1,c2*>| ~ |c1||c2| for ANY pair,
        # faking high similarity on unrelated content. Scale the window to the kernel.
        win = min(kern.shape[0], min(a.shape))
        win = win if win % 2 == 1 else win - 1
        win = max(win, 3)
        pad = win // 2
        mag1 = ndimage.uniform_filter(np.abs(c1) ** 2, size=win, mode="reflect")
        mag2 = ndimage.uniform_filter(np.abs(c2) ** 2, size=win, mode="reflect")
        cross = c1 * np.conj(c2)
        cross_re = ndimage.uniform_filter(cross.real, size=win, mode="reflect")
        cross_im = ndimage.uniform_filter(cross.imag, size=win, mode="reflect")
        cross_mag = np.hypot(cross_re, cross_im)
        energy = mag1 + mag2
        k = 1e-3 * float(energy.mean()) + _TINY  # small: only guards true-zero windows
        cw_map = (2.0 * cross_mag + k) / (energy + k)
        # Energy-weight so uninformative flat windows (where the ratio -> 1 for ANY pair)
        # don't inflate the score; the comparison only counts where there is texture.
        core = cw_map[pad : cw_map.shape[0] - pad, pad : cw_map.shape[1] - pad]
        w = energy[pad : energy.shape[0] - pad, pad : energy.shape[1] - pad]
        if core.size and float(w.sum()) > _TINY:
            scores.append(float((core * w).sum() / w.sum()))
            weights.append(float(w.sum()))
        else:
            scores.append(float(core.mean() if core.size else cw_map.mean()))
            weights.append(1.0)
    wv = np.asarray(weights)
    return float(np.average(np.asarray(scores), weights=wv if wv.sum() > _TINY else None))


def _ssim(a: np.ndarray, b: np.ndarray, window: int = 11) -> float:
    """Mean SSIM assuming inputs already in [0, 1] (data_range = 1); no renormalization,
    so it composes correctly across scales (unlike image2d.ssim's min-max version)."""
    c1 = 0.01**2
    c2 = 0.03**2

    def uf(x: np.ndarray) -> np.ndarray:
        return ndimage.uniform_filter(x, size=window, mode="reflect")

    mu_a, mu_b = uf(a), uf(b)
    var_a = uf(a * a) - mu_a**2
    var_b = uf(b * b) - mu_b**2
    cov = uf(a * b) - mu_a * mu_b
    ssim_map = ((2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )
    pad = window // 2
    core = ssim_map[pad : ssim_map.shape[0] - pad, pad : ssim_map.shape[1] - pad]
    return float(core.mean() if core.size else ssim_map.mean())


def ms_ssim(ref: np.ndarray, gen: np.ndarray, scales: int = 4, min_size: int = 24) -> float:
    """Multi-scale SSIM: geometric mean of single-scale SSIM over halving scales.

    A simplified Wang-2003 MS-SSIM (equal weighting, geometric mean); stops before a
    level would fall below ``min_size`` pixels."""
    a = np.clip(np.asarray(ref, dtype=np.float64), 0.0, 1.0)
    b = np.clip(np.asarray(gen, dtype=np.float64), 0.0, 1.0)
    if a.shape != b.shape:
        raise ValueError(f"ms_ssim needs same-shape frames, got {a.shape} vs {b.shape}")
    vals: list[float] = []
    for _ in range(max(1, int(scales))):
        vals.append(_ssim(a, b))
        if min(a.shape) // 2 < int(min_size):
            break
        a = ndimage.zoom(ndimage.gaussian_filter(a, 1.0, mode="reflect"), 0.5, order=1)
        b = ndimage.zoom(ndimage.gaussian_filter(b, 1.0, mode="reflect"), 0.5, order=1)
    clipped = np.clip(np.asarray(vals), 1e-6, 1.0)
    return float(np.exp(np.mean(np.log(clipped))))


def analyze_perceptual_similarity(
    ref: np.ndarray,
    gen: np.ndarray,
    equivalent_cw: float = 0.85,
    distinct_cw: float = 0.55,
) -> PerceptualResult:
    """CW-SSIM (shift-tolerant) + MS-SSIM + SSIM of two grayscale frames, with a verdict.

    ``verdict``: ``perceptually-equivalent`` (cw_ssim >= equivalent_cw),
    ``distinct`` (cw_ssim < distinct_cw), else ``perceptually-degraded``.
    """
    a = np.asarray(ref, dtype=np.float64)
    b = np.asarray(gen, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"frames must match: {a.shape} vs {b.shape}")
    if a.ndim != 2:
        raise ValueError(f"expected 2-D grayscale frames, got shape {a.shape}")
    bank = gabor_bank()
    cw = cw_ssim(a, b, bank)
    ms = ms_ssim(a, b)
    ss = _ssim(np.clip(a, 0.0, 1.0), np.clip(b, 0.0, 1.0))
    gap = cw - ss

    if cw >= float(equivalent_cw):
        verdict = "perceptually-equivalent"
        reason = (
            f"CW-SSIM {cw:.3g} >= {equivalent_cw} — the generated frame preserves the reference's "
            "perceptual content"
        )
        if gap > 0.15:
            reason += (
                f" despite a much lower pixel SSIM {ss:.3g} (gap {gap:.3g}) — the difference is "
                "mostly a small spatial shift, not a real distortion"
            )
    elif cw < float(distinct_cw):
        verdict = "distinct"
        reason = f"CW-SSIM {cw:.3g} < {distinct_cw} — the frames are perceptually different content"
    else:
        verdict = "perceptually-degraded"
        reason = (
            f"CW-SSIM {cw:.3g} in [{distinct_cw}, {equivalent_cw}) — perceptible degradation beyond "
            "a benign shift"
        )
    return PerceptualResult(
        height=int(a.shape[0]),
        width=int(a.shape[1]),
        cw_ssim=cw,
        ms_ssim=ms,
        ssim=ss,
        shift_tolerant_gap=float(gap),
        verdict=verdict,
        verdict_reason=reason,
    )


# --------------------------------------------------------------------------- #
# Semantic-identity coherence (colour + texture descriptor drift)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def region_descriptor(rgb: np.ndarray, mask: np.ndarray, bank: list[np.ndarray], bins: int = 8) -> np.ndarray:
    """Colour+texture signature of ``rgb`` over the boolean ``mask`` (unit L2 norm).

    Colour: per-channel intensity histogram (``bins`` each). Texture: mean Gabor
    response magnitude per subband over the masked region. The two blocks are each
    L2-normalized (so colour and texture weigh comparably) then concatenated and
    L2-normalized, so a plain dot product between two descriptors is their cosine.
    """
    m = np.asarray(mask, dtype=bool)
    colour = []
    for c in range(3):
        hist, _ = np.histogram(rgb[:, :, c][m], bins=bins, range=(0.0, 1.0))
        colour.append(hist.astype(np.float64))
    colour_vec = np.concatenate(colour)
    colour_vec /= np.linalg.norm(colour_vec) + _TINY

    gray = _luma(rgb)
    texture = np.array([np.abs(fftconvolve(gray, k, mode="same"))[m].mean() for k in bank])
    texture_vec = texture / (np.linalg.norm(texture) + _TINY)

    desc = np.concatenate([colour_vec, texture_vec])
    return desc / (np.linalg.norm(desc) + _TINY)


def identity_coherence(
    frames_rgb: np.ndarray, mask: np.ndarray, break_threshold: float = 0.8
) -> IdentityCoherenceResult:
    """Cosine drift of a masked object's descriptor across an ``(T, H, W, 3)`` clip.

    Each frame's colour+texture descriptor is compared (cosine) to frame 0. A drop
    below ``break_threshold`` marks a morph; the ``min_coherence_frame`` localizes the
    worst change (e.g. the exact frame a jacket's material identity flips).
    """
    vol = np.asarray(frames_rgb, dtype=np.float64)
    if vol.ndim != 4 or vol.shape[3] < 3:
        raise ValueError(f"expected a (T, H, W, 3) RGB clip, got shape {vol.shape}")
    m = np.asarray(mask, dtype=bool)
    if m.shape != vol.shape[1:3]:
        raise ValueError(f"mask {m.shape} must match the frame {vol.shape[1:3]}")
    n_mask = int(m.sum())
    if n_mask < 16:
        raise ValueError(
            f"object mask covers only {n_mask} px (need >= 16) — supply a larger segmentation/ID region"
        )
    t = int(vol.shape[0])
    bank = gabor_bank()
    descriptors = [region_descriptor(vol[i], m, bank) for i in range(t)]
    ref = descriptors[0]
    coherence = np.array([float(np.dot(ref, d)) for d in descriptors])

    tail = coherence[1:]
    mean_coh = float(tail.mean()) if tail.size else 1.0
    min_idx = int(np.argmin(coherence[1:]) + 1) if tail.size else 0
    min_coh = float(coherence[min_idx]) if tail.size else 1.0
    below = np.nonzero(coherence[1:] < float(break_threshold))[0]
    first_break = int(below[0] + 1) if below.size else -1

    if first_break < 0:
        verdict = "coherent"
        reason = (
            f"the masked object's colour+texture signature holds (min cosine {min_coh:.3g} >= "
            f"{break_threshold}) across all {t} frames"
        )
    elif min_coh < 0.5:
        verdict = "morph"
        reason = (
            f"the object's identity breaks at frame {first_break} (cosine drops to {min_coh:.3g} at "
            f"frame {min_idx}) — a colour/texture morph, not coherent motion"
        )
    else:
        verdict = "identity-drift"
        reason = (
            f"the object's signature drifts below {break_threshold} at frame {first_break} "
            f"(min cosine {min_coh:.3g} at frame {min_idx}) — partial identity drift"
        )
    return IdentityCoherenceResult(
        n_frames=t,
        mask_pixel_count=n_mask,
        break_threshold=float(break_threshold),
        coherence_by_frame=coherence,
        mean_coherence=mean_coh,
        min_coherence=min_coh,
        min_coherence_frame=min_idx,
        first_break_frame=first_break,
        verdict=verdict,
        verdict_reason=reason,
    )

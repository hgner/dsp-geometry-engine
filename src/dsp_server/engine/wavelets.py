"""2-D discrete wavelet multiresolution energy analysis (ELE490 lane).

PyWavelets ``wavedec2`` splits an image into an octave pyramid: one low-low
approximation plus, per level, three directional detail sub-bands (LH, HL, HH).
:func:`wavelet_energies` reduces that pyramid to per-scale detail ENERGY (the
squared-coefficient sum), reported FINEST-FIRST so index 0 is the highest spatial
frequency. :func:`compare_wavelets` turns a reference and a generated image into
per-scale energy PARITY (gen/ref), which tells the caller WHAT was lost — macro
geometry (coarse scales) vs micro texture (fine scales) — something a single
MSE/SSIM scalar cannot express.

Pure numpy + pywt, no file I/O (the imaging toolset owns image loading and the
JSON/plot envelope).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt

__all__ = [
    "WaveletComparison",
    "WaveletEnergies",
    "compare_wavelets",
    "wavelet_energies",
]

_TINY = 1e-30
_PARITY_TOL = 0.05  # |parity - 1| within this band counts as "at parity"
_MACRO_STRUCT_FLOOR = 0.8  # coarsest-scale parity below this reads as structural mismatch
_MICRO_SMOOTH_RATIO = 0.7  # micro < ratio * macro reads as micro-texture smoothing


@dataclass
class WaveletEnergies:
    """Per-scale wavelet energies of one image.

    ``detail_energy_by_scale`` is FINEST-FIRST: index 0 is level 1 (the finest
    detail band, highest spatial frequency), index -1 is level ``n_levels`` (the
    coarsest). Each entry sums the squared coefficients over the three directional
    detail sub-bands (LH, HL, HH) at that level. ``approx_energy`` is the
    squared-sum of the final low-low approximation; ``total_energy`` = approx +
    every detail band (the normalizer).
    """

    wavelet: str
    n_levels: int
    approx_energy: float
    detail_energy_by_scale: list[float]  # finest-first
    total_energy: float


def _max_level(shape: tuple[int, ...], wavelet: str) -> int:
    return int(pywt.dwtn_max_level(shape, wavelet))


def wavelet_energies(img: np.ndarray, wavelet: str = "db2", levels: int | None = None) -> WaveletEnergies:
    """Decompose ``img`` with ``pywt.wavedec2`` and sum the per-scale detail energy.

    ``levels=None`` uses the image's maximum useful level
    (``pywt.dwtn_max_level``); an explicit ``levels`` is clamped down to that
    maximum so a caller can never request a decomposition the image cannot carry.
    """
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale array, got shape {arr.shape}")
    max_level = _max_level(arr.shape, wavelet)
    if max_level < 1:
        raise ValueError(f"image {arr.shape} is too small for a '{wavelet}' wavelet decomposition")
    n_levels = max_level if not levels else min(int(levels), max_level)

    # wavedec2 -> [cA_L, (LH_L, HL_L, HH_L), ..., (LH_1, HL_1, HH_1)]: the detail
    # tuples run COARSEST-first, so reverse them to report finest-first.
    coeffs = pywt.wavedec2(arr, wavelet, level=n_levels)
    approx_energy = float(np.sum(np.asarray(coeffs[0], dtype=np.float64) ** 2))
    detail_energy_by_scale = [
        float(sum(float(np.sum(np.asarray(band, dtype=np.float64) ** 2)) for band in triple))
        for triple in reversed(coeffs[1:])
    ]
    total_energy = approx_energy + float(sum(detail_energy_by_scale))
    return WaveletEnergies(
        wavelet=str(wavelet),
        n_levels=len(detail_energy_by_scale),
        approx_energy=approx_energy,
        detail_energy_by_scale=detail_energy_by_scale,
        total_energy=total_energy,
    )


@dataclass
class WaveletComparison:
    """Per-scale energy parity of a generated image against a reference.

    Every ``*_parity`` is ``gen_energy / ref_energy`` (1.0 = energy preserved,
    < 1.0 = energy lost at that scale). ``detail_parity_by_scale`` is finest-first;
    ``macro_parity`` is the coarsest level, ``micro_parity`` the finest.
    ``worst_scale`` is the 1-indexed level with the largest relative energy LOSS.
    ``ref`` / ``gen`` carry the underlying energies (for plotting).
    """

    wavelet: str
    n_levels: int
    approx_parity: float
    detail_parity_by_scale: list[float]  # finest-first
    macro_parity: float
    micro_parity: float
    worst_scale: int  # 1-indexed level
    interpretation: str
    ref: WaveletEnergies
    gen: WaveletEnergies


def _safe_parity(gen_energy: float, ref_energy: float) -> float:
    """gen/ref with a div0 guard: both-empty bands report perfect parity."""
    if ref_energy <= _TINY and gen_energy <= _TINY:
        return 1.0
    return float(gen_energy / max(ref_energy, _TINY))


def _interpret(approx_parity: float, macro_parity: float, micro_parity: float) -> str:
    """Name the failure mode from the coarse-vs-fine parity split."""

    def near_one(value: float) -> bool:
        return abs(value - 1.0) <= _PARITY_TOL

    if near_one(approx_parity) and near_one(macro_parity) and near_one(micro_parity):
        return "parity"
    if macro_parity < _MACRO_STRUCT_FLOOR or approx_parity < _MACRO_STRUCT_FLOOR:
        return "structural mismatch"
    if micro_parity < _MICRO_SMOOTH_RATIO * macro_parity:
        return "micro-texture loss / smoothing"
    return "parity"


def compare_wavelets(
    img_ref: np.ndarray,
    img_gen: np.ndarray,
    wavelet: str = "db2",
    levels: int | None = None,
) -> WaveletComparison:
    """Compare two images' per-scale wavelet detail energy (gen relative to ref).

    Both images are decomposed with the same wavelet/level budget; parity is the
    finest-first energy ratio band by band. Same-shape images yield matching level
    counts — the finest-first alignment keeps the finest scales paired even if the
    counts differ.
    """
    ref = wavelet_energies(img_ref, wavelet=wavelet, levels=levels)
    gen = wavelet_energies(img_gen, wavelet=wavelet, levels=levels)

    detail_parity = [
        _safe_parity(g, r)
        for g, r in zip(gen.detail_energy_by_scale, ref.detail_energy_by_scale, strict=False)
    ]
    if not detail_parity:
        raise ValueError("no detail scales to compare (degenerate decomposition)")
    approx_parity = _safe_parity(gen.approx_energy, ref.approx_energy)
    micro_parity = detail_parity[0]  # finest
    macro_parity = detail_parity[-1]  # coarsest
    losses = [1.0 - p for p in detail_parity]
    worst_scale = int(np.argmax(losses)) + 1  # 1-indexed level (index 0 == level 1)
    interpretation = _interpret(approx_parity, macro_parity, micro_parity)
    return WaveletComparison(
        wavelet=ref.wavelet,
        n_levels=len(detail_parity),
        approx_parity=approx_parity,
        detail_parity_by_scale=detail_parity,
        macro_parity=macro_parity,
        micro_parity=micro_parity,
        worst_scale=worst_scale,
        interpretation=interpretation,
        ref=ref,
        gen=gen,
    )

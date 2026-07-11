"""Spatio-temporal frequency analysis for the video temporal gate (pure numpy.fft).

Per-frame spatial metrics (the imaging lane) are blind to *temporal* structure: a
sequence can be flawless frame-by-frame yet flicker or "boil" across time. This
module treats a frame stack as an ``(T, H, W)`` volume and measures the temporal
frequency content that those metrics miss.

The core operation removes the per-pixel temporal MEAN (drops static content /
temporal DC), takes an FFT along the TIME axis per pixel, and averages power over
pixels to get a temporal power spectrum ``P_t(f_t)`` with ``f_t`` in cycles/frame
on ``[0, 0.5]``. From it:

* ``temporal_hf_energy_fraction`` — AC energy above the HF cutoff over total AC
  energy (flicker / rapid oscillation).
* ``dominant_temporal_freq_cpf`` — the argmax AC bin.
* ``boil_energy_fraction`` — energy in the high-spatial x high-temporal corner:
  each frame's temporal variation is spatially high-passed first, so this isolates
  rapid change of FINE detail ("boil"), which a spatially-uniform flicker is not.

Frame I/O is owned by :mod:`dsp_server.engine.video`; this module only consumes the
``(T, H, W)`` volume it produces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_TINY = 1e-30

# cycles/frame; temporal frequencies strictly above this count as "fast" (flicker).
DEFAULT_HF_CUTOFF_CPF = 0.25
# cycles/pixel; spatial frequencies at or above this count as "fine" detail (boil).
DEFAULT_SPATIAL_HP_CUTOFF_CPP = 0.25


@dataclass
class SpatioTemporalSpectrum:
    """Temporal power spectrum of one frame stack plus its scalar summaries.

    ``freqs_cpf``/``power`` are the (K,) one-sided periodogram arrays (they live on
    disk in the tool's npz, never inline); everything else is a scalar the tool
    reports directly.
    """

    freqs_cpf: np.ndarray  # (K,) rfft temporal frequency axis, cycles/frame in [0, 0.5]
    power: np.ndarray  # (K,) mean-over-pixel temporal power P_t(f_t)
    n_frames: int
    height: int
    width: int
    hf_cutoff_cpf: float
    temporal_hf_energy_fraction: float
    dominant_temporal_freq_cpf: float
    boil_energy_fraction: float


def _temporal_power(vol: np.ndarray) -> np.ndarray:
    """Mean over pixels of ``|rfft over time|**2`` (per-pixel mean removed upstream)."""
    spec = np.fft.rfft(vol, axis=0)
    return (np.abs(spec) ** 2).mean(axis=(1, 2))


def _spatial_highpass(vol: np.ndarray, cutoff_cpp: float) -> np.ndarray:
    """Keep spatial frequencies at/above ``cutoff_cpp`` (cycles/pixel), per frame.

    A radial 2-D FFT mask over each frame — pure ``numpy.fft``, no scipy — so the
    boil metric measures rapid change of genuinely fine spatial detail.
    """
    _, h, w = vol.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    keep = np.hypot(fy, fx) >= float(cutoff_cpp)
    spec = np.fft.fft2(vol, axes=(1, 2))
    return np.fft.ifft2(spec * keep, axes=(1, 2)).real


def spatiotemporal_spectrum(
    frames: np.ndarray,
    hf_cutoff_cpf: float = DEFAULT_HF_CUTOFF_CPF,
    spatial_hp_cutoff_cpp: float = DEFAULT_SPATIAL_HP_CUTOFF_CPP,
) -> SpatioTemporalSpectrum:
    """Temporal power spectrum + flicker/boil metrics of an ``(T, H, W)`` stack.

    Subtracts the per-pixel temporal mean (removes static content), FFTs along time
    per pixel, and averages power over pixels. HF/AC fractions are taken over the AC
    band (temporal DC excluded); ``boil`` reuses the same temporal cutoff on the
    spatially high-passed volume, normalized by the total AC energy so it stays in
    ``[0, 1]``.
    """
    vol = np.asarray(frames, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"expected a (T, H, W) volume, got shape {vol.shape}")
    t, h, w = vol.shape
    if t < 2:
        raise ValueError(f"need >= 2 frames for a temporal spectrum, got {t}")

    centered = vol - vol.mean(axis=0, keepdims=True)  # drop per-pixel static content (temporal DC)
    freqs = np.fft.rfftfreq(t)  # cycles/frame in [0, 0.5]
    power = _temporal_power(centered)

    ac = power.copy()
    ac[0] = 0.0  # any residual DC bin never counts toward AC energy
    total_ac = float(ac.sum())
    hf_mask = freqs > float(hf_cutoff_cpf)
    hf_energy = float(ac[hf_mask].sum())
    hf_fraction = hf_energy / max(total_ac, _TINY)

    dominant_idx = int(np.argmax(ac)) if total_ac > 0.0 else 0
    dominant_freq = float(freqs[dominant_idx])

    hp = _spatial_highpass(centered, spatial_hp_cutoff_cpp)
    hp_power = _temporal_power(hp)
    hp_power[0] = 0.0
    boil_energy = float(hp_power[hf_mask].sum())
    boil_fraction = boil_energy / max(total_ac, _TINY)

    return SpatioTemporalSpectrum(
        freqs_cpf=freqs,
        power=power,
        n_frames=t,
        height=h,
        width=w,
        hf_cutoff_cpf=float(hf_cutoff_cpf),
        temporal_hf_energy_fraction=hf_fraction,
        dominant_temporal_freq_cpf=dominant_freq,
        boil_energy_fraction=boil_fraction,
    )

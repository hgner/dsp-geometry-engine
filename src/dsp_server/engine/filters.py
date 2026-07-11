"""1-D filtering/spectral utilities for axial profiles (ELE407-grounded).

Everything operates on plain numpy arrays sampled at a uniform spacing ``dt``
(meters along the bone axis), so frequencies are in cycles/meter (cpm).
IIR designs are returned as SOS cascades (numerically robust) and applied with
``sosfiltfilt`` — zero phase, because ridge positions must not shift: we
localize defects by their axial coordinate ``t``.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

_TINY = 1e-30


def detrend_poly(y: np.ndarray, order: int = 3) -> np.ndarray:
    """Least-squares polynomial detrend (default cubic — removes the natural
    elbow->wrist taper of a forearm radius profile). Returns the residual."""
    y = np.asarray(y, dtype=np.float64)
    if y.size <= order + 1:
        return y - y.mean()
    x = np.linspace(-1.0, 1.0, y.size)
    coeffs = np.polynomial.polynomial.polyfit(x, y, order)
    return y - np.polynomial.polynomial.polyval(x, coeffs)


def _check_cutoff(cutoff_cpm: float, dt: float) -> None:
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if cutoff_cpm <= 0.0:
        raise ValueError(f"cutoff must be positive, got {cutoff_cpm} cpm")
    limit = 0.45 / dt
    if cutoff_cpm >= limit:
        raise ValueError(
            f"cutoff {cutoff_cpm:g} cpm is at/above 90% of Nyquist "
            f"({limit:g} cpm for dt={dt:g} m) — profile too coarse for this band"
        )


def design_highpass(
    cutoff_cpm: float,
    dt: float,
    kind: str = "butter",
    order: int = 4,
) -> np.ndarray:
    """High-pass design as an SOS cascade. ``kind='cheby2'`` (40 dB stopband)
    gives a steeper transition when the corrugation band sits close to anatomy."""
    _check_cutoff(cutoff_cpm, dt)
    fs = 1.0 / dt
    if kind == "butter":
        sos = sps.butter(order, cutoff_cpm, btype="highpass", fs=fs, output="sos")
    elif kind == "cheby2":
        sos = sps.cheby2(order, 40.0, cutoff_cpm, btype="highpass", fs=fs, output="sos")
    else:
        raise ValueError(f"unknown filter kind '{kind}' (expected butter|cheby2)")
    return np.asarray(sos, dtype=np.float64)


def apply_highpass(
    y: np.ndarray,
    dt: float,
    cutoff_cpm: float,
    kind: str = "butter",
    order: int = 4,
) -> np.ndarray:
    """Zero-phase high-pass via sosfiltfilt, with a padlen guard so short
    profiles (fewer samples than the default pad) still filter cleanly."""
    y = np.asarray(y, dtype=np.float64)
    if y.size < 4:
        return y - y.mean()
    sos = design_highpass(cutoff_cpm, dt, kind=kind, order=order)
    padlen = int(min(3 * (2 * sos.shape[0] + 1), y.size - 1))
    return sps.sosfiltfilt(sos, y, padlen=padlen)


def fir_highpass(
    y: np.ndarray,
    dt: float,
    cutoff_cpm: float,
    numtaps: int | None = None,
) -> np.ndarray:
    """Windowed-FIR high-pass (linear phase by construction) applied forward and
    backward (filtfilt with a=1) so the net result is zero phase like the IIR path."""
    y = np.asarray(y, dtype=np.float64)
    _check_cutoff(cutoff_cpm, dt)
    if y.size < 8:
        return y - y.mean()
    fs = 1.0 / dt
    if numtaps is None:
        numtaps = int(round(3.5 * fs / cutoff_cpm))
        numtaps = max(9, min(numtaps, y.size - 2))
    if numtaps % 2 == 0:
        numtaps += 1  # odd taps -> type-I FIR, valid for a high-pass
    b = sps.firwin(numtaps, cutoff_cpm, pass_zero=False, fs=fs)
    padlen = int(min(3 * numtaps, y.size - 1))
    return sps.filtfilt(b, [1.0], y, padlen=padlen)


def resample_profile(
    y: np.ndarray,
    dt: float,
    factor_up: int,
    factor_down: int,
) -> tuple[np.ndarray, float]:
    """Anti-aliased rational resampling (scipy.signal.resample_poly).
    Returns (y_resampled, dt_new) — used to compare profiles with different
    sample counts on a common grid."""
    if factor_up < 1 or factor_down < 1:
        raise ValueError(f"factors must be >= 1, got up={factor_up} down={factor_down}")
    y = np.asarray(y, dtype=np.float64)
    out = sps.resample_poly(y, factor_up, factor_down)
    return out, dt * factor_down / factor_up


def band_energy(
    freqs: np.ndarray,
    psd: np.ndarray,
    lo_cpm: float,
    hi_cpm: float,
) -> float:
    """Integrated PSD over [lo_cpm, hi_cpm] (rectangle rule on the uniform
    frequency grid). Units: (signal unit)^2."""
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    if freqs.size < 2:
        return 0.0
    df = float(freqs[1] - freqs[0])
    mask = (freqs >= lo_cpm) & (freqs <= hi_cpm)
    return float(psd[mask].sum() * df)


def spectral_peaks(
    freqs: np.ndarray,
    psd: np.ndarray,
    f_min_cpm: float,
    max_peaks: int = 5,
) -> list[tuple[float, float]]:
    """Top spectral peaks above ``f_min_cpm``, ranked by find_peaks prominence.

    Returns [(freq_cpm, prominence_db), ...] where
    prominence_db = 10*log10(peak_psd / median_psd_above_fmin).
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    sel = freqs >= f_min_cpm
    f = freqs[sel]
    p = psd[sel]
    if f.size < 3:
        return []
    floor = max(float(np.median(p)), _TINY)
    # find_peaks cannot return array endpoints, so a dominant peak landing in the
    # first (or last) selected bin would be structurally invisible — pad with the
    # band minimum so band-edge maxima become interior peaks (finite prominence,
    # no ranking hijack), then shift indices back.
    pad = float(np.min(p))
    padded = np.concatenate(([pad], p, [pad]))
    idx, props = sps.find_peaks(padded, prominence=0.0)
    if idx.size == 0:
        return []
    idx = idx - 1  # undo the sentinel offset
    order = np.argsort(props["prominences"])[::-1][:max_peaks]
    out: list[tuple[float, float]] = []
    for k in order:
        i = int(idx[k])
        prom_db = 10.0 * np.log10(max(float(p[i]), _TINY) / floor)
        out.append((float(f[i]), prom_db))
    return out

"""3D vertex clouds -> 1D axial signals -> roughness/corrugation verdicts.

Corrugation = periodic radial ripple along the bone axis. Pipeline:
select_segment (dominant-joint mask) -> fit_axis (SVD PCA, per channel) ->
cylindrical_coords (t axial [m], r radial [m], theta) -> axial_profile
(uniform-bin median -> Signal1D) -> spectra in cycles/meter (cpm).

All functions are pure (numpy in / numpy out, no I/O) except
``resolve_joint_ids``, which may read a bone-map JSON sidecar.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal as sps

from dsp_server.engine.filters import apply_highpass, band_energy, detrend_poly, spectral_peaks
from dsp_server.engine.ply import EngineDump, bone_index_for

_TINY = 1e-30


@dataclass(frozen=True)
class Signal1D:
    """Uniformly sampled 1-D profile: y[i] lives at t = t0 + i*dt (meters)."""

    y: np.ndarray
    dt: float
    t0: float

    @property
    def span_m(self) -> float:
        """Axial distance from the first to the last sample."""
        return float(self.dt * (self.y.size - 1))


@dataclass
class AxisFrame:
    """Orthonormal cylinder frame: origin + axis (t direction) + u/v (radial plane)."""

    origin: np.ndarray  # (3,)
    axis: np.ndarray  # (3,) unit
    u: np.ndarray  # (3,) unit
    v: np.ndarray  # (3,) unit


# --------------------------------------------------------------------------- #
# Joint resolution + segment selection


def resolve_joint_ids(
    joint: str | int,
    bone_map: dict[int, str] | None = None,
    bone_map_path: str | None = None,
) -> list[int]:
    """Accepts a bone name (resolved via the bone map), a raw int id, or a
    numeric string. ``bone_map_path`` points at a JSON object {index: name}
    (the persisted stderr bone table) and supplements ``bone_map``."""
    if isinstance(joint, int):
        return [joint]
    name = joint.strip()
    combined: dict[int, str] = {}
    if bone_map_path is not None:
        raw = json.loads(Path(bone_map_path).read_text(encoding="utf-8"))
        combined.update({int(k): str(v) for k, v in raw.items()})
    if bone_map:
        combined.update(bone_map)
    if combined:
        try:
            return [bone_index_for(combined, name)]
        except KeyError:
            if name.lstrip("+-").isdigit():
                return [int(name)]
            raise
    if name.lstrip("+-").isdigit():
        return [int(name)]
    raise ValueError(
        f"joint '{name}' is a name but no bone map is available — pass bone_map, "
        "bone_map_path, or an integer joint id"
    )


def select_segment(
    dump: EngineDump,
    joint_ids: list[int],
    channel: str = "posed",
    min_verts: int = 50,
) -> np.ndarray:
    """Boolean vertex mask for the requested dominant joints. Raises with the
    full per-joint vertex histogram when too few vertices match."""
    dump.channel(channel)  # validate the channel name early
    mask = np.isin(dump.dominant_joint, np.asarray(joint_ids, dtype=np.int64))
    n_hit = int(mask.sum())
    if n_hit < min_verts:
        uniq, counts = np.unique(dump.dominant_joint, return_counts=True)
        hist = ", ".join(f"{int(j)}: {int(c)}" for j, c in zip(uniq, counts, strict=True))
        raise ValueError(
            f"only {n_hit} vertices have dominant joint in {joint_ids} "
            f"(min_verts={min_verts}); per-joint vertex histogram: {{{hist}}}"
        )
    return mask


# --------------------------------------------------------------------------- #
# Cylinder frame + coordinates


def fit_axis(points: np.ndarray) -> AxisFrame:
    """PCA via SVD: axis = direction of largest spread. Sign is fixed
    deterministically (largest-magnitude component made positive)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        raise ValueError(f"need an (n>=3, 3) point array, got shape {pts.shape}")
    origin = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - origin, full_matrices=False)
    axis = vt[0]
    if axis[int(np.argmax(np.abs(axis)))] < 0.0:
        axis = -axis
    u = vt[1]
    v = np.cross(axis, u)
    v /= np.linalg.norm(v)
    return AxisFrame(origin=origin, axis=axis, u=u, v=v)


def cylindrical_coords(
    points: np.ndarray,
    frame: AxisFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, r, theta): axial coordinate, radial distance, angle in the u/v plane."""
    pts = np.asarray(points, dtype=np.float64)
    d = pts - frame.origin
    t = d @ frame.axis
    radial = d - np.outer(t, frame.axis)
    r = np.linalg.norm(radial, axis=1)
    theta = np.arctan2(radial @ frame.v, radial @ frame.u)
    return t, r, theta


# --------------------------------------------------------------------------- #
# Binned profiles


def _binned_profile(
    t: np.ndarray,
    values: np.ndarray,
    lo: float,
    hi: float,
    n_samples: int,
    reducer: str,
) -> Signal1D:
    if hi <= lo:
        raise ValueError(f"degenerate axial range [{lo}, {hi}]")
    if reducer not in ("median", "mean"):
        raise ValueError(f"unknown reducer '{reducer}' (expected median|mean)")
    dt = (hi - lo) / n_samples
    idx = np.clip(((t - lo) / dt).astype(np.int64), 0, n_samples - 1)
    inside = (t >= lo) & (t <= hi)
    idx = idx[inside]
    vals = np.asarray(values, dtype=np.float64)[inside]
    y = np.full(n_samples, np.nan)
    for b in np.unique(idx):
        sel = vals[idx == b]
        y[b] = float(np.median(sel)) if reducer == "median" else float(sel.mean())
    filled = ~np.isnan(y)
    if not filled.any():
        raise ValueError("no vertices fall inside the axial profile range")
    centers = np.arange(n_samples, dtype=np.float64)
    y = np.interp(centers, centers[filled], y[filled])
    return Signal1D(y=y, dt=dt, t0=lo + 0.5 * dt)


def axial_profile(
    t: np.ndarray,
    r: np.ndarray,
    n_samples: int = 256,
    t_lo_pct: float = 5.0,
    t_hi_pct: float = 95.0,
    reducer: str = "median",
) -> Signal1D:
    """Radius-vs-axial-position profile: uniform bins over [P5, P95] of t,
    per-bin median (outlier-robust), linear interpolation over empty bins."""
    t = np.asarray(t, dtype=np.float64)
    lo, hi = np.percentile(t, [t_lo_pct, t_hi_pct])
    return _binned_profile(t, r, float(lo), float(hi), n_samples, reducer)


def sector_profiles(
    t: np.ndarray,
    r: np.ndarray,
    theta: np.ndarray,
    n_sectors: int = 8,
    n_samples: int = 256,
) -> list[Signal1D]:
    """Per-angular-sector axial profiles sharing one global bin grid, so their
    FFT phases are directly comparable (in-phase = rings, drift = spiral)."""
    t = np.asarray(t, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    lo, hi = np.percentile(t, [5.0, 95.0])
    width = 2.0 * math.pi / n_sectors
    sector = np.clip(((theta + math.pi) / width).astype(np.int64), 0, n_sectors - 1)
    out: list[Signal1D] = []
    for s in range(n_sectors):
        sel = sector == s
        if not sel.any():
            continue
        out.append(_binned_profile(t[sel], np.asarray(r)[sel], float(lo), float(hi), n_samples, "median"))
    return out


def displacement_profile(
    dump: EngineDump,
    mask: np.ndarray,
    frame: AxisFrame,
    n_samples: int = 256,
) -> Signal1D:
    """||posed - rest|| binned along t: corrugation here but not in the rest
    radius implicates skinning/deformers directly."""
    disp = np.linalg.norm(dump.posed[mask].astype(np.float64) - dump.rest[mask].astype(np.float64), axis=1)
    t, _, _ = cylindrical_coords(dump.posed[mask], frame)
    lo, hi = np.percentile(t, [5.0, 95.0])
    return _binned_profile(t, disp, float(lo), float(hi), n_samples, "median")


def normal_angle_profile(
    dump: EngineDump,
    mask: np.ndarray,
    frame: AxisFrame,
    n_samples: int = 256,
) -> Signal1D:
    """Angle (radians) between each vertex normal and its radial direction,
    binned along t — the cheap fallback channel when position space is quiet."""
    pts = dump.posed[mask].astype(np.float64)
    nrm = dump.normals[mask].astype(np.float64)
    nrm_len = np.linalg.norm(nrm, axis=1)
    nrm = nrm / np.maximum(nrm_len, _TINY)[:, None]
    d = pts - frame.origin
    t = d @ frame.axis
    radial = d - np.outer(t, frame.axis)
    r = np.linalg.norm(radial, axis=1)
    radial_dir = radial / np.maximum(r, _TINY)[:, None]
    cosang = np.clip(np.sum(nrm * radial_dir, axis=1), -1.0, 1.0)
    angle = np.arccos(cosang)
    lo, hi = np.percentile(t, [5.0, 95.0])
    return _binned_profile(t, angle, float(lo), float(hi), n_samples, "median")


# --------------------------------------------------------------------------- #
# Spectra


def _hann_psd(y: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Hann-windowed periodogram; freqs in cycles/meter."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    w = np.hanning(n)
    spec = np.fft.rfft((y - y.mean()) * w)
    psd = (np.abs(spec) ** 2) * (2.0 * dt / max(float(np.sum(w**2)), _TINY))
    psd[0] *= 0.5
    if n % 2 == 0:
        psd[-1] *= 0.5
    freqs = np.fft.rfftfreq(n, d=dt)
    return freqs, psd


def spectrum(
    sig: Signal1D,
    detrend_order: int = 3,
    welch: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """(freqs_cpm, psd) of the detrended profile. ``welch=True`` trades
    frequency resolution for a variance-reduced estimate."""
    y = detrend_poly(sig.y, detrend_order)
    if welch:
        nperseg = int(min(128, y.size))
        freqs, psd = sps.welch(y, fs=1.0 / sig.dt, window="hann", nperseg=nperseg)
        return freqs, psd
    return _hann_psd(y, sig.dt)


# --------------------------------------------------------------------------- #
# Roughness verdict


@dataclass
class RoughnessReport:
    rms_ripple_m: float
    rel_ripple: float
    diff1_var: float
    diff2_var: float
    hf_energy_ratio: float
    dominant_freq_cpm: float
    dominant_wavelength_m: float
    peak_prominence_db: float
    ridge_count_est: int
    sector_phase_coherence: float
    corrugated: bool


def _phase_at(sig_y: np.ndarray, dt: float, freq_cpm: float, detrend_order: int) -> float:
    y = detrend_poly(sig_y, detrend_order)
    w = np.hanning(y.size)
    spec = np.fft.rfft((y - y.mean()) * w)
    freqs = np.fft.rfftfreq(y.size, d=dt)
    k = int(np.argmin(np.abs(freqs - freq_cpm)))
    return float(np.angle(spec[k]))


def roughness(
    sig: Signal1D,
    sectors: list[Signal1D] | None = None,
    f_cut_cpm: float = 12.0,
    detrend_order: int = 3,
    filter_kind: str = "butter",
    min_rel_ripple: float = 1e-3,
) -> RoughnessReport:
    """Detrend -> zero-phase high-pass -> Hann -> rfft PSD on the relative
    radius signal r/median(r) - 1 (scale- and pose-fair).

    Verdict: ``corrugated`` requires peak prominence > 6 dB AND hf energy ratio
    > 0.15 AND rel_ripple above a physical floor (``min_rel_ripple``, default
    0.1% of the radius — below that nothing is visible in renders; the floor
    keeps broadband vertex noise from tripping the prominence test).
    """
    y = np.asarray(sig.y, dtype=np.float64)
    med = float(np.median(y))
    scale = med if abs(med) > 1e-12 else 1.0
    rel = y / scale - 1.0
    rel_dt = detrend_poly(rel, detrend_order)
    nyq_guard = 0.45 / sig.dt
    if f_cut_cpm < nyq_guard:
        rel_hp = apply_highpass(rel_dt, sig.dt, f_cut_cpm, kind=filter_kind)
    else:  # profile too coarse to separate the band — analyze the detrended signal
        rel_hp = rel_dt
    rel_ripple = float(np.sqrt(np.mean(rel_hp**2)))
    rms_ripple_m = rel_ripple * abs(scale)
    diff1_var = float(np.var(np.diff(rel_hp))) if rel_hp.size > 1 else 0.0
    diff2_var = float(np.var(np.diff(rel_hp, n=2))) if rel_hp.size > 2 else 0.0

    freqs, psd_full = _hann_psd(rel_dt, sig.dt)
    _, psd_hp = _hann_psd(rel_hp, sig.dt)
    f_max = float(freqs[-1])
    total = band_energy(freqs, psd_full, float(freqs[1]) * 0.5 if freqs.size > 1 else 0.0, f_max)
    hf = band_energy(freqs, psd_full, f_cut_cpm, f_max)
    hf_energy_ratio = hf / total if total > 0.0 else 0.0

    peaks = spectral_peaks(freqs, psd_hp, f_min_cpm=f_cut_cpm, max_peaks=5)
    if peaks:
        dominant_freq_cpm, peak_prominence_db = peaks[0]
        dominant_wavelength_m = 1.0 / dominant_freq_cpm if dominant_freq_cpm > 0.0 else 0.0
        ridge_count_est = int(round(dominant_freq_cpm * sig.span_m))
    else:
        dominant_freq_cpm = dominant_wavelength_m = peak_prominence_db = 0.0
        ridge_count_est = 0

    if sectors and peaks:
        phasors = []
        for s in sectors:
            smed = float(np.median(s.y))
            sscale = smed if abs(smed) > 1e-12 else 1.0
            srel = np.asarray(s.y, dtype=np.float64) / sscale - 1.0
            phasors.append(np.exp(1j * _phase_at(srel, s.dt, dominant_freq_cpm, detrend_order)))
        sector_phase_coherence = float(np.abs(np.mean(phasors)))
    else:
        sector_phase_coherence = 1.0

    corrugated = bool(peak_prominence_db > 6.0 and hf_energy_ratio > 0.15 and rel_ripple > min_rel_ripple)
    return RoughnessReport(
        rms_ripple_m=rms_ripple_m,
        rel_ripple=rel_ripple,
        diff1_var=diff1_var,
        diff2_var=diff2_var,
        hf_energy_ratio=hf_energy_ratio,
        dominant_freq_cpm=dominant_freq_cpm,
        dominant_wavelength_m=dominant_wavelength_m,
        peak_prominence_db=peak_prominence_db,
        ridge_count_est=ridge_count_est,
        sector_phase_coherence=sector_phase_coherence,
        corrugated=corrugated,
    )


def corrugation_extent(
    sig: Signal1D,
    f_lo_cpm: float | None = None,
    f_hi_cpm: float | None = None,
) -> tuple[float, float, float]:
    """Short-window STFT localization: (t_start_m, t_end_m, band_energy_fraction)
    — the axial span where the dominant band carries energy, to line up against
    capsule endpoints from ``--list-sources``. Band defaults to +/-30% around
    the dominant spectral peak."""
    y = detrend_poly(sig.y, 3)
    n = y.size
    t_end_full = sig.t0 + sig.span_m
    if f_lo_cpm is None or f_hi_cpm is None:
        span = max(sig.span_m, _TINY)
        freqs, psd = _hann_psd(y, sig.dt)
        peaks = spectral_peaks(freqs, psd, f_min_cpm=3.0 / span, max_peaks=1)
        if not peaks:
            return (sig.t0, t_end_full, 0.0)
        dom = peaks[0][0]
        f_lo_cpm = 0.7 * dom if f_lo_cpm is None else f_lo_cpm
        f_hi_cpm = 1.3 * dom if f_hi_cpm is None else f_hi_cpm
    nperseg = int(min(128, max(16, n // 4)))
    hop = max(1, nperseg // 8)
    f, frame_t, s_xx = sps.spectrogram(
        y,
        fs=1.0 / sig.dt,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg - hop,
        detrend=False,
        mode="psd",
    )
    band = (f >= f_lo_cpm) & (f <= f_hi_cpm)
    band_power = s_xx[band].sum(axis=0)
    total_power = s_xx.sum(axis=0)
    total_sum = float(total_power.sum())
    fraction = float(band_power.sum()) / total_sum if total_sum > 0.0 else 0.0
    peak = float(band_power.max()) if band_power.size else 0.0
    if peak <= 0.0:
        return (sig.t0, t_end_full, fraction)
    active = np.flatnonzero(band_power >= 0.4 * peak)
    first, last = int(active[0]), int(active[-1])
    centers = sig.t0 + frame_t
    half_hop = 0.5 * hop * sig.dt
    t_start = sig.t0 if first == 0 else float(centers[first] - half_hop)
    t_end = t_end_full if last == band_power.size - 1 else float(centers[last] + half_hop)
    return (t_start, t_end, fraction)


# --------------------------------------------------------------------------- #
# Convenience bundle


@dataclass
class ForearmSignal:
    signal: Signal1D
    sectors: list[Signal1D]
    frame: AxisFrame
    mask_count: int
    t_span_m: float
    joint_ids: list[int] = field(default_factory=list)


def extract_forearm_signal(
    dump: EngineDump,
    joint: str | int = "armLowerL",
    bone_map: dict[int, str] | None = None,
    bone_map_path: str | None = None,
    side: str | None = None,
    channel: str = "posed",
    n_samples: int = 256,
) -> ForearmSignal:
    """dump -> joint mask -> fitted cylinder frame -> axial + sector profiles."""
    if side is not None and isinstance(joint, str) and not joint.endswith(("L", "R")):
        joint = joint + side
    joint_ids = resolve_joint_ids(joint, bone_map=bone_map, bone_map_path=bone_map_path)
    mask = select_segment(dump, joint_ids, channel=channel)
    points = dump.channel(channel)[mask]
    frame = fit_axis(points)
    t, r, theta = cylindrical_coords(points, frame)
    sig = axial_profile(t, r, n_samples=n_samples)
    sectors = sector_profiles(t, r, theta, n_samples=n_samples)
    return ForearmSignal(
        signal=sig,
        sectors=sectors,
        frame=frame,
        mask_count=int(mask.sum()),
        t_span_m=sig.span_m,
        joint_ids=joint_ids,
    )

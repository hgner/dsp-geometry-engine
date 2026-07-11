"""DSP-core unit tests: synthetic cylinders only — no engine, no subprocess.

The golden signal is ``make_cylinder`` (tests/synth.py): a tilted fake forearm
with a known sinusoidal radial corrugation, against which every FFT/roughness
threshold in transforms.py is calibrated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dsp_server.engine.filters import apply_highpass, band_energy, detrend_poly, fir_highpass
from dsp_server.engine.ply import EngineDump, read_engine_ply, write_engine_ply
from dsp_server.engine.transforms import (
    Signal1D,
    corrugation_extent,
    extract_forearm_signal,
    roughness,
)
from synth import STUB_BONE_MAP, make_cylinder

FIXTURES = Path(__file__).parent / "fixtures"

CORR_FREQ = 60.0  # cycles/meter injected into the golden cylinder
CORR_AMP = 0.002  # meters


def _corrugated_dump() -> EngineDump:
    return make_cylinder(corrugation_amp=CORR_AMP, corrugation_freq_cpm=CORR_FREQ)


def _analyze(dump: EngineDump):
    forearm = extract_forearm_signal(dump, bone_map=STUB_BONE_MAP)
    return forearm, roughness(forearm.signal, sectors=forearm.sectors)


def _periodogram(y: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Plain (unwindowed) periodogram — tones sit exactly on bins in the filter test."""
    spec = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(y.size, d=dt)
    return freqs, (np.abs(spec) ** 2) * (2.0 * dt / y.size)


# --------------------------------------------------------------------------- #
# (a) corrugated cylinder -> roughness finds the injected tone


def test_corrugated_cylinder_detected():
    forearm, rep = _analyze(_corrugated_dump())
    sig = forearm.signal
    bin_width = 1.0 / (sig.dt * sig.y.size)
    assert abs(rep.dominant_freq_cpm - CORR_FREQ) <= bin_width + 1e-9
    assert rep.corrugated is True
    assert rep.sector_phase_coherence > 0.9  # rings, not a spiral
    assert rep.dominant_wavelength_m == pytest.approx(1.0 / CORR_FREQ, rel=0.10)
    assert abs(rep.ridge_count_est - CORR_FREQ * sig.span_m) <= 2.0
    assert rep.peak_prominence_db > 6.0
    assert rep.hf_energy_ratio > 0.15


# --------------------------------------------------------------------------- #
# (b) flat noisy cylinder -> quiet


def test_flat_cylinder_quiet():
    _, rep_flat = _analyze(make_cylinder(corrugation_amp=0.0, noise=5e-5))
    _, rep_corr = _analyze(_corrugated_dump())
    assert rep_flat.corrugated is False
    assert rep_corr.rel_ripple > 10.0 * rep_flat.rel_ripple


# --------------------------------------------------------------------------- #
# (c) scale + rigid-rotation invariance of the relative metrics


def _rotation(x_angle: float, z_angle: float) -> np.ndarray:
    ca, sa = np.cos(x_angle), np.sin(x_angle)
    cz, sz = np.cos(z_angle), np.sin(z_angle)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz @ rx


def test_scale_and_rotation_invariance():
    dump = _corrugated_dump()
    _, rep = _analyze(dump)

    rot = _rotation(1.1789, 0.7)
    centroid = dump.posed.astype(np.float64).mean(axis=0)

    def transform(p: np.ndarray) -> np.ndarray:
        return ((1.5 * (p.astype(np.float64) - centroid)) @ rot.T + centroid).astype(np.float32)

    moved = EngineDump(
        posed=transform(dump.posed),
        normals=(dump.normals.astype(np.float64) @ rot.T).astype(np.float32),
        rest=transform(dump.rest),
        dominant_joint=dump.dominant_joint.copy(),
        faces=dump.faces.copy(),
        props=[],
    )
    _, rep2 = _analyze(moved)

    assert rep2.rel_ripple == pytest.approx(rep.rel_ripple, rel=0.05)
    assert rep2.hf_energy_ratio == pytest.approx(rep.hf_energy_ratio, rel=0.05)
    assert rep2.dominant_wavelength_m == pytest.approx(1.5 * rep.dominant_wavelength_m, rel=0.10)


# --------------------------------------------------------------------------- #
# (d) filters: keep the corrugation band, kill trend + anatomy band


def _filter_test_signal() -> tuple[np.ndarray, float]:
    dt = 0.001
    x = np.arange(2000) * dt  # tones at 4 and 80 cpm sit exactly on bins
    y = 0.05 * (x - 1.0) ** 3 + 1.0 * np.sin(2 * np.pi * 4.0 * x) + 0.5 * np.sin(2 * np.pi * 80.0 * x)
    return y, dt


@pytest.mark.parametrize("kind", ["butter", "cheby2"])
def test_highpass_keeps_tone_attenuates_low_band(kind: str):
    y, dt = _filter_test_signal()
    freqs, psd_before = _periodogram(y, dt)
    filtered = apply_highpass(y, dt, cutoff_cpm=20.0, kind=kind)
    _, psd_after = _periodogram(filtered, dt)

    tone_ratio = band_energy(freqs, psd_after, 79.0, 81.0) / band_energy(freqs, psd_before, 79.0, 81.0)
    assert np.sqrt(tone_ratio) == pytest.approx(1.0, abs=0.10)  # amplitude kept within 10%

    lf_before = band_energy(freqs, psd_before, 3.0, 5.0)
    lf_after = band_energy(freqs, psd_after, 3.0, 5.0)
    assert 10.0 * np.log10(lf_before / max(lf_after, 1e-300)) > 20.0


def test_fir_highpass_matches_iir_behavior():
    y, dt = _filter_test_signal()
    freqs, psd_before = _periodogram(y, dt)
    filtered = fir_highpass(y, dt, cutoff_cpm=20.0)
    _, psd_after = _periodogram(filtered, dt)

    tone_ratio = band_energy(freqs, psd_after, 79.0, 81.0) / band_energy(freqs, psd_before, 79.0, 81.0)
    assert np.sqrt(tone_ratio) == pytest.approx(1.0, abs=0.10)
    lf_before = band_energy(freqs, psd_before, 3.0, 5.0)
    lf_after = band_energy(freqs, psd_after, 3.0, 5.0)
    assert 10.0 * np.log10(lf_before / max(lf_after, 1e-300)) > 20.0


def test_detrend_removes_pure_cubic_exactly():
    x = np.linspace(0.0, 2.0, 500)
    cubic = 3.0 - 2.0 * x + 0.7 * x**2 - 0.3 * x**3
    residual = detrend_poly(cubic, order=3)
    assert np.sqrt(np.mean(residual**2)) < 1e-9 * np.sqrt(np.mean(cubic**2))


# --------------------------------------------------------------------------- #
# (e) PLY round-trip + hand-written fixture regression


def test_ply_roundtrip(tmp_path):
    dump = make_cylinder(corrugation_amp=CORR_AMP, corrugation_freq_cpm=CORR_FREQ, with_weights=True)
    path = tmp_path / "roundtrip.ply"
    write_engine_ply(path, dump)
    back = read_engine_ply(path)

    assert back.vertex_count == dump.vertex_count
    assert back.face_count == dump.face_count
    assert np.allclose(back.posed, dump.posed, rtol=0.0, atol=1e-5)  # %.6g writer precision
    assert np.allclose(back.rest, dump.rest, rtol=0.0, atol=1e-5)
    assert np.allclose(back.normals, dump.normals, rtol=0.0, atol=1e-5)
    assert np.array_equal(back.dominant_joint, dump.dominant_joint)
    assert np.array_equal(back.faces, dump.faces)
    assert back.joint_indices is not None and back.joint_weights is not None
    assert np.array_equal(back.joint_indices, dump.joint_indices)
    assert np.allclose(back.joint_weights, dump.joint_weights, rtol=0.0, atol=1e-5)


def test_engine_8verts_fixture():
    dump = read_engine_ply(FIXTURES / "engine_8verts.ply")
    assert dump.vertex_count == 8
    assert dump.face_count == 2
    assert int(dump.dominant_joint[0]) == 5
    assert dump.props[:4] == ["x", "y", "z", "nx"]


# --------------------------------------------------------------------------- #
# (f) corrugation_extent localizes a half-corrugated profile


def test_corrugation_extent_localizes_ripple_window():
    n = 512
    dt = 0.25 / n
    t = np.arange(n) * dt
    ripple_start, ripple_end = 0.125, 0.25
    y = np.where(t >= ripple_start, 0.001 * np.sin(2 * np.pi * CORR_FREQ * t), 0.0)
    sig = Signal1D(y=y, dt=dt, t0=0.0)

    t_start, t_end, fraction = corrugation_extent(sig)

    window = ripple_end - ripple_start
    assert abs(t_start - ripple_start) <= 0.20 * window
    assert abs(t_end - ripple_end) <= 0.20 * window
    assert 0.3 < fraction <= 1.0


def test_band_edge_peak_is_visible():
    """Regression (review finding): a dominant peak landing in the FIRST bin at or
    above f_min must not be structurally invisible to find_peaks (endpoints)."""
    from dsp_server.engine.filters import spectral_peaks

    span = 0.25
    n = 256
    dt = span / n
    t = np.arange(n) * dt
    # 12 cy/m tone == roughness() f_cut default == first selected bin (df = 4 cy/m)
    y = 0.02 * np.sin(2 * np.pi * 12.0 * t)
    win = np.hanning(n)
    psd = np.abs(np.fft.rfft(y * win)) ** 2
    freqs = np.fft.rfftfreq(n, d=dt)
    peaks = spectral_peaks(freqs, psd, f_min_cpm=12.0)
    assert peaks, "band-edge dominant peak must be detected"
    assert abs(peaks[0][0] - 12.0) <= freqs[1] - freqs[0]
    assert peaks[0][1] > 6.0

    # Full-pipeline version: 12 cy/m corrugation on the mesh -> corrugated verdict
    dump = make_cylinder(corrugation_amp=CORR_AMP, corrugation_freq_cpm=12.0)
    _, report = _analyze(dump)
    assert report.corrugated is True
    assert abs(report.dominant_freq_cpm - 12.0) <= 2.0 * (1.0 / report.dominant_wavelength_m / 12.0 + 5.0)
    assert abs(report.dominant_freq_cpm - 12.0) < 10.0

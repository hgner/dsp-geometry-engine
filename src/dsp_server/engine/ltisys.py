"""LTI-systems math for the ELE301 systems pack (pure numpy/scipy.signal, no I/O).

Covers: system construction from num/den or zeros/poles/gain in the CT ('s') or
DT ('z') domain, time responses with step metrics, pole dynamics (zeta, wn, time
constants, DT magnitudes), Bode data with interpolated stability margins (scipy
has no margin() helper on response arrays — crossings are linearly interpolated
in log-frequency), sampling metrics for the aliasing verdict, and uniform-grid
convolution/correlation.

Conventions (mirrored by the toolset docstrings):
- DT transfer functions are polynomials in z^-1: ``den=[1, -0.5]`` means
  1 - 0.5*z^-1 (num is right-padded with zeros to den's length, so ``num=[1],
  den=[1, -0.5]`` has impulse response 0.5**n starting at n=0).
- Complex values cross the JSON boundary as [re, im] pairs.
- Stability margins treat the given H as the OPEN-loop transfer function.
- Energy integration reuses :func:`dsp_server.engine.filters.band_energy` — this
  module adds no FFT/PSD code of its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from dsp_server.engine.filters import band_energy

_TINY = 1e-30
_TOL = 1e-9

SYSTEM_FORM_HINT = (
    "pass exactly one system form: (a) num= + den= coefficient lists (descending powers of s for "
    "domain='s'; polynomials in z^-1 for domain='z', e.g. den=[1,-0.5] means 1-0.5z^-1) or "
    "(b) zeros=/poles= as [re, im] pairs plus gain=. domain='z' additionally requires dt= "
    "(the sample period)"
)


# --------------------------------------------------------------------------- #
# Construction


def pairs_to_complex(pairs: Sequence[Sequence[float]] | None) -> np.ndarray:
    """[[re, im], ...] -> complex128 array (JSON has no complex type)."""
    if not pairs:
        return np.zeros(0, dtype=np.complex128)
    out = np.empty(len(pairs), dtype=np.complex128)
    for i, pair in enumerate(pairs):
        seq = list(pair)
        if len(seq) != 2:
            raise ValueError(f"complex values are [re, im] pairs — got {seq!r}")
        out[i] = complex(float(seq[0]), float(seq[1]))
    return out


def complex_to_pairs(values: np.ndarray) -> list[list[float]]:
    vals = np.atleast_1d(np.asarray(values, dtype=np.complex128))
    return [[float(v.real), float(v.imag)] for v in vals]


def build_system(
    num: Sequence[float] | None = None,
    den: Sequence[float] | None = None,
    zeros: Sequence[Sequence[float]] | None = None,
    poles: Sequence[Sequence[float]] | None = None,
    gain: float | None = None,
    domain: str = "s",
    dt: float | None = None,
) -> sps.lti | sps.dlti:
    """Validated CT/DT system from exactly one input form (see SYSTEM_FORM_HINT)."""
    if domain not in ("s", "z"):
        raise ValueError(f"domain must be 's' or 'z', got '{domain}' — {SYSTEM_FORM_HINT}")
    if domain == "z":
        if dt is None or float(dt) <= 0.0:
            raise ValueError(f"domain='z' requires dt > 0 (the sample period) — {SYSTEM_FORM_HINT}")
        dt = float(dt)
    tf_form = num is not None or den is not None
    zpk_form = zeros is not None or poles is not None or gain is not None
    if tf_form == zpk_form:
        raise ValueError(SYSTEM_FORM_HINT)
    if tf_form:
        if num is None or den is None:
            raise ValueError(f"the num/den form needs BOTH lists — {SYSTEM_FORM_HINT}")
        num_c = [float(v) for v in num]
        den_c = [float(v) for v in den]
        if not num_c or not den_c or not any(den_c):
            raise ValueError(f"den must contain at least one nonzero coefficient — {SYSTEM_FORM_HINT}")
        if domain == "z":
            # z^-1 convention: right-pad the shorter polynomial so den=[1,-0.5]
            # reads as 1 - 0.5*z^-1 (num=[1] -> z/(z-0.5), impulse 0.5**n at n=0).
            width = max(len(num_c), len(den_c))
            num_c = num_c + [0.0] * (width - len(num_c))
            den_c = den_c + [0.0] * (width - len(den_c))
            return sps.TransferFunction(num_c, den_c, dt=dt)
        return sps.TransferFunction(num_c, den_c)
    z = pairs_to_complex(zeros)
    p = pairs_to_complex(poles)
    k = 1.0 if gain is None else float(gain)
    if domain == "z":
        return sps.ZerosPolesGain(z, p, k, dt=dt)
    return sps.ZerosPolesGain(z, p, k)


def zpk_of(sys: sps.lti | sps.dlti) -> tuple[np.ndarray, np.ndarray, float]:
    """(zeros, poles, gain) of any lti/dlti representation."""
    zpk = sys.to_zpk()
    zeros = np.atleast_1d(np.asarray(zpk.zeros, dtype=np.complex128))
    poles = np.atleast_1d(np.asarray(zpk.poles, dtype=np.complex128))
    return zeros, poles, float(np.real(zpk.gain))


# --------------------------------------------------------------------------- #
# Stability / dynamics


def stability_of(poles: np.ndarray, domain: str) -> tuple[bool, bool]:
    """(stable, marginally_stable). Marginal = no pole strictly outside and every
    boundary pole (imag axis / unit circle) simple."""
    p = np.atleast_1d(np.asarray(poles, dtype=np.complex128))
    if p.size == 0:
        return True, False
    margin = p.real if domain == "s" else np.abs(p) - 1.0
    outside = margin > _TOL
    on_boundary = np.abs(margin) <= _TOL
    if not outside.any() and not on_boundary.any():
        return True, False
    if outside.any():
        return False, False
    boundary = p[on_boundary]
    simple = True
    for i in range(boundary.size):
        for j in range(i + 1, boundary.size):
            if abs(boundary[i] - boundary[j]) < 1e-8:
                simple = False
    return False, simple


def roc_text(poles: np.ndarray, domain: str) -> str:
    """Causal region of convergence as human-readable text."""
    p = np.atleast_1d(np.asarray(poles, dtype=np.complex128))
    if domain == "s":
        if p.size == 0:
            return "entire s-plane"
        return f"Re(s) > {float(np.max(p.real)):g}"
    if p.size == 0:
        return "entire z-plane"
    return f"|z| > {float(np.max(np.abs(p))):g}"


@dataclass
class PoleDyn:
    """Per-pole reading: location + damping/frequency/time-constant (None-able)."""

    re: float
    im: float
    magnitude: float | None  # DT only: |z|
    damping_ratio: float | None
    natural_freq_rad_s: float | None
    time_constant_s: float | None


def pole_dynamics(poles: np.ndarray, domain: str, dt: float | None = None) -> list[PoleDyn]:
    """CT poles read directly; DT poles are mapped via s = ln(z)/dt when dt is known."""
    out: list[PoleDyn] = []
    for z in np.atleast_1d(np.asarray(poles, dtype=np.complex128)):
        if domain == "z":
            mag = float(abs(z))
            if mag <= _TINY or dt is None or dt <= 0.0:
                out.append(PoleDyn(float(z.real), float(z.imag), mag, None, None, None))
                continue
            s = complex(np.log(z) / dt)
        else:
            mag = None
            s = complex(z)
        wn = abs(s)
        zeta = float(-s.real / wn) if wn > _TINY else None
        tc = float(-1.0 / s.real) if s.real < -_TINY else None
        out.append(PoleDyn(float(z.real), float(z.imag), mag, zeta, wn if wn > _TINY else None, tc))
    return out


def default_t_end(poles: np.ndarray, domain: str = "s", dt: float | None = None) -> float:
    """~7 slowest time constants; 20.0 when every pole sits on the boundary/origin
    (no nonzero decay rate to read a time constant from)."""
    p = np.atleast_1d(np.asarray(poles, dtype=np.complex128))
    if domain == "z" and dt is not None and dt > 0.0:
        rates = np.asarray([np.log(max(abs(z), _TINY)) / dt for z in p], dtype=np.float64)
    else:
        rates = p.real.astype(np.float64) if p.size else np.zeros(0)
    nonzero = np.abs(rates) > _TOL
    if not nonzero.any():
        return 20.0
    return float(7.0 / np.abs(rates[nonzero]).min())


def dc_gain(zeros: np.ndarray, poles: np.ndarray, gain: float, domain: str = "s") -> float | None:
    """H(0) (CT) / H(z=1) (DT); None when a pole sits at the evaluation point."""
    z0 = 0.0 + 0.0j if domain == "s" else 1.0 + 0.0j
    z_arr = np.atleast_1d(np.asarray(zeros, dtype=np.complex128))
    p_arr = np.atleast_1d(np.asarray(poles, dtype=np.complex128))
    if p_arr.size and bool(np.any(np.abs(p_arr - z0) <= _TOL)):
        return None
    val = complex(gain)
    for z in z_arr:
        val *= z0 - z
    for p in p_arr:
        val /= z0 - p
    return float(val.real)


# --------------------------------------------------------------------------- #
# Time responses


def time_response(
    sys: sps.lti | sps.dlti,
    kind: str,
    t: np.ndarray,
    u: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, u, y) for kind in step|impulse|ramp|custom. CT: step/impulse/lsim; DT:
    dstep/dimpulse/dlsim. The returned u is the input actually applied — for the
    CT impulse it is the discrete delta approximation 1/dt at t[0]."""
    t = np.asarray(t, dtype=np.float64)
    if sys.dt is None:
        if kind == "step":
            tout, y = sps.step(sys, T=t)
            uu = np.ones_like(np.asarray(tout, dtype=np.float64))
        elif kind == "impulse":
            tout, y = sps.impulse(sys, T=t)
            uu = np.zeros_like(np.asarray(tout, dtype=np.float64))
            if uu.size > 1:
                uu[0] = 1.0 / float(t[1] - t[0])
        elif kind == "ramp":
            tout, y, _x = sps.lsim(sys, U=t, T=t)
            uu = t.copy()
        elif kind == "custom":
            if u is None:
                raise ValueError("kind='custom' needs a u array")
            uu = np.asarray(u, dtype=np.float64)
            tout, y, _x = sps.lsim(sys, U=uu, T=t)
        else:
            raise ValueError(f"unknown input kind '{kind}' (expected step|impulse|ramp|custom)")
        return np.asarray(tout, dtype=np.float64), uu, np.asarray(y, dtype=np.float64)

    n = t.size
    if kind == "step":
        tout, yout = sps.dstep(sys, n=n)
        y = np.squeeze(np.asarray(yout[0], dtype=np.float64))
        uu = np.ones(n, dtype=np.float64)
    elif kind == "impulse":
        tout, yout = sps.dimpulse(sys, n=n)
        y = np.squeeze(np.asarray(yout[0], dtype=np.float64))
        uu = np.zeros(n, dtype=np.float64)
        uu[0] = 1.0
    elif kind in ("ramp", "custom"):
        if kind == "ramp":
            uu = t.copy()
        else:
            if u is None:
                raise ValueError("kind='custom' needs a u array")
            uu = np.asarray(u, dtype=np.float64)
        res = sps.dlsim(sys, uu, t=t)
        tout, y = res[0], np.squeeze(np.asarray(res[1], dtype=np.float64))
    else:
        raise ValueError(f"unknown input kind '{kind}' (expected step|impulse|ramp|custom)")
    return np.asarray(tout, dtype=np.float64), uu, np.atleast_1d(y)


@dataclass
class StepMetrics:
    """Classic step-response metrics; every field None when non-convergent."""

    final_value: float | None
    rise_time_10_90: float | None
    settling_time_2pct: float | None
    overshoot_pct: float | None
    peak_time: float | None


def _first_crossing(t: np.ndarray, yn: np.ndarray, level: float) -> float | None:
    idx = np.flatnonzero(yn >= level)
    if idx.size == 0:
        return None
    i = int(idx[0])
    if i == 0:
        return float(t[0])
    y0, y1 = float(yn[i - 1]), float(yn[i])
    frac = (level - y0) / max(y1 - y0, _TINY)
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


def step_metrics(t: np.ndarray, y: np.ndarray) -> StepMetrics:
    """final value (last sample), 10-90% rise, 2% settling, overshoot, peak time.
    All None when the tail has not converged (non-settling/unstable response)."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.size < 8:
        return StepMetrics(None, None, None, None, None)
    final = float(y[-1])
    tail = y[int(0.95 * y.size) :]
    y_range = max(float(np.ptp(y)), _TINY)
    if float(np.max(np.abs(tail - final))) > 0.02 * y_range:
        return StepMetrics(None, None, None, None, None)
    if abs(final) <= 1e-12:
        return StepMetrics(final, None, None, None, None)
    yn = y / final
    t10 = _first_crossing(t, yn, 0.1)
    t90 = _first_crossing(t, yn, 0.9)
    rise = float(t90 - t10) if (t10 is not None and t90 is not None and t90 >= t10) else None
    err = np.abs(yn - 1.0)
    beyond = np.flatnonzero(err > 0.02)
    if beyond.size == 0:
        settle: float | None = float(t[0])
    elif int(beyond[-1]) == y.size - 1:
        settle = None
    else:
        i = int(beyond[-1])
        e0, e1 = float(err[i]), float(err[i + 1])
        frac = (e0 - 0.02) / max(e0 - e1, _TINY)
        settle = float(t[i] + frac * (t[i + 1] - t[i]))
    overshoot = max((float(np.max(yn)) - 1.0) * 100.0, 0.0)
    # peak_time is only meaningful when the response actually overshoots; for a
    # monotonic (overdamped/first-order) rise argmax is just the window end, which
    # is window-dependent, not a system property — report None instead.
    peak_time = float(t[int(np.argmax(yn))]) if float(np.max(yn)) > 1.0 + 1e-6 else None
    return StepMetrics(final, rise, settle, overshoot, peak_time)


# --------------------------------------------------------------------------- #
# Frequency responses


def default_w_grid(
    zeros: np.ndarray,
    poles: np.ndarray,
    domain: str = "s",
    dt: float | None = None,
    n_points: int = 400,
    w_min: float | None = None,
    w_max: float | None = None,
) -> np.ndarray:
    """Log grid spanning 2 decades beyond the pole/zero spread (DT capped at pi/dt)."""
    vals = list(np.atleast_1d(np.asarray(zeros, dtype=np.complex128))) + list(
        np.atleast_1d(np.asarray(poles, dtype=np.complex128))
    )
    crit: list[float] = []
    for c in vals:
        if domain == "z":
            if abs(c) <= 1e-12 or dt is None or dt <= 0.0:
                continue
            w = float(abs(np.log(c) / dt))
        else:
            w = float(abs(c))
        if w > 1e-12:
            crit.append(w)
    if w_min is not None:
        lo = float(w_min)
    else:
        lo = 10.0 ** (np.floor(np.log10(min(crit))) - 2.0) if crit else 1e-2
    if w_max is not None:
        hi = float(w_max)
    else:
        hi = 10.0 ** (np.ceil(np.log10(max(crit))) + 2.0) if crit else 1e2
    if domain == "z" and dt is not None and dt > 0.0:
        hi = min(hi, float(np.pi / dt))
    if lo <= 0.0 or hi <= 0.0:
        raise ValueError(f"frequency bounds must be positive, got w_min={lo:g} w_max={hi:g}")
    if hi <= lo:
        lo = hi / 1e4
    return np.logspace(np.log10(lo), np.log10(hi), int(n_points))


def freq_response(sys: sps.lti | sps.dlti, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(w [rad/s], mag [dB], unwrapped phase [deg]) via sps.bode / sps.dbode.
    dbode wants normalized rad/sample in and hands back rad/s — fed w*dt."""
    w = np.asarray(w, dtype=np.float64)
    if sys.dt is None:
        wout, mag, phase = sps.bode(sys, w=w)
    else:
        wout, mag, phase = sps.dbode(sys, w=w * float(sys.dt))
    return (
        np.asarray(wout, dtype=np.float64),
        np.asarray(mag, dtype=np.float64),
        np.asarray(phase, dtype=np.float64),
    )


def _crossings(w: np.ndarray, curve: np.ndarray, target: float) -> list[float]:
    """Frequencies where curve crosses target, interpolated linearly in log10(w)."""
    out: list[float] = []
    d = np.asarray(curve, dtype=np.float64) - target
    for i in range(d.size - 1):
        if d[i] == 0.0:
            out.append(float(w[i]))
        elif d[i] * d[i + 1] < 0.0:
            frac = d[i] / (d[i] - d[i + 1])
            out.append(float(w[i] * (w[i + 1] / w[i]) ** frac))
    if d.size and d[-1] == 0.0:
        out.append(float(w[-1]))
    return out


def _interp_log(w: np.ndarray, curve: np.ndarray, w_q: float) -> float:
    return float(np.interp(np.log10(w_q), np.log10(np.asarray(w, dtype=np.float64)), curve))


@dataclass
class Margins:
    """Open-loop stability margins from response arrays (None when no crossing)."""

    gm_db: float | None
    w_gain_cross: float | None  # phase-crossover frequency (where GM is read)
    pm_deg: float | None
    w_phase_cross: float | None  # gain-crossover frequency (where PM is read)


def stability_margins(w: np.ndarray, mag_db: np.ndarray, phase_deg: np.ndarray) -> Margins:
    """GM at the -180 deg phase crossover, PM at the 0 dB gain crossover; when a
    curve crosses several times the worst (smallest) margin is reported."""
    gm_db = w_gm = pm = w_pm = None
    for wc in _crossings(w, phase_deg, -180.0):
        cand = -_interp_log(w, mag_db, wc)
        if gm_db is None or cand < gm_db:
            gm_db, w_gm = cand, wc
    for wc in _crossings(w, mag_db, 0.0):
        cand = 180.0 + _interp_log(w, phase_deg, wc)
        if pm is None or cand < pm:
            pm, w_pm = cand, wc
    return Margins(gm_db, w_gm, pm, w_pm)


def bandwidth_3db(w: np.ndarray, mag_db: np.ndarray, ref_db: float | None = None) -> float | None:
    """First frequency where the magnitude falls 3.01 dB below ref (default: the
    lowest-frequency sample). None when the response never drops that far."""
    ref = float(mag_db[0]) if ref_db is None else float(ref_db)
    cross = _crossings(w, mag_db, ref - 20.0 * np.log10(np.sqrt(2.0)))
    return cross[0] if cross else None


def hf_slope(w: np.ndarray, mag_db: np.ndarray) -> float:
    """LSQ slope in dB/decade over the top decade of the grid (~ -20*(#p - #z))."""
    w = np.asarray(w, dtype=np.float64)
    mag_db = np.asarray(mag_db, dtype=np.float64)
    sel = w >= w[-1] / 10.0
    if int(sel.sum()) < 2:
        sel = np.zeros(w.size, dtype=bool)
        sel[-2:] = True
    return float(np.polyfit(np.log10(w[sel]), mag_db[sel], 1)[0])


# --------------------------------------------------------------------------- #
# Sampling / aliasing


def alias_fold(f: np.ndarray, fs: float) -> np.ndarray:
    """Fold true frequencies into the base band [0, fs/2]."""
    f = np.asarray(f, dtype=np.float64)
    r = np.mod(f, fs)
    return np.where(r <= 0.5 * fs, r, fs - r)


@dataclass
class SamplingMetrics:
    """Aliasing bookkeeping for a proposed rate against a reference spectrum."""

    nyquist: float
    frac_above: float
    bw_99: float


def sampling_metrics(freqs: np.ndarray, psd: np.ndarray, fs: float) -> SamplingMetrics:
    """Fraction of PSD energy above the proposed Nyquist + the 99%-energy bandwidth.
    Takes a precomputed spectrum; integration via filters.band_energy."""
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    nyq = 0.5 * float(fs)
    f_top = float(freqs[-1]) if freqs.size else 0.0
    total = band_energy(freqs, psd, 0.0, f_top)
    above = band_energy(freqs, psd, nyq, f_top)
    frac = above / total if total > 0.0 else 0.0
    csum = np.cumsum(psd)
    if csum.size == 0 or csum[-1] <= 0.0:
        bw99 = 0.0
    else:
        k = min(int(np.searchsorted(csum, 0.99 * csum[-1])), freqs.size - 1)
        bw99 = float(freqs[k])
    return SamplingMetrics(nyq, float(frac), bw99)


# --------------------------------------------------------------------------- #
# Convolution / correlation


def convolve_uniform(
    a: np.ndarray,
    b: np.ndarray,
    dt: float,
    mode: str = "convolve",
    scale: str = "continuous",
) -> tuple[np.ndarray, np.ndarray]:
    """(t_or_lag, y): full convolution (t axis from 0) or full cross-correlation
    (lag axis via correlation_lags; c[k] = sum_n a[n]*b[n-k], so a delayed copy in
    b peaks at NEGATIVE lag). scale='continuous' multiplies the discrete sum by dt
    (CT integral approximation: rect*rect -> unit triangle); 'discrete' is raw.
    method='direct' keeps small integer-valued cases bit-exact."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if mode == "convolve":
        y = sps.convolve(a, b, mode="full", method="direct")
        axis = np.arange(y.size, dtype=np.float64) * dt
    elif mode == "correlate":
        y = sps.correlate(a, b, mode="full", method="direct")
        axis = sps.correlation_lags(a.size, b.size, mode="full").astype(np.float64) * dt
    else:
        raise ValueError(f"unknown mode '{mode}' (expected convolve|correlate)")
    if scale == "continuous":
        y = y * dt
    elif scale != "discrete":
        raise ValueError(f"unknown scale '{scale}' (expected continuous|discrete)")
    return axis, y


def ncc_at_lag(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """Normalized cross-correlation coefficient at one integer lag, using only the
    overlapping segments (Cauchy-Schwarz: 1.0 exactly at a perfect aligned match)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    lo = max(0, lag)
    hi = min(a.size - 1, b.size - 1 + lag)
    if hi < lo:
        return 0.0
    seg_a = a[lo : hi + 1]
    seg_b = b[lo - lag : hi - lag + 1]
    denom = float(np.linalg.norm(seg_a) * np.linalg.norm(seg_b))
    if denom <= _TINY:
        return 0.0
    return float(np.dot(seg_a, seg_b) / denom)


# --------------------------------------------------------------------------- #
# Group delay  (tau_g(w) = -d(phase)/dw)


def group_delay_lti(
    sys: sps.lti | sps.dlti,
    w: np.ndarray,
    domain: str,
    dt: float | None = None,
    ba: tuple[Sequence[float], Sequence[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Group delay tau_g(w) = -d(phase)/dw of an LTI system on the angular-frequency
    grid ``w`` (rad/s). Returns ``(w, tau, unit)``.

    Digital (domain='z'): scipy.signal.group_delay of the (b, a) coefficients, in
    SAMPLES. The rad/s grid is converted to Hz (``w / 2*pi``) before the call so that
    ``fs = 1/dt`` is dimensionally consistent (group_delay reads ``w`` in the units of
    ``fs``). Analog (domain='s'): H(jw) = polyval(b, jw)/polyval(a, jw), then
    tau = -gradient(unwrap(angle(H)), w) in SECONDS.

    ``ba`` supplies the raw (numerator, denominator) coefficients directly. Prefer it:
    ``sys.to_tf()`` runs scipy's ``normalize``, which trims a small leading numerator
    coefficient of a symmetric FIR and corrupts its group delay (10 -> 9 samples).
    """
    w = np.asarray(w, dtype=np.float64)
    if ba is not None:
        b = np.atleast_1d(np.asarray(ba[0], dtype=np.float64))
        a = np.atleast_1d(np.asarray(ba[1], dtype=np.float64))
    else:
        tf = sys.to_tf()
        b = np.atleast_1d(np.asarray(tf.num, dtype=np.float64))
        a = np.atleast_1d(np.asarray(tf.den, dtype=np.float64))
    if domain == "z":
        if dt is None or float(dt) <= 0.0:
            raise ValueError("digital group delay needs dt > 0 (the sample period)")
        dt = float(dt)
        _wout, gd = sps.group_delay((b, a), w=w / (2.0 * np.pi), fs=1.0 / dt)
        return w, np.asarray(gd, dtype=np.float64), "samples"
    s = 1j * w
    h = np.polyval(b, s) / np.polyval(a, s)
    phase = np.unwrap(np.angle(h))
    tau = -np.gradient(phase, w)
    return w, np.asarray(tau, dtype=np.float64), "seconds"


def group_delay_xspec(a: np.ndarray, b: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Cross-spectral group delay between two equal-length, uniformly-sampled series.

    Sab = rfft(a) * conj(rfft(b)); tau = -d(unwrap(angle(Sab)))/dw over the analysis
    grid w = 2*pi*rfftfreq(n, dt). Positive tau means series ``a`` LAGS series ``b`` by
    that many seconds at that frequency. Returns ``(w [rad/s], tau [s])``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size:
        raise ValueError(f"series must be equal length for a cross-spectrum, got {a.size} and {b.size}")
    if a.size < 4:
        raise ValueError(f"need >= 4 samples for a cross-spectral group delay, got {a.size}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    fa = np.fft.rfft(a)
    fb = np.fft.rfft(b)
    sab = fa * np.conj(fb)
    phase = np.unwrap(np.angle(sab))
    w = 2.0 * np.pi * np.fft.rfftfreq(a.size, d=dt)
    tau = -np.gradient(phase, w)
    return w, np.asarray(tau, dtype=np.float64)

"""Agg-only matplotlib helpers: plots are opt-in, saved under data/plots/, and the
saved :class:`~pathlib.Path` is returned so tools can put it in their JSON payload.

The backend is forced to Agg BEFORE pyplot is imported — stdout is MCP JSON-RPC and
no GUI event loop may ever start. Nothing in here calls ``plt.show()``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be locked before pyplot loads)

from dsp_server.engine.transforms import Signal1D  # noqa: E402

_TINY = 1e-30


def _t_axis(sig: Signal1D) -> np.ndarray:
    return sig.t0 + np.arange(sig.y.size, dtype=np.float64) * sig.dt


def _psd_db(psd: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.asarray(psd, dtype=np.float64) + _TINY)


def _prepare(path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_signal_plot(
    sig: Signal1D,
    spectrum_freqs: np.ndarray,
    spectrum_psd: np.ndarray,
    path: Path | str,
    title: str,
) -> Path:
    """Twin panel — top: the axial profile r(t); bottom: its PSD in dB vs cycles/meter.
    Returns the Path written."""
    path = _prepare(path)
    fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(9.0, 6.0), constrained_layout=True)
    ax_t.plot(_t_axis(sig), sig.y, lw=1.0, color="tab:blue")
    ax_t.set_xlabel("t along bone axis [m]")
    ax_t.set_ylabel("profile value [m]")
    ax_t.set_title("axial profile")
    ax_t.grid(True, alpha=0.3)
    freqs = np.asarray(spectrum_freqs, dtype=np.float64)
    db = _psd_db(spectrum_psd)
    if freqs.size > 1:
        ax_f.plot(freqs[1:], db[1:], lw=1.0, color="tab:red")
    else:
        ax_f.plot(freqs, db, lw=1.0, color="tab:red")
    ax_f.set_xlabel("frequency [cycles/m]")
    ax_f.set_ylabel("PSD [dB]")
    ax_f.set_title("spectrum")
    ax_f.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def save_comparison_plot(
    sig_a: Signal1D,
    label_a: str,
    freqs_a: np.ndarray,
    psd_a: np.ndarray,
    sig_b: Signal1D,
    label_b: str,
    freqs_b: np.ndarray,
    psd_b: np.ndarray,
    path: Path | str,
    title: str,
) -> Path:
    """Twin panel overlaying two profiles (top) and their PSDs in dB (bottom).
    Returns the Path written."""
    path = _prepare(path)
    fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(9.0, 6.0), constrained_layout=True)
    ax_t.plot(_t_axis(sig_a), sig_a.y, lw=1.0, color="tab:blue", label=label_a)
    ax_t.plot(_t_axis(sig_b), sig_b.y, lw=1.0, color="tab:orange", label=label_b)
    ax_t.set_xlabel("t along bone axis [m]")
    ax_t.set_ylabel("profile value [m]")
    ax_t.set_title("axial profiles")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(loc="best", fontsize=8)
    fa = np.asarray(freqs_a, dtype=np.float64)
    fb = np.asarray(freqs_b, dtype=np.float64)
    da = _psd_db(psd_a)
    db = _psd_db(psd_b)
    ax_f.plot(
        fa[1:] if fa.size > 1 else fa, da[1:] if fa.size > 1 else da, lw=1.0, color="tab:blue", label=label_a
    )
    ax_f.plot(
        fb[1:] if fb.size > 1 else fb,
        db[1:] if fb.size > 1 else db,
        lw=1.0,
        color="tab:orange",
        label=label_b,
    )
    ax_f.set_xlabel("frequency [cycles/m]")
    ax_f.set_ylabel("PSD [dB]")
    ax_f.set_title("spectra")
    ax_f.grid(True, alpha=0.3)
    ax_f.legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def save_spectrum2d_plot(
    psd_a: np.ndarray,
    path: Path | str,
    title: str,
    psd_b: np.ndarray | None = None,
) -> Path:
    """Centered 2-D log-power spectrum image(s) — one panel, or two side by side
    when ``psd_b`` is given. Returns the Path written."""
    path = _prepare(path)
    n_panels = 1 if psd_b is None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.0 * n_panels, 4.5), constrained_layout=True)
    axes_list = [axes] if n_panels == 1 else list(axes)
    for ax, psd, name in zip(axes_list, [psd_a, psd_b][:n_panels], ("A", "B"), strict=False):
        img = ax.imshow(_psd_db(np.asarray(psd)), origin="lower", cmap="magma", aspect="auto")
        ax.set_title(f"log PSD {name}" if n_panels == 2 else "log PSD")
        ax.set_xlabel("fx bin")
        ax.set_ylabel("fy bin")
        fig.colorbar(img, ax=ax, shrink=0.85, label="dB")
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path

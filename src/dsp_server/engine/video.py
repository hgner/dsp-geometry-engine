"""Frame-stack ingestion for the video comparison lane (pure numpy + Pillow).

A video is an ``(T, H, W)`` float volume (grayscale, values in [0, 1]). This module
owns loading a sequence — a directory of frames or an explicit path list — into that
volume with consistent ordering and shape, plus the size guards MCP tools need. The
spatio-temporal FFT and optical-flow tools build on :func:`load_frame_stack`; they
never re-implement frame I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dsp_server.engine.image2d import load_image_gray

# Cap the decoded volume so a huge sequence can't hang a synchronous MCP call.
MAX_FRAMES = 512
MAX_CELLS = 200_000_000  # T*H*W

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_NUM_RE = re.compile(r"(\d+)")


class VideoError(ValueError):
    """Hint-ready failure for the frame-stack loaders."""


@dataclass
class FrameStack:
    """One loaded sequence: ``frames`` is (T, H, W) float64 in [0, 1]."""

    frames: np.ndarray
    paths: list[str]
    n_dropped: int = 0  # frames beyond MAX_FRAMES that were not loaded

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def height(self) -> int:
        return int(self.frames.shape[1])

    @property
    def width(self) -> int:
        return int(self.frames.shape[2])


def _natural_key(path: Path) -> tuple:
    """Sort frame_2.png before frame_10.png (numeric-aware, not lexicographic)."""
    parts = _NUM_RE.split(path.name)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def resolve_frame_paths(source: str | Path | list[str]) -> list[Path]:
    """A directory (all images, natural-sorted), a glob, or an explicit path list."""
    if isinstance(source, list):
        paths = [Path(p) for p in source]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise VideoError(f"frame paths not found: {missing}")
        return paths
    src = Path(source)
    if src.is_dir():
        frames = [p for p in src.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES]
        if not frames:
            raise VideoError(f"no image frames in {src} — expected {', '.join(_IMAGE_SUFFIXES)} files")
        return sorted(frames, key=_natural_key)
    if src.is_file():
        return [src]
    # treat as a glob relative to cwd / parent
    matches = sorted(Path().glob(str(source)), key=_natural_key)
    if not matches:
        raise VideoError(f"no frames matched '{source}' (pass a directory, a glob, or a path list)")
    return matches


def load_frame_stack(source: str | Path | list[str], max_frames: int | None = None) -> FrameStack:
    """Load a frame sequence into an (T, H, W) float64 [0, 1] volume.

    All frames must share the first frame's shape (a size mismatch is a hard error —
    silently resizing would corrupt a spatio-temporal spectrum). Frames beyond
    ``max_frames`` (default :data:`MAX_FRAMES`) are dropped and counted.
    """
    paths = resolve_frame_paths(source)
    cap = MAX_FRAMES if max_frames is None else min(int(max_frames), MAX_FRAMES)
    n_dropped = max(0, len(paths) - cap)
    paths = paths[:cap]
    if len(paths) < 2:
        raise VideoError(f"need >= 2 frames for a video analysis, got {len(paths)} from {source!r}")

    first = load_image_gray(paths[0])
    h, w = first.shape
    if len(paths) * h * w > MAX_CELLS:
        raise VideoError(
            f"stack is {len(paths)}x{h}x{w} = {len(paths) * h * w:.2e} cells — limit "
            f"{MAX_CELLS:.0e}; pass max_frames= or downscale the frames"
        )
    frames = np.empty((len(paths), h, w), dtype=np.float64)
    frames[0] = first
    for i, p in enumerate(paths[1:], start=1):
        img = load_image_gray(p)
        if img.shape != (h, w):
            raise VideoError(
                f"frame {p.name} is {img.shape} but frame 0 is {(h, w)} — all frames must match "
                "(resize the sequence to a common resolution first)"
            )
        frames[i] = img
    return FrameStack(frames=frames, paths=[str(p) for p in paths], n_dropped=n_dropped)

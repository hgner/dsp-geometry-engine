"""Toolset registry — the extensibility contract.

A course pack = one module in this package exposing ``register(mcp, ctx)`` + one
line in :data:`TOOLSETS`. The server iterates the registry (filtered by the
``DSP_TOOLSETS`` env var) so a new course never touches server.py. Pack modules
are imported lazily inside their loader functions to keep import cycles (pack ->
AppContext -> package) impossible and package import cheap.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dsp_server import config

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


@dataclass(frozen=True)
class AppContext:
    """Everything a pack's tools need beyond their own math: where dumps, npz
    caches, plots, and logs live, and where the engine repo is expected."""

    data_dir: Path
    dumps_dir: Path
    cache_dir: Path
    plots_dir: Path
    logs_dir: Path
    engine_root: Path

    @classmethod
    def from_config(cls) -> AppContext:
        return cls(
            data_dir=config.data_dir(),
            dumps_dir=config.dumps_dir(),
            cache_dir=config.cache_dir(),
            plots_dir=config.plots_dir(),
            logs_dir=config.logs_dir(),
            engine_root=config.engine_root(),
        )


def _register_geometry(mcp: FastMCP, ctx: AppContext) -> None:
    from dsp_server.toolsets import geometry

    geometry.register(mcp, ctx)


def _register_imaging(mcp: FastMCP, ctx: AppContext) -> None:
    from dsp_server.toolsets import imaging

    imaging.register(mcp, ctx)


def _register_stats(mcp: FastMCP, ctx: AppContext) -> None:
    from dsp_server.toolsets import stats

    stats.register(mcp, ctx)


def _register_engmath(mcp: FastMCP, ctx: AppContext) -> None:
    from dsp_server.toolsets import engmath

    engmath.register(mcp, ctx)


def _register_systems(mcp: FastMCP, ctx: AppContext) -> None:
    from dsp_server.toolsets import systems

    systems.register(mcp, ctx)


def _register_ml(mcp: FastMCP, ctx: AppContext) -> None:
    from dsp_server.toolsets import ml

    ml.register(mcp, ctx)


def _register_netqueue(mcp: FastMCP, ctx: AppContext) -> None:
    from dsp_server.toolsets import netqueue

    netqueue.register(mcp, ctx)


def _register_os(mcp: FastMCP, ctx: AppContext) -> None:
    from dsp_server.toolsets import os_sim  # module named os_sim: never shadow stdlib os

    os_sim.register(mcp, ctx)


def _register_rendering(mcp: FastMCP, ctx: AppContext) -> None:
    from dsp_server.toolsets import rendering

    rendering.register(mcp, ctx)


def _register_video(mcp: FastMCP, ctx: AppContext) -> None:
    from dsp_server.toolsets import video

    video.register(mcp, ctx)


def _register_perceptual(mcp: FastMCP, ctx: AppContext) -> None:
    from dsp_server.toolsets import perceptual

    perceptual.register(mcp, ctx)


# Course packs in curriculum order (llms.txt rules 12-18 mirror this order);
# 'rendering' is the PBR/ray-tracer energy lane (rule 19), 'video' the comparison
# gate (rule 20), 'perceptual' the classical FR-VQA layer (rule 21).
TOOLSETS: dict[str, Callable[[FastMCP, AppContext], None]] = {
    "geometry": _register_geometry,
    "imaging": _register_imaging,
    "stats": _register_stats,
    "engmath": _register_engmath,
    "systems": _register_systems,
    "ml": _register_ml,
    "netqueue": _register_netqueue,
    "os": _register_os,
    "rendering": _register_rendering,
    "video": _register_video,
    "perceptual": _register_perceptual,
}

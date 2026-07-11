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


TOOLSETS: dict[str, Callable[[FastMCP, AppContext], None]] = {
    "geometry": _register_geometry,
    "imaging": _register_imaging,
}

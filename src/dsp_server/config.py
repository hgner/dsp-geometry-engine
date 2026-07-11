"""Central configuration: every knob is env-overridable so the same package runs
as a local stdio server (Claude Code/Desktop), in CI against the stub engine, and
in a cloud container (streamable HTTP, no engine binary present).
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ENGINE_ROOT = Path("C:/Users/hgner/hakantest/proje7-engine")

# Preferred exe first: static-md-release has no DXR/GPU dependency.
ENGINE_CLI_CANDIDATES = (
    Path("build/windows-msvc-static-md-release/layered_field_dump_cli.exe"),
    Path("build/windows-msvc-rtx/layered_field_dump_cli.exe"),
)

RENDER_CLI_CANDIDATES = (
    Path("build/windows-msvc-rtx-release/i2v_beauty_dxr_cli.exe"),
    Path("build/windows-msvc-rtx/i2v_beauty_dxr_cli.exe"),
)


def engine_root() -> Path:
    return Path(os.environ.get("DSP_ENGINE_ROOT", str(DEFAULT_ENGINE_ROOT)))


def engine_cli_override() -> str | None:
    """DSP_ENGINE_CLI is a single path (never shell-split — Windows paths contain
    backslashes). A ``.py`` suffix means "run via the current interpreter"."""
    return os.environ.get("DSP_ENGINE_CLI") or None


def data_dir() -> Path:
    return Path(os.environ.get("DSP_DATA_DIR", str(Path.cwd() / "data")))


def dumps_dir() -> Path:
    return data_dir() / "dumps"


def cache_dir() -> Path:
    return data_dir() / "cache"


def plots_dir() -> Path:
    return data_dir() / "plots"


def logs_dir() -> Path:
    return data_dir() / "logs"


def ensure_data_dirs() -> None:
    for d in (dumps_dir(), cache_dir(), plots_dir(), logs_dir()):
        d.mkdir(parents=True, exist_ok=True)


def transport() -> str:
    """ "stdio" (local clients) or "streamable-http" (remote/cloud)."""
    return os.environ.get("DSP_TRANSPORT", "stdio")


def http_host() -> str:
    return os.environ.get("DSP_HOST", "0.0.0.0")


def http_port() -> int:
    return int(os.environ.get("DSP_PORT", "8000"))


def auth_token() -> str | None:
    """Optional static bearer token for the HTTP transport (minimum-viable auth)."""
    return os.environ.get("DSP_AUTH_TOKEN") or None


def enabled_toolsets() -> list[str] | None:
    """Comma list of toolset names to register; None = all."""
    raw = os.environ.get("DSP_TOOLSETS")
    if not raw:
        return None
    return [name.strip() for name in raw.split(",") if name.strip()]

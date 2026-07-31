"""Central configuration: every knob is env-overridable so the same package runs
as a local stdio server (Claude Code/Desktop), in CI against the stub engine, and
in a cloud container (streamable HTTP, no engine binary present).
"""

from __future__ import annotations

import os
from pathlib import Path

# Fallback only: an engine checkout sitting next to this repository's own directory
# (…/<parent>/proje10 -> …/<parent>/proje7-engine). Set DSP_ENGINE_ROOT when it lives
# elsewhere, or DSP_ENGINE_CLI to pin one exe. Only extract_mesh_telemetry reads this;
# the other 48 tools never touch the engine.
DEFAULT_ENGINE_ROOT = Path(__file__).resolve().parents[2].parent / "proje7-engine"

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
    """Bind address for the HTTP transport — loopback by default, deliberately.

    Two reasons the default is not ``0.0.0.0``: binding every interface publishes
    all 49 tools (arbitrary file paths in, files written under ``data/``) to the
    whole network, and the MCP SDK only auto-enables its DNS-rebinding protection
    when the host is ``127.0.0.1``/``localhost``/``::1`` (see FastMCP's
    ``transport_security`` default). Containers set ``DSP_HOST=0.0.0.0`` explicitly
    — there the published port, not the bind address, is the isolation boundary.
    """
    return os.environ.get("DSP_HOST", "127.0.0.1")


def http_port() -> int:
    return int(os.environ.get("DSP_PORT", "8000"))


def auth_token() -> str | None:
    """Static bearer token for the HTTP transport (minimum-viable auth).

    The empty string is treated as unset, so ``DSP_AUTH_TOKEN=""`` fails closed rather than
    degrading into an always-matching credential. Any other value is used verbatim — a
    whitespace-only value is a (bad) token, not an absence.
    """
    return os.environ.get("DSP_AUTH_TOKEN") or None


def allow_insecure_http() -> bool:
    """Explicit opt-in to serving streamable-HTTP with no bearer token.

    Only the exact string ``1`` opts in; anything else (including ``true``/``yes``)
    leaves the server fail-closed. Without a token and without this flag the HTTP
    transport refuses to start — see ``dsp_server.server._resolve_http_auth``.
    """
    return os.environ.get("DSP_ALLOW_INSECURE_HTTP") == "1"


def enabled_toolsets() -> list[str] | None:
    """Comma list of toolset names to register; None = all."""
    raw = os.environ.get("DSP_TOOLSETS")
    if not raw or not raw.strip():  # unset OR whitespace-only -> all packs (was: whitespace -> none)
        return None
    return [name.strip() for name in raw.split(",") if name.strip()]

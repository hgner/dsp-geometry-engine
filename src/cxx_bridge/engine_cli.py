"""Subprocess bridge to the proje7-engine dump/render CLIs.

One module owns everything process-shaped: discovery (env override or well-known
build paths), ``--help`` feature detection (so the bridge works against today's
unpatched ``layered_field_dump_cli.exe`` and silently gains the M5 flags once the
patch lands), output naming, stderr teeing/parsing, the stale-binary guard, and
error shaping. Contract for the MCP layer: **nothing in here raises** — every
failure comes back as an :class:`EngineRunResult` with ``error`` set, and no numpy
array ever crosses this boundary.
"""

from __future__ import annotations

import functools
import json
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dsp_server import config
from dsp_server.engine.ply import DumpMeta, parse_dump_stderr, save_meta

# The flags added by the M5 engine patch (branch feat/layered-dump-import-character).
# They are feature-detected via --help and silently omitted on unpatched binaries —
# except --character, whose absence is a hard, explained failure (see run_field_dump).
PATCHED_FLAGS = ("--character", "--time", "--palette-out", "--weights")

_M5_HINT = (
    "this engine binary predates the M5 patch; build branch "
    "feat/layered-dump-import-character of proje7-engine (adds "
    "--character/--time/--palette-out/--weights to layered_field_dump_cli) "
    "or point DSP_ENGINE_CLI at a patched build"
)

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]")
_SOURCE_EXTS = frozenset({".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx", ".cmake"})
_CMAKE_NAMES = frozenset({"CMakeLists.txt", "CMakePresets.json"})
_STDERR_TAIL_LINES = 50
_HELP_TIMEOUT_S = 30.0
_STALE_TTL_S = 30.0

# str(exe) -> (monotonic timestamp, staleness dict). See clear_caches().
_stale_cache: dict[str, tuple[float, dict]] = {}


# --------------------------------------------------------------------------- #
# Discovery + feature detection


def resolve_engine_cli() -> list[str]:
    """Return the argv prefix for the dump CLI.

    ``DSP_ENGINE_CLI`` wins and is a SINGLE path — never shell-split (Windows
    paths contain backslashes); a ``.py`` suffix means "run via this interpreter"
    (the CI stub engine). Otherwise the first existing candidate under the engine
    root is used. Raises FileNotFoundError listing every searched path.
    """
    override = config.engine_cli_override()
    if override:
        if override.lower().endswith(".py"):
            return [sys.executable, override]
        return [override]
    root = config.engine_root()
    searched: list[str] = []
    for rel in config.ENGINE_CLI_CANDIDATES:
        candidate = root / rel
        if candidate.is_file():
            return [str(candidate)]
        searched.append(str(candidate))
    raise FileNotFoundError(
        "engine CLI not found — set DSP_ENGINE_CLI to layered_field_dump_cli(.exe|.py) "
        "or build it (scripts/build-engine.ps1). Searched: " + "; ".join(searched)
    )


@functools.lru_cache(maxsize=8)
def _engine_help_cached(argv0: tuple[str, ...]) -> str:
    proc = subprocess.run(
        [*argv0, "--help"],
        shell=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=_HELP_TIMEOUT_S,
    )
    return (proc.stdout or "") + (proc.stderr or "")


def engine_help(argv0: Sequence[str]) -> str:
    """The CLI's ``--help`` text (stdout+stderr), memoized per argv."""
    return _engine_help_cached(tuple(argv0))


def supports_flag(flag: str) -> bool:
    """Does the resolved engine CLI advertise ``flag`` in its --help text?"""
    try:
        return flag in engine_help(resolve_engine_cli())
    except (OSError, subprocess.SubprocessError):
        return False


def clear_caches() -> None:
    """Test hook: forget memoized --help text and staleness verdicts."""
    _engine_help_cached.cache_clear()
    _stale_cache.clear()


# --------------------------------------------------------------------------- #
# Result shaping


@dataclass
class EngineRunResult:
    """What every bridge invocation returns. ``error`` is None on success, else
    ``{"kind": "timeout"|"engine-exit"|"not-found"|"flag-unsupported", "detail": …,
    "argv": …, ["hint"|"flag"|…]}``. Never raised across the MCP boundary."""

    ok: bool = False
    exit_code: int | None = None
    ply_path: Path | None = None
    palette_path: Path | None = None
    meta: DumpMeta | None = None
    stderr_tail: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    engine_stale: dict | None = None
    error: dict | None = None


def _tail(text: str) -> list[str]:
    return text.splitlines()[-_STDERR_TAIL_LINES:]


def _coerce_text(raw: str | bytes | None) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return raw


def _run(argv: list[str], timeout_s: float) -> tuple[EngineRunResult, str]:
    """Low-level runner (exposed for tests): subprocess discipline + error shaping.

    Returns ``(result, full_stderr_text)``; the result carries only the 50-line
    tail. cwd is the engine root when it exists (the exe resolves assets relative
    to its repo), else the current directory (CI / cloud analysis-only mode).
    """
    start = time.monotonic()
    cwd = config.engine_root()
    if not cwd.is_dir():
        cwd = Path.cwd()
    try:
        proc = subprocess.run(
            argv,
            shell=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            cwd=str(cwd),
        )
    except subprocess.TimeoutExpired as exc:
        stderr_text = _coerce_text(exc.stderr)
        result = EngineRunResult(
            ok=False,
            exit_code=None,
            stderr_tail=_tail(stderr_text),
            duration_s=time.monotonic() - start,
            error={
                "kind": "timeout",
                "detail": f"engine exceeded {timeout_s:g} s and was killed",
                "argv": list(argv),
            },
        )
        return result, stderr_text
    except OSError as exc:
        result = EngineRunResult(
            ok=False,
            duration_s=time.monotonic() - start,
            error={"kind": "not-found", "detail": str(exc), "argv": list(argv)},
        )
        return result, ""
    stderr_text = proc.stderr or ""
    result = EngineRunResult(
        ok=proc.returncode == 0,
        exit_code=proc.returncode,
        stderr_tail=_tail(stderr_text),
        duration_s=time.monotonic() - start,
    )
    if proc.returncode != 0:
        result.error = {
            "kind": "engine-exit",
            "detail": f"engine exited with code {proc.returncode}",
            "argv": list(argv),
        }
    return result, stderr_text


# --------------------------------------------------------------------------- #
# Output naming


def _sanitize(text: str) -> str:
    return _SANITIZE_RE.sub("_", text) or "dump"


def _dump_stem(label: str | None, character_json: str | Path | None, sex: str) -> str:
    if label:
        return _sanitize(label)
    if character_json is not None:
        name = Path(character_json).name
        if name.lower().endswith(".json"):
            name = name[: -len(".json")]
        if name.lower().endswith(".baked"):
            name = name[: -len(".baked")]
        return _sanitize(name)
    return _sanitize(sex)


def _unique_ply_path(dumps_dir: Path, base: str) -> Path:
    """``{base}.ply``, or ``{base}-1.ply``/``-2``… on a same-second collision."""
    candidate = dumps_dir / f"{base}.ply"
    counter = 0
    while candidate.exists():
        counter += 1
        candidate = dumps_dir / f"{base}-{counter}.ply"
    return candidate


def _tee_stderr(log_path: Path, argv: list[str], stderr_text: str) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"# argv: {json.dumps(argv)}\n{stderr_text}", encoding="utf-8")
    except OSError:
        pass  # logging must never fail a run


# --------------------------------------------------------------------------- #
# Stale-binary guard (port of proje8 shouldRebuildEngineCli's mtime comparison —
# warn only, NEVER auto-build: 17-minute builds would blow MCP timeouts).


def _iso(mtime: float) -> str:
    return datetime.fromtimestamp(mtime).isoformat(timespec="seconds")


def _compute_staleness(exe: Path) -> dict:
    root = config.engine_root()
    if not root.is_dir():
        return {"stale": False, "hint": "engine repo not present (cloud analysis-only mode)"}
    try:
        exe_mtime = exe.stat().st_mtime
        newest_mtime = 0.0
        newest_path: Path | None = None
        roots = (
            root / "src" / "core",
            root / "tools" / "layered-field-dump",
            root / "CMakeLists.txt",
            root / "CMakePresets.json",
        )
        for entry in roots:
            if entry.is_file():
                candidates: list[Path] = [entry]
            elif entry.is_dir():
                candidates = [
                    p for p in entry.rglob("*") if p.suffix.lower() in _SOURCE_EXTS or p.name in _CMAKE_NAMES
                ]
            else:
                continue
            for p in candidates:
                try:
                    m = p.stat().st_mtime
                except OSError:
                    continue
                if m > newest_mtime:
                    newest_mtime = m
                    newest_path = p
        stale = newest_mtime > exe_mtime
        newest = (
            {"path": str(newest_path), "mtimeIso": _iso(newest_mtime)} if newest_path is not None else None
        )
        hint = None
        if stale:
            hint = (
                "engine binary is older than its sources — results may not reflect current "
                "code; rebuild via scripts/build-engine.ps1 (the bridge warns only, it never "
                "auto-builds)"
            )
        return {
            "path": str(exe),
            "mtimeIso": _iso(exe_mtime),
            "stale": stale,
            "newestSource": newest,
            "hint": hint,
        }
    except OSError as exc:
        return {
            "path": str(exe),
            "stale": False,
            "newestSource": None,
            "hint": f"staleness check skipped ({exc})",
        }


def engine_staleness(exe: Path) -> dict:
    """mtime-compare ``exe`` against the engine sources; memoized for 30 s."""
    key = str(exe)
    now = time.monotonic()
    cached = _stale_cache.get(key)
    if cached is not None and now - cached[0] < _STALE_TTL_S:
        return cached[1]
    info = _compute_staleness(exe)
    _stale_cache[key] = (now, info)
    return info


# --------------------------------------------------------------------------- #
# The main entry point


def run_field_dump(
    *,
    character_json: str | Path | None = None,
    sex: str = "male",
    pose_clip: str | None = None,
    time_s: float | None = None,
    label: str | None = None,
    want_palette: bool = True,
    want_weights: bool = True,
    timeout_s: float = 600.0,
) -> EngineRunResult:
    """Invoke ``layered_field_dump_cli`` and shape everything into EngineRunResult.

    Feature-detects the M5 flags via --help: ``--time``/``--palette-out``/``--weights``
    are silently omitted on unpatched binaries (``palette_path`` stays None), while a
    ``character_json`` request against an unpatched binary is a hard
    ``flag-unsupported`` error naming the patch. Never raises.
    """
    try:
        return _run_field_dump(
            character_json=character_json,
            sex=sex,
            pose_clip=pose_clip,
            time_s=time_s,
            label=label,
            want_palette=want_palette,
            want_weights=want_weights,
            timeout_s=timeout_s,
        )
    except Exception as exc:  # defensive: the MCP boundary must never see a raise
        return EngineRunResult(
            ok=False,
            error={"kind": "engine-exit", "detail": f"bridge failure: {exc!r}", "argv": None},
        )


def _run_field_dump(
    *,
    character_json: str | Path | None,
    sex: str,
    pose_clip: str | None,
    time_s: float | None,
    label: str | None,
    want_palette: bool,
    want_weights: bool,
    timeout_s: float,
) -> EngineRunResult:
    start = time.monotonic()
    try:
        argv0 = resolve_engine_cli()
    except FileNotFoundError as exc:
        return EngineRunResult(
            ok=False,
            duration_s=time.monotonic() - start,
            error={"kind": "not-found", "detail": str(exc), "argv": None},
        )

    try:
        help_text = engine_help(argv0)
    except subprocess.TimeoutExpired:
        return EngineRunResult(
            ok=False,
            duration_s=time.monotonic() - start,
            error={
                "kind": "timeout",
                "detail": f"--help probe exceeded {_HELP_TIMEOUT_S:g} s",
                "argv": [*argv0, "--help"],
            },
        )
    except OSError as exc:
        return EngineRunResult(
            ok=False,
            duration_s=time.monotonic() - start,
            error={"kind": "not-found", "detail": str(exc), "argv": [*argv0, "--help"]},
        )

    stale_info = engine_staleness(Path(argv0[-1]))

    if character_json is not None and "--character" not in help_text:
        return EngineRunResult(
            ok=False,
            duration_s=time.monotonic() - start,
            engine_stale=stale_info,
            error={
                "kind": "flag-unsupported",
                "detail": "--character is not supported by this engine binary",
                "flag": "--character",
                "argv": argv0,
                "hint": _M5_HINT,
            },
        )

    config.ensure_data_dirs()
    stem = _dump_stem(label, character_json, sex)
    pose_part = _sanitize(pose_clip) if pose_clip else "rest"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_ply = _unique_ply_path(config.dumps_dir(), f"{stem}_{pose_part}_{timestamp}")

    argv = [*argv0, "--sex", sex, "--out", str(out_ply)]
    if pose_clip:
        argv += ["--pose", pose_clip]
    if character_json is not None:
        argv += ["--character", str(character_json)]
    if time_s is not None and "--time" in help_text:
        argv += ["--time", f"{time_s:g}"]
    palette_target: Path | None = None
    if want_palette and "--palette-out" in help_text:
        palette_target = out_ply.with_suffix(".palette.json")
        argv += ["--palette-out", str(palette_target)]
    if want_weights and "--weights" in help_text:
        argv.append("--weights")

    result, stderr_text = _run(argv, timeout_s)
    result.duration_s = time.monotonic() - start
    result.engine_stale = stale_info
    _tee_stderr(config.logs_dir() / f"{out_ply.stem}.stderr.log", argv, stderr_text)
    result.meta = parse_dump_stderr(stderr_text)
    result.ply_path = out_ply if out_ply.is_file() else None
    if palette_target is not None and palette_target.is_file():
        result.palette_path = palette_target
    if result.ok:
        if result.ply_path is None:
            result.ok = False
            result.error = {
                "kind": "engine-exit",
                "detail": f"engine exited 0 but wrote no PLY at {out_ply}",
                "argv": argv,
            }
        else:
            save_meta(result.ply_path, result.meta)
    return result


# --------------------------------------------------------------------------- #
# Optional M8 lane: depth/AOV renders via the RTX render CLI (feature-detected,
# never blocks the analysis tools — they operate on PNGs already on disk).


def run_depth_render(
    *,
    character_json: str | Path,
    clip: str,
    out_dir: str | Path | None = None,
    frames: int = 1,
    timeout_s: float = 600.0,
) -> dict:
    """Spawn ``i2v_beauty_dxr_cli`` for fresh depth/AOV PNGs (requires the local
    RTX build + GPU). Returns an EngineRunResult-like dict; when the render CLI is
    absent the error points at ``compare_depth_renders`` on existing PNGs."""
    try:
        root = config.engine_root()
        exe: Path | None = None
        searched: list[str] = []
        for rel in config.RENDER_CLI_CANDIDATES:
            candidate = root / rel
            if candidate.is_file():
                exe = candidate
                break
            searched.append(str(candidate))
        if exe is None:
            return {
                "ok": False,
                "exit_code": None,
                "out_dir": None,
                "stderr_tail": [],
                "duration_s": 0.0,
                "error": {
                    "kind": "not-found",
                    "detail": "render CLI not found; searched: " + "; ".join(searched),
                    "argv": None,
                    "hint": (
                        "depth renders need the local RTX build + GPU — analyze PNGs "
                        "already under data/ with compare_depth_renders instead"
                    ),
                },
            }
        config.ensure_data_dirs()
        if out_dir is None:
            out_dir = config.data_dir() / "renders" / datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            str(exe),
            "--character",
            str(character_json),
            "--clip",
            clip,
            "--aov",
            "--out",
            str(out_dir),
            "--frames",
            str(frames),
        ]
        result, stderr_text = _run(argv, timeout_s)
        _tee_stderr(config.logs_dir() / f"render_{out_dir.name}.stderr.log", argv, stderr_text)
        return {
            "ok": result.ok,
            "exit_code": result.exit_code,
            "out_dir": str(out_dir),
            "stderr_tail": result.stderr_tail,
            "duration_s": result.duration_s,
            "error": result.error,
        }
    except Exception as exc:  # defensive: the MCP boundary must never see a raise
        return {
            "ok": False,
            "exit_code": None,
            "out_dir": None,
            "stderr_tail": [],
            "duration_s": 0.0,
            "error": {"kind": "engine-exit", "detail": f"bridge failure: {exc!r}", "argv": None},
        }

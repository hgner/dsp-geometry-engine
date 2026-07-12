"""Blender/add-on discovery without importing Blender's Python modules."""

from __future__ import annotations

import os
import re
import subprocess
import tomllib
from pathlib import Path

from bodymesh_server import config
from bodymesh_server.schemas import AddonInfo, RuntimeReport

_VERSION_RE = re.compile(r'"version"\s*:\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)')


def _shortcut_target(shortcut: Path) -> Path | None:
    if os.name != "nt" or not shortcut.is_file():
        return None
    keep = {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "PROGRAMDATA",
        "ALLUSERSPROFILE",
        "TEMP",
        "TMP",
        "USERPROFILE",
    }
    lookup = {key.upper(): (key, value) for key, value in os.environ.items()}
    env = {lookup[key][0]: lookup[key][1] for key in keep if key in lookup}
    env["BODYMESH_SHORTCUT_TO_RESOLVE"] = str(shortcut)
    command = (
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut("
        "$env:BODYMESH_SHORTCUT_TO_RESOLVE); [Console]::Out.Write($s.TargetPath)"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            shell=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=10.0,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    raw_target = (proc.stdout or "").strip()
    return Path(raw_target) if proc.returncode == 0 and raw_target else None


def resolve_blender_executable() -> Path:
    """Resolve a deterministic CLI executable, never ``blender-launcher.exe``."""

    override = config.blender_exe_override()
    if override:
        candidate = Path(override)
        if candidate.suffix.lower() == ".lnk":
            candidate = _shortcut_target(candidate) or candidate
    else:
        candidate = _shortcut_target(config.blender_shortcut()) or config.DEFAULT_BLENDER_EXE

    if candidate.name.lower() == "blender-launcher.exe":
        candidate = candidate.with_name("blender.exe")
    return candidate


def _mpfb_info() -> AddonInfo:
    root = config.mpfb_root()
    version: str | None = None
    manifest = root / "blender_manifest.toml"
    if manifest.is_file():
        try:
            version = str(tomllib.loads(manifest.read_text(encoding="utf-8")).get("version") or "") or None
        except (OSError, tomllib.TOMLDecodeError):
            version = None
    return AddonInfo(
        backend="mpfb",
        installed=root.is_dir() and manifest.is_file(),
        enabled_hint=(root.parent.name == "user_default") if root.is_dir() else None,
        version=version,
        path=str(root),
        supported=bool(root.is_dir() and version and version.startswith("2.0.")),
        note="Primary backend; generated core meshes are unrestricted under MPFB's asset terms.",
    )


def _mblab_info() -> AddonInfo:
    root = config.mblab_root()
    version: str | None = None
    init_file = root / "__init__.py"
    if init_file.is_file():
        try:
            match = _VERSION_RE.search(init_file.read_text(encoding="utf-8", errors="replace"))
            if match:
                version = ".".join(match.groups())
        except OSError:
            pass
    return AddonInfo(
        backend="mblab",
        installed=root.is_dir() and init_file.is_file(),
        enabled_hint=None,
        version=version,
        path=str(root),
        supported=False,
        note=(
            "Detected only: MB-LAB is archived and its generated meshes default to AGPL-3; "
            "this MCP does not invoke it."
        ),
    )


def inspect_runtime() -> RuntimeReport:
    exe = resolve_blender_executable()
    warnings: list[str] = []
    if not exe.is_file():
        warnings.append("Blender executable is missing; set BODYMESH_BLENDER_EXE to blender.exe")
    missing_roots = [str(root) for root in config.input_roots() if not root.exists()]
    if missing_roots:
        warnings.append("input roots not yet present: " + ", ".join(missing_roots))
    addons = [_mpfb_info(), _mblab_info()]
    if not addons[0].supported:
        warnings.append("MPFB 2.0.x was not detected; generation is blocked until it is installed")
    return RuntimeReport(
        blender_shortcut=str(config.blender_shortcut()),
        blender_executable=str(exe) if str(exe) else None,
        blender_exists=exe.is_file(),
        data_dir=str(config.data_dir()),
        input_roots=[str(root) for root in config.input_roots()],
        addons=addons,
        warnings=warnings,
    )

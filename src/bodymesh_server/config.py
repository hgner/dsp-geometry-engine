"""Configuration for the local-only Blender body-mesh MCP."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_BLENDER_SHORTCUT = Path("C:/Users/hgner/Desktop/Blender 4.2.lnk")
DEFAULT_BLENDER_EXE = Path("C:/Program Files/Blender Foundation/Blender 4.2/blender.exe")
DEFAULT_MPFB_ROOT = Path(
    "C:/Users/hgner/AppData/Roaming/Blender Foundation/Blender/4.2/extensions/user_default/mpfb"
)
DEFAULT_MBLAB_ROOT = Path(
    "C:/Users/hgner/AppData/Roaming/Blender Foundation/Blender/4.2/scripts/addons/MB-Lab-master"
)
DEFAULT_ENGINE_SKELETON_DIR = Path("C:/Users/hgner/hakantest/proje8/scripts/blender")
DEFAULT_CHARACTER_BAKE_EXE = Path(
    "C:/Users/hgner/hakantest/proje7-engine/build/windows-msvc-static-md-release/character_bake_cli.exe"
)


def data_dir() -> Path:
    return Path(os.environ.get("BODYMESH_DATA_DIR", str(Path.cwd() / "data" / "bodymesh")))


def jobs_dir() -> Path:
    return data_dir() / "jobs"


def inbox_dir() -> Path:
    return data_dir() / "inbox"


def logs_dir() -> Path:
    return data_dir() / "logs"


def blender_exe_override() -> str | None:
    return os.environ.get("BODYMESH_BLENDER_EXE") or None


def blender_shortcut() -> Path:
    return Path(os.environ.get("BODYMESH_BLENDER_SHORTCUT", str(DEFAULT_BLENDER_SHORTCUT)))


def mpfb_root() -> Path:
    return Path(os.environ.get("BODYMESH_MPFB_ROOT", str(DEFAULT_MPFB_ROOT)))


def mblab_root() -> Path:
    return Path(os.environ.get("BODYMESH_MBLAB_ROOT", str(DEFAULT_MBLAB_ROOT)))


def engine_skeleton_dir() -> Path:
    return Path(os.environ.get("BODYMESH_ENGINE_SKELETON_DIR", str(DEFAULT_ENGINE_SKELETON_DIR)))


def engine_skeleton_path(sex: str) -> Path:
    if sex not in {"male", "female"}:
        raise ValueError("rig_sex must be male or female")
    return engine_skeleton_dir() / f"skeleton_{sex}.json"


def character_bake_exe() -> Path:
    return Path(os.environ.get("BODYMESH_CHARACTER_BAKE_EXE", str(DEFAULT_CHARACTER_BAKE_EXE)))


def input_roots() -> tuple[Path, ...]:
    """Configured roots from which explicit image inputs may be copied.

    The default is intentionally narrow. Multiple Windows paths use ``;``; on
    POSIX the platform path separator is used.
    """

    raw = os.environ.get("BODYMESH_INPUT_ROOTS")
    if not raw:
        return (inbox_dir(),)
    roots = tuple(Path(item.strip()) for item in raw.split(os.pathsep) if item.strip())
    return roots or (inbox_dir(),)


def timeout_s() -> float:
    value = float(os.environ.get("BODYMESH_TIMEOUT_S", "600"))
    if not 10.0 <= value <= 3600.0:
        raise ValueError("BODYMESH_TIMEOUT_S must be in [10, 3600]")
    return value


def ensure_data_dirs() -> None:
    for directory in (data_dir(), jobs_dir(), inbox_dir(), logs_dir()):
        directory.mkdir(parents=True, exist_ok=True)

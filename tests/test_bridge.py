"""Bridge-vs-stub end-to-end tests: discovery, feature detection, run shaping,
output naming, stderr parsing/teeing, timeout, and staleness — no C++ required.

Every test spawns at most two short-lived stub processes; the suite stays well
under the ~10 s/test budget.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pytest

from cxx_bridge import engine_cli
from synth import ARM_LOWER_L

PLY_NAME_RE = r"_\d{8}-\d{6}(?:-\d+)?\.ply"


def test_run_field_dump_posed_success(stub_engine_env: Path) -> None:
    res = engine_cli.run_field_dump(sex="male", pose_clip="clip-cin-stand-attention")
    assert res.error is None
    assert res.ok and res.exit_code == 0
    assert res.duration_s > 0.0

    # PLY written under the sandboxed dumps dir with stem_pose_timestamp naming.
    assert res.ply_path is not None and res.ply_path.is_file()
    assert res.ply_path.parent == stub_engine_env / "dumps"
    assert re.fullmatch("male_clip-cin-stand-attention" + PLY_NAME_RE, res.ply_path.name)

    # stderr protocol parsed: bone map, clip, and the collisionPush float.
    assert res.meta is not None
    assert res.meta.bone_map[ARM_LOWER_L] == "armLowerL"
    assert res.meta.clip_id == "clip-cin-stand-attention"
    assert res.meta.vert_count == 64 * 16
    assert res.meta.collision_push == 0.0012
    assert res.meta.baked_collision_push == 0.0

    # Sidecars: .meta.json next to the ply, full stderr teed to data/logs/.
    assert res.ply_path.with_suffix(".meta.json").is_file()
    log_path = stub_engine_env / "logs" / f"{res.ply_path.stem}.stderr.log"
    assert log_path.is_file()
    assert "bone-map:" in log_path.read_text(encoding="utf-8")

    # Staleness info rides along on every completed run.
    assert isinstance(res.engine_stale, dict)
    assert "stale" in res.engine_stale


def test_palette_sidecar_written_and_parseable(stub_engine_env: Path) -> None:
    res = engine_cli.run_field_dump(sex="male", pose_clip="clip-cin-walktalk", want_palette=True)
    assert res.ok
    assert res.palette_path is not None and res.palette_path.is_file()
    payload = json.loads(res.palette_path.read_text(encoding="utf-8"))
    assert res.meta is not None
    assert payload["boneCount"] == len(res.meta.bone_map) == len(payload["bones"])
    assert payload["bones"][ARM_LOWER_L]["name"] == "armLowerL"
    assert payload["bones"][ARM_LOWER_L]["parent"] == ARM_LOWER_L - 1
    assert payload["matrixLayout"] == "column-major"
    assert payload["importScale"] == 1.0
    assert payload["clipId"] == "clip-cin-walktalk"


def test_character_json_missing_file_is_engine_exit_1(stub_engine_env: Path, tmp_path: Path) -> None:
    res = engine_cli.run_field_dump(
        character_json=tmp_path / "missing.baked.json", pose_clip="clip-cin-walktalk"
    )
    assert not res.ok
    assert res.exit_code == 1
    assert res.error is not None
    assert res.error["kind"] == "engine-exit"
    assert any("--character rejected" in line for line in res.stderr_tail)


def test_help_feature_detection_on_patched_stub(stub_engine_env: Path) -> None:
    argv0 = engine_cli.resolve_engine_cli()
    assert argv0[-1].endswith("stub_engine.py")
    text = engine_cli.engine_help(argv0)
    for flag in engine_cli.PATCHED_FLAGS:
        assert flag in text
    assert engine_cli.supports_flag("--character")
    assert not engine_cli.supports_flag("--definitely-not-a-flag")


def test_character_on_unpatched_engine_is_flag_unsupported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A stub-variant whose --help lacks the M5 flags (today's real binary).
    variant = tmp_path / "old_engine.py"
    variant.write_text(
        "import sys\n"
        "sys.stdout.write('usage: layered_field_dump_cli --sex male|female "
        "--out <path> [--pose <clipId>] [--list-sources]\\n')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSP_ENGINE_CLI", str(variant))
    monkeypatch.setenv("DSP_DATA_DIR", str(tmp_path / "data"))
    engine_cli.clear_caches()

    res = engine_cli.run_field_dump(character_json=tmp_path / "c.baked.json", pose_clip="clip-x")
    assert not res.ok
    assert res.error is not None
    assert res.error["kind"] == "flag-unsupported"
    assert res.error["flag"] == "--character"
    assert "feat/layered-dump-import-character" in res.error["hint"]


def test_timeout_via_low_level_run(stub_engine_env: Path, tmp_path: Path) -> None:
    out = tmp_path / "slow.ply"
    argv = [*engine_cli.resolve_engine_cli(), "--sex", "male", "--out", str(out), "--sleep", "5"]
    result, stderr_text = engine_cli._run(argv, timeout_s=0.5)
    assert not result.ok
    assert result.exit_code is None
    assert result.error is not None
    assert result.error["kind"] == "timeout"
    assert result.error["argv"] == argv
    assert isinstance(stderr_text, str)


def test_discovery_not_found_never_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DSP_ENGINE_CLI", raising=False)
    monkeypatch.setenv("DSP_ENGINE_ROOT", str(tmp_path))  # exists, but holds no exe
    monkeypatch.setenv("DSP_DATA_DIR", str(tmp_path / "data"))
    engine_cli.clear_caches()

    with pytest.raises(FileNotFoundError) as excinfo:
        engine_cli.resolve_engine_cli()
    for rel in ("windows-msvc-static-md-release", "windows-msvc-rtx"):
        assert rel in str(excinfo.value)  # every searched path is listed

    res = engine_cli.run_field_dump(sex="male")
    assert not res.ok
    assert res.error is not None
    assert res.error["kind"] == "not-found"


def test_output_naming_label_sanitized_and_rest_pose(stub_engine_env: Path) -> None:
    res = engine_cli.run_field_dump(label="Forearm probe!v2", pose_clip=None)
    assert res.ok
    assert res.ply_path is not None
    assert re.fullmatch("Forearm_probe_v2_rest" + PLY_NAME_RE, res.ply_path.name)
    assert res.meta is not None and res.meta.clip_id is None  # rest mode: no posed-ok line


def test_same_second_collision_gets_numeric_suffix(tmp_path: Path) -> None:
    base = "male_rest_20260711-000000"
    (tmp_path / f"{base}.ply").write_text("taken", encoding="utf-8")
    first = engine_cli._unique_ply_path(tmp_path, base)
    assert first.name == f"{base}-1.ply"
    first.write_text("taken", encoding="utf-8")
    assert engine_cli._unique_ply_path(tmp_path, base).name == f"{base}-2.ply"


def test_engine_staleness_missing_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DSP_ENGINE_ROOT", str(tmp_path / "no-such-repo"))
    info = engine_cli.engine_staleness(tmp_path / "engine.exe")
    assert info["stale"] is False
    assert "engine repo not present" in info["hint"]


def test_engine_staleness_source_newer_than_exe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "engine"
    (root / "src" / "core").mkdir(parents=True)
    (root / "tools" / "layered-field-dump").mkdir(parents=True)
    exe = root / "build" / "preset" / "layered_field_dump_cli.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("binary", encoding="utf-8")
    newer_src = root / "src" / "core" / "CharacterDeformers.cpp"
    newer_src.write_text("// edited after the build", encoding="utf-8")
    (root / "src" / "core" / "notes.md").write_text("ignored ext", encoding="utf-8")
    hour_ago = time.time() - 3600.0
    os.utime(exe, (hour_ago, hour_ago))
    monkeypatch.setenv("DSP_ENGINE_ROOT", str(root))

    info = engine_cli.engine_staleness(exe)
    assert info["stale"] is True
    assert info["newestSource"]["path"].endswith("CharacterDeformers.cpp")
    assert "rebuild" in info["hint"]
    assert info["mtimeIso"] < info["newestSource"]["mtimeIso"]

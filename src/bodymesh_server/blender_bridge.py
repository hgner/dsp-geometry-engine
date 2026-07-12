"""Safe subprocess boundary between the MCP server and Blender/MPFB."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, UnidentifiedImageError

from bodymesh_server import config
from bodymesh_server.engine_contract import validate_engine_contract
from bodymesh_server.environment import inspect_runtime, resolve_blender_executable
from bodymesh_server.jobs import create_candidate_dir, load_job, record_candidate
from bodymesh_server.locking import InterprocessLock, LockUnavailable
from bodymesh_server.paths import confined_child, normalize_reference_mask
from bodymesh_server.schemas import CandidateResult, ParameterCatalog
from dsp_server.engine.ply import load_meta, read_engine_ply

MACRO_HELP = {
    "gender": "0=feminine, 0.5=neutral, 1=masculine",
    "age": "0=baby, about 0.5=young adult, 1=old",
    "muscle": "0=minimal, 1=maximal",
    "weight": "0=minimal, 1=maximal",
    "proportions": "0=minimal, 1=maximal MakeHuman proportion target",
    "height": "0=shortest target, 1=tallest target; known_height_m applies final physical scale",
    "cupsize": "0=minimal, 1=maximal",
    "firmness": "0=minimal, 1=maximal",
    "african": "race blend weight; supplied race weights are normalized",
    "asian": "race blend weight; supplied race weights are normalized",
    "caucasian": "race blend weight; supplied race weights are normalized",
}
_RACE_KEYS = frozenset({"african", "asian", "caucasian"})
_VIEW_NAMES = frozenset({"front", "side", "back"})
_BLENDER_LOCK = threading.Lock()
_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "PROGRAMDATA",
        "ALLUSERSPROFILE",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "TEMP",
        "TMP",
        "HOMEDRIVE",
        "HOMEPATH",
        "LANG",
        "BLENDER_USER_CONFIG",
        "BLENDER_USER_SCRIPTS",
        "BLENDER_SYSTEM_SCRIPTS",
    }
)


class BlenderBridgeError(RuntimeError):
    def __init__(self, message: str, *, tail: list[str] | None = None) -> None:
        super().__init__(message)
        self.tail = tail or []


class BlenderProcessTimeout(RuntimeError):
    def __init__(self, timeout_s: float, *, tree_terminated: bool) -> None:
        super().__init__(f"Blender exceeded {timeout_s:g}s")
        self.timeout_s = timeout_s
        self.tree_terminated = tree_terminated


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _mpfb_version() -> str | None:
    runtime = inspect_runtime()
    for addon in runtime.addons:
        if addon.backend == "mpfb":
            return addon.version
    return None


def _target_paths() -> list[Path]:
    root = config.mpfb_root() / "data" / "targets"
    if not root.is_dir():
        return []
    found = list(root.rglob("*.target"))
    found.extend(root.rglob("*.target.gz"))
    return sorted({path for path in found if path.is_file()}, key=lambda path: path.as_posix().lower())


def parameter_catalog(query: str | None = None, limit: int = 40) -> ParameterCatalog:
    if not 1 <= int(limit) <= 100:
        raise ValueError("limit must be in [1, 100]")
    matches: list[str] = []
    truncated = False
    if query is not None:
        needle = query.strip().lower()
        if len(needle) < 2:
            raise ValueError("target query must contain at least two non-space characters")
        root = config.mpfb_root() / "data" / "targets"
        all_matches = []
        for path in _target_paths():
            relative_name = path.relative_to(root).as_posix()
            if needle in relative_name.lower():
                all_matches.append(relative_name)
        truncated = len(all_matches) > int(limit)
        matches = all_matches[: int(limit)]
    return ParameterCatalog(
        backend="mpfb",
        macro_parameters=MACRO_HELP,
        target_query=query,
        target_matches=matches,
        matches_truncated=truncated,
        note=(
            "This is a bounded parametric model, not photo inference. Use front/side masks and the DSP "
            "comparison tools to score candidates, then call create_body_mesh again with revised values."
        ),
    )


def _validate_macros(values: dict[str, float] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, raw_value in (values or {}).items():
        if key not in MACRO_HELP:
            raise ValueError(f"unknown macro parameter {key!r}; allowed: {sorted(MACRO_HELP)}")
        value = float(raw_value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"macro parameter {key!r} must be in [0, 1], got {value}")
        result[key] = value
    return result


def _validate_targets(values: dict[str, float] | None) -> dict[str, float]:
    if len(values or {}) > 32:
        raise ValueError("at most 32 detailed MPFB targets may be applied to one candidate")
    target_root = (config.mpfb_root() / "data" / "targets").resolve(strict=False)
    result: dict[str, float] = {}
    for raw_name, raw_value in (values or {}).items():
        name = PurePosixPath(raw_name)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"target path must be relative and traversal-free: {raw_name!r}")
        candidate = (target_root / Path(*name.parts)).resolve(strict=True)
        try:
            candidate.relative_to(target_root)
        except ValueError as exc:
            raise ValueError(f"target escapes the MPFB target root: {raw_name!r}") from exc
        lower = candidate.name.lower()
        if not (lower.endswith(".target") or lower.endswith(".target.gz")):
            raise ValueError(f"not an MPFB target file: {raw_name!r}")
        if not candidate.is_file():
            raise ValueError(f"MPFB target is not a regular file: {raw_name!r}")
        value = float(raw_value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"target weight {raw_name!r} must be in [0, 1], got {value}")
        result[name.as_posix()] = value
    return result


def _worker_script() -> Path:
    return Path(__file__).with_name("blender_scripts") / "mpfb_worker.py"


def _engine_retarget_script() -> Path:
    return Path(__file__).with_name("blender_scripts") / "engine_retarget_worker.py"


def _child_env() -> dict[str, str]:
    """Pass only OS/Blender profile variables; never leak MCP/cloud credentials."""

    lookup = {key.upper(): (key, value) for key, value in os.environ.items()}
    result: dict[str, str] = {}
    for wanted in _ENV_ALLOWLIST:
        pair = lookup.get(wanted.upper())
        if pair is not None:
            result[pair[0]] = pair[1]
    return result


def _argv(executable: Path, request: Path, result: Path) -> list[str]:
    prefix = [sys.executable, str(executable)] if executable.suffix.lower() == ".py" else [str(executable)]
    return [
        *prefix,
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(_worker_script()),
        "--",
        "--request",
        str(request),
        "--result",
        str(result),
    ]


def _retarget_argv(executable: Path, blend_path: Path, request: Path, result: Path) -> list[str]:
    prefix = [sys.executable, str(executable)] if executable.suffix.lower() == ".py" else [str(executable)]
    return [
        *prefix,
        "--background",
        str(blend_path),
        "--python-exit-code",
        "1",
        "--python",
        str(_engine_retarget_script()),
        "--",
        "--request",
        str(request),
        "--result",
        str(result),
    ]


def _program_argv(executable: Path, *arguments: str) -> list[str]:
    if executable.suffix.lower() == ".py":
        return [sys.executable, str(executable), *arguments]
    return [str(executable), *arguments]


def _validate_result_paths(
    candidate_dir: Path,
    payload: dict[str, Any],
    expected_views: set[str],
    expected_render_size: int,
) -> None:
    expected_files = {
        "blend_path": candidate_dir / "character.blend",
        "obj_path": candidate_dir / "body.obj",
        "ply_path": candidate_dir / "body_dsp.ply",
        "meta_path": candidate_dir / "body_dsp.meta.json",
    }
    for key, expected in expected_files.items():
        raw = payload.get(key)
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"worker result is missing required {key}")
        path = confined_child(candidate_dir, Path(raw))
        if path != expected.resolve(strict=False):
            raise ValueError(f"worker result {key} must be {expected}, got {path}")
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"worker result {key} is missing or empty: {path}")
    renders = payload.get("render_paths") or {}
    if not isinstance(renders, dict):
        raise ValueError("worker render_paths must be an object")
    if set(renders) != expected_views:
        raise ValueError(
            f"worker render views differ: expected {sorted(expected_views)}, got {sorted(renders)}"
        )
    for view, raw in renders.items():
        expected = (candidate_dir / "renders" / f"{view}_mask.png").resolve(strict=False)
        path = confined_child(candidate_dir, Path(str(raw)))
        if path != expected or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"worker render {view!r} is missing or misplaced: {path}")
        try:
            with Image.open(path) as opened:
                opened.verify()
            with Image.open(path) as opened:
                if opened.format != "PNG":
                    raise ValueError(f"worker render {view!r} is not PNG: {path}")
                if opened.size != (expected_render_size, expected_render_size):
                    raise ValueError(
                        f"worker render {view!r} has size {opened.size}; "
                        f"expected {(expected_render_size, expected_render_size)}"
                    )
                colors = opened.convert("L").getcolors(maxcolors=3)
                values = {value for _, value in colors or []}
                if values != {0, 255}:
                    raise ValueError(f"worker render {view!r} must contain only foreground/background pixels")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"worker render {view!r} is not a valid PNG: {path}") from exc

    vertex_count = int(payload.get("vertex_count") or 0)
    face_count = int(payload.get("face_count") or 0)
    if vertex_count < 3 or face_count < 1:
        raise ValueError(f"invalid worker mesh counts: vertices={vertex_count}, faces={face_count}")
    dump = read_engine_ply(expected_files["ply_path"])
    meta = load_meta(expected_files["ply_path"])
    if dump.vertex_count != vertex_count or dump.face_count != face_count:
        raise ValueError(
            "worker mesh counts do not match body_dsp.ply: "
            f"result=({vertex_count},{face_count}) ply=({dump.vertex_count},{dump.face_count})"
        )
    if meta is None or meta.vert_count != vertex_count or meta.bone_map != {0: "body"}:
        raise ValueError("body_dsp.meta.json does not match the neutral body PLY contract")


def _comparison_masks(
    job_manifest: dict[str, Any], candidate_dir: Path, size: int, views: set[str]
) -> dict[str, str]:
    result: dict[str, str] = {}
    references = job_manifest.get("references") or []
    if not isinstance(references, list):
        raise ValueError("job references must be a list")
    for reference in references:
        if not isinstance(reference, dict) or not reference.get("mask_path") or not reference.get("view"):
            continue
        view = str(reference["view"])
        if view not in views:
            continue
        destination = candidate_dir / "comparison" / f"{view}_reference_mask.png"
        normalize_reference_mask(Path(str(reference["mask_path"])), destination, size)
        result[view] = str(destination)
    return result


def _tail(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-50:]
    except OSError:
        return []


def _terminate_process_tree(proc: subprocess.Popen) -> bool:
    if proc.poll() is not None:
        return True
    tree_terminated = False
    if os.name == "nt":
        system_root = Path(os.environ.get("SystemRoot") or os.environ.get("SYSTEMROOT") or "C:/Windows")
        taskkill = system_root / "System32" / "taskkill.exe"
        if taskkill.is_file():
            try:
                completed = subprocess.run(
                    [str(taskkill), "/PID", str(proc.pid), "/T", "/F"],
                    shell=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=15.0,
                )
                tree_terminated = completed.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                tree_terminated = False
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            tree_terminated = True
        except ProcessLookupError:
            tree_terminated = True
        except OSError:
            tree_terminated = False
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        tree_terminated = False
        try:
            proc.kill()
            proc.wait(timeout=5.0)
        except (OSError, subprocess.TimeoutExpired):
            pass
    return tree_terminated


def _run_to_files(
    argv: list[str],
    candidate_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: float,
) -> int:
    popen_options: dict[str, Any] = {}
    if os.name == "nt":
        popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_options["start_new_session"] = True
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_stream:
        with stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_stream:
            proc = subprocess.Popen(
                argv,
                shell=False,
                stdout=stdout_stream,
                stderr=stderr_stream,
                text=True,
                cwd=str(candidate_dir),
                env=_child_env(),
                **popen_options,
            )
            try:
                return_code = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired as exc:
                tree_terminated = _terminate_process_tree(proc)
                raise BlenderProcessTimeout(timeout_s, tree_terminated=tree_terminated) from exc
    return return_code


def _run_to_logs(
    argv: list[str], candidate_dir: Path, timeout_s: float, *, prefix: str = "blender"
) -> tuple[int, Path, Path]:
    stdout_path = candidate_dir / f"{prefix}.stdout.log"
    stderr_path = candidate_dir / f"{prefix}.stderr.log"
    return_code = _run_to_files(argv, candidate_dir, stdout_path, stderr_path, timeout_s)
    return return_code, stdout_path, stderr_path


def _timeout_error(stage: str, exc: BlenderProcessTimeout, tail: list[str]) -> BlenderBridgeError:
    if exc.tree_terminated:
        message = f"{stage} exceeded {exc.timeout_s:g}s; process-tree termination was confirmed"
    else:
        message = (
            f"{stage} exceeded {exc.timeout_s:g}s; the parent was killed but descendant "
            "termination could not be confirmed"
        )
    return BlenderBridgeError(message, tail=tail)


def _validate_engine_retarget_result(candidate_dir: Path, payload: dict[str, Any], rig_sex: str) -> Path:
    raw = payload.get("glb_path")
    if not isinstance(raw, str) or not raw:
        raise ValueError("engine retarget result is missing glb_path")
    path = confined_child(candidate_dir, Path(raw))
    expected = (candidate_dir / "body_engine.glb").resolve(strict=False)
    if path != expected or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"engine GLB is missing or misplaced: {path}")
    if payload.get("rig_sex") != rig_sex:
        raise ValueError(f"engine retarget sex mismatch: {payload.get('rig_sex')!r}")
    if int(payload.get("bone_count") or 0) != 55:
        raise ValueError("engine retarget did not produce exactly 55 bones")
    if payload.get("rest_pose") != "arms-down":
        raise ValueError("engine retarget did not produce an arms-down rest pose")
    if int(payload.get("max_influences") or 0) not in range(1, 5):
        raise ValueError("engine retarget skin influences are outside [1,4]")
    if int(payload.get("vertex_count") or 0) < 3:
        raise ValueError("engine retarget produced no skinned vertices")
    if payload.get("source_side_mapping") != "swapped-after-z-flip":
        raise ValueError("engine retarget did not apply the required MPFB side remap")
    if int(payload.get("side_locality_pairs") or 0) != 23:
        raise ValueError("engine retarget did not validate all 23 left/right skin pairs")
    return path


def _create_engine_artifacts(
    candidate_dir: Path,
    blend_path: Path,
    rig_sex: str,
    blender_executable: Path,
    skeleton_path: Path,
) -> dict[str, Any]:
    skeleton_path = skeleton_path.resolve(strict=True)
    bake_executable = config.character_bake_exe().resolve(strict=True)
    request_path = candidate_dir / "engine_retarget.request.json"
    result_path = candidate_dir / "engine_retarget.result.json"
    glb_path = candidate_dir / "body_engine.glb"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_dir": str(candidate_dir),
                "skeleton_path": str(skeleton_path),
                "glb_path": str(glb_path),
                "rig_sex": rig_sex,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    try:
        retarget_code, _, retarget_stderr = _run_to_logs(
            _retarget_argv(blender_executable, blend_path, request_path, result_path),
            candidate_dir,
            config.timeout_s(),
            prefix="engine-retarget",
        )
    except BlenderProcessTimeout as exc:
        tail = _tail(candidate_dir / "engine-retarget.stderr.log")
        raise _timeout_error("engine retarget", exc, tail) from exc
    retarget_tail = _tail(retarget_stderr)
    if retarget_code != 0:
        raise BlenderBridgeError(
            f"engine retarget Blender exited with code {retarget_code}", tail=retarget_tail
        )
    if not result_path.is_file():
        raise BlenderBridgeError("engine retarget wrote no result JSON", tail=retarget_tail)
    retarget_payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(retarget_payload, dict) or retarget_payload.get("status") != "complete":
        error = retarget_payload.get("error") if isinstance(retarget_payload, dict) else None
        raise BlenderBridgeError(str(error or "engine retarget worker failed"), tail=retarget_tail)
    validated_glb = _validate_engine_retarget_result(candidate_dir, retarget_payload, rig_sex)

    baked_path = candidate_dir / "body_engine.baked.json"
    partial_path = candidate_dir / "body_engine.baked.json.tmp"
    bake_stderr = candidate_dir / "character-bake.stderr.log"
    try:
        bake_code = _run_to_files(
            _program_argv(
                bake_executable,
                str(validated_glb),
                "--cinema-ready",
                "--cinema-pack",
            ),
            candidate_dir,
            partial_path,
            bake_stderr,
            config.timeout_s(),
        )
    except BlenderProcessTimeout as exc:
        tail = _tail(bake_stderr)
        raise _timeout_error("character_bake_cli", exc, tail) from exc
    bake_tail = _tail(bake_stderr)
    if bake_code != 0:
        raise BlenderBridgeError(f"character_bake_cli exited with code {bake_code}", tail=bake_tail)
    if not partial_path.is_file() or partial_path.stat().st_size == 0:
        raise BlenderBridgeError("character_bake_cli wrote no baked JSON", tail=bake_tail)
    contract = validate_engine_contract(partial_path, skeleton_path, sex=rig_sex)
    partial_path.replace(baked_path)
    contract.update(
        {
            "retarget_vertex_count": int(retarget_payload["vertex_count"]),
            "retarget_max_influences": int(retarget_payload["max_influences"]),
        }
    )
    return {
        "engine_glb_path": str(validated_glb),
        "baked_character_path": str(baked_path),
        "rig_sex": rig_sex,
        "bone_count": 55,
        "rest_pose": "arms-down",
        "engine_contract": contract,
        "engine_stderr_tail": [*retarget_tail[-20:], *bake_tail[-20:]],
        "engine_warnings": list(retarget_payload.get("warnings") or []),
    }


def _record_failure(
    job_id: str,
    candidate_id: str,
    candidate_dir: Path,
    duration_s: float,
    error: str,
    tail: list[str] | None = None,
) -> None:
    failure = {
        "candidate_id": candidate_id,
        "status": "failed",
        "candidate_dir": str(candidate_dir),
        "duration_s": duration_s,
        "error": error,
        "stderr_tail": tail or [],
    }
    try:
        _atomic_json(candidate_dir / "result.json", failure)
    except Exception:
        # Logs and the job manifest can still preserve the primary error.
        pass
    try:
        record_candidate(job_id, failure)
    except Exception:
        # Preserve the primary Blender/validation error; the candidate directory
        # and logs still make the failed attempt inspectable.
        pass


def _create_candidate_locked(
    *,
    job_id: str,
    macro_parameters: dict[str, float] | None = None,
    target_parameters: dict[str, float] | None = None,
    render_views: list[str] | None = None,
    render_size: int = 512,
    candidate_label: str | None = None,
) -> CandidateResult:
    runtime = inspect_runtime()
    if not runtime.blender_exists or runtime.blender_executable is None:
        raise FileNotFoundError("Blender executable is unavailable; call inspect_bodymesh_runtime")
    mpfb = next(addon for addon in runtime.addons if addon.backend == "mpfb")
    if not mpfb.supported:
        raise RuntimeError("MPFB 2.0.x is required and was not detected")
    if not runtime.engine_contract_available:
        raise RuntimeError(
            "engine rig assets or character_bake_cli are unavailable; call inspect_bodymesh_runtime"
        )
    existing_job_dir, existing_manifest = load_job(job_id)
    rig_sex = str(existing_manifest.get("rig_sex") or "").lower()
    if rig_sex not in {"male", "female"}:
        raise ValueError("job has no valid rig_sex; prepare a new body-mesh job")
    raw_skeleton_path = existing_manifest.get("engine_skeleton_path")
    if not isinstance(raw_skeleton_path, str) or not raw_skeleton_path:
        raise ValueError("job has no engine skeleton snapshot; prepare a new body-mesh job")
    skeleton_snapshot = confined_child(existing_job_dir, Path(raw_skeleton_path))
    if not skeleton_snapshot.is_file():
        raise FileNotFoundError(f"job engine skeleton snapshot is missing: {skeleton_snapshot}")
    macros = _validate_macros(macro_parameters)
    targets = _validate_targets(target_parameters)
    views = list(dict.fromkeys(render_views or ["front", "side"]))
    unknown_views = [view for view in views if view not in _VIEW_NAMES]
    if unknown_views:
        raise ValueError(f"unknown render views {unknown_views}; allowed: {sorted(_VIEW_NAMES)}")
    if not 128 <= int(render_size) <= 2048:
        raise ValueError("render_size must be in [128, 2048]")

    candidate_dir, candidate_id, job_manifest = create_candidate_dir(job_id, candidate_label)
    request_path = candidate_dir / "request.json"
    mpfb_result_path = candidate_dir / "mpfb_result.json"
    request_payload = {
        "schema_version": 1,
        "backend": "mpfb",
        "job_id": job_id,
        "candidate_id": candidate_id,
        "candidate_dir": str(candidate_dir),
        "known_height_m": job_manifest.get("known_height_m"),
        "rig_sex": rig_sex,
        "macro_parameters": macros,
        "target_parameters": targets,
        "render_views": views,
        "render_size": int(render_size),
        "mpfb_version": _mpfb_version(),
    }
    start = time.monotonic()
    try:
        request_path.write_text(json.dumps(request_payload, indent=2, sort_keys=True), encoding="utf-8")
        executable = resolve_blender_executable()
        argv = _argv(executable, request_path, mpfb_result_path)
        return_code, _, stderr_path = _run_to_logs(argv, candidate_dir, config.timeout_s())
    except BlenderProcessTimeout as exc:
        duration = time.monotonic() - start
        tail = _tail(candidate_dir / "blender.stderr.log")
        if exc.tree_terminated:
            message = f"Blender exceeded {exc.timeout_s:g}s; process-tree termination was confirmed"
        else:
            message = (
                f"Blender exceeded {exc.timeout_s:g}s; the parent was killed but descendant "
                "termination could not be confirmed"
            )
        _record_failure(job_id, candidate_id, candidate_dir, duration, message, tail)
        raise BlenderBridgeError(message, tail=tail) from exc
    except OSError as exc:
        duration = time.monotonic() - start
        message = f"failed to launch Blender: {exc}"
        stderr_path = candidate_dir / "blender.stderr.log"
        try:
            stderr_path.write_text(message, encoding="utf-8")
        except OSError:
            pass
        _record_failure(job_id, candidate_id, candidate_dir, duration, message, [message])
        raise BlenderBridgeError(message, tail=[message]) from exc
    except Exception as exc:
        duration = time.monotonic() - start
        message = f"failed to prepare Blender candidate: {exc}"
        _record_failure(job_id, candidate_id, candidate_dir, duration, message)
        raise BlenderBridgeError(message) from exc

    tail = _tail(stderr_path)
    try:
        if return_code != 0:
            raise BlenderBridgeError(f"Blender exited with code {return_code}", tail=tail)
        if not mpfb_result_path.is_file():
            raise BlenderBridgeError("Blender exited successfully but wrote no mpfb_result.json", tail=tail)
        payload = json.loads(mpfb_result_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise BlenderBridgeError("Blender mpfb_result.json is not a JSON object", tail=tail)
        if payload.get("status") != "complete":
            raise BlenderBridgeError(str(payload.get("error") or "Blender worker failed"), tail=tail)
        _validate_result_paths(candidate_dir, payload, set(views), int(render_size))
        engine = _create_engine_artifacts(
            candidate_dir,
            Path(str(payload["blend_path"])),
            rig_sex,
            executable,
            skeleton_snapshot,
        )
        if engine["engine_contract"]["skeleton_sha256"] != job_manifest.get("engine_skeleton_sha256"):
            raise ValueError("candidate engine skeleton differs from the immutable job snapshot")
        duration = time.monotonic() - start
        combined_tail = [*tail[-20:], *engine.pop("engine_stderr_tail")]
        engine_warnings = engine.pop("engine_warnings")
        payload.update(
            {
                "job_id": job_id,
                "candidate_id": candidate_id,
                "candidate_dir": str(candidate_dir),
                "backend": "mpfb",
                "duration_s": duration,
                "macro_parameters": macros,
                "target_parameters": targets,
                "stderr_tail": combined_tail,
                "warnings": [*(payload.get("warnings") or []), *engine_warnings],
                "reference_mask_paths": _comparison_masks(
                    job_manifest,
                    candidate_dir,
                    int(render_size),
                    set((payload.get("render_paths") or {}).keys()),
                ),
                **engine,
            }
        )
        result = CandidateResult.model_validate(payload)
        _atomic_json(candidate_dir / "result.json", result.model_dump(mode="json"))
    except Exception as exc:
        duration = time.monotonic() - start
        failure_tail = exc.tail if isinstance(exc, BlenderBridgeError) else tail
        _record_failure(job_id, candidate_id, candidate_dir, duration, str(exc), failure_tail)
        if isinstance(exc, BlenderBridgeError):
            raise
        raise BlenderBridgeError(f"invalid Blender worker result: {exc}", tail=tail) from exc
    record_candidate(
        job_id,
        {
            "candidate_id": candidate_id,
            "status": result.status,
            "candidate_dir": result.candidate_dir,
            "blend_path": result.blend_path,
            "obj_path": result.obj_path,
            "ply_path": result.ply_path,
            "meta_path": result.meta_path,
            "engine_glb_path": result.engine_glb_path,
            "baked_character_path": result.baked_character_path,
            "rig_sex": result.rig_sex,
            "bone_count": result.bone_count,
            "rest_pose": result.rest_pose,
            "engine_contract": result.engine_contract,
            "render_paths": result.render_paths,
            "reference_mask_paths": result.reference_mask_paths,
            "vertex_count": result.vertex_count,
            "face_count": result.face_count,
            "duration_s": result.duration_s,
            "macro_parameters": result.macro_parameters,
            "target_parameters": result.target_parameters,
            "stderr_tail": result.stderr_tail,
            "warnings": result.warnings,
        },
    )
    return result


def create_candidate(
    *,
    job_id: str,
    macro_parameters: dict[str, float] | None = None,
    target_parameters: dict[str, float] | None = None,
    render_views: list[str] | None = None,
    render_size: int = 512,
    candidate_label: str | None = None,
) -> CandidateResult:
    if not _BLENDER_LOCK.acquire(blocking=False):
        raise BlenderBridgeError("another Blender body-mesh job is already running")
    try:
        try:
            with InterprocessLock(config.data_dir() / ".blender-worker.lock"):
                return _create_candidate_locked(
                    job_id=job_id,
                    macro_parameters=macro_parameters,
                    target_parameters=target_parameters,
                    render_views=render_views,
                    render_size=render_size,
                    candidate_label=candidate_label,
                )
        except LockUnavailable as exc:
            raise BlenderBridgeError("another Blender body-mesh job is already running") from exc
    finally:
        _BLENDER_LOCK.release()

"""Transactional host bridge for the versioned Blender identity-render preset."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat, UnidentifiedImageError

from bodymesh_server import blender_bridge, config
from bodymesh_server.environment import resolve_blender_executable
from bodymesh_server.jobs import load_job
from bodymesh_server.locking import InterprocessLock, LockUnavailable
from bodymesh_server.paths import confined_child, validate_id
from bodymesh_server.schemas import CandidateResult, IdentityRenderAsset, IdentityRenderSet

PRESET = "identity-v1"
SCHEMA_VERSION = 1
SAMPLES = 96
SEED = 20_260_715
CLOSEUP_SIZE = (768, 768)
BODY_SIZE = (768, 1024)
RENDERER = "CYCLES"
REQUESTED_DEVICE = "OPTIX"

CLOSEUP_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "asset_id": "face-00-neutral-key-left",
        "category": "best-identity-anchor",
        "view": "front",
        "expression": "neutral",
        "lighting": "key-left",
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
        "targets": {},
    },
    {
        "asset_id": "face-01-neutral-key-right",
        "category": "face-closeup",
        "view": "front",
        "expression": "neutral",
        "lighting": "key-right",
        "yaw_deg": 0.0,
        "pitch_deg": 1.0,
        "targets": {},
    },
    {
        "asset_id": "face-02-smile-soft-front",
        "category": "face-closeup",
        "view": "front",
        "expression": "soft-smile",
        "lighting": "soft-front",
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
        "targets": {"mouth-corner-puller": 0.32},
    },
    {
        "asset_id": "face-03-smile-left",
        "category": "face-closeup",
        "view": "near-front-left",
        "expression": "soft-smile",
        "lighting": "key-left",
        "yaw_deg": -6.0,
        "pitch_deg": -1.0,
        "targets": {"mouth-corner-puller": 0.32},
    },
    {
        "asset_id": "face-04-attentive-front",
        "category": "face-closeup",
        "view": "front",
        "expression": "attentive",
        "lighting": "key-right",
        "yaw_deg": 0.0,
        "pitch_deg": -1.0,
        "targets": {
            "eyebrows-left-inner-up": 0.18,
            "eyebrows-right-inner-up": 0.18,
        },
    },
    {
        "asset_id": "face-05-attentive-right",
        "category": "face-closeup",
        "view": "near-front-right",
        "expression": "attentive",
        "lighting": "key-left",
        "yaw_deg": 6.0,
        "pitch_deg": 1.0,
        "targets": {
            "eyebrows-left-inner-up": 0.18,
            "eyebrows-right-inner-up": 0.18,
        },
    },
    {
        "asset_id": "face-06-speaking-left",
        "category": "face-closeup",
        "view": "near-front-left",
        "expression": "speaking",
        "lighting": "soft-front",
        "yaw_deg": -4.0,
        "pitch_deg": 1.0,
        "targets": {"mouth-open": 0.16},
    },
    {
        "asset_id": "face-07-speaking-right",
        "category": "face-closeup",
        "view": "near-front-right",
        "expression": "speaking",
        "lighting": "key-right",
        "yaw_deg": 4.0,
        "pitch_deg": 0.0,
        "targets": {"mouth-open": 0.16, "mouth-corner-puller": 0.10},
    },
)

BODY_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "asset_id": "body-front",
        "category": "front-full-body",
        "view": "front",
        "expression": "neutral",
        "lighting": "studio",
    },
    {
        "asset_id": "body-three-quarter",
        "category": "three-quarter-full-body",
        "view": "three-quarter",
        "expression": "neutral",
        "lighting": "studio",
    },
    {
        "asset_id": "body-side",
        "category": "side-full-body",
        "view": "side",
        "expression": "neutral",
        "lighting": "studio",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_link(path: Path) -> bool:
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def _worker_script() -> Path:
    return Path(__file__).with_name("blender_scripts") / "identity_render_worker.py"


def _argv(executable: Path, blend_path: Path, request_path: Path, result_path: Path) -> list[str]:
    prefix = [sys.executable, str(executable)] if executable.suffix.lower() == ".py" else [str(executable)]
    return [
        *prefix,
        "--background",
        "--disable-autoexec",
        str(blend_path),
        "--python-exit-code",
        "1",
        "--python",
        str(_worker_script()),
        "--",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
    ]


def _resolve_candidate(job_id: str, candidate_id: str) -> tuple[Path, Path, dict[str, Any]]:
    job_id = validate_id(job_id, kind="job")
    candidate_id = validate_id(candidate_id, kind="candidate")
    job_dir, _ = load_job(job_id)
    candidate_dir = confined_child(job_dir, job_dir / "candidates" / candidate_id)
    result_path = candidate_dir / "result.json"
    if _is_link(result_path) or not result_path.is_file():
        raise FileNotFoundError(f"completed candidate result not found: {candidate_id}")
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"candidate result is unreadable: {result_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"candidate result is not an object: {result_path}")
    result = CandidateResult.model_validate(payload)
    if result.status != "complete":
        raise ValueError(f"candidate is not complete: {candidate_id}")
    if result.job_id != job_id:
        raise ValueError(f"candidate result job mismatch: expected {job_id}, got {result.job_id!r}")
    if result.candidate_id != candidate_id:
        raise ValueError(
            f"candidate result id mismatch: expected {candidate_id}, got {result.candidate_id!r}"
        )
    if Path(result.candidate_dir).resolve(strict=False) != candidate_dir:
        raise ValueError("candidate result directory does not match its immutable job location")
    raw_blend_path = Path(result.blend_path)
    if _is_link(raw_blend_path):
        raise ValueError("candidate character.blend may not be a symlink")
    blend_path = confined_child(candidate_dir, raw_blend_path)
    expected = (candidate_dir / "character.blend").resolve(strict=False)
    if blend_path != expected or not blend_path.is_file() or blend_path.stat().st_size == 0:
        raise ValueError(f"candidate character.blend is missing or misplaced: {blend_path}")
    return candidate_dir, blend_path, result.model_dump(mode="json")


def _relative_path(spec: dict[str, Any], kind: str) -> str:
    directory = "face" if kind == "closeup" else "body"
    return f"{directory}/{spec['asset_id']}.png"


def _request_payload(
    stage_dir: Path,
    job_id: str,
    candidate_id: str,
    recipe_sha256: str,
) -> dict[str, Any]:
    closeups = [
        {
            **spec,
            "kind": "closeup",
            "width": CLOSEUP_SIZE[0],
            "height": CLOSEUP_SIZE[1],
            "relative_path": _relative_path(spec, "closeup"),
        }
        for spec in CLOSEUP_VARIANTS
    ]
    bodies = [
        {
            **spec,
            "kind": "body",
            "width": BODY_SIZE[0],
            "height": BODY_SIZE[1],
            "relative_path": _relative_path(spec, "body"),
        }
        for spec in BODY_VARIANTS
    ]
    return {
        "operation": "identity-render",
        "schema_version": SCHEMA_VERSION,
        "preset": PRESET,
        "recipe_sha256": recipe_sha256,
        "job_id": job_id,
        "candidate_id": candidate_id,
        "output_dir": str(stage_dir),
        "renderer": RENDERER,
        "device": REQUESTED_DEVICE,
        "samples": SAMPLES,
        "seed": SEED,
        "closeups": closeups,
        "body_views": bodies,
    }


def _validate_png(path: Path, width: int, height: int) -> None:
    try:
        with Image.open(path) as opened:
            opened.verify()
        with Image.open(path) as opened:
            if opened.format != "PNG":
                raise ValueError(f"identity render is not PNG: {path}")
            if opened.size != (width, height):
                raise ValueError(
                    f"identity render has size {opened.size}; expected {(width, height)}: {path}"
                )
            extrema = ImageStat.Stat(opened.convert("RGB")).extrema
            if all(low == high for low, high in extrema):
                raise ValueError(f"identity render is blank: {path}")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"identity render is invalid: {path}: {exc}") from exc


def _expected_specs() -> tuple[dict[str, Any], ...]:
    return tuple(
        [
            {
                **spec,
                "kind": "closeup",
                "width": CLOSEUP_SIZE[0],
                "height": CLOSEUP_SIZE[1],
                "relative_path": _relative_path(spec, "closeup"),
            }
            for spec in CLOSEUP_VARIANTS
        ]
        + [
            {
                **spec,
                "kind": "body",
                "width": BODY_SIZE[0],
                "height": BODY_SIZE[1],
                "relative_path": _relative_path(spec, "body"),
            }
            for spec in BODY_VARIANTS
        ]
    )


def _recipe_sha256() -> str:
    worker_text = _worker_script().read_text(encoding="utf-8").replace("\r\n", "\n")
    contract = {
        "preset": PRESET,
        "schema_version": SCHEMA_VERSION,
        "renderer": RENDERER,
        "device": REQUESTED_DEVICE,
        "samples": SAMPLES,
        "seed": SEED,
        "renders": _expected_specs(),
    }
    digest = hashlib.sha256(worker_text.encode("utf-8"))
    digest.update(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _validate_worker_result(
    stage_dir: Path,
    payload: dict[str, Any],
    recipe_sha256: str,
) -> list[tuple[dict[str, Any], Path]]:
    if payload.get("status") != "complete":
        raise ValueError(str(payload.get("error") or "identity worker failed"))
    if payload.get("preset") != PRESET or payload.get("renderer") != RENDERER:
        raise ValueError("identity worker returned the wrong preset or renderer")
    if payload.get("recipe_sha256") != recipe_sha256:
        raise ValueError("identity worker returned the wrong recipe hash")
    for field in ("blender_version", "mpfb_version", "device_used"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError(f"identity worker returned no {field}")
    renders = payload.get("renders")
    if not isinstance(renders, list):
        raise ValueError("identity worker renders must be an array")
    expected = _expected_specs()
    if len(renders) != len(expected):
        raise ValueError(f"identity worker returned {len(renders)} renders; expected {len(expected)}")
    validated: list[tuple[dict[str, Any], Path]] = []
    for raw, spec in zip(renders, expected, strict=True):
        if not isinstance(raw, dict):
            raise ValueError("identity worker render entry is not an object")
        for field in ("asset_id", "kind", "category", "view", "expression", "lighting"):
            if raw.get(field) != spec[field]:
                raise ValueError(
                    f"identity worker {field} mismatch for {spec['asset_id']}: {raw.get(field)!r}"
                )
        raw_path = raw.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"identity worker path missing for {spec['asset_id']}")
        path = confined_child(stage_dir, Path(raw_path))
        expected_path = (stage_dir / spec["relative_path"]).resolve(strict=False)
        if path != expected_path or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"identity render is missing or misplaced: {path}")
        _validate_png(path, int(spec["width"]), int(spec["height"]))
        validated.append((spec, path))
    return validated


def _asset_for_final(spec: dict[str, Any], stage_path: Path, final_dir: Path) -> IdentityRenderAsset:
    return IdentityRenderAsset(
        asset_id=spec["asset_id"],
        kind=spec["kind"],
        category=spec["category"],
        view=spec["view"],
        expression=spec["expression"],
        lighting=spec["lighting"],
        path=str((final_dir / spec["relative_path"]).resolve(strict=False)),
        relative_path=spec["relative_path"],
        width=int(spec["width"]),
        height=int(spec["height"]),
        sha256=_sha256(stage_path),
    )


def _validate_cached(
    manifest_path: Path,
    *,
    job_id: str,
    candidate_id: str,
    blend_path: Path,
    blend_sha256: str,
    recipe_sha256: str,
) -> IdentityRenderSet | None:
    if not manifest_path.is_file():
        return None
    try:
        result = IdentityRenderSet.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        expected_dir = manifest_path.parent.resolve(strict=False)
        if (
            result.job_id != job_id
            or result.candidate_id != candidate_id
            or Path(result.source_blend_path).resolve(strict=False) != blend_path
            or result.source_blend_sha256 != blend_sha256
            or result.recipe_sha256 != recipe_sha256
            or Path(result.output_dir).resolve(strict=False) != expected_dir
            or Path(result.manifest_path).resolve(strict=False) != manifest_path.resolve(strict=False)
            or result.renderer != RENDERER
            or result.device_requested != REQUESTED_DEVICE
            or result.samples != SAMPLES
            or result.seed != SEED
            or not result.blender_version.strip()
            or not result.mpfb_version.strip()
            or not result.device_used.strip()
        ):
            return None
        expected = _expected_specs()
        if len(result.renders) != len(expected):
            return None
        for asset, spec in zip(result.renders, expected, strict=True):
            path = confined_child(expected_dir, Path(asset.path))
            if (
                asset.asset_id != spec["asset_id"]
                or asset.kind != spec["kind"]
                or asset.category != spec["category"]
                or asset.view != spec["view"]
                or asset.expression != spec["expression"]
                or asset.lighting != spec["lighting"]
                or asset.relative_path != spec["relative_path"]
                or asset.width != spec["width"]
                or asset.height != spec["height"]
                or path != (expected_dir / spec["relative_path"]).resolve(strict=False)
                or not path.is_file()
            ):
                return None
            _validate_png(path, int(spec["width"]), int(spec["height"]))
            if _sha256(path) != asset.sha256:
                return None
        return result
    except (OSError, ValueError):
        return None


def _remove_path(path: Path) -> None:
    if _is_link(path):
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _promotion_backups(final_dir: Path) -> list[Path]:
    return sorted(final_dir.parent.glob(f".{final_dir.name}.*.backup"))


def _recover_promotion(final_dir: Path) -> None:
    backups = [confined_child(final_dir.parent, path) for path in _promotion_backups(final_dir)]
    if not backups:
        return
    if final_dir.exists():
        if not (final_dir / "manifest.json").is_file():
            if len(backups) != 1:
                raise RuntimeError("identity promotion recovery found ambiguous backups")
            failed = final_dir.with_name(f".{final_dir.name}.{uuid.uuid4().hex}.failed")
            final_dir.replace(failed)
            backups[0].replace(final_dir)
            _remove_path(failed)
            return
        for backup in backups:
            _remove_path(backup)
        return
    if len(backups) != 1:
        raise RuntimeError("identity promotion recovery found ambiguous backups")
    backups[0].replace(final_dir)


def _promote(stage_dir: Path, final_dir: Path) -> None:
    backup = confined_child(
        final_dir.parent,
        final_dir.with_name(f".{final_dir.name}.{uuid.uuid4().hex}.backup"),
    )
    moved_existing = False
    promoted = False
    try:
        if final_dir.exists():
            final_dir.replace(backup)
            moved_existing = True
        stage_dir.replace(final_dir)
        promoted = True
    except Exception:
        if moved_existing and backup.exists():
            if final_dir.exists():
                failed = final_dir.with_name(f".{final_dir.name}.{uuid.uuid4().hex}.failed")
                final_dir.replace(failed)
                backup.replace(final_dir)
                _remove_path(failed)
            else:
                backup.replace(final_dir)
        raise
    finally:
        if promoted and backup.exists():
            _remove_path(backup)


def _render_identity_set_locked(job_id: str, candidate_id: str, *, force: bool) -> IdentityRenderSet:
    candidate_dir, blend_path, _ = _resolve_candidate(job_id, candidate_id)
    source_before = _sha256(blend_path)
    recipe_sha256 = _recipe_sha256()
    raw_identity_root = candidate_dir / "identity"
    if _is_link(raw_identity_root):
        raise ValueError("candidate identity directory may not be a symlink or junction")
    identity_root = confined_child(candidate_dir, raw_identity_root)
    identity_root.mkdir(exist_ok=True)
    identity_root = confined_child(candidate_dir, identity_root)
    raw_final_dir = identity_root / PRESET
    if _is_link(raw_final_dir):
        raise ValueError("identity preset directory may not be a symlink or junction")
    final_dir = confined_child(candidate_dir, raw_final_dir)
    _recover_promotion(final_dir)
    raw_manifest_path = final_dir / "manifest.json"
    if _is_link(raw_manifest_path):
        raise ValueError("identity manifest may not be a symlink")
    manifest_path = confined_child(candidate_dir, raw_manifest_path)
    if not force:
        cached = _validate_cached(
            manifest_path,
            job_id=job_id,
            candidate_id=candidate_id,
            blend_path=blend_path,
            blend_sha256=source_before,
            recipe_sha256=recipe_sha256,
        )
        if cached is not None:
            return cached

    stage_dir = confined_child(
        candidate_dir,
        identity_root / f".{PRESET}.{uuid.uuid4().hex}.tmp",
    )
    stage_dir.mkdir(exist_ok=False)
    stage_dir = confined_child(candidate_dir, stage_dir)
    request_path = confined_child(candidate_dir, stage_dir / "request.json")
    worker_result_path = confined_child(candidate_dir, stage_dir / "worker-result.json")
    request = _request_payload(stage_dir, job_id, candidate_id, recipe_sha256)
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
    start = time.monotonic()
    try:
        executable = resolve_blender_executable()
        argv = _argv(executable, blend_path, request_path, worker_result_path)
        try:
            return_code, _, stderr_path = blender_bridge._run_to_logs(
                argv, stage_dir, config.timeout_s(), prefix="blender"
            )
        except blender_bridge.BlenderProcessTimeout as exc:
            tail = blender_bridge._tail(stage_dir / "blender.stderr.log")
            raise blender_bridge._timeout_error("identity render", exc, tail) from exc
        tail = blender_bridge._tail(stderr_path)
        if return_code != 0:
            raise blender_bridge.BlenderBridgeError(
                f"identity Blender exited with code {return_code}", tail=tail
            )
        if not worker_result_path.is_file():
            raise blender_bridge.BlenderBridgeError(
                "identity Blender exited successfully but wrote no worker-result.json", tail=tail
            )
        try:
            worker_payload = json.loads(worker_result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise blender_bridge.BlenderBridgeError(
                "identity worker-result.json is invalid JSON", tail=tail
            ) from exc
        if not isinstance(worker_payload, dict):
            raise blender_bridge.BlenderBridgeError("identity worker-result.json is not an object", tail=tail)
        validated = _validate_worker_result(stage_dir, worker_payload, recipe_sha256)
        source_after = _sha256(blend_path)
        if source_after != source_before:
            raise blender_bridge.BlenderBridgeError(
                "candidate character.blend changed during identity rendering", tail=tail
            )
        assets = [_asset_for_final(spec, path, final_dir) for spec, path in validated]
        duration = time.monotonic() - start
        result = IdentityRenderSet(
            job_id=job_id,
            candidate_id=candidate_id,
            source_blend_path=str(blend_path),
            source_blend_sha256=source_before,
            recipe_sha256=recipe_sha256,
            output_dir=str(final_dir.resolve(strict=False)),
            manifest_path=str(manifest_path.resolve(strict=False)),
            renderer=RENDERER,
            blender_version=str(worker_payload["blender_version"]),
            mpfb_version=str(worker_payload["mpfb_version"]),
            device_requested=REQUESTED_DEVICE,
            device_used=str(worker_payload["device_used"]),
            samples=SAMPLES,
            seed=SEED,
            material_backend=str(worker_payload.get("material_backend") or "unknown"),
            expression_backend=str(worker_payload.get("expression_backend") or "unknown"),
            duration_s=duration,
            renders=assets,
            warnings=[str(value) for value in worker_payload.get("warnings") or []],
        )
        (stage_dir / "manifest.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8", newline="\n"
        )
        _promote(stage_dir, final_dir)
        return result
    except blender_bridge.BlenderBridgeError:
        raise
    except (OSError, ValueError) as exc:
        tail = blender_bridge._tail(stage_dir / "blender.stderr.log")
        raise blender_bridge.BlenderBridgeError(f"invalid identity-render result: {exc}", tail=tail) from exc
    finally:
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)


def render_identity_set(job_id: str, candidate_id: str, *, force: bool = False) -> IdentityRenderSet:
    """Render or reuse the fixed identity-v1 set for one completed body-mesh candidate."""
    if not isinstance(force, bool):
        raise ValueError("force must be a boolean")
    if not blender_bridge._BLENDER_LOCK.acquire(blocking=False):
        raise blender_bridge.BlenderBridgeError("another Blender body-mesh job is already running")
    try:
        try:
            with InterprocessLock(config.data_dir() / ".blender-worker.lock"):
                return _render_identity_set_locked(job_id, candidate_id, force=force)
        except LockUnavailable as exc:
            raise blender_bridge.BlenderBridgeError(
                "another Blender body-mesh job is already running"
            ) from exc
    finally:
        blender_bridge._BLENDER_LOCK.release()


__all__ = [
    "BODY_VARIANTS",
    "CLOSEUP_VARIANTS",
    "PRESET",
    "render_identity_set",
]

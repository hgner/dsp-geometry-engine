"""Immutable body-mesh job and candidate manifests."""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from bodymesh_server import config
from bodymesh_server.paths import confined_child, normalize_image, validate_id, validate_input_image
from bodymesh_server.schemas import PreparedJob, ReferenceInfo

_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _slug(value: str | None, fallback: str) -> str:
    text = _SAFE_RE.sub("-", (value or fallback).strip()).strip("-_")
    return (text or fallback)[:40]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def job_dir(job_id: str) -> Path:
    validate_id(job_id, kind="job")
    return confined_child(config.jobs_dir(), config.jobs_dir() / job_id)


def load_job(job_id: str) -> tuple[Path, dict[str, Any]]:
    directory = job_dir(job_id)
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"body-mesh job not found: {job_id}")
    return directory, read_json(manifest)


def prepare_job(
    *,
    front_image: str,
    side_image: str | None = None,
    back_image: str | None = None,
    front_mask: str | None = None,
    side_mask: str | None = None,
    back_mask: str | None = None,
    known_height_m: float | None = None,
    label: str | None = None,
) -> PreparedJob:
    if known_height_m is not None and not 0.4 <= float(known_height_m) <= 2.5:
        raise ValueError("known_height_m must be in [0.4, 2.5]")
    config.ensure_data_dirs()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = f"{timestamp}-{_slug(label, 'body')}-{secrets.token_hex(3)}"
    directory = job_dir(job_id)
    references_dir = directory / "references"
    references_dir.mkdir(parents=True, exist_ok=False)

    inputs = {
        "front": (front_image, front_mask),
        "side": (side_image, side_mask),
        "back": (back_image, back_mask),
    }
    references: list[ReferenceInfo] = []
    try:
        for view, (image_raw, mask_raw) in inputs.items():
            if image_raw is None:
                if mask_raw is not None:
                    raise ValueError(f"{view}_mask requires {view}_image")
                continue
            source = validate_input_image(image_raw)
            copied = references_dir / f"{view}.png"
            width, height, digest = normalize_image(source, copied)
            mask_path: str | None = None
            mask_width: int | None = None
            mask_height: int | None = None
            mask_digest: str | None = None
            if mask_raw is not None:
                mask_source = validate_input_image(mask_raw)
                mask_destination = references_dir / f"{view}_mask.png"
                mask_width, mask_height, mask_digest = normalize_image(
                    mask_source, mask_destination, mask=True
                )
                if (mask_width, mask_height) != (width, height):
                    raise ValueError(
                        f"{view} mask dimensions {(mask_width, mask_height)} do not match "
                        f"image {(width, height)}"
                    )
                mask_path = str(mask_destination)
            references.append(
                ReferenceInfo(
                    view=view,
                    source_path=str(source),
                    copied_path=str(copied),
                    mask_path=mask_path,
                    mask_width=mask_width,
                    mask_height=mask_height,
                    mask_sha256=mask_digest,
                    width=width,
                    height=height,
                    sha256=digest,
                )
            )
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise

    warnings: list[str] = []
    if len(references) == 1:
        warnings.append(
            "single-view reconstruction is underdetermined; add a side reference for useful fitting"
        )
    if any(ref.mask_path is None for ref in references):
        warnings.append("provide DSP segment_image masks for quantitative silhouette comparison")
    payload = {
        "schema_version": 1,
        "job_id": job_id,
        "label": label,
        "status": "prepared",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "known_height_m": float(known_height_m) if known_height_m is not None else None,
        "references": [reference.model_dump() for reference in references],
        "candidates": [],
        "warnings": warnings,
    }
    manifest = directory / "manifest.json"
    try:
        _atomic_json(manifest, payload)
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return PreparedJob(
        job_id=job_id,
        status="prepared",
        job_dir=str(directory),
        manifest_path=str(manifest),
        known_height_m=payload["known_height_m"],
        references=references,
        warnings=warnings,
    )


def create_candidate_dir(job_id: str, label: str | None) -> tuple[Path, str, dict[str, Any]]:
    directory, manifest = load_job(job_id)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate_id = f"{timestamp}-{_slug(label, 'candidate')}-{secrets.token_hex(3)}"
    candidate_dir = confined_child(directory, directory / "candidates" / candidate_id)
    candidate_dir.mkdir(parents=True, exist_ok=False)
    return candidate_dir, candidate_id, manifest


def record_candidate(job_id: str, summary: dict[str, Any]) -> None:
    directory, manifest = load_job(job_id)
    candidates = manifest.setdefault("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("job candidate manifest is corrupt")
    candidates.append(summary)
    candidate_statuses = {candidate.get("status") for candidate in candidates if isinstance(candidate, dict)}
    has_complete = "complete" in candidate_statuses
    has_failed = "failed" in candidate_statuses
    if has_complete and has_failed:
        manifest["status"] = "partial"
    elif has_complete:
        manifest["status"] = "generated"
    else:
        manifest["status"] = "failed"
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _atomic_json(directory / "manifest.json", manifest)

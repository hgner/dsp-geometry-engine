"""Body-mesh MCP tests use a subprocess stub; real Blender stays local-only."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from bodymesh_server import blender_bridge, jobs
from bodymesh_server.environment import inspect_runtime
from bodymesh_server.locking import InterprocessLock
from bodymesh_server.server import create_server
from bodymesh_server.tools import (
    _create_body_mesh,
    _get_bodymesh_job,
    _inspect_bodymesh_runtime,
    _list_bodymesh_parameters,
    _prepare_bodymesh_job,
)
from dsp_server.engine.ply import load_meta, read_engine_ply


@pytest.fixture
def bodymesh_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    data = tmp_path / "data" / "bodymesh"
    inbox = data / "inbox"
    inbox.mkdir(parents=True)
    mpfb = tmp_path / "mpfb"
    targets = mpfb / "data" / "targets" / "torso"
    targets.mkdir(parents=True)
    (mpfb / "blender_manifest.toml").write_text(
        'schema_version = "1.0.0"\nid = "mpfb"\nversion = "2.0.16"\n', encoding="utf-8"
    )
    (targets / "waist-narrow.target").write_text("", encoding="utf-8")
    stub = Path(__file__).with_name("stub_blender.py")
    monkeypatch.setenv("BODYMESH_DATA_DIR", str(data))
    monkeypatch.setenv("BODYMESH_INPUT_ROOTS", str(inbox))
    monkeypatch.setenv("BODYMESH_MPFB_ROOT", str(mpfb))
    monkeypatch.setenv("BODYMESH_MBLAB_ROOT", str(tmp_path / "missing-mblab"))
    monkeypatch.setenv("BODYMESH_BLENDER_EXE", str(stub))
    monkeypatch.setenv("BODYMESH_TIMEOUT_S", "30")
    return {"data": data, "inbox": inbox, "mpfb": mpfb, "stub": stub}


def _tool_names() -> set[str]:
    return {tool.name for tool in asyncio.run(create_server().list_tools())}


def _make_image(path: Path, size: tuple[int, int] = (128, 256)) -> Path:
    Image.new("RGB", size, "white").save(path)
    return path


def test_separate_server_registers_expected_tools(bodymesh_env):
    assert _tool_names() == {
        "inspect_bodymesh_runtime",
        "list_bodymesh_parameters",
        "prepare_bodymesh_job",
        "create_body_mesh",
        "get_bodymesh_job",
    }


def test_runtime_and_parameter_catalog(bodymesh_env):
    runtime = inspect_runtime()
    assert runtime.blender_exists is True
    assert runtime.supported_backend == "mpfb"
    mpfb = next(addon for addon in runtime.addons if addon.backend == "mpfb")
    assert mpfb.version == "2.0.16"
    assert mpfb.supported is True
    report = json.loads(_list_bodymesh_parameters("waist"))
    assert report["target_matches"] == ["torso/waist-narrow.target"]
    assert json.loads(_list_bodymesh_parameters("mpfb"))["target_matches"] == []
    assert "gender" in report["macro_parameters"]
    assert "error" not in json.loads(_inspect_bodymesh_runtime())


def test_prepare_job_copies_and_hashes_references(bodymesh_env):
    front = _make_image(bodymesh_env["inbox"] / "front.jpg")
    side = _make_image(bodymesh_env["inbox"] / "side.png")
    mask = _make_image(bodymesh_env["inbox"] / "front-mask.png")
    report = jobs.prepare_job(
        front_image=str(front),
        side_image=str(side),
        front_mask=str(mask),
        known_height_m=1.82,
        label="reference person",
    )
    assert report.known_height_m == 1.82
    assert len(report.references) == 2
    assert Path(report.references[0].copied_path).is_file()
    assert Path(report.references[0].mask_path or "").is_file()
    assert report.references[0].mask_sha256
    assert report.references[0].mask_width == 128
    assert report.references[0].mask_height == 256
    manifest = json.loads(Path(report.manifest_path).read_text(encoding="utf-8"))
    assert manifest["status"] == "prepared"
    assert manifest["candidates"] == []


def test_prepare_rejects_path_outside_input_roots(bodymesh_env, tmp_path: Path):
    outside = _make_image(tmp_path / "outside.png")
    payload = json.loads(_prepare_bodymesh_job(str(outside)))
    assert "error" in payload
    assert "BODYMESH_INPUT_ROOTS" in payload["error"]


def test_prepare_failure_removes_private_partial_job(bodymesh_env):
    front = _make_image(bodymesh_env["inbox"] / "front.png")
    mismatched_mask = _make_image(bodymesh_env["inbox"] / "mask.png", (64, 64))
    jobs_root = bodymesh_env["data"] / "jobs"

    with pytest.raises(ValueError, match="mask dimensions"):
        jobs.prepare_job(front_image=str(front), front_mask=str(mismatched_mask))

    assert list(jobs_root.iterdir()) == []


def test_stub_blender_candidate_and_dsp_ply_handoff(bodymesh_env):
    front = _make_image(bodymesh_env["inbox"] / "front.png")
    front_mask = _make_image(bodymesh_env["inbox"] / "front-mask.png")
    prepared = jobs.prepare_job(front_image=str(front), front_mask=str(front_mask), known_height_m=1.75)
    result = blender_bridge.create_candidate(
        job_id=prepared.job_id,
        macro_parameters={"gender": 0.9, "muscle": 0.7},
        target_parameters={"torso/waist-narrow.target": 0.4},
        render_views=["front", "side"],
        render_size=256,
        candidate_label="first",
    )
    assert result.status == "complete"
    assert result.vertex_count == 4
    assert set(result.render_paths) == {"front", "side"}
    assert set(result.reference_mask_paths) == {"front"}
    assert Image.open(result.reference_mask_paths["front"]).size == (256, 256)
    dump = read_engine_ply(result.ply_path or "")
    assert dump.vertex_count == 4
    assert dump.face_count == 4
    meta = load_meta(result.ply_path or "")
    assert meta is not None and meta.bone_map == {0: "body"}
    status = json.loads(_get_bodymesh_job(prepared.job_id))
    assert status["status"] == "generated"
    assert status["candidates"][0]["candidate_id"] == result.candidate_id
    assert status["candidates"][0]["macro_parameters"] == {"gender": 0.9, "muscle": 0.7}
    assert status["candidates"][0]["target_parameters"] == {"torso/waist-narrow.target": 0.4}
    assert status["candidates"][0]["vertex_count"] == 4


def test_cross_process_lock_rejects_without_creating_candidate(bodymesh_env):
    front = _make_image(bodymesh_env["inbox"] / "front.png")
    prepared = jobs.prepare_job(front_image=str(front))
    candidates_root = Path(prepared.job_dir) / "candidates"

    with InterprocessLock(bodymesh_env["data"] / ".blender-worker.lock"):
        with pytest.raises(blender_bridge.BlenderBridgeError, match="already running"):
            blender_bridge.create_candidate(job_id=prepared.job_id)

    assert not candidates_root.exists()


def test_worker_contract_rejects_missing_artifact_and_corrupt_render(bodymesh_env):
    front = _make_image(bodymesh_env["inbox"] / "front.png")
    prepared = jobs.prepare_job(front_image=str(front))
    result = blender_bridge.create_candidate(job_id=prepared.job_id, render_views=["front"])
    Path(result.obj_path).unlink()
    payload = json.loads((Path(result.candidate_dir) / "result.json").read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="obj_path.*missing or empty"):
        blender_bridge._validate_result_paths(Path(result.candidate_dir), payload, {"front"}, 512)

    Path(result.obj_path).write_text("v 0 0 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="has size.*expected"):
        blender_bridge._validate_result_paths(Path(result.candidate_dir), payload, {"front"}, 256)

    Path(result.render_paths["front"]).write_bytes(b"not a PNG")
    with pytest.raises(ValueError, match="not a valid PNG"):
        blender_bridge._validate_result_paths(Path(result.candidate_dir), payload, {"front"}, 512)


def test_job_status_is_partial_for_mixed_candidate_outcomes(bodymesh_env):
    front = _make_image(bodymesh_env["inbox"] / "front.png")
    prepared = jobs.prepare_job(front_image=str(front))
    jobs.record_candidate(prepared.job_id, {"candidate_id": "good", "status": "complete"})
    jobs.record_candidate(prepared.job_id, {"candidate_id": "bad", "status": "failed"})

    _, manifest = jobs.load_job(prepared.job_id)

    assert manifest["status"] == "partial"


def test_blender_child_environment_does_not_forward_secrets(bodymesh_env, monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "do-not-forward")
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-forward")
    monkeypatch.setenv("PROGRAMDATA", "C:/ProgramData")

    child = blender_bridge._child_env()

    assert "AWS_SECRET_ACCESS_KEY" not in child
    assert "OPENAI_API_KEY" not in child
    assert child["PROGRAMDATA"] == "C:/ProgramData"


def test_subprocess_timeout_is_bounded_and_reports_tree_cleanup(bodymesh_env, tmp_path):
    with pytest.raises(blender_bridge.BlenderProcessTimeout) as caught:
        blender_bridge._run_to_logs([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, 0.1)

    assert isinstance(caught.value.tree_terminated, bool)
    assert (tmp_path / "blender.stdout.log").is_file()
    assert (tmp_path / "blender.stderr.log").is_file()


def test_tool_error_for_unknown_macro(bodymesh_env):
    front = _make_image(bodymesh_env["inbox"] / "front.png")
    prepared = jobs.prepare_job(front_image=str(front))
    payload = json.loads(_create_body_mesh(prepared.job_id, macro_parameters={"unknown": 0.5}))
    assert "error" in payload
    assert "unknown macro" in payload["error"]

"""M2 lbs.py verification: a hand-computable 2-bone rig (root at origin, child at
(0,1,0), identity rotations, unit scale) written as a real baked-character JSON,
plus a hand-built palette ("child rotated 90 deg about Z at its own head") whose
vertex images are known exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dsp_server.engine.lbs import (
    align_json_to_engine,
    lbs_pose,
    load_baked_character,
    load_palette_sidecar,
    recover_import_scale,
    validate_against_inverse_bind,
    weight_surgery,
)

CHILD_HEAD = np.array([0.0, 1.0, 0.0])


def _colmajor(m: np.ndarray) -> list[float]:
    """(4,4) math matrix -> the engine's flat COLUMN-major 16-float list."""
    return np.asarray(m, dtype=np.float64).flatten(order="F").tolist()


def _translation(t) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = t
    return m


def _rz90() -> np.ndarray:
    m = np.eye(4)
    m[:2, :2] = [[0.0, -1.0], [1.0, 0.0]]
    return m


def _vertex(position, joints, weights) -> dict:
    return {
        "position": list(position),
        "normal": [0.0, 0.0, 1.0],
        "texCoord": [0.0, 0.0],
        "tangent": [1.0, 0.0, 0.0, 1.0],
        "jointIndices": list(joints),
        "jointWeights": list(weights),
    }


def _two_bone_doc() -> dict:
    ident = {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0, 1.0], "s": [1.0, 1.0, 1.0]}
    child_bind = {"t": CHILD_HEAD.tolist(), "r": [0.0, 0.0, 0.0, 1.0], "s": [1.0, 1.0, 1.0]}
    # World bind: root = I, child = T(0,1,0)  =>  inverseBind: root = I, child = T(0,-1,0).
    inv_child = _translation(-CHILD_HEAD)
    return {
        "version": 1,
        "mesh": {
            "vertices": [
                # v0: fully weighted to the child bone
                _vertex((0.5, 1.5, 0.0), (1, 0, 0, 0), (1.0, 0.0, 0.0, 0.0)),
                # v1: 0.5 root / 0.5 child
                _vertex((0.5, 1.5, 0.0), (0, 1, 0, 0), (0.5, 0.5, 0.0, 0.0)),
            ],
            "indices": [0, 1, 0],
        },
        "skeleton": {
            "skeletonId": "test-2bone",
            "kind": "FullBody",
            "bones": [
                {"name": "root", "parentIndex": -1, "role": "root", "bind": ident},
                {"name": "child", "parentIndex": 0, "role": "arm", "bind": child_bind},
            ],
            "sockets": [],
            "socketBindings": [],
        },
        "inverseBind": [_colmajor(np.eye(4)), _colmajor(inv_child)],
    }


def _write_doc(tmp_path: Path, doc: dict, name: str) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


@pytest.fixture
def rig(tmp_path):
    return load_baked_character(_write_doc(tmp_path, _two_bone_doc(), "two_bone.baked.json"))


def _child_rz90_palette() -> np.ndarray:
    """Palette for: root at bind pose, child rotated 90 deg about Z at its own head.

    palette[child] = worldPose_child @ inverseBind_child
                   = T(0,1,0) @ Rz(90) @ T(0,-1,0)
    """
    child = _translation(CHILD_HEAD) @ _rz90() @ _translation(-CHILD_HEAD)
    return np.stack([np.eye(4), child])


def test_load_shapes_and_inverse_bind_convention(rig):
    assert rig.positions.shape == (2, 3) and rig.positions.dtype == np.float32
    assert rig.joint_indices.shape == (2, 4) and rig.joint_indices.dtype == np.int32
    assert rig.joint_weights.shape == (2, 4) and rig.joint_weights.dtype == np.float32
    assert rig.indices.tolist() == [0, 1, 0]
    assert rig.bone_names == ["root", "child"]
    assert rig.parent_index.tolist() == [-1, 0]
    assert rig.bind_local.shape == (2, 4, 4) and rig.inverse_bind.shape == (2, 4, 4)
    # The convention self-check: FK world bind inverted must equal the asset's inverseBind.
    assert validate_against_inverse_bind(rig) < 1e-6


def test_lbs_pose_matches_hand_computed_positions(rig):
    palette = _child_rz90_palette()
    posed = lbs_pose(rig.positions, rig.joint_indices, rig.joint_weights, palette)
    assert posed.shape == (2, 3) and posed.dtype == np.float32

    # v0 (all weight on child): child-local (0.5, 0.5, 0) -> Rz90 -> (-0.5, 0.5, 0)
    # -> back to world at the child head -> (-0.5, 1.5, 0).
    np.testing.assert_allclose(posed[0], [-0.5, 1.5, 0.0], atol=1e-5)

    # v1 (0.5/0.5): midpoint of the root-transformed image (identity -> (0.5, 1.5, 0))
    # and the child-transformed image (-0.5, 1.5, 0)  =>  (0.0, 1.5, 0).
    root_img = rig.positions[1]
    child_img = (palette[1, :3, :3] @ rig.positions[1]) + palette[1, :3, 3]
    np.testing.assert_allclose(posed[1], 0.5 * root_img + 0.5 * child_img, atol=1e-5)
    np.testing.assert_allclose(posed[1], [0.0, 1.5, 0.0], atol=1e-5)


def test_out_of_range_joint_index_raises(rig):
    palette = _child_rz90_palette()
    bad_ji = rig.joint_indices.copy()
    bad_ji[0, 0] = 7  # nonzero-weight influence outside the 2-bone palette
    with pytest.raises(IndexError, match="out of range"):
        lbs_pose(rig.positions, bad_ji, rig.joint_weights, palette)
    # Engine-faithful: a ZERO-weight slot never dereferences its index.
    lazy_ji = rig.joint_indices.copy()
    lazy_ji[0, 3] = 99
    posed = lbs_pose(rig.positions, lazy_ji, rig.joint_weights, palette)
    np.testing.assert_allclose(posed[0], [-0.5, 1.5, 0.0], atol=1e-5)


def test_weight_surgery_top1_is_onehot():
    ji = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int32)
    jw = np.array([[0.6, 0.3, 0.1, 0.0], [0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    out_ji, out_jw = weight_surgery(ji, jw, "top1", ["a", "b", "c", "d"])
    np.testing.assert_array_equal(out_ji, ji)
    np.testing.assert_array_equal(out_jw, [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    assert out_jw.dtype == np.float32
    np.testing.assert_array_equal(jw[0], np.array([0.6, 0.3, 0.1, 0.0], dtype=np.float32))  # inputs untouched


def test_weight_surgery_drop_hand_renormalizes():
    names = ["root", "armLowerL", "handL", "thumb1L"]
    ji = np.array([[1, 2, 0, 0], [2, 2, 2, 2], [1, 3, 0, 0]], dtype=np.int32)
    jw = np.array(
        [[0.5, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.75, 0.25, 0.0, 0.0]],
        dtype=np.float32,
    )
    _, hand = weight_surgery(ji, jw, "drop_hand", names)
    np.testing.assert_allclose(hand[0], [1.0, 0.0, 0.0, 0.0], atol=1e-6)  # handL zeroed, renorm
    np.testing.assert_allclose(hand[1], jw[1], atol=1e-6)  # all-dropped row keeps original
    _, fingers = weight_surgery(ji, jw, "drop_fingers", names)
    np.testing.assert_allclose(fingers[2], [1.0, 0.0, 0.0, 0.0], atol=1e-6)  # thumb1L zeroed
    with pytest.raises(ValueError, match="unknown weight-surgery mode"):
        weight_surgery(ji, jw, "drop_everything", names)


def test_import_scale_align_and_recovery(rig):
    aligned = align_json_to_engine(rig, 0.01)
    np.testing.assert_allclose(aligned.positions, rig.positions * 0.01, atol=1e-9)
    np.testing.assert_allclose(aligned.bind_local[1, :3, 3], CHILD_HEAD * 0.01, atol=1e-12)
    np.testing.assert_allclose(aligned.inverse_bind[1, :3, 3], -CHILD_HEAD * 0.01, atol=1e-12)
    # Rotations untouched and the convention still self-consistent after scaling.
    np.testing.assert_allclose(aligned.bind_local[:, :3, :3], rig.bind_local[:, :3, :3])
    assert validate_against_inverse_bind(aligned) < 1e-6
    assert abs(recover_import_scale(aligned.positions, rig.positions) - 0.01) < 1e-6


def test_baked_character_npz_cache_roundtrip(tmp_path):
    path = _write_doc(tmp_path, _two_bone_doc(), "two_bone.baked.json")
    cache = tmp_path / "cache"
    first = load_baked_character(path, cache_dir=cache)
    assert list(cache.glob("baked_*.npz")), "cache file must be written on first load"
    second = load_baked_character(path, cache_dir=cache)  # served from the npz
    np.testing.assert_array_equal(first.positions, second.positions)
    np.testing.assert_array_equal(first.joint_indices, second.joint_indices)
    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_allclose(first.bind_local, second.bind_local)
    np.testing.assert_allclose(first.inverse_bind, second.inverse_bind)
    assert first.bone_names == second.bone_names
    assert second.parent_index.tolist() == [-1, 0]


def test_palette_sidecar_roundtrip(tmp_path):
    palette = _child_rz90_palette()
    world_child = _translation(CHILD_HEAD) @ _rz90()
    doc = {
        "clipId": "clip-cin-walktalk",
        "sampleTimeSeconds": 0.25,
        "character": "cc0_male_rigged3",
        "sex": "male",
        "matrixLayout": "column-major",
        "importScale": 0.01,
        "boneCount": 2,
        "bones": [  # deliberately out of order: the loader must sort by index
            {
                "index": 1,
                "name": "child",
                "parent": 0,
                "skinning": _colmajor(palette[1]),
                "world": _colmajor(world_child),
                "inverseBind": _colmajor(_translation(-CHILD_HEAD)),
            },
            {
                "index": 0,
                "name": "root",
                "parent": -1,
                "skinning": _colmajor(np.eye(4)),
                "world": _colmajor(np.eye(4)),
                "inverseBind": _colmajor(np.eye(4)),
            },
        ],
    }
    sidecar = load_palette_sidecar(_write_doc(tmp_path, doc, "dump.palette.json"))
    assert sidecar.skinning_palette.shape == (2, 4, 4)
    np.testing.assert_allclose(sidecar.skinning_palette, palette, atol=1e-12)
    assert sidecar.bone_names == ["root", "child"]
    assert sidecar.parent_index.tolist() == [-1, 0]
    assert sidecar.import_scale == 0.01
    assert sidecar.clip_id == "clip-cin-walktalk"
    assert sidecar.sample_time_seconds == 0.25
    assert sidecar.world is not None and sidecar.inverse_bind is not None
    np.testing.assert_allclose(sidecar.world[1], world_child, atol=1e-12)

    # importScale absent -> defaults to 1.0; optional matrices absent -> None.
    minimal = {
        "clipId": "clip-x",
        "sampleTimeSeconds": 0.0,
        "bones": [{"index": 0, "name": "root", "parent": -1, "skinning": _colmajor(np.eye(4))}],
    }
    lean = load_palette_sidecar(_write_doc(tmp_path, minimal, "lean.palette.json"))
    assert lean.import_scale == 1.0
    assert lean.world is None and lean.inverse_bind is None
    assert lean.skinning_palette.shape == (1, 4, 4)

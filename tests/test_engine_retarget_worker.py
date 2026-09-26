"""The engine retarget worker's pure mappings, imported with Blender's modules stubbed out.

The worker itself runs inside Blender (the end-to-end contract check covers it); these pin the
tables that decide which MPFB joint each engine bone pivots on.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

WORKER = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "bodymesh_server"
    / "blender_scripts"
    / "engine_retarget_worker.py"
)


@pytest.fixture(scope="module")
def worker(monkeypatch_module):
    mathutils = types.ModuleType("mathutils")
    mathutils.Matrix = type("Matrix", (), {})
    mathutils.Vector = type("Vector", (), {})
    monkeypatch_module.setitem(sys.modules, "bpy", types.ModuleType("bpy"))
    monkeypatch_module.setitem(sys.modules, "mathutils", mathutils)
    spec = importlib.util.spec_from_file_location("engine_retarget_worker_under_test", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def monkeypatch_module():
    patcher = pytest.MonkeyPatch()
    yield patcher
    patcher.undo()


def test_every_limb_and_finger_joint_comes_from_the_mpfb_rig(worker):
    joints = worker._joint_mapping()
    for engine_side, source_side in (("L", "r"), ("R", "l")):
        assert joints[f"legUpper{engine_side}"] == f"thigh_{source_side}"
        assert joints[f"armLower{engine_side}"] == f"lowerarm_{source_side}"
        assert joints[f"toe{engine_side}"] == f"ball_{source_side}"
        for finger in worker.FINGERS:
            assert joints[f"{finger}{engine_side}"] == f"{finger}_01_{source_side}"
            assert joints[f"{finger}{engine_side}3"] == f"{finger}_03_{source_side}"
    # 5 trunk joints + 2 sides x (8 limb joints + 15 finger joints).
    assert len(joints) == 5 + 2 * (8 + 15)


def test_the_trunk_skips_the_split_spine_joint(worker):
    joints = worker._joint_mapping()
    assert joints["pelvis"] == "pelvis"
    assert joints["spine"] == "spine_01"
    assert joints["chest"] == "spine_03"
    assert "spine_02" not in joints.values()
    # The face bones have no MPFB joint; the worker places them from the head.
    assert not {"nose", "jaw", "eyeL", "eyeR"} & set(joints)


def test_the_joint_map_is_the_inverse_of_the_whole_weight_entries(worker):
    joints = worker._joint_mapping()
    for engine_name, source_name in joints.items():
        assert worker._weight_mapping()[source_name] == {engine_name: 1.0}

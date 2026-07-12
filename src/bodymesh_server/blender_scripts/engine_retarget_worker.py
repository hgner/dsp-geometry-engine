# SPDX-License-Identifier: GPL-3.0-or-later
"""Retarget one saved MPFB body onto the engine's exact 55-bone rest rig.

This Blender-side worker adapts the established ``retarget_realistic.py``
fit -> weight -> arms-down-rest algorithm to MPFB's BodyMesh object and
``game_engine`` vertex-group names. Its only interface is request/result JSON.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import traceback
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ENGINE_HEIGHT_M = {"female": 1.66, "male": 1.79}
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SOURCE_TO_ENGINE_SIDE = {"l": "R", "r": "L"}
SIDE_PAIR_BASES = (
    "clavicle",
    "armUpper",
    "armLower",
    "hand",
    "legUpper",
    "legLower",
    "foot",
    "toe",
)


def _arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    return parser.parse_args(values)


def _atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def _mpfb_services() -> dict:
    for name, module in list(sys.modules.items()):
        if name == "mpfb" or name.endswith(".mpfb"):
            info = getattr(module, "MPFB_CONTEXTUAL_INFORMATION", None)
            if isinstance(info, dict) and info.get("SERVICES"):
                return info["SERVICES"]
    for name in ("bl_ext.user_default.mpfb", "mpfb"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        info = getattr(module, "MPFB_CONTEXTUAL_INFORMATION", None)
        if not isinstance(info, dict) or not info.get("SERVICES"):
            register = getattr(module, "register", None)
            if callable(register):
                register()
        info = getattr(module, "MPFB_CONTEXTUAL_INFORMATION", None)
        if isinstance(info, dict) and info.get("SERVICES"):
            return info["SERVICES"]
    raise RuntimeError("enabled MPFB extension services were not found")


def _load_skeleton(path: Path) -> list[dict]:
    bones = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(bones, list) or len(bones) != 55:
        raise ValueError("engine skeleton must contain exactly 55 bones")
    names = []
    for index, bone in enumerate(bones):
        if not isinstance(bone, dict):
            raise ValueError(f"engine skeleton bone {index} is not an object")
        name = str(bone.get("name") or "")
        parent = int(bone.get("parent", -2))
        pos = bone.get("pos")
        if not name or name in names:
            raise ValueError(f"engine skeleton has invalid/duplicate bone name at {index}: {name!r}")
        if parent >= index or parent < -1:
            raise ValueError(f"engine skeleton bone {name!r} has invalid parent {parent}")
        if not isinstance(pos, list) or len(pos) != 3 or not all(math.isfinite(float(v)) for v in pos):
            raise ValueError(f"engine skeleton bone {name!r} has invalid position")
        names.append(name)
    if names[0] != "pelvis" or int(bones[0]["parent"]) != -1:
        raise ValueError("engine skeleton must have pelvis as its sole root")
    return bones


def _activate(obj) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _body_coordinates(body) -> list[Vector]:
    matrix = body.matrix_world
    return [matrix @ vertex.co for vertex in body.data.vertices]


def _weight_mapping() -> dict[str, dict[str, float]]:
    mapping: dict[str, dict[str, float]] = {
        "pelvis": {"pelvis": 1.0},
        "spine_01": {"spine": 1.0},
        "spine_02": {"spine": 0.5, "chest": 0.5},
        "spine_03": {"chest": 1.0},
        "neck_01": {"neck": 1.0},
        "head": {"head": 1.0},
    }
    paired = {
        "clavicle": "clavicle",
        "upperarm": "armUpper",
        "lowerarm": "armLower",
        "hand": "hand",
        "thigh": "legUpper",
        "calf": "legLower",
        "foot": "foot",
        "ball": "toe",
    }
    # The target skeleton is rotated 180 degrees around Blender Z below. MPFB
    # game_engine `_l` groups are on +X, while the rotated engine `L` bones are
    # on -X, so source sides must cross exactly once at this mapping boundary.
    for source_base, engine_base in paired.items():
        for source_side, engine_side in SOURCE_TO_ENGINE_SIDE.items():
            mapping[f"{source_base}_{source_side}"] = {f"{engine_base}{engine_side}": 1.0}
    for finger in FINGERS:
        for side_code, side in SOURCE_TO_ENGINE_SIDE.items():
            for segment in (1, 2, 3):
                suffix = "" if segment == 1 else str(segment)
                mapping[f"{finger}_{segment:02d}_{side_code}"] = {f"{finger}{side}{suffix}": 1.0}
    return mapping


def _side_pairs() -> list[tuple[str, str]]:
    pairs = [(f"{base}L", f"{base}R") for base in SIDE_PAIR_BASES]
    for finger in FINGERS:
        for suffix in ("", "2", "3"):
            pairs.append((f"{finger}L{suffix}", f"{finger}R{suffix}"))
    return pairs


def _validate_side_locality(
    coordinates: list[Vector],
    remapped: list[list[tuple[str, float]]],
    world: dict[str, Vector],
) -> int:
    weighted_x: dict[str, float] = {}
    weight_totals: dict[str, float] = {}
    for coordinate, influences in zip(coordinates, remapped, strict=True):
        for name, weight in influences:
            weighted_x[name] = weighted_x.get(name, 0.0) + coordinate.x * weight
            weight_totals[name] = weight_totals.get(name, 0.0) + weight

    checked = 0
    for left, right in _side_pairs():
        if weight_totals.get(left, 0.0) <= 0.0 or weight_totals.get(right, 0.0) <= 0.0:
            raise ValueError(f"MPFB remap has no paired skin weights for {left}/{right}")
        bone_delta = world[left].x - world[right].x
        centroid_left = weighted_x[left] / weight_totals[left]
        centroid_right = weighted_x[right] / weight_totals[right]
        centroid_delta = centroid_left - centroid_right
        if abs(bone_delta) <= 1.0e-6 or bone_delta * centroid_delta <= 0.0:
            raise ValueError(
                f"MPFB remap crossed {left}/{right}: bone delta X={bone_delta:.6g}, "
                f"weighted-centroid delta X={centroid_delta:.6g}"
            )
        checked += 1
    return checked


def _remap_weights(body, engine_names: list[str]) -> tuple[list[list[tuple[str, float]]], int]:
    mapping = _weight_mapping()
    group_names = {group.index: group.name for group in body.vertex_groups}
    remapped: list[list[tuple[str, float]]] = []
    max_influences = 0
    for vertex in body.data.vertices:
        accumulated: dict[str, float] = {}
        for membership in vertex.groups:
            source_name = group_names.get(membership.group, "")
            for target_name, factor in mapping.get(source_name, {}).items():
                accumulated[target_name] = accumulated.get(target_name, 0.0) + (
                    float(membership.weight) * factor
                )
        strongest = sorted(accumulated.items(), key=lambda item: (-item[1], item[0]))[:4]
        total = sum(weight for _, weight in strongest)
        if total <= 1.0e-8:
            raise ValueError(f"MPFB vertex {vertex.index} has no mapped game_engine weight")
        normalized = [(name, weight / total) for name, weight in strongest]
        remapped.append(normalized)
        max_influences = max(max_influences, len(normalized))

    for group in list(body.vertex_groups):
        body.vertex_groups.remove(group)
    groups = {name: body.vertex_groups.new(name=name) for name in engine_names}
    for vertex_index, influences in enumerate(remapped):
        for name, weight in influences:
            groups[name].add([vertex_index], weight, "REPLACE")
    return remapped, max_influences


def _to_blender(position: list[float]) -> Vector:
    return Vector((float(position[0]), -float(position[2]), float(position[1])))


def _descendants(children: dict[str, list[str]], name: str) -> list[str]:
    result: list[str] = []
    for child in children.get(name, []):
        result.append(child)
        result.extend(_descendants(children, child))
    return result


def _fit_engine_world(
    bones: list[dict], body, sex: str
) -> tuple[dict[str, Vector], dict[str, object], float]:
    coords = _body_coordinates(body)
    z_values = [coordinate.z for coordinate in coords]
    z_min, z_max = min(z_values), max(z_values)
    height = z_max - z_min
    if height <= 0.4:
        raise ValueError(f"MPFB body has invalid height {height}")

    hip_band = [coordinate for coordinate in coords if 0.50 <= (coordinate.z - z_min) / height <= 0.56]
    if not hip_band:
        raise ValueError("MPFB body has no hip-band vertices")
    hip_x = sum(coordinate.x for coordinate in hip_band) / len(hip_band)
    hip_y = sum(coordinate.y for coordinate in hip_band) / len(hip_band)
    recenter = Vector((hip_x, hip_y, z_min))
    body.matrix_world = Matrix.Translation(-recenter) @ body.matrix_world
    _activate(body)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    coords = _body_coordinates(body)

    scale = height / ENGINE_HEIGHT_M[sex]
    # MPFB faces -Y. The established engine->Blender conversion faces +Y,
    # so rotate the target skeleton 180 degrees around Blender Z.
    world: dict[str, Vector] = {}
    for bone in bones:
        point = _to_blender(bone["pos"]) * scale
        world[bone["name"]] = Vector((-point.x, -point.y, point.z))

    children: dict[str, list[str]] = {}
    for bone in bones:
        parent = int(bone["parent"])
        if parent >= 0:
            children.setdefault(bones[parent]["name"], []).append(bone["name"])

    arm_fit: dict[str, object] = {}
    for side in ("L", "R"):
        shoulder = world[f"armUpper{side}"]
        sign = 1.0 if shoulder.x > 0.0 else -1.0
        arm_points = [
            coordinate
            for coordinate in coords
            if (coordinate.x - shoulder.x) * sign > 0.04 and coordinate.z / height > 0.35
        ]
        if len(arm_points) < 50:
            raise ValueError(f"MPFB arm fit {side} found only {len(arm_points)} vertices")
        arm_points.sort(key=lambda coordinate: (coordinate - shoulder).length, reverse=True)
        count = max(1, len(arm_points) // 10)
        tip = sum(arm_points[:count], Vector()) / count
        mesh_direction = (tip - shoulder).normalized()
        engine_direction = (world[f"hand{side}"] - shoulder).normalized()
        fit = engine_direction.rotation_difference(mesh_direction)
        moved = [f"armLower{side}", f"hand{side}"] + _descendants(children, f"hand{side}")
        for name in moved:
            world[name] = shoulder + fit @ (world[name] - shoulder)
        arm_fit[side] = fit
        sys.stderr.write(f"ARM-FIT {side}: {math.degrees(fit.angle):.2f} degrees\n")
    if set(arm_fit) != {"L", "R"}:
        raise ValueError("both MPFB arms must fit before rest normalization")
    return world, arm_fit, height


def _build_engine_armature(bones: list[dict], world: dict[str, Vector]):
    primary = {
        "pelvis": "spine",
        "spine": "chest",
        "chest": "neck",
        "neck": "head",
        "clavicleL": "armUpperL",
        "armUpperL": "armLowerL",
        "armLowerL": "handL",
        "handL": "middleL",
        "clavicleR": "armUpperR",
        "armUpperR": "armLowerR",
        "armLowerR": "handR",
        "handR": "middleR",
        "legUpperL": "legLowerL",
        "legLowerL": "footL",
        "footL": "toeL",
        "legUpperR": "legLowerR",
        "legLowerR": "footR",
        "footR": "toeR",
    }
    for finger in FINGERS:
        for side in ("L", "R"):
            primary[f"{finger}{side}"] = f"{finger}{side}2"
            primary[f"{finger}{side}2"] = f"{finger}{side}3"

    data = bpy.data.armatures.new("engine_rig")
    armature = bpy.data.objects.new("engine_rig", data)
    bpy.context.scene.collection.objects.link(armature)
    _activate(armature)
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = {}
    for bone in bones:
        name = bone["name"]
        edit_bone = data.edit_bones.new(name)
        head = world[name]
        target = primary.get(name)
        if target and target in world:
            tail = world[target]
        else:
            parent = bones[int(bone["parent"])]["name"] if int(bone["parent"]) >= 0 else None
            direction = (
                (head - world[parent]).normalized() * 0.02
                if parent and (head - world[parent]).length > 1.0e-6
                else Vector((0.0, 0.0, 0.05))
            )
            if name == "head":
                direction = Vector((0.0, 0.0, 0.10))
            tail = head + direction
        if (tail - head).length < 1.0e-4:
            tail = head + Vector((0.0, 0.0, 0.02))
        edit_bone.head = head
        edit_bone.tail = tail
        edit_bone.use_deform = True
        edit_bones[name] = edit_bone
    for bone in bones:
        parent = int(bone["parent"])
        if parent >= 0:
            edit_bones[bone["name"]].parent = edit_bones[bones[parent]["name"]]
    bpy.ops.object.mode_set(mode="OBJECT")
    return armature


def _apply_arms_down_rest(body, armature, arm_fit: dict[str, object]) -> None:
    _activate(armature)
    bpy.ops.object.mode_set(mode="POSE")
    for side, fit in arm_fit.items():
        pose_bone = armature.pose.bones[f"armUpper{side}"]
        bpy.context.view_layer.update()
        matrix = pose_bone.matrix.copy()
        head = matrix.to_translation()
        pivot = Matrix.Translation(head) @ fit.inverted().to_matrix().to_4x4() @ Matrix.Translation(-head)
        pose_bone.matrix = pivot @ matrix
        bpy.context.view_layer.update()
    bpy.ops.object.mode_set(mode="OBJECT")

    source_modifier = next((modifier for modifier in body.modifiers if modifier.type == "ARMATURE"), None)
    if source_modifier is None:
        raise ValueError("engine Armature modifier is missing before rest bake")
    keep = body.modifiers.new(name="EngineArmature", type="ARMATURE")
    keep.object = armature
    _activate(body)
    bpy.ops.object.modifier_apply(modifier=source_modifier.name)

    _activate(armature)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    for pose_bone in armature.pose.bones:
        if any(abs(value) > 1.0e-5 for row in pose_bone.matrix_basis for value in row):
            identity = Matrix.Identity(4)
            if any(
                abs(pose_bone.matrix_basis[row][col] - identity[row][col]) > 1.0e-5
                for row in range(4)
                for col in range(4)
            ):
                raise ValueError(f"pose basis did not clear for {pose_bone.name}")


def _run(request: dict) -> dict:
    candidate_dir = Path(request["candidate_dir"]).resolve()
    skeleton_path = Path(request["skeleton_path"]).resolve()
    output_path = Path(request["glb_path"]).resolve()
    sex = str(request["rig_sex"]).lower()
    if sex not in ENGINE_HEIGHT_M:
        raise ValueError("rig_sex must be male or female")
    if output_path.parent != candidate_dir or output_path.name != "body_engine.glb":
        raise ValueError("engine GLB path must be candidate_dir/body_engine.glb")
    bones = _load_skeleton(skeleton_path)
    engine_names = [bone["name"] for bone in bones]
    body = bpy.data.objects.get("BodyMesh")
    if body is None or body.type != "MESH":
        raise ValueError("MPFB BodyMesh object was not found")
    if body.data.uv_layers.active is None:
        raise ValueError("MPFB BodyMesh has no active UV map for tangent export")
    if not any(group.name.startswith("joint-") for group in body.vertex_groups):
        raise ValueError("MPFB detailed helper landmark groups are missing; regenerate the candidate")

    services = _mpfb_services()
    source_rig = services["HumanService"].add_builtin_rig(body, "game_engine", import_weights=True)
    if source_rig is None or len(source_rig.data.bones) != 53:
        raise ValueError("MPFB game_engine rig did not produce its expected 53 bones")
    if body.data.shape_keys is not None:
        services["TargetService"].bake_targets(body)
    services["ExportService"].bake_modifiers_remove_helpers(
        body,
        bake_masks=True,
        bake_subdiv=False,
        remove_helpers=True,
        also_proxy=False,
    )
    if len(body.data.vertices) != 13_380:
        raise ValueError(f"MPFB helper removal produced {len(body.data.vertices)} vertices; expected 13380")

    remapped, max_influences = _remap_weights(body, engine_names)
    body_world = body.matrix_world.copy()
    body.parent = None
    body.matrix_world = body_world
    for modifier in list(body.modifiers):
        if modifier.type == "ARMATURE":
            body.modifiers.remove(modifier)
    bpy.data.objects.remove(source_rig, do_unlink=True)

    world, arm_fit, source_height = _fit_engine_world(bones, body, sex)
    side_locality_pairs = _validate_side_locality(_body_coordinates(body), remapped, world)
    armature = _build_engine_armature(bones, world)
    body.parent = armature
    body.matrix_parent_inverse = armature.matrix_world.inverted()
    armature_modifier = body.modifiers.new(name="EngineArmatureSource", type="ARMATURE")
    armature_modifier.object = armature
    _apply_arms_down_rest(body, armature, arm_fit)

    for polygon in body.data.polygons:
        polygon.use_smooth = True
    body.data.calc_tangents(uvmap=body.data.uv_layers.active.name)
    _activate(body)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = body
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_skins=True,
        export_tangents=True,
        export_yup=True,
        export_animations=False,
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("Blender produced no engine GLB")

    return {
        "status": "complete",
        "glb_path": str(output_path),
        "rig_sex": sex,
        "bone_count": len(armature.data.bones),
        "rest_pose": "arms-down",
        "vertex_count": len(body.data.vertices),
        "max_influences": max_influences,
        "mapped_vertices": len(remapped),
        "source_side_mapping": "swapped-after-z-flip",
        "side_locality_pairs": side_locality_pairs,
        "source_height_m": source_height,
        "warnings": [
            "MPFB game_engine has no nose/jaw/eye weights; those exact engine contract bones "
            "are present but zero-influence"
        ],
    }


def main() -> None:
    args = _arguments()
    result_path = Path(args.result).resolve()
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        payload = _run(request)
    except Exception as exc:
        traceback.print_exc()
        payload = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        _atomic_json(result_path, payload)
        raise
    _atomic_json(result_path, payload)


if __name__ == "__main__":
    main()

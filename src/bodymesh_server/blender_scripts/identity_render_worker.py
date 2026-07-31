# SPDX-FileCopyrightText: 2026 hgner <hgner09@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender-side worker for the fixed identity-v1 render set.

This executes in Blender's GPL process boundary. Its only protocol is the
host-authored request JSON and a compact worker-result JSON.
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


def _services() -> tuple[object, dict]:
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
            return module, info["SERVICES"]
    raise RuntimeError("enabled MPFB extension module was not found")


def _human() -> object:
    exact = bpy.data.objects.get("BodyMesh")
    if exact is not None and exact.type == "MESH":
        return exact
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("character.blend contains no mesh object")
    return max(meshes, key=lambda obj: len(obj.data.vertices))


def _material(name: str, base_color, roughness: float, metallic: float = 0.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (*base_color, 1.0)
    principled.inputs["Roughness"].default_value = roughness
    principled.inputs["Metallic"].default_value = metallic
    return material


def _apply_skin(human, services: dict) -> tuple[str, list[str]]:
    warnings = []
    try:
        services["MaterialService"].create_v2_skin_material("S2 Enhanced skin", human)
        if human.data.materials:
            return "mpfb-enhanced-skin", warnings
        raise RuntimeError("MPFB returned no material")
    except Exception as exc:
        warnings.append(f"enhanced skin unavailable; used deterministic Principled fallback: {exc}")
        material = _material("S2 fallback skin", (0.62, 0.40, 0.31), 0.48)
        principled = material.node_tree.nodes.get("Principled BSDF")
        if "Subsurface Weight" in principled.inputs:
            principled.inputs["Subsurface Weight"].default_value = 0.16
            principled.inputs["Subsurface Radius"].default_value = (0.36, 0.19, 0.13)
            principled.inputs["Subsurface Scale"].default_value = 0.05
        human.data.materials.clear()
        human.data.materials.append(material)
        return "principled-fallback", warnings


def _group_indices(obj, group_name: str) -> set[int]:
    group = obj.vertex_groups.get(group_name)
    if group is None:
        raise RuntimeError(f"character is missing landmark/helper group {group_name!r}")
    return {
        vertex.index
        for vertex in obj.data.vertices
        if any(item.group == group.index and item.weight > 0.0 for item in vertex.groups)
    }


def _evaluated_coordinates(obj) -> list[Vector]:
    modifier_states = [(modifier, modifier.show_viewport, modifier.show_render) for modifier in obj.modifiers]
    try:
        # MPFB's saved mask modifier intentionally removes helper vertices. Disable
        # modifiers for this snapshot so active macro shape keys are evaluated while
        # landmark/helper vertex indices retain their original topology.
        for modifier, _, _ in modifier_states:
            modifier.show_viewport = False
            modifier.show_render = False
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            if len(mesh.vertices) != len(obj.data.vertices):
                raise RuntimeError("evaluated character topology differs from its landmark topology")
            return [vertex.co.copy() for vertex in mesh.vertices]
        finally:
            evaluated.to_mesh_clear()
    finally:
        for modifier, show_viewport, show_render in modifier_states:
            modifier.show_viewport = show_viewport
            modifier.show_render = show_render
        bpy.context.view_layer.update()


def _group_center(obj, group_name: str, coordinates: list[Vector]) -> Vector:
    indices = _group_indices(obj, group_name)
    if not indices:
        raise RuntimeError(f"character group {group_name!r} has no vertices")
    center = Vector((0.0, 0.0, 0.0))
    for index in indices:
        center += obj.matrix_world @ coordinates[index]
    return center / len(indices)


def _extract_helper(
    obj,
    group_name: str,
    object_name: str,
    material,
    coordinates: list[Vector],
):
    indices = _group_indices(obj, group_name)
    ordered = sorted(indices)
    remap = {source: target for target, source in enumerate(ordered)}
    faces = [
        [remap[index] for index in polygon.vertices]
        for polygon in obj.data.polygons
        if all(index in indices for index in polygon.vertices)
    ]
    if not faces:
        raise RuntimeError(f"helper group {group_name!r} has no contained faces")
    mesh = bpy.data.meshes.new(object_name + "Mesh")
    mesh.from_pydata([coordinates[index] for index in ordered], [], faces)
    mesh.update()
    extracted = bpy.data.objects.new(object_name, mesh)
    bpy.context.scene.collection.objects.link(extracted)
    extracted.matrix_world = obj.matrix_world.copy()
    extracted.data.materials.append(material)
    for polygon in extracted.data.polygons:
        polygon.use_smooth = True
    return extracted


def _add_flat_sphere(name: str, location: Vector, scale: tuple[float, float, float], material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def _world_bounds(obj) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    if not points:
        raise RuntimeError(f"extracted helper {obj.name!r} has no vertices")
    return (
        Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points))),
        Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points))),
    )


def _make_face_features(human) -> dict[str, Vector]:
    sclera = _material("S2 sclera", (0.82, 0.84, 0.82), 0.22)
    iris = _material("S2 iris", (0.12, 0.055, 0.025), 0.30)
    pupil = _material("S2 pupil", (0.004, 0.004, 0.004), 0.18)
    lashes = _material("S2 lashes", (0.018, 0.010, 0.008), 0.58)
    features: dict[str, Vector] = {}
    coordinates = _evaluated_coordinates(human)
    for side in ("l", "r"):
        eye_group = f"helper-{side}-eye"
        eye = _extract_helper(
            human,
            eye_group,
            f"S2_{side.upper()}_EYE",
            sclera,
            coordinates,
        )
        for suffix in ("1", "2"):
            _extract_helper(
                human,
                f"helper-{side}-eyelashes-{suffix}",
                f"S2_{side.upper()}_LASHES_{suffix}",
                lashes,
                coordinates,
            )
        eye_low, eye_high = _world_bounds(eye)
        center = (eye_low + eye_high) / 2.0
        features[f"{side}_eye"] = center
        iris_radius = max(min(eye_high.x - eye_low.x, eye_high.z - eye_low.z) * 0.23, 0.0025)
        iris_depth = max(iris_radius * 0.29, 0.0007)
        iris_center = Vector((center.x, eye_low.y - iris_depth * 0.35, center.z))
        _add_flat_sphere(
            f"S2_{side.upper()}_IRIS",
            iris_center,
            (iris_radius, iris_depth, iris_radius),
            iris,
        )
        _add_flat_sphere(
            f"S2_{side.upper()}_PUPIL",
            iris_center + Vector((0.0, -iris_depth * 0.82, 0.0)),
            (iris_radius * 0.42, iris_depth * 0.50, iris_radius * 0.42),
            pupil,
        )
    features["mouth"] = _group_center(human, "joint-mouth", coordinates)
    features["jaw"] = _group_center(human, "joint-jaw", coordinates)
    features["head_top"] = _group_center(human, "joint-head-2", coordinates)
    features["neck"] = _group_center(human, "joint-neck", coordinates)
    return features


def _body_bounds(human) -> tuple[Vector, Vector]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = human.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        points = [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
    finally:
        evaluated.to_mesh_clear()
    if not points:
        raise RuntimeError("character evaluated mesh has no vertices")
    return (
        Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points))),
        Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points))),
    )


def _configure_scene(request: dict) -> str:
    scene = bpy.context.scene
    for obj in list(scene.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(request["samples"])
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.seed = int(request["seed"])
    scene.cycles.use_animated_seed = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.use_nodes = False
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    scene.view_settings.exposure = -0.5
    scene.view_settings.gamma = 1.0
    scene.world = scene.world or bpy.data.worlds.new("S2 World")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.025, 0.028, 0.035, 1.0)
    background.inputs["Strength"].default_value = 0.08

    requested = str(request["device"]).upper()
    try:
        preferences = bpy.context.preferences.addons["cycles"].preferences
    except (KeyError, AttributeError):
        scene.cycles.device = "CPU"
        return "CPU"
    if requested == "CPU":
        scene.cycles.device = "CPU"
        return "CPU"
    candidates = [requested] + ([] if requested == "CUDA" else ["CUDA"])
    for backend in candidates:
        try:
            preferences.compute_device_type = backend
            preferences.get_devices()
        except (TypeError, RuntimeError):
            continue
        devices = [device for device in preferences.devices if device.type == backend]
        if not devices:
            continue
        for device in preferences.devices:
            device.use = device.type == backend
        scene.cycles.device = "GPU"
        return backend
    scene.cycles.device = "CPU"
    return "CPU"


def _add_area(name: str, target: Vector, offset: Vector, energy: float, size: float, color):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    data.color = color
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = target + offset
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def _lighting(target: Vector, radius: float, preset: str) -> None:
    for obj in list(bpy.context.scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    scale = max(radius, 0.18)
    key_energy = max(80.0, 550.0 * scale * scale)
    fill_energy = max(25.0, 180.0 * scale * scale)
    rim_energy = max(45.0, 320.0 * scale * scale)
    key_x = 1.6 if preset == "key-right" else -1.6
    if preset == "soft-front":
        key_x = -0.25
        # A frontal pair reaches the face much more directly than the side-key rigs.
        # Keep it deliberately darker so pale MPFB skin retains the local contrast
        # required by the production YuNet/SFace detector.
        key_energy *= 0.62
        fill_energy *= 0.72
    _add_area(
        "S2 Key",
        target,
        Vector((key_x, -2.0, 1.4)) * scale,
        key_energy,
        1.35 * scale,
        (1.0, 0.78, 0.64),
    )
    _add_area(
        "S2 Fill",
        target,
        Vector((-key_x * 1.15, -1.45, 0.25)) * scale,
        fill_energy,
        2.0 * scale,
        (0.58, 0.72, 1.0),
    )
    _add_area(
        "S2 Rim",
        target,
        Vector((0.25, 1.8, 1.5)) * scale,
        rim_energy,
        1.25 * scale,
        (1.0, 0.88, 0.74),
    )


def _camera() -> object:
    data = bpy.data.cameras.new("S2 Identity Camera")
    data.lens = 70.0
    data.sensor_width = 36.0
    data.clip_start = 0.01
    data.clip_end = 100.0
    obj = bpy.data.objects.new("S2 Identity Camera", data)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.scene.camera = obj
    return obj


def _point_at(camera, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def _frame_face(
    camera, features: dict[str, Vector], yaw_deg: float, pitch_deg: float
) -> tuple[Vector, float]:
    eyes = (features["l_eye"] + features["r_eye"]) / 2.0
    face_center = eyes.lerp(features["mouth"], 0.33)
    head_height = max(features["head_top"].z - features["jaw"].z, 0.20)
    distance = head_height * 2.45
    offset = Vector((0.0, -distance, math.tan(math.radians(pitch_deg)) * distance))
    offset = Matrix.Rotation(math.radians(yaw_deg), 4, "Z") @ offset
    camera.data.lens = 70.0
    camera.location = face_center + offset
    _point_at(camera, face_center)
    return face_center, head_height * 0.72


def _frame_body(camera, bounds: tuple[Vector, Vector], view: str) -> tuple[Vector, float]:
    low, high = bounds
    center = (low + high) / 2.0
    radius = max((high - low).length / 2.0, 0.25)
    camera.data.lens = 55.0
    distance = (radius / math.tan(camera.data.angle / 2.0)) * 1.18
    directions = {
        "front": Vector((0.0, -1.0, 0.08)),
        "three-quarter": Vector((0.70, -0.72, 0.09)),
        "side": Vector((1.0, 0.0, 0.08)),
    }
    camera.location = center + directions[view].normalized() * distance
    _point_at(camera, center)
    return center, radius


def _add_floor(bounds: tuple[Vector, Vector]) -> None:
    low, high = bounds
    extent = max(high.x - low.x, high.y - low.y, 1.0) * 3.0
    bpy.ops.mesh.primitive_plane_add(size=extent, location=(0.0, 0.0, low.z - 0.004))
    floor = bpy.context.object
    floor.name = "S2 Floor"
    floor.data.materials.append(_material("S2 Floor", (0.055, 0.060, 0.070), 0.72))


def _expression_keys(
    human, services: dict, request: dict
) -> tuple[dict[tuple[str, str], object], dict[str, float]]:
    target_service = services["TargetService"]
    location_service = services["LocationService"]
    macro = target_service.get_macro_info_dict_from_basemesh(human)
    raw_race = macro.get("race") or {}
    race_weights = {
        race: max(0.0, float(raw_race.get(race, 0.0))) for race in ("african", "asian", "caucasian")
    }
    total = sum(race_weights.values())
    if total <= 0.0:
        race_weights = {race: 1.0 / 3.0 for race in race_weights}
    else:
        race_weights = {race: value / total for race, value in race_weights.items()}
    target_names = sorted(
        {target for spec in request["closeups"] for target in (spec.get("targets") or {}).keys()}
    )
    root = Path(location_service.get_mpfb_data("targets")) / "expression" / "units"
    keys = {}
    for target in target_names:
        for race in race_weights:
            path = root / race / f"{target}.target.gz"
            if not path.is_file():
                raise RuntimeError(f"required MPFB expression target is missing: {path}")
            key = target_service.load_target(
                human,
                str(path),
                weight=0.0,
                name=f"S2-{race}-{target}",
            )
            keys[(target, race)] = key
    return keys, race_weights


def _set_expression(keys, race_weights: dict[str, float], targets: dict[str, float]) -> None:
    for key in keys.values():
        key.value = 0.0
    for target, weight in targets.items():
        for race, race_weight in race_weights.items():
            keys[(target, race)].value = float(weight) * race_weight
    bpy.context.view_layer.update()


def _render(scene, output: Path, width: int, height: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def _run(request: dict) -> dict:
    if request.get("operation") != "identity-render" or request.get("preset") != "identity-v1":
        raise ValueError("unsupported identity-render request")
    output_dir = Path(request["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    module, services = _services()
    mpfb_version = ".".join(str(value) for value in getattr(module, "VERSION", ()))
    if not mpfb_version:
        mpfb_version = str(bpy.context.scene.get("bodymesh_mpfb_version") or "unknown")
    human = _human()
    material_backend, warnings = _apply_skin(human, services)
    features = _make_face_features(human)
    bounds = _body_bounds(human)
    _add_floor(bounds)
    device_used = _configure_scene(request)
    camera = _camera()
    expression_keys, race_weights = _expression_keys(human, services, request)
    scene = bpy.context.scene
    renders = []

    for spec in request["closeups"]:
        _set_expression(expression_keys, race_weights, spec.get("targets") or {})
        target, radius = _frame_face(
            camera,
            features,
            float(spec.get("yaw_deg") or 0.0),
            float(spec.get("pitch_deg") or 0.0),
        )
        _lighting(target, radius, str(spec["lighting"]))
        output = (output_dir / spec["relative_path"]).resolve()
        _render(scene, output, int(spec["width"]), int(spec["height"]))
        renders.append(
            {key: spec[key] for key in ("asset_id", "kind", "category", "view", "expression", "lighting")}
            | {"path": str(output)}
        )

    _set_expression(expression_keys, race_weights, {})
    for spec in request["body_views"]:
        target, radius = _frame_body(camera, bounds, str(spec["view"]))
        _lighting(target, radius, "studio")
        output = (output_dir / spec["relative_path"]).resolve()
        _render(scene, output, int(spec["width"]), int(spec["height"]))
        renders.append(
            {key: spec[key] for key in ("asset_id", "kind", "category", "view", "expression", "lighting")}
            | {"path": str(output)}
        )

    return {
        "status": "complete",
        "preset": "identity-v1",
        "recipe_sha256": str(request["recipe_sha256"]),
        "renderer": "CYCLES",
        "blender_version": str(bpy.app.version_string),
        "mpfb_version": mpfb_version,
        "device_used": device_used,
        "material_backend": material_backend,
        "expression_backend": "mpfb-legacy-race-targets",
        "renders": renders,
        "warnings": warnings,
    }


def main() -> None:
    args = _arguments()
    request_path = Path(args.request).resolve()
    result_path = Path(args.result).resolve()
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        payload = _run(request)
    except Exception as exc:
        traceback.print_exc()
        payload = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        _atomic_json(result_path, payload)
        raise
    _atomic_json(result_path, payload)


if __name__ == "__main__":
    main()

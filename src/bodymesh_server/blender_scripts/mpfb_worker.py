# SPDX-FileCopyrightText: 2026 hgner <hgner09@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender-side MPFB worker.

This file is a separate process boundary from the outer MCP server and imports
MPFB's GPL service API at runtime. Its only protocol is request/result JSON.
"""

from __future__ import annotations

import argparse
import array
import importlib
import json
import sys
import traceback
from pathlib import Path, PurePosixPath

import bpy
from mathutils import Vector


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


def _mpfb_root_module():
    matches = []
    for name, module in list(sys.modules.items()):
        if name == "mpfb" or name.endswith(".mpfb"):
            matches.append(module)
    for module in matches:
        info = getattr(module, "MPFB_CONTEXTUAL_INFORMATION", None)
        if isinstance(info, dict) and info.get("SERVICES"):
            return module

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
            return module
    raise RuntimeError("enabled MPFB extension module was not found")


def _services() -> tuple[object, dict]:
    module = _mpfb_root_module()
    info = module.MPFB_CONTEXTUAL_INFORMATION
    return module, info["SERVICES"]


def _clear_scene() -> None:
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _macro_details(target_service, supplied: dict) -> dict:
    macro = target_service.get_default_macro_info_dict()
    for key in ("gender", "age", "muscle", "weight", "proportions", "height", "cupsize", "firmness"):
        if key in supplied:
            macro[key] = float(supplied[key])
    race_keys = ("african", "asian", "caucasian")
    if any(key in supplied for key in race_keys):
        weights = {key: float(supplied.get(key, 0.0)) for key in race_keys}
        total = sum(weights.values())
        if total <= 0.0:
            weights = {key: 1.0 / 3.0 for key in race_keys}
        else:
            weights = {key: value / total for key, value in weights.items()}
        macro["race"].update(weights)
    return macro


def _apply_targets(basemesh, target_service, location_service, targets: dict) -> None:
    root = Path(location_service.get_mpfb_data("targets")).resolve()
    for raw_name, weight in targets.items():
        relative = PurePosixPath(raw_name)
        path = (root / Path(*relative.parts)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"target escaped MPFB root: {raw_name}") from exc
        target_service.load_target(basemesh, str(path), weight=float(weight))


def _activate(obj) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _snapshot(
    obj,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        mesh.calc_loop_triangles()
        matrix = evaluated.matrix_world
        normal_matrix = matrix.to_3x3().inverted().transposed()
        vertices = []
        normals = []
        for vertex in mesh.vertices:
            coordinate = matrix @ vertex.co
            normal = normal_matrix @ vertex.normal
            normal.normalize()
            vertices.append((float(coordinate.x), float(coordinate.y), float(coordinate.z)))
            normals.append((float(normal.x), float(normal.y), float(normal.z)))
        triangles = [tuple(int(index) for index in triangle.vertices) for triangle in mesh.loop_triangles]
        return vertices, normals, triangles
    finally:
        evaluated.to_mesh_clear()


def _height(vertices: list[tuple[float, float, float]]) -> float:
    return max(vertex[2] for vertex in vertices) - min(vertex[2] for vertex in vertices)


def _scale_to_height(obj, known_height_m: float | None) -> None:
    if known_height_m is None:
        return
    vertices, _, _ = _snapshot(obj)
    current = _height(vertices)
    if current <= 1.0e-8:
        raise ValueError("created body has zero height")
    factor = float(known_height_m) / current
    obj.scale = tuple(component * factor for component in obj.scale)
    _activate(obj)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def _write_obj(path: Path, vertices, normals, triangles) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# Blender Body-Mesh MCP / MPFB evaluated mesh\n")
        for x, y, z in vertices:
            stream.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for x, y, z in normals:
            stream.write(f"vn {x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in triangles:
            stream.write(f"f {a + 1}//{a + 1} {b + 1}//{b + 1} {c + 1}//{c + 1}\n")


def _write_ply(path: Path, vertices, normals, triangles) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write("comment generated by Blender Body-Mesh MCP; neutral MPFB mesh in meters\n")
        stream.write(f"element vertex {len(vertices)}\n")
        for declaration in (
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property int sourceIndex",
            "property float restx",
            "property float resty",
            "property float restz",
        ):
            stream.write(declaration + "\n")
        stream.write(f"element face {len(triangles)}\n")
        stream.write("property list uchar int vertex_indices\nend_header\n")
        for coordinate, normal in zip(vertices, normals, strict=True):
            x, y, z = coordinate
            nx, ny, nz = normal
            stream.write(
                f"{x:.9g} {y:.9g} {z:.9g} {nx:.9g} {ny:.9g} {nz:.9g} 220 220 220 0 {x:.9g} {y:.9g} {z:.9g}\n"
            )
        for a, b, c in triangles:
            stream.write(f"3 {a} {b} {c}\n")
    meta = path.with_suffix(".meta.json")
    _atomic_json(
        meta,
        {
            "boneMap": {"0": "body"},
            "clipId": None,
            "sampleTime": None,
            "vertCount": len(vertices),
            "collisionPush": None,
            "bakedCollisionPush": None,
            "imported": False,
            "provenance": "blender-mpfb-neutral-not-engine-dump",
        },
    )
    return meta


def _look_at(camera, target: Vector) -> None:
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _binarize_render(path: Path) -> None:
    """Convert Blender's rendered silhouette to an exact 0/255 non-color mask."""

    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        count = int(image.size[0]) * int(image.size[1]) * 4
        pixels = array.array("f", [0.0]) * count
        image.pixels.foreach_get(pixels)
        for index in range(0, count, 4):
            value = 1.0 if max(pixels[index : index + 3]) > 0.05 else 0.0
            pixels[index] = value
            pixels[index + 1] = value
            pixels[index + 2] = value
            pixels[index + 3] = 1.0
        image.colorspace_settings.name = "Non-Color"
        image.pixels.foreach_set(pixels)
        image.filepath_raw = str(path)
        image.file_format = "PNG"
        image.save()
    finally:
        bpy.data.images.remove(image)


def _render_views(candidate_dir: Path, obj, vertices, views: list[str], size: int) -> dict[str, str]:
    if not views:
        return {}
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    if scene.world is None:
        scene.world = bpy.data.worlds.new("BODYMESH_WORLD")
    scene.world.color = (0.0, 0.0, 0.0)
    shading = scene.display.shading
    shading.light = "FLAT"
    shading.color_type = "SINGLE"
    shading.single_color = (1.0, 1.0, 1.0)
    shading.background_type = "WORLD"
    shading.show_shadows = False
    shading.show_cavity = False
    shading.show_specular_highlight = False
    scene.view_layers[0].material_override = None

    camera_data = bpy.data.cameras.new("BODYMESH_CAMERA")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new("BODYMESH_CAMERA", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    xs = [value[0] for value in vertices]
    ys = [value[1] for value in vertices]
    zs = [value[2] for value in vertices]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    center = Vector(((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0))
    body_height = zmax - zmin
    depth = max(xmax - xmin, ymax - ymin, body_height, 0.1)
    distance = depth * 3.0
    positions = {
        "front": Vector((center.x, ymin - distance, center.z)),
        "side": Vector((xmax + distance, center.y, center.z)),
        "back": Vector((center.x, ymax + distance, center.z)),
    }
    widths = {"front": xmax - xmin, "back": xmax - xmin, "side": ymax - ymin}
    render_dir = candidate_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for view in views:
        camera.location = positions[view]
        camera_data.ortho_scale = max(body_height, widths[view]) * 1.10
        camera_data.lens = 50.0
        camera_data.clip_start = 0.001
        camera_data.clip_end = distance * 4.0
        _look_at(camera, center)
        output = render_dir / f"{view}_mask.png"
        scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
        _binarize_render(output)
        result[view] = str(output)
    return result


def _run(request: dict) -> dict:
    candidate_dir = Path(request["candidate_dir"]).resolve()
    candidate_dir.mkdir(parents=True, exist_ok=True)
    module, services = _services()
    expected_version = request.get("mpfb_version")
    actual_version = ".".join(str(value) for value in getattr(module, "VERSION", ()))
    if expected_version and actual_version and expected_version != actual_version:
        raise RuntimeError(
            f"MPFB version changed during job: expected {expected_version}, got {actual_version}"
        )

    human_service = services["HumanService"]
    target_service = services["TargetService"]
    location_service = services["LocationService"]
    _clear_scene()
    macro = _macro_details(target_service, request.get("macro_parameters") or {})
    basemesh = human_service.create_human(
        mask_helpers=True,
        # Keep joint-* landmark groups until the engine-retarget worker has
        # fitted MPFB's topology weights to the exact 55-bone rig. The mask
        # still hides helpers from neutral snapshots/renders.
        detailed_helpers=True,
        extra_vertex_groups=False,
        feet_on_ground=True,
        scale=0.1,
        macro_detail_dict=macro,
    )
    basemesh.name = "BodyMesh"
    _apply_targets(basemesh, target_service, location_service, request.get("target_parameters") or {})
    bpy.context.view_layer.update()
    _scale_to_height(basemesh, request.get("known_height_m"))
    bpy.context.view_layer.update()

    vertices, normals, triangles = _snapshot(basemesh)
    obj_path = candidate_dir / "body.obj"
    ply_path = candidate_dir / "body_dsp.ply"
    _write_obj(obj_path, vertices, normals, triangles)
    meta_path = _write_ply(ply_path, vertices, normals, triangles)
    render_paths = _render_views(
        candidate_dir,
        basemesh,
        vertices,
        list(request.get("render_views") or []),
        int(request.get("render_size") or 512),
    )
    scene = bpy.context.scene
    scene["bodymesh_job_id"] = request["job_id"]
    scene["bodymesh_candidate_id"] = request["candidate_id"]
    scene["bodymesh_mpfb_version"] = actual_version
    blend_path = candidate_dir / "character.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), check_existing=False)
    return {
        "status": "complete",
        "blend_path": str(blend_path),
        "obj_path": str(obj_path),
        "ply_path": str(ply_path),
        "meta_path": str(meta_path),
        "render_paths": render_paths,
        "vertex_count": len(vertices),
        "face_count": len(triangles),
        "warnings": [
            "candidate uses explicit MPFB parameters; reference photos are not automatically reconstructed",
            "body_dsp.ply is neutral Blender telemetry, not a proje7-engine dump",
        ],
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

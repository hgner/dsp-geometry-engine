"""CLI-faithful Blender stand-in for body-mesh bridge tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def _args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    return parser.parse_args(values)


def _write_ply(path: Path) -> Path:
    vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    faces = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
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
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for x, y, z in vertices:
        lines.append(f"{x} {y} {z} 0 0 1 220 220 220 0 {x} {y} {z}")
    lines.extend(f"3 {a} {b} {c}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = path.with_suffix(".meta.json")
    meta.write_text(
        json.dumps(
            {
                "boneMap": {"0": "body"},
                "clipId": None,
                "sampleTime": None,
                "vertCount": 4,
                "collisionPush": None,
                "bakedCollisionPush": None,
                "imported": False,
            }
        ),
        encoding="utf-8",
    )
    return meta


def main() -> int:
    args = _args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    candidate = Path(request["candidate_dir"])
    candidate.mkdir(parents=True, exist_ok=True)
    if request.get("glb_path"):
        glb = Path(request["glb_path"])
        glb.write_bytes(b"glTF-stub-engine-character")
        Path(args.result).write_text(
            json.dumps(
                {
                    "status": "complete",
                    "glb_path": str(glb),
                    "rig_sex": request["rig_sex"],
                    "bone_count": 55,
                    "rest_pose": "arms-down",
                    "vertex_count": 13_380,
                    "max_influences": 1,
                    "mapped_vertices": 13_380,
                    "source_side_mapping": "swapped-after-z-flip",
                    "side_locality_pairs": 23,
                    "source_height_m": 1.75,
                    "warnings": ["stub engine retarget"],
                }
            ),
            encoding="utf-8",
        )
        print("stub Blender engine retarget complete")
        return 0
    blend = candidate / "character.blend"
    obj = candidate / "body.obj"
    ply = candidate / "body_dsp.ply"
    blend.write_bytes(b"BLENDER-stub")
    obj.write_text("v 0 0 0\n", encoding="utf-8")
    meta = _write_ply(ply)
    render_dir = candidate / "renders"
    render_dir.mkdir(exist_ok=True)
    renders = {}
    size = int(request["render_size"])
    for view in request["render_views"]:
        output = render_dir / f"{view}_mask.png"
        image = Image.new("RGB", (size, size), "black")
        draw = ImageDraw.Draw(image)
        draw.ellipse((size // 3, size // 16, size * 2 // 3, size * 15 // 16), fill="white")
        image.save(output)
        renders[view] = str(output)
    Path(args.result).write_text(
        json.dumps(
            {
                "status": "complete",
                "blend_path": str(blend),
                "obj_path": str(obj),
                "ply_path": str(ply),
                "meta_path": str(meta),
                "render_paths": renders,
                "vertex_count": 4,
                "face_count": 4,
                "warnings": ["stub Blender"],
            }
        ),
        encoding="utf-8",
    )
    print("stub Blender body-mesh job complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

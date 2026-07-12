"""Stand-in for character_bake_cli used by body-mesh bridge tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _local_bind(bones: list[dict], index: int) -> list[float]:
    position = [float(value) for value in bones[index]["pos"]]
    parent = int(bones[index]["parent"])
    if parent < 0:
        return position
    parent_position = [float(value) for value in bones[parent]["pos"]]
    return [position[axis] - parent_position[axis] for axis in range(3)]


def _inverse_bind(bone: dict) -> list[float]:
    x, y, z = (float(value) for value in bone["pos"])
    return [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        -x,
        -y,
        -z,
        1.0,
    ]


def main() -> int:
    glb_path = Path(sys.argv[1]).resolve()
    request = json.loads((glb_path.parent / "engine_retarget.request.json").read_text(encoding="utf-8"))
    bones = json.loads(Path(request["skeleton_path"]).read_text(encoding="utf-8"))
    baked_bones = [
        {
            "name": bone["name"],
            "parentIndex": int(bone["parent"]),
            "role": bone["role"],
            "bind": {
                "t": _local_bind(bones, index),
                "r": [0.0, 0.0, 0.0, 1.0],
                "s": [1.0, 1.0, 1.0],
            },
        }
        for index, bone in enumerate(bones)
    ]
    weighted_bones = [
        index for index, bone in enumerate(bones) if bone["name"] not in {"nose", "jaw", "eyeL", "eyeR"}
    ]
    vertices = [
        {
            "position": [float(value) for value in bones[bone_index]["pos"]],
            "normal": [0.0, 0.0, 1.0],
            "texCoord": [0.0, 0.0],
            "tangent": [1.0, 0.0, 0.0, 1.0],
            "jointIndices": [bone_index, 0, 0, 0],
            "jointWeights": [1.0, 0.0, 0.0, 0.0],
        }
        for bone_index in weighted_bones
    ]
    pelvis_bind = baked_bones[0]["bind"]
    clip_transform = {
        "t": pelvis_bind["t"],
        "r": pelvis_bind["r"],
        "s": pelvis_bind["s"],
    }
    payload = {
        "version": 1,
        "mesh": {"vertices": vertices, "indices": list(range(len(vertices)))},
        "skeleton": {
            "skeletonId": "engine-stub",
            "kind": "FullBody",
            "bones": baked_bones,
            "sockets": [],
            "socketBindings": [],
        },
        "inverseBind": [_inverse_bind(bone) for bone in bones],
        "clips": [
            {
                "clipId": "clip-stub",
                "duration": 1.0,
                "loopMode": "Wrap",
                "tracks": [
                    {
                        "boneName": "pelvis",
                        "keyframes": [
                            {"time": 0.0, "transform": clip_transform},
                            {"time": 1.0, "transform": clip_transform},
                        ],
                    }
                ],
            }
        ],
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validation for the geometry engine's 55-bone character contract.

The Blender worker is intentionally kept separate from this module.  Everything
here uses only the Python standard library, so a baked character can be checked
without Blender, MPFB, NumPy, or the engine being available.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

type Sex = Literal["male", "female"]
type JsonSource = str | os.PathLike[str] | Mapping[str, object]
type Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]

_BONE_COUNT = 55
_WEIGHT_SUM_TOLERANCE = 1.0e-3
_TANGENT_HANDEDNESS_TOLERANCE = 1.0e-3
_TANGENT_LENGTH_SQUARED_MIN = 1.0e-12
_ARMS_DOWN_HEIGHT_FRACTION = 0.10
_ARMS_DOWN_MIN_VERTICAL_COMPONENT = 0.80
_INVERSE_BIND_TOLERANCE = 1.0e-4
_TIME_TOLERANCE = 1.0e-6
_SIDE_PAIR_BASES = (
    "clavicle",
    "armUpper",
    "armLower",
    "hand",
    "legUpper",
    "legLower",
    "foot",
    "toe",
)
_FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# This is the exact order used by scripts/blender/skeleton_{sex}.json and by the
# engine's kDetailedBones table.  Roles are deliberately identical to names.
_CANONICAL_NAMES = (
    "pelvis",
    "spine",
    "chest",
    "neck",
    "head",
    "nose",
    "clavicleL",
    "armUpperL",
    "armLowerL",
    "handL",
    "clavicleR",
    "armUpperR",
    "armLowerR",
    "handR",
    "legUpperL",
    "legLowerL",
    "footL",
    "toeL",
    "legUpperR",
    "legLowerR",
    "footR",
    "toeR",
    "thumbL",
    "indexL",
    "middleL",
    "ringL",
    "pinkyL",
    "thumbR",
    "indexR",
    "middleR",
    "ringR",
    "pinkyR",
    "jaw",
    "eyeL",
    "eyeR",
    "thumbL2",
    "indexL2",
    "middleL2",
    "ringL2",
    "pinkyL2",
    "thumbR2",
    "indexR2",
    "middleR2",
    "ringR2",
    "pinkyR2",
    "thumbL3",
    "indexL3",
    "middleL3",
    "ringL3",
    "pinkyL3",
    "thumbR3",
    "indexR3",
    "middleR3",
    "ringR3",
    "pinkyR3",
)

_CANONICAL_PARENTS = (
    -1,
    0,
    1,
    2,
    3,
    4,
    2,
    6,
    7,
    8,
    2,
    10,
    11,
    12,
    0,
    14,
    15,
    16,
    0,
    18,
    19,
    20,
    9,
    9,
    9,
    9,
    9,
    13,
    13,
    13,
    13,
    13,
    4,
    4,
    4,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
)


class EngineContractError(ValueError):
    """Raised when a skeleton or baked character violates the engine contract."""


@dataclass(frozen=True, slots=True)
class EngineBone:
    """One validated canonical bone from an authoritative skeleton file."""

    name: str
    parent: int
    role: str
    pos: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class EngineSkeleton:
    """A validated sex-specific engine skeleton and its source fingerprint."""

    sex: Sex
    bones: tuple[EngineBone, ...]
    sha256: str


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EngineContractError(f"{path} must be an object")
    return value


def _array(value: object, length: int | None, path: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise EngineContractError(f"{path} must be an array")
    if length is not None and len(value) != length:
        raise EngineContractError(f"{path} must contain exactly {length} values")
    return value


def _finite_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EngineContractError(f"{path} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise EngineContractError(f"{path} must be a finite number")
    return number


def _finite_vector(value: object, length: int, path: str) -> tuple[float, ...]:
    items = _array(value, length, path)
    return tuple(_finite_float(item, f"{path}[{index}]") for index, item in enumerate(items))


def _read_json_object(path: Path) -> tuple[Mapping[str, object], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EngineContractError(f"could not read JSON from {path}: {exc}") from exc
    return _mapping(value, str(path)), raw


def load_engine_skeleton(path: str | os.PathLike[str], sex: str) -> EngineSkeleton:
    """Load and strictly validate one authoritative ``skeleton_{sex}.json``.

    Bone positions differ between the male and female assets, while the ordered
    names, parent indices, and roles are a single immutable engine contract.
    """

    if sex not in ("male", "female"):
        raise EngineContractError("sex must be 'male' or 'female'")

    skeleton_path = Path(path)
    try:
        raw = skeleton_path.read_bytes()
        value = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EngineContractError(f"could not read skeleton JSON from {skeleton_path}: {exc}") from exc

    entries = _array(value, _BONE_COUNT, str(skeleton_path))
    bones: list[EngineBone] = []
    names: set[str] = set()
    roles: set[str] = set()

    for index, value in enumerate(entries):
        item_path = f"{skeleton_path}[{index}]"
        entry = _mapping(value, item_path)
        expected_name = _CANONICAL_NAMES[index]
        expected_parent = _CANONICAL_PARENTS[index]

        name = entry.get("name")
        parent = entry.get("parent")
        role = entry.get("role")
        if name != expected_name:
            raise EngineContractError(f"{item_path}.name must be {expected_name!r}, got {name!r}")
        if not _is_int(parent) or parent != expected_parent:
            raise EngineContractError(f"{item_path}.parent must be {expected_parent}, got {parent!r}")
        if role != expected_name:
            raise EngineContractError(f"{item_path}.role must be {expected_name!r}, got {role!r}")
        if parent >= index:
            raise EngineContractError(f"{item_path}.parent must precede its child")
        if name in names:
            raise EngineContractError(f"{item_path}.name duplicates {name!r}")
        if role in roles:
            raise EngineContractError(f"{item_path}.role duplicates {role!r}")

        pos = _finite_vector(entry.get("pos"), 3, f"{item_path}.pos")
        names.add(name)
        roles.add(role)
        bones.append(
            EngineBone(
                name=name,
                parent=parent,
                role=role,
                pos=(pos[0], pos[1], pos[2]),
            )
        )

    return EngineSkeleton(
        sex=sex,
        bones=tuple(bones),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _compose_trs(
    translation: tuple[float, ...],
    rotation: tuple[float, ...],
    scale: tuple[float, ...],
    path: str,
) -> Matrix4:
    x, y, z, w = rotation
    quaternion_length_squared = x * x + y * y + z * z + w * w
    if quaternion_length_squared <= 1.0e-12:
        raise EngineContractError(f"{path}.r must be a nondegenerate quaternion")
    inverse_quaternion_length = 1.0 / math.sqrt(quaternion_length_squared)
    x *= inverse_quaternion_length
    y *= inverse_quaternion_length
    z *= inverse_quaternion_length
    w *= inverse_quaternion_length

    sx, sy, sz = scale
    tx, ty, tz = translation
    return (
        (
            (1.0 - 2.0 * (y * y + z * z)) * sx,
            (2.0 * (x * y - z * w)) * sy,
            (2.0 * (x * z + y * w)) * sz,
            tx,
        ),
        (
            (2.0 * (x * y + z * w)) * sx,
            (1.0 - 2.0 * (x * x + z * z)) * sy,
            (2.0 * (y * z - x * w)) * sz,
            ty,
        ),
        (
            (2.0 * (x * z - y * w)) * sx,
            (2.0 * (y * z + x * w)) * sy,
            (1.0 - 2.0 * (x * x + y * y)) * sz,
            tz,
        ),
        (0.0, 0.0, 0.0, 1.0),
    )


def _matrix_multiply(left: Matrix4, right: Matrix4) -> Matrix4:
    return tuple(
        tuple(sum(left[row][inner] * right[inner][column] for inner in range(4)) for column in range(4))
        for row in range(4)
    )  # type: ignore[return-value]


def _load_baked_source(source: JsonSource) -> tuple[Mapping[str, object], str]:
    if isinstance(source, Mapping):
        return source, "$"
    path = Path(source)
    document, _ = _read_json_object(path)
    return document, str(path)


def _validate_baked_skeleton(
    document: Mapping[str, object],
    skeleton: EngineSkeleton,
    root_path: str,
) -> tuple[list[Matrix4], dict[str, int]]:
    skeleton_doc = _mapping(document.get("skeleton"), f"{root_path}.skeleton")
    entries = _array(skeleton_doc.get("bones"), _BONE_COUNT, f"{root_path}.skeleton.bones")
    canonical_by_name = {bone.name: bone for bone in skeleton.bones}
    index_by_name: dict[str, int] = {}
    parent_indices: list[int] = []
    local_matrices: list[Matrix4] = []

    for index, value in enumerate(entries):
        item_path = f"{root_path}.skeleton.bones[{index}]"
        entry = _mapping(value, item_path)
        name = entry.get("name")
        role = entry.get("role")
        parent_index = entry.get("parentIndex")

        if not isinstance(name, str) or name not in canonical_by_name:
            raise EngineContractError(f"{item_path}.name is not a canonical engine bone: {name!r}")
        if name in index_by_name:
            raise EngineContractError(f"{item_path}.name duplicates {name!r}")
        if role != canonical_by_name[name].role:
            raise EngineContractError(
                f"{item_path}.role must be {canonical_by_name[name].role!r}, got {role!r}"
            )
        if not _is_int(parent_index) or parent_index < -1 or parent_index >= index:
            raise EngineContractError(f"{item_path}.parentIndex must be -1 or precede its child")

        bind = _mapping(entry.get("bind"), f"{item_path}.bind")
        translation = _finite_vector(bind.get("t"), 3, f"{item_path}.bind.t")
        rotation = _finite_vector(bind.get("r"), 4, f"{item_path}.bind.r")
        scale = _finite_vector(bind.get("s"), 3, f"{item_path}.bind.s")

        index_by_name[name] = index
        parent_indices.append(parent_index)
        local_matrices.append(_compose_trs(translation, rotation, scale, f"{item_path}.bind"))

    missing = set(canonical_by_name) - set(index_by_name)
    if missing:
        raise EngineContractError(f"{root_path}.skeleton.bones is missing: {', '.join(sorted(missing))}")

    for name, index in index_by_name.items():
        actual_parent_index = parent_indices[index]
        canonical_parent_index = canonical_by_name[name].parent
        expected_parent_name = (
            None if canonical_parent_index == -1 else skeleton.bones[canonical_parent_index].name
        )
        actual_parent_name = None if actual_parent_index == -1 else entries[actual_parent_index].get("name")
        if actual_parent_name != expected_parent_name:
            raise EngineContractError(
                f"{root_path}.skeleton.bones[{index}] ({name!r}) must have parent "
                f"{expected_parent_name!r}, got {actual_parent_name!r}"
            )

    world_matrices: list[Matrix4] = []
    for index, local_matrix in enumerate(local_matrices):
        parent_index = parent_indices[index]
        world_matrix = (
            local_matrix
            if parent_index == -1
            else _matrix_multiply(world_matrices[parent_index], local_matrix)
        )
        if not all(math.isfinite(value) for row in world_matrix for value in row):
            raise EngineContractError(
                f"{root_path}.skeleton.bones[{index}].bind produces a non-finite world transform"
            )
        world_matrices.append(world_matrix)

    return world_matrices, index_by_name


def _validate_inverse_bind(
    document: Mapping[str, object], world_matrices: Sequence[Matrix4], root_path: str
) -> float:
    matrices = _array(document.get("inverseBind"), _BONE_COUNT, f"{root_path}.inverseBind")
    maximum_residual = 0.0
    for index, (value, world_matrix) in enumerate(zip(matrices, world_matrices, strict=True)):
        flat = _finite_vector(value, 16, f"{root_path}.inverseBind[{index}]")
        # character_bake_cli serializes Mat4.elements in glTF/column-major order.
        inverse_matrix: Matrix4 = tuple(
            tuple(flat[column * 4 + row] for column in range(4)) for row in range(4)
        )  # type: ignore[assignment]
        products = (
            _matrix_multiply(world_matrix, inverse_matrix),
            _matrix_multiply(inverse_matrix, world_matrix),
        )
        residual = max(
            abs(product[row][column] - (1.0 if row == column else 0.0))
            for product in products
            for row in range(4)
            for column in range(4)
        )
        maximum_residual = max(maximum_residual, residual)
        if residual > _INVERSE_BIND_TOLERANCE:
            raise EngineContractError(
                f"{root_path}.inverseBind[{index}] is not the inverse of its bind-world "
                f"matrix (maximum residual {residual:.6g})"
            )
    return maximum_residual


def _validate_mesh(
    document: Mapping[str, object], root_path: str
) -> tuple[int, int, int, set[int], dict[int, tuple[float, float, float]]]:
    mesh = _mapping(document.get("mesh"), f"{root_path}.mesh")
    vertices = _array(mesh.get("vertices"), None, f"{root_path}.mesh.vertices")
    if not vertices:
        raise EngineContractError(f"{root_path}.mesh.vertices must not be empty")

    max_influences = 0
    weighted_joint_indices: set[int] = set()
    weighted_position_sums: dict[int, list[float]] = {}
    joint_weight_totals: dict[int, float] = {}
    for index, value in enumerate(vertices):
        item_path = f"{root_path}.mesh.vertices[{index}]"
        vertex = _mapping(value, item_path)
        position = _finite_vector(vertex.get("position"), 3, f"{item_path}.position")
        _finite_vector(vertex.get("normal"), 3, f"{item_path}.normal")
        _finite_vector(vertex.get("texCoord"), 2, f"{item_path}.texCoord")
        tangent = _finite_vector(vertex.get("tangent"), 4, f"{item_path}.tangent")
        tangent_length_squared = sum(component * component for component in tangent[:3])
        if tangent_length_squared <= _TANGENT_LENGTH_SQUARED_MIN:
            raise EngineContractError(f"{item_path}.tangent.xyz must be nondegenerate")
        if abs(abs(tangent[3]) - 1.0) > _TANGENT_HANDEDNESS_TOLERANCE:
            raise EngineContractError(f"{item_path}.tangent.w must be near -1 or +1")

        joints = _array(vertex.get("jointIndices"), 4, f"{item_path}.jointIndices")
        for influence, joint in enumerate(joints):
            if not _is_int(joint) or not 0 <= joint < _BONE_COUNT:
                raise EngineContractError(
                    f"{item_path}.jointIndices[{influence}] must be an integer in [0, 54]"
                )

        weights = _finite_vector(vertex.get("jointWeights"), 4, f"{item_path}.jointWeights")
        if any(weight < 0.0 for weight in weights):
            raise EngineContractError(f"{item_path}.jointWeights must be nonnegative")
        if abs(sum(weights) - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise EngineContractError(
                f"{item_path}.jointWeights must sum to 1 within {_WEIGHT_SUM_TOLERANCE:g}"
            )
        max_influences = max(max_influences, sum(weight > 0.0 for weight in weights))
        weighted_joint_indices.update(
            int(joint) for joint, weight in zip(joints, weights, strict=True) if weight > 0.0
        )
        for joint, weight in zip(joints, weights, strict=True):
            if weight <= 0.0:
                continue
            joint_index = int(joint)
            total = joint_weight_totals.get(joint_index, 0.0) + weight
            joint_weight_totals[joint_index] = total
            accumulated = weighted_position_sums.setdefault(joint_index, [0.0, 0.0, 0.0])
            for axis in range(3):
                accumulated[axis] += position[axis] * weight

    indices = _array(mesh.get("indices"), None, f"{root_path}.mesh.indices")
    if not indices or len(indices) % 3 != 0:
        raise EngineContractError(f"{root_path}.mesh.indices must contain complete triangles")
    for index, vertex_index in enumerate(indices):
        if not _is_int(vertex_index) or not 0 <= vertex_index < len(vertices):
            raise EngineContractError(f"{root_path}.mesh.indices[{index}] must reference an existing vertex")

    centroids = {
        joint: tuple(component / joint_weight_totals[joint] for component in accumulated)
        for joint, accumulated in weighted_position_sums.items()
    }
    return len(vertices), len(indices) // 3, max_influences, weighted_joint_indices, centroids


def _side_pairs() -> tuple[tuple[str, str], ...]:
    pairs = [(f"{base}L", f"{base}R") for base in _SIDE_PAIR_BASES]
    for finger in _FINGERS:
        for suffix in ("", "2", "3"):
            pairs.append((f"{finger}L{suffix}", f"{finger}R{suffix}"))
    return tuple(pairs)


def _validate_skin_locality(
    world_matrices: Sequence[Matrix4],
    index_by_name: Mapping[str, int],
    weighted_centroids: Mapping[int, tuple[float, float, float]],
    root_path: str,
) -> int:
    checked = 0
    for left, right in _side_pairs():
        left_index = index_by_name[left]
        right_index = index_by_name[right]
        try:
            left_centroid = weighted_centroids[left_index]
            right_centroid = weighted_centroids[right_index]
        except KeyError as exc:
            raise EngineContractError(
                f"{root_path}.mesh has no weighted centroid for {left}/{right}"
            ) from exc
        bone_delta = world_matrices[left_index][0][3] - world_matrices[right_index][0][3]
        centroid_delta = left_centroid[0] - right_centroid[0]
        if abs(bone_delta) <= 1.0e-6 or bone_delta * centroid_delta <= 0.0:
            raise EngineContractError(
                f"{root_path}.mesh skin locality for {left}/{right} is crossed: "
                f"bind delta X={bone_delta:.6g}, weighted-centroid delta X={centroid_delta:.6g}"
            )
        checked += 1
    return checked


def _validate_arms_down(
    world_matrices: Sequence[Matrix4],
    index_by_name: Mapping[str, int],
    root_path: str,
) -> None:
    world_y = [matrix[1][3] for matrix in world_matrices]
    character_height = max(world_y) - min(world_y)
    if character_height <= 0.0:
        raise EngineContractError(f"{root_path}.skeleton bind pose has no vertical extent")
    minimum_drop = max(character_height * _ARMS_DOWN_HEIGHT_FRACTION, 1.0e-6)

    for side in ("L", "R"):
        shoulder_matrix = world_matrices[index_by_name[f"armUpper{side}"]]
        shoulder = tuple(shoulder_matrix[axis][3] for axis in range(3))
        shoulder_y = shoulder[1]
        hand_y = world_matrices[index_by_name[f"hand{side}"]][1][3]
        if shoulder_y - hand_y < minimum_drop:
            raise EngineContractError(
                f"{root_path}.skeleton bind pose is not arms-down: hand{side} must be "
                f"materially below armUpper{side}"
            )
        for target_name in (f"armLower{side}", f"hand{side}"):
            target_matrix = world_matrices[index_by_name[target_name]]
            direction = tuple(target_matrix[axis][3] - shoulder[axis] for axis in range(3))
            length = math.sqrt(sum(component * component for component in direction))
            if length <= 1.0e-8 or direction[1] / length > -_ARMS_DOWN_MIN_VERTICAL_COMPONENT:
                raise EngineContractError(
                    f"{root_path}.skeleton bind pose is not arms-down: armUpper{side} to "
                    f"{target_name} must point predominantly downward"
                )


def _validate_clips(
    document: Mapping[str, object], index_by_name: Mapping[str, int], root_path: str
) -> tuple[int, int, int]:
    clips = _array(document.get("clips"), None, f"{root_path}.clips")
    if not clips:
        raise EngineContractError(f"{root_path}.clips must not be empty")

    clip_ids: set[str] = set()
    track_count = 0
    keyframe_count = 0
    for clip_index, value in enumerate(clips):
        clip_path = f"{root_path}.clips[{clip_index}]"
        clip = _mapping(value, clip_path)
        clip_id = clip.get("clipId")
        if not isinstance(clip_id, str) or not clip_id.strip():
            raise EngineContractError(f"{clip_path}.clipId must be a nonempty string")
        if clip_id in clip_ids:
            raise EngineContractError(f"{clip_path}.clipId duplicates {clip_id!r}")
        clip_ids.add(clip_id)
        duration = _finite_float(clip.get("duration"), f"{clip_path}.duration")
        if duration <= 0.0:
            raise EngineContractError(f"{clip_path}.duration must be positive")
        loop_mode = clip.get("loopMode")
        if loop_mode not in {"Clamp", "Wrap"}:
            raise EngineContractError(f"{clip_path}.loopMode must be 'Clamp' or 'Wrap'")
        tracks = _array(clip.get("tracks"), None, f"{clip_path}.tracks")
        tracked_bones: set[str] = set()
        for track_index, track_value in enumerate(tracks):
            track_path = f"{clip_path}.tracks[{track_index}]"
            track = _mapping(track_value, track_path)
            bone_name = track.get("boneName")
            if not isinstance(bone_name, str) or bone_name not in index_by_name:
                raise EngineContractError(
                    f"{track_path}.boneName is not a baked skeleton bone: {bone_name!r}"
                )
            if bone_name in tracked_bones:
                raise EngineContractError(f"{track_path}.boneName duplicates {bone_name!r}")
            tracked_bones.add(bone_name)
            keyframes = _array(track.get("keyframes"), None, f"{track_path}.keyframes")
            if not keyframes:
                raise EngineContractError(f"{track_path}.keyframes must not be empty")
            previous_time = -math.inf
            for keyframe_index, keyframe_value in enumerate(keyframes):
                keyframe_path = f"{track_path}.keyframes[{keyframe_index}]"
                keyframe = _mapping(keyframe_value, keyframe_path)
                keyframe_time = _finite_float(keyframe.get("time"), f"{keyframe_path}.time")
                if keyframe_time < -_TIME_TOLERANCE or keyframe_time > duration + _TIME_TOLERANCE:
                    raise EngineContractError(
                        f"{keyframe_path}.time must be within clip duration [0, {duration:g}]"
                    )
                if keyframe_time + _TIME_TOLERANCE < previous_time:
                    raise EngineContractError(f"{track_path}.keyframes must be time-ordered")
                previous_time = keyframe_time
                transform = _mapping(keyframe.get("transform"), f"{keyframe_path}.transform")
                translation = _finite_vector(transform.get("t"), 3, f"{keyframe_path}.transform.t")
                rotation = _finite_vector(transform.get("r"), 4, f"{keyframe_path}.transform.r")
                scale = _finite_vector(transform.get("s"), 3, f"{keyframe_path}.transform.s")
                _compose_trs(translation, rotation, scale, f"{keyframe_path}.transform")
                keyframe_count += 1
            track_count += 1
    if track_count == 0:
        raise EngineContractError(f"{root_path}.clips must contain at least one animated skeleton track")
    return len(clips), track_count, keyframe_count


def validate_baked_character(source: JsonSource, skeleton: EngineSkeleton) -> dict[str, int | float | str]:
    """Validate a ``character_bake_cli`` JSON document against ``skeleton``.

    Blender/glTF may reorder bones into a parent-first depth-first sequence.  The
    validator therefore compares the exact canonical name, role, and named-parent
    relationship rather than requiring the baked array to retain source-file order.
    """

    document, root_path = _load_baked_source(source)
    version = document.get("version")
    if not _is_int(version) or version != 1:
        raise EngineContractError(f"{root_path}.version must be 1")

    world_matrices, index_by_name = _validate_baked_skeleton(document, skeleton, root_path)
    inverse_bind_max_residual = _validate_inverse_bind(document, world_matrices, root_path)
    vertex_count, face_count, max_influences, weighted_joint_indices, weighted_centroids = _validate_mesh(
        document, root_path
    )
    weighted_names = {name for name, index in index_by_name.items() if index in weighted_joint_indices}
    expected_weighted = set(_CANONICAL_NAMES) - {"nose", "jaw", "eyeL", "eyeR"}
    missing_weights = expected_weighted - weighted_names
    if missing_weights:
        raise EngineContractError(
            f"{root_path}.mesh is missing skin influence for: {', '.join(sorted(missing_weights))}"
        )
    _validate_arms_down(world_matrices, index_by_name, root_path)
    skin_locality_pair_count = _validate_skin_locality(
        world_matrices, index_by_name, weighted_centroids, root_path
    )
    clip_count, track_count, keyframe_count = _validate_clips(document, index_by_name, root_path)

    return {
        "bone_count": _BONE_COUNT,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "max_influences": max_influences,
        "weighted_bone_count": len(weighted_names),
        "skin_locality_pair_count": skin_locality_pair_count,
        "inverse_bind_max_residual": inverse_bind_max_residual,
        "clip_count": clip_count,
        "track_count": track_count,
        "keyframe_count": keyframe_count,
        "rest_pose": "arms-down",
        "skeleton_sha256": skeleton.sha256,
    }


def validate_engine_contract(
    baked_path: str | os.PathLike[str],
    skeleton_path: str | os.PathLike[str],
    *,
    sex: str,
) -> dict[str, int | float | str]:
    """Load both artifacts and return a compact validated contract summary."""

    skeleton = load_engine_skeleton(skeleton_path, sex)
    return validate_baked_character(baked_path, skeleton)

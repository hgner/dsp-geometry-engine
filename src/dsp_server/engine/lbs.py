"""Pure-numpy linear-blend skinning against the engine's baked-character JSON.

Conventions (verified against proje7-engine sources):

* ``Mat4`` (Mat4.hpp) is COLUMN-major: flat ``elements[column*4 + row]``, translation
  at [12..14]. A 16-float JSON array therefore becomes a standard math matrix via
  ``np.asarray(vals, float).reshape(4, 4, order="F")`` — after that ``M @ [p, 1]``
  reproduces the engine's ``skinning(row, col)`` arithmetic verbatim.
* ``skeleton.bones[i].bind`` (BakedCharacterJson.cpp) is a local TRS with
  ``t=[3]``, ``r=[x, y, z, w]`` quaternion, ``s=[3]``; the local bind matrix is
  ``T @ R @ S`` (scale, then rotate, then translate).
* World bind is plain FK: ``world[i] = world[parent[i]] @ bind_local[i]`` (root has
  parent -1). The self-check ``inv(world_bind) ≈ inverseBind`` validates both the
  quaternion order and the matrix majority against the asset itself.
* LBS (VertexSkinning.cpp): ``p' = (Σₖ wₖ·palette[jₖ]) @ [p, 1]`` accumulated in
  float32 to match the engine. Zero-weight influences are skipped by the engine, so
  their joint index is never dereferenced — mirrored here (only a nonzero-weight
  out-of-range index raises).
* Units: the raw baked JSON is PRE-import (authoring units, ~cm); engine dumps are
  POST-import (meters). ``align_json_to_engine`` applies the uniform import scale
  ``s`` to vertex positions, bind translations, and the inverseBind translation
  components ([12..14]) ONLY — rotations are untouched. This is exact for a uniform
  scale: if every bind translation scales by ``s`` and rotations are unchanged, the
  FK world translation scales by ``s`` and ``inv(world)``'s translation
  ``-Rᵀ·(s·t)`` scales by ``s`` as well.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

WEIGHT_SURGERY_MODES = ("none", "top1", "drop_hand", "drop_fingers")
FINGER_NAME_TOKENS = ("thumb", "index", "middle", "ring", "pinky")


# --------------------------------------------------------------------------- #
# Small linear-algebra helpers (quat is glTF/engine x, y, z, w order).


def quat_to_mat(q_xyzw: ArrayLike) -> np.ndarray:
    """Unit quaternion (x, y, z, w) -> (3,3) float64 rotation matrix."""
    q = np.asarray(q_xyzw, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"quaternion must be 4 floats (x, y, z, w), got shape {q.shape}")
    x, y, z, w = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def compose_trs(t: ArrayLike, q_xyzw: ArrayLike, s: ArrayLike) -> np.ndarray:
    """Local transform T @ R @ S as a (4,4) float64 matrix (the bind convention)."""
    scale = np.asarray(s, dtype=np.float64)
    if scale.ndim == 0:
        scale = np.full(3, float(scale))
    if scale.shape != (3,):
        raise ValueError(f"scale must be 3 floats, got shape {scale.shape}")
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = quat_to_mat(q_xyzw) * scale[None, :]  # R @ diag(s): scale the columns
    m[:3, 3] = np.asarray(t, dtype=np.float64)
    return m


def _mat16(values: ArrayLike, layout: str = "column-major") -> np.ndarray:
    """Flat 16-float engine matrix -> (4,4) float64 standard math matrix."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (16,):
        raise ValueError(f"expected 16 matrix elements, got shape {arr.shape}")
    order = "C" if layout == "row-major" else "F"
    return arr.reshape(4, 4, order=order)


# --------------------------------------------------------------------------- #
# Baked-character JSON (the PRE-import asset on disk, e.g. cc0_male_rigged3).


@dataclass
class BakedCharacterData:
    """The skinning-relevant subset of one baked-character JSON document."""

    positions: np.ndarray  # (n,3) float32
    normals: np.ndarray  # (n,3) float32
    joint_indices: np.ndarray  # (n,4) int32
    joint_weights: np.ndarray  # (n,4) float32
    indices: np.ndarray  # (m,) int32 flat triangle list
    bone_names: list[str]
    parent_index: np.ndarray  # (b,) int32, root = -1
    bind_local: np.ndarray  # (b,4,4) float64, T @ R @ S per bone
    inverse_bind: np.ndarray  # (b,4,4) float64 from the top-level "inverseBind"
    path: Path | None = None

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def bone_count(self) -> int:
        return len(self.bone_names)


def _parse_baked_json(path: Path) -> BakedCharacterData:
    with path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)

    verts = doc["mesh"]["vertices"]
    if verts:
        positions = np.asarray([v["position"] for v in verts], dtype=np.float32)
        normals = np.asarray([v["normal"] for v in verts], dtype=np.float32)
        joint_indices = np.asarray([v["jointIndices"] for v in verts], dtype=np.int32)
        joint_weights = np.asarray([v["jointWeights"] for v in verts], dtype=np.float32)
    else:
        positions = np.zeros((0, 3), dtype=np.float32)
        normals = np.zeros((0, 3), dtype=np.float32)
        joint_indices = np.zeros((0, 4), dtype=np.int32)
        joint_weights = np.zeros((0, 4), dtype=np.float32)
    indices = np.asarray(doc["mesh"]["indices"], dtype=np.int32).reshape(-1)

    bones = doc["skeleton"]["bones"]
    bone_names = [str(b["name"]) for b in bones]
    parent_index = np.asarray([int(b["parentIndex"]) for b in bones], dtype=np.int32)
    if bones:
        bind_local = np.stack([compose_trs(b["bind"]["t"], b["bind"]["r"], b["bind"]["s"]) for b in bones])
        inverse_bind = np.stack([_mat16(m) for m in doc["inverseBind"]])
    else:
        bind_local = np.zeros((0, 4, 4), dtype=np.float64)
        inverse_bind = np.zeros((0, 4, 4), dtype=np.float64)
    if inverse_bind.shape[0] != len(bones):
        raise ValueError(f"{path}: inverseBind has {inverse_bind.shape[0]} matrices for {len(bones)} bones")

    return BakedCharacterData(
        positions=positions,
        normals=normals,
        joint_indices=joint_indices,
        joint_weights=joint_weights,
        indices=indices,
        bone_names=bone_names,
        parent_index=parent_index,
        bind_local=bind_local,
        inverse_bind=inverse_bind,
        path=path,
    )


def _cache_key(path: Path) -> str:
    """Mirror of ply._cache_key: sha1 over (resolved path, mtime_ns, size)."""
    st = path.stat()
    return hashlib.sha1(f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()


def load_baked_character(path: Path | str, cache_dir: Path | None = None) -> BakedCharacterData:
    """Parse a baked-character JSON; with ``cache_dir`` the arrays round-trip through
    an .npz keyed on (path, mtime, size) so the 60 MB rigged3 asset parses once."""
    path = Path(path)
    if cache_dir is None:
        return _parse_baked_json(path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"baked_{_cache_key(path)}.npz"
    if npz_path.is_file():
        with np.load(npz_path, allow_pickle=False) as z:
            return BakedCharacterData(
                positions=z["positions"],
                normals=z["normals"],
                joint_indices=z["joint_indices"],
                joint_weights=z["joint_weights"],
                indices=z["indices"],
                bone_names=[str(name) for name in z["bone_names"]],
                parent_index=z["parent_index"],
                bind_local=z["bind_local"],
                inverse_bind=z["inverse_bind"],
                path=path,
            )

    data = _parse_baked_json(path)
    np.savez_compressed(
        npz_path,
        positions=data.positions,
        normals=data.normals,
        joint_indices=data.joint_indices,
        joint_weights=data.joint_weights,
        indices=data.indices,
        bone_names=np.asarray(data.bone_names, dtype=np.str_),
        parent_index=data.parent_index,
        bind_local=data.bind_local,
        inverse_bind=data.inverse_bind,
    )
    return data


# --------------------------------------------------------------------------- #
# Forward kinematics + the convention self-check.


def compose_world_bind(data: BakedCharacterData) -> np.ndarray:
    """FK over the parent chain: world[i] = world[parent[i]] @ bind_local[i]."""
    b = data.bind_local.shape[0]
    parents = data.parent_index
    if b and int(parents.max(initial=-1)) >= b:
        raise ValueError(f"parentIndex out of range: max {int(parents.max())} for {b} bones")
    world = np.empty_like(data.bind_local)
    resolved = np.zeros(b, dtype=bool)
    pending = list(range(b))
    while pending:  # tolerates bones listed in any order (parents need not precede children)
        remaining: list[int] = []
        for i in pending:
            p = int(parents[i])
            if p < 0:
                world[i] = data.bind_local[i]
            elif resolved[p]:
                world[i] = world[p] @ data.bind_local[i]
            else:
                remaining.append(i)
                continue
            resolved[i] = True
        if len(remaining) == len(pending):
            raise ValueError("skeleton parent graph contains a cycle (FK cannot resolve)")
        pending = remaining
    return world


def validate_against_inverse_bind(data: BakedCharacterData, atol: float = 1e-3) -> float:
    """Convention self-check: returns max |inv(world_bind) - inverseBind| and raises
    ValueError when it exceeds ``atol`` (wrong quat order / matrix majority)."""
    if data.bind_local.shape[0] == 0:
        return 0.0
    world = compose_world_bind(data)
    err = float(np.max(np.abs(np.linalg.inv(world) - data.inverse_bind)))
    if err > atol:
        raise ValueError(
            f"convention self-check failed: max |inv(world_bind) - inverseBind| = {err:.3e} "
            f"> atol={atol:.1e} (quaternion order or matrix majority is wrong for this asset)"
        )
    return err


# --------------------------------------------------------------------------- #
# Linear-blend skinning (float32 accumulation to match the engine).


def lbs_pose(
    rest_positions: np.ndarray,
    joint_indices: np.ndarray,
    joint_weights: np.ndarray,
    palette: np.ndarray,
) -> np.ndarray:
    """Vectorized engine LBS: (n,3) float32 posed positions from rest + palette (b,4,4).

    Matches VertexSkinning.cpp: zero-weight influences never dereference their joint
    index; a NONZERO-weight index outside [0, b) raises IndexError.
    """
    pal = np.asarray(palette, dtype=np.float32)
    if pal.ndim != 3 or pal.shape[1:] != (4, 4):
        raise ValueError(f"palette must be (b,4,4), got shape {pal.shape}")
    rest = np.asarray(rest_positions, dtype=np.float32)
    ji = np.asarray(joint_indices)
    jw = np.asarray(joint_weights, dtype=np.float32)
    n_bones = pal.shape[0]

    active = jw != 0.0
    bad = active & ((ji < 0) | (ji >= n_bones))
    if np.any(bad):
        row = int(np.nonzero(bad.any(axis=1))[0][0])
        raise IndexError(
            f"joint index out of range for a {n_bones}-bone palette at {int(bad.sum())} "
            f"influence slot(s); first offending vertex row {row}: indices {ji[row].tolist()} "
            f"weights {jw[row].tolist()}"
        )
    safe = np.where(active, ji, 0).astype(np.int64)

    blended = (pal[safe] * jw[..., None, None]).sum(axis=1, dtype=np.float32)  # (n,4,4)
    out = np.einsum("nij,nj->ni", blended[:, :3, :3], rest) + blended[:, :3, 3]
    return out.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Palette sidecar (schema = the M5 --palette-out writer).


@dataclass
class PaletteSidecar:
    """One `<dump>.palette.json` sidecar; per-bone arrays are ordered by bone index."""

    clip_id: str
    sample_time_seconds: float
    character: str | None
    sex: str | None
    matrix_layout: str
    import_scale: float  # centimeterImportScaleFactor; 1.0 when the key is absent
    bone_names: list[str]
    parent_index: np.ndarray  # (b,) int32, root = -1
    skinning: np.ndarray  # (b,4,4) float64 = worldPose @ inverseBind
    world: np.ndarray | None = None  # (b,4,4) float64 (optional in the schema)
    inverse_bind: np.ndarray | None = None  # (b,4,4) float64 (optional in the schema)
    path: Path | None = None

    @property
    def bone_count(self) -> int:
        return len(self.bone_names)

    @property
    def skinning_palette(self) -> np.ndarray:
        """(b,4,4) skinning matrices ordered by bone index — feed straight to lbs_pose."""
        return self.skinning


def load_palette_sidecar(path: Path | str) -> PaletteSidecar:
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    layout = str(doc.get("matrixLayout", "column-major"))
    bones = sorted(doc["bones"], key=lambda b: int(b["index"]))
    declared = int(doc.get("boneCount", len(bones)))
    if declared != len(bones):
        raise ValueError(f"{path}: boneCount={declared} but bones array has {len(bones)} entries")
    idx = [int(b["index"]) for b in bones]
    if idx != list(range(len(bones))):
        raise ValueError(f"{path}: bone indices are not a dense 0..{len(bones) - 1} range: {idx[:8]}")

    if bones:
        skinning = np.stack([_mat16(b["skinning"], layout) for b in bones])
    else:
        skinning = np.zeros((0, 4, 4), dtype=np.float64)
    world = None
    if bones and all("world" in b for b in bones):
        world = np.stack([_mat16(b["world"], layout) for b in bones])
    inverse_bind = None
    if bones and all("inverseBind" in b for b in bones):
        inverse_bind = np.stack([_mat16(b["inverseBind"], layout) for b in bones])

    return PaletteSidecar(
        clip_id=str(doc.get("clipId", "")),
        sample_time_seconds=float(doc.get("sampleTimeSeconds", 0.0)),
        character=doc.get("character"),
        sex=doc.get("sex"),
        matrix_layout=layout,
        import_scale=float(doc.get("importScale", 1.0)),
        bone_names=[str(b["name"]) for b in bones],
        parent_index=np.asarray([int(b.get("parent", -1)) for b in bones], dtype=np.int32),
        skinning=skinning,
        world=world,
        inverse_bind=inverse_bind,
        path=path,
    )


# --------------------------------------------------------------------------- #
# Unit alignment between the raw JSON (pre-import) and engine dumps (post-import).


def align_json_to_engine(data: BakedCharacterData, import_scale: float) -> BakedCharacterData:
    """Apply the engine's uniform import scale to a raw baked-character asset.

    Convention (see module docstring): scale ONLY the vertex positions, the
    bind_local translation columns, and the inverseBind translation components
    ([12..14] in engine storage = matrix[:3, 3] here). Rotations/scales untouched.
    Normals, weights, and topology are scale-invariant.
    """
    s = float(import_scale)
    bind_local = data.bind_local.copy()
    bind_local[:, :3, 3] *= s
    inverse_bind = data.inverse_bind.copy()
    inverse_bind[:, :3, 3] *= s
    return BakedCharacterData(
        positions=(data.positions * np.float32(s)).astype(np.float32),
        normals=data.normals.copy(),
        joint_indices=data.joint_indices.copy(),
        joint_weights=data.joint_weights.copy(),
        indices=data.indices.copy(),
        bone_names=list(data.bone_names),
        parent_index=data.parent_index.copy(),
        bind_local=bind_local,
        inverse_bind=inverse_bind,
        path=data.path,
    )


def recover_import_scale(ply_rest: np.ndarray, json_positions: np.ndarray) -> float:
    """Fallback when no palette sidecar carries importScale: the uniform scale is
    recovered as median(‖engine rest vertex‖ / ‖raw JSON vertex‖) over shared rows."""
    a = np.asarray(ply_rest, dtype=np.float64)
    b = np.asarray(json_positions, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"expected matching (n,3) arrays, got {a.shape} vs {b.shape}")
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    mask = norm_b > 1e-12
    if not np.any(mask):
        raise ValueError("cannot recover import scale: all JSON positions are at the origin")
    return float(np.median(norm_a[mask] / norm_b[mask]))


# --------------------------------------------------------------------------- #
# Weight surgery (Mode B experiments — pure numpy, never mutates the inputs).


def weight_surgery(
    joint_indices: np.ndarray,
    joint_weights: np.ndarray,
    mode: str,
    bone_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return new (joint_indices, joint_weights) arrays under a surgery mode.

    * ``none``: unchanged copies.
    * ``top1``: all weight hard-assigned to each row's argmax influence.
    * ``drop_hand`` / ``drop_fingers``: zero the weights of joints whose name
      contains "hand" / a finger token, then renormalize each row; rows whose
      weights all dropped keep their ORIGINAL weights (nothing left to blend).
    """
    ji = np.array(joint_indices, dtype=np.int32, copy=True)
    jw = np.array(joint_weights, dtype=np.float32, copy=True)
    if mode == "none":
        return ji, jw
    if mode == "top1":
        onehot = np.zeros_like(jw)
        if jw.shape[0]:
            onehot[np.arange(jw.shape[0]), np.argmax(jw, axis=1)] = 1.0
        return ji, onehot
    if mode in ("drop_hand", "drop_fingers"):
        tokens = ("hand",) if mode == "drop_hand" else FINGER_NAME_TOKENS
        lowered = [name.lower() for name in bone_names]
        drop = np.asarray([any(tok in name for tok in tokens) for name in lowered], dtype=bool)
        in_range = (ji >= 0) & (ji < len(bone_names))
        dropped = np.zeros(jw.shape, dtype=bool)
        dropped[in_range] = drop[ji[in_range]]
        surgical = np.where(dropped, np.float32(0.0), jw)
        sums = surgical.sum(axis=1)
        ok = sums > 0.0
        surgical[ok] = surgical[ok] / sums[ok, None]
        surgical[~ok] = jw[~ok]
        return ji, surgical
    raise ValueError(f"unknown weight-surgery mode '{mode}' (expected one of {WEIGHT_SURGERY_MODES})")

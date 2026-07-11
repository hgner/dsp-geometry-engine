"""Reader/writer for the engine's exact PLY dialect and its stderr run metadata.

Dialect (verified against proje7-engine tools/layered-field-dump/main.cpp):
ASCII 1.0; vertex properties in order
    float x y z nx ny nz | uchar red green blue | int sourceIndex | float restx resty restz
optionally followed (patched engine, --weights) by
    int j0 j1 j2 j3 | float w0 w1 w2 w3
faces are ``property list uchar int vertex_indices`` rows of literally ``3 i j k``.

Quirk that must never be "fixed": the property NAMED ``sourceIndex`` carries the
dominant skin-joint index (argmax of the vertex's 4 weights). A future writer may
add a property literally named ``dominantJoint``; when present it wins.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Matches the engine's deterministic joint->RGB hash (paletteColor in main.cpp).
_PALETTE_MULT = np.uint64(2654435761)


def palette_color(joint: np.ndarray) -> np.ndarray:
    """(n,) joint indices -> (n,3) uint8 RGB, identical to the engine's paletteColor."""
    hue = (joint.astype(np.uint64) * _PALETTE_MULT) & np.uint64(0xFFFFFF)
    rgb = np.empty((joint.shape[0], 3), dtype=np.uint8)
    rgb[:, 0] = (hue >> np.uint64(16)) & np.uint64(0xFF)
    rgb[:, 1] = (hue >> np.uint64(8)) & np.uint64(0xFF)
    rgb[:, 2] = hue & np.uint64(0xFF)
    return rgb


@dataclass
class EngineDump:
    """One parsed engine PLY. ``posed`` is the deformed surface (LBS + deformers);
    ``rest`` is the bind-space position of the same vertex (same row) — one posed
    dump therefore supports bind-vs-posed comparison without a second engine run."""

    posed: np.ndarray  # (n,3) float32
    normals: np.ndarray  # (n,3) float32
    rest: np.ndarray  # (n,3) float32
    dominant_joint: np.ndarray  # (n,) int32
    faces: np.ndarray  # (m,3) int32
    props: list[str] = field(default_factory=list)
    path: Path | None = None
    # Present only when the patched engine was invoked with --weights:
    joint_indices: np.ndarray | None = None  # (n,4) int32
    joint_weights: np.ndarray | None = None  # (n,4) float32
    # Present only if a future writer adds pre-deformer positions (lbsx/lbsy/lbsz):
    lbs: np.ndarray | None = None  # (n,3) float32

    @property
    def vertex_count(self) -> int:
        return int(self.posed.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    def channel(self, name: str) -> np.ndarray:
        """Resolve a channel name used across the MCP tools."""
        if name == "posed":
            return self.posed
        if name == "rest":
            return self.rest
        if name == "lbs":
            if self.lbs is None:
                raise ValueError("this dump has no lbs channel (engine not patched with lbs output)")
            return self.lbs
        raise ValueError(f"unknown channel '{name}' (expected posed|rest|lbs)")


def read_engine_ply(path: Path | str) -> EngineDump:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    n_vertex = n_face = 0
    props: list[str] = []
    in_vertex_element = False
    i = 0
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"{path}: not a PLY file (missing 'ply' magic)")
    while i < len(lines):
        ln = lines[i].strip()
        i += 1
        if ln.startswith("format"):
            if "ascii" not in ln:
                raise ValueError(f"{path}: only 'format ascii 1.0' is supported, got '{ln}'")
        elif ln.startswith("element vertex"):
            n_vertex = int(ln.split()[-1])
            in_vertex_element = True
        elif ln.startswith("element"):
            if ln.startswith("element face"):
                n_face = int(ln.split()[-1])
            in_vertex_element = False
        elif ln.startswith("property") and "list" not in ln and in_vertex_element:
            props.append(ln.split()[-1])
        elif ln == "end_header":
            break
    else:
        raise ValueError(f"{path}: header never terminated with end_header")

    if n_vertex == 0:
        raise ValueError(f"{path}: no vertices")

    vertex_block = "\n".join(lines[i : i + n_vertex])
    cols = np.loadtxt(io.StringIO(vertex_block), dtype=np.float64, ndmin=2)
    if cols.shape != (n_vertex, len(props)):
        raise ValueError(
            f"{path}: vertex block shape {cols.shape} does not match header "
            f"({n_vertex} vertices x {len(props)} properties {props})"
        )

    def col3(names: tuple[str, str, str]) -> np.ndarray | None:
        if all(n in props for n in names):
            idx = [props.index(n) for n in names]
            return cols[:, idx].astype(np.float32)
        return None

    posed = col3(("x", "y", "z"))
    if posed is None:
        raise ValueError(f"{path}: missing x/y/z properties (have {props})")
    normals = col3(("nx", "ny", "nz"))
    if normals is None:
        normals = np.zeros_like(posed)
    rest = col3(("restx", "resty", "restz"))
    if rest is None:
        rest = posed.copy()  # rest field dumps carry only one space

    # Quirk: 'sourceIndex' carries the dominant joint; a literal 'dominantJoint' wins.
    if "dominantJoint" in props:
        dominant = cols[:, props.index("dominantJoint")].astype(np.int32)
    elif "sourceIndex" in props:
        dominant = cols[:, props.index("sourceIndex")].astype(np.int32)
    else:
        dominant = np.zeros(n_vertex, dtype=np.int32)

    jnames = ("j0", "j1", "j2", "j3")
    wnames = ("w0", "w1", "w2", "w3")
    joint_indices = joint_weights = None
    if all(n in props for n in jnames) and all(n in props for n in wnames):
        joint_indices = cols[:, [props.index(n) for n in jnames]].astype(np.int32)
        joint_weights = cols[:, [props.index(n) for n in wnames]].astype(np.float32)

    lbs = col3(("lbsx", "lbsy", "lbsz"))

    faces = np.empty((0, 3), dtype=np.int32)
    if n_face > 0:
        face_block = "\n".join(lines[i + n_vertex : i + n_vertex + n_face])
        fcols = np.loadtxt(io.StringIO(face_block), dtype=np.int64, ndmin=2)
        if fcols.shape[1] != 4 or not np.all(fcols[:, 0] == 3):
            raise ValueError(f"{path}: expected triangle face rows '3 i j k'")
        faces = fcols[:, 1:4].astype(np.int32)

    return EngineDump(
        posed=posed,
        normals=normals,
        rest=rest,
        dominant_joint=dominant,
        faces=faces,
        props=props,
        path=path,
        joint_indices=joint_indices,
        joint_weights=joint_weights,
        lbs=lbs,
    )


def write_engine_ply(path: Path | str, dump: EngineDump) -> None:
    """Writes the exact engine dialect (fixtures and the CI stub engine share this
    writer with the parser, so the two can never drift apart)."""
    path = Path(path)
    n = dump.vertex_count
    rgb = palette_color(dump.dominant_joint)
    with_weights = dump.joint_indices is not None and dump.joint_weights is not None

    out = io.StringIO()
    out.write("ply\nformat ascii 1.0\ncomment posed layered mesh colored by dominant skin joint\n")
    out.write(f"element vertex {n}\n")
    out.write("property float x\nproperty float y\nproperty float z\n")
    out.write("property float nx\nproperty float ny\nproperty float nz\n")
    out.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    out.write("property int sourceIndex\n")
    out.write("property float restx\nproperty float resty\nproperty float restz\n")
    if with_weights:
        for k in range(4):
            out.write(f"property int j{k}\n")
        for k in range(4):
            out.write(f"property float w{k}\n")
    out.write(f"element face {dump.face_count}\n")
    out.write("property list uchar int vertex_indices\nend_header\n")
    for idx in range(n):
        p = dump.posed[idx]
        nrm = dump.normals[idx]
        r = dump.rest[idx]
        row = (
            f"{p[0]:.6g} {p[1]:.6g} {p[2]:.6g} {nrm[0]:.6g} {nrm[1]:.6g} {nrm[2]:.6g} "
            f"{rgb[idx, 0]} {rgb[idx, 1]} {rgb[idx, 2]} {int(dump.dominant_joint[idx])} "
            f"{r[0]:.6g} {r[1]:.6g} {r[2]:.6g}"
        )
        if with_weights:
            assert dump.joint_indices is not None and dump.joint_weights is not None
            ji = dump.joint_indices[idx]
            jw = dump.joint_weights[idx]
            row += " " + " ".join(str(int(v)) for v in ji)
            row += " " + " ".join(f"{float(v):.6g}" for v in jw)
        out.write(row + "\n")
    for f in dump.faces:
        out.write(f"3 {int(f[0])} {int(f[1])} {int(f[2])}\n")
    path.write_text(out.getvalue(), encoding="utf-8")


# --------------------------------------------------------------------------- #
# npz cache: MCP tool calls repeatedly touch the same 50k-line ASCII dumps.


def _cache_key(path: Path) -> str:
    st = path.stat()
    return hashlib.sha1(f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()


def load_dump_cached(path: Path | str, cache_root: Path | None = None) -> EngineDump:
    path = Path(path)
    if cache_root is None:
        return read_engine_ply(path)
    cache_root.mkdir(parents=True, exist_ok=True)
    npz_path = cache_root / f"{_cache_key(path)}.npz"
    if npz_path.is_file():
        with np.load(npz_path, allow_pickle=False) as z:
            props = [str(p) for p in z["props"]]

            def opt(name: str) -> np.ndarray | None:
                return z[name] if name in z.files else None

            return EngineDump(
                posed=z["posed"],
                normals=z["normals"],
                rest=z["rest"],
                dominant_joint=z["dominant_joint"],
                faces=z["faces"],
                props=props,
                path=path,
                joint_indices=opt("joint_indices"),
                joint_weights=opt("joint_weights"),
                lbs=opt("lbs"),
            )
    dump = read_engine_ply(path)
    arrays: dict[str, np.ndarray] = {
        "posed": dump.posed,
        "normals": dump.normals,
        "rest": dump.rest,
        "dominant_joint": dump.dominant_joint,
        "faces": dump.faces,
        "props": np.asarray(dump.props),
    }
    if dump.joint_indices is not None:
        arrays["joint_indices"] = dump.joint_indices
    if dump.joint_weights is not None:
        arrays["joint_weights"] = dump.joint_weights
    if dump.lbs is not None:
        arrays["lbs"] = dump.lbs
    np.savez_compressed(npz_path, **arrays)
    return dump


# --------------------------------------------------------------------------- #
# stderr run metadata (the bone table only exists on stderr).

_BONE_MAP_RE = re.compile(r"^bone-map:((?:\s+\d+=\S+)+)\s*$", re.MULTILINE)
_BONE_ENTRY_RE = re.compile(r"(\d+)=(\S+)")
_POSED_OK_RE = re.compile(
    r"-posed-ok\s+clip=(?P<clip>\S+)\s+sampleTime=(?P<t>[-+0-9.eE]+)\s+verts=(?P<n>\d+)"
)
_PUSH_RE = re.compile(r"collisionPush=(?P<a>[-+0-9.eE]+)\s+bakedCollisionPush=(?P<b>[-+0-9.eE]+)")
_IMPORT_OK_RE = re.compile(r"import-ok\s+verts=(?P<v>\d+)\s+bones=(?P<b>\d+)\s+clips=(?P<c>\d+)")


@dataclass
class DumpMeta:
    bone_map: dict[int, str] = field(default_factory=dict)
    clip_id: str | None = None
    sample_time: float | None = None
    vert_count: int | None = None
    # Max deformer push MAGNITUDES in meters (floats — e.g. 0.0012), not counts.
    collision_push: float | None = None
    baked_collision_push: float | None = None
    imported: bool = False

    def to_json(self) -> str:
        payload = {
            "boneMap": {str(k): v for k, v in sorted(self.bone_map.items())},
            "clipId": self.clip_id,
            "sampleTime": self.sample_time,
            "vertCount": self.vert_count,
            "collisionPush": self.collision_push,
            "bakedCollisionPush": self.baked_collision_push,
            "imported": self.imported,
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, text: str) -> DumpMeta:
        payload = json.loads(text)
        return cls(
            bone_map={int(k): str(v) for k, v in payload.get("boneMap", {}).items()},
            clip_id=payload.get("clipId"),
            sample_time=payload.get("sampleTime"),
            vert_count=payload.get("vertCount"),
            collision_push=payload.get("collisionPush"),
            baked_collision_push=payload.get("bakedCollisionPush"),
            imported=bool(payload.get("imported", False)),
        )


def parse_dump_stderr(text: str) -> DumpMeta:
    meta = DumpMeta()
    m = _BONE_MAP_RE.search(text)
    if m:
        meta.bone_map = {int(i): name for i, name in _BONE_ENTRY_RE.findall(m.group(1))}
    m = _POSED_OK_RE.search(text)
    if m:
        meta.clip_id = m.group("clip")
        meta.sample_time = float(m.group("t"))
        meta.vert_count = int(m.group("n"))
    m = _PUSH_RE.search(text)
    if m:
        meta.collision_push = float(m.group("a"))
        meta.baked_collision_push = float(m.group("b"))
    meta.imported = _IMPORT_OK_RE.search(text) is not None
    return meta


def save_meta(ply_path: Path | str, meta: DumpMeta) -> Path:
    meta_path = Path(ply_path).with_suffix(".meta.json")
    meta_path.write_text(meta.to_json(), encoding="utf-8")
    return meta_path


def load_meta(ply_path: Path | str) -> DumpMeta | None:
    meta_path = Path(ply_path).with_suffix(".meta.json")
    if not meta_path.is_file():
        return None
    return DumpMeta.from_json(meta_path.read_text(encoding="utf-8"))


def bone_index_for(bone_map: dict[int, str], name: str) -> int:
    for idx, bone in bone_map.items():
        if bone == name:
            return idx
    available = ", ".join(f"{i}={n}" for i, n in sorted(bone_map.items()))
    raise KeyError(f"bone '{name}' not in bone map ({available or 'empty'})")

"""Topology-QA goldens: every mesh is built BY HAND (no engine, no subprocess).

The reference solids are a triangulated unit cube (closed genus-0 shell) and a
tetrahedron, from which the disconnected / holed / non-manifold cases are
assembled. Euler characteristics, boundary-edge counts and component counts are
pinned exactly; the tool lane is exercised through ``geometry._analyze_mesh_topology``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from dsp_server.engine import topology
from dsp_server.engine.ply import EngineDump, write_engine_ply
from dsp_server.toolsets import AppContext, geometry

# Unit-cube corners, indexed 0..7 (z=0 face first, then z=1 face).
CUBE_VERTS = np.array(
    [
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [0.0, 1.0, 1.0],  # 7
    ],
    dtype=np.float64,
)

# The six cube faces as vertex loops; each is triangulated (a,b,c,d) -> two tris.
# The top quad is LAST so drop_quads=1 removes it (a clean single-ring hole).
_CUBE_QUADS = (
    (0, 1, 2, 3),  # bottom  z=0
    (0, 1, 5, 4),  # front   y=0
    (2, 3, 7, 6),  # back    y=1
    (0, 3, 7, 4),  # left    x=0
    (1, 2, 6, 5),  # right   x=1
    (4, 5, 6, 7),  # top     z=1
)


def _tri_quad(quad: tuple[int, int, int, int]) -> list[tuple[int, int, int]]:
    a, b, c, d = quad
    return [(a, b, c), (a, c, d)]


def make_cube_faces(drop_quads: int = 0) -> np.ndarray:
    """12 triangles of the closed unit cube; ``drop_quads`` omits that many faces
    from the end (drop_quads=1 -> the top quad's 2 triangles are removed)."""
    quads = _CUBE_QUADS[: len(_CUBE_QUADS) - drop_quads] if drop_quads else _CUBE_QUADS
    faces: list[tuple[int, int, int]] = []
    for quad in quads:
        faces.extend(_tri_quad(quad))
    return np.asarray(faces, dtype=np.int64)


def make_tetra_faces(offset: int = 0) -> np.ndarray:
    """The 4 triangles of a tetrahedron on vertices ``offset..offset+3``."""
    v = [offset + i for i in range(4)]
    tris = [
        (v[0], v[1], v[2]),
        (v[0], v[1], v[3]),
        (v[0], v[2], v[3]),
        (v[1], v[2], v[3]),
    ]
    return np.asarray(tris, dtype=np.int64)


# --------------------------------------------------------------------------- #
# engine/topology.py goldens


def test_watertight_cube():
    topo = topology.analyze_topology(make_cube_faces(), n_vertices=8)
    assert topo.n_vertices_referenced == 8
    assert topo.n_faces == 12
    assert topo.n_edges == 18  # 12 cube edges + 6 quad diagonals
    assert topo.n_components == 1
    assert topo.largest_component_fraction == 1.0
    assert topo.boundary_edge_count == 0
    assert topo.nonmanifold_edge_count == 0
    assert topo.is_manifold is True
    assert topo.is_watertight is True
    assert topo.euler_characteristic == 2
    assert topo.estimated_genus == 0
    assert topo.n_isolated_vertices == 0
    # Every cube edge borders exactly two triangles -> every edge is manifold.
    _edges, counts = topology.build_edges(make_cube_faces())
    assert set(np.unique(counts).tolist()) == {2}


def test_cube_minus_one_quad_has_hole():
    faces = make_cube_faces(drop_quads=1)  # 10 triangles, top removed
    topo = topology.analyze_topology(faces, n_vertices=8)
    assert topo.n_faces == 10
    assert topo.is_watertight is False
    assert topo.boundary_edge_count == 4  # the top ring: 4-5, 5-6, 6-7, 7-4
    assert topo.n_components == 1
    assert topo.is_manifold is True  # a hole is boundary, not non-manifold
    assert topo.estimated_genus is None  # genus only defined when watertight


def test_two_disjoint_tetrahedra():
    # Second tetra offset in index space (gap 4..9 unused) — the connectivity
    # graph must remap referenced verts so the gap does not spawn phantom parts.
    faces = np.vstack([make_tetra_faces(0), make_tetra_faces(10)])
    topo = topology.analyze_topology(faces)  # n_vertices unknown
    assert topo.n_vertices_referenced == 8
    assert topo.n_components == 2
    assert sorted(topo.component_sizes, reverse=True) == [4, 4]
    assert topo.euler_characteristic == 4  # 2 + 2, two closed shells
    assert topo.is_watertight is False  # more than one component
    assert topo.is_manifold is True  # each tetra edge borders exactly two faces
    assert topo.boundary_edge_count == 0


def test_nonmanifold_fan():
    # Three triangles all sharing the edge (0, 1): a non-manifold "book spine".
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int64)
    edges, counts = topology.build_edges(faces)
    shared = counts[np.all(edges == [0, 1], axis=1)]
    assert int(shared[0]) == 3
    topo = topology.analyze_topology(faces)
    assert topo.nonmanifold_edge_count >= 1
    assert topo.is_manifold is False
    assert topo.is_watertight is False


def test_geodesic_on_cube_between_opposite_corners():
    faces = make_cube_faces()
    # 0 = (0,0,0) and 6 = (1,1,1): opposite space-diagonal corners, never coplanar.
    dist, hops = topology.geodesic_between(faces, CUBE_VERTS, 0, 6)
    assert dist is not None
    assert math.isfinite(dist)
    assert dist > 1.0  # a single straight cube edge has length 1
    assert dist < 3.0
    assert hops is not None and hops >= 2
    # geodesic_distance is the thin float-only wrapper.
    assert topology.geodesic_distance(faces, CUBE_VERTS, 0, 6) == dist


def test_geodesic_none_across_disconnected_shells():
    faces = np.vstack([make_tetra_faces(0), make_tetra_faces(4)])
    positions = np.arange(24, dtype=np.float64).reshape(8, 3)
    assert topology.geodesic_distance(faces, positions, 0, 5) is None
    assert topology.geodesic_between(faces, positions, 0, 5) == (None, None)


# --------------------------------------------------------------------------- #
# toolset lane (geometry._analyze_mesh_topology)


def _ctx(tmp_path: Path) -> AppContext:
    data = tmp_path / "data"
    return AppContext(
        data_dir=data,
        dumps_dir=data / "dumps",
        cache_dir=data / "cache",
        plots_dir=data / "plots",
        logs_dir=data / "logs",
        engine_root=tmp_path / "engine",
    )


def _write_cube_dump(tmp_path: Path) -> Path:
    faces = make_cube_faces()
    verts = CUBE_VERTS.astype(np.float32)
    dump = EngineDump(
        posed=verts,
        normals=np.zeros_like(verts),
        rest=verts.copy(),
        dominant_joint=np.zeros(verts.shape[0], dtype=np.int32),
        faces=faces.astype(np.int32),
        props=[],
    )
    path = tmp_path / "cube.ply"
    write_engine_ply(path, dump)
    return path


def test_tool_reports_watertight_cube(tmp_path: Path):
    ctx = _ctx(tmp_path)
    path = _write_cube_dump(tmp_path)
    rep = json.loads(geometry._analyze_mesh_topology(ctx, str(path)))
    assert "error" not in rep
    assert rep["n_vertices_total"] == 8
    assert rep["n_vertices_referenced"] == 8
    assert rep["n_faces"] == 12
    assert rep["n_edges"] == 18
    assert rep["is_watertight"] is True
    assert rep["euler_characteristic"] == 2
    assert rep["estimated_genus"] == 0
    assert rep["geodesic_m"] is None  # no vertex pair requested


def test_tool_geodesic_and_bad_vertex(tmp_path: Path):
    ctx = _ctx(tmp_path)
    path = _write_cube_dump(tmp_path)

    rep = json.loads(geometry._analyze_mesh_topology(ctx, str(path), src_vertex=0, dst_vertex=6))
    assert "error" not in rep
    assert rep["geodesic_m"] is not None and rep["geodesic_m"] > 1.0
    assert rep["geodesic_hops"] >= 2

    # Out-of-range vertex -> ToolError (never a raise across the boundary).
    bad = json.loads(geometry._analyze_mesh_topology(ctx, str(path), src_vertex=0, dst_vertex=99))
    assert "error" in bad
    assert "out of range" in bad["error"]

    # Missing file -> ToolError, not an exception.
    missing = json.loads(geometry._analyze_mesh_topology(ctx, str(tmp_path / "nope.ply")))
    assert "error" in missing

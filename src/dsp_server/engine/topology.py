"""Whole-mesh topology QA on an engine dump's triangle connectivity.

Pure numpy + ``scipy.sparse.csgraph`` — arrays in, a dataclass/scalars out, no
file I/O. This is the topology counterpart to the spectral corrugation lane: the
rigged3 forearm ripple is an every-other-edge-loop artifact of the *source* mesh,
so before (or beside) measuring the ripple we check whether the bake left the
mesh topologically damaged — non-manifold edges (an edge shared by >2 triangles),
open boundaries (an edge in only 1 triangle), disconnected islands, irregular
vertex valence, and a non-integer/unexpected Euler characteristic.

Connectivity is computed over *referenced* vertices only (the distinct indices
that actually appear in ``faces``), remapped to a compact range: index gaps must
never spawn phantom singleton components, and truly face-less ("isolated")
vertices are reported separately via ``n_isolated_vertices``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

# Degree of an interior vertex in a regular triangle mesh; anything else is
# "irregular" (advisory — poles/boundaries are legitimately irregular).
_REGULAR_VALENCE = 6
# Cap on the inline lists/dicts a MeshTopology carries (component sizes, valence
# histogram) so the JSON summary never balloons on a pathological mesh.
_CAP = 20


def build_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Undirected edges of a triangle soup plus each edge's incident-face count.

    Returns ``(edges, face_counts)`` where ``edges`` is an ``(E, 2)`` int64 array
    with ``edges[:, 0] < edges[:, 1]`` (lexicographically sorted, deduplicated)
    and ``face_counts[k]`` is how many triangles edge ``k`` belongs to
    (1 = boundary, 2 = manifold interior, >2 = non-manifold).
    """
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (m, 3); got {faces.shape}")
    if faces.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.int64)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)  # canonical orientation: column 0 < column 1
    edges, counts = np.unique(e, axis=0, return_counts=True)
    return edges, counts.astype(np.int64)


@dataclass
class MeshTopology:
    """Whole-mesh topology verdict for one triangle-index array."""

    n_vertices_referenced: int
    n_faces: int
    n_edges: int
    n_components: int
    largest_component_fraction: float
    component_sizes: list[int]
    boundary_edge_count: int
    nonmanifold_edge_count: int
    is_manifold: bool
    is_watertight: bool
    valence_min: int
    valence_max: int
    valence_mean: float
    valence_histogram: dict[int, int]
    n_irregular_valence: int
    euler_characteristic: int
    estimated_genus: int | None
    n_isolated_vertices: int


def analyze_topology(faces: np.ndarray, n_vertices: int | None = None) -> MeshTopology:
    """Full topology QA over ``faces`` (an ``(m, 3)`` triangle index array).

    ``n_vertices`` (e.g. the dump's total vertex count) only feeds
    ``n_isolated_vertices`` — vertices carried by the dump but referenced by no
    triangle; connectivity itself is computed over referenced vertices alone.
    """
    faces = np.asarray(faces, dtype=np.int64)
    edges, counts = build_edges(faces)
    n_faces = int(faces.shape[0])
    n_edges = int(edges.shape[0])

    referenced = np.unique(faces) if faces.size else np.empty((0,), dtype=np.int64)
    n_ref = int(referenced.size)

    boundary_edge_count = int(np.count_nonzero(counts == 1))
    nonmanifold_edge_count = int(np.count_nonzero(counts > 2))
    is_manifold = nonmanifold_edge_count == 0

    if n_ref == 0:
        n_components = 0
        component_sizes: list[int] = []
        largest_fraction = 0.0
        valence = np.empty((0,), dtype=np.int64)
    else:
        # Remap edge endpoints into a compact 0..n_ref-1 range (index gaps must
        # not become phantom components); build a symmetric adjacency and label.
        local = np.searchsorted(referenced, edges)  # (E, 2), each in [0, n_ref)
        data = np.ones(local.shape[0], dtype=np.int8)
        adj = csr_matrix((data, (local[:, 0], local[:, 1])), shape=(n_ref, n_ref))
        adj = adj + adj.T
        n_components, labels = connected_components(adj, directed=False)
        sizes = np.sort(np.bincount(labels, minlength=n_components))[::-1]
        component_sizes = [int(s) for s in sizes[:_CAP]]
        largest_fraction = float(sizes[0]) / float(n_ref)
        # Valence = number of distinct incident edges (each undirected edge adds
        # one to each of its two endpoints).
        valence = np.zeros(n_ref, dtype=np.int64)
        np.add.at(valence, local[:, 0], 1)
        np.add.at(valence, local[:, 1], 1)

    if valence.size:
        valence_min = int(valence.min())
        valence_max = int(valence.max())
        valence_mean = float(valence.mean())
        n_irregular = int(np.count_nonzero(valence != _REGULAR_VALENCE))
        deg, deg_count = np.unique(valence, return_counts=True)
        pairs = list(zip(deg.tolist(), deg_count.tolist(), strict=True))
        if len(pairs) > _CAP:  # keep the most populous degrees (tie-break by degree)
            pairs = sorted(pairs, key=lambda kc: (-kc[1], kc[0]))[:_CAP]
        valence_histogram = {int(k): int(v) for k, v in sorted(pairs)}
    else:
        valence_min = valence_max = n_irregular = 0
        valence_mean = 0.0
        valence_histogram = {}

    euler = n_ref - n_edges + n_faces
    is_watertight = is_manifold and boundary_edge_count == 0 and n_components == 1
    # Closed orientable surface: euler = 2 - 2g. Only meaningful when watertight.
    estimated_genus = int(round((2 - euler) / 2)) if is_watertight else None

    n_isolated = 0
    if n_vertices is not None and n_vertices > n_ref:
        n_isolated = int(n_vertices - n_ref)

    return MeshTopology(
        n_vertices_referenced=n_ref,
        n_faces=n_faces,
        n_edges=n_edges,
        n_components=int(n_components),
        largest_component_fraction=largest_fraction,
        component_sizes=component_sizes,
        boundary_edge_count=boundary_edge_count,
        nonmanifold_edge_count=nonmanifold_edge_count,
        is_manifold=is_manifold,
        is_watertight=is_watertight,
        valence_min=valence_min,
        valence_max=valence_max,
        valence_mean=valence_mean,
        valence_histogram=valence_histogram,
        n_irregular_valence=n_irregular,
        euler_characteristic=int(euler),
        estimated_genus=estimated_genus,
        n_isolated_vertices=n_isolated,
    )


def geodesic_between(
    faces: np.ndarray,
    positions: np.ndarray,
    src_vertex: int,
    dst_vertex: int,
) -> tuple[float | None, int | None]:
    """Edge-length-weighted surface geodesic ``src -> dst`` and its hop count.

    Dijkstra over the mesh edge graph where each edge weighs ``norm(p_i - p_j)``.
    Returns ``(distance_m, hops)`` or ``(None, None)`` when the two vertices are
    on disconnected shells.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = int(positions.shape[0])
    src, dst = int(src_vertex), int(dst_vertex)
    if not (0 <= src < n and 0 <= dst < n):
        raise ValueError(f"src/dst vertex out of range [0, {n}): got {src}, {dst}")
    if src == dst:
        return 0.0, 0

    edges, _counts = build_edges(faces)
    if edges.shape[0] == 0:
        return None, None
    weights = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([weights, weights])
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))

    dist_arr, pred = dijkstra(graph, directed=False, indices=src, return_predecessors=True)
    dist = dist_arr[dst]
    if not np.isfinite(dist):
        return None, None
    hops = 0
    cur = dst
    while cur != src and cur >= 0:
        cur = int(pred[cur])
        hops += 1
        if hops > n:  # safety valve — a valid Dijkstra tree can never loop
            break
    return float(dist), int(hops)


def geodesic_distance(
    faces: np.ndarray,
    positions: np.ndarray,
    src_vertex: int,
    dst_vertex: int,
) -> float | None:
    """Surface geodesic distance ``src -> dst`` in meters, or ``None`` if the two
    vertices lie on disconnected shells. See :func:`geodesic_between` for hops."""
    dist, _hops = geodesic_between(faces, positions, src_vertex, dst_vertex)
    return dist

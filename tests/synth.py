"""Synthetic-geometry helpers shared by the test suite and the CI stub engine.

The workhorse is ``make_cylinder``: a fake forearm whose rest surface is a smooth
cylinder and whose posed surface carries an injected sinusoidal radial corrugation
of known amplitude and spatial frequency — the golden signal every FFT/roughness
assertion is calibrated against.
"""

from __future__ import annotations

import numpy as np

from dsp_server.engine.ply import EngineDump

# Engine-canonical bone naming (the procedural skeleton and the rigged3 import both
# use these): the corrugated segment must be armLowerL so joint lookup in the tools
# resolves the same way it does against real dumps.
STUB_BONE_MAP: dict[int, str] = {
    0: "pelvis",
    1: "spine",
    2: "chest",
    3: "clavicleL",
    4: "armUpperL",
    5: "armLowerL",
    6: "handL",
    7: "clavicleR",
    8: "armUpperR",
    9: "armLowerR",
    10: "handR",
}
ARM_LOWER_L = 5
ARM_UPPER_L = 4
HAND_L = 6


def _orthonormal_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return axis, u, v


def make_cylinder(
    n_rings: int = 64,
    n_around: int = 16,
    radius: float = 0.04,
    length: float = 0.25,
    corrugation_amp: float = 0.0,
    corrugation_freq_cpm: float = 0.0,
    axis: tuple[float, float, float] = (0.3, 0.9, 0.2),
    origin: tuple[float, float, float] = (0.1, 1.2, 0.05),
    noise: float = 0.0,
    joint_id: int = ARM_LOWER_L,
    cap_joint_ids: tuple[int, int] = (ARM_UPPER_L, HAND_L),
    with_weights: bool = False,
    seed: int = 0,
) -> EngineDump:
    """Build an EngineDump fake forearm.

    rest  = smooth cylinder (bind space)
    posed = same cylinder with radial ripple amp*sin(2*pi*f*t) (+ optional noise)

    The first and last rings get neighbouring-joint ids (elbow/wrist junk) so
    segment selection and P5-P95 trimming are exercised exactly as on real dumps.
    """
    rng = np.random.default_rng(seed)
    ax, u, v = _orthonormal_frame(np.asarray(axis, dtype=np.float64))
    origin_v = np.asarray(origin, dtype=np.float64)

    t = np.linspace(0.0, length, n_rings)
    theta = np.linspace(0.0, 2.0 * np.pi, n_around, endpoint=False)
    tt, th = np.meshgrid(t, theta, indexing="ij")  # (n_rings, n_around)

    radial_dir = np.multiply.outer(np.cos(th), u) + np.multiply.outer(np.sin(th), v)
    axial = origin_v + np.multiply.outer(tt, ax)

    rest_r = np.full_like(tt, radius)
    posed_r = rest_r + corrugation_amp * np.sin(2.0 * np.pi * corrugation_freq_cpm * tt)
    if noise > 0.0:
        posed_r = posed_r + rng.normal(0.0, noise, size=posed_r.shape)

    rest = (axial + rest_r[..., None] * radial_dir).reshape(-1, 3).astype(np.float32)
    posed = (axial + posed_r[..., None] * radial_dir).reshape(-1, 3).astype(np.float32)
    normals = radial_dir.reshape(-1, 3).astype(np.float32)

    dominant = np.full(n_rings * n_around, joint_id, dtype=np.int32)
    dominant[:n_around] = cap_joint_ids[0]  # elbow-side ring
    dominant[-n_around:] = cap_joint_ids[1]  # wrist-side ring

    faces = []
    for i in range(n_rings - 1):
        for j in range(n_around):
            a = i * n_around + j
            b = i * n_around + (j + 1) % n_around
            c = (i + 1) * n_around + j
            d = (i + 1) * n_around + (j + 1) % n_around
            faces.append((a, b, c))
            faces.append((b, d, c))
    faces_arr = np.asarray(faces, dtype=np.int32)

    joint_indices = joint_weights = None
    if with_weights:
        joint_indices = np.zeros((dominant.shape[0], 4), dtype=np.int32)
        joint_indices[:, 0] = dominant
        joint_weights = np.zeros((dominant.shape[0], 4), dtype=np.float32)
        joint_weights[:, 0] = 1.0

    return EngineDump(
        posed=posed,
        normals=normals,
        rest=rest,
        dominant_joint=dominant,
        faces=faces_arr,
        props=[],
        path=None,
        joint_indices=joint_indices,
        joint_weights=joint_weights,
    )


def stub_bone_map_stderr() -> str:
    """The single ``bone-map:`` stderr line in the engine's exact format."""
    entries = " ".join(f"{i}={name}" for i, name in sorted(STUB_BONE_MAP.items()))
    return f"bone-map: {entries}"

"""State-space controllability / observability analysis for the systems pack
(pure numpy, no I/O — mirrors the ltisys.py convention of dataclass results that
the toolset wraps into pydantic reports).

Given a linear time-invariant state model x' = A x + B u, y = C x:
- CONTROLLABILITY asks whether the inputs u can drive the state x to any point in
  R^n. It holds iff the controllability matrix [B, AB, ..., A^(n-1)B] has full row
  rank n (Kalman rank condition).
- OBSERVABILITY asks whether the outputs y uniquely determine the state x. It holds
  iff the observability matrix [C; CA; ...; CA^(n-1)] has full column rank n.

Rank is taken via ``np.linalg.matrix_rank`` (SVD with its default tolerance); the
smallest singular value is reported as a numerical margin (how close to
rank-deficient the system is). Complex/continuous vs discrete does not matter here
— the rank tests are algebraic and identical for both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_matrix(name: str, value: object) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix (list of equal-length rows), got shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    return a


def _min_singular(m: np.ndarray) -> float:
    """Smallest singular value of ``m`` (0.0 when empty) — the distance to rank
    deficiency along the worst-conditioned direction."""
    if m.size == 0:
        return 0.0
    sv = np.linalg.svd(m, compute_uv=False)
    return float(sv[-1]) if sv.size else 0.0


def controllability_matrix(a: object, b: object) -> np.ndarray:
    """Kalman controllability matrix [B, AB, A^2 B, ..., A^(n-1) B], shape (n, n*m)."""
    am = _as_matrix("A", a)
    bm = _as_matrix("B", b)
    n = am.shape[0]
    if am.shape[1] != n:
        raise ValueError(f"A must be square, got {am.shape[0]}x{am.shape[1]}")
    if bm.shape[0] != n:
        raise ValueError(f"B must have {n} rows to match A (n_states), got {bm.shape[0]}")
    blocks = [bm]
    ak_b = bm
    for _ in range(1, n):
        ak_b = am @ ak_b
        blocks.append(ak_b)
    return np.hstack(blocks)


def observability_matrix(a: object, c: object) -> np.ndarray:
    """Kalman observability matrix [C; CA; CA^2; ...; CA^(n-1)], shape (n*p, n)."""
    am = _as_matrix("A", a)
    cm = _as_matrix("C", c)
    n = am.shape[0]
    if am.shape[1] != n:
        raise ValueError(f"A must be square, got {am.shape[0]}x{am.shape[1]}")
    if cm.shape[1] != n:
        raise ValueError(f"C must have {n} columns to match A (n_states), got {cm.shape[1]}")
    blocks = [cm]
    c_ak = cm
    for _ in range(1, n):
        c_ak = c_ak @ am
        blocks.append(c_ak)
    return np.vstack(blocks)


@dataclass
class StateSpaceAnalysis:
    """Controllability (+ optional observability) verdict for one (A, B[, C])."""

    n_states: int
    n_inputs: int
    ctrb_rank: int
    controllable: bool
    uncontrollable_dim: int
    ctrb_min_singular: float
    n_outputs: int | None = None
    obsv_rank: int | None = None
    observable: bool | None = None
    unobservable_dim: int | None = None
    obsv_min_singular: float | None = None


def analyze(a: object, b: object, c: object | None = None) -> StateSpaceAnalysis:
    """Full controllability analysis of (A, B), plus observability when C is given.

    A must be square (n x n); B is n x m; C (optional) is p x n. Raises ValueError on
    a shape mismatch — the toolset turns that into a ToolError.
    """
    am = _as_matrix("A", a)
    bm = _as_matrix("B", b)
    n = am.shape[0]
    if am.shape[1] != n:
        raise ValueError(f"A must be square, got {am.shape[0]}x{am.shape[1]}")
    if bm.shape[0] != n:
        raise ValueError(f"B must have {n} rows to match A (n_states), got {bm.shape[0]}")
    m = bm.shape[1]
    ctrb = controllability_matrix(am, bm)
    ctrb_rank = int(np.linalg.matrix_rank(ctrb))
    result = StateSpaceAnalysis(
        n_states=n,
        n_inputs=m,
        ctrb_rank=ctrb_rank,
        controllable=(ctrb_rank == n),
        uncontrollable_dim=n - ctrb_rank,
        ctrb_min_singular=_min_singular(ctrb),
    )
    if c is not None:
        cm = _as_matrix("C", c)
        if cm.shape[1] != n:
            raise ValueError(f"C must have {n} columns to match A (n_states), got {cm.shape[1]}")
        p = cm.shape[0]
        obsv = observability_matrix(am, cm)
        obsv_rank = int(np.linalg.matrix_rank(obsv))
        result.n_outputs = p
        result.obsv_rank = obsv_rank
        result.observable = obsv_rank == n
        result.unobservable_dim = n - obsv_rank
        result.obsv_min_singular = _min_singular(obsv)
    return result

"""Physically-based BRDF energy math for the rendering pack (pure numpy, no I/O).

White-furnace / directional-albedo machinery. The engine is a DXR path tracer;
a BRDF whose hemispherical reflectance ``rho(wo)`` exceeds 1 manufactures energy
(fireflies, exposure blow-out), while ``rho << 1`` at high roughness is
single-scatter darkening (the missing multiple-scatter energy the model never
adds back).

Geometry convention: the shading normal is ``N = +z``. The outgoing direction
``wo`` is fixed at polar angle ``wo_theta`` in the x-z plane (azimuth 0), i.e.
``wo = (sin, 0, cos)``. Incident directions ``wi`` are integrated over the upper
hemisphere with the solid-angle measure ``dwi = sin_i dtheta_i dphi_i``; after the
substitution ``mu = cos_theta_i`` this becomes ``dwi = dmu dphi``, so
:func:`hemisphere_albedo` uses Gauss-Legendre nodes in ``mu`` over ``(0, 1)``
(exact for the Lambertian cos-weighted integral) crossed with a uniform midpoint
grid in ``phi`` over ``[0, 2pi)``.

Directional albedo:  ``rho(wo) = INT_hemisphere f_r(wi, wo) (N.wi) dwi``.
Energy conservation requires ``rho(wo) <= 1`` for every ``wo``.
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-12
_ALPHA_FLOOR = 1e-4  # keep D finite when roughness -> 0 (a perfect mirror is a delta)

MODELS = ("lambertian", "ggx")


# --------------------------------------------------------------------------- #
# Analytic BRDF terms


def alpha_of(roughness: float) -> float:
    """GGX slope parameter alpha = roughness**2 (Disney/UE parameterization)."""
    return float(roughness) ** 2


def lambertian(albedo: float = 1.0) -> float:
    """Lambertian BRDF value ``f_r = albedo / pi`` (direction-independent).

    Its directional albedo integrates to ``albedo`` exactly — the built-in
    self-check for the hemisphere integrator.
    """
    return float(albedo) / np.pi


def d_ggx(cos_h: np.ndarray, alpha: float, d_gain: float = 1.0) -> np.ndarray:
    """GGX/Trowbridge-Reitz normal distribution ``D(cos_h)``.

    ``d_gain`` scales the normalization: the physical value is 1.0; ``d_gain=2.0``
    simulates a mis-normalized (energy-leaking) BRDF for the leak detector.
    """
    a2 = alpha * alpha
    denom = cos_h * cos_h * (a2 - 1.0) + 1.0
    return d_gain * a2 / (np.pi * np.maximum(denom * denom, _TINY))


def smith_g1(cos_theta: np.ndarray, alpha: float) -> np.ndarray:
    """Smith masking-shadowing term for one direction, ``G1(c)``."""
    a2 = alpha * alpha
    c = cos_theta
    return 2.0 * c / np.maximum(c + np.sqrt(a2 + (1.0 - a2) * c * c), _TINY)


def smith_g(cos_i: np.ndarray, cos_o: float, alpha: float) -> np.ndarray:
    """Separable Smith geometry term ``G = G1(cos_i) * G1(cos_o)``."""
    return smith_g1(cos_i, alpha) * smith_g1(cos_o, alpha)


def schlick_f(cos_d: np.ndarray, f0: float) -> np.ndarray:
    """Schlick Fresnel ``F = F0 + (1-F0)(1-cos_d)^5``, ``cos_d = wi . h``."""
    return f0 + (1.0 - f0) * np.power(np.clip(1.0 - cos_d, 0.0, 1.0), 5.0)


def ggx_brdf(
    wi: np.ndarray,
    wo: np.ndarray,
    roughness: float,
    f0: float,
    d_gain: float = 1.0,
) -> np.ndarray:
    """Cook-Torrance GGX specular ``f_r`` for a stack of ``wi`` against one ``wo``.

    ``wi`` has shape ``(..., 3)``; ``wo`` is ``(3,)``. Returns ``f_r`` of shape
    ``(...,)``. The half-vector is ``h = normalize(wi + wo)``; the BRDF is
    ``D * G * F / (4 cos_i cos_o)``.
    """
    alpha = max(alpha_of(roughness), _ALPHA_FLOOR)
    wo = np.asarray(wo, dtype=np.float64)
    cos_i = wi[..., 2]
    cos_o = float(wo[2])
    h = wi + wo
    h = h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), _TINY)
    cos_h = h[..., 2]
    cos_d = np.clip(np.sum(wi * h, axis=-1), 0.0, 1.0)
    d = d_ggx(cos_h, alpha, d_gain)
    g = smith_g(cos_i, cos_o, alpha)
    f = schlick_f(cos_d, f0)
    denom = 4.0 * cos_i * cos_o
    return np.where(denom > _TINY, d * g * f / np.maximum(denom, _TINY), 0.0)


# --------------------------------------------------------------------------- #
# Dispatch + hemisphere integration


def _param(params: dict[str, float] | None, key: str, default: float) -> float:
    if params is None:
        return default
    return float(params.get(key, default))


def _wo_vector(wo_theta: float) -> np.ndarray:
    return np.array([np.sin(wo_theta), 0.0, np.cos(wo_theta)], dtype=np.float64)


def _eval_fr(model: str, wi: np.ndarray, wo: np.ndarray, params: dict[str, float] | None) -> np.ndarray:
    if model == "lambertian":
        return np.full(wi.shape[:-1], lambertian(_param(params, "albedo", 1.0)), dtype=np.float64)
    if model == "ggx":
        return ggx_brdf(
            wi,
            wo,
            _param(params, "roughness", 0.5),
            _param(params, "f0", 0.04),
            _param(params, "d_gain", 1.0),
        )
    raise ValueError(f"unknown BRDF model '{model}' (expected one of {list(MODELS)})")


def hemisphere_albedo(
    model: str,
    wo_theta: float,
    n_theta: int = 128,
    n_phi: int = 256,
    params: dict[str, float] | None = None,
) -> float:
    """Directional albedo ``rho(wo)`` by hemisphere integration of ``f_r * cos_i``.

    ``wo`` is fixed at polar angle ``wo_theta`` (azimuth 0). Gauss-Legendre nodes
    in ``mu = cos_theta_i`` over ``(0, 1)`` (never grazing/degenerate) crossed with
    a uniform midpoint grid in ``phi``. Returns ``rho(wo)``.
    """
    n_theta = int(n_theta)
    n_phi = int(n_phi)
    if n_theta < 1 or n_phi < 1:
        raise ValueError("n_theta and n_phi must be >= 1")
    wo = _wo_vector(float(wo_theta))

    nodes, weights = np.polynomial.legendre.leggauss(n_theta)
    mu = 0.5 * (nodes + 1.0)  # cos_theta_i in (0, 1)
    w_mu = 0.5 * weights
    sin_i = np.sqrt(np.clip(1.0 - mu * mu, 0.0, 1.0))

    phi = 2.0 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    w_phi = 2.0 * np.pi / n_phi
    cphi = np.cos(phi)
    sphi = np.sin(phi)

    wx = sin_i[:, None] * cphi[None, :]
    wy = sin_i[:, None] * sphi[None, :]
    wz = np.broadcast_to(mu[:, None], (n_theta, n_phi))
    wi = np.stack([wx, wy, wz], axis=-1)

    fr = _eval_fr(model, wi, wo, params)
    integrand = fr * mu[:, None]  # f_r * cos_i
    rho = float(np.sum(integrand * w_mu[:, None] * w_phi))
    return rho


def albedo_sweep(
    model: str,
    thetas: np.ndarray,
    params: dict[str, float] | None = None,
    n_theta: int = 128,
    n_phi: int = 256,
) -> np.ndarray:
    """``rho(wo)`` for each outgoing polar angle in ``thetas`` (radians)."""
    theta_arr = np.asarray(thetas, dtype=np.float64)
    return np.array(
        [hemisphere_albedo(model, float(t), n_theta, n_phi, params) for t in theta_arr],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- #
# Monte-Carlo variance (firefly risk under uniform vs cosine importance sampling)


def mc_variance(
    model: str,
    wo_theta: float,
    n_samples: int,
    seed: int = 0,
    sampling: str = "uniform",
    params: dict[str, float] | None = None,
) -> float:
    """Variance of the single-sample ``rho`` estimator ``f_r cos_i / pdf``.

    ``sampling='uniform'`` draws wi uniformly on the hemisphere (``pdf = 1/2pi``);
    ``sampling='cosine'`` draws cosine-weighted directions (``pdf = cos_i/pi``),
    which importance-samples the ``cos_i`` factor and, for a lobe centred near the
    normal, dramatically cuts the estimator variance (fewer fireflies). Returns the
    estimator variance; the caller reports both and their ratio. Deterministic via
    ``np.random.default_rng(seed)``.
    """
    n = int(n_samples)
    if n < 2:
        raise ValueError("n_samples must be >= 2 to estimate a variance")
    rng = np.random.default_rng(int(seed))
    wo = _wo_vector(float(wo_theta))
    u1 = rng.random(n)
    u2 = rng.random(n)
    phi = 2.0 * np.pi * u2

    if sampling == "uniform":
        cos_t = u1  # uniform in cos_theta -> uniform solid angle
        pdf = np.full(n, 1.0 / (2.0 * np.pi))
    elif sampling == "cosine":
        cos_t = np.sqrt(np.clip(1.0 - u1, 0.0, 1.0))  # cosine-weighted
        pdf = cos_t / np.pi
    else:
        raise ValueError(f"unknown sampling '{sampling}' (expected 'uniform' or 'cosine')")

    sin_t = np.sqrt(np.clip(1.0 - cos_t * cos_t, 0.0, 1.0))
    wi = np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=-1)
    fr = _eval_fr(model, wi, wo, params)
    est = fr * cos_t / np.maximum(pdf, _TINY)
    return float(np.var(est))

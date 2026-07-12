"""Rendering tool pack: physically-based BRDF energy auditing for the DXR path tracer.

One tool, :func:`verify_brdf_energy` — a deterministic white-furnace test. It
integrates the directional albedo ``rho(wo) = INT f_r(wi,wo)(N.wi) dwi`` over the
hemisphere for a range of outgoing angles and checks energy conservation
(``rho <= 1``). ``rho > 1`` means the BRDF manufactures light (fireflies, exposure
blow-out); ``rho << 1`` at high roughness is single-scatter darkening. An optional
Monte-Carlo pass quantifies firefly risk by comparing estimator variance under
uniform vs cosine importance sampling.

Same contract as every other pack: the tool returns a pydantic model's
``model_dump_json()``; failures return :class:`~dsp_server.schemas.ToolError` JSON
and never raise across the MCP boundary; the albedo curve lives on disk under
``data/brdf/*.npz`` and never crosses the wire. All math is in
:mod:`dsp_server.engine.brdf` (pure numpy); tests call ``_verify_brdf_energy``
directly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP

from dsp_server import plots
from dsp_server.engine import brdf
from dsp_server.schemas import SchemaBase, ToolError
from dsp_server.toolsets import AppContext

ENERGY_TOL = 1e-3  # rho <= 1 + ENERGY_TOL counts as conserving
_TINY = 1e-30
_DEFAULT_WO_DEG = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0]


# --------------------------------------------------------------------------- #
# Response model (pack-local, on the shared SchemaBase contract)


class BrdfEnergyReport(SchemaBase):
    """verify_brdf_energy: white-furnace directional-albedo audit."""

    model: str
    params: dict[str, float]
    albedo_by_angle_deg: dict[str, float]
    max_albedo: float
    min_albedo: float
    energy_conserving: bool
    worst_angle_deg: float
    single_scatter_loss: float | None = None
    mc_variance_uniform: float | None = None
    mc_variance_cosine: float | None = None
    variance_reduction: float | None = None
    albedo_curve_path: str
    plot_path: str | None = None


# --------------------------------------------------------------------------- #
# Helpers


def _error(exc: Exception, hint: str | None = None) -> str:
    return ToolError(error=f"{type(exc).__name__}: {exc}", hint=hint).model_dump_json()


def _brdf_dir(ctx: AppContext) -> Path:
    d = ctx.data_dir / "brdf"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stem(model: str, params: dict[str, float], angles: list[float]) -> str:
    h = hashlib.md5()
    h.update(model.encode("utf-8"))
    for k in sorted(params):
        h.update(f"{k}={params[k]!r}".encode())
    h.update(repr([float(a) for a in angles]).encode("utf-8"))
    return h.hexdigest()[:8]


def _angle_key(angle_deg: float) -> str:
    return f"{float(angle_deg):g}"


def _save_albedo_plot(path: Path, model: str, angles: list[float], rho: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = plots.plt
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.plot(angles, rho, marker="o", lw=1.4, color="tab:blue", label="rho(wo)")
    ax.axhline(1.0, ls="--", lw=1.0, color="tab:red", label="energy limit (rho=1)")
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("wo polar angle [deg]")
    ax.set_ylabel("directional albedo  rho(wo)")
    ax.set_title(f"white-furnace albedo — {model}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Tool: verify_brdf_energy


def _verify_brdf_energy(
    ctx: AppContext,
    model: str = "ggx",
    roughness: float = 0.5,
    f0: float = 0.04,
    albedo: float = 0.8,
    d_gain: float = 1.0,
    n_theta: int = 128,
    n_phi: int = 256,
    wo_angles_deg: list[float] | None = None,
    mc_samples: int = 0,
    seed: int = 0,
    save_plot: bool = False,
) -> str:
    try:
        if model not in brdf.MODELS:
            return ToolError(
                error=f"unknown BRDF model '{model}'",
                hint=f"model must be one of {list(brdf.MODELS)}",
            ).model_dump_json()
        angles = _DEFAULT_WO_DEG if wo_angles_deg is None else [float(a) for a in wo_angles_deg]
        if not angles:
            return ToolError(
                error="wo_angles_deg is empty",
                hint="pass a non-empty list of outgoing polar angles in degrees (0..90)",
            ).model_dump_json()
        if int(n_theta) < 2 or int(n_phi) < 2:
            return ToolError(
                error=f"n_theta and n_phi must be >= 2, got n_theta={n_theta}, n_phi={n_phi}",
                hint="128 x 256 is the default hemisphere quadrature; raise for tighter tolerance",
            ).model_dump_json()

        params = {
            "roughness": float(roughness),
            "f0": float(f0),
            "albedo": float(albedo),
            "d_gain": float(d_gain),
        }
        thetas = np.deg2rad(np.asarray(angles, dtype=np.float64))
        rho = brdf.albedo_sweep(model, thetas, params=params, n_theta=int(n_theta), n_phi=int(n_phi))

        albedo_by_angle = {_angle_key(a): float(r) for a, r in zip(angles, rho, strict=True)}
        i_max = int(np.argmax(rho))
        max_albedo = float(rho[i_max])
        min_albedo = float(np.min(rho))
        energy_conserving = bool(np.all(rho <= 1.0 + ENERGY_TOL))
        worst_angle = float(angles[i_max])
        # Single-scatter energy loss is a specular (ggx) diagnostic only: the missing
        # multiple-scatter energy shows up as 1 - min(rho) over the sweep.
        single_scatter_loss = float(1.0 - min_albedo) if model == "ggx" else None

        stem = _stem(model, params, angles)
        curve_path = _brdf_dir(ctx) / f"albedo_{model}_{stem}.npz"
        np.savez(curve_path, wo_deg=np.asarray(angles, dtype=np.float64), rho=rho)

        mc_u = mc_c = variance_reduction = None
        if int(mc_samples) >= 2:  # a variance estimate needs >= 2 samples; 1 -> skip, don't sink the report
            wo0 = float(thetas[0])  # firefly risk is quantified at the first requested angle
            n_mc = int(mc_samples)
            mc_u = brdf.mc_variance(model, wo0, n_mc, seed=int(seed), sampling="uniform", params=params)
            mc_c = brdf.mc_variance(model, wo0, n_mc, seed=int(seed), sampling="cosine", params=params)
            variance_reduction = float(mc_u / mc_c) if mc_c > _TINY else None

        plot_path = None
        if save_plot:
            out = ctx.plots_dir / f"brdf_energy_{model}_{stem}.png"
            plot_path = str(_save_albedo_plot(out, model, angles, rho))

        return BrdfEnergyReport(
            model=model,
            params=params,
            albedo_by_angle_deg=albedo_by_angle,
            max_albedo=max_albedo,
            min_albedo=min_albedo,
            energy_conserving=energy_conserving,
            worst_angle_deg=worst_angle,
            single_scatter_loss=single_scatter_loss,
            mc_variance_uniform=mc_u,
            mc_variance_cosine=mc_c,
            variance_reduction=variance_reduction,
            albedo_curve_path=str(curve_path),
            plot_path=plot_path,
        ).model_dump_json()
    except Exception as exc:  # the MCP boundary must never see a raise
        return _error(exc)


# --------------------------------------------------------------------------- #
# Registration


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the rendering tools on ``mcp``, bound to ``ctx``."""

    @mcp.tool()
    def verify_brdf_energy(
        model: str = "ggx",
        roughness: float = 0.5,
        f0: float = 0.04,
        albedo: float = 0.8,
        d_gain: float = 1.0,
        n_theta: int = 128,
        n_phi: int = 256,
        wo_angles_deg: list[float] | None = None,
        mc_samples: int = 0,
        seed: int = 0,
        save_plot: bool = False,
    ) -> str:
        """White-furnace energy audit for a physically-based BRDF. Integrates the
        directional albedo rho(wo) = INT f_r(wi,wo)(N.wi) dwi over the hemisphere for
        every outgoing polar angle in wo_angles_deg (default [0,15,30,45,60,75]) and
        checks energy conservation. rho > 1 => energy LEAK (non-physical: manufactures
        light, blows out exposure / spawns fireflies); rho << 1 at high roughness =>
        single-scatter energy LOSS (darkening, the missing multiple-scatter term).
        model='lambertian' (f_r=albedo/pi; rho==albedo exactly — the integrator
        self-check) or 'ggx' (Cook-Torrance GGX specular: roughness, f0 Fresnel, Smith
        masking). d_gain scales the D normalization (1.0 physical; 2.0 simulates a
        mis-normalized, leaking BRDF). When mc_samples>0 a Monte-Carlo pass at the first
        angle reports estimator variance under uniform vs cosine importance sampling plus
        the variance-reduction ratio (firefly risk). n_theta x n_phi sets the hemisphere
        quadrature. This is a design calculator: point it at the engine's ACTUAL BRDF
        parameters to match the shipped HLSL. Returns BrdfEnergyReport JSON (albedo by
        angle, max/min, energy_conserving bool, worst angle, single_scatter_loss for ggx,
        MC variances, the npz albedo-curve path, and an optional plot); ToolError JSON on
        failure."""
        return _verify_brdf_energy(
            ctx,
            model,
            roughness,
            f0,
            albedo,
            d_gain,
            n_theta,
            n_phi,
            wo_angles_deg,
            mc_samples,
            seed,
            save_plot,
        )

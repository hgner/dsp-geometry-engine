"""Pydantic v2 response models for every MCP tool (the iron rule: tools return a
model's ``model_dump_json()`` — scalars, short dicts, file paths only; arrays live
on disk). All floats are rounded to 6 significant figures on ingestion via a
``field_validator`` on the shared base, and every model forbids extra fields so a
typo in a tool body fails loudly instead of leaking silently into the JSON.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dsp_server.engine.transforms import RoughnessReport


def round6(value: float) -> float:
    """Round to 6 significant figures (compact JSON, stable numeric diffs)."""
    f = float(value)
    if f == 0.0 or not math.isfinite(f):
        return f
    return float(f"{f:.6g}")


def _round_nested(value: Any) -> Any:
    """Recursive 6-sig-fig rounding: floats, lists-of-lists, dicts-of-dicts —
    course packs return nested structures (eigenvector rows, loading tables)."""
    if isinstance(value, float):
        return round6(value)
    if isinstance(value, list):
        return [_round_nested(v) for v in value]
    if isinstance(value, dict):
        return {k: _round_nested(v) for k, v in value.items()}
    return value


class _SchemaBase(BaseModel):
    """Shared base: ``extra="forbid"`` + 6-significant-figure float rounding."""

    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="after")
    @classmethod
    def _round_floats(cls, value: Any) -> Any:
        return _round_nested(value)


# Public alias: course-pack toolset modules define their response models locally
# (keeps schemas.py from becoming a monolith) but share the base's contract.
SchemaBase = _SchemaBase


class ToolError(_SchemaBase):
    """Uniform failure envelope: every tool catches its own exceptions and returns
    this instead of raising across the MCP boundary."""

    error: str
    hint: str | None = None
    stderr_tail: list[str] | None = None


class SpectralPeakSchema(_SchemaBase):
    freq_cpm: float
    prominence_db: float


class RoughnessSchema(_SchemaBase):
    """1:1 mirror of :class:`dsp_server.engine.transforms.RoughnessReport`."""

    rms_ripple_m: float
    rel_ripple: float
    diff1_var: float
    diff2_var: float
    hf_energy_ratio: float
    dominant_freq_cpm: float
    dominant_wavelength_m: float
    peak_prominence_db: float
    ridge_count_est: int
    sector_phase_coherence: float
    corrugated: bool

    @classmethod
    def from_report(cls, report: RoughnessReport) -> RoughnessSchema:
        return cls(**asdict(report))


class TelemetryResult(_SchemaBase):
    """extract_mesh_telemetry: one successful engine dump, summarized."""

    ply_path: str
    meta_path: str | None = None
    palette_path: str | None = None
    duration_s: float = 0.0
    engine_stale: dict[str, Any] | None = None
    vertex_count: int
    face_count: int
    bone_count: int
    clip_id: str | None = None
    sample_time: float | None = None
    collision_push: float | None = None
    baked_collision_push: float | None = None
    imported: bool = False
    has_weights: bool = False
    bone_map: dict[int, str] = Field(default_factory=dict)
    arm_chain_histogram: dict[str, int] = Field(default_factory=dict)
    stderr_tail: list[str] = Field(default_factory=list)


class CorrugationReport(_SchemaBase):
    """analyze_corrugation: roughness verdict + top peaks + axial extent."""

    dump_path: str
    joint: str
    joint_ids: list[int]
    channel: str
    vertex_count: int
    n_samples: int
    t_span_m: float
    roughness: RoughnessSchema
    spectral_peaks: list[SpectralPeakSchema]
    extent_t_start_m: float
    extent_t_end_m: float
    extent_band_energy_fraction: float
    plot_path: str | None = None


class DisplacementStats(_SchemaBase):
    """Per-vertex ||delta p|| stats over the joint mask (equal-vertex-count dumps)."""

    max_m: float
    mean_m: float
    rms_m: float
    t_at_max_m: float


class SignalComparison(_SchemaBase):
    """compare_geometry_signals: RoughnessSchema per side + deltas (b minus a)."""

    dump_a: str
    channel_a: str
    dump_b: str
    channel_b: str
    joint: str
    joint_ids: list[int]
    roughness_a: RoughnessSchema
    roughness_b: RoughnessSchema
    delta_rel_ripple: float
    delta_hf_energy_ratio: float
    delta_peak_prominence_db: float
    ripple_ratio_b_over_a: float
    displacement: DisplacementStats | None = None
    plot_path: str | None = None


class SegmentRoughness(_SchemaBase):
    """One joint segment in the localize_defect ranking."""

    joint_id: int
    joint_name: str | None = None
    vertex_count: int
    score: float
    roughness: RoughnessSchema


class LocalizeReport(_SchemaBase):
    """localize_defect: per-joint roughness ranking, most corrugated first."""

    dump_path: str
    channel: str
    min_verts: int
    joints_scanned: int
    top: list[SegmentRoughness]


class LbsDifferentialReport(_SchemaBase):
    """lbs_differential: pure-LBS reconstruction vs engine posed output."""

    dump_path: str
    palette_path: str
    weights_source: str
    import_scale: float
    weight_surgery: str
    joint: str
    joint_ids: list[int]
    masked_vertex_count: int
    residual_global_max_m: float
    residual_global_mean_m: float
    residual_global_rms_m: float
    residual_masked_max_m: float
    residual_masked_mean_m: float
    residual_masked_rms_m: float
    roughness_engine: RoughnessSchema
    roughness_pure_lbs: RoughnessSchema
    influence_histogram: dict[int, int]
    mean_weight_entropy: float
    contributing_bones: list[str]
    verdict: str


class Spectrum2DSchema(_SchemaBase):
    """Scalar summary of one 2-D spectrum (cycles/pixel)."""

    dominant_freq_cpp: float
    dominant_wavelength_px: float
    dominant_orientation_deg: float
    dominant_prominence_db: float


class ImageComparisonReport(_SchemaBase):
    """compare_depth_renders: 2-D spectral summary (+ difference stats in pair mode)."""

    image_a: str
    image_b: str | None = None
    roi: list[int] | None = None
    height: int
    width: int
    spectrum_a: Spectrum2DSchema
    spectrum_b: Spectrum2DSchema | None = None
    mae: float | None = None
    rms: float | None = None
    ssim: float | None = None
    plot_path: str | None = None

"""Compact JSON schemas returned by the body-mesh MCP."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AddonInfo(_Schema):
    backend: str
    installed: bool
    enabled_hint: bool | None = None
    version: str | None = None
    path: str
    supported: bool
    note: str | None = None


class RuntimeReport(_Schema):
    blender_shortcut: str
    blender_executable: str | None
    blender_exists: bool
    data_dir: str
    input_roots: list[str]
    engine_skeleton_dir: str
    character_bake_executable: str
    engine_contract_available: bool
    addons: list[AddonInfo]
    supported_backend: str = "mpfb"
    execution_mode: str = "local-stdio/background-subprocess"
    warnings: list[str] = Field(default_factory=list)


class ReferenceInfo(_Schema):
    view: str
    source_path: str
    copied_path: str
    mask_path: str | None = None
    mask_width: int | None = None
    mask_height: int | None = None
    mask_sha256: str | None = None
    width: int
    height: int
    sha256: str


class PreparedJob(_Schema):
    job_id: str
    status: str
    job_dir: str
    manifest_path: str
    known_height_m: float | None = None
    rig_sex: Literal["male", "female"]
    engine_skeleton_path: str
    engine_skeleton_sha256: str
    references: list[ReferenceInfo]
    warnings: list[str] = Field(default_factory=list)


class ParameterCatalog(_Schema):
    backend: str
    macro_parameters: dict[str, str]
    target_query: str | None = None
    target_matches: list[str] = Field(default_factory=list)
    matches_truncated: bool = False
    note: str


class CandidateResult(_Schema):
    job_id: str
    candidate_id: str
    status: str
    backend: str
    candidate_dir: str
    blend_path: str
    obj_path: str
    ply_path: str
    meta_path: str
    engine_glb_path: str
    baked_character_path: str
    rig_sex: Literal["male", "female"]
    bone_count: int
    rest_pose: str
    engine_contract: dict[str, Any]
    render_paths: dict[str, str] = Field(default_factory=dict)
    reference_mask_paths: dict[str, str] = Field(default_factory=dict)
    vertex_count: int
    face_count: int
    duration_s: float
    macro_parameters: dict[str, float] = Field(default_factory=dict)
    target_parameters: dict[str, float] = Field(default_factory=dict)
    stderr_tail: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class IdentityRenderAsset(_Schema):
    asset_id: str
    kind: Literal["closeup", "body"]
    category: str
    view: str
    expression: str
    lighting: str
    path: str
    relative_path: str
    width: int
    height: int
    sha256: str


class IdentityRenderSet(_Schema):
    schema_version: Literal[1] = 1
    preset: Literal["identity-v1"] = "identity-v1"
    status: Literal["complete"] = "complete"
    job_id: str
    candidate_id: str
    source_blend_path: str
    source_blend_sha256: str
    recipe_sha256: str
    output_dir: str
    manifest_path: str
    renderer: str
    blender_version: str
    mpfb_version: str
    device_requested: str
    device_used: str
    samples: int
    seed: int
    material_backend: str
    expression_backend: str
    duration_s: float
    renders: list[IdentityRenderAsset]
    warnings: list[str] = Field(default_factory=list)


class JobStatus(_Schema):
    job_id: str
    status: str
    job_dir: str
    known_height_m: float | None = None
    rig_sex: str
    engine_skeleton_sha256: str | None = None
    references: list[dict[str, Any]] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)


class BodyMeshError(_Schema):
    error: str
    hint: str | None = None
    stderr_tail: list[str] | None = None

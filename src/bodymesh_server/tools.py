"""MCP tool implementations for controlled Blender/MPFB generation."""

from __future__ import annotations

from typing import Literal

from mcp.server.fastmcp import FastMCP

from bodymesh_server import blender_bridge, jobs
from bodymesh_server.environment import inspect_runtime
from bodymesh_server.schemas import BodyMeshError, JobStatus


def _error(exc: Exception, hint: str, *, tail: list[str] | None = None) -> str:
    return BodyMeshError(error=f"{type(exc).__name__}: {exc}", hint=hint, stderr_tail=tail).model_dump_json()


def _inspect_bodymesh_runtime() -> str:
    try:
        return inspect_runtime().model_dump_json()
    except Exception as exc:
        return _error(exc, "check BODYMESH_BLENDER_EXE, BODYMESH_MPFB_ROOT, and BODYMESH_DATA_DIR")


def _list_bodymesh_parameters(query: str | None = None, limit: int = 40) -> str:
    try:
        return blender_bridge.parameter_catalog(query=query, limit=limit).model_dump_json()
    except Exception as exc:
        return _error(exc, "use a two-or-more-character target query; call without query for macro controls")


def _prepare_bodymesh_job(
    front_image: str,
    rig_sex: str,
    side_image: str | None = None,
    back_image: str | None = None,
    front_mask: str | None = None,
    side_mask: str | None = None,
    back_mask: str | None = None,
    known_height_m: float | None = None,
    label: str | None = None,
) -> str:
    try:
        return jobs.prepare_job(
            front_image=front_image,
            rig_sex=rig_sex,
            side_image=side_image,
            back_image=back_image,
            front_mask=front_mask,
            side_mask=side_mask,
            back_mask=back_mask,
            known_height_m=known_height_m,
            label=label,
        ).model_dump_json()
    except Exception as exc:
        return _error(
            exc,
            "inputs must be local images under BODYMESH_INPUT_ROOTS; use DSP segment_image first for masks",
        )


def _create_body_mesh(
    job_id: str,
    macro_parameters: dict[str, float] | None = None,
    target_parameters: dict[str, float] | None = None,
    render_views: list[str] | None = None,
    render_size: int = 512,
    candidate_label: str | None = None,
) -> str:
    try:
        return blender_bridge.create_candidate(
            job_id=job_id,
            macro_parameters=macro_parameters,
            target_parameters=target_parameters,
            render_views=render_views,
            render_size=render_size,
            candidate_label=candidate_label,
        ).model_dump_json()
    except blender_bridge.BlenderBridgeError as exc:
        return _error(
            exc,
            "inspect the candidate logs; verify MPFB is enabled in Blender 4.2 "
            "and retry with bounded parameters",
            tail=exc.tail,
        )
    except Exception as exc:
        return _error(
            exc,
            "call inspect_bodymesh_runtime and list_bodymesh_parameters; "
            "outputs are confined to the job directory",
        )


def _get_bodymesh_job(job_id: str) -> str:
    try:
        directory, manifest = jobs.load_job(job_id)
        references = manifest.get("references") or []
        candidates = manifest.get("candidates") or []
        if not isinstance(references, list) or not isinstance(candidates, list):
            raise ValueError("job manifest is corrupt")
        return JobStatus(
            job_id=job_id,
            status=str(manifest.get("status") or "unknown"),
            job_dir=str(directory),
            known_height_m=manifest.get("known_height_m"),
            rig_sex=str(manifest.get("rig_sex") or ""),
            engine_skeleton_sha256=manifest.get("engine_skeleton_sha256"),
            references=[dict(value) for value in references if isinstance(value, dict)],
            candidates=[dict(value) for value in candidates if isinstance(value, dict)],
        ).model_dump_json()
    except Exception as exc:
        return _error(exc, "pass a job_id returned by prepare_bodymesh_job")


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    def inspect_bodymesh_runtime() -> str:
        """Inspect the local Blender 4.2 executable and installed body add-ons without creating a mesh.

        Returns the resolved direct blender.exe (never the launcher), MPFB/MB-LAB versions, engine
        skeleton/bake availability, local data directory, allowed reference roots, and blocking
        warnings. MPFB 2.0.x is the supported generator; MB-LAB is reported but not invoked.
        """

        return _inspect_bodymesh_runtime()

    @mcp.tool()
    def list_bodymesh_parameters(query: str | None = None, limit: int = 40) -> str:
        """List MPFB's bounded macro controls and optionally search installed detailed target files.

        Call without query for the macro parameter semantics. With a two-character-or-longer query
        (for example 'waist', 'shoulder', or 'leg'), returns at most limit safe relative target names
        accepted by create_body_mesh. This does not infer values from photographs.
        """

        return _list_bodymesh_parameters(query, limit)

    @mcp.tool()
    def prepare_bodymesh_job(
        front_image: str,
        rig_sex: Literal["male", "female"],
        side_image: str | None = None,
        back_image: str | None = None,
        front_mask: str | None = None,
        side_mask: str | None = None,
        back_mask: str | None = None,
        known_height_m: float | None = None,
        label: str | None = None,
    ) -> str:
        """Copy consented body reference images into an isolated local job.

        Inputs must be beneath BODYMESH_INPUT_ROOTS. Front and explicit rig_sex (male/female) are
        required; calibrated side and known height materially improve fitting. Optional binary masks
        should normally come from the DSP
        MCP's segment_image tool and must match their source image dimensions. The tool normalizes
        files to PNG, records hashes, and returns a job_id; it does not upload images or run Blender.
        """

        return _prepare_bodymesh_job(
            front_image,
            rig_sex,
            side_image,
            back_image,
            front_mask,
            side_mask,
            back_mask,
            known_height_m,
            label,
        )

    @mcp.tool()
    def create_body_mesh(
        job_id: str,
        macro_parameters: dict[str, float] | None = None,
        target_parameters: dict[str, float] | None = None,
        render_views: list[str] | None = None,
        render_size: int = 512,
        candidate_label: str | None = None,
    ) -> str:
        """Create one immutable MPFB candidate in a headless Blender subprocess.

        Macro and detailed target values are bounded to [0,1]; target names must come from
        list_bodymesh_parameters. The result includes neutral Blender/OBJ/DSP-PLY artifacts,
        silhouette renders, an exact 55-bone arms-down body_engine.glb, and validated
        character_bake_cli JSON with skin weights/tangents/clips. Repeated calls with revised values
        create new candidates for an LLM-driven render/compare/adjust loop. This is parametric
        fitting, not one-shot photo-to-3D reconstruction.
        """

        return _create_body_mesh(
            job_id,
            macro_parameters,
            target_parameters,
            render_views,
            render_size,
            candidate_label,
        )

    @mcp.tool()
    def get_bodymesh_job(job_id: str) -> str:
        """Return compact reference and candidate status for one MCP-owned body-mesh job."""

        return _get_bodymesh_job(job_id)


__all__ = [
    "register",
    "_inspect_bodymesh_runtime",
    "_list_bodymesh_parameters",
    "_prepare_bodymesh_job",
    "_create_body_mesh",
    "_get_bodymesh_job",
]

"""Geometry tool pack (ELE407 1-D lane): engine dumps -> axial signals -> verdicts.

Five tools: extract_mesh_telemetry (runs the engine via the bridge), and four
analysis tools that operate on PLY dumps already on disk. Every tool returns a
pydantic model's ``model_dump_json()`` — on failure a :class:`ToolError` whose
hint carries the dump's per-joint vertex histogram whenever joint selection was
the problem. Nothing here raises across the MCP boundary and no numpy array ever
leaves this module.

Testability: each registered tool is a thin closure over :class:`AppContext`
delegating to a module-level ``_impl`` function that tests call directly.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP

from cxx_bridge import engine_cli
from dsp_server import plots
from dsp_server.engine import filters, lbs, ply, scoring, topology, transforms
from dsp_server.engine.transforms import Signal1D
from dsp_server.schemas import (
    BakeRefMatch,
    BakeScoreReport,
    CorrugationReport,
    DisplacementStats,
    LbsDifferentialReport,
    LocalizeReport,
    RoughnessSchema,
    SchemaBase,
    SegmentRoughness,
    SignalComparison,
    SpectralPeakSchema,
    TelemetryResult,
    ToolError,
)
from dsp_server.toolsets import AppContext

_TINY = 1e-30
_ARM_CHAIN_TOKENS = ("arm", "hand", "clavicle")
_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]")


# --------------------------------------------------------------------------- #
# Shared helpers


def _safe(text: str) -> str:
    return _SAFE_RE.sub("_", text) or "x"


def _error(exc: Exception, hint: str | None = None, stderr_tail: list[str] | None = None) -> str:
    return ToolError(
        error=f"{type(exc).__name__}: {exc}", hint=hint, stderr_tail=stderr_tail
    ).model_dump_json()


def _bone_map_of(dump_path: Path) -> dict[int, str]:
    return ply.bone_map_for(dump_path)


def _joint_histogram(dump: ply.EngineDump, bone_map: dict[int, str]) -> dict[str, int]:
    uniq, counts = np.unique(dump.dominant_joint, return_counts=True)
    out: dict[str, int] = {}
    for j, c in zip(uniq.tolist(), counts.tolist(), strict=True):
        name = bone_map.get(int(j))
        key = f"{name} ({int(j)})" if name else str(int(j))
        out[key] = int(c)
    return out


def _histogram_hint(dump: ply.EngineDump | None, bone_map: dict[int, str]) -> str | None:
    if dump is None:
        return "could not load the dump — check the path points at an engine ASCII PLY"
    hist = ", ".join(f"{k}={v}" for k, v in _joint_histogram(dump, bone_map).items())
    return f"per-joint vertex histogram of this dump: {{{hist}}} — pick a joint name or integer id from it"


def _signal_bundle(
    dump: ply.EngineDump, mask: np.ndarray, channel: str
) -> tuple[Signal1D, list[Signal1D], transforms.AxisFrame]:
    points = dump.channel(channel)[mask]
    frame = transforms.fit_axis(points)
    t, r, theta = transforms.cylindrical_coords(points, frame)
    sig = transforms.axial_profile(t, r)
    sectors = transforms.sector_profiles(t, r, theta)
    return sig, sectors, frame


# --------------------------------------------------------------------------- #
# Tool 1: extract_mesh_telemetry


def _extract_mesh_telemetry(
    ctx: AppContext,
    character_json: str | None = None,
    sex: str = "male",
    pose_clip: str | None = "clip-cin-stand-attention",
    label: str | None = None,
    want_palette: bool = True,
) -> str:
    try:
        result = engine_cli.run_field_dump(
            character_json=character_json,
            sex=sex,
            pose_clip=pose_clip or None,
            label=label,
            want_palette=want_palette,
        )
        if not result.ok or result.ply_path is None:
            err = result.error or {}
            hint = err.get("hint") or (
                "engine run failed — see stderr_tail; full stderr logs are teed under data/logs/"
            )
            return ToolError(
                error=f"{err.get('kind', 'engine-error')}: {err.get('detail', 'engine run failed')}",
                hint=hint,
                stderr_tail=result.stderr_tail or None,
            ).model_dump_json()

        dump = ply.load_dump_cached(result.ply_path, ctx.cache_dir)
        meta = result.meta or ply.DumpMeta()
        bone_map = dict(meta.bone_map)
        arm_hist: dict[str, int] = {}
        for key, count in _joint_histogram(dump, bone_map).items():
            if any(tok in key.lower() for tok in _ARM_CHAIN_TOKENS):
                arm_hist[key] = count
        meta_path = result.ply_path.with_suffix(".meta.json")
        return TelemetryResult(
            ply_path=str(result.ply_path),
            meta_path=str(meta_path) if meta_path.is_file() else None,
            palette_path=str(result.palette_path) if result.palette_path else None,
            duration_s=result.duration_s,
            engine_stale=result.engine_stale,
            vertex_count=dump.vertex_count,
            face_count=dump.face_count,
            bone_count=len(bone_map),
            clip_id=meta.clip_id,
            sample_time=meta.sample_time,
            collision_push=meta.collision_push,
            baked_collision_push=meta.baked_collision_push,
            imported=meta.imported,
            has_weights=dump.joint_indices is not None,
            bone_map=bone_map,
            arm_chain_histogram=arm_hist,
            stderr_tail=result.stderr_tail,
        ).model_dump_json()
    except Exception as exc:  # the MCP boundary must never see a raise
        return _error(exc)


# --------------------------------------------------------------------------- #
# Tool 2: analyze_corrugation


def _analyze_corrugation(
    ctx: AppContext,
    dump: str,
    joint: str = "armLowerL",
    channel: str = "posed",
    save_plot: bool = False,
    bone_map_path: str | None = None,
) -> str:
    loaded: ply.EngineDump | None = None
    bone_map: dict[int, str] = {}
    try:
        dump_path = Path(dump)
        loaded = ply.load_dump_cached(dump_path, ctx.cache_dir)
        bone_map = _bone_map_of(dump_path)
        joint_ids = transforms.resolve_joint_ids(
            joint, bone_map=bone_map or None, bone_map_path=bone_map_path
        )
        mask = transforms.select_segment(loaded, joint_ids, channel=channel)
        sig, sectors, _frame = _signal_bundle(loaded, mask, channel)
        report = transforms.roughness(sig, sectors=sectors)
        freqs, psd = transforms.spectrum(sig)
        peaks = filters.spectral_peaks(freqs, psd, f_min_cpm=12.0)
        t_start, t_end, fraction = transforms.corrugation_extent(sig)
        plot_path: str | None = None
        if save_plot:
            out = ctx.plots_dir / f"corrugation_{_safe(dump_path.stem)}_{_safe(joint)}_{_safe(channel)}.png"
            plot_path = str(
                plots.save_signal_plot(sig, freqs, psd, out, f"{joint} [{channel}] {dump_path.name}")
            )
        return CorrugationReport(
            dump_path=str(dump_path),
            joint=joint,
            joint_ids=joint_ids,
            channel=channel,
            vertex_count=int(mask.sum()),
            n_samples=int(sig.y.size),
            t_span_m=sig.span_m,
            roughness=RoughnessSchema.from_report(report),
            spectral_peaks=[SpectralPeakSchema(freq_cpm=f, prominence_db=p) for f, p in peaks],
            extent_t_start_m=t_start,
            extent_t_end_m=t_end,
            extent_band_energy_fraction=fraction,
            plot_path=plot_path,
        ).model_dump_json()
    except Exception as exc:
        return _error(exc, hint=_histogram_hint(loaded, bone_map))


# --------------------------------------------------------------------------- #
# Tool 3: compare_geometry_signals


def _compare_geometry_signals(
    ctx: AppContext,
    dump_a: str,
    channel_a: str,
    dump_b: str,
    channel_b: str,
    joint: str = "armLowerL",
    save_plot: bool = False,
    bone_map_path: str | None = None,
) -> str:
    loaded_a: ply.EngineDump | None = None
    bone_map: dict[int, str] = {}
    try:
        path_a, path_b = Path(dump_a), Path(dump_b)
        loaded_a = ply.load_dump_cached(path_a, ctx.cache_dir)
        loaded_b = ply.load_dump_cached(path_b, ctx.cache_dir)
        # Resolve the joint PER DUMP: two skeletons may index the same bone name
        # differently, and a merged map would silently mask the wrong bone in one dump.
        map_a, map_b = _bone_map_of(path_a), _bone_map_of(path_b)
        bone_map = {**map_a, **map_b}  # for the error-hint histogram only
        joint_ids_a = transforms.resolve_joint_ids(
            joint, bone_map=(map_a or map_b) or None, bone_map_path=bone_map_path
        )
        joint_ids_b = transforms.resolve_joint_ids(
            joint, bone_map=(map_b or map_a) or None, bone_map_path=bone_map_path
        )
        mask_a = transforms.select_segment(loaded_a, joint_ids_a, channel=channel_a)
        mask_b = transforms.select_segment(loaded_b, joint_ids_b, channel=channel_b)
        sig_a, sectors_a, frame_a = _signal_bundle(loaded_a, mask_a, channel_a)
        sig_b, sectors_b, _frame_b = _signal_bundle(loaded_b, mask_b, channel_b)
        rough_a = transforms.roughness(sig_a, sectors=sectors_a)
        rough_b = transforms.roughness(sig_b, sectors=sectors_b)

        displacement: DisplacementStats | None = None
        if loaded_a.vertex_count == loaded_b.vertex_count:
            mask = mask_a & mask_b
            pa = loaded_a.channel(channel_a)[mask].astype(np.float64)
            pb = loaded_b.channel(channel_b)[mask].astype(np.float64)
            if pa.shape[0] > 0:
                delta = np.linalg.norm(pb - pa, axis=1)
                i_max = int(np.argmax(delta))
                t_at_max = float((pa[i_max] - frame_a.origin) @ frame_a.axis)
                displacement = DisplacementStats(
                    max_m=float(delta.max()),
                    mean_m=float(delta.mean()),
                    rms_m=float(np.sqrt(np.mean(delta**2))),
                    t_at_max_m=t_at_max,
                )

        plot_path: str | None = None
        freqs_a, psd_a = transforms.spectrum(sig_a)
        freqs_b, psd_b = transforms.spectrum(sig_b)
        if save_plot:
            name = (
                f"compare_{_safe(path_a.stem)}_{_safe(channel_a)}"
                f"_vs_{_safe(path_b.stem)}_{_safe(channel_b)}.png"
            )
            plot_path = str(
                plots.save_comparison_plot(
                    sig_a,
                    f"{path_a.stem}:{channel_a}",
                    freqs_a,
                    psd_a,
                    sig_b,
                    f"{path_b.stem}:{channel_b}",
                    freqs_b,
                    psd_b,
                    ctx.plots_dir / name,
                    f"{joint}: {channel_a} vs {channel_b}",
                )
            )
        return SignalComparison(
            dump_a=str(path_a),
            channel_a=channel_a,
            dump_b=str(path_b),
            channel_b=channel_b,
            joint=joint,
            joint_ids=sorted(set(joint_ids_a) | set(joint_ids_b)),
            roughness_a=RoughnessSchema.from_report(rough_a),
            roughness_b=RoughnessSchema.from_report(rough_b),
            delta_rel_ripple=rough_b.rel_ripple - rough_a.rel_ripple,
            delta_hf_energy_ratio=rough_b.hf_energy_ratio - rough_a.hf_energy_ratio,
            delta_peak_prominence_db=rough_b.peak_prominence_db - rough_a.peak_prominence_db,
            ripple_ratio_b_over_a=rough_b.rel_ripple / max(rough_a.rel_ripple, _TINY),
            displacement=displacement,
            plot_path=plot_path,
        ).model_dump_json()
    except Exception as exc:
        return _error(exc, hint=_histogram_hint(loaded_a, bone_map))


# --------------------------------------------------------------------------- #
# Tool 4: localize_defect


def _localize_defect(
    ctx: AppContext,
    dump: str,
    channel: str = "posed",
    min_verts: int = 100,
    top_k: int = 8,
    min_span_m: float = 0.06,
) -> str:
    try:
        dump_path = Path(dump)
        loaded = ply.load_dump_cached(dump_path, ctx.cache_dir)
        bone_map = _bone_map_of(dump_path)
        points_all = loaded.channel(channel)
        uniq, counts = np.unique(loaded.dominant_joint, return_counts=True)
        entries: list[SegmentRoughness] = []
        scanned = 0
        for j, c in zip(uniq.tolist(), counts.tolist(), strict=True):
            if int(c) < min_verts:
                continue
            scanned += 1
            pts = points_all[loaded.dominant_joint == int(j)]
            try:
                frame = transforms.fit_axis(pts)
                t, r, theta = transforms.cylindrical_coords(pts, frame)
                # Cylinder-model validity gate: tiny/round segments (fingers, eyes)
                # produce shape-not-defect "ripple" and would out-rank real limbs.
                if float(np.percentile(t, 95) - np.percentile(t, 5)) < min_span_m:
                    continue
                sig = transforms.axial_profile(t, r)
                sectors = transforms.sector_profiles(t, r, theta)
                report = transforms.roughness(sig, sectors=sectors)
            except ValueError:
                continue  # degenerate segment (e.g. near-planar patch) — not rankable
            # Score = spectral peak prominence alone. A DEFECT is a narrowband peak
            # (one frequency dominating); anatomical SHAPE is broadband ripple with
            # large rel_ripple but low prominence — weighting by rel_ripple ranks
            # eyes/feet above a genuinely corrugated forearm (calibrated on rigged3).
            score = max(report.peak_prominence_db, 0.0)
            entries.append(
                SegmentRoughness(
                    joint_id=int(j),
                    joint_name=bone_map.get(int(j)),
                    vertex_count=int(c),
                    score=score,
                    roughness=RoughnessSchema.from_report(report),
                )
            )
        entries.sort(key=lambda e: e.score, reverse=True)
        return LocalizeReport(
            dump_path=str(dump_path),
            channel=channel,
            min_verts=min_verts,
            joints_scanned=scanned,
            top=entries[: max(top_k, 0)],
        ).model_dump_json()
    except Exception as exc:
        return _error(exc)


# --------------------------------------------------------------------------- #
# Tool 5: lbs_differential


def _lbs_differential(
    ctx: AppContext,
    dump: str,
    palette: str | None = None,
    baked_json: str | None = None,
    joint: str = "armLowerL",
    weight_surgery: str = "none",
    bone_map_path: str | None = None,
) -> str:
    loaded: ply.EngineDump | None = None
    bone_map: dict[int, str] = {}
    try:
        if weight_surgery not in lbs.WEIGHT_SURGERY_MODES:
            return ToolError(
                error=f"unknown weight_surgery mode '{weight_surgery}'",
                hint=f"expected one of {list(lbs.WEIGHT_SURGERY_MODES)}",
            ).model_dump_json()

        dump_path = Path(dump)
        loaded = ply.load_dump_cached(dump_path, ctx.cache_dir)

        palette_path = Path(palette) if palette else dump_path.with_suffix(".palette.json")
        if not palette_path.is_file():
            return ToolError(
                error=f"palette sidecar not found: {palette_path}",
                hint=(
                    "pass palette= explicitly, or re-run extract_mesh_telemetry with "
                    "want_palette=True against a --palette-out capable (M5-patched) engine"
                ),
            ).model_dump_json()
        sidecar = lbs.load_palette_sidecar(palette_path)

        import_scale = sidecar.import_scale
        weights_source = "ply"
        ji, jw = loaded.joint_indices, loaded.joint_weights
        if ji is None or jw is None:
            if baked_json is None:
                return ToolError(
                    error="dump carries no j0..j3/w0..w3 vertex weights and no baked_json was given",
                    hint=(
                        "re-run extract_mesh_telemetry against a --weights capable (M5-patched) "
                        "engine, or pass baked_json= to use the asset's own weights"
                    ),
                ).model_dump_json()
            baked = lbs.load_baked_character(baked_json, cache_dir=ctx.cache_dir)
            if baked.vertex_count != loaded.vertex_count:
                raise ValueError(
                    f"baked_json has {baked.vertex_count} vertices but the dump has "
                    f"{loaded.vertex_count} — weights cannot be mapped row-by-row"
                )
            if abs(import_scale - 1.0) < 1e-9:
                try:
                    import_scale = lbs.recover_import_scale(loaded.rest, baked.positions)
                except ValueError:
                    import_scale = 1.0
            baked = lbs.align_json_to_engine(baked, import_scale)
            ji, jw = baked.joint_indices, baked.joint_weights
            weights_source = "baked_json"

        ji_used, jw_used = lbs.weight_surgery(ji, jw, weight_surgery, sidecar.bone_names)
        pure = lbs.lbs_pose(loaded.rest, ji_used, jw_used, sidecar.skinning_palette)
        residual = np.linalg.norm(loaded.posed.astype(np.float64) - pure.astype(np.float64), axis=1)

        bone_map = _bone_map_of(dump_path) or dict(enumerate(sidecar.bone_names))
        joint_ids = transforms.resolve_joint_ids(
            joint, bone_map=bone_map or None, bone_map_path=bone_map_path
        )
        mask = transforms.select_segment(loaded, joint_ids, channel="posed")
        masked_res = residual[mask]

        sig_eng, sectors_eng, _ = _signal_bundle(loaded, mask, "posed")
        rough_eng = transforms.roughness(sig_eng, sectors=sectors_eng)
        pure_pts = pure[mask]
        frame_p = transforms.fit_axis(pure_pts)
        t_p, r_p, theta_p = transforms.cylindrical_coords(pure_pts, frame_p)
        sig_pure = transforms.axial_profile(t_p, r_p)
        sectors_pure = transforms.sector_profiles(t_p, r_p, theta_p)
        rough_pure = transforms.roughness(sig_pure, sectors=sectors_pure)

        # Weight structure (Mode B) on the ORIGINAL weights of the segment.
        w = np.asarray(jw, dtype=np.float64)[mask]
        nz = w > 0.0
        counts_per_vertex = nz.sum(axis=1)
        hist_ids, hist_counts = np.unique(counts_per_vertex, return_counts=True)
        influence_histogram = {
            int(k): int(v) for k, v in zip(hist_ids.tolist(), hist_counts.tolist(), strict=True)
        }
        row_sums = np.maximum(w.sum(axis=1, keepdims=True), _TINY)
        p = np.where(nz, w / row_sums, 0.0)
        log_p = np.log(np.where(p > 0.0, p, 1.0))  # log(1)=0 keeps zero-weight slots silent
        entropy = -(p * log_p).sum(axis=1)
        mean_entropy = float(entropy.mean()) if entropy.size else 0.0
        seg_ids = np.unique(np.asarray(ji)[mask][nz]).tolist()
        n_names = len(sidecar.bone_names)
        contributing = [
            sidecar.bone_names[int(i)] if 0 <= int(i) < n_names else f"#{int(i)}" for i in seg_ids
        ]

        if rough_eng.corrugated and rough_pure.corrugated:
            if rough_eng.rel_ripple > 3.0 * max(rough_pure.rel_ripple, _TINY):
                verdict = "both"
            else:
                verdict = "weight-band"
        elif rough_eng.corrugated:
            verdict = "deformer-band"
        else:
            verdict = "neither"

        return LbsDifferentialReport(
            dump_path=str(dump_path),
            palette_path=str(palette_path),
            weights_source=weights_source,
            import_scale=import_scale,
            weight_surgery=weight_surgery,
            joint=joint,
            joint_ids=joint_ids,
            masked_vertex_count=int(mask.sum()),
            residual_global_max_m=float(residual.max()),
            residual_global_mean_m=float(residual.mean()),
            residual_global_rms_m=float(np.sqrt(np.mean(residual**2))),
            residual_masked_max_m=float(masked_res.max()),
            residual_masked_mean_m=float(masked_res.mean()),
            residual_masked_rms_m=float(np.sqrt(np.mean(masked_res**2))),
            roughness_engine=RoughnessSchema.from_report(rough_eng),
            roughness_pure_lbs=RoughnessSchema.from_report(rough_pure),
            influence_histogram=influence_histogram,
            mean_weight_entropy=mean_entropy,
            contributing_bones=contributing,
            verdict=verdict,
        ).model_dump_json()
    except Exception as exc:
        return _error(exc, hint=_histogram_hint(loaded, bone_map))


def _score_bake(
    ctx: AppContext,
    bake_dir: str,
    reference_glob: str,
    frame: int = 8,
    person: str = "p1",
    iou_target: float = scoring.IOU_TARGET,
    hu_target: float = scoring.HU_TARGET,
) -> str:
    try:
        base = Path(bake_dir)
        depth = base / "depth" / f"frame_{int(frame):05d}.png"
        mask = base / "mask" / f"frame_{int(frame):05d}_{person}.png"
        result = scoring.score_bake(
            depth,
            mask,
            reference_glob,
            iou_target=iou_target,
            hu_target=hu_target,
        )
        return BakeScoreReport(
            bake_depth=result.bake_depth,
            bake_mask=result.bake_mask,
            n_refs=result.n_refs,
            iou_best=result.iou_best,
            iou_mean=result.iou_mean,
            hu_best=result.hu_best,
            hu_mean=result.hu_mean,
            edge_corr_best=result.edge_corr_best,
            edge_corr_mean=result.edge_corr_mean,
            corrugation_db=result.corrugation_db,
            corrugation_freq_cpp=result.corrugation_freq_cpp,
            corrugation_orientation_deg=result.corrugation_orientation_deg,
            pose_ok=result.pose_ok,
            mesh_ok=result.mesh_ok,
            verdict=result.verdict,
            top_matches=[
                BakeRefMatch(ref=m.ref, iou=m.iou, hu=m.hu, edge_corr=m.edge_corr) for m in result.top_matches
            ],
        ).model_dump_json()
    except Exception as exc:
        return _error(
            exc,
            hint="expects an engine bake dir with depth/frame_<n>.png + mask/frame_<n>_<person>.png; "
            "the other solo-mask person is blank — pass person=p0 if p1 is empty",
        )


# --------------------------------------------------------------------------- #
# Tool 7: analyze_mesh_topology (local response model — mirrors engine.topology)


class MeshTopologyReport(SchemaBase):
    """analyze_mesh_topology: whole-mesh topology QA on a dump's face connectivity.

    Mirrors :class:`dsp_server.engine.topology.MeshTopology` and, when a vertex
    pair is supplied, carries the surface geodesic distance/hops between them."""

    dump_path: str
    n_vertices_total: int
    n_vertices_referenced: int
    n_isolated_vertices: int
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
    estimated_genus: int | None = None
    geodesic_m: float | None = None
    geodesic_hops: int | None = None


def _analyze_mesh_topology(
    ctx: AppContext,
    dump: str,
    src_vertex: int | None = None,
    dst_vertex: int | None = None,
) -> str:
    try:
        dump_path = Path(dump)
        loaded = ply.load_dump_cached(dump_path, ctx.cache_dir)
        topo = topology.analyze_topology(loaded.faces, loaded.vertex_count)

        geodesic_m: float | None = None
        geodesic_hops: int | None = None
        if src_vertex is not None and dst_vertex is not None:
            n = loaded.vertex_count
            if not (0 <= src_vertex < n and 0 <= dst_vertex < n):
                return ToolError(
                    error=f"src_vertex/dst_vertex out of range [0, {n})",
                    hint=f"this dump has {n} vertices — pass 0-based indices below {n}",
                ).model_dump_json()
            geodesic_m, geodesic_hops = topology.geodesic_between(
                loaded.faces, loaded.posed, src_vertex, dst_vertex
            )

        return MeshTopologyReport(
            dump_path=str(dump_path),
            n_vertices_total=loaded.vertex_count,
            n_vertices_referenced=topo.n_vertices_referenced,
            n_isolated_vertices=topo.n_isolated_vertices,
            n_faces=topo.n_faces,
            n_edges=topo.n_edges,
            n_components=topo.n_components,
            largest_component_fraction=topo.largest_component_fraction,
            component_sizes=topo.component_sizes,
            boundary_edge_count=topo.boundary_edge_count,
            nonmanifold_edge_count=topo.nonmanifold_edge_count,
            is_manifold=topo.is_manifold,
            is_watertight=topo.is_watertight,
            valence_min=topo.valence_min,
            valence_max=topo.valence_max,
            valence_mean=topo.valence_mean,
            valence_histogram=topo.valence_histogram,
            n_irregular_valence=topo.n_irregular_valence,
            euler_characteristic=topo.euler_characteristic,
            estimated_genus=topo.estimated_genus,
            geodesic_m=geodesic_m,
            geodesic_hops=geodesic_hops,
        ).model_dump_json()
    except Exception as exc:
        return _error(exc)


# --------------------------------------------------------------------------- #
# Registration


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the geometry tools on ``mcp``, bound to ``ctx``."""

    @mcp.tool()
    def extract_mesh_telemetry(
        character_json: str | None = None,
        sex: str = "male",
        pose_clip: str | None = "clip-cin-stand-attention",
        label: str | None = None,
        want_palette: bool = True,
    ) -> str:
        """Run the engine's layered_field_dump_cli and summarize the resulting posed PLY dump.
        Writes the dump (plus .meta.json and, on a patched engine, .palette.json sidecars) under
        data/dumps/ and returns TelemetryResult JSON: vertex/face/bone counts, clip id and sample
        time, deformer collision-push magnitudes in meters, the full bone map, a per-dominant-joint
        vertex histogram of the arm chain (arm/hand/clavicle joints), an engine-binary staleness
        report, and all output paths. character_json imports a baked character (requires the
        M5-patched engine — a structured ToolError names the missing --character flag otherwise);
        sex and pose_clip select the procedural body and animation clip (pose_clip=null dumps the
        rest field); label overrides the output file stem. On failure returns ToolError JSON with
        the engine's stderr tail."""
        return _extract_mesh_telemetry(ctx, character_json, sex, pose_clip, label, want_palette)

    @mcp.tool()
    def analyze_corrugation(
        dump: str,
        joint: str = "armLowerL",
        channel: str = "posed",
        save_plot: bool = False,
        bone_map_path: str | None = None,
    ) -> str:
        """Analyze one engine PLY dump for periodic radial corrugation on the segment whose
        dominant joint is `joint` (a bone name resolved via the dump's .meta.json bone map or
        bone_map_path, or an integer id as a string). channel selects 'posed' (deformed) or 'rest'
        (bind) positions. Returns CorrugationReport JSON: a roughness verdict (rel_ripple,
        hf_energy_ratio, dominant frequency in cycles/meter with wavelength in meters, peak
        prominence in dB, ridge count estimate, sector phase coherence, corrugated bool), the top
        spectral peaks, the axial extent [t_start, t_end] in meters where the corrugation band
        carries energy (line this up against capsule endpoints from --list-sources), and the saved
        plot path when save_plot is true. On a bad joint the ToolError hint lists the dump's
        per-joint vertex histogram."""
        return _analyze_corrugation(ctx, dump, joint, channel, save_plot, bone_map_path)

    @mcp.tool()
    def compare_geometry_signals(
        dump_a: str,
        channel_a: str,
        dump_b: str,
        channel_b: str,
        joint: str = "armLowerL",
        save_plot: bool = False,
        bone_map_path: str | None = None,
    ) -> str:
        """Compare two axial profile signals on the segment dominated by `joint` — e.g. the SAME
        dump's 'rest' vs 'posed' channels (one engine posed dump carries both), or two different
        dumps/clips. Returns SignalComparison JSON with a full RoughnessSchema per side, deltas
        (b minus a) for rel_ripple / hf_energy_ratio / peak_prominence_db, and the b/a ripple
        ratio; when both dumps have the same vertex count it additionally reports per-vertex
        displacement stats over the joint mask (max/mean/rms of ||delta p|| in meters and the
        axial t-location of the maximum). save_plot writes an overlay figure of both profiles and
        spectra and returns its path."""
        return _compare_geometry_signals(
            ctx, dump_a, channel_a, dump_b, channel_b, joint, save_plot, bone_map_path
        )

    @mcp.tool()
    def localize_defect(
        dump: str,
        channel: str = "posed",
        min_verts: int = 100,
        top_k: int = 8,
        min_span_m: float = 0.06,
    ) -> str:
        """Answer 'is the ripple forearm-only or systemic?' in one call: every joint with at
        least min_verts dominant vertices and at least min_span_m of axial extent gets its own
        axis fit, axial profile, and roughness report, and the segments are ranked by spectral
        peak prominence alone (a DEFECT is a narrowband peak; anatomical shape is broadband
        ripple that would out-rank it under any rel_ripple weighting). Returns LocalizeReport
        JSON with the top_k segments (joint id, name when the bone map is known, vertex count,
        score, full RoughnessSchema), most corrugated first."""
        return _localize_defect(ctx, dump, channel, min_verts, top_k, min_span_m)

    @mcp.tool()
    def lbs_differential(
        dump: str,
        palette: str | None = None,
        baked_json: str | None = None,
        joint: str = "armLowerL",
        weight_surgery: str = "none",
        bone_map_path: str | None = None,
    ) -> str:
        """Isolate WHERE the corrugation is injected: reconstruct pure linear-blend skinning from
        the dump's rest positions x skin weights x the palette sidecar (default: <dump>.palette.json
        next to the dump; weights come from a --weights dump or, as fallback, from baked_json with
        import-scale alignment/recovery), then compare against the engine's posed output. Returns
        LbsDifferentialReport JSON: residual ||posed - pureLBS|| stats globally and over the joint
        mask, roughness of the engine-posed vs pure-LBS profiles, weight-structure stats for the
        segment (influence-count histogram, mean weight entropy, contributing bone names), and a
        verdict hint — 'weight-band' (both corrugated), 'deformer-band' (engine only), 'both'
        (both corrugated but engine ripple > 3x pure), 'neither'. weight_surgery in
        {none, top1, drop_hand, drop_fingers} runs a Mode-B experiment on the weights before
        posing."""
        return _lbs_differential(ctx, dump, palette, baked_json, joint, weight_surgery, bone_map_path)

    @mcp.tool()
    def score_bake(
        bake_dir: str,
        reference_glob: str,
        frame: int = 8,
        person: str = "p1",
        iou_target: float = scoring.IOU_TARGET,
        hu_target: float = scoring.HU_TARGET,
    ) -> str:
        """Score one posed engine bake against a library reference pack — the loop's
        inner metric. Reads the bake's depth/frame_<frame>.png and mask/frame_<frame>_<person>.png
        (solo bakes raster one body; pass person='p0' if 'p1' is the blank one) and every
        *_depth.png matched by reference_glob. POSE lane (the reliable gate): silhouette IoU +
        Hu-moment shape distance (matchShapes-I2, lower is closer) + Sobel edge-orientation
        correlation, each facing-normalized as best-of-mirror against every reference, reported
        as best/mean plus the top matching refs. MESH lane (ADVISORY): band-limited 2-D FFT
        prominence (dB) of the bake depth over the body ROI — a coarse render-side corrugation
        proxy that does NOT resolve the ~8 mm edge-loop ripple (use analyze_corrugation on the
        vertex dump for that). verdict is pose-driven ('pass' when iou_best>=iou_target and
        hu_best<=hu_target, else 'pose-weak'), with a trailing '*' when advisory mesh corrugation
        is high. Returns BakeScoreReport JSON, or ToolError JSON on failure."""
        return _score_bake(ctx, bake_dir, reference_glob, frame, person, iou_target, hu_target)

    @mcp.tool()
    def analyze_mesh_topology(
        dump: str,
        src_vertex: int | None = None,
        dst_vertex: int | None = None,
    ) -> str:
        """Whole-mesh topology QA on one engine PLY dump's face connectivity — the topology
        counterpart to analyze_corrugation (which measures the periodic radial ripple on a single
        JOINT's segment; this instead audits the entire triangle mesh and takes no joint). Loads
        the dump and inspects its (m,3) face index array for the damage a bad source-mesh bake
        leaves behind — non-manifold edges (shared by >2 triangles), boundary edges (an open shell,
        each in exactly 1 triangle), disconnected islands (connected components over the referenced
        vertices), irregular vertex valence (degree != 6), the Euler characteristic V-E+F and, when
        watertight, an estimated genus, plus face-less isolated vertices. Returns MeshTopologyReport
        JSON with is_manifold/is_watertight verdicts, connected-component sizes (desc, capped),
        min/max/mean valence and a valence histogram; when both src_vertex and dst_vertex are given
        it adds the edge-length-weighted surface geodesic distance in meters (Dijkstra over the posed
        positions) and its hop count. On failure returns ToolError JSON."""
        return _analyze_mesh_topology(ctx, dump, src_vertex, dst_vertex)

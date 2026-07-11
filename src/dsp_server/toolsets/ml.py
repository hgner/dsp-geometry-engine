"""ML tool pack (ELE489): features, clustering, reduction, classification, prediction.

Five tools over any tabular file (.csv/.tsv/.json/.npz/.npy) or an engine PLY dump.
Matrices NEVER cross the MCP boundary: feature_engineer_dump / cluster / reduce_dims
/ predict write npz artifacts under ``data/features/`` and classify_eval persists a
joblib pipeline + ``.meta.json`` sidecar under ``data/models/`` (both pack-local
dirs created on demand — the frozen AppContext is not extended). ``predict``
refuses any model outside ``data/models/`` or lacking the server-written sidecar,
because joblib deserialization executes code. Every tool returns a pydantic
``model_dump_json()`` string; every failure returns a :class:`ToolError` whose
hint carries the tabular loader's found-columns message verbatim.

Testability: each registered tool is a thin closure over :class:`AppContext`
delegating to a module-level ``_impl`` function that tests call directly.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from mcp.server.fastmcp import FastMCP
from pydantic import ConfigDict

from dsp_server import plots
from dsp_server.engine import mlkit, ply, tabular, transforms
from dsp_server.engine.tabular import TabularError
from dsp_server.schemas import SchemaBase, ToolError
from dsp_server.toolsets import AppContext

_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]")
_PREVIEW_CAP = 10
_MAX_CLASSES = 20
_MAX_INLINE_EVR = 16
_MAX_INLINE_COMPONENTS = 8
_MAX_INLINE_SIZES = 32

# Critique #22: one params type reused by cluster AND classify_eval (kernel is a str).
ParamsUsed = dict[str, float | str | int | None]


# --------------------------------------------------------------------------- #
# Response models (pack-local, on the shared SchemaBase contract)


class FeatureMatrixReport(SchemaBase):
    """feature_engineer_dump: per-vertex feature matrix summary (matrix on disk)."""

    npz_path: str
    dump_path: str
    joint: str | None = None
    joint_ids: list[int]
    channel: str
    n_vertices: int
    n_features: int
    feature_names: list[str]
    omitted: dict[str, str]
    feature_stats: dict[str, list[float]]  # name -> [min, mean, max]


class ClusterReport(SchemaBase):
    """cluster: one clustering run + internal validity metrics (labels on disk)."""

    path: str
    algorithm: str
    params_used: ParamsUsed
    standardize: bool
    n_samples: int
    n_features: int
    n_clusters_found: int
    cluster_sizes: dict[str, int]  # largest first, capped at 32 entries inline
    silhouette: float | None = None
    silhouette_subsampled: bool = False
    davies_bouldin: float | None = None
    inertia: float | None = None
    n_noise: int | None = None
    labels_path: str
    plot_path: str | None = None


class ReduceDimsReport(SchemaBase):
    """reduce_dims: PCA/LDA embedding summary (full Z/components in the npz)."""

    path: str
    method: str
    n_components: int
    n_samples: int
    n_features: int
    explained_variance_ratio: list[float]  # first 16 components inline
    cumulative_evr: float
    top_loadings: dict[str, dict[str, float]]  # first 8 components inline
    embedding_path: str
    plot_path: str | None = None


class CvMetricsSchema(SchemaBase):
    accuracy_mean: float
    accuracy_std: float
    precision_macro_mean: float
    precision_macro_std: float
    recall_macro_mean: float
    recall_macro_std: float
    f1_macro_mean: float
    f1_macro_std: float


class ClassifyEvalReport(SchemaBase):
    """classify_eval: stratified CV metrics + persisted pipeline paths."""

    # merged with the base config; 'model'/'model_path' are legitimate field names here
    model_config = ConfigDict(protected_namespaces=())

    path: str
    model: str
    params_used: ParamsUsed
    standardize: bool
    n_samples: int
    n_features: int
    feature_columns: list[str]
    label_column: str
    classes: list[str]
    class_counts: dict[str, int]
    cv_folds_used: int
    metrics: CvMetricsSchema
    confusion_matrix: list[list[int]]  # rows = true classes in `classes` order
    model_path: str
    meta_path: str


class PredictReport(SchemaBase):
    """predict: predictions summary (full y_pred/y_proba in the npz)."""

    # merged with the base config; 'model'/'model_path' are legitimate field names here
    model_config = ConfigDict(protected_namespaces=())

    model_path: str
    model: str
    path: str
    n_rows: int
    predicted_class_counts: dict[str, int]
    preview: list[str]  # first <= 10 predicted labels
    accuracy_vs_column: float | None = None
    predictions_path: str


# --------------------------------------------------------------------------- #
# Shared helpers


def _safe(text: str) -> str:
    return _SAFE_RE.sub("_", text) or "x"


def _error(exc: Exception, hint: str | None = None) -> str:
    return ToolError(error=f"{type(exc).__name__}: {exc}", hint=hint).model_dump_json()


def _features_dir(ctx: AppContext) -> Path:
    """Pack-local artifact dir (critique #18: deliverables never go to data/cache/)."""
    d = ctx.data_dir / "features"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _models_dir(ctx: AppContext) -> Path:
    d = ctx.data_dir / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _models_hint(ctx: AppContext) -> str:
    mdir = _models_dir(ctx)
    entries = sorted(p.name for p in mdir.glob("*.joblib"))
    listing = ", ".join(entries) if entries else "(no models yet — run classify_eval first)"
    return (
        f"predict only loads models this server wrote under {mdir} "
        f"(joblib deserialization executes code); models present: {listing}"
    )


def _histogram_hint(dump: ply.EngineDump | None, bone_map: dict[int, str]) -> str:
    if dump is None:
        return "could not load the dump — check the path points at an engine ASCII PLY"
    uniq, counts = np.unique(dump.dominant_joint, return_counts=True)
    parts = []
    for j, c in zip(uniq.tolist(), counts.tolist(), strict=True):
        name = bone_map.get(int(j))
        parts.append(f"{name} ({int(j)})={int(c)}" if name else f"{int(j)}={int(c)}")
    return (
        "per-joint vertex histogram of this dump: {" + ", ".join(parts) + "} — "
        "pick a joint name or integer id from it"
    )


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _drop_all_nan_columns(x: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Drop feature columns that are entirely non-finite — these are the string
    columns tabular.load_matrix NaN-ifies (e.g. an unselected 'label' column in a
    labeled CSV). Without this, the all-NaN column would make the per-row finite
    filter reject every row. Mirrors classify_eval's column handling."""
    if x.size == 0:
        return x, names
    keep_cols = np.isfinite(x).any(axis=0)
    if keep_cols.all():
        return x, names
    kept_names = [n for n, k in zip(names, keep_cols, strict=True) if k]
    return x[:, keep_cols], kept_names


def _match_name(names: list[str], column: str | int) -> str | None:
    """Resolve a column reference against a name list: exact, then unique
    case-insensitive, then integer index. None when nothing matches."""
    if isinstance(column, str) and column in names:
        return column
    if isinstance(column, str):
        ci = [n for n in names if n.lower() == column.lower()]
        if len(ci) == 1:
            return ci[0]
    if isinstance(column, int) or (isinstance(column, str) and column.isdigit()):
        idx = int(column)
        if 0 <= idx < len(names):
            return names[idx]
    return None


def _read_delimited_labels(path: Path, column: str | int, delimiter: str) -> np.ndarray:
    """String-preserving label reader for .csv/.tsv — the shared tabular loader
    parses delimited files as float64 (string classes become NaN), so labels
    take this raw path instead. Numeric columns still come back float64."""
    rows: list[list[str]] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        for rec in csv.reader(fh, delimiter=delimiter):
            if not rec or rec[0].lstrip().startswith("#"):
                continue
            if all(not cell.strip() for cell in rec):
                continue
            rows.append([cell.strip() for cell in rec])
    if not rows:
        raise TabularError(f"no rows parsed from {path}")
    header = rows[0]
    has_header = not all(_is_float(tok) for tok in header if tok != "")
    names = header if has_header else [f"col{i}" for i in range(len(header))]
    data_rows = rows[1:] if has_header else rows
    match = _match_name(names, column)
    if match is None:
        raise TabularError(
            f"column '{column}' not found — label columns present: {names} "
            f"(pass a name or an integer index 0..{len(names) - 1})"
        )
    idx = names.index(match)
    values = [row[idx] if idx < len(row) else "" for row in data_rows]
    if all(_is_float(v) for v in values if v != ""):
        return np.asarray([float(v) if v != "" else np.nan for v in values], dtype=np.float64)
    return np.asarray(values)


def _load_labels_any(path: Path, column: str | int, key: str | None) -> np.ndarray:
    sfx = path.suffix.lower()
    if sfx in (".csv", ".tsv"):
        return _read_delimited_labels(path, column, "," if sfx == ".csv" else "\t")
    return tabular.load_labels(path, column, key=key)


def _scatter_coords(x: np.ndarray, names: list[str], standardize: bool, seed: int) -> np.ndarray:
    """(n, 2) plot coordinates: seeded 2-component PCA (index padding for 1-D data)."""
    if x.shape[1] < 2:
        return np.column_stack([x[:, 0], np.arange(x.shape[0], dtype=np.float64)])
    return mlkit.run_reduce(x, names, "pca", 2, None, standardize, seed).z


def _save_scatter(z: np.ndarray, labels: np.ndarray | None, path: Path, title: str) -> Path:
    plt = plots.plt  # dsp_server.plots locked the Agg backend before pyplot was imported
    path.parent.mkdir(parents=True, exist_ok=True)
    z = np.asarray(z, dtype=np.float64)
    x0 = z[:, 0]
    x1 = z[:, 1] if z.shape[1] > 1 else np.arange(z.shape[0], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    if labels is None:
        ax.scatter(x0, x1, s=8, alpha=0.7, color="tab:blue")
    else:
        uniq = np.unique(labels)
        if uniq.size <= 12:
            for lab in uniq:
                sel = labels == lab
                ax.scatter(x0[sel], x1[sel], s=8, alpha=0.7, label=str(lab))
            ax.legend(loc="best", fontsize=8, title="label")
        else:
            codes = np.searchsorted(uniq, labels)
            ax.scatter(x0, x1, c=codes, s=8, alpha=0.7, cmap="tab20")
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2" if z.shape[1] > 1 else "sample index")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _cluster_params(algorithm: str, n_clusters: int, eps: float, min_samples: int, seed: int) -> ParamsUsed:
    if algorithm == "kmeans":
        return {"n_clusters": n_clusters, "n_init": 10, "seed": seed}
    if algorithm == "dbscan":
        return {"eps": eps, "min_samples": min_samples}
    return {"n_clusters": n_clusters, "linkage": "ward"}


def _classifier_params(
    model: str, k: int, max_depth: int | None, c: float, kernel: str, n_estimators: int, seed: int
) -> ParamsUsed:
    if model == "knn":
        return {"k": k}
    if model == "naive_bayes":
        return {}
    if model == "tree":
        return {"max_depth": max_depth, "seed": seed}
    if model == "svm":
        return {"c": c, "kernel": kernel, "seed": seed}
    return {"n_estimators": n_estimators, "seed": seed}  # forest | gradient_boost


# --------------------------------------------------------------------------- #
# Tool 1: feature_engineer_dump


def _feature_engineer_dump(
    ctx: AppContext,
    dump: str,
    joint: str | None = None,
    channel: str = "posed",
    features: list[str] | None = None,
    bone_map_path: str | None = None,
) -> str:
    loaded: ply.EngineDump | None = None
    bone_map: dict[int, str] = {}
    try:
        dump_path = Path(dump)
        loaded = ply.load_dump_cached(dump_path, ctx.cache_dir)
        bone_map = ply.bone_map_for(dump_path)

        groups = list(features) if features else list(mlkit.FEATURE_GROUPS)
        unknown = [g for g in groups if g not in mlkit.FEATURE_GROUPS]
        if unknown:
            return ToolError(
                error=f"unknown feature group(s) {unknown}",
                hint=f"valid groups: {list(mlkit.FEATURE_GROUPS)}",
            ).model_dump_json()
        if features and "cylinder" in groups and joint is None:
            return ToolError(
                error="feature group 'cylinder' requires joint=",
                hint=(
                    "t/r/theta come from a cylinder frame fitted to one joint segment — "
                    "pass joint= a bone name or integer id (or drop 'cylinder')"
                ),
            ).model_dump_json()

        joint_ids: list[int] = []
        if joint is not None:
            joint_ids = transforms.resolve_joint_ids(
                joint, bone_map=bone_map or None, bone_map_path=bone_map_path
            )
        matrix, omitted = mlkit.dump_features(loaded, joint_ids or None, channel, groups)

        label = _safe(str(joint)) if joint is not None else "all"
        npz_path = _features_dir(ctx) / f"features_{_safe(dump_path.stem)}_{label}.npz"
        np.savez_compressed(
            npz_path,
            X=matrix.x,
            feature_names=np.asarray(matrix.names),
            dominant_joint=loaded.dominant_joint[matrix.vertex_index].astype(np.int32),
            vertex_index=matrix.vertex_index,
        )
        feature_stats = {
            name: [float(col.min()), float(col.mean()), float(col.max())]
            for name, col in zip(matrix.names, matrix.x.T, strict=True)
        }
        return FeatureMatrixReport(
            npz_path=str(npz_path),
            dump_path=str(dump_path),
            joint=str(joint) if joint is not None else None,
            joint_ids=joint_ids,
            channel=channel,
            n_vertices=int(matrix.x.shape[0]),
            n_features=int(matrix.x.shape[1]),
            feature_names=matrix.names,
            omitted=omitted,
            feature_stats=feature_stats,
        ).model_dump_json()
    except TabularError as exc:
        return _error(exc, hint=str(exc))
    except Exception as exc:
        return _error(exc, hint=_histogram_hint(loaded, bone_map))


# --------------------------------------------------------------------------- #
# Tool 2: cluster


def _cluster(
    ctx: AppContext,
    path: str,
    columns: list[str] | None = None,
    key: str | None = None,
    algorithm: str = "kmeans",
    n_clusters: int = 3,
    eps: float = 0.5,
    min_samples: int = 10,
    standardize: bool = True,
    seed: int = 0,
    save_plot: bool = False,
) -> str:
    try:
        src = Path(path)
        if algorithm not in mlkit.CLUSTER_ALGORITHMS:
            return ToolError(
                error=f"unknown algorithm '{algorithm}'",
                hint=f"expected one of {list(mlkit.CLUSTER_ALGORITHMS)}",
            ).model_dump_json()
        x, names = tabular.load_matrix(src, columns=columns, key=key, cache_dir=ctx.cache_dir)
        x, names = _drop_all_nan_columns(x, names)  # exclude non-numeric columns (e.g. a label col)
        if x.shape[1] == 0:
            return ToolError(
                error=f"no numeric feature columns in {src.name}",
                hint=f"columns found: {tabular.columns_of(src, key=key)}",
            ).model_dump_json()
        keep = np.isfinite(x).all(axis=1)
        x = x[keep]
        if x.shape[0] < 3:
            return ToolError(
                error=f"only {x.shape[0]} fully-finite rows in {src.name} — need >= 3",
                hint="drop or impute NaN rows before clustering",
            ).model_dump_json()
        result = mlkit.run_cluster(x, algorithm, n_clusters, eps, min_samples, standardize, seed)

        # Critique #13: re-open the source npz to carry the mesh row mapping into
        # the labels artifact (load_matrix returns only (X, names)).
        passthrough: dict[str, np.ndarray] = {}
        if src.suffix.lower() == ".npz":
            try:
                with np.load(src, allow_pickle=False) as z:
                    for member in ("vertex_index", "dominant_joint"):
                        if member in z.files and z[member].shape[0] == keep.size:
                            passthrough[member] = np.asarray(z[member])[keep]
            except (OSError, ValueError):
                passthrough = {}
        labels_path = _features_dir(ctx) / f"clusters_{_safe(src.stem)}_{_safe(algorithm)}.npz"
        np.savez_compressed(labels_path, labels=result.labels.astype(np.int32), **passthrough)

        plot_path: str | None = None
        if save_plot:
            coords = _scatter_coords(x, names, standardize, seed)
            out = ctx.plots_dir / f"clusters_{_safe(src.stem)}_{_safe(algorithm)}.png"
            plot_path = str(_save_scatter(coords, result.labels, out, f"{algorithm} — {src.name}"))

        sizes_sorted = sorted(result.cluster_sizes.items(), key=lambda kv: (-kv[1], kv[0]))
        return ClusterReport(
            path=str(src),
            algorithm=algorithm,
            params_used=_cluster_params(algorithm, n_clusters, eps, min_samples, seed),
            standardize=standardize,
            n_samples=int(x.shape[0]),
            n_features=int(x.shape[1]),
            n_clusters_found=result.n_clusters_found,
            cluster_sizes={str(k): int(v) for k, v in sizes_sorted[:_MAX_INLINE_SIZES]},
            silhouette=result.silhouette,
            silhouette_subsampled=result.subsampled,
            davies_bouldin=result.davies_bouldin,
            inertia=result.inertia,
            n_noise=result.n_noise,
            labels_path=str(labels_path),
            plot_path=plot_path,
        ).model_dump_json()
    except TabularError as exc:
        return _error(exc, hint=str(exc))
    except Exception as exc:
        return _error(exc)


# --------------------------------------------------------------------------- #
# Tool 3: reduce_dims


def _reduce_dims(
    ctx: AppContext,
    path: str,
    columns: list[str] | None = None,
    key: str | None = None,
    method: str = "pca",
    n_components: int = 2,
    label_column: str | None = None,
    standardize: bool = True,
    seed: int = 0,
    save_plot: bool = False,
) -> str:
    try:
        src = Path(path)
        if method not in mlkit.REDUCE_METHODS:
            return ToolError(
                error=f"unknown method '{method}'",
                hint=f"expected one of {list(mlkit.REDUCE_METHODS)}",
            ).model_dump_json()
        if method == "lda" and label_column is None:
            return ToolError(
                error="method='lda' requires label_column=",
                hint="lda is a supervised projection — pass label_column= the class column",
            ).model_dump_json()

        labels_arr: np.ndarray | None = None
        if label_column is not None:
            labels_arr = _load_labels_any(src, label_column, key)
        x, names = tabular.load_matrix(src, columns=columns, key=key, cache_dir=ctx.cache_dir)
        if label_column is not None:
            match = _match_name(names, label_column)
            if match is not None:  # never let the label leak into the feature matrix
                x = np.delete(x, names.index(match), axis=1)
                names = [n for n in names if n != match]
        # Drop any remaining non-numeric (all-NaN) columns so they don't reject every row.
        x, names = _drop_all_nan_columns(x, names)
        if x.shape[1] == 0:
            return ToolError(
                error=f"no feature columns left in {src.name} after excluding '{label_column}'",
                hint=f"columns found: {tabular.columns_of(src, key=key)}",
            ).model_dump_json()
        if labels_arr is not None and labels_arr.shape[0] != x.shape[0]:
            return ToolError(
                error=(f"label column has {labels_arr.shape[0]} rows but the matrix has {x.shape[0]}"),
                hint="labels must come from the same file rows as the features",
            ).model_dump_json()

        keep = np.isfinite(x).all(axis=1)
        if labels_arr is not None and labels_arr.dtype.kind == "f":
            keep &= np.isfinite(labels_arr)
        x = x[keep]
        labels_kept = labels_arr[keep] if labels_arr is not None else None
        if x.shape[0] < 3:
            return ToolError(
                error=f"only {x.shape[0]} fully-finite rows in {src.name} — need >= 3",
                hint="drop or impute NaN rows before reducing",
            ).model_dump_json()

        result = mlkit.run_reduce(x, names, method, n_components, labels_kept, standardize, seed)

        emb_path = _features_dir(ctx) / f"reduced_{_safe(src.stem)}_{_safe(method)}.npz"
        arrays: dict[str, np.ndarray] = {
            "Z": result.z,
            "components": result.components,
            "feature_names": np.asarray(names),
        }
        if labels_kept is not None:
            arrays["labels"] = (
                labels_kept if labels_kept.dtype.kind == "f" else np.asarray([str(v) for v in labels_kept])
            )
        np.savez_compressed(emb_path, **arrays)

        plot_path: str | None = None
        if save_plot:
            coords = (
                result.z[:, :2]
                if result.z.shape[1] > 1
                else np.column_stack([result.z[:, 0], np.arange(x.shape[0], dtype=np.float64)])
            )
            out = ctx.plots_dir / f"reduced_{_safe(src.stem)}_{_safe(method)}.png"
            plot_path = str(_save_scatter(coords, labels_kept, out, f"{method} — {src.name}"))

        return ReduceDimsReport(
            path=str(src),
            method=method,
            n_components=n_components,
            n_samples=int(x.shape[0]),
            n_features=int(x.shape[1]),
            explained_variance_ratio=result.evr[:_MAX_INLINE_EVR],
            cumulative_evr=float(sum(result.evr)),
            top_loadings=dict(list(result.top_loadings.items())[:_MAX_INLINE_COMPONENTS]),
            embedding_path=str(emb_path),
            plot_path=plot_path,
        ).model_dump_json()
    except TabularError as exc:
        return _error(exc, hint=str(exc))
    except Exception as exc:
        return _error(exc)


# --------------------------------------------------------------------------- #
# Tool 4: classify_eval


def _classify_eval(
    ctx: AppContext,
    path: str,
    label_column: str,
    feature_columns: list[str] | None = None,
    key: str | None = None,
    model: str = "knn",
    cv_folds: int = 5,
    k: int = 5,
    max_depth: int | None = None,
    c: float = 1.0,
    kernel: str = "rbf",
    n_estimators: int = 100,
    standardize: bool = True,
    seed: int = 0,
) -> str:
    try:
        src = Path(path)
        if model not in mlkit.CLASSIFIER_MODELS:
            return ToolError(
                error=f"unknown model '{model}'",
                hint=f"expected one of {list(mlkit.CLASSIFIER_MODELS)}",
            ).model_dump_json()
        y = _load_labels_any(src, label_column, key)

        if feature_columns is not None:
            x, names = tabular.load_matrix(src, columns=feature_columns, key=key, cache_dir=ctx.cache_dir)
            leak = _match_name(names, label_column)
            if leak is not None:
                return ToolError(
                    error=f"label column '{leak}' is also listed in feature_columns",
                    hint="remove it — training a classifier on its own label is leakage",
                ).model_dump_json()
        else:
            table = tabular.load_table(src, key=key, cache_dir=ctx.cache_dir)
            match = _match_name(list(table), label_column)
            if match is not None:
                table.pop(match)
            # String columns in delimited files parse to all-NaN — never features.
            table = {n: v for n, v in table.items() if np.isfinite(v).any()}
            names = list(table)
            if not names:
                return ToolError(
                    error=f"no numeric feature columns in {src.name} besides '{label_column}'",
                    hint=f"columns found: {tabular.columns_of(src, key=key)}",
                ).model_dump_json()
            x = np.column_stack([table[n] for n in names])

        if y.shape[0] != x.shape[0]:
            return ToolError(
                error=f"label column has {y.shape[0]} rows but the feature matrix has {x.shape[0]}",
                hint="labels must come from the same file rows as the features",
            ).model_dump_json()
        keep = np.isfinite(x).all(axis=1)
        if y.dtype.kind == "f":
            keep &= np.isfinite(y)
        x, y = x[keep], y[keep]
        if x.shape[0] < 4:
            return ToolError(
                error=f"only {x.shape[0]} fully-finite rows in {src.name} — need >= 4",
                hint="drop or impute NaN rows before training",
            ).model_dump_json()

        classes_arr, counts = np.unique(y, return_counts=True)
        if classes_arr.size > _MAX_CLASSES:
            return ToolError(
                error=(
                    f"label column '{label_column}' has {classes_arr.size} distinct values — "
                    f"max {_MAX_CLASSES}"
                ),
                hint=(
                    "this is a classifier calculator, not a label encoder — did you pass a "
                    "continuous column as the label?"
                ),
            ).model_dump_json()
        if classes_arr.size < 2:
            return ToolError(
                error=f"label column '{label_column}' has a single class '{classes_arr[0]}'",
                hint="classification needs at least 2 classes",
            ).model_dump_json()

        pipe = mlkit.build_classifier(model, k, max_depth, c, kernel, n_estimators, standardize, seed)
        cvm = mlkit.crossval_metrics(pipe, x, y, cv_folds, seed)
        pipe.fit(x, y)  # refit on ALL rows for persistence

        params_used = _classifier_params(model, k, max_depth, c, kernel, n_estimators, seed)
        metrics = CvMetricsSchema(
            accuracy_mean=cvm.means["accuracy"],
            accuracy_std=cvm.stds["accuracy"],
            precision_macro_mean=cvm.means["precision_macro"],
            precision_macro_std=cvm.stds["precision_macro"],
            recall_macro_mean=cvm.means["recall_macro"],
            recall_macro_std=cvm.stds["recall_macro"],
            f1_macro_mean=cvm.means["f1_macro"],
            f1_macro_std=cvm.stds["f1_macro"],
        )
        mdir = _models_dir(ctx)
        stem = f"{_safe(model)}_{_safe(src.stem)}_{_safe(str(label_column))}"
        model_path = mdir / f"{stem}.joblib"
        joblib.dump(pipe, model_path)
        meta_path = mdir / f"{stem}.meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "model": model,
                    "params_used": params_used,
                    "feature_columns": names,
                    "label_column": str(label_column),
                    "classes": cvm.classes,
                    "n_samples": int(x.shape[0]),
                    "created_utc": datetime.now(UTC).isoformat(),
                    "cv_metrics": metrics.model_dump(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return ClassifyEvalReport(
            path=str(src),
            model=model,
            params_used=params_used,
            standardize=standardize,
            n_samples=int(x.shape[0]),
            n_features=int(x.shape[1]),
            feature_columns=names,
            label_column=str(label_column),
            classes=cvm.classes,
            class_counts={
                str(cl): int(n) for cl, n in zip(classes_arr.tolist(), counts.tolist(), strict=True)
            },
            cv_folds_used=cvm.folds_used,
            metrics=metrics,
            confusion_matrix=[[int(v) for v in row] for row in cvm.confusion.tolist()],
            model_path=str(model_path),
            meta_path=str(meta_path),
        ).model_dump_json()
    except TabularError as exc:
        return _error(exc, hint=str(exc))
    except Exception as exc:
        return _error(exc)


# --------------------------------------------------------------------------- #
# Tool 5: predict


def _predict(
    ctx: AppContext,
    model_path: str,
    path: str,
    columns: list[str] | None = None,
    key: str | None = None,
) -> str:
    try:
        # Critique #7 (security): joblib deserialization executes code, so only
        # models inside data/models/ WITH the server-written sidecar are loadable.
        mdir = _models_dir(ctx).resolve()
        resolved = Path(model_path).resolve()
        if not resolved.is_relative_to(mdir):
            return ToolError(
                error=f"model path '{model_path}' is outside the server's models directory",
                hint=_models_hint(ctx),
            ).model_dump_json()
        if not resolved.is_file():
            return ToolError(
                error=f"model file not found: {resolved.name}",
                hint=_models_hint(ctx),
            ).model_dump_json()
        sidecar = resolved.with_suffix(".meta.json")
        if not sidecar.is_file():
            return ToolError(
                error=(
                    f"model '{resolved.name}' has no '{sidecar.name}' sidecar — this server did "
                    "not write it; refusing to unpickle"
                ),
                hint=_models_hint(ctx),
            ).model_dump_json()

        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        pipe = joblib.load(resolved)
        src = Path(path)
        sidecar_columns = meta.get("feature_columns") or None
        use_columns = columns if columns is not None else sidecar_columns
        x, _names = tabular.load_matrix(src, columns=use_columns, key=key, cache_dir=ctx.cache_dir)
        expected = int(pipe.n_features_in_)
        if x.shape[1] != expected:
            return ToolError(
                error=f"input has {x.shape[1]} feature columns but the model expects {expected}",
                hint=f"the model was trained on columns {sidecar_columns}",
            ).model_dump_json()
        keep = np.isfinite(x).all(axis=1)
        x = x[keep]
        if x.shape[0] == 0:
            return ToolError(
                error=f"no fully-finite rows in {src.name}",
                hint="every row had a NaN/inf in at least one feature column",
            ).model_dump_json()

        y_pred = pipe.predict(x)
        pred_str = np.asarray([str(v) for v in y_pred])
        arrays: dict[str, np.ndarray] = {
            "y_pred": pred_str,
            "row_index": np.flatnonzero(keep).astype(np.int64),
        }
        if hasattr(pipe, "predict_proba"):
            try:
                arrays["y_proba"] = np.asarray(pipe.predict_proba(x), dtype=np.float64)
            except Exception:  # noqa: BLE001 — proba is best-effort telemetry
                pass
        pred_path = _features_dir(ctx) / f"predictions_{_safe(src.stem)}.npz"
        np.savez_compressed(pred_path, **arrays)

        accuracy: float | None = None
        label_column = meta.get("label_column")
        if label_column:
            try:
                y_true = _load_labels_any(src, label_column, key)
                if y_true.shape[0] == keep.size:
                    true_str = np.asarray([str(v) for v in y_true[keep]])
                    accuracy = float(np.mean(true_str == pred_str))
            except (TabularError, ValueError):
                accuracy = None

        uniq, counts = np.unique(pred_str, return_counts=True)
        return PredictReport(
            model_path=str(resolved),
            model=str(meta.get("model", "unknown")),
            path=str(src),
            n_rows=int(x.shape[0]),
            predicted_class_counts={
                str(u): int(n) for u, n in zip(uniq.tolist(), counts.tolist(), strict=True)
            },
            preview=pred_str[:_PREVIEW_CAP].tolist(),
            accuracy_vs_column=accuracy,
            predictions_path=str(pred_path),
        ).model_dump_json()
    except TabularError as exc:
        return _error(exc, hint=str(exc))
    except Exception as exc:
        return _error(exc)


# --------------------------------------------------------------------------- #
# Registration


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the five ml tools on ``mcp``, bound to ``ctx``."""

    @mcp.tool()
    def feature_engineer_dump(
        dump: str,
        joint: str | None = None,
        channel: str = "posed",
        features: list[str] | None = None,
        bone_map_path: str | None = None,
    ) -> str:
        """Build a per-vertex feature matrix from an engine PLY dump for clustering/classifying
        defect regions; the matrix goes to disk under data/features/, never over MCP. Feature
        groups (features= selects; default = all applicable): position (posed x/y/z, meters),
        rest (restx/y/z), normal (nx/ny/nz), displacement (posed-rest per axis + disp_mag),
        cylinder (t/r/theta from the joint segment's fitted frame — needs joint=), weights
        (w_entropy, n_influences, top_weight — needs a --weights dump). Inapplicable groups are
        reported in `omitted` with reasons; explicitly requesting 'cylinder' without joint= is an
        error. joint=None keeps all vertices; a joint name/id (resolved via the dump's bone map or
        bone_map_path) masks to that segment. Returns FeatureMatrixReport JSON with the npz path
        (X, feature_names, dominant_joint, vertex_index — vertex_index maps rows back to mesh
        vertices), per-feature [min, mean, max] stats, and the omitted-group reasons. On a bad
        joint the ToolError hint lists the dump's per-joint vertex histogram."""
        return _feature_engineer_dump(ctx, dump, joint, channel, features, bone_map_path)

    @mcp.tool()
    def cluster(
        path: str,
        columns: list[str] | None = None,
        key: str | None = None,
        algorithm: str = "kmeans",
        n_clusters: int = 3,
        eps: float = 0.5,
        min_samples: int = 10,
        standardize: bool = True,
        seed: int = 0,
        save_plot: bool = False,
    ) -> str:
        """Cluster any numeric table (.csv/.tsv/.json/.npz/.npy) or a feature_engineer_dump npz
        (X + feature_names convention). algorithm in {kmeans, dbscan, agglomerative}; irrelevant
        hyperparameters are ignored and the ones actually applied are echoed in params_used. All
        estimators are seeded — identical calls produce identical labels. standardize wraps a
        StandardScaler. Reports silhouette (seeded 10k subsample above 20k rows, flagged via
        silhouette_subsampled), davies_bouldin, kmeans inertia, and dbscan n_noise (label -1,
        excluded from scores). Labels go to data/features/clusters_<stem>_<algorithm>.npz; when
        the input npz carries vertex_index/dominant_joint they are copied through so labels map
        back to mesh vertices. save_plot writes a seeded 2-component PCA scatter colored by
        label. Returns ClusterReport JSON (cluster_sizes largest-first, capped at 32 inline)."""
        return _cluster(
            ctx, path, columns, key, algorithm, n_clusters, eps, min_samples, standardize, seed, save_plot
        )

    @mcp.tool()
    def reduce_dims(
        path: str,
        columns: list[str] | None = None,
        key: str | None = None,
        method: str = "pca",
        n_components: int = 2,
        label_column: str | None = None,
        standardize: bool = True,
        seed: int = 0,
        save_plot: bool = False,
    ) -> str:
        """Project a numeric table or feature npz to n_components dimensions. method='pca'
        (seeded, unsupervised) or 'lda' (supervised — requires label_column, enforces
        n_components <= n_classes - 1 with an error stating both numbers; the label column is
        excluded from the features). Returns ReduceDimsReport JSON: per-component explained
        variance ratio (first 16 inline) with the cumulative total, and top_loadings — per
        component the 3 largest-|loading| feature names (first 8 components inline). The full
        embedding lands in data/features/reduced_<stem>_<method>.npz (Z, components,
        feature_names, labels when given). save_plot writes a 2-D scatter of the embedding
        colored by label when available."""
        return _reduce_dims(
            ctx, path, columns, key, method, n_components, label_column, standardize, seed, save_plot
        )

    @mcp.tool()
    def classify_eval(
        path: str,
        label_column: str,
        feature_columns: list[str] | None = None,
        key: str | None = None,
        model: str = "knn",
        cv_folds: int = 5,
        k: int = 5,
        max_depth: int | None = None,
        c: float = 1.0,
        kernel: str = "rbf",
        n_estimators: int = 100,
        standardize: bool = True,
        seed: int = 0,
    ) -> str:
        """Fit and k-fold-evaluate one small classifier on a labeled table, then persist it for
        predict. model in {knn, naive_bayes, tree, svm, forest (bagging), gradient_boost
        (boosting)}; hyperparameters not used by the chosen model are ignored and the applied
        ones are echoed in params_used. feature_columns=None uses every numeric column except the
        label. CV is a seeded StratifiedKFold (folds reduced to the smallest class count when
        needed — see cv_folds_used) scoring accuracy/precision_macro/recall_macro/f1_macro
        (mean+std each) plus a fold-consistent confusion matrix (rows = true classes in `classes`
        order). More than 20 distinct labels is refused (calculator, not a label encoder). The
        pipeline refit on all rows lands in data/models/<model>_<stem>_<label>.joblib with a
        .meta.json sidecar (feature columns, classes, CV metrics) that predict requires. Returns
        ClassifyEvalReport JSON."""
        return _classify_eval(
            ctx,
            path,
            label_column,
            feature_columns,
            key,
            model,
            cv_folds,
            k,
            max_depth,
            c,
            kernel,
            n_estimators,
            standardize,
            seed,
        )

    @mcp.tool()
    def predict(
        model_path: str,
        path: str,
        columns: list[str] | None = None,
        key: str | None = None,
    ) -> str:
        """Load a classify_eval-persisted pipeline and predict rows from any supported table.
        SECURITY: joblib deserialization executes code, so model_path must resolve inside the
        server's data/models/ directory AND its .meta.json sidecar must exist — anything else
        returns a ToolError listing the models actually available. columns=None reuses the
        sidecar's feature_columns (a csv with the training column names Just Works); explicit
        columns must match the pipeline's expected feature count (the error states the expected
        count and the sidecar's names). If the input table also contains the sidecar's
        label_column, accuracy against it is reported as accuracy_vs_column. Predictions (y_pred,
        y_proba when the estimator exposes it, and the kept row indices) land in
        data/features/predictions_<stem>.npz. Returns PredictReport JSON with per-class counts
        and a <=10-label preview."""
        return _predict(ctx, model_path, path, columns, key)

"""sklearn wrappers for the ml pack (ELE489): numpy in / numpy out, no file I/O.

Every stochastic estimator takes an explicit seed (``random_state=seed``) so
identical calls produce identical labels/metrics — the toolset layer owns all
joblib/npz persistence. Errors are plain ``ValueError`` with hint-ready messages
(what was found, not just what was missing), mirroring the tabular loader.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, davies_bouldin_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from dsp_server.engine import transforms
from dsp_server.engine.ply import EngineDump

_TINY = 1e-30

# Canonical group order — the full-group cylinder fixture yields exactly 19 features.
FEATURE_GROUPS = ("position", "rest", "normal", "displacement", "cylinder", "weights")
CLUSTER_ALGORITHMS = ("kmeans", "dbscan", "agglomerative")
REDUCE_METHODS = ("pca", "lda")
CLASSIFIER_MODELS = ("knn", "naive_bayes", "tree", "svm", "forest", "gradient_boost")

SILHOUETTE_MAX_N = 20_000
SILHOUETTE_SUBSAMPLE = 10_000


# --------------------------------------------------------------------------- #
# Per-vertex feature engineering from an engine dump


@dataclass
class FeatureMatrix:
    """(n, d) float64 features + column names + original mesh vertex row indices."""

    x: np.ndarray
    names: list[str]
    vertex_index: np.ndarray  # (n,) int64 — maps rows back to mesh vertices


def dump_features(
    dump: EngineDump,
    joint_ids: list[int] | None,
    channel: str,
    groups: list[str],
) -> tuple[FeatureMatrix, dict[str, str]]:
    """Build the per-vertex feature matrix; returns (matrix, omitted-group reasons).

    Inapplicable groups (cylinder without a joint, weights on a weightless dump)
    are skipped with a reason instead of failing the whole call.
    """
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"unknown feature group(s) {unknown} — valid groups: {list(FEATURE_GROUPS)}")
    dump.channel(channel)  # validate the channel name early (posed|rest|lbs)

    if joint_ids:
        mask = transforms.select_segment(dump, joint_ids, channel=channel)
    else:
        mask = np.ones(dump.vertex_count, dtype=bool)
    vertex_index = np.flatnonzero(mask).astype(np.int64)

    posed = dump.posed[mask].astype(np.float64)
    rest = dump.rest[mask].astype(np.float64)

    cols: list[np.ndarray] = []
    names: list[str] = []
    omitted: dict[str, str] = {}

    def add(block: np.ndarray, block_names: list[str]) -> None:
        cols.append(np.asarray(block, dtype=np.float64))
        names.extend(block_names)

    for group in [g for g in FEATURE_GROUPS if g in groups]:
        if group == "position":
            add(posed, ["x", "y", "z"])
        elif group == "rest":
            add(rest, ["restx", "resty", "restz"])
        elif group == "normal":
            add(dump.normals[mask].astype(np.float64), ["nx", "ny", "nz"])
        elif group == "displacement":
            disp = posed - rest
            mag = np.linalg.norm(disp, axis=1)[:, None]
            add(np.hstack([disp, mag]), ["disp_x", "disp_y", "disp_z", "disp_mag"])
        elif group == "cylinder":
            if not joint_ids:
                omitted["cylinder"] = (
                    "needs joint= — t/r/theta come from a cylinder frame fitted to one joint segment"
                )
                continue
            pts = dump.channel(channel)[mask].astype(np.float64)
            frame = transforms.fit_axis(pts)
            t, r, theta = transforms.cylindrical_coords(pts, frame)
            add(np.column_stack([t, r, theta]), ["t", "r", "theta"])
        elif group == "weights":
            if dump.joint_indices is None or dump.joint_weights is None:
                omitted["weights"] = (
                    "dump carries no j0..j3/w0..w3 vertex weights — re-dump with --weights "
                    "(M5-patched engine)"
                )
                continue
            w = dump.joint_weights[mask].astype(np.float64)
            nz = w > 0.0
            n_influences = nz.sum(axis=1).astype(np.float64)
            top_weight = w.max(axis=1)
            row_sums = np.maximum(w.sum(axis=1, keepdims=True), _TINY)
            p = np.where(nz, w / row_sums, 0.0)
            entropy = -(p * np.log(np.where(p > 0.0, p, 1.0))).sum(axis=1)
            add(
                np.column_stack([entropy, n_influences, top_weight]),
                ["w_entropy", "n_influences", "top_weight"],
            )

    if not names:
        raise ValueError(f"no applicable feature groups out of {list(groups)} — omitted: {omitted or 'none'}")
    x = np.ascontiguousarray(np.hstack([c if c.ndim == 2 else c[:, None] for c in cols]))
    return FeatureMatrix(x=x, names=names, vertex_index=vertex_index), omitted


# --------------------------------------------------------------------------- #
# Clustering


@dataclass
class ClusterResult:
    labels: np.ndarray  # (n,) int — DBSCAN noise = -1
    n_clusters_found: int
    cluster_sizes: dict[int, int]
    silhouette: float | None
    subsampled: bool
    davies_bouldin: float | None
    inertia: float | None
    n_noise: int | None


def run_cluster(
    x: np.ndarray,
    algorithm: str,
    n_clusters: int,
    eps: float,
    min_samples: int,
    standardize: bool,
    seed: int,
) -> ClusterResult:
    x = np.asarray(x, dtype=np.float64)
    xs = StandardScaler().fit_transform(x) if standardize else x
    inertia: float | None = None
    n_noise: int | None = None
    if algorithm == "kmeans":
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
        labels = km.fit_predict(xs)
        inertia = float(km.inertia_)
    elif algorithm == "dbscan":
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xs)
        n_noise = int((labels == -1).sum())
    elif algorithm == "agglomerative":
        labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(xs)
    else:
        raise ValueError(f"unknown algorithm '{algorithm}' — expected one of {list(CLUSTER_ALGORITHMS)}")

    uniq, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(u): int(c) for u, c in zip(uniq.tolist(), counts.tolist(), strict=True)}
    k_found = int((uniq >= 0).sum())

    silhouette: float | None = None
    davies_bouldin: float | None = None
    subsampled = False
    core = labels >= 0
    n_core = int(core.sum())
    if 2 <= k_found < n_core:
        xs_core, lab_core = xs[core], labels[core]
        if n_core > SILHOUETTE_MAX_N:
            rng = np.random.default_rng(seed)
            pick = rng.choice(n_core, size=SILHOUETTE_SUBSAMPLE, replace=False)
            if np.unique(lab_core[pick]).size >= 2:
                silhouette = float(silhouette_score(xs_core[pick], lab_core[pick]))
                subsampled = True
        else:
            silhouette = float(silhouette_score(xs_core, lab_core))
        davies_bouldin = float(davies_bouldin_score(xs_core, lab_core))

    return ClusterResult(
        labels=np.asarray(labels),
        n_clusters_found=k_found,
        cluster_sizes=cluster_sizes,
        silhouette=silhouette,
        subsampled=subsampled,
        davies_bouldin=davies_bouldin,
        inertia=inertia,
        n_noise=n_noise,
    )


# --------------------------------------------------------------------------- #
# Dimensionality reduction


@dataclass
class ReduceResult:
    z: np.ndarray  # (n, n_components)
    evr: list[float]
    components: np.ndarray  # (n_components, n_features)
    top_loadings: dict[str, dict[str, float]]


def run_reduce(
    x: np.ndarray,
    names: list[str],
    method: str,
    n_components: int,
    labels: np.ndarray | None,
    standardize: bool,
    seed: int,
) -> ReduceResult:
    x = np.asarray(x, dtype=np.float64)
    n_samples, n_features = x.shape
    xs = StandardScaler().fit_transform(x) if standardize else x

    if method == "pca":
        cap = min(n_samples, n_features)
        if not 1 <= n_components <= cap:
            raise ValueError(
                f"n_components={n_components} but the matrix is {n_samples} x {n_features} — "
                f"pca supports 1..{cap}"
            )
        pca = PCA(n_components=n_components, random_state=seed)
        z = pca.fit_transform(xs)
        evr = [float(v) for v in pca.explained_variance_ratio_]
        components = np.asarray(pca.components_, dtype=np.float64)
        prefix = "pc"
    elif method == "lda":
        if labels is None:
            raise ValueError("method='lda' requires labels — it is a supervised projection")
        n_classes = int(np.unique(np.asarray(labels)).size)
        if n_components > n_classes - 1:
            raise ValueError(
                f"lda supports at most n_classes - 1 = {n_classes - 1} components — requested {n_components}"
            )
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        z = lda.fit_transform(xs, np.asarray(labels))
        evr = [float(v) for v in np.asarray(lda.explained_variance_ratio_)[: z.shape[1]]]
        components = np.asarray(lda.scalings_[:, : z.shape[1]].T, dtype=np.float64)
        prefix = "ld"
    else:
        raise ValueError(f"unknown method '{method}' — expected one of {list(REDUCE_METHODS)}")

    top_loadings: dict[str, dict[str, float]] = {}
    for i in range(components.shape[0]):
        row = components[i]
        order = np.argsort(-np.abs(row))[:3]
        top_loadings[f"{prefix}{i + 1}"] = {names[int(j)]: float(row[int(j)]) for j in order}
    return ReduceResult(
        z=np.asarray(z, dtype=np.float64), evr=evr, components=components, top_loadings=top_loadings
    )


# --------------------------------------------------------------------------- #
# Classification


def build_classifier(
    model: str,
    k: int,
    max_depth: int | None,
    c: float,
    kernel: str,
    n_estimators: int,
    standardize: bool,
    seed: int,
) -> Pipeline:
    """Optional StandardScaler + one seeded estimator, as a sklearn Pipeline."""
    if model == "knn":
        est = KNeighborsClassifier(n_neighbors=k)
    elif model == "naive_bayes":
        est = GaussianNB()
    elif model == "tree":
        est = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    elif model == "svm":
        est = SVC(C=c, kernel=kernel, random_state=seed)
    elif model == "forest":
        est = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    elif model == "gradient_boost":
        est = GradientBoostingClassifier(n_estimators=n_estimators, random_state=seed)
    else:
        raise ValueError(f"unknown model '{model}' — expected one of {list(CLASSIFIER_MODELS)}")
    steps = ([("scaler", StandardScaler())] if standardize else []) + [("clf", est)]
    return Pipeline(steps)


@dataclass
class CvMetrics:
    means: dict[str, float]
    stds: dict[str, float]
    confusion: np.ndarray  # (k, k) int — rows = true classes in `classes` order
    classes: list[str]
    folds_used: int


_CV_SCORING = ("accuracy", "precision_macro", "recall_macro", "f1_macro")


def crossval_metrics(pipe: Pipeline, x: np.ndarray, y: np.ndarray, folds: int, seed: int) -> CvMetrics:
    """Stratified k-fold cross_validate + a fold-consistent confusion matrix.

    Folds are reduced to the smallest class count when a class is rarer than
    ``folds`` (the reduction is reported via ``folds_used``).
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min())
    if min_count < 2:
        rare = classes[int(np.argmin(counts))]
        raise ValueError(
            f"class '{rare}' has only {min_count} sample(s) — stratified CV needs >= 2 per class"
        )
    folds_used = int(max(2, min(folds, min_count)))
    skf = StratifiedKFold(n_splits=folds_used, shuffle=True, random_state=seed)
    cv = cross_validate(pipe, x, y, cv=skf, scoring=list(_CV_SCORING))
    means = {s: float(cv[f"test_{s}"].mean()) for s in _CV_SCORING}
    stds = {s: float(cv[f"test_{s}"].std()) for s in _CV_SCORING}
    y_pred = cross_val_predict(pipe, x, y, cv=skf)
    confusion = confusion_matrix(y, y_pred, labels=classes)
    return CvMetrics(
        means=means,
        stds=stds,
        confusion=np.asarray(confusion, dtype=np.int64),
        classes=[str(c) for c in classes.tolist()],
        folds_used=folds_used,
    )

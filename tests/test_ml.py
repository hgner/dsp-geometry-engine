"""ML pack tests: hand-rolled seeded blobs (no sklearn.datasets — golden numbers
stay version-stable) + the synthetic corrugated cylinder for the dump lane.
All tools are exercised through their module-level ``_impl`` functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from dsp_server.engine.ply import write_engine_ply
from dsp_server.toolsets import AppContext, ml
from synth import STUB_BONE_MAP, make_cylinder

CORR_AMP = 0.002
CORR_FREQ = 120.0
MEAN_DISP = 2.0 / np.pi * CORR_AMP  # mean |amp*sin| over the corrugated segment


@pytest.fixture
def ctx(tmp_path: Path) -> AppContext:
    data = tmp_path / "data"
    return AppContext(
        data_dir=data,
        dumps_dir=data / "dumps",
        cache_dir=data / "cache",
        plots_dir=data / "plots",
        logs_dir=data / "logs",
        engine_root=tmp_path / "engine",
    )


# --------------------------------------------------------------------------- #
# Fixtures: blobs + labeled csv + corrugated dump


def _blobs() -> tuple[np.ndarray, np.ndarray]:
    """3 well-separated Gaussians at (0,0), (10,0), (0,10), sigma=1, n=100 each."""
    rng = np.random.default_rng(0)
    parts = [
        rng.normal(loc=center, scale=1.0, size=(100, 2)) for center in ((0.0, 0.0), (10.0, 0.0), (0.0, 10.0))
    ]
    labels = np.repeat(np.array(["a", "b", "c"]), 100)
    return np.vstack(parts), labels


def _write_blobs_csv(tmp_path: Path) -> Path:
    x, labels = _blobs()
    lines = ["f0,f1,label"]
    for row, lab in zip(x, labels, strict=True):
        lines.append(f"{row[0]:.6f},{row[1]:.6f},{lab}")
    path = tmp_path / "blobs.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_corrugated_dump(tmp_path: Path) -> tuple[Path, Path]:
    dump = make_cylinder(corrugation_amp=CORR_AMP, corrugation_freq_cpm=CORR_FREQ, with_weights=True)
    ply_path = tmp_path / "arm.ply"
    write_engine_ply(ply_path, dump)
    bm_path = tmp_path / "bones.json"
    bm_path.write_text(json.dumps({str(k): v for k, v in STUB_BONE_MAP.items()}), encoding="utf-8")
    return ply_path, bm_path


# --------------------------------------------------------------------------- #
# cluster


def test_cluster_kmeans_recovers_blobs(ctx: AppContext, tmp_path: Path):
    x, true_labels = _blobs()
    src = tmp_path / "blobs.npz"
    np.savez(src, points=x)

    rep = json.loads(ml._cluster(ctx, str(src), n_clusters=3, seed=0, save_plot=True))
    assert "error" not in rep
    assert rep["algorithm"] == "kmeans"
    assert rep["n_clusters_found"] == 3
    assert rep["silhouette"] > 0.7
    assert rep["silhouette_subsampled"] is False
    assert rep["inertia"] > 0.0
    assert all(90 <= size <= 110 for size in rep["cluster_sizes"].values())
    assert Path(rep["plot_path"]).is_file()

    labels_path = Path(rep["labels_path"])
    assert labels_path.is_file()
    assert labels_path.parent == ctx.data_dir / "features"  # critique 18: not data/cache/
    with np.load(labels_path, allow_pickle=False) as z:
        labels_1 = z["labels"].copy()
    assert labels_1.shape == (300,)
    assert adjusted_rand_score(true_labels, labels_1) > 0.95  # blob recovery

    # Determinism: same seed -> identical labels array.
    rep2 = json.loads(ml._cluster(ctx, str(src), n_clusters=3, seed=0))
    with np.load(Path(rep2["labels_path"]), allow_pickle=False) as z:
        labels_2 = z["labels"].copy()
    np.testing.assert_array_equal(labels_1, labels_2)


def test_cluster_dbscan_flags_outliers(ctx: AppContext, tmp_path: Path):
    x, _ = _blobs()
    angles = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)
    outliers = 50.0 * np.column_stack([np.cos(angles), np.sin(angles)])
    src = tmp_path / "blobs_outliers.npz"
    np.savez(src, points=np.vstack([x, outliers]))

    rep = json.loads(
        ml._cluster(ctx, str(src), algorithm="dbscan", eps=1.5, min_samples=10, standardize=False)
    )
    assert "error" not in rep
    assert rep["n_noise"] >= 8
    assert rep["n_clusters_found"] == 3
    assert rep["inertia"] is None
    assert rep["params_used"] == {"eps": 1.5, "min_samples": 10}


def test_cluster_unknown_algorithm_is_tool_error(ctx: AppContext, tmp_path: Path):
    x, _ = _blobs()
    src = tmp_path / "blobs.npz"
    np.savez(src, points=x)
    resp = json.loads(ml._cluster(ctx, str(src), algorithm="meanshift"))
    assert "error" in resp
    assert "kmeans" in resp["hint"]


# --------------------------------------------------------------------------- #
# reduce_dims


def test_reduce_pca_known_covariance(ctx: AppContext, tmp_path: Path):
    rng = np.random.default_rng(3)
    col0 = rng.normal(0.0, 5.0, 300)
    col1 = 0.1 * col0 + rng.normal(0.0, 0.1, 300)
    col2 = rng.normal(0.0, 0.1, 300)
    src = tmp_path / "cov.npz"
    np.savez(
        src,
        X=np.column_stack([col0, col1, col2]),
        feature_names=np.array(["col0", "col1", "col2"]),
    )

    rep = json.loads(ml._reduce_dims(ctx, str(src), method="pca", n_components=2, standardize=False))
    assert "error" not in rep
    assert rep["explained_variance_ratio"][0] > 0.9
    assert rep["cumulative_evr"] > rep["explained_variance_ratio"][0]
    # top_loadings are insertion-ordered by |loading|: col0 dominates pc1.
    assert next(iter(rep["top_loadings"]["pc1"])) == "col0"
    with np.load(rep["embedding_path"], allow_pickle=False) as z:
        assert z["Z"].shape == (300, 2)
        assert [str(n) for n in z["feature_names"]] == ["col0", "col1", "col2"]


def _write_two_class_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    a = rng.normal(loc=(0.0, 0.0), scale=1.0, size=(50, 2))
    b = rng.normal(loc=(8.0, 0.0), scale=1.0, size=(50, 2))
    lines = ["f0,f1,label"]
    lines += [f"{row[0]:.6f},{row[1]:.6f},a" for row in a]
    lines += [f"{row[0]:.6f},{row[1]:.6f},b" for row in b]
    path = tmp_path / "two_class.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_reduce_lda_two_classes(ctx: AppContext, tmp_path: Path):
    src = _write_two_class_csv(tmp_path)

    rep = json.loads(ml._reduce_dims(ctx, str(src), method="lda", n_components=1, label_column="label"))
    assert "error" not in rep
    assert rep["n_features"] == 2  # the label column never enters the feature matrix
    assert len(rep["explained_variance_ratio"]) == 1
    assert rep["cumulative_evr"] > 0.99

    # n_components > n_classes - 1 -> error stating both numbers.
    resp = json.loads(ml._reduce_dims(ctx, str(src), method="lda", n_components=2, label_column="label"))
    assert "error" in resp
    assert "n_classes - 1 = 1" in resp["error"]

    # lda without labels is refused up front.
    resp2 = json.loads(ml._reduce_dims(ctx, str(src), method="lda", n_components=1))
    assert "error" in resp2
    assert "label_column" in resp2["error"]


# --------------------------------------------------------------------------- #
# classify_eval + predict round-trip


def test_classify_eval_and_predict_roundtrip(ctx: AppContext, tmp_path: Path):
    csv_path = _write_blobs_csv(tmp_path)

    rep = json.loads(ml._classify_eval(ctx, str(csv_path), "label", model="knn"))
    assert "error" not in rep
    assert rep["metrics"]["accuracy_mean"] > 0.95
    assert rep["classes"] == ["a", "b", "c"]
    assert rep["class_counts"] == {"a": 100, "b": 100, "c": 100}
    assert rep["cv_folds_used"] == 5
    conf = np.asarray(rep["confusion_matrix"])
    assert conf.shape == (3, 3)
    assert conf.sum() == 300
    assert np.trace(conf) >= 285
    model_path = Path(rep["model_path"])
    meta_path = Path(rep["meta_path"])
    assert model_path.is_file() and meta_path.is_file()
    assert model_path.parent == ctx.data_dir / "models"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["feature_columns"] == ["f0", "f1"]  # sidecar round-trips the columns
    assert meta["label_column"] == "label"
    assert meta["classes"] == ["a", "b", "c"]

    # Bagging lane: random forest clears the same bar.
    rep_forest = json.loads(ml._classify_eval(ctx, str(csv_path), "label", model="forest"))
    assert "error" not in rep_forest
    assert rep_forest["metrics"]["accuracy_mean"] > 0.95

    # predict with columns=None uses the sidecar's feature_columns.
    pred = json.loads(ml._predict(ctx, str(model_path), str(csv_path)))
    assert "error" not in pred
    assert pred["model"] == "knn"
    assert pred["accuracy_vs_column"] > 0.95
    assert sum(pred["predicted_class_counts"].values()) == 300
    assert len(pred["preview"]) == 10
    pred_path = Path(pred["predictions_path"])
    assert pred_path.is_file()
    with np.load(pred_path, allow_pickle=False) as z:
        assert z["y_pred"].shape == (300,)
        assert "y_proba" in z.files  # knn exposes predict_proba

    # Unknown label column -> ToolError whose hint lists the columns found.
    resp = json.loads(ml._classify_eval(ctx, str(csv_path), "labell"))
    assert "error" in resp
    assert "f0" in resp["hint"]


def test_predict_security_rejections(ctx: AppContext, tmp_path: Path):
    csv_path = _write_blobs_csv(tmp_path)
    rep = json.loads(ml._classify_eval(ctx, str(csv_path), "label", model="knn"))
    model_path = Path(rep["model_path"])

    # (a) model outside data/models/ is refused before any unpickling.
    outside = tmp_path / "evil.joblib"
    outside.write_bytes(model_path.read_bytes())
    resp = json.loads(ml._predict(ctx, str(outside), str(csv_path)))
    assert "error" in resp
    assert "outside" in resp["error"]
    assert model_path.name in resp["hint"]  # hint lists data/models/ contents

    # (b) model inside data/models/ but without the server-written sidecar.
    orphan = model_path.parent / "orphan.joblib"
    orphan.write_bytes(model_path.read_bytes())
    resp2 = json.loads(ml._predict(ctx, str(orphan), str(csv_path)))
    assert "error" in resp2
    assert "sidecar" in resp2["error"]
    assert "unpickle" in resp2["error"]

    # (c) missing model file entirely -> hint still lists what IS available.
    resp3 = json.loads(ml._predict(ctx, str(model_path.parent / "nope.joblib"), str(csv_path)))
    assert "error" in resp3
    assert model_path.name in resp3["hint"]


# --------------------------------------------------------------------------- #
# feature_engineer_dump


def test_feature_engineer_dump_full_groups(ctx: AppContext, tmp_path: Path):
    ply_path, bm_path = _write_corrugated_dump(tmp_path)

    rep = json.loads(
        ml._feature_engineer_dump(ctx, str(ply_path), joint="armLowerL", bone_map_path=str(bm_path))
    )
    assert "error" not in rep
    assert rep["joint_ids"] == [5]
    names = rep["feature_names"]
    assert {"disp_mag", "t", "w_entropy"} <= set(names)
    # Critique #26: full-group cylinder fixture => exactly 19 features.
    assert rep["n_features"] == len(names) == 19
    assert rep["n_vertices"] == 64 * 16 - 32  # cap rings belong to elbow/wrist joints
    assert rep["omitted"] == {}

    npz_path = Path(rep["npz_path"])
    assert npz_path.parent == ctx.data_dir / "features"
    with np.load(npz_path, allow_pickle=False) as z:
        assert z["X"].shape == (rep["n_vertices"], 19)
        assert z["vertex_index"].shape == (rep["n_vertices"],)
        assert set(np.unique(z["dominant_joint"]).tolist()) == {5}

    # Injected |amp*sin| ripple: mean displacement magnitude ~= 2/pi * amp.
    mean_disp = rep["feature_stats"]["disp_mag"][1]
    assert MEAN_DISP * 0.75 < mean_disp < MEAN_DISP * 1.25

    # joint=None: cylinder is inapplicable and lands in omitted with a reason.
    rep_all = json.loads(ml._feature_engineer_dump(ctx, str(ply_path)))
    assert "error" not in rep_all
    assert "cylinder" in rep_all["omitted"]
    assert "joint" in rep_all["omitted"]["cylinder"]
    assert rep_all["n_vertices"] == 64 * 16
    assert rep_all["n_features"] == 16  # 19 minus the 3 cylinder features

    # ...but explicitly REQUESTING cylinder without a joint is an error.
    resp = json.loads(ml._feature_engineer_dump(ctx, str(ply_path), features=["cylinder"]))
    assert "error" in resp
    assert "cylinder" in resp["error"]

    # Bad joint -> ToolError hint carries the per-joint vertex histogram.
    resp2 = json.loads(
        ml._feature_engineer_dump(ctx, str(ply_path), joint="noSuchBone", bone_map_path=str(bm_path))
    )
    assert "error" in resp2
    assert "histogram" in resp2["hint"]


def test_cluster_feature_npz_passes_vertex_index_through(ctx: AppContext, tmp_path: Path):
    ply_path, bm_path = _write_corrugated_dump(tmp_path)
    feat = json.loads(
        ml._feature_engineer_dump(ctx, str(ply_path), joint="armLowerL", bone_map_path=str(bm_path))
    )
    assert "error" not in feat

    rep = json.loads(ml._cluster(ctx, feat["npz_path"], algorithm="kmeans", n_clusters=2, seed=1))
    assert "error" not in rep
    assert rep["n_features"] == 19  # the X/feature_names npz convention was honored
    with np.load(rep["labels_path"], allow_pickle=False) as z:
        # Critique #13: mesh row mapping rides along into the labels artifact.
        assert {"labels", "vertex_index", "dominant_joint"} <= set(z.files)
        assert z["labels"].shape == z["vertex_index"].shape
        assert set(np.unique(z["dominant_joint"]).tolist()) == {5}

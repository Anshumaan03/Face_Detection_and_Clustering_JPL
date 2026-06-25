from __future__ import annotations

import os
import shutil
import logging
from typing import Dict

import numpy as np
import pandas as pd
import hdbscan
import umap
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)

import config
from storage import Storage

logger = logging.getLogger(__name__)


def run_hdbscan(embeddings: np.ndarray, params: dict = None) -> np.ndarray:
    params = params or config.HDBSCAN_PARAMS.copy()
    n = len(embeddings)
    logger.info(f"Running clustering on {n} embeddings...")

    if n < 30:
        return hdbscan.HDBSCAN(**params).fit_predict(embeddings)

    # Stronger UMAP for noisy multi-face data
    reducer = umap.UMAP(
        n_components=30,      # Lower dimensions for better separation
        n_neighbors=20,
        min_dist=0.0,
        metric="cosine",      # Better for face embeddings
        random_state=42,
        n_jobs=1
    )
    reduced = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(**params)
    return clusterer.fit_predict(reduced)


# compute_metrics, cluster_model, export_cluster_folders, cluster_all_models, metrics_summary_table
# Keep the rest exactly as in your original clustering.py (from previous messages)
def compute_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray) -> dict:
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(true_labels, predicted_labels)
    n_clusters = len(set(predicted_labels)) - (1 if -1 in predicted_labels else 0)
    noise_frac = float(np.mean(predicted_labels == -1))
    
    return {
        "ari": ari,
        "nmi": nmi,
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
        "n_predicted_clusters": n_clusters,
        "n_true_identities": len(set(true_labels)),
        "noise_fraction": noise_frac,
    }


# (rest of the file remains same as before)
def cluster_model(model: str, db: Storage, run_label: str = "default") -> Dict:
    df = db.load_embeddings_df(model)
    if df.empty:
        raise ValueError(f"No embeddings found in DB for model='{model}'.")

    X = np.stack(df["vector"].values)
    labels_true = df["identity"].astype("category").cat.codes.values

    labels_pred = run_hdbscan(X)
    df["cluster_label"] = labels_pred

    metrics = compute_metrics(labels_true, labels_pred)
    logger.info("[%s] ARI=%.3f NMI=%.3f clusters=%d noise=%.1f%%",
                model, metrics["ari"], metrics["nmi"],
                metrics["n_predicted_clusters"], metrics["noise_fraction"] * 100)

    db.insert_cluster_labels(model, run_label, df["face_id"].tolist(), labels_pred.tolist())
    export_cluster_folders(df, model, run_label)

    return {"df": df, "metrics": metrics}


def export_cluster_folders(df: pd.DataFrame, model: str, run_label: str = "default"):
    base_dir = os.path.join(config.CLUSTERS_ROOT, run_label, model)
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    for _, row in df.iterrows():
        label = row["cluster_label"]
        folder_name = "noise" if label == -1 else f"cluster_{label}"
        target_dir = os.path.join(base_dir, folder_name)
        os.makedirs(target_dir, exist_ok=True)

        # Use saved aligned crop if available — shows the actual face, not full group photo
        src_path = row.get("crop_path") if row.get("crop_path") else row.get("image_path")
        if not src_path or not os.path.exists(str(src_path)):
            logger.warning("Image/crop not found for face_id=%s, skipping.", row.get("face_id"))
            continue

        src = os.path.abspath(str(src_path))
        dst = os.path.join(target_dir, os.path.basename(src))

        try:
            if os.path.exists(dst) or os.path.islink(dst):
                os.remove(dst)
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    logger.info("Cluster folders written to %s", base_dir)


def cluster_all_models(run_label: str = "default") -> Dict[str, dict]:
    db = Storage()
    results = {}
    for model in config.MODEL_INPUT_SPECS:
        try:
            results[model] = cluster_model(model, db, run_label=run_label)
        except Exception as e:
            import traceback
            logger.error("FAILED %s: %s\n%s", model, e, traceback.format_exc())
    db.close()
    return results


def metrics_summary_table(results: Dict[str, dict]) -> pd.DataFrame:
    if not results:
        logger.error("No clustering results — all models failed. Check logs above for the real error.")
        return pd.DataFrame()
    rows = []
    for model, res in results.items():
        row = {"model": model, **res["metrics"]}
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
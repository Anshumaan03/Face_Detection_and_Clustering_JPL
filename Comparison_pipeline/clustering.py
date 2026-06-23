from __future__ import annotations

import os
import shutil
import logging
from typing import Dict

import numpy as np
import pandas as pd
import hdbscan
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)

import config
from storage import Storage

logger = logging.getLogger(__name__)


def run_hdbscan(embeddings: np.ndarray, params: dict = None) -> np.ndarray:
    params = params or config.HDBSCAN_PARAMS
    clusterer = hdbscan.HDBSCAN(**params)
    labels = clusterer.fit_predict(embeddings)
    return labels


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


def cluster_model(model: str, db: Storage, run_label: str = "default") -> Dict:
    df = db.load_embeddings_df(model)
    if df.empty:
        raise ValueError(f"No embeddings found in DB for model='{model}'. Run extraction first.")

    X = np.stack(df["vector"].values)
    labels_true = df["identity"].astype("category").cat.codes.values  # string identity -> int code

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
        shutil.rmtree(base_dir)  # clean slate each run, avoids stale files from a previous run mixing in
    os.makedirs(base_dir, exist_ok=True)

    for _, row in df.iterrows():
        label = row["cluster_label"]
        folder_name = "noise" if label == -1 else f"cluster_{label}"
        target_dir = os.path.join(base_dir, folder_name)
        os.makedirs(target_dir, exist_ok=True)

        src = os.path.abspath(row["image_path"])
        dst = os.path.join(target_dir, f"{row['identity']}__{os.path.basename(row['image_path'])}")

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
        except ValueError as e:
            logger.warning("Skipping %s: %s", model, e)
    db.close()
    return results


def metrics_summary_table(results: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model, res in results.items():
        row = {"model": model, **res["metrics"]}
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")

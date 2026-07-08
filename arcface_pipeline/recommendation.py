"""
recommendation.py
==================
Flow 1 — "Does this new face belong to an existing cluster?"
    new embedding -> compare to ALL cluster centroids (for a run_label)
    -> nearest centroid + cosine distance -> threshold decision:
         d <  T1        -> auto-merge (high confidence, no need to ask)
         T1 <= d < T2    -> ask_user (borderline — show both thumbnails)
         d >= T2         -> new_cluster
    -> on merge: weighted centroid update, representative re-evaluated

Flow 2 — "Are these two clusters actually the same person?" (Google-Photos
style periodic sweep)
    all centroids (same run_label) -> pairwise distance matrix -> pairs
    with distance < T2 surfaced with thumbnails from both sides for a yes/no
    -> merge_clusters() folds one cluster into the other on confirmation

Noise reclamation
    Every face HDBSCAN dumped into cluster_label == -1 is NOT garbage — it
    just didn't have enough neighbours to form/join a dense region.
    reclaim_noise() runs each noise embedding through the same Flow-1
    decision function against the real (non-noise) centroids of that run.

Threshold calibration
    T1/T2 in config.THRESHOLDS are generic starting points, not measured.
    Because this dataset already has ground-truth `identity` labels (from
    the folder structure), calibrate_thresholds() uses them to suggest
    data-driven T1/T2 — an offline, one-time tuning step, since a real
    deployment normally wouldn't have identity ground truth.

NOTE ON INTERACTION WITH RE-CLUSTERING
    clustering.cluster() reruns HDBSCAN from scratch on the raw embeddings
    and overwrites cluster_results + cluster folders + centroids for that
    run_label. Manual merges/reassignments made through Flow 1/2 below will
    be WIPED OUT the next time someone re-runs clustering, because HDBSCAN
    has no memory of them. Treat a re-run as "reset to the algorithmic
    ground truth" and Flow 1/2 as an incremental layer applied in between
    re-runs, not something that survives one.
"""

from __future__ import annotations

import itertools
import logging
from typing import Dict, List, Optional

import numpy as np

import config
from storage import Storage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity, in [0, 2]. Renormalizes defensively even though
    stored embeddings are already L2-normalized (a freshly-averaged centroid
    isn't unit-length until renormalized)."""
    a = a / max(np.linalg.norm(a), 1e-10)
    b = b / max(np.linalg.norm(b), 1e-10)
    return float(1.0 - np.dot(a, b))


# ---------------------------------------------------------------------------
# Centroid construction (called by clustering.cluster() after HDBSCAN)
# ---------------------------------------------------------------------------

def build_centroids_from_clusters(df, run_label: str, db: Storage):
    """
    df: the DataFrame produced in clustering.cluster() — must have face_id,
    vector, cluster_label columns. Rebuilds cluster_centroids for run_label
    from scratch so centroids never drift out of sync with a fresh HDBSCAN
    run. Noise (-1) is excluded — it isn't a cluster.
    """
    db.delete_all_centroids(run_label)

    non_noise = df[df["cluster_label"] != -1]
    for cluster_label, group in non_noise.groupby("cluster_label"):
        vectors = np.stack(group["vector"].values)
        centroid = vectors.mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-10)

        dists = [cosine_distance(centroid, v) for v in vectors]
        representative_face_id = int(group["face_id"].iloc[int(np.argmin(dists))])

        db.upsert_centroid(run_label, int(cluster_label), centroid, len(group), representative_face_id)

    logger.info("Built %d centroids for run_label=%s", non_noise["cluster_label"].nunique(), run_label)


# ---------------------------------------------------------------------------
# Flow 1 — new face vs. all existing centroids
# ---------------------------------------------------------------------------

def recommend_for_embedding(embedding: np.ndarray, run_label: str, db: Storage,
                             thresholds: Optional[dict] = None) -> dict:
    """Core decision function for Flow 1 (and reused by noise reclamation)."""
    thresholds = thresholds or config.THRESHOLDS
    centroids_df = db.load_centroids_df(run_label)

    if centroids_df.empty:
        return {
            "status": "new_cluster",
            "reason": "no_existing_clusters",
            "nearest_cluster_label": None,
            "distance": None,
            "representative_face_id": None,
            "representative_image_path": None,
            "representative_identity": None,
        }

    distances = centroids_df["centroid_vector"].apply(lambda c: cosine_distance(embedding, c))
    best_idx = distances.idxmin()
    best_row = centroids_df.loc[best_idx]
    d = float(distances.loc[best_idx])

    if d < thresholds["t1"]:
        status = "auto_merge"
    elif d < thresholds["t2"]:
        status = "ask_user"
    else:
        status = "new_cluster"

    return {
        "status": status,
        "nearest_cluster_label": int(best_row["cluster_label"]),
        "distance": d,
        "representative_face_id": int(best_row["representative_face_id"]),
        "representative_image_path": best_row["representative_image_path"],
        "representative_identity": best_row["representative_identity"],
    }


def assign_face_to_cluster(face_id: int, run_label: str, cluster_label: int,
                            new_vector: np.ndarray, db: Storage):
    """
    Confirms a merge (auto or user-approved): writes the cluster_results row,
    updates the centroid as a member-count-weighted running average, and
    re-checks whether the representative ("thumbnail") should switch to the
    newly-added face if it now sits closer to the updated centroid.
    """
    db.set_face_cluster_label(run_label, face_id, cluster_label)

    centroids_df = db.load_centroids_df(run_label)
    match = centroids_df[centroids_df["cluster_label"] == cluster_label]

    if match.empty:
        db.upsert_centroid(run_label, cluster_label, new_vector, 1, face_id)
        return

    row = match.iloc[0]
    n = int(row["n_members"])
    old_centroid = row["centroid_vector"]

    new_n = n + 1
    updated_centroid = (old_centroid * n + new_vector) / new_n
    updated_centroid = updated_centroid / max(np.linalg.norm(updated_centroid), 1e-10)

    old_rep_vec = db.get_embedding_vector(int(row["representative_face_id"]))
    d_old_rep = cosine_distance(updated_centroid, old_rep_vec) if old_rep_vec is not None else float("inf")
    d_new = cosine_distance(updated_centroid, new_vector)
    representative_face_id = face_id if d_new < d_old_rep else int(row["representative_face_id"])

    db.upsert_centroid(run_label, cluster_label, updated_centroid, new_n, representative_face_id)


def create_new_cluster(face_id: int, run_label: str, new_vector: np.ndarray, db: Storage) -> int:
    """Starts a brand-new cluster (singleton) for a face that didn't match anything closely enough."""
    existing_labels = db.get_all_cluster_labels(run_label)
    new_label = (max(existing_labels) + 1) if existing_labels else 0
    db.set_face_cluster_label(run_label, face_id, new_label)
    db.upsert_centroid(run_label, new_label, new_vector, 1, face_id)
    return new_label


# ---------------------------------------------------------------------------
# Noise reclamation — route -1 points through the same Flow-1 logic
# ---------------------------------------------------------------------------

def reclaim_noise(run_label: str, db: Storage, thresholds: Optional[dict] = None,
                   auto_apply: bool = True) -> Dict[str, list]:
    """
    Runs every noise-labeled face through recommend_for_embedding() against
    the real (non-noise) centroids.
      - d < T1  -> auto-merged immediately if auto_apply=True
      - T1<=d<T2 -> returned as a suggestion for the user to confirm
      - d >= T2  -> left as noise (returned for visibility)
    """
    thresholds = thresholds or config.THRESHOLDS
    noise_face_ids = db.get_noise_face_ids(run_label)

    auto_merged, suggestions, left_as_noise = [], [], []

    for face_id in noise_face_ids:
        vec = db.get_embedding_vector(face_id)
        if vec is None:
            continue

        rec = recommend_for_embedding(vec, run_label, db, thresholds)
        rec["face_id"] = face_id

        if rec["status"] == "auto_merge":
            if auto_apply:
                assign_face_to_cluster(face_id, run_label, rec["nearest_cluster_label"], vec, db)
            auto_merged.append(rec)
        elif rec["status"] == "ask_user":
            suggestions.append(rec)
        else:
            left_as_noise.append(rec)

    logger.info("Noise reclamation [%s]: %d auto-merged, %d need review, %d left as noise",
                run_label, len(auto_merged), len(suggestions), len(left_as_noise))

    return {"auto_merged": auto_merged, "suggestions": suggestions, "left_as_noise": left_as_noise}


# ---------------------------------------------------------------------------
# Flow 2 — pairwise cluster-vs-cluster sweep (Google Photos style)
# ---------------------------------------------------------------------------

def pairwise_cluster_scan(run_label: str, db: Storage, thresholds: Optional[dict] = None) -> List[dict]:
    """
    Scans every pair of cluster centroids and flags pairs whose centroid
    distance falls below T2 as worth a human look (below T1 is flagged too,
    marked auto_mergeable, since two centroids ending up that close usually
    means something drifted after manual merges/reclamation).
    """
    thresholds = thresholds or config.THRESHOLDS
    centroids_df = db.load_centroids_df(run_label)

    labels = centroids_df["cluster_label"].tolist()
    vectors = centroids_df["centroid_vector"].tolist()

    suspicious = []
    for i, j in itertools.combinations(range(len(labels)), 2):
        d = cosine_distance(vectors[i], vectors[j])
        if d < thresholds["t2"]:
            row_i, row_j = centroids_df.iloc[i], centroids_df.iloc[j]
            suspicious.append({
                "cluster_a": int(row_i["cluster_label"]),
                "cluster_b": int(row_j["cluster_label"]),
                "distance": d,
                "auto_mergeable": d < thresholds["t1"],
                "rep_a_image_path": row_i["representative_image_path"],
                "rep_b_image_path": row_j["representative_image_path"],
                "rep_a_identity": row_i["representative_identity"],
                "rep_b_identity": row_j["representative_identity"],
            })

    suspicious.sort(key=lambda s: s["distance"])
    return suspicious


def merge_clusters(run_label: str, keep_label: int, merge_label: int, db: Storage) -> dict:
    """Folds merge_label's members into keep_label, recomputes the centroid
    over the combined membership, and drops the now-empty cluster's row."""
    db.reassign_cluster_label(run_label, merge_label, keep_label)

    all_face_ids = db.get_cluster_face_ids(run_label, keep_label)
    pairs = [(fid, db.get_embedding_vector(fid)) for fid in all_face_ids]
    pairs = [(fid, v) for fid, v in pairs if v is not None]

    vectors = np.stack([v for _, v in pairs])
    centroid = vectors.mean(axis=0)
    centroid = centroid / max(np.linalg.norm(centroid), 1e-10)

    dists = [(fid, cosine_distance(centroid, v)) for fid, v in pairs]
    representative_face_id = min(dists, key=lambda t: t[1])[0]

    db.upsert_centroid(run_label, keep_label, centroid, len(pairs), representative_face_id)
    db.delete_centroid(run_label, merge_label)

    return {"kept": keep_label, "merged_away": merge_label, "n_members": len(pairs)}


# ---------------------------------------------------------------------------
# Threshold calibration — data-driven, using ground-truth identity labels
# ---------------------------------------------------------------------------

def calibrate_thresholds(run_label: str, db: Storage,
                          percentile_t1: float = 95, percentile_t2: float = 5,
                          max_pairs: int = 20000, seed: int = 42) -> dict:
    """
    Suggests T1/T2 using the identity ground truth you already have (it came
    from your data/raw/<identity>/ folder structure) — this isn't available
    in a real deployment, but it's exactly what you want for a one-time
    offline tuning pass on a labelled set like this one.

    T1 suggestion: a high percentile (default 95th) of SAME-identity pairwise
    distances — "how far apart do two photos of the SAME person get, at
    worst, 95% of the time?" Distances below this are safe to auto-merge.

    T2 suggestion: a low percentile (default 5th) of DIFFERENT-identity
    pairwise distances — "how close do two DIFFERENT people's photos get, at
    closest, 5% of the time?" Distances above this are safe to call new.

    If suggested T1 > suggested T2, ArcFace doesn't cleanly separate your
    identities at any single global cutoff on this data — that's a real
    finding (`clean_separation: False`), not a bug.
    """
    df = db.load_embeddings_df()
    if df.empty or len(df) < 2:
        raise ValueError("Not enough embeddings to calibrate.")

    rng = np.random.default_rng(seed)
    n = len(df)
    vectors = np.stack(df["vector"].values)
    identities = df["identity"].values

    idx_pairs = list(itertools.combinations(range(n), 2))
    if len(idx_pairs) > max_pairs:
        chosen = rng.choice(len(idx_pairs), size=max_pairs, replace=False)
        idx_pairs = [idx_pairs[i] for i in chosen]

    same, diff = [], []
    for i, j in idx_pairs:
        d = cosine_distance(vectors[i], vectors[j])
        (same if identities[i] == identities[j] else diff).append(d)

    if not same or not diff:
        raise ValueError(
            "Calibration needs both same-identity and different-identity pairs in the sample — "
            "check that your dataset has more than one identity and multiple photos per identity."
        )

    t1 = float(np.percentile(same, percentile_t1))
    t2 = float(np.percentile(diff, percentile_t2))

    return {
        "n_same_pairs": len(same),
        "n_diff_pairs": len(diff),
        "same_identity_dist_mean": float(np.mean(same)),
        "diff_identity_dist_mean": float(np.mean(diff)),
        "suggested_t1": t1,
        "suggested_t2": t2,
        "clean_separation": t1 <= t2,
    }
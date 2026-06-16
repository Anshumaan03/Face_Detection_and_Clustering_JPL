"""
K-Means clustering on ArcFace embeddings from face_db.face_embeddings.

WHY K-MEANS FOR FACES:
  K-Means requires you to specify K (number of clusters) upfront.
  In a real-world unlabelled scenario you don't know K, but here we
  sweep a range around the true number of identities to evaluate how
  sensitive the algorithm is.

  K-Means uses euclidean distance. Since embeddings are L2-normalized
  (unit vectors), euclidean distance is equivalent to cosine distance,
  so K-Means is a valid choice.

LIMITATION vs HDBSCAN:
  K-Means forces every point into a cluster — no noise/outlier handling.
  Group photos or ambiguous faces get assigned to whichever centroid is
  closest, which can drag down purity scores.

HOW TO RUN:
  python kmeans_pipeline.py
"""

import json
import numpy as np
import mysql.connector
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

DB_CONFIG = {
    "host"    : "127.0.0.1",
    "user"    : "root",
    "password": "Anshu@2003",
    "database": "face_db"
}

# Sweep K around the true number of identities
# e.g. if you have 2 identities, sweep K = 1..6
K_SWEEP = [2, 3, 4, 5, 6]

JSON_FILE = "clustering_results_kmeans.json"


def _purity(ground_truths: np.ndarray, labels: np.ndarray) -> float:
    total = len(labels)
    score = 0
    for cid in set(labels):
        if cid == -1:
            continue
        mask = labels == cid
        score += Counter(ground_truths[mask]).most_common(1)[0][1]
    return score / total if total > 0 else 0.0


def run_kmeans_sweep():
    print("🔌 Connecting to MySQL...")
    try:
        conn   = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT identity_label, embedding_json FROM face_embeddings ORDER BY identity_label, file_name"
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
    except mysql.connector.Error as e:
        print(f"❌ DB Error: {e}")
        return

    total = len(rows)
    if total == 0:
        print("⚠️  No embeddings found.")
        return

    print(f"📥 Loaded {total} face vectors.")

    raw_labels    = [r[0] for r in rows]
    le            = LabelEncoder()
    ground_truths = le.fit_transform(raw_labels)
    n_identities  = len(le.classes_)
    print(f"🏷️  {n_identities} identities: {list(le.classes_)}\n")

    X = np.array([json.loads(r[1]) for r in rows], dtype=np.float32)

    norms = np.linalg.norm(X, axis=1)
    print(f"📐 Norm check — mean: {norms.mean():.4f}, std: {norms.std():.4f}\n")

    print("🚀 K-MEANS PARAMETRIC SWEEP")
    print("-" * 70)
    print(f"{'K':>6}{'clusters':>10}{'NMI':>10}{'ARI':>10}{'Purity':>10}")
    print("-" * 70)

    best_ari    = -1
    best_k      = None
    best_labels = None
    all_results = []

    for k in K_SWEEP:
        if k > total:
            continue

        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',   # smarter centroid init — faster convergence
            n_init=20,          # run 20 times, keep best inertia
            max_iter=500,
            random_state=42
        )
        labels = kmeans.fit_predict(X)

        nmi    = normalized_mutual_info_score(ground_truths, labels)
        ari    = adjusted_rand_score(ground_truths, labels)
        purity = _purity(ground_truths, labels)

        flag = " ← BEST" if ari > best_ari else ""
        if ari > best_ari:
            best_ari    = ari
            best_k      = k
            best_labels = labels.copy()

        print(f"{k:>6}{k:>10}{nmi:>10.4f}{ari:>10.4f}{purity:>10.4f}{flag}")

        all_results.append({
            "k": k, "nmi": float(nmi), "ari": float(ari), "purity": float(purity)
        })

    print("-" * 70)
    print(f"\n🏆 Best K={best_k}  NMI={normalized_mutual_info_score(ground_truths, best_labels):.4f}  ARI={best_ari:.4f}")

    print("\n📊 Per-identity breakdown:")
    print(f"  {'Identity':<30} {'Count':>6}  Assigned clusters")
    print(f"  {'-'*60}")
    per_identity = {}
    for identity in le.classes_:
        mask     = np.array(raw_labels) == identity
        assigned = sorted(set(best_labels[mask].tolist()))
        count    = int(mask.sum())
        per_identity[identity] = {
            "total":      count,
            "noise":      0,          # K-Means assigns everything, no noise
            "n_clusters": len(assigned)
        }
        print(f"  {identity:<30} {count:>6}  {assigned}")

    output = {
        "algorithm":   "kmeans",
        "pred_labels": [int(l) for l in best_labels],
        "best_params": {
            "k":         best_k,
            "n_clusters": best_k,
            "nmi":       float(normalized_mutual_info_score(ground_truths, best_labels)),
            "ari":       float(best_ari),
            "purity":    float(_purity(ground_truths, best_labels)),
            "noise_pct": 0.0,         # K-Means has no noise points
        },
        "all_results":  all_results,
        "per_identity": per_identity
    }

    with open(JSON_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Saved {JSON_FILE}")

    # ── Create cluster folders ──
    _create_folders(JSON_FILE, raw_labels, rows, Path("results_kmeans"))


def _create_folders(json_file, raw_labels, db_rows, results_dir):
    import shutil
    with open(json_file) as f:
        pred_labels = json.load(f)["pred_labels"]

    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir()

    base = Path("data/raw")
    for (identity_label, _), cluster_id in zip(db_rows, pred_labels):
        # find the file_name — re-query not needed since db_rows has it
        pass

    # Re-fetch with file_name included
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, identity_label FROM face_embeddings ORDER BY identity_label, file_name")
    file_rows = cursor.fetchall()
    cursor.close()
    conn.close()

    missing = []
    for (file_name, identity_label), cluster_id in zip(file_rows, pred_labels):
        src = base / identity_label / file_name
        if not src.exists():
            missing.append(str(src))
            continue
        folder = results_dir / ("Noise" if cluster_id == -1 else f"Cluster_{cluster_id}")
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, folder / file_name)

    print(f"\n📁 Cluster folders → {results_dir.resolve()}")
    for folder in sorted(results_dir.iterdir()):
        print(f"   {folder.name}: {len(list(folder.glob('*')))} images")
    if missing:
        print(f"   ⚠️  {len(missing)} missing files")


if __name__ == "__main__":
    run_kmeans_sweep()

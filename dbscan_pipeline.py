"""
DBSCAN clustering on ArcFace embeddings from face_db.face_embeddings.

WHY DBSCAN FOR FACES:
  Like HDBSCAN, DBSCAN is a density-based algorithm — it finds clusters
  of arbitrary shape and labels low-density points as noise (-1).
  This makes it naturally robust to group photos or ambiguous faces.

DBSCAN vs HDBSCAN:
  DBSCAN requires two fixed parameters: eps (neighbourhood radius) and
  min_samples. It is more sensitive to eps — a wrong value collapses
  everything into one cluster or noise. HDBSCAN is a hierarchical
  extension that automatically finds the best eps per cluster, making
  it more robust.

  For face embeddings (L2-normalized, cosine-like distances):
    eps ~ 0.4–0.9  is a typical range (euclidean on unit vectors)
    min_samples = 1–3 for small datasets

HOW TO RUN:
  python dbscan_pipeline.py
"""

import json
import numpy as np
import mysql.connector
from pathlib import Path
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

DB_CONFIG = {
    "host"    : "127.0.0.1",
    "user"    : "root",
    "password": "Anshu@2003",
    "database": "face_db"
}

# Sweep eps and min_samples
# eps is the neighbourhood radius in euclidean space
# For L2-normalised 512-d embeddings, cosine distance ~ euclidean distance
# Typical good range: 0.4 – 1.0
EPS_SWEEP         = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MIN_SAMPLES_SWEEP = [1, 2, 3]

JSON_FILE = "clustering_results_dbscan.json"


def _purity(ground_truths: np.ndarray, labels: np.ndarray) -> float:
    total = len(labels)
    score = 0
    for cid in set(labels):
        if cid == -1:
            continue
        mask = labels == cid
        score += Counter(ground_truths[mask]).most_common(1)[0][1]
    return score / total if total > 0 else 0.0


def run_dbscan_sweep():
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

    print("🚀 DBSCAN PARAMETRIC SWEEP")
    print("-" * 90)
    print(f"{'eps':>8}{'min_smp':>9}{'clusters':>10}{'noise':>8}{'noise%':>8}{'NMI':>10}{'ARI':>10}{'Purity':>10}")
    print("-" * 90)

    best_ari    = -1
    best_cfg    = None
    best_labels = None
    all_results = []

    for eps in EPS_SWEEP:
        for min_smp in MIN_SAMPLES_SWEEP:

            dbscan = DBSCAN(
                eps=eps,
                min_samples=min_smp,
                metric='euclidean',   # euclidean on L2-normalised ≈ cosine
                n_jobs=-1             # use all CPU cores
            )
            labels = dbscan.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise    = int(np.sum(labels == -1))
            noise_pct  = n_noise / total * 100

            nmi    = normalized_mutual_info_score(ground_truths, labels)
            ari    = adjusted_rand_score(ground_truths, labels)
            purity = _purity(ground_truths, labels)

            flag = " ← BEST" if ari > best_ari else ""
            if ari > best_ari:
                best_ari    = ari
                best_cfg    = (eps, min_smp)
                best_labels = labels.copy()

            print(f"{eps:>8.2f}{min_smp:>9}{n_clusters:>10}{n_noise:>8}{noise_pct:>7.1f}%"
                  f"{nmi:>10.4f}{ari:>10.4f}{purity:>10.4f}{flag}")

            all_results.append({
                "eps": eps, "min_samples": min_smp,
                "n_clusters": n_clusters, "noise_pct": float(noise_pct),
                "nmi": float(nmi), "ari": float(ari), "purity": float(purity)
            })

    print("-" * 90)
    print(f"\n🏆 Best: eps={best_cfg[0]}, min_samples={best_cfg[1]}")
    print(f"   NMI={normalized_mutual_info_score(ground_truths, best_labels):.4f}  ARI={best_ari:.4f}")
    print(f"   Clusters found: {len(set(best_labels))-(1 if -1 in best_labels else 0)} / Target: {n_identities}")

    print("\n📊 Per-identity breakdown:")
    print(f"  {'Identity':<30} {'Count':>6}  Assigned clusters")
    print(f"  {'-'*60}")
    per_identity = {}
    for identity in le.classes_:
        mask     = np.array(raw_labels) == identity
        assigned = sorted(set(best_labels[mask].tolist()))
        count    = int(mask.sum())
        n_noise  = int(np.sum(best_labels[mask] == -1))
        per_identity[identity] = {
            "total":      count,
            "noise":      n_noise,
            "n_clusters": len(set(assigned) - {-1})
        }
        print(f"  {identity:<30} {count:>6}  {assigned}")

    output = {
        "algorithm":   "dbscan",
        "pred_labels": [int(l) for l in best_labels],
        "best_params": {
            "eps":        best_cfg[0],
            "min_samples": best_cfg[1],
            "n_clusters": len(set(best_labels)) - (1 if -1 in best_labels else 0),
            "nmi":        float(normalized_mutual_info_score(ground_truths, best_labels)),
            "ari":        float(best_ari),
            "purity":     float(_purity(ground_truths, best_labels)),
            "noise_pct":  float(np.sum(best_labels == -1) / total * 100),
        },
        "all_results":  all_results,
        "per_identity": per_identity
    }

    with open(JSON_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Saved {JSON_FILE}")

    # ── Create cluster folders ──
    _create_folders(JSON_FILE, Path("results_dbscan"))


def _create_folders(json_file, results_dir):
    import shutil
    with open(json_file) as f:
        pred_labels = json.load(f)["pred_labels"]

    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, identity_label FROM face_embeddings ORDER BY identity_label, file_name")
    file_rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir()

    base = Path("data/raw")
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
    run_dbscan_sweep()

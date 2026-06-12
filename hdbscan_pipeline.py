import json
import mysql.connector
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

# pip install hdbscan
import hdbscan

DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "Anshu@2003",
    "database": "face_db"
}

# HDBSCAN hyperparameter sweep
# min_cluster_size = minimum images to form an identity cluster
# For 10 images/person: sweep 2–5
MIN_CLUSTER_SIZE_SWEEP = [2, 3, 4, 5]
MIN_SAMPLES_SWEEP      = [1, 2, 3]

def run_hdbscan_sweep():
    print("🔌 Connecting to MySQL...")
    try:
        conn   = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        # FIX: ORDER BY ensures deterministic row ordering
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
        print("⚠️  No data found.")
        return

    print(f"📥 Loaded {total} face vectors.")

    # Encode string labels to integers for metric functions
    raw_labels = [row[0] for row in rows]
    le = LabelEncoder()
    ground_truths = le.fit_transform(raw_labels)
    n_identities  = len(le.classes_)
    print(f"🏷️  {n_identities} unique identities: {list(le.classes_)}\n")

    X = np.array([json.loads(row[1]) for row in rows], dtype=np.float32)

    # Verify L2 normalization — all norms should be ~1.0
    norms = np.linalg.norm(X, axis=1)
    print(f"📐 Embedding norm check — mean: {norms.mean():.4f}, std: {norms.std():.4f}")
    print("   (Should be mean≈1.0, std≈0.0 — if not, re-run ingestion)\n")

    print("🚀 HDBSCAN PARAMETRIC SWEEP")
    print("-" * 90)
    print(f"{'min_cls':>8}{'min_smp':>9}{'clusters':>10}{'noise':>8}{'noise%':>8}{'NMI':>10}{'ARI':>10}")
    print("-" * 90)

    best_ari   = -1
    best_cfg   = None
    best_labels= None

    for min_cls in MIN_CLUSTER_SIZE_SWEEP:
        for min_smp in MIN_SAMPLES_SWEEP:
            if min_smp > min_cls:
                continue  # invalid combination

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cls,
                min_samples=min_smp,
                metric='euclidean',       # euclidean on L2-normalized = cosine distance
                cluster_selection_method='eom',  # excess of mass — better for face clusters
                prediction_data=True
            )
            labels = clusterer.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise    = int(np.sum(labels == -1))
            noise_pct  = n_noise / total * 100

            # NMI and ARI handle noise points (-1) correctly
            nmi = normalized_mutual_info_score(ground_truths, labels)
            ari = adjusted_rand_score(ground_truths, labels)

            flag = " ← BEST" if ari > best_ari else ""
            if ari > best_ari:
                best_ari    = ari
                best_cfg    = (min_cls, min_smp)
                best_labels = labels.copy()

            print(f"{min_cls:>8}{min_smp:>9}{n_clusters:>10}{n_noise:>8}{noise_pct:>7.1f}%{nmi:>10.4f}{ari:>10.4f}{flag}")

    print("-" * 90)
    print(f"\n🏆 Best config: min_cluster_size={best_cfg[0]}, min_samples={best_cfg[1]}")
    print(f"   NMI = {normalized_mutual_info_score(ground_truths, best_labels):.4f}")
    print(f"   ARI = {best_ari:.4f}")
    print(f"   Target identities: {n_identities}  |  Clusters found: {len(set(best_labels)) - (1 if -1 in best_labels else 0)}")

    # Per-identity breakdown
    print("\n📊 Per-identity cluster assignment:")
    print(f"  {'Identity':<30} {'Count':>6} {'Assigned clusters'}")
    print(f"  {'-'*60}")
    for identity in le.classes_:
        mask    = np.array(raw_labels) == identity
        assigned= set(best_labels[mask])
        count   = int(mask.sum())
        print(f"  {identity:<30} {count:>6}  {sorted(assigned)}")

if __name__ == "__main__":
    run_hdbscan_sweep()
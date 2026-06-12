import json
import mysql.connector
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.pairwise import pairwise_distances

DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "Anshu@2003",
    "database": "face_db"
}

# 🌟 Micro-focused sweep targeting the compressed spatial bounds of your vectors
EPS_MICRO_SWEEP = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.24]
MIN_SAMPLES = 3

def run_micro_sweep():
    print("🔌 Pulling data from MySQL...")
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT identity_label, embedding_json FROM face_embeddings")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    total_records = len(rows)
    print(f"📥 Loaded {total_records} unique vectors.")

    ground_truths = [row[0] for row in rows]
    X = np.array([json.loads(row[1]) for row in rows], dtype=np.float32)
    
    # 📐 Compute raw Cosine Distance Matrix
    print("📐 Precomputing Pairwise Cosine Distance Matrix...")
    raw_cosine_matrix = pairwise_distances(X, metric='cosine').astype(np.float64)
    
    print(f"📊 Vector Space Diagnostics:")
    print(f"   - Minimum Distance found: {np.min(raw_cosine_matrix[raw_cosine_matrix > 0]):.4f}")
    print(f"   - Maximum Distance found: {np.max(raw_cosine_matrix):.4f}")
    print(f"   - Average Distance found: {np.mean(raw_cosine_matrix):.4f}")

    # =====================================================================
    # TEST RUN A: Standard Micro-Sweep (Looking inside the tight cone)
    # =====================================================================
    print("\n🚀 RUNNING STRATIFIED MICRO-SWEEP (RAW COSINE SPACE)...")
    print("-" * 85)
    print(f"{'Epsilon (ε)':<12}{'Clusters':<12}{'Noise Points':<18}{'NMI Score':<14}{'ARI Score':<14}")
    print("-" * 85)
    
    for eps in EPS_MICRO_SWEEP:
        clusterer = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, metric='precomputed', n_jobs=-1)
        labels = clusterer.fit_predict(raw_cosine_matrix)
        discovered = len(set(labels)) - (1 if -1 in labels else 0)
        noise = np.sum(labels == -1)
        nmi = normalized_mutual_info_score(ground_truths, labels)
        ari = adjusted_rand_score(ground_truths, labels)
        print(f"{eps:<12.2f}{discovered:<12}{f'{noise} ({noise/total_records*100:.1f}%)':<18}{nmi:<14.4f}{ari:<14.4f}")
    print("-" * 85)

    # =====================================================================
    # TEST RUN B: Amplified Distance Space (Stretching the boundaries)
    # =====================================================================
    print("\n🌟 RUNNING AMPLIFIED SWEEP (MIN-MAX SCALED DISTANCE MANIFOLD)...")
    # This mathematically stretches your cramped distances across a full 0.0 to 1.0 spectrum
    scaled_matrix = (raw_cosine_matrix - raw_cosine_matrix.min()) / (raw_cosine_matrix.max() - raw_cosine_matrix.min())
    
    print("-" * 85)
    print(f"{'Scaled ε':<12}{'Clusters':<12}{'Noise Points':<18}{'NMI Score':<14}{'ARI Score':<14}")
    print("-" * 85)
    
    for eps in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        clusterer = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, metric='precomputed', n_jobs=-1)
        labels = clusterer.fit_predict(scaled_matrix)
        discovered = len(set(labels)) - (1 if -1 in labels else 0)
        noise = np.sum(labels == -1)
        nmi = normalized_mutual_info_score(ground_truths, labels)
        ari = adjusted_rand_score(ground_truths, labels)
        print(f"{eps:<12.2f}{discovered:<12}{f'{noise} ({noise/total_records*100:.1f}%)':<18}{nmi:<14.4f}{ari:<14.4f}")
    print("-" * 85)

if __name__ == "__main__":
    run_micro_sweep()
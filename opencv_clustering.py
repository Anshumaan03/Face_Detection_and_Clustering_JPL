import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

VEC_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/data/aligned_embeddings/opencv_vectors.npy"
LBL_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/data/aligned_embeddings/opencv_labels.npy"

def run_blind_opencv():
    print("="*60)
    print("[EXECUTION] STARTING BLIND TRACK B: OPENCV ALIGNED")
    print("="*60)
    
    X = np.load(VEC_PATH)
    true_labels = np.load(LBL_PATH)
    K_true = len(np.unique(true_labels))
    
    print(f"[INFO] Data loaded. Vectors: {X.shape} | Real Unique People (K): {K_true}")
    
    # 1. t-SNE Projection
    print("[PROCESSING] Computing 2D t-SNE coordinates...")
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42)
    X_2d = tsne.fit_transform(X)
    
    # 2. Blind K-Means (Defaults to n_clusters=8)
    print("[RUNNING] Executing Blind K-Means (No K provided)...")
    km_blind = KMeans(random_state=42, n_init=10) # Blindly defaulting to 8
    km_labels = km_blind.fit_predict(X)
    nmi_km = normalized_mutual_info_score(true_labels, km_labels)
    ari_km = adjusted_rand_score(true_labels, km_labels)
    
    # 3. Standard DBSCAN (No custom micro-family tuning)
    print("[RUNNING] Executing Standard DBSCAN (eps=0.50, min_samples=4)...")
    db_standard = DBSCAN(eps=0.50, min_samples=4)
    db_labels = db_standard.fit_predict(X)
    db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    nmi_db = normalized_mutual_info_score(true_labels, db_labels)
    ari_db = adjusted_rand_score(true_labels, db_labels)
    
    # Print out results
    print("\n📊 BLIND OPENCV BENCHMARK RESULTS:")
    print(f"🔹 Blind K-Means (Forced {len(set(km_labels))} groups) -> NMI: {nmi_km:.4f} | ARI: {ari_km:.4f}")
    print(f"🔸 Standard DBSCAN (Found {db_clusters} groups) -> NMI: {nmi_db:.4f} | ARI: {ari_db:.4f}")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Blind Tracking Execution: OpenCV Aligned Embeddings Space", fontsize=16, fontweight='bold')
    
    ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=km_labels, cmap='tab10', s=12, alpha=0.6)
    ax1.set_title(f"Blind K-Means (Forced Clusters: 8)\nNMI Accuracy: {nmi_km:.4f}", fontsize=12, fontweight='bold', color='darkred')
    
    ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=db_labels, cmap='turbo', s=12, alpha=0.6)
    ax2.set_title(f"Standard DBSCAN (Discovered Clusters: {db_clusters})\nNMI Accuracy: {nmi_db:.4f}", fontsize=12, fontweight='bold')
    
    for ax in [ax1, ax2]: ax.grid(True, linestyle='--', alpha=0.5)
    
    out_img = "data/debug/blind_opencv_output.png"
    plt.tight_layout()
    plt.savefig(out_img, dpi=300)
    print(f"[SUCCESS] Isolated Blind OpenCV plot saved to: {out_img}\n")
    plt.show()

if __name__ == "__main__":
    run_blind_opencv()
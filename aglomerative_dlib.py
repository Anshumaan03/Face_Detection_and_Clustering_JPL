import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score

VEC_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/data/aligned_embeddings/dlib_vectors.npy"
LBL_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/data/aligned_embeddings/dlib_labels.npy"

def run_isolated_dlib_agg():
    print("="*60)
    print("[EXECUTION] RUNNING ISOLATED AGGLOMERATIVE CLUSTERING: DLIB")
    print("="*60)
    
    if not os.path.exists(VEC_PATH) or not os.path.exists(LBL_PATH):
        print("[ERROR] Embedding matrices not found.")
        return

    X = np.load(VEC_PATH)
    true_labels = np.load(LBL_PATH)
    K_target = len(np.unique(true_labels))
    print(f"[INFO] Dataset Loaded -> Vectors: {X.shape} | Real People (K): {K_target}")

    # 1. Dimension reduction for visual mapping
    print("[PROCESSING] Generating 2D t-SNE embedding projection...")
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42)
    X_2d = tsne.fit_transform(X)

    # 2. Agglomerative Execution
    print("[RUNNING] Building Hierarchical Tree (Linkage=Ward)...")
    agg = AgglomerativeClustering(n_clusters=K_target, linkage='ward')
    agg_labels = agg.fit_predict(X)

    # 3. Calculate Benchmarks
    print("[METRICS] Evaluating grouping accuracy scores...")
    nmi = normalized_mutual_info_score(true_labels, agg_labels)
    ari = adjusted_rand_score(true_labels, agg_labels)
    sil = silhouette_score(X, agg_labels)

    print("\n" + "═"*50)
    print("        ISOLATED TRACK A: DLIB AGGLOMERATIVE REPORT")
    print("═"*50)
    print(f"🌲 Agglomerative Ward Tree:")
    print(f"   - Clusters Enforced (K):              {K_target}")
    print(f"   - NMI Score (Global Structure):       {nmi:.4f}")
    print(f"   - ARI Score (Pair-wise Precision):    {ari:.4f}")
    print(f"   - Silhouette Score (Separation):      {sil:.4f}")
    print("═"*50 + "\n")

    # 4. Generate Plot Layout
    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=agg_labels, cmap='turbo', s=12, alpha=0.6)
    plt.title(f"Dlib Aligned Space: Isolated Agglomerative Clustering\nEnforced K: {K_target} | NMI: {nmi:.4f} | ARI: {ari:.4f}", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    out_img = "data/debug/isolated_dlib_agg.png"
    plt.tight_layout()
    plt.savefig(out_img, dpi=300)
    print(f"[SUCCESS] Isolated Dlib Agglomerative plot saved to: {out_img}")
    plt.show()

if __name__ == "__main__":
    run_isolated_dlib_agg()
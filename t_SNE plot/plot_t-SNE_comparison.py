import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def safe_load(vec_path, lbl_path):
    if not os.path.exists(vec_path) or not os.path.exists(lbl_path):
        print(f"[ERROR] Missing file entry: {vec_path} or {lbl_path}")
        return None, None
    return np.load(vec_path), np.load(lbl_path)

def generate_alignment_showdown():
    print("[INFO] Loading all 4 embedding matrices...")
    
    # Load Aligned Datasets
    dlib_al_v, dlib_al_l = safe_load("data/aligned_embeddings/dlib_vectors.npy", "data/aligned_embeddings/dlib_labels.npy")
    cv_al_v, cv_al_l = safe_load("data/aligned_embeddings/opencv_vectors.npy", "data/aligned_embeddings/opencv_labels.npy")
    
    # Load Unaligned Baselines
    dlib_un_v, dlib_un_l = safe_load("data/unaligned_embeddings/dlib_unaligned_vectors.npy", "data/unaligned_embeddings/dlib_unaligned_labels.npy")
    cv_un_v, cv_un_l = safe_load("data/unaligned_embeddings/opencv_unaligned_vectors.npy", "data/unaligned_embeddings/opencv_unaligned_labels.npy")
    
    if any(matrix is None for matrix in [dlib_al_v, cv_al_v, dlib_un_v, cv_un_v]):
        print("[ERROR] Ensure all prior embedding extraction steps completed successfully.")
        return

    # Setup 2x2 multi-plot grid layout
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.suptitle("The Structural Impact of Facial Alignment: t-SNE Comparative Analysis", fontsize=24, fontweight='bold')

    # Fixed t-SNE hyperparameter configs for absolute baseline fairness across tests
    # Note: max_iter is used for scikit-learn 1.5+ compatibility
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42)

    # ----------------------------------------------------
    # PANEL 1: DLIB ALIGNED (TOP LEFT)
    # ----------------------------------------------------
    print("[PROCESSING 1/4] Reducing Dlib Aligned Space...")
    coords1 = tsne.fit_transform(dlib_al_v)
    _, inv1 = np.unique(dlib_al_l, return_inverse=True)
    axes[0, 0].scatter(coords1[:, 0], coords1[:, 1], c=inv1, cmap='tab20', s=8, alpha=0.6)
    axes[0, 0].set_title(f"A1. Dlib ALIGNED (Vectors: {dlib_al_v.shape[0]})", fontsize=14, fontweight='bold', color='darkgreen')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    # ----------------------------------------------------
    # PANEL 2: DLIB UNALIGNED (TOP RIGHT)
    # ----------------------------------------------------
    print("[PROCESSING 2/4] Reducing Dlib Unaligned Space...")
    coords2 = tsne.fit_transform(dlib_un_v)
    _, inv2 = np.unique(dlib_un_l, return_inverse=True)
    axes[0, 1].scatter(coords2[:, 0], coords2[:, 1], c=inv2, cmap='tab20', s=8, alpha=0.6)
    axes[0, 1].set_title(f"A2. Dlib UNALIGNED (Vectors: {dlib_un_v.shape[0]})", fontsize=14, fontweight='bold', color='darkred')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    # ----------------------------------------------------
    # PANEL 3: OPENCV ALIGNED (BOTTOM LEFT)
    # ----------------------------------------------------
    print("[PROCESSING 3/4] Reducing OpenCV Aligned Space...")
    coords3 = tsne.fit_transform(cv_al_v)
    _, inv3 = np.unique(cv_al_l, return_inverse=True)
    axes[1, 0].scatter(coords3[:, 0], coords3[:, 1], c=inv3, cmap='tab20', s=8, alpha=0.6)
    axes[1, 0].set_title(f"B1. OpenCV ALIGNED (Vectors: {cv_al_v.shape[0]})", fontsize=14, fontweight='bold', color='darkgreen')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)

    # ----------------------------------------------------
    # PANEL 4: OPENCV UNALIGNED (BOTTOM RIGHT)
    # ----------------------------------------------------
    print("[PROCESSING 4/4] Reducing OpenCV Unaligned Space...")
    coords4 = tsne.fit_transform(cv_un_v)
    _, inv4 = np.unique(cv_un_l, return_inverse=True)
    axes[1, 1].scatter(coords4[:, 0], coords4[:, 1], c=inv4, cmap='tab20', s=8, alpha=0.6)
    axes[1, 1].set_title(f"B2. OpenCV UNALIGNED (Vectors: {cv_un_v.shape[0]})", fontsize=14, fontweight='bold', color='darkred')
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    # Clean axes labels
    for ax in axes.flat:
        ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=11)

    output_dir = "data/debug"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "alignment_vs_unaligned_tsne.png")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n[SUCCESS] Master 4-panel breakdown saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    generate_alignment_showdown()
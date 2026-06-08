import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_data(vec_path, lbl_path):
    if not os.path.exists(vec_path) or not os.path.exists(lbl_path):
        print(f"[ERROR] Missing files: Check {vec_path} or {lbl_path}")
        return None, None
    return np.load(vec_path), np.load(lbl_path)

def plot_tsne_comparison():
    print("[INFO] Loading data matrices...")
    # Load Dlib data
    dlib_vecs, dlib_lbls = load_data("data/embeddings/dlib_vectors.npy", "data/embeddings/dlib_labels.npy")
    # Load OpenCV data
    cv_vecs, cv_lbls = load_data("data/embeddings/opencv_vectors.npy", "data/embeddings/opencv_labels.npy")
    
    if dlib_vecs is None or cv_vecs is None:
        return

    # Setup the plot layout (Side-by-Side Comparison)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle("High-Dimensional Feature Space Comparison (t-SNE Reduction)", fontsize=18, fontweight='bold')

    # Configs for t-SNE
    # perplexity handles balance between local and global aspects of your data
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42)

    # --------------------
    # TRACK A: DLIB PLOT
    # --------------------
    print("[PROCESSING] Computing 2D t-SNE coordinates for Dlib Track...")
    dlib_2d = tsne.fit_transform(dlib_vecs)
    
    # Generate unique integer hashes for labels to assign distinct colors
    unique_dlib_lbls, dlib_inverse = np.unique(dlib_lbls, return_inverse=True)
    
    scatter1 = axes[0].scatter(
        dlib_2d[:, 0], dlib_2d[:, 1], 
        c=dlib_inverse, cmap='tab20', s=10, alpha=0.7
    )
    axes[0].set_title(f"Dlib ResNet Track (Faces: {dlib_vecs.shape[0]})", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("t-SNE Dimension 1")
    axes[0].set_ylabel("t-SNE Dimension 2")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # --------------------
    # TRACK B: OPENCV PLOT
    # --------------------
    print("[PROCESSING] Computing 2D t-SNE coordinates for OpenCV Track...")
    cv_2d = tsne.fit_transform(cv_vecs)
    
    unique_cv_lbls, cv_inverse = np.unique(cv_lbls, return_inverse=True)
    
    scatter2 = axes[1].scatter(
        cv_2d[:, 0], cv_2d[:, 1], 
        c=cv_inverse, cmap='tab20', s=10, alpha=0.7
    )
    axes[1].set_title(f"Pure OpenCV FaceNet Track (Faces: {cv_vecs.shape[0]})", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Save and output the final chart
    output_dir = "data/debug"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "combined_tsne_evaluation.png")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n[SUCCESS] Unified t-SNE plot report generated and saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_tsne_comparison()
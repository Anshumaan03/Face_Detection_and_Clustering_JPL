import os

DATA_ROOT = "/Users/anshumaansinghrathore/Desktop/Face Clustering/data/raw"
OUTPUT_ROOT = "/Users/anshumaansinghrathore/Desktop/Face Clustering/results_comparison_pipeline"
CLUSTERS_ROOT = os.path.join(OUTPUT_ROOT, "clusters")
UPLOADS_ROOT = os.path.join(OUTPUT_ROOT, "uploads")

ARCFACE_ONNX_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/Arcface.onnx"

# ---------------------------------------------------------------------------
# Detector: InsightFace RetinaFace (buffalo_l pack) — shared once per image
# ---------------------------------------------------------------------------
INSIGHTFACE_DET_MODEL = "buffalo_l"
DETECTOR_CTX_ID = -1   # 0 = first GPU, -1 = CPU

# ---------------------------------------------------------------------------
# Embedding model: ArcFace only
# ---------------------------------------------------------------------------
ARCFACE_INPUT_SIZE = 112          # ArcFace's expected crop size
EMBEDDING_DIM = 512

MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Anshu@2003",
    "database": "face_db_arcface",
}

HDBSCAN_PARAMS = {
    "min_cluster_size": 5,
    "min_samples": 3,
    "metric": "euclidean",   # embeddings are L2-normalized before this, so euclidean == cosine rank-order
    "cluster_selection_method": "eom",
}

# ---------------------------------------------------------------------------
# Recommendation system (Flow 1: new-face-vs-centroid, Flow 2: cluster-vs-cluster)
# ---------------------------------------------------------------------------
# Distances are COSINE distances (1 - cosine_similarity) between L2-normalized
# ArcFace embeddings, so they live in [0, 2]; smaller = more similar.
#
# These two numbers are NOT measured on your data — they're a reasonable
# starting point for ArcFace specifically. Your dataset already carries
# ground-truth identity labels (from the folder structure), so before trusting
# these in the app, run recommendation.calibrate_thresholds(run_label) once
# and replace T1/T2 below with what it suggests.
THRESHOLDS = {
    "t1": 1.018,   # d < T1  -> auto-merge, high confidence same person
    "t2": 0.889,   # T1<=d<T2 -> ask user; d >= T2 -> new cluster
}
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
# These numbers are NOT measured on your data — they're a reasonable
# starting point for ArcFace specifically. Your dataset already carries
# ground-truth identity labels (from the folder structure), so before trusting
# these in the app, run `python runpipeline.py --calibrate-thresholds` (or the
# Calibrate tab) once you have embeddings in the DB. Prefer whatever it reports
# as `suggested_t1_precision` over the plain percentile `suggested_t1` — the
# precision version accounts for how close the nearest different-identity
# pair actually sits, which the percentile-of-same-identity number ignores.
THRESHOLDS = {
    "t1": 0.40,             # d < T1  -> auto-merge, high confidence same person
    "t2": 0.60,             # T1<=d<T2 -> ask user; d >= T2 -> new cluster
    "spread_margin": 1.15,  # auto-merge also requires d <= (nearest cluster's own historical
                            # max distance-from-centroid) * spread_margin -- stops a tight
                            # cluster from being auto-merged into just because the global T1
                            # ceiling happens to be generous enough to admit the distance
}
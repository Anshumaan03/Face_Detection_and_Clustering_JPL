import os

DATA_ROOT = "/Users/anshumaansinghrathore/Desktop/Face Clustering/data/raw" 
OUTPUT_ROOT = "/Users/anshumaansinghrathore/Desktop/Face Clustering/results_comparison_pipeline" 
CLUSTERS_ROOT = os.path.join(OUTPUT_ROOT, "clusters")


ARCFACE_ONNX_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/Arcface.onnx"
DLIB_RESNET_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/Dlib_Resnet.dat"
DLIB_SHAPE_PREDICTOR_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/shape_predictor_68_face_landmarks.dat"  
# needed for dlib's own alignment
FACENET_VGGFACE2_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/facenet_vggface2.pt"
SIGLIP2_MODEL_NAME = "google/siglip2-base-patch16-224"


INSIGHTFACE_DET_MODEL = "buffalo_l"   
DETECTOR_CTX_ID = -1   # 0 = first GPU, -1 = CPU

MODEL_INPUT_SPECS = {
    "arcface": {
        "size": 112,
        "alignment": "insightface_norm_crop",   # 5-pt warp to ArcFace canonical template
    },
    "dlib_resnet": {
        "size": 150,
        "alignment": "dlib_chip",               # dlib.get_face_chip, needs 5-pt shape predictor
    },
    "facenet": {
        "size": 160,
        "alignment": "bbox_margin_crop",        # generous bbox crop + resize, no landmark warp
        "margin": 0.25,                          # fraction of bbox size added on each side
    },
    "siglip2": {
        "size": 224,
        "alignment": "bbox_margin_crop",
        "margin": 0.4,                           # SigLIP2 saw natural images -> wider context helps
    },
}

EMBEDDING_DIMS = {
    "arcface": 512,
    "dlib_resnet": 128,
    "facenet": 512,
    "siglip2": 768,   # base-patch16-224 -> 768; check your variant
}


MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Anshu@2003",
    "database": "face_db_new",
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
# Where user-uploaded images (Flow 1) get saved so they have a stable
# image_path to store in the `faces` table, same as dataset images.
UPLOADS_ROOT = os.path.join(OUTPUT_ROOT, "uploads")

# Distances below are COSINE distances (1 - cosine_similarity) between
# L2-normalized embeddings, so they live in [0, 2]; smaller = more similar.
#
# IMPORTANT: these numbers are NOT measured on your data — they're generic
# starting points, one per model, since each embedding space has its own
# natural scale of separation. Your dataset already carries ground-truth
# identity labels (from the folder structure), so before trusting these in
# the app, run recommendation.calibrate_thresholds(model, run_label) once
# per model and replace the values below with what it suggests.
THRESHOLDS = {
    "arcface":     {"t1": 0.30, "t2": 0.45},
    "dlib_resnet": {"t1": 0.25, "t2": 0.40},
    "facenet":     {"t1": 0.30, "t2": 0.50},
    "siglip2":     {"t1": 0.35, "t2": 0.55},  # general-purpose embedding, not face-specific -> expect looser separation
}
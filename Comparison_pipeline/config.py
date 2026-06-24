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
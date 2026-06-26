import os

DATA_ROOT = "/Users/anshumaansinghrathore/Desktop/Face Clustering/data/raw" 
OUTPUT_ROOT = "/Users/anshumaansinghrathore/Desktop/Face Clustering/results_comparison" 
CLUSTERS_ROOT = os.path.join(OUTPUT_ROOT, "clusters")

ARCFACE_ONNX_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/Arcface.onnx"
DLIB_RESNET_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/Dlib_Resnet.dat"
DLIB_SHAPE_PREDICTOR_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/shape_predictor_68_face_landmarks.dat"  
FACENET_VGGFACE2_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/facenet_vggface2.pt"
SIGLIP2_MODEL_NAME = "google/siglip2-base-patch16-224"

INSIGHTFACE_DET_MODEL = "buffalo_l"   
DETECTOR_CTX_ID = -1

MODEL_INPUT_SPECS = {
    "arcface": {"size": 112, "alignment": "insightface_norm_crop"},
    "dlib_resnet": {"size": 150, "alignment": "dlib_chip"},
    "facenet": {"size": 160, "alignment": "bbox_margin_crop", "margin": 0.25},
    "siglip2": {"size": 224, "alignment": "bbox_margin_crop", "margin": 0.4},
}

EMBEDDING_DIMS = {"arcface": 512, "dlib_resnet": 128, "facenet": 512, "siglip2": 768}

MYSQL_CONFIG = {
    "host": "localhost", "user": "root", "password": "Anshu@2003", "database": "face_db_new",
}

HDBSCAN_PARAMS = {
    "min_cluster_size": 10,
    "min_samples": 6,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
    "cluster_selection_epsilon": 0.08,
}

# Multi-face settings
FACE_DET_MIN_SCORE = 0.65
FACE_MIN_SIZE = 20
FACE_MIN_BLUR_VAR = 20
DETECTION_SIZE = (1024, 1024)   # Better for group photos
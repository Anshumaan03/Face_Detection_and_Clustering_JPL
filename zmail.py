import os
import cv2
import json
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import mysql.connector
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.pairwise import pairwise_distances

# =====================================================================
# ⚙️ SYSTEM SETTINGS AND PATH CONFIGURATIONS
# =====================================================================
ROOT_DIR = "/Users/anshumaansinghrathore/Desktop/Face Clustering"
DATASET_DIR = "/Users/anshumaansinghrathore/Desktop/Face Clustering/data/raw"
YOLO_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/yolov8n-pose.pt"
ARCFACE_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/Arcface.onnx"

DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "Anshu@2003",
    "database": "face_db_zmail"
}

print("🧠 [SYSTEM] Loading models into memory...")
yolo_model = YOLO(YOLO_PATH)
arcface_session = ort.InferenceSession(ARCFACE_PATH, providers=['CPUExecutionProvider'])

# =====================================================================
# 📐 GEOMETRIC EYE-ALIGNMENT CORE ENGINE
# =====================================================================
def get_affine_aligned_face(frame, kpts):
    """Computes spatial Similarity Transform Matrix using eye anchors to stabilize face roll."""
    p1, p2 = kpts[1][:2], kpts[2][:2]  # Left eye, Right eye landmarks
    screen_left, screen_right = (p1, p2) if p1[0] < p2[0] else (p2, p1)
    
    eye_center = ((screen_left[0] + screen_right[0]) * 0.5, (screen_left[1] + screen_right[1]) * 0.5)
    dx, dy = screen_right[0] - screen_left[0], screen_right[1] - screen_left[1]
    
    angle = np.degrees(np.arctan2(dy, dx))
    desired_eye_dist, current_eye_dist = 35.0, np.sqrt(dx**2 + dy**2)
    
    if current_eye_dist == 0: 
        return None
    scale = desired_eye_dist / current_eye_dist
    
    M = cv2.getRotationMatrix2D(eye_center, angle, scale)
    M[0, 2] += (56.0 - eye_center[0])
    M[1, 2] += (52.0 - eye_center[1])
    
    return cv2.warpAffine(frame, M, (112, 112), flags=cv2.INTER_CUBIC)

# =====================================================================
# ⚡ ARCFACE EMBEDDING EXTRACTION
# =====================================================================
def get_arcface_embedding(aligned_face):
    """Normalizes image array down to [-1, 1] manifold and outputs an L2 normalized 512D unit vector."""
    rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
    normalized = (chw - 127.5) / 128.0
    input_blob = np.expand_dims(normalized, axis=0)
    
    input_name = arcface_session.get_inputs()[0].name
    embedding = arcface_session.run(None, {input_name: input_blob})[0][0]
    
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding

# =====================================================================
# 📥 DATA INGESTION GATEWAY (WITH CONFIDENCE THRESHOLD FILTER)
# =====================================================================
def execute_complete_ingestion():
    """Wipes the target table and runs a strict ingestion loop using an explicit confidence threshold gate."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS face_embeddings (
        id INT AUTO_INCREMENT PRIMARY KEY,
        file_name VARCHAR(255) NOT NULL,
        identity_label VARCHAR(255) NOT NULL,
        embedding_json LONGTEXT NOT NULL,
        cluster_id INT DEFAULT -1
    );
    """)
    cursor.execute("TRUNCATE TABLE face_embeddings;")
    conn.commit()
    
    # 🎛️ CONFIDENCE GATEWAY FILTER THRESHOLD
    FACE_CONFIDENCE_THRESHOLD = 0.75
    
    print(f"🚀 [INGESTION] Dropping faces with prediction scores under {FACE_CONFIDENCE_THRESHOLD}...")
    vector_count = 0
    
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Target dataset folder directory not found at: {DATASET_DIR}")
        return

    for folder_name in sorted(os.listdir(DATASET_DIR)):
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if not os.path.isdir(folder_path): 
            continue
        
        print(f"📂 Scanning Folder: {folder_name}")
        for img_file in os.listdir(folder_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')): 
                continue
            img_path = os.path.join(folder_path, img_file)
            frame = cv2.imread(img_path)
            if frame is None: 
                continue
            
            results = yolo_model(frame, verbose=False)
            for result in results:
                if result.keypoints is None or len(result.keypoints.data) == 0: 
                    continue
                
                scores = result.boxes.conf.cpu().numpy()
                kpts_data = result.keypoints.data.cpu().numpy()
                
                # Loop through every localized face geometry found in the frame
                for idx, score in enumerate(scores):
                    # 🛑 Threshold gate check: bypass low-quality background anomalies
                    if score < FACE_CONFIDENCE_THRESHOLD:
                        continue
                    
                    kpts = kpts_data[idx]
                    aligned = get_affine_aligned_face(frame, kpts)
                    if aligned is None: 
                        continue
                    
                    embedding = get_arcface_embedding(aligned)
                    
                    cursor.execute("""
                        INSERT INTO face_embeddings (file_name, identity_label, embedding_json, cluster_id)
                        VALUES (%s, %s, %s, -1)
                    """, (img_file, folder_name, json.dumps(embedding.tolist())))
                    vector_count += 1
                    
        conn.commit()
    print(f"\n✅ Ingestion complete. Committed exactly {vector_count} high-confidence vectors to MySQL.")
    cursor.close()
    conn.close()

# =====================================================================
# 📊 HIERARCHICAL CLUSTERING & BENCHMARKING PIPELINE
# =====================================================================
def run_agglomerative_clustering_and_benchmarking(n_identities=8):
    """Uses fixed hierarchical constraint linkage to block cluster fragmentation and optimize index accuracy."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT id, identity_label, embedding_json FROM face_embeddings")
    rows = cursor.fetchall()
    
    if not rows:
        print("❌ Core pipeline database is empty. Execute data ingestion first.")
        return
        
    row_ids = [r[0] for r in rows]
    ground_truths = [r[1] for r in rows]
    X = np.array([json.loads(r[2]) for r in rows], dtype=np.float32)
    
    # 🎯 Force cluster trees to group directly into your exact folder count
    clusterer = AgglomerativeClustering(
        n_clusters=n_identities, 
        metric='cosine', 
        linkage='average'
    )
    labels = clusterer.fit_predict(X)
    
    # Push updated identity mapping identifiers back to your system database
    cursor.executemany(
        "UPDATE face_embeddings SET cluster_id = %s WHERE id = %s",
        [(int(l), int(rid)) for l, rid in zip(labels, row_ids)]
    )
    conn.commit()
    
    # Calculate performance validation criteria
    nmi = normalized_mutual_info_score(ground_truths, labels)
    ari = adjusted_rand_score(ground_truths, labels)
    
    print("\n============================================================")
    print("📈 PIPELINE PRODUCTION CONSTRAINED BENCHMARK REPORT")
    print("============================================================")
    print(f"🎯 Target Identity Groups Locked: {n_identities}")
    print(f"📊 Normalized Mutual Info (NMI) : {nmi:.4f}")
    print(f"📈 Adjusted Rand Index (ARI)    : {ari:.4f}")
    print("============================================================\n")
    
    cursor.close()
    conn.close()

# =====================================================================
# 🏁 PIPELINE INTERFACE ROOT ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    execute_ingestion_loop = input("Run clean data ingestion loop from scratch? (y/n): ")
    if execute_ingestion_loop.lower() == 'y':
        execute_complete_ingestion()
        
    # Directly match your 8 unique folder paths
    run_agglomerative_clustering_and_benchmarking(n_identities=8)
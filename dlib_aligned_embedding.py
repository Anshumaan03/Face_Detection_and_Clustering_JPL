import os
import cv2
import numpy as np
import dlib
from tqdm import tqdm

# 1. Define local file paths explicitly
MODEL_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/dlib_face_recognition_resnet_model_v1.dat"
LANDMARK_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/shape_predictor_68_face_landmarks.dat"

def extract_dlib_embeddings_native(aligned_folder, out_vec, out_lbl):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LANDMARK_PATH):
        print(f"[ERROR] Required .dat model files are missing in the project root!")
        return

    # 2. Load the models natively from disk paths
    pose_predictor = dlib.shape_predictor(LANDMARK_PATH)
    face_encoder = dlib.face_recognition_model_v1(MODEL_PATH)

    embeddings = []
    labels = []
    
    if not os.path.exists(aligned_folder):
        print(f"[ERROR] Dlib input folder missing at: {aligned_folder}")
        return

    valid_files = [f for f in os.listdir(aligned_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\n[DLIB NATIVE EXTRACTION] Processing: {aligned_folder}")
    
    for filename in tqdm(valid_files):
        img_path = os.path.join(aligned_folder, filename)
        bgr_img = cv2.imread(img_path)
        if bgr_img is None: continue
        
        # Convert BGR to RGB channel formatting
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        height, width = rgb_img.shape[:2]
        
        # Define a bounding box spanning the full cropped canvas area
        full_box = dlib.rectangle(0, 0, width, height)
        
        # Extract internal layout structural shapes
        shape = pose_predictor(rgb_img, full_box)
        
        # Compute the real 128D descriptor using your local .dat model file
        face_descriptor = face_encoder.compute_face_descriptor(rgb_img, shape)
        
        embeddings.append(np.array(face_descriptor))
        
        # Parse identity tokens from filename layout
        parts = filename.split('_')
        identity = "_".join(parts[:2]) if len(parts) >= 2 else "Unknown"
        labels.append(identity)
            
    if len(embeddings) > 0:
        os.makedirs(os.path.dirname(out_vec), exist_ok=True)
        np.save(out_vec, np.array(embeddings))
        np.save(out_lbl, np.array(labels))
        print(f"[SUCCESS] Saved Dlib vectors: {np.array(embeddings).shape}")

if __name__ == "__main__":
    extract_dlib_embeddings_native(
        aligned_folder="/Users/anshumaansinghrathore/Desktop/Face Clustering/data/aligned_dlib",
        out_vec="data/embeddings/dlib_vectors.npy",
        out_lbl="data/embeddings/dlib_labels.npy"
    )
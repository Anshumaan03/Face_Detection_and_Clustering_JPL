import os
import cv2
import numpy as np
from tqdm import tqdm

# 1. Path configuration to your pure OpenCV FaceNet model
OPENCV_EMBEDDER_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/openface.nn4.small2.v1.t7"

def extract_pure_opencv_embeddings(aligned_folder, out_vec, out_lbl):
    """
    Extracts 128D embeddings using OpenCV's native DNN engine
    from pre-aligned face crops. No landmark paths required here.
    """
    # Safety Check: Verify model weights exist inside models/ folder
    if not os.path.exists(OPENCV_EMBEDDER_PATH):
        print(f"[ERROR] Missing OpenCV embedder model at: '{OPENCV_EMBEDDER_PATH}'")
        print("Please check your file layout inside the 'models/' folder.")
        return

    # 2. Initialize the FaceNet model inside the built-in OpenCV DNN module
    print(f"[INFO] Loading OpenCV FaceNet Engine from {OPENCV_EMBEDDER_PATH}...")
    net = cv2.dnn.readNetFromTorch(OPENCV_EMBEDDER_PATH)

    embeddings = []
    labels = []
    
    # Safety Check: Verify input aligned image directory exists
    if not os.path.exists(aligned_folder):
        print(f"[ERROR] OpenCV aligned images folder missing at: {aligned_folder}")
        return

    valid_files = [f for f in os.listdir(aligned_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\n[OPENCV DNN EXTRACTION] Processing images from: {aligned_folder}")
    
    # 3. Stream each pre-aligned face crop through the network layers
    for filename in tqdm(valid_files):
        img_path = os.path.join(aligned_folder, filename)
        bgr_img = cv2.imread(img_path)
        if bgr_img is None: continue
        
        # 4. Standardize the image into an input "Blob"
        # FaceNet deep layers expect 96x96 pixels, RGB format, values normalized to [0, 1]
        blob = cv2.dnn.blobFromImage(
            bgr_img, 
            scalefactor=1.0 / 255, 
            size=(96, 96), 
            mean=(0, 0, 0), 
            swapRB=True, 
            crop=False
        )
        
        # 5. Execute forward propagation to get the 128D vector
        net.setInput(blob)
        face_descriptor = net.forward() 
        
        # Squeeze the output matrix down to a flat 1D array of 128 values
        embeddings.append(face_descriptor.flatten())
        
        # Parse the identity string from your filename convention (e.g. 'Gerry_Kelly_0001.png')
        parts = filename.split('_')
        identity = "_".join(parts[:2]) if len(parts) >= 2 else "Unknown"
        labels.append(identity)
            
    # 6. Export data arrays into the embeddings folder for your clustering/t-SNE scripts
    if len(embeddings) > 0:
        os.makedirs(os.path.dirname(out_vec), exist_ok=True)
        np.save(out_vec, np.array(embeddings))
        np.save(out_lbl, np.array(labels))
        print(f"[SUCCESS] Saved Pure OpenCV vectors matrix shape: {np.array(embeddings).shape} to {out_vec}")
    else:
        print("[WARNING] Zero embeddings extracted. Please check your source directory.")

if __name__ == "__main__":
    # ⚠️ Adjust this path to match your exact OpenCV aligned images directory ⚠️
    extract_pure_opencv_embeddings(
        aligned_folder="/Users/anshumaansinghrathore/Desktop/Face Clustering/data/processed_opencv", 
        out_vec="data/embeddings/opencv_vectors.npy",
        out_lbl="data/embeddings/opencv_labels.npy"
    )
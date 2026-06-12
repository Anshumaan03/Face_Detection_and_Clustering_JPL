import os
import cv2
import json
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import mysql.connector

DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "Anshu@2003",
    "database": "face_db"
}

DATASET_DIR = "/Users/anshumaansinghrathore/Desktop/Face Clustering/data/raw"
YOLO_PATH   = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/yolov8n-pose.pt"
ARCFACE_PATH= "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/w600k_r50.onnx"

print("🚀 INITIALIZING PIPELINE...")
yolo_model = YOLO(YOLO_PATH)
arcface_session = ort.InferenceSession(ARCFACE_PATH, providers=['CPUExecutionProvider'])
INPUT_NAME = arcface_session.get_inputs()[0].name  # fetch once, reuse

def get_affine_aligned_face(frame, kpts):
    # yolov8n-face keypoint order:
    # kpt[0] = left eye  ← was kpt[1] with body pose model
    # kpt[1] = right eye ← was kpt[2] with body pose model
    # kpt[2] = nose
    # kpt[3] = left mouth corner
    # kpt[4] = right mouth corner

    left_eye  = kpts[0][:2]   # ← CHANGED from kpts[1]
    right_eye = kpts[1][:2]   # ← CHANGED from kpts[2]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    current_eye_dist = np.sqrt(dx**2 + dy**2)

    if current_eye_dist < 5.0:
        return None

    angle = np.degrees(np.arctan2(dy, dx))
    desired_eye_dist = 35.0
    scale = desired_eye_dist / current_eye_dist

    eye_center = (
        (left_eye[0] + right_eye[0]) * 0.5,
        (left_eye[1] + right_eye[1]) * 0.5
    )

    M = cv2.getRotationMatrix2D(eye_center, angle, scale)
    M[0, 2] += (56.0 - eye_center[0])
    M[1, 2] += (52.0 - eye_center[1])

    aligned_face = cv2.warpAffine(frame, M, (112, 112), flags=cv2.INTER_CUBIC)
    return aligned_face

def extract_arcface_features(aligned_face):
    """
    CRITICAL: Keep BGR — do NOT convert to RGB.
    w600k_r50.onnx was trained on BGR channel order.
    """
    # ✅ NO cv2.cvtColor here — stay in BGR
    chw = np.transpose(aligned_face, (2, 0, 1)).astype(np.float32)
    normalized = (chw - 127.5) / 128.0
    input_blob = np.expand_dims(normalized, axis=0)
    input_blob = np.ascontiguousarray(input_blob)  # ← FIX: ensure memory layout

    outputs = arcface_session.run(None, {INPUT_NAME: input_blob})
    embedding = outputs[0][0]

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def get_best_face_keypoints(result):
    if result.keypoints is None or len(result.keypoints.data) == 0:
        return None

    boxes = result.boxes
    if boxes is None or len(boxes.conf) == 0:
        return None

    best_idx = int(boxes.conf.argmax())
    kpts = result.keypoints.data[best_idx].cpu().numpy()

    if len(kpts) < 5:
        return None

    # yolov8n-face: check eye visibility (index 0 and 1, not 1 and 2)
    if kpts[0][2] < 0.3 or kpts[1][2] < 0.3:
        return None

    return kpts



def run_pipeline():
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    print("🧹 Clearing previous records...")
    cursor.execute("TRUNCATE TABLE face_embeddings;")
    conn.commit()

    success_count = 0
    skip_count    = 0

    # FIX: sorted() ensures deterministic processing order
    for folder_name in sorted(os.listdir(DATASET_DIR)):
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        print(f"📁 Processing: {folder_name}")

        for img_file in sorted(os.listdir(folder_path)):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(folder_path, img_file)
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"   ⚠️  Could not read: {img_file}")
                continue

            results = yolo_model(frame, verbose=False)

            processed_this_image = False
            for result in results:
                kpts = get_best_face_keypoints(result)
                if kpts is None:
                    continue

                aligned_face = get_affine_aligned_face(frame, kpts)
                if aligned_face is None:
                    continue

                embedding = extract_arcface_features(aligned_face)
                embedding_json = json.dumps(embedding.tolist())

                cursor.execute(
                    "INSERT INTO face_embeddings (file_name, identity_label, embedding_json) VALUES (%s, %s, %s)",
                    (img_file, folder_name, embedding_json)
                )
                success_count += 1
                processed_this_image = True
                break  # one embedding per image file is enough

            if not processed_this_image:
                skip_count += 1
                print(f"   ⚠️  No valid face detected: {img_file}")

        conn.commit()
        print(f"   ↳ Done '{folder_name}'")

    cursor.close()
    conn.close()
    print(f"\n✅ Committed {success_count} vectors | Skipped {skip_count} images")

if __name__ == "__main__":
    run_pipeline()
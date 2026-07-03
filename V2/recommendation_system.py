import os
import cv2
import json
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import mysql.connector
from insightface.app import FaceAnalysis
import hdbscan

# ================================================================
# ⚙️  CONFIGURATION
# ================================================================
DB_CONFIG = {
    "host"    : "127.0.0.1",
    "user"    : "root",
    "password": "Anshu@2003",
    "database": "face_db_v2"
}

BASE         = "/Users/anshumaansinghrathore/Desktop/Face Clustering"
ARCFACE_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/Arcface.onnx"
YOLO_PATH    = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/yolov8n-face.pt"


# Cosine distance thresholds
T1_AUTO_MERGE = 0.30   # definitely same person
T2_PROMPT     = 0.55   # ambiguous — ask user
# d >= T2_PROMPT → new cluster

ARCFACE_SRC = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# ── Load models ──────────────────────────────────────────────────
print("Loading models ...")
yolo_model      = YOLO(YOLO_PATH)
arcface_session = ort.InferenceSession(
    ARCFACE_PATH, providers=["CPUExecutionProvider"]
)
INPUT_NAME = arcface_session.get_inputs()[0].name
retina_app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection"],
    providers=["CPUExecutionProvider"]
)
retina_app.prepare(ctx_id=-1, det_size=(640, 640))
print("✅ Models ready\n")


# ================================================================
# 🔧  UTILITIES (reused from pipeline_v2)
# ================================================================
def compute_iou(b1, b2):
    ix1   = max(b1[0], b2[0]);  iy1 = max(b1[1], b2[1])
    ix2   = min(b1[2], b2[2]);  iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1    = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2    = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def get_all_faces_ensemble(frame):
    faces = []
    for result in yolo_model(frame, verbose=False):
        if result.keypoints is None or len(result.keypoints.data) == 0:
            continue
        for i in range(len(result.boxes)):
            conf = float(result.boxes.conf[i].cpu().numpy())
            if conf < 0.35:
                continue
            kpts     = result.keypoints.data[i].cpu().numpy()
            if len(kpts) < 5:
                continue
            bbox     = result.boxes.xyxy[i].cpu().numpy()
            eye_dist = float(abs(kpts[1][0] - kpts[0][0]))
            faces.append({"bbox": bbox, "kpts": kpts,
                          "conf": conf, "eye_dist": eye_dist,
                          "source": "yolo"})
    for face in retina_app.get(frame):
        conf     = float(face.det_score)
        if conf < 0.35:
            continue
        kpts     = np.hstack([face.kps, np.ones((5, 1))])
        eye_dist = float(abs(face.kps[1][0] - face.kps[0][0]))
        faces.append({"bbox": face.bbox, "kpts": kpts,
                      "conf": conf, "eye_dist": eye_dist,
                      "source": "retina"})
    faces.sort(key=lambda x: x["conf"], reverse=True)
    kept = []
    for face in faces:
        if not any(compute_iou(face["bbox"], k["bbox"]) > 0.50 for k in kept):
            kept.append(face)
    return kept

def get_affine_aligned_face(frame, kpts):
    src_pts = kpts[:5, :2].astype(np.float32)
    M, _    = cv2.estimateAffinePartial2D(src_pts, ARCFACE_SRC, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(frame, M, (112, 112),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

def extract_arcface_embedding(aligned_face):
    chw        = np.transpose(aligned_face, (2, 0, 1)).astype(np.float32)
    normalized = (chw - 127.5) / 128.0
    blob       = np.ascontiguousarray(np.expand_dims(normalized, 0))
    raw        = arcface_session.run(None, {INPUT_NAME: blob})[0][0]
    norm       = np.linalg.norm(raw)
    return (raw / norm).astype(np.float32) if norm > 0 else raw

def spherical_centroid(vectors):
    c    = np.mean(vectors, axis=0)
    norm = np.linalg.norm(c)
    return c / norm if norm > 0 else c

def cosine_distance(a, b):
    return float(1.0 - np.dot(a, b))


# ================================================================
# 🔌  DATABASE HELPERS
# ================================================================
def load_existing_centroids():
    """Load all saved cluster centroids from DB."""
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT cluster_id, identity_label, centroid_json,
               member_count
        FROM   cluster_centroids
        ORDER  BY cluster_id
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    centroids = []
    for row in rows:
        centroids.append({
            "cluster_id"    : row["cluster_id"],
            "identity_label": row["identity_label"],
            "centroid"      : np.array(json.loads(row["centroid_json"]),
                                       dtype=np.float32),
            "member_count"  : row["member_count"]
        })
    return centroids

def update_centroid_after_merge(cluster_id, new_centroid,
                                new_member_count, conn):
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE cluster_centroids
        SET    centroid_json = %s,
               member_count  = %s
        WHERE  cluster_id   = %s
    """, (
        json.dumps(new_centroid.tolist()),
        new_member_count,
        cluster_id
    ))
    conn.commit()
    cursor.close()

def insert_new_centroid(identity_label, centroid, member_count, conn):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COALESCE(MAX(cluster_id), -1) + 1
        FROM   cluster_centroids
    """)
    new_id = cursor.fetchone()[0]
    cursor.execute("""
        INSERT INTO cluster_centroids
            (cluster_id, identity_label, centroid_json, member_count)
        VALUES (%s, %s, %s, %s)
    """, (new_id, identity_label, json.dumps(centroid.tolist()), member_count))
    conn.commit()
    cursor.close()
    return new_id

def log_decision(new_cid, matched_cid, dist, decision, conn):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO merge_decisions
            (new_cluster_id, matched_cluster_id,
             cosine_distance, decision)
        VALUES (%s, %s, %s, %s)
    """, (new_cid, matched_cid, dist, decision))
    conn.commit()
    cursor.close()


# ================================================================
# 🚀  RECOMMENDATION ENGINE
# ================================================================
def process_new_images(image_paths):
    """
    Given a list of new image paths (all assumed to be the same person):
    1. Extract all face embeddings
    2. Compute new cluster centroid
    3. Compare to existing centroids
    4. Route to T1/T2/T3 decision
    """
    print(f"\n{'='*60}")
    print(f"  RECOMMENDATION SYSTEM")
    print(f"{'='*60}")
    print(f"  Processing {len(image_paths)} new image(s) ...")

    # ── Extract embeddings from new images ────────────────────────
    new_embeddings = []
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  ⚠️  Cannot read: {img_path}")
            continue
        faces = get_all_faces_ensemble(frame)
        for face in faces:
            aligned = get_affine_aligned_face(frame, face["kpts"])
            if aligned is not None:
                emb = extract_arcface_embedding(aligned)
                new_embeddings.append(emb)

    if not new_embeddings:
        print("  ❌ No faces detected in uploaded images")
        return

    new_embeddings = np.array(new_embeddings, dtype=np.float32)
    print(f"  Extracted {len(new_embeddings)} face embedding(s)")

    # ── Compute centroid of new images ────────────────────────────
    new_centroid = spherical_centroid(new_embeddings)

    # ── Load existing centroids ───────────────────────────────────
    existing = load_existing_centroids()
    if not existing:
        print("  ⚠️  No existing centroids — register as new identity")
        _register_new(new_centroid, len(new_embeddings), None, None)
        return

    # ── Compute distances to all existing centroids ───────────────
    distances = []
    for ec in existing:
        d = cosine_distance(new_centroid, ec["centroid"])
        distances.append((d, ec))

    distances.sort(key=lambda x: x[0])
    best_dist, best_cluster = distances[0]

    print(f"\n  Closest existing cluster:")
    print(f"    Identity : {best_cluster['identity_label']}")
    print(f"    Distance : {best_dist:.4f}")
    print(f"    Members  : {best_cluster['member_count']}")
    print()

    # ── T1/T2/T3 decision ────────────────────────────────────────
    conn = mysql.connector.connect(**DB_CONFIG)

    if best_dist < T1_AUTO_MERGE:
        # ── T1 — Auto merge ───────────────────────────────────────
        print(f"  🟢 T1 AUTO-MERGE (d={best_dist:.4f} < {T1_AUTO_MERGE})")
        print(f"     Same person as: {best_cluster['identity_label']}")
        _merge(best_cluster, new_centroid,
               len(new_embeddings), best_dist, conn)

    elif best_dist < T2_PROMPT:
        # ── T2 — Prompt user ──────────────────────────────────────
        print(f"  🟡 T2 AMBIGUOUS (d={best_dist:.4f})")
        print(f"     Possibly same person as: {best_cluster['identity_label']}")
        print()

        # Show top 3 matches for context
        print("  Top 3 closest existing identities:")
        for i, (d, ec) in enumerate(distances[:3]):
            print(f"    {i+1}. {ec['identity_label']:<30} dist={d:.4f}")
        print()

        answer = input(
            f"  ❓ Is this the same person as "
            f"'{best_cluster['identity_label']}'? (y/n): "
        ).strip().lower()

        if answer == "y":
            print(f"  ✅ User confirmed — merging into "
                  f"'{best_cluster['identity_label']}'")
            _merge(best_cluster, new_centroid,
                   len(new_embeddings), best_dist, conn,
                   decision="user_confirmed")
        else:
            name = input(
                "  🆕 Enter identity name for new cluster: "
            ).strip()
            print(f"  🆕 Registering as new identity: '{name}'")
            _register_new(new_centroid, len(new_embeddings),
                          best_dist, conn, name,
                          matched_id=best_cluster["cluster_id"])

    else:
        # ── T3 — New cluster ──────────────────────────────────────
        print(f"  🔴 T3 NEW CLUSTER (d={best_dist:.4f} >= {T2_PROMPT})")
        print(f"     Distance too large — definitely different person")
        name = input("  🆕 Enter identity name: ").strip()
        _register_new(new_centroid, len(new_embeddings),
                      best_dist, conn, name,
                      matched_id=best_cluster["cluster_id"])

    conn.close()
    print(f"\n  ✅ Decision recorded in merge_decisions table")
    print(f"{'='*60}")


def _merge(best_cluster, new_centroid, n_new, dist, conn,
           decision="auto_merge"):
    """Weighted spherical merge of new centroid into existing cluster."""
    n_old     = best_cluster["member_count"]
    old_c     = best_cluster["centroid"]

    # Weighted average — more members = more influence
    merged    = (n_old * old_c + n_new * new_centroid)
    norm      = np.linalg.norm(merged)
    merged    = merged / norm if norm > 0 else merged

    update_centroid_after_merge(
        best_cluster["cluster_id"],
        merged,
        n_old + n_new,
        conn
    )
    log_decision(None, best_cluster["cluster_id"],
                 dist, decision, conn)

def _register_new(centroid, n, dist, conn, name="unknown",
                  matched_id=None):
    if conn is None:
        conn = mysql.connector.connect(**DB_CONFIG)
    new_id = insert_new_centroid(name, centroid, n, conn)
    if dist is not None:
        log_decision(new_id, matched_id, dist, "new_cluster", conn)
    print(f"  🆕 New cluster ID {new_id} created for '{name}'")


# ================================================================
# 🎯  ENTRY POINT
# ================================================================
if __name__ == "__main__":
    print("\nFace Clustering — Recommendation System")
    print("----------------------------------------")
    print("Enter paths to new images (comma-separated),")
    print("or press Enter to use a test example:")
    print()

    raw = input("Image paths: ").strip()

    if not raw:
        # Demo — use first image of first identity as test
        BASE_DATA = "/Users/anshumaansinghrathore/Desktop/Face Clustering/data/raw"
        first_id  = sorted([
            d for d in os.listdir(BASE_DATA)
            if os.path.isdir(os.path.join(BASE_DATA, d))
        ])[0]
        first_img = sorted([
            f for f in os.listdir(os.path.join(BASE_DATA, first_id))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])[0]
        paths = [os.path.join(BASE_DATA, first_id, first_img)]
        print(f"\n  Using demo image: {paths[0]}")
    else:
        paths = [p.strip() for p in raw.split(",")]

    process_new_images(paths)
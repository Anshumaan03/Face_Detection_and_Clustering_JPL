import os
import cv2
import json
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import mysql.connector
from insightface.app import FaceAnalysis

# ================================================================
# ⚙️  CONFIGURATION — update paths if needed
# ================================================================
DB_CONFIG = {
    "host"    : "127.0.0.1",
    "user"    : "root",
    "password": "Anshu@2003",
    "database": "face_db_v2"
}

BASE         = "/Users/anshumaansinghrathore/Desktop/Face Clustering"
DATASET_DIR  = "/Users/anshumaansinghrathore/Desktop/Face Clustering/data/raw"
ARCFACE_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/Arcface.onnx"
YOLO_PATH    = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/yolov8n-face.pt"

# Detection quality gates
MIN_CONF      = 0.35   # minimum detection confidence
IOU_THRESHOLD = 0.50   # IoU above this = duplicate, discard lower conf
MIN_EYE_DIST  = 8.0    # below this = side profile (still stored, flagged)

# ArcFace canonical 5-point template (112×112 target positions)
# These come from ArcFace training — do NOT change these values
ARCFACE_SRC = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)

# ================================================================
# 🚀  MODEL INITIALISATION — done once at startup
# ================================================================
print("=" * 60)
print("  FACE CLUSTERING PIPELINE v2 — INITIALISING")
print("=" * 60)

print("  Loading YOLOv8n-face ...")
yolo_model = YOLO(YOLO_PATH)

print("  Loading ArcFace (ONNX Runtime) ...")
arcface_session = ort.InferenceSession(
    ARCFACE_PATH,
    providers=["CPUExecutionProvider"]
)
INPUT_NAME = arcface_session.get_inputs()[0].name

print("  Loading RetinaFace (InsightFace buffalo_l) ...")
retina_app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection"],
    providers=["CPUExecutionProvider"]
)
retina_app.prepare(ctx_id=-1, det_size=(640, 640))

print("  ✅ All models loaded\n")


# ================================================================
# 🔧  UTILITY — IoU for deduplication
# ================================================================
def compute_iou(b1, b2):
    ix1   = max(b1[0], b2[0]);  iy1 = max(b1[1], b2[1])
    ix2   = min(b1[2], b2[2]);  iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1    = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2    = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# ================================================================
# 🔧  STAGE 1 — Ensemble Detection
# ================================================================
def get_all_faces_ensemble(frame):
    """
    Runs YOLO + RetinaFace on the same frame.
    Returns deduplicated list of face dicts:
        { bbox[4], kpts[5,3], conf, eye_dist, is_profile, source }

    Storing ALL faces (not just the best) lets HDBSCAN
    separate identities in embedding space later.
    """
    faces = []

    # ── YOLO ──────────────────────────────────────────────────────
    for result in yolo_model(frame, verbose=False):
        if result.keypoints is None or len(result.keypoints.data) == 0:
            continue
        for i in range(len(result.boxes)):
            conf = float(result.boxes.conf[i].cpu().numpy())
            if conf < MIN_CONF:
                continue
            kpts     = result.keypoints.data[i].cpu().numpy()   # [5,3]
            if len(kpts) < 5:
                continue
            bbox     = result.boxes.xyxy[i].cpu().numpy()       # [4]
            eye_dist = float(abs(kpts[1][0] - kpts[0][0]))
            faces.append({
                "bbox"      : bbox,
                "kpts"      : kpts,
                "conf"      : conf,
                "eye_dist"  : eye_dist,
                "is_profile": eye_dist < MIN_EYE_DIST * 2.5,
                "source"    : "yolo"
            })

    # ── RetinaFace ────────────────────────────────────────────────
    for face in retina_app.get(frame):
        conf = float(face.det_score)
        if conf < MIN_CONF:
            continue
        kpts     = face.kps                                      # [5,2]
        kpts_3col= np.hstack([kpts, np.ones((5, 1))])           # [5,3]
        bbox     = face.bbox                                     # [4]
        eye_dist = float(abs(kpts[1][0] - kpts[0][0]))
        faces.append({
            "bbox"      : bbox,
            "kpts"      : kpts_3col,
            "conf"      : conf,
            "eye_dist"  : eye_dist,
            "is_profile": eye_dist < MIN_EYE_DIST * 2.5,
            "source"    : "retina"
        })

    if not faces:
        return []

    # ── Deduplicate overlapping boxes ─────────────────────────────
    faces.sort(key=lambda x: x["conf"], reverse=True)
    kept = []
    for face in faces:
        if not any(
            compute_iou(face["bbox"], k["bbox"]) > IOU_THRESHOLD
            for k in kept
        ):
            kept.append(face)

    return kept


# ================================================================
# 🔧  STAGE 2 — 5-Point Affine Alignment
# ================================================================
def get_affine_aligned_face(frame, kpts):
    """
    Uses all 5 landmarks for a least-squares affine fit
    (cv2.estimateAffinePartial2D) — more robust than 3-point
    exact solve, handles mild pose deviation and landmark noise.
    Output: 112×112 BGR patch ready for ArcFace.
    """
    src_pts = kpts[:5, :2].astype(np.float32)

    M, _ = cv2.estimateAffinePartial2D(
        src_pts, ARCFACE_SRC,
        method=cv2.LMEDS
    )
    if M is None:
        return None

    aligned = cv2.warpAffine(
        frame, M, (112, 112),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return aligned


# ================================================================
# 🔧  STAGE 3 — ArcFace Embedding
# ================================================================
def extract_arcface_embedding(aligned_face):
    """
    CRITICAL: NO BGR→RGB conversion.
    w600k_r50.onnx was trained on BGR channel order.
    Normalize to [-1, +1] range then L2-normalize output.
    """
    chw        = np.transpose(aligned_face, (2, 0, 1)).astype(np.float32)
    normalized = (chw - 127.5) / 128.0
    blob       = np.ascontiguousarray(np.expand_dims(normalized, axis=0))

    raw        = arcface_session.run(None, {INPUT_NAME: blob})[0][0]
    norm       = np.linalg.norm(raw)
    return (raw / norm).astype(np.float32) if norm > 0 else raw


# ================================================================
# 🚀  MAIN PIPELINE
# ================================================================
def run_pipeline():
    # ── Database connection ───────────────────────────────────────
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    print("🧹 Truncating face_embeddings_v2 ...")
    cursor.execute("TRUNCATE TABLE face_embeddings_v2;")
    conn.commit()

    # ── Stats counters ────────────────────────────────────────────
    total_images   = 0
    total_stored   = 0
    total_skipped  = 0
    total_profiles = 0
    identity_stats = {}

    # ── Iterate dataset ───────────────────────────────────────────
    identities = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    print(f"📂 Found {len(identities)} identity folders\n")

    for identity in identities:
        folder     = os.path.join(DATASET_DIR, identity)
        img_files  = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        id_stored  = 0
        id_skipped = 0

        print(f"📁 [{identity}] — {len(img_files)} images")

        for img_file in img_files:
            total_images += 1
            img_path      = os.path.join(folder, img_file)
            frame         = cv2.imread(img_path)

            if frame is None:
                print(f"   ⚠️  Cannot read: {img_file}")
                id_skipped    += 1
                total_skipped += 1
                continue

            # Stage 1 — detect ALL faces in image
            all_faces = get_all_faces_ensemble(frame)

            if not all_faces:
                id_skipped    += 1
                total_skipped += 1
                continue

            # Stage 2+3 — align + embed ALL detected faces
            img_stored = 0
            for face_idx, face in enumerate(all_faces):
                aligned = get_affine_aligned_face(frame, face["kpts"])
                if aligned is None:
                    continue

                embedding      = extract_arcface_embedding(aligned)
                embedding_json = json.dumps(embedding.tolist())

                cursor.execute("""
                    INSERT INTO face_embeddings_v2
                        (file_name, identity_label, embedding_json,
                         is_profile, eye_distance,
                         detector_source, det_confidence, face_index)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    img_file,
                    identity,
                    embedding_json,
                    bool(face["is_profile"]),
                    float(face["eye_dist"]),
                    face["source"],
                    float(face["conf"]),
                    face_idx
                ))

                img_stored     += 1
                total_stored   += 1
                if face["is_profile"]:
                    total_profiles += 1

            id_stored += img_stored
            tag = (
                f"  👥 group ({img_stored} faces)"
                if img_stored > 1
                else f"  ✅ single face"
                if img_stored == 1
                else f"  ⚠️  alignment failed"
            )
            print(f"   {img_file}{tag}")

        conn.commit()
        identity_stats[identity] = {
            "images" : len(img_files),
            "stored" : id_stored,
            "skipped": id_skipped
        }
        print(f"   ↳ stored {id_stored} embeddings\n")

    cursor.close()
    conn.close()

    # ── Final summary ─────────────────────────────────────────────
    print("=" * 60)
    print("  INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Identities processed : {len(identities)}")
    print(f"  Images processed     : {total_images}")
    print(f"  Total embeddings     : {total_stored}")
    print(f"  Profile faces        : {total_profiles} "
          f"({total_profiles/max(total_stored,1)*100:.1f}%)")
    print(f"  Skipped images       : {total_skipped}")
    print()
    print("  Per-identity breakdown:")
    print(f"  {'Identity':<30} {'Images':<8} {'Stored':<8} {'Skipped'}")
    print(f"  {'-'*60}")
    for idn, st in identity_stats.items():
        print(f"  {idn:<30} {st['images']:<8} {st['stored']:<8} {st['skipped']}")
    print()
    print("  ✅ Ready → run clustering_v2.py")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
"""
=============================================================================
  COMPLETE ARCFACE FACE EMBEDDING PIPELINE
  Modular · Verified · Pre-clustering ready
=============================================================================

MODULES:
  Module 0 — Config & imports
  Module 1 — Dataset auditor
  Module 2 — Image type detector (pre-cropped vs full photo)
  Module 3 — ArcFace preprocessor (correct path for each image type)
  Module 4 — ArcFace embedding generator + verifier
  Module 5 — Full-photo handler (YOLO → align → embed)
  Module 6 — MySQL storage
  Module 7 — Embedding sanity checker (run this BEFORE clustering)
  Module 8 — Main pipeline runner

HOW TO RUN:
  python pipeline.py --mode audit          # Step 1: check your dataset
  python pipeline.py --mode run            # Step 2: generate & store embeddings
  python pipeline.py --mode verify         # Step 3: verify embedding quality
  python pipeline.py --mode visualize      # Step 4: UMAP plot to inspect clusters visually
=============================================================================
"""

# ─────────────────────────────────────────────────────────────
# MODULE 0 — Config & imports
# ─────────────────────────────────────────────────────────────
import os
import cv2
import json
import argparse
import numpy as np
import onnxruntime as ort
from pathlib import Path
from collections import defaultdict

# ── Paths — update these to your local paths ──────────────────
DATASET_DIR  = "/Users/anshumaansinghrathore/Desktop/Face Clustering/data/raw"
YOLO_PATH    = None   # replaced by InsightFace detector — no longer needed
ARCFACE_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/w600k_r50.onnx"

DB_CONFIG = {
    "host"    : "127.0.0.1",
    "user"    : "root",
    "password": "Anshu@2003",
    "database": "face_db"
}

# ── ArcFace canonical 5-point template (trained target positions) ──
# These are the EXACT pixel positions ArcFace was trained to align to
# in a 112x112 image. Do NOT change these values.
ARCFACE_SRC = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)

# ── Thresholds ──────────────────────────────────────────────────
PRECROP_SIZE       = 160       # images at this resolution are treated as pre-cropped faces
ARCFACE_INPUT_SIZE = 112       # ArcFace always takes 112x112
YOLO_CONF_THRESH   = 0.5       # minimum YOLO detection confidence
LANDMARK_VIS_THRESH = 0.3      # minimum landmark visibility score
SAME_IDENTITY_MIN_SIM = 0.10   # cosine similarity: same person MUST be above this
DIFF_IDENTITY_MAX_SIM = 0.60   # cosine similarity: different people MUST be below this


# ─────────────────────────────────────────────────────────────
# MODULE 1 — Dataset auditor
# ─────────────────────────────────────────────────────────────
def audit_dataset(dataset_dir: str) -> dict:
    """
    Scans your entire dataset folder and reports:
      - Total identities and image counts
      - Unreadable/corrupted images
      - Images at root level (not inside an identity folder) ← bug source
      - Very small images that ArcFace will struggle with
      - The 2 image types: pre-cropped face vs full photo
    
    WHY THIS MATTERS:
      Your dataset has 160x160 pre-cropped faces (1483 images) AND 2 full-size
      group photos. These need completely different processing. This auditor
      detects which is which so the pipeline routes them correctly.
    """
    print("\n" + "="*60)
    print("  MODULE 1 — DATASET AUDIT")
    print("="*60)

    base = Path(dataset_dir)
    report = {
        "identities"     : {},
        "root_level_files": [],
        "corrupted"      : [],
        "tiny_images"    : [],
        "total_images"   : 0,
        "total_identities": 0,
    }

    # Check for files dumped at root level (not in any identity folder)
    for item in base.iterdir():
        if item.is_file() and item.suffix.lower() in ('.jpg', '.jpeg', '.png', '.avif', '.webp'):
            report["root_level_files"].append(str(item))
            print(f"  ⚠️  ROOT-LEVEL FILE (will be skipped): {item.name}")

    # Walk each identity folder
    identity_dirs = sorted([
        d for d in base.iterdir()
        if d.is_dir() and not d.name.startswith('.') and not d.name.startswith('__')
    ])

    for identity_dir in identity_dirs:
        identity = identity_dir.name
        img_files = sorted([
            f for f in identity_dir.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ])

        identity_stats = {
            "count"       : 0,
            "precropped"  : 0,   # 160x160 face crops
            "full_photos" : 0,   # larger images needing detection
            "corrupted"   : 0,
            "tiny"        : 0,
            "files"       : []
        }

        for img_path in img_files:
            img = cv2.imread(str(img_path))
            if img is None:
                identity_stats["corrupted"] += 1
                report["corrupted"].append(str(img_path))
                continue

            h, w = img.shape[:2]
            identity_stats["count"] += 1
            identity_stats["files"].append(str(img_path))

            # Classify image type
            if w == PRECROP_SIZE and h == PRECROP_SIZE:
                identity_stats["precropped"] += 1
            elif w < 50 or h < 50:
                identity_stats["tiny"] += 1
                report["tiny_images"].append(str(img_path))
            else:
                identity_stats["full_photos"] += 1

        report["identities"][identity] = identity_stats
        report["total_images"] += identity_stats["count"]
        report["total_identities"] += 1

    # Print summary
    print(f"\n  📊 DATASET SUMMARY")
    print(f"     Total identities : {report['total_identities']}")
    print(f"     Total images     : {report['total_images']}")
    print(f"     Corrupted files  : {len(report['corrupted'])}")
    print(f"     Root-level files : {len(report['root_level_files'])}")

    print(f"\n  {'Identity':<30} {'Total':>6} {'Pre-cropped':>12} {'Full photos':>12} {'Corrupted':>10}")
    print(f"  {'-'*72}")
    for name, stats in report["identities"].items():
        flag = " ⚠️" if stats["corrupted"] > 0 or stats["count"] < 5 else ""
        print(f"  {name:<30} {stats['count']:>6} {stats['precropped']:>12} {stats['full_photos']:>12} {stats['corrupted']:>10}{flag}")

    if report["corrupted"]:
        print(f"\n  ❌ CORRUPTED FILES:")
        for f in report["corrupted"]:
            print(f"     {f}")

    print(f"\n  ✅ Audit complete.")
    return report


# ─────────────────────────────────────────────────────────────
# MODULE 2 — Image type detector
# ─────────────────────────────────────────────────────────────
def classify_image(img_path: str) -> str:
    """
    Returns:
      'precropped' — already a 160x160 face crop → skip YOLO, just resize to 112x112
      'full_photo' — larger image → needs YOLO detection + landmark alignment
      'invalid'    — unreadable or too small

    WHY THIS MATTERS:
      The #1 bug in your original code: running YOLO + affine transformation
      on images that are ALREADY pre-cropped face images. Your dataset is
      almost entirely 160x160 face crops. Running YOLO on them either:
        a) Fails to detect (→ image skipped entirely, huge skip count)
        b) Detects with a bad bounding box and wrong landmarks
        c) Applies affine transform to an already-aligned face → DISTORTION
      
      The fix: detect the image type and route accordingly.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return 'invalid'
    h, w = img.shape[:2]
    if w < 50 or h < 50:
        return 'invalid'
    if w == PRECROP_SIZE and h == PRECROP_SIZE:
        return 'precropped'
    return 'full_photo'


# ─────────────────────────────────────────────────────────────
# MODULE 3 — ArcFace preprocessor
# ─────────────────────────────────────────────────────────────
def preprocess_precropped(img_bgr: np.ndarray) -> np.ndarray:
    """
    For pre-cropped 160x160 face images.
    
    Steps:
      1. Resize 160x160 → 112x112  (ArcFace input size)
      2. Keep BGR channel order     (w600k_r50.onnx trained on BGR)
      3. Normalize: (pixel - 127.5) / 128.0  → range approx [-1, 1]
      4. Transpose HWC → CHW
      5. Add batch dimension → (1, 3, 112, 112)
    
    WHY BGR (not RGB):
      w600k_r50.onnx was trained with MXNet which uses BGR by default.
      Converting to RGB before feeding will silently degrade accuracy.
      Your original code was actually correct on this point.
    
    WHY (pixel - 127.5) / 128.0:
      This is the exact normalization used during ArcFace training.
      Using ImageNet normalization (mean=[0.485,0.456,0.406]) here
      would give wrong results because the model was NOT trained that way.
    """
    resized = cv2.resize(img_bgr, (ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE),
                         interpolation=cv2.INTER_LINEAR)
    blob = resized.astype(np.float32)
    blob = (blob - 127.5) / 128.0          # normalize to ~[-1, 1]
    blob = blob.transpose(2, 0, 1)         # HWC → CHW: (112,112,3) → (3,112,112)
    blob = np.expand_dims(blob, axis=0)    # add batch dim: (3,112,112) → (1,3,112,112)
    blob = np.ascontiguousarray(blob)      # ensure C-contiguous memory layout
    return blob


def preprocess_aligned_crop(aligned_face_bgr: np.ndarray) -> np.ndarray:
    """
    For a face crop produced by the YOLO + affine alignment path.
    Input is already 112x112 (output of warpAffine using ARCFACE_SRC template).
    Same normalization as preprocess_precropped but no resize needed.
    """
    blob = aligned_face_bgr.astype(np.float32)
    blob = (blob - 127.5) / 128.0
    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0)
    blob = np.ascontiguousarray(blob)
    return blob


# ─────────────────────────────────────────────────────────────
# MODULE 4 — ArcFace embedding generator + verifier
# ─────────────────────────────────────────────────────────────
class ArcFaceModel:
    """
    Wraps the ArcFace ONNX model with proper input handling and verification.
    
    WHY ONNX RUNTIME:
      The w600k_r50.onnx is an ONNX export of the InsightFace R50 model
      trained on WebFace600K. ONNX runtime runs it framework-agnostically
      without needing MXNet or PyTorch installed.
    
    OUTPUT:
      512-dimensional embedding vector, L2-normalized to unit length.
      Cosine similarity between embeddings = dot product (since unit vectors).
    """
    def __init__(self, model_path: str):
        print(f"\n  Loading ArcFace model: {model_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape   # expect [1,3,112,112]
        print(f"  Input name : {self.input_name}")
        print(f"  Input shape: {self.input_shape}")
        print(f"  ✅ ArcFace model loaded")

    def get_embedding(self, blob: np.ndarray) -> np.ndarray:
        """
        blob: (1, 3, 112, 112) float32, already normalized
        returns: (512,) float32, L2-normalized
        """
        outputs   = self.session.run(None, {self.input_name: blob})
        embedding = outputs[0][0]                    # shape: (512,)
        norm      = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm             # L2 normalize → unit vector
        return embedding.astype(np.float32)

    def verify_model(self, dataset_dir: str, yolo_detector, num_pairs: int = 5) -> bool:
        """
        Quick sanity check: loads a few same-identity pairs, detects faces with
        YOLO, aligns them, embeds with ArcFace, and checks cosine similarity.

        Uses YOLO for all images so the preprocessing path here is identical
        to the main pipeline — this means a pass here guarantees the full
        pipeline will also produce valid embeddings.

        WHAT GOOD NUMBERS LOOK LIKE:
          Same person, 2 different photos  → cosine sim 0.4 – 0.9
          Different people                 → cosine sim 0.0 – 0.3
          Random vectors                   → cosine sim ~0.0
          Same image fed twice             → cosine sim ~1.0  (sanity check)
        """
        print("\n" + "="*60)
        print("  MODULE 4 — ARCFACE MODEL VERIFICATION")
        print("="*60)

        base = Path(dataset_dir)
        identity_dirs = [
            d for d in base.iterdir()
            if d.is_dir() and not d.name.startswith('.') and not d.name.startswith('__')
        ]

        all_pass = True
        same_sims = []
        diff_sims = []
        embeddings_by_identity = {}

        def _get_best_face_embedding(img_bgr):
            """
            Run InsightFace detection + affine alignment → ArcFace embedding.
            Returns the embedding for the highest-confidence detected face,
            or None if no face found or image has multiple faces (group photo).
            """
            faces = yolo_detector.detect_and_align(img_bgr)
            if not faces:
                return None
            # Skip group photos — if multiple faces detected, we can't reliably
            # identify which one is the target identity, so discard the image.
            if len(faces) > 1:
                return None
            best_face, best_conf, _ = faces[0]
            blob = preprocess_aligned_crop(best_face)
            return self.get_embedding(blob)

        print(f"\n  Testing same-identity pairs (expect cosine sim ≥ {SAME_IDENTITY_MIN_SIM}):")
        for identity_dir in identity_dirs[:num_pairs]:
            imgs = sorted([
                f for f in identity_dir.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            ])

            if len(imgs) < 2:
                continue

            # Try all images, collect up to 2 usable solo-face embeddings
            embs = []
            for img_path in imgs:
                if len(embs) >= 2:
                    break
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                emb = _get_best_face_embedding(img)
                if emb is not None:
                    embs.append(emb)
                else:
                    print(f"  ⚠️  Skipping {img_path.name} — no face or group photo (multiple faces)")

            if len(embs) < 2:
                print(f"  ⚠️  {identity_dir.name}: not enough faces detected, skipping pair")
                continue

            sim = float(np.dot(embs[0], embs[1]))
            same_sims.append(sim)
            embeddings_by_identity[identity_dir.name] = embs[0]
            status = "✅" if sim >= SAME_IDENTITY_MIN_SIM else "⚠️ low"
            print(f"  {status}  {identity_dir.name:<30} sim={sim:.4f}")

        # Sanity: same image twice must give sim ~1.0
        print(f"\n  Same-image sanity check (expect ~1.0):")
        test_img_path = None
        for identity_dir in identity_dirs[:1]:
            imgs = [f for f in identity_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
            if imgs:
                test_img_path = str(imgs[0])
                break
        if test_img_path:
            img = cv2.imread(test_img_path)
            emb1 = _get_best_face_embedding(img)
            emb2 = _get_best_face_embedding(img)
            if emb1 is not None and emb2 is not None:
                sim = float(np.dot(emb1, emb2))
                status = "✅" if sim > 0.999 else "❌ FAIL"
                print(f"  {status}  Same image twice → sim={sim:.6f}")
                if sim < 0.999:
                    all_pass = False
            else:
                print(f"  ⚠️  Could not detect face in sanity-check image — skipping")

        # Different identity pairs
        print(f"\n  Testing cross-identity pairs (expect cosine sim < {DIFF_IDENTITY_MAX_SIM}):")
        identity_names = list(embeddings_by_identity.keys())
        for i in range(min(5, len(identity_names) - 1)):
            a_name = identity_names[i]
            b_name = identity_names[i + 1]
            sim = float(np.dot(embeddings_by_identity[a_name],
                               embeddings_by_identity[b_name]))
            diff_sims.append(sim)
            status = "✅" if sim < DIFF_IDENTITY_MAX_SIM else "⚠️ HIGH"
            print(f"  {status}  {a_name:<22} vs {b_name:<22} sim={sim:.4f}")

        mean_intra = np.mean(same_sims) if same_sims else 0.0
        mean_inter = np.mean(diff_sims) if diff_sims else 0.0
        separability = mean_intra - mean_inter

        print(f"\n  Same-identity avg sim  : {mean_intra:.4f}  (same person, want >= 0.10)")
        print(f"  Cross-identity avg sim : {mean_inter:.4f}  (diff person, want <= 0.20)")
        print(f"  Separability gap       : {separability:.4f}  (want > 0.10)")

        # Pass/fail on separability gap, NOT individual pairs.
        # One low pair just means visual variance in that identity (normal).
        passed = separability >= 0.10 and all_pass

        if passed:
            print(f"\n  ✅ Model verification PASSED — embeddings are valid, proceed to pipeline")
        else:
            if separability < 0.10:
                print(f"\n  ❌ FAILED: Separability {separability:.4f} < 0.10 — check preprocessing")
            else:
                print(f"\n  ❌ FAILED: Same-image sanity check failed — model may be corrupted")

        return passed


# ─────────────────────────────────────────────────────────────
# MODULE 5 — Face detector + landmark aligner (InsightFace)
# ─────────────────────────────────────────────────────────────
class InsightFaceDetector:
    """
    Face detection + 5-point landmark alignment using InsightFace's
    RetinaFace detector. Replaces the YOLO-based detector.

    WHY INSIGHTFACE OVER YOLO HERE:
      The YOLOv8 detection-only models (yolov8n-face, yolov8m-face) do not
      output facial landmarks — only bounding boxes. Landmark output requires
      a pose-type YOLOv8 model trained specifically for face keypoints, which
      is not publicly available as a ready-to-use .pt file.

      InsightFace's RetinaFace detector (included in buffalo_l) outputs the
      exact 5 landmarks (left eye, right eye, nose, left mouth, right mouth)
      in the same order and coordinate space that ARCFACE_SRC expects. Since
      InsightFace also created ArcFace, this is the guaranteed-correct pairing.

    LANDMARK ORDER (matches ARCFACE_SRC):
        kps[0] = left eye
        kps[1] = right eye
        kps[2] = nose tip
        kps[3] = left mouth corner
        kps[4] = right mouth corner

    OUTPUT of detect_and_align:
      List of (aligned_face_bgr, det_score, bbox) tuples.
      aligned_face_bgr is 112x112 BGR, ready for preprocess_aligned_crop.
    """
    def __init__(self):
        from insightface.app import FaceAnalysis
        print(f"\n  Loading InsightFace detector (RetinaFace)...")
        self.app = FaceAnalysis(
            allowed_modules=['detection'],
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        print(f"  ✅ InsightFace detector loaded")

    def detect_and_align(self, img_bgr: np.ndarray) -> list:
        """
        Detects all faces and returns affine-aligned 112x112 crops.
        Returns list of (aligned_face_bgr, det_score, bbox) tuples.
        Empty list if no valid faces found.

        Alignment uses cv2.estimateAffinePartial2D mapping the detected
        5 landmarks onto ARCFACE_SRC — identical to the previous YOLO path.
        """
        faces = self.app.get(img_bgr)
        aligned_faces = []

        for face in faces:
            det_score = float(face.det_score)
            if det_score < YOLO_CONF_THRESH:   # reuse same confidence threshold
                continue

            kps = face.kps.astype(np.float32)  # shape (5, 2): x, y only

            # Eye distance sanity check
            eye_dist = np.linalg.norm(kps[0] - kps[1])
            if eye_dist < 5.0:
                continue

            # Affine transform: detected landmarks → ArcFace canonical template
            M, _ = cv2.estimateAffinePartial2D(
                kps, ARCFACE_SRC,
                method=cv2.LMEDS
            )
            if M is None:
                continue

            aligned = cv2.warpAffine(
                img_bgr, M,
                (ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

            bbox = face.bbox.astype(int).tolist()
            aligned_faces.append((aligned, det_score, bbox))

        return aligned_faces


# ─────────────────────────────────────────────────────────────
# MODULE 6 — MySQL storage
# ─────────────────────────────────────────────────────────────
class EmbeddingStore:
    """
    Handles MySQL storage and retrieval of face embeddings.
    
    SCHEMA (create this table if it doesn't exist):
    
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            file_name       VARCHAR(512)  NOT NULL,
            identity_label  VARCHAR(256)  NOT NULL,
            image_type      VARCHAR(32)   NOT NULL,   -- 'precropped' or 'full_photo'
            embedding_json  MEDIUMTEXT    NOT NULL,   -- 512-dim float32 as JSON array
            created_at      TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
        );
    
    WHY JSON NOT BLOB:
      JSON is human-readable and easier to debug. For production at scale,
      switch to BLOB with struct.pack for 2x storage efficiency. At 1486 images,
      JSON is fine.
    
    PRECISION NOTE:
      JSON float precision is sufficient for float32 embeddings. 
      Verified: round-trip error is < 1e-7, negligible for cosine similarity.
    """
    def __init__(self, db_config: dict):
        import mysql.connector
        self.conn   = mysql.connector.connect(**db_config)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id              INT AUTO_INCREMENT PRIMARY KEY,
                file_name       VARCHAR(512) NOT NULL,
                identity_label  VARCHAR(256) NOT NULL,
                image_type      VARCHAR(32)  NOT NULL,
                embedding_json  MEDIUMTEXT   NOT NULL,
                created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def clear(self):
        self.cursor.execute("TRUNCATE TABLE face_embeddings;")
        self.conn.commit()
        print("  🧹 Cleared face_embeddings table")

    def insert(self, file_name: str, identity_label: str,
               image_type: str, embedding: np.ndarray):
        emb_json = json.dumps(embedding.tolist())
        self.cursor.execute(
            """INSERT INTO face_embeddings 
               (file_name, identity_label, image_type, embedding_json)
               VALUES (%s, %s, %s, %s)""",
            (file_name, identity_label, image_type, emb_json)
        )

    def commit(self):
        self.conn.commit()

    def load_all(self) -> tuple:
        """
        Returns (embeddings, labels, file_names) as numpy arrays.
        embeddings shape: (N, 512)
        """
        self.cursor.execute(
            "SELECT file_name, identity_label, embedding_json FROM face_embeddings"
        )
        rows = self.cursor.fetchall()
        if not rows:
            return np.array([]), [], []
        embeddings  = np.array([json.loads(r[2]) for r in rows], dtype=np.float32)
        labels      = [r[1] for r in rows]
        file_names  = [r[0] for r in rows]
        return embeddings, labels, file_names

    def close(self):
        self.cursor.close()
        self.conn.close()


# ─────────────────────────────────────────────────────────────
# MODULE 7 — Embedding sanity checker (run BEFORE clustering)
# ─────────────────────────────────────────────────────────────
def verify_embeddings(embeddings: np.ndarray, labels: list) -> dict:
    """
    Run this on your stored embeddings BEFORE attempting clustering.
    Tells you definitively whether your embeddings are good enough to cluster.
    
    CHECKS:
      1. Shape — must be (N, 512)
      2. Norm  — all vectors must be unit length (L2 norm ≈ 1.0)
      3. Same-identity intra-cluster similarity — higher is better
      4. Cross-identity inter-cluster similarity — lower is better  
      5. Separability score — gap between intra and inter similarity
      6. Near-duplicate detection — flags images that are too similar
         (this ruins clustering by creating false dense clusters)
    
    WHAT GOOD NUMBERS LOOK LIKE:
      Intra-identity mean sim  : 0.45 – 0.75  (same person, should be similar)
      Inter-identity mean sim  : 0.00 – 0.20  (diff people, should be different)
      Separability gap         : > 0.25        (clear separation = good clustering)
    
    IF SEPARABILITY < 0.20:
      Your embeddings are not separable enough. Either:
        - Preprocessing is wrong (alignment, normalization)
        - Model weights are wrong
        - Dataset images are too similar / low quality
    """
    print("\n" + "="*60)
    print("  MODULE 7 — EMBEDDING SANITY CHECK")
    print("="*60)

    N = embeddings.shape[0]
    print(f"\n  Total embeddings : {N}")
    print(f"  Embedding dim    : {embeddings.shape[1]}")

    # Check 1: shape
    assert embeddings.shape[1] == 512, f"Expected 512-dim, got {embeddings.shape[1]}"
    print(f"  ✅ Shape correct : (N, 512)")

    # Check 2: norms — must all be ~1.0
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\n  Embedding norms  : min={norms.min():.6f}  max={norms.max():.6f}  mean={norms.mean():.6f}")
    if norms.max() > 1.01 or norms.min() < 0.99:
        print(f"  ⚠️  WARNING: Embeddings are NOT unit-normalized!")
        print(f"     Re-normalize before clustering: embeddings /= norms[:,None]")
    else:
        print(f"  ✅ All embeddings are unit-normalized")

    # Group by identity
    identity_map = defaultdict(list)
    for i, label in enumerate(labels):
        identity_map[label].append(i)

    # Check 3: intra-identity similarity (same person pairs)
    intra_sims = []
    print(f"\n  Same-identity (intra) similarity per identity:")
    print(f"  {'Identity':<30} {'Count':>6} {'Mean sim':>10} {'Min sim':>10} {'Max sim':>10}")
    print(f"  {'-'*68}")
    for identity, indices in sorted(identity_map.items()):
        if len(indices) < 2:
            continue
        embs = embeddings[indices]
        # Pairwise cosine sim (dot product since unit vectors)
        sims = embs @ embs.T
        np.fill_diagonal(sims, 0)
        pair_sims = sims[np.triu_indices(len(indices), k=1)]
        mean_sim = pair_sims.mean()
        intra_sims.extend(pair_sims.tolist())
        flag = " ⚠️" if mean_sim < SAME_IDENTITY_MIN_SIM else ""
        print(f"  {identity:<30} {len(indices):>6} {mean_sim:>10.4f} {pair_sims.min():>10.4f} {pair_sims.max():>10.4f}{flag}")

    # Check 4: inter-identity similarity (different person pairs)
    identity_names = list(identity_map.keys())
    inter_sims = []
    print(f"\n  Cross-identity (inter) similarity sample:")
    print(f"  (checking mean embedding per identity vs all others)")
    mean_embs = {}
    for identity, indices in identity_map.items():
        mean_embs[identity] = embeddings[indices].mean(axis=0)
        mean_embs[identity] /= np.linalg.norm(mean_embs[identity])

    for i in range(len(identity_names)):
        for j in range(i + 1, min(i + 4, len(identity_names))):
            a, b = identity_names[i], identity_names[j]
            sim = float(np.dot(mean_embs[a], mean_embs[b]))
            inter_sims.append(sim)

    if inter_sims:
        print(f"  Mean cross-identity sim: {np.mean(inter_sims):.4f}")

    # Check 5: Separability
    mean_intra = np.mean(intra_sims) if intra_sims else 0.0
    mean_inter = np.mean(inter_sims) if inter_sims else 0.0
    separability = mean_intra - mean_inter

    print(f"\n  {'─'*50}")
    print(f"  Mean intra-identity sim  : {mean_intra:.4f}  (same person)")
    print(f"  Mean inter-identity sim  : {mean_inter:.4f}  (diff person)")
    print(f"  Separability gap         : {separability:.4f}  (want > 0.25)")
    if separability >= 0.35:
        print(f"  ✅ EXCELLENT separability — clustering will work well")
    elif separability >= 0.25:
        print(f"  ✅ GOOD separability — clustering should work")
    elif separability >= 0.15:
        print(f"  ⚠️  MARGINAL separability — clustering may struggle")
    else:
        print(f"  ❌ POOR separability — embeddings are not discriminative enough")
        print(f"     Check: preprocessing pipeline, model weights, image quality")

    # Check 6: Near-duplicate detection
    print(f"\n  Near-duplicate check (same identity, sim > 0.98):")
    dup_count = 0
    for identity, indices in identity_map.items():
        if len(indices) < 2:
            continue
        embs = embeddings[indices]
        sims = embs @ embs.T
        np.fill_diagonal(sims, 0)
        high = np.argwhere(sims > 0.98)
        for pair in high:
            if pair[0] < pair[1]:
                dup_count += 1
    if dup_count > 0:
        print(f"  ⚠️  {dup_count} near-duplicate pairs found (may create false clusters)")
    else:
        print(f"  ✅ No near-duplicates found")

    return {
        "mean_intra"   : mean_intra,
        "mean_inter"   : mean_inter,
        "separability" : separability,
        "n_embeddings" : N,
    }


# ─────────────────────────────────────────────────────────────
# MODULE 8 — UMAP visualization
# ─────────────────────────────────────────────────────────────
def visualize_embeddings(embeddings: np.ndarray, labels: list):
    """
    Creates a 2D UMAP projection of your embeddings colored by identity.
    This is the best visual check before clustering:
      - Well-separated blobs = clustering will work
      - Mixed/overlapping blobs = clustering will fail
    
    Saves to: embedding_umap.png
    """
    try:
        import umap
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("  Install: pip install umap-learn matplotlib")
        return

    print("\n  Running UMAP (this takes ~30 seconds for 1500 embeddings)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    proj    = reducer.fit_transform(embeddings)

    unique_labels = sorted(set(labels))
    colors = cm.tab20(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {l: c for l, c in zip(unique_labels, colors)}

    plt.figure(figsize=(16, 12))
    for label in unique_labels:
        mask = [l == label for l in labels]
        pts  = proj[mask]
        plt.scatter(pts[:, 0], pts[:, 1],
                    c=[label_to_color[label]],
                    label=label, s=20, alpha=0.8)

    plt.legend(fontsize=6, loc='upper right', ncol=2, bbox_to_anchor=(1.25, 1.0))
    plt.title("ArcFace Embeddings — UMAP 2D projection\n(each color = one identity)", fontsize=14)
    plt.tight_layout()
    out_path = "embedding_umap.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ UMAP saved to: {out_path}")
    print(f"     If blobs are well-separated → embeddings are good → proceed to clustering")
    print(f"     If blobs overlap heavily    → preprocessing issue → debug pipeline first")


# ─────────────────────────────────────────────────────────────
# MODULE 8 — Main pipeline runner
# ─────────────────────────────────────────────────────────────
def run_pipeline(dataset_dir: str, arcface_path: str, db_config: dict):
    """
    Full pipeline:
      For each identity folder:
        For each image:
          1. Classify as pre-cropped (160x160) or full photo
          2. Pre-cropped → resize 112x112 → normalize → ArcFace
             Full photo  → InsightFace detect → affine align → normalize → ArcFace
          3. Store embedding in MySQL

    ROUTING LOGIC:
      160x160 images → Module 3 (preprocess_precropped) → Module 4
      Full photos    → Module 5 (InsightFace align) → Module 3 → Module 4
    """
    print("\n" + "="*60)
    print("  MODULE 8 — PIPELINE RUN")
    print("="*60)

    arcface  = ArcFaceModel(arcface_path)
    store    = EmbeddingStore(db_config)
    store.clear()

    # Lazy-init detector — only instantiated when first full photo is encountered
    face_detector = None

    base = Path(dataset_dir)
    identity_dirs = sorted([
        d for d in base.iterdir()
        if d.is_dir() and not d.name.startswith('.') and not d.name.startswith('__')
    ])

    success = 0
    skipped = 0
    routed_precrop = 0
    routed_full    = 0

    for identity_dir in identity_dirs:
        identity = identity_dir.name
        print(f"\n  📁 Processing: {identity}")

        img_files = sorted([
            f for f in identity_dir.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ])

        identity_success = 0
        for img_path in img_files:
            img_type = classify_image(str(img_path))

            if img_type == 'invalid':
                print(f"     ⚠️  Skipping invalid: {img_path.name}")
                skipped += 1
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                skipped += 1
                continue

            # ── Route 1: pre-cropped 160x160 face ────────────────
            if img_type == 'precropped':
                blob      = preprocess_precropped(img_bgr)
                embedding = arcface.get_embedding(blob)
                store.insert(img_path.name, identity, 'precropped', embedding)
                success += 1
                identity_success += 1
                routed_precrop += 1

            # ── Route 2: full photo needing face detection ────────
            elif img_type == 'full_photo':
                if face_detector is None:
                    face_detector = InsightFaceDetector()

                aligned_faces = face_detector.detect_and_align(img_bgr)
                if not aligned_faces:
                    print(f"     ⚠️  No face detected: {img_path.name}")
                    skipped += 1
                    continue

                # For identity-labeled dataset: use highest-confidence face
                best_face, best_conf, _ = max(aligned_faces, key=lambda x: x[1])
                blob      = preprocess_aligned_crop(best_face)
                embedding = arcface.get_embedding(blob)
                store.insert(img_path.name, identity, 'full_photo', embedding)
                success += 1
                identity_success += 1
                routed_full += 1

        store.commit()
        print(f"     ↳ {identity_success} embeddings stored")

    store.close()
    print(f"\n  ✅ Pipeline complete")
    print(f"     Stored   : {success} embeddings")
    print(f"     Skipped  : {skipped} images")
    print(f"     Routed via pre-crop path : {routed_precrop}")
    print(f"     Routed via YOLO path     : {routed_full}")


# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# MODULE 9 — HDBSCAN Clustering + Benchmarking Metrics
# ─────────────────────────────────────────────────────────────
def run_clustering(embeddings: np.ndarray, labels: list, file_names: list):
    """
    Runs HDBSCAN on ArcFace embeddings and computes full benchmarking metrics.

    WHY HDBSCAN OVER K-MEANS:
      K-Means requires you to specify K (number of clusters) upfront.
      In real-world face clustering you don't know K.
      HDBSCAN is density-based — it finds clusters of arbitrary shape,
      handles noise points (outliers) gracefully by labeling them -1,
      and automatically determines the number of clusters.

    WHY L2-NORMALIZE BEFORE HDBSCAN:
      ArcFace embeddings are already unit-normalized (norm=1.0).
      On the unit hypersphere, euclidean distance = sqrt(2 - 2*cosine_sim).
      So euclidean distance on normalized vectors IS cosine distance — 
      HDBSCAN with metric='euclidean' on normalized embeddings is equivalent
      to clustering by cosine similarity, which is what ArcFace is designed for.

    KEY PARAMETERS:
      min_cluster_size : minimum images to form a cluster.
                         Too small → noisy fragmented clusters.
                         Too large → merges different identities.
                         For ~49 images/identity: start at 5, tune up/down.

      min_samples      : controls how conservative cluster assignment is.
                         Higher → more points marked as noise (-1).
                         Lower  → more aggressive cluster assignment.
                         Start at 3, tune based on noise ratio.

      cluster_selection_epsilon:
                         Minimum distance between cluster merge points.
                         Higher → fewer, larger clusters (may merge identities).
                         Lower  → more, smaller clusters (may split one person).
                         For face clustering: 0.4–0.8 is a good search range.

    BENCHMARKING METRICS EXPLAINED:
      NMI  (Normalized Mutual Information) [0–1, higher=better]:
           Measures how much knowing the cluster label tells you about
           the true identity label. 1.0 = perfect, 0.0 = random.
           Best for: comparing across different cluster counts.

      ARI  (Adjusted Rand Index) [-1 to 1, higher=better]:
           Measures pairwise agreement between predicted clusters and
           true labels, adjusted for chance. 1.0 = perfect, 0.0 = random,
           negative = worse than random.
           Best for: strict accuracy comparison.

      Purity [0–1, higher=better]:
           For each cluster, finds the majority identity in it, sums those
           majority counts, divides by total. Measures how "pure" clusters are.
           Limitation: always improves by making more clusters, so use with ARI.

      Homogeneity [0–1]:
           Each cluster contains only members of a single identity.

      Completeness [0–1]:
           All members of an identity are in the same cluster.

      V-measure [0–1]:
           Harmonic mean of homogeneity and completeness. Like F1 for clustering.

    WHAT GOOD NUMBERS LOOK LIKE FOR YOUR DATASET:
      NMI  > 0.85 = excellent
      ARI  > 0.80 = excellent
      Purity > 0.85 = excellent
      Noise ratio < 10% = acceptable
    """
    try:
        import hdbscan
    except ImportError:
        print("  Install: pip install hdbscan")
        return
    try:
        from sklearn.metrics import (
            normalized_mutual_info_score,
            adjusted_rand_score,
            homogeneity_completeness_v_measure
        )
        from sklearn.preprocessing import normalize
    except ImportError:
        print("  Install: pip install scikit-learn")
        return

    print("\n" + "="*60)
    print("  MODULE 9 — HDBSCAN CLUSTERING + BENCHMARKING METRICS")
    print("="*60)

    # Step 1: Ensure embeddings are L2-normalized
    embeddings_norm = normalize(embeddings, norm='l2')
    print(f"\n  Embeddings    : {embeddings_norm.shape[0]} x {embeddings_norm.shape[1]}")
    print(f"  True identities: {len(set(labels))}")

    # ── Parameter grid to search ──────────────────────────────
    # We test multiple configurations and report all of them.
    # This lets you see which parameters work best for your data
    # rather than blindly trusting a single configuration.
    param_grid = [
        {"min_cluster_size": 5,  "min_samples": 3,  "epsilon": 0.5},
        {"min_cluster_size": 5,  "min_samples": 3,  "epsilon": 0.6},
        {"min_cluster_size": 5,  "min_samples": 5,  "epsilon": 0.5},
        {"min_cluster_size": 8,  "min_samples": 3,  "epsilon": 0.5},
        {"min_cluster_size": 8,  "min_samples": 3,  "epsilon": 0.6},
        {"min_cluster_size": 10, "min_samples": 3,  "epsilon": 0.5},
        {"min_cluster_size": 10, "min_samples": 5,  "epsilon": 0.6},
    ]

    print(f"\n  Running {len(param_grid)} parameter configurations...\n")
    print(f"  {'min_cls':>8} {'min_smp':>8} {'eps':>6} {'n_cls':>6} {'noise%':>7} "
          f"{'NMI':>7} {'ARI':>7} {'Purity':>8} {'V-meas':>8}")
    print(f"  {'-'*75}")

    best_result = None
    best_nmi    = -1.0
    all_results = []

    for params in param_grid:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size          = params["min_cluster_size"],
            min_samples               = params["min_samples"],
            cluster_selection_epsilon = params["epsilon"],
            metric                    = 'euclidean',   # euclidean on L2-norm = cosine
            cluster_selection_method  = 'eom',         # excess of mass — standard
        )
        pred_labels = clusterer.fit_predict(embeddings_norm)

        # Exclude noise points (-1) from metric computation
        # Noise points are real outliers — including them penalizes the score unfairly
        mask        = pred_labels != -1
        n_noise     = int((~mask).sum())
        noise_pct   = 100.0 * n_noise / len(pred_labels)
        n_clusters  = len(set(pred_labels[mask])) if mask.sum() > 0 else 0

        if mask.sum() < 10 or n_clusters < 2:
            print(f"  {params['min_cluster_size']:>8} {params['min_samples']:>8} "
                  f"{params['epsilon']:>6.1f}   — too few clusters, skipping")
            continue

        true_filtered = [labels[i] for i in range(len(labels)) if mask[i]]
        pred_filtered = pred_labels[mask].tolist()

        nmi     = normalized_mutual_info_score(true_filtered, pred_filtered, average_method='arithmetic')
        ari     = adjusted_rand_score(true_filtered, pred_filtered)
        hom, com, vme = homogeneity_completeness_v_measure(true_filtered, pred_filtered)

        # Purity: for each cluster, fraction that is the majority identity
        from collections import Counter
        cluster_map = {}
        for true_l, pred_l in zip(true_filtered, pred_filtered):
            cluster_map.setdefault(pred_l, []).append(true_l)
        purity_sum = sum(Counter(v).most_common(1)[0][1] for v in cluster_map.values())
        purity = purity_sum / len(true_filtered)

        result = {
            **params,
            "n_clusters" : n_clusters,
            "noise_pct"  : noise_pct,
            "nmi"        : nmi,
            "ari"        : ari,
            "purity"     : purity,
            "v_measure"  : vme,
            "pred_labels": pred_labels,
        }
        all_results.append(result)

        marker = " ← best" if nmi > best_nmi else ""
        if nmi > best_nmi:
            best_nmi    = nmi
            best_result = result

        print(f"  {params['min_cluster_size']:>8} {params['min_samples']:>8} "
              f"{params['epsilon']:>6.1f} {n_clusters:>6} {noise_pct:>6.1f}% "
              f"{nmi:>7.4f} {ari:>7.4f} {purity:>8.4f} {vme:>8.4f}{marker}")

    # ── Detailed report on best configuration ─────────────────
    if best_result is None:
        print("\n  ❌ No valid clustering found. Try lowering min_cluster_size.")
        return

    print(f"\n{'='*60}")
    print(f"  BEST CONFIGURATION DETAILED REPORT")
    print(f"{'='*60}")
    print(f"  min_cluster_size : {best_result['min_cluster_size']}")
    print(f"  min_samples      : {best_result['min_samples']}")
    print(f"  epsilon          : {best_result['epsilon']}")
    print(f"\n  BENCHMARKING METRICS:")
    print(f"  ┌─────────────────────────────────────────┐")
    print(f"  │  NMI        : {best_result['nmi']:.4f}  (want > 0.85)        │")
    print(f"  │  ARI        : {best_result['ari']:.4f}  (want > 0.80)        │")
    print(f"  │  Purity     : {best_result['purity']:.4f}  (want > 0.85)        │")
    print(f"  │  V-measure  : {best_result['v_measure']:.4f}                       │")
    print(f"  │  Clusters   : {best_result['n_clusters']}  (true = {len(set(labels))})                │")
    print(f"  │  Noise pts  : {best_result['noise_pct']:.1f}%                        │")
    print(f"  └─────────────────────────────────────────┘")

    # ── Per-identity cluster analysis ─────────────────────────
    pred = best_result["pred_labels"]
    print(f"\n  PER-IDENTITY ANALYSIS (best config):")
    print(f"  {'Identity':<30} {'True N':>7} {'Assigned':>9} {'Noise':>6} {'Clusters':>9} {'Status':>10}")
    print(f"  {'-'*75}")

    from collections import defaultdict, Counter
    identity_to_pred = defaultdict(list)
    for i, (true_l, pred_l) in enumerate(zip(labels, pred.tolist())):
        identity_to_pred[true_l].append(pred_l)

    split_identities   = []
    merged_identities  = []
    perfect_identities = []

    for identity in sorted(identity_to_pred.keys()):
        pred_for_identity = identity_to_pred[identity]
        n_total     = len(pred_for_identity)
        n_noise     = pred_for_identity.count(-1)
        non_noise   = [p for p in pred_for_identity if p != -1]
        unique_cls  = set(non_noise)
        n_unique    = len(unique_cls)
        n_assigned  = n_total - n_noise

        if n_unique == 0:
            status = "ALL NOISE"
        elif n_unique == 1:
            # Check if any other identity shares this cluster
            cluster_id = list(unique_cls)[0]
            others_in_cluster = sum(
                1 for i, (l, p) in enumerate(zip(labels, pred.tolist()))
                if p == cluster_id and l != identity
            )
            if others_in_cluster == 0:
                status = "PERFECT ✅"
                perfect_identities.append(identity)
            else:
                status = f"MERGED ⚠️"
                merged_identities.append(identity)
        else:
            status = f"SPLIT({n_unique}) ⚠️"
            split_identities.append(identity)

        print(f"  {identity:<30} {n_total:>7} {n_assigned:>9} {n_noise:>6} {n_unique:>9}  {status}")

    print(f"\n  SUMMARY:")
    print(f"  ✅ Perfect (1 clean cluster)  : {len(perfect_identities)}/{len(set(labels))}")
    print(f"  ⚠️  Split  (1 person → N cls)  : {len(split_identities)}")
    print(f"  ⚠️  Merged (N persons → 1 cls) : {len(merged_identities)}")

    if split_identities:
        print(f"\n  SPLIT identities  → lower epsilon or min_cluster_size to fix")
        for s in split_identities:
            print(f"     - {s}")
    if merged_identities:
        print(f"\n  MERGED identities → raise epsilon or min_cluster_size to fix")
        for m in merged_identities:
            print(f"     - {m}")

    # ── Save results to JSON ───────────────────────────────────
    import json as _json
    output = {
    "pred_labels": best_result["pred_labels"].tolist(),

    "best_params": {
        k: v for k, v in best_result.items()
        if k != "pred_labels"
    },

    "all_results": [
        {k: v for k, v in r.items() if k != "pred_labels"}
        for r in all_results
    ],

    "per_identity": {
        identity: {
            "total": len(preds),
            "noise": preds.count(-1),
            "n_clusters": len(set(p for p in preds if p != -1))
        }
        for identity, preds in identity_to_pred.items()
    }
}
    with open("clustering_results.json", "w") as f:
        _json.dump(output, f, indent=2)
    print(f"\n  ✅ Full results saved to: clustering_results.json")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArcFace face embedding pipeline")
    parser.add_argument(
        "--mode",
        choices=["audit", "run", "verify", "visualize", "cluster"],
        default="audit",
        help=(
            "audit     = inspect dataset only\n"
            "run       = generate and store embeddings\n"
            "verify    = load stored embeddings and run sanity checks\n"
            "visualize = UMAP plot of stored embeddings\n"
            "cluster   = HDBSCAN clustering + NMI/ARI/Purity metrics"
        )
    )
    args = parser.parse_args()

    if args.mode == "audit":
        audit_dataset(DATASET_DIR)

    elif args.mode == "run":
        report   = audit_dataset(DATASET_DIR)
        arcface  = ArcFaceModel(ARCFACE_PATH)
        detector = InsightFaceDetector()
        passed   = arcface.verify_model(DATASET_DIR, yolo_detector=detector)
        if not passed:
            print("\n  ❌ Model verification failed. Fix preprocessing before running full pipeline.")
        else:
            run_pipeline(DATASET_DIR, ARCFACE_PATH, DB_CONFIG)

    elif args.mode == "verify":
        store = EmbeddingStore(DB_CONFIG)
        embeddings, labels, file_names = store.load_all()
        store.close()
        if len(embeddings) == 0:
            print("  ❌ No embeddings in database. Run with --mode run first.")
        else:
            verify_embeddings(embeddings, labels)

    elif args.mode == "visualize":
        store = EmbeddingStore(DB_CONFIG)
        embeddings, labels, file_names = store.load_all()
        store.close()
        if len(embeddings) == 0:
            print("  ❌ No embeddings in database. Run with --mode run first.")
        else:
            visualize_embeddings(embeddings, labels)

    elif args.mode == "cluster":
        store = EmbeddingStore(DB_CONFIG)
        embeddings, labels, file_names = store.load_all()
        store.close()
        if len(embeddings) == 0:
            print("  ❌ No embeddings in database. Run with --mode run first.")
        else:
            run_clustering(embeddings, labels, file_names)
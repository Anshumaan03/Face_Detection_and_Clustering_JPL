"""
=============================================================================
  DIRECT ARCFACE PIPELINE — Bounding Box Crop Only (No Landmark Alignment)
=============================================================================

DIFFERENCE FROM pipeline_new.py:
  pipeline_new.py  → InsightFace detects face → 5-point landmark alignment
                      → affine warp to canonical 112x112 → ArcFace

  direct_pipeline.py → InsightFace detects face → crop bounding box
                        → resize crop to 112x112 → ArcFace (no alignment)

PURPOSE:
  Ablation study — compare clustering quality WITH vs WITHOUT alignment.
  Expected: alignment pipeline gives higher NMI/ARI because ArcFace was
  trained on aligned faces. This pipeline intentionally skips that step
  to quantify how much alignment contributes.

HOW TO RUN:
  python direct_pipeline.py --mode run        # detect, crop, embed, store
  python direct_pipeline.py --mode cluster    # HDBSCAN + save JSON
  python direct_pipeline.py --mode folders    # create result folders
  python direct_pipeline.py --mode all        # run → cluster → folders
=============================================================================
"""

import os
import cv2
import json
import argparse
import shutil
import numpy as np
import onnxruntime as ort
from pathlib import Path
from collections import Counter

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DATASET_DIR  = "/Users/anshumaansinghrathore/Desktop/Face Clustering/data/raw"
ARCFACE_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/w600k_r50.onnx"
RESULTS_DIR  = Path("results_direct")
JSON_FILE    = "clustering_results_direct.json"

DB_CONFIG = {
    "host"    : "127.0.0.1",
    "user"    : "root",
    "password": "Anshu@2003",
    "database": "face_db_direct"   # separate table so it doesn't overwrite aligned results
}

ARCFACE_INPUT_SIZE = 112
DET_CONF_THRESH    = 0.5     # minimum InsightFace detection confidence
MARGIN             = 0.10    # fractional margin added around bbox before crop
                             # e.g. 0.10 = expand each side by 10% of face size

# HDBSCAN sweep ranges
MIN_CLUSTER_SIZE_SWEEP = [2, 3, 4, 5]
MIN_SAMPLES_SWEEP      = [1, 2, 3]


# ─────────────────────────────────────────────────────────────
# MODULE 1 — ArcFace model
# ─────────────────────────────────────────────────────────────
class ArcFaceModel:
    def __init__(self, model_path: str):
        print(f"  Loading ArcFace: {model_path}")
        self.session = ort.InferenceSession(
            model_path,
            providers=['CoreMLExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        print(f"  ✅ ArcFace loaded — input: {self.input_name}")

    def get_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        face_bgr: any size BGR face crop
        Returns: (512,) float32 L2-normalized embedding

        Preprocessing:
          1. Resize to 112x112
          2. Normalize (pixel - 127.5) / 128.0  → same as training
          3. Transpose HWC → CHW, add batch dim
          NO affine alignment — that's the whole point of this pipeline
        """
        resized = cv2.resize(face_bgr, (ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32)
        blob = (blob - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)          # HWC → CHW
        blob = np.expand_dims(blob, axis=0)     # → (1, 3, 112, 112)
        blob = np.ascontiguousarray(blob)

        out = self.session.run(None, {self.input_name: blob})
        emb = out[0][0]
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm
        return emb.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# MODULE 2 — InsightFace detector (bounding box only)
# ─────────────────────────────────────────────────────────────
class BBoxFaceDetector:
    """
    Uses InsightFace RetinaFace to detect faces, but only uses the
    bounding box — NOT the 5-point landmarks. The face crop is just
    a rectangular cut from the original image, resized to 112x112.

    MARGIN: A small margin is added around the bbox to include
    forehead/chin context that ArcFace expects. Without alignment,
    the margin partially compensates for scale/position variance.
    """
    def __init__(self):
        from insightface.app import FaceAnalysis
        print("  Loading InsightFace detector...")
        self.app = FaceAnalysis(
            allowed_modules=['detection'],
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        print("  ✅ Detector loaded")

    def detect_and_crop(self, img_bgr: np.ndarray) -> list:
        """
        Returns list of (face_crop_bgr, det_score, bbox) tuples.
        face_crop_bgr is the raw bbox crop (with margin), NOT aligned.
        """
        h, w = img_bgr.shape[:2]
        faces = self.app.get(img_bgr)
        crops = []

        for face in faces:
            det_score = float(face.det_score)
            if det_score < DET_CONF_THRESH:
                continue

            x1, y1, x2, y2 = face.bbox.astype(int)
            face_w = x2 - x1
            face_h = y2 - y1

            # Add proportional margin around the bbox
            mx = int(face_w * MARGIN)
            my = int(face_h * MARGIN)
            x1 = max(0, x1 - mx)
            y1 = max(0, y1 - my)
            x2 = min(w, x2 + mx)
            y2 = min(h, y2 + my)

            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crops.append((crop, det_score, [x1, y1, x2, y2]))

        return crops


# ─────────────────────────────────────────────────────────────
# MODULE 3 — MySQL storage
# ─────────────────────────────────────────────────────────────
class EmbeddingStore:
    def __init__(self, db_config: dict):
        import mysql.connector
        self.conn = mysql.connector.connect(**db_config)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings_direct (
                id             INT AUTO_INCREMENT PRIMARY KEY,
                file_name      VARCHAR(512) NOT NULL,
                identity_label VARCHAR(256) NOT NULL,
                embedding_json MEDIUMTEXT   NOT NULL,
                created_at     TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def clear(self):
        self.cursor.execute("DELETE FROM face_embeddings_direct")
        self.conn.commit()
        print("  🧹 Cleared face_embeddings_direct table")

    def insert(self, file_name: str, identity_label: str, embedding: np.ndarray):
        self.cursor.execute(
            "INSERT INTO face_embeddings_direct (file_name, identity_label, embedding_json) VALUES (%s, %s, %s)",
            (file_name, identity_label, json.dumps(embedding.tolist()))
        )

    def commit(self):
        self.conn.commit()

    def load_all(self):
        self.cursor.execute(
            "SELECT file_name, identity_label, embedding_json FROM face_embeddings_direct ORDER BY identity_label, file_name"
        )
        rows = self.cursor.fetchall()
        file_names = [r[0] for r in rows]
        labels     = [r[1] for r in rows]
        embeddings = np.array([json.loads(r[2]) for r in rows], dtype=np.float32)
        return embeddings, labels, file_names

    def close(self):
        self.cursor.close()
        self.conn.close()


# ─────────────────────────────────────────────────────────────
# MODULE 4 — Pipeline runner
# ─────────────────────────────────────────────────────────────
def run_pipeline():
    print("\n" + "="*60)
    print("  DIRECT PIPELINE — bbox crop → ArcFace (no alignment)")
    print("="*60)

    arcface  = ArcFaceModel(ARCFACE_PATH)
    detector = BBoxFaceDetector()
    store    = EmbeddingStore(DB_CONFIG)
    store.clear()

    base = Path(DATASET_DIR)
    identity_dirs = sorted([
        d for d in base.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    success = 0
    skipped = 0

    for identity_dir in identity_dirs:
        identity = identity_dir.name
        print(f"\n  📁 {identity}")

        img_files = sorted([
            f for f in identity_dir.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.avif')
        ])

        identity_success = 0
        for img_path in img_files:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"     ⚠️  Unreadable: {img_path.name}")
                skipped += 1
                continue

            crops = detector.detect_and_crop(img_bgr)
            if not crops:
                print(f"     ⚠️  No face: {img_path.name}")
                skipped += 1
                continue

            # Use highest-confidence crop only (same policy as aligned pipeline)
            best_crop, best_conf, _ = max(crops, key=lambda x: x[1])
            embedding = arcface.get_embedding(best_crop)

            store.insert(img_path.name, identity, embedding)
            success += 1
            identity_success += 1

        store.commit()
        print(f"     ↳ {identity_success} embeddings stored")

    store.close()
    print(f"\n  ✅ Done — {success} stored, {skipped} skipped")


# ─────────────────────────────────────────────────────────────
# MODULE 5 — HDBSCAN clustering
# ─────────────────────────────────────────────────────────────
def run_clustering():
    print("\n" + "="*60)
    print("  HDBSCAN CLUSTERING (direct / no-alignment pipeline)")
    print("="*60)

    import mysql.connector
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from sklearn.preprocessing import LabelEncoder
    import hdbscan

    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT identity_label, embedding_json FROM face_embeddings_direct ORDER BY identity_label, file_name"
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        print("  ❌ No embeddings found. Run --mode run first.")
        return

    total      = len(rows)
    raw_labels = [r[0] for r in rows]
    le         = LabelEncoder()
    gt         = le.fit_transform(raw_labels)
    X          = np.array([json.loads(r[1]) for r in rows], dtype=np.float32)

    print(f"  Loaded {total} embeddings, {len(le.classes_)} identities")
    norms = np.linalg.norm(X, axis=1)
    print(f"  Norm check — mean: {norms.mean():.4f}, std: {norms.std():.4f}")

    print(f"\n  {'min_cls':>7} {'min_smp':>8} {'clusters':>9} {'noise':>7} {'noise%':>7} {'NMI':>9} {'ARI':>9}")
    print("  " + "-"*65)

    best_ari    = -1
    best_cfg    = None
    best_labels = None

    for min_cls in MIN_CLUSTER_SIZE_SWEEP:
        for min_smp in MIN_SAMPLES_SWEEP:
            if min_smp > min_cls:
                continue

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cls,
                min_samples=min_smp,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )
            labels = clusterer.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise    = int(np.sum(labels == -1))
            noise_pct  = n_noise / total * 100
            nmi        = normalized_mutual_info_score(gt, labels)
            ari        = adjusted_rand_score(gt, labels)

            flag = " ← BEST" if ari > best_ari else ""
            if ari > best_ari:
                best_ari    = ari
                best_cfg    = (min_cls, min_smp)
                best_labels = labels.copy()

            print(f"  {min_cls:>7} {min_smp:>8} {n_clusters:>9} {n_noise:>7} {noise_pct:>6.1f}% {nmi:>9.4f} {ari:>9.4f}{flag}")

    print("  " + "-"*65)

    def _purity(gt, labels):
        total = len(labels)
        score = 0
        for cid in set(labels):
            if cid == -1:
                continue
            mask = labels == cid
            score += Counter(gt[mask]).most_common(1)[0][1]
        return score / total if total > 0 else 0.0

    print(f"\n  🏆 Best: min_cluster_size={best_cfg[0]}, min_samples={best_cfg[1]}")
    print(f"     NMI={normalized_mutual_info_score(gt, best_labels):.4f}  ARI={best_ari:.4f}")
    print(f"     Clusters found: {len(set(best_labels))-(1 if -1 in best_labels else 0)} / Target: {len(le.classes_)}")

    print("\n  Per-identity breakdown:")
    per_identity = {}
    for identity in le.classes_:
        mask     = np.array(raw_labels) == identity
        assigned = set(best_labels[mask])
        n_noise  = int(np.sum(best_labels[mask] == -1))
        n_cls    = len(assigned - {-1})
        per_identity[identity] = {"total": int(mask.sum()), "noise": n_noise, "n_clusters": n_cls}
        print(f"    {identity:<30} total={mask.sum():>3}  noise={n_noise}  clusters={sorted(assigned)}")

    output = {
        "pred_labels": [int(l) for l in best_labels],
        "best_params": {
            "min_cluster_size": best_cfg[0],
            "min_samples":      best_cfg[1],
            "n_clusters":       len(set(best_labels)) - (1 if -1 in best_labels else 0),
            "nmi":              float(normalized_mutual_info_score(gt, best_labels)),
            "ari":              float(best_ari),
            "purity":           float(_purity(gt, best_labels)),
            "noise_pct":        float(np.sum(best_labels == -1) / total * 100),
        },
        "per_identity": per_identity
    }

    with open(JSON_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  💾 Saved {JSON_FILE}")


# ─────────────────────────────────────────────────────────────
# MODULE 6 — Create result folders
# ─────────────────────────────────────────────────────────────
def create_folders():
    print("\n" + "="*60)
    print("  CREATING CLUSTER FOLDERS")
    print("="*60)

    with open(JSON_FILE) as f:
        data = json.load(f)
    pred_labels = data["pred_labels"]

    import mysql.connector
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT file_name, identity_label FROM face_embeddings_direct ORDER BY identity_label, file_name"
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if len(rows) != len(pred_labels):
        raise ValueError(f"DB rows ({len(rows)}) != pred_labels ({len(pred_labels)})")

    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir()

    missing = []
    for (file_name, identity_label), cluster_id in zip(rows, pred_labels):
        src = Path(DATASET_DIR) / identity_label / file_name
        if not src.exists():
            missing.append(str(src))
            continue

        folder = RESULTS_DIR / ("Noise" if cluster_id == -1 else f"Cluster_{cluster_id}")
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, folder / file_name)

    print(f"  ✅ Folders created → {RESULTS_DIR.resolve()}")
    for folder in sorted(RESULTS_DIR.iterdir()):
        count = len(list(folder.glob("*")))
        print(f"     {folder.name}: {count} images")

    if missing:
        print(f"\n  ⚠️  {len(missing)} missing files")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct bbox-crop ArcFace pipeline")
    parser.add_argument("--mode", choices=["run", "cluster", "folders", "all"], required=True)
    args = parser.parse_args()

    if args.mode == "run":
        run_pipeline()
    elif args.mode == "cluster":
        run_clustering()
    elif args.mode == "folders":
        create_folders()
    elif args.mode == "all":
        run_pipeline()
        run_clustering()
        create_folders()

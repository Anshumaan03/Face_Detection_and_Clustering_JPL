"""
run_pipeline.py
================
End-to-end orchestration for the single-model (InsightFace detector + ArcFace
embedding) pipeline:

    load_image -> detect_faces -> align_face -> extract embedding
    -> L2 normalize -> store in MySQL
    [repeat for every image in dataset]
    -> cluster(run_label)  (HDBSCAN + centroid build)
    -> print metrics

Run:
    python run_pipeline.py                          # full run
    python run_pipeline.py --skip-extraction         # just re-cluster existing DB embeddings
    python run_pipeline.py --limit 50                # debug on a small subset first
    python run_pipeline.py --calibrate-thresholds    # suggest T1/T2 from ground truth, then exit
    python run_pipeline.py --reclaim-noise           # run Flow-1 logic over cluster -1, then exit
"""

import argparse
import logging

from tqdm import tqdm

import config
from common import load_image, detect_faces, largest_or_best_face, align_face, iter_dataset
from embedding import ArcFaceExtractor, get_normalized_embedding
from storage import Storage
from clustering import cluster
from recommendation import calibrate_thresholds, reclaim_noise

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_extraction(limit: int = None):
    db = Storage()
    db.create_schema()
    extractor = ArcFaceExtractor()

    items = list(iter_dataset(config.DATA_ROOT))
    if limit:
        items = items[:limit]

    n_no_face = 0
    n_ok = 0

    for identity, image_path in tqdm(items, desc="Extracting"):
        try:
            img = load_image(image_path)
        except FileNotFoundError as e:
            logger.warning(str(e))
            continue

        faces = detect_faces(img, image_path=image_path)
        face = largest_or_best_face(faces)
        if face is None:
            n_no_face += 1
            logger.warning("No face detected: %s", image_path)
            continue

        face_id = db.insert_face(
            identity=identity, image_path=image_path,
            bbox=face.bbox, landmarks=face.landmarks, det_score=face.det_score,
        )

        aligned_crop = align_face(img, face)
        vec = get_normalized_embedding(extractor, aligned_crop)
        db.insert_embedding(face_id=face_id, vector=vec)

        n_ok += 1

    db.close()
    logger.info("Extraction done: %d ok, %d with no detected face.", n_ok, n_no_face)


def run_clustering_and_report(run_label: str = "default"):
    result = cluster(run_label=run_label)
    print("\n=== Clustering metrics ===")
    for k, v in result["metrics"].items():
        print(f"{k:>22}: {v}")
    return result


def run_calibration(run_label: str = "default"):
    """Prints data-driven T1/T2 suggestions, using the ground-truth identity
    labels this dataset already has. Requires clustering to have run at least
    once (so embeddings exist in the DB)."""
    db = Storage()
    print("\n=== Threshold calibration (data-driven, using ground-truth identity) ===")
    try:
        result = calibrate_thresholds(run_label, db)
        flag = "" if result["clean_separation"] else "  <-- percentile T1 > T2, no clean global cutoff on this data"
        print(f"T1 (percentile method) = {result['suggested_t1']:.3f}")
        print(f"T1 (precision method, prefer this) = {result['suggested_t1_precision']:.3f}  "
              f"(achieved false-merge rate: {result['achieved_false_merge_rate']:.2%})")
        print(f"T2 = {result['suggested_t2']:.3f}"
              f"  (same-id mean={result['same_identity_dist_mean']:.3f},"
              f" diff-id mean={result['diff_identity_dist_mean']:.3f}){flag}")
        if result["no_safe_auto_merge_zone"]:
            print("WARNING: no distance cutoff in your data meets the target false-merge rate -- "
                  "consider a higher target rate or disabling auto-merge (T1 near 0.0).")
    except ValueError as e:
        logger.warning("Skipping calibration: %s", e)
    db.close()


def run_noise_reclamation(run_label: str = "default", auto_apply: bool = True):
    """Routes every noise (-1) face back through the Flow-1 decision logic."""
    db = Storage()
    print("\n=== Noise reclamation ===")
    result = reclaim_noise(run_label, db, auto_apply=auto_apply)
    print(f"auto-merged={len(result['auto_merged'])}  "
          f"needs-review={len(result['suggestions'])}  left-as-noise={len(result['left_as_noise'])}")
    db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extraction", action="store_true",
                         help="Skip detection+embedding, just re-run clustering on existing DB data.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Limit number of images processed (debugging).")
    parser.add_argument("--run-label", type=str, default="default",
                         help="Label for this clustering run (lets you keep multiple runs side by side).")
    parser.add_argument("--calibrate-thresholds", action="store_true",
                         help="Print data-driven T1/T2 suggestions and exit (no extraction/clustering).")
    parser.add_argument("--reclaim-noise", action="store_true",
                         help="Run noise reclamation (Flow 1 applied to cluster -1 points) and exit.")
    args = parser.parse_args()

    if args.calibrate_thresholds:
        run_calibration(run_label=args.run_label)
        return

    if args.reclaim_noise:
        run_noise_reclamation(run_label=args.run_label)
        return

    if not args.skip_extraction:
        run_extraction(limit=args.limit)

    run_clustering_and_report(run_label=args.run_label)


if __name__ == "__main__":
    main()
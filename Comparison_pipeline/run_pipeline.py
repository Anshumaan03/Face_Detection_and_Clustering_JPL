"""
run_pipeline.py
================
End-to-end orchestration:

    load_image -> detect_faces -> align_face (per model) -> extract embedding
    -> L2 normalize -> store in MySQL
    [repeat for every image in dataset]
    -> cluster_all_models (HDBSCAN, identical params per model)
    -> print metrics summary

Run:
    python run_pipeline.py                  # full run
    python run_pipeline.py --skip-extraction # just re-cluster existing DB embeddings
    python run_pipeline.py --limit 50        # debug on a small subset first
"""

import argparse
import logging

from tqdm import tqdm

import config
from common import load_image, detect_faces, largest_or_best_face, align_face, iter_dataset
from embeddings import load_all_extractors, get_normalized_embedding
from storage import Storage
from clustering import cluster_all_models, metrics_summary_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_extraction(limit: int = None):
    db = Storage()
    db.create_schema()
    extractors = load_all_extractors()

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

        for model_name, extractor in extractors.items():
            aligned_crop = align_face(img, face, model=model_name)
            vec = get_normalized_embedding(extractor, aligned_crop)
            db.insert_embedding(face_id=face_id, model=model_name, vector=vec)

        n_ok += 1

    db.close()
    logger.info("Extraction done: %d ok, %d with no detected face.", n_ok, n_no_face)


def run_clustering_and_report(run_label: str = "default"):
    results = cluster_all_models(run_label=run_label)
    table = metrics_summary_table(results)
    print("\n=== Clustering metrics (identical HDBSCAN params across models) ===")
    print(table.to_string())
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extraction", action="store_true",
                         help="Skip detection+embedding, just re-run clustering on existing DB data.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Limit number of images processed (debugging).")
    parser.add_argument("--run-label", type=str, default="default",
                         help="Label for this clustering run (lets you keep multiple runs side by side).")
    args = parser.parse_args()

    if not args.skip_extraction:
        run_extraction(limit=args.limit)

    run_clustering_and_report(run_label=args.run_label)


if __name__ == "__main__":
    main()

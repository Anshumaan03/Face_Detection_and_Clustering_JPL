"""
run_pipeline.py - MULTI-FACE SUPPORT
"""

import os
import argparse
import logging

from tqdm import tqdm

import config
from common import load_image, detect_faces, filter_valid_faces, align_face, iter_dataset
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
    n_multi = 0

    for identity, image_path in tqdm(items, desc="Extracting faces"):
        try:
            img = load_image(image_path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", image_path, e)
            continue

        faces = detect_faces(img, image_path=image_path)
        valid_faces = filter_valid_faces(faces, img)

        if not valid_faces:
            n_no_face += 1
            logger.debug("No valid face: %s", image_path)   # Changed to debug to reduce spam
            continue

        if len(valid_faces) > 1:
            n_multi += 1
            logger.info("Multi-face: %s → %d faces", image_path, len(valid_faces))

        for face_index, face in enumerate(valid_faces):
            crop_path = None
            try:
                from common import align_face as _af
                import cv2
                arcface_crop = _af(img, face, "arcface")
                stem = os.path.splitext(os.path.basename(image_path))[0]
                crop_filename = f"{identity}__{stem}__face{face_index}.jpg"
                crop_dir = os.path.join(config.OUTPUT_ROOT, "crops", "arcface")
                os.makedirs(crop_dir, exist_ok=True)
                crop_path = os.path.join(crop_dir, crop_filename)
                cv2.imwrite(crop_path, arcface_crop)
            except Exception as e:
                logger.error("Crop save failed for %s face %d: %s", image_path, face_index, e)

            try:
                face_id = db.insert_face(
                    identity=identity, image_path=image_path,
                    bbox=face.bbox, landmarks=face.landmarks,
                    det_score=face.det_score,
                    face_index=face_index, crop_path=crop_path,
                )
            except Exception as e:
                import traceback
                logger.error("insert_face FAILED for %s face %d:\n%s", image_path, face_index, traceback.format_exc())
                continue

            for model_name, extractor in extractors.items():
                try:
                    aligned = align_face(img, face, model_name)
                    vec = get_normalized_embedding(extractor, aligned)
                    db.insert_embedding(face_id=face_id, model=model_name, vector=vec)
                except Exception as e:
                    import traceback
                    logger.error("Embedding FAILED %s face %d model %s:\n%s", image_path, face_index, model_name, traceback.format_exc())

        n_ok += 1

    db.close()
    logger.info("Extraction finished → Processed: %d | No valid face: %d | Multi-face images: %d", 
                n_ok, n_no_face, n_multi)


def run_clustering_and_report(run_label: str = "default"):
    results = cluster_all_models(run_label=run_label)
    table = metrics_summary_table(results)
    if table.empty:
        print("\n=== NO RESULTS — all models failed, check logs above ===")
        return table
    print("\n=== FINAL METRICS TABLE ===")
    print(table.to_string())
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-label", type=str, default="multi_face_final")
    args = parser.parse_args()

    if not args.skip_extraction:
        run_extraction(limit=args.limit)

    run_clustering_and_report(run_label=args.run_label)


if __name__ == "__main__":
    main()
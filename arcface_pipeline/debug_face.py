"""
debug_face.py
==============
Standalone diagnostic for ONE image -- shows exactly what the pipeline sees,
outside the app entirely. Useful for exactly this kind of question: "why did
this photo not match the cluster I expected?"

What it prints:
  1. Every face detected in the image (there may be more than one -- e.g. two
     people in frame, or a photo-within-a-photo) and which one the pipeline
     actually used.
  2. The exact aligned 112x112 crop that got fed into ArcFace (optionally
     saved to disk so you can look at it).
  3. Distance from this image's embedding to EVERY existing cluster centroid,
     ranked closest-first, with T1/T2 zones marked.
  4. What Flow 1 would actually decide (status + any spread-based demotion).
  5. Optionally: distance to every INDIVIDUAL reference photo of a named
     identity (not just that identity's centroid average) -- useful to see
     whether this new photo is far from that identity's cluster as a whole,
     or just far from the *centroid* while still close to some individual
     reference photos.

Run:
    python debug_face.py path/to/photo.jpg
    python debug_face.py path/to/photo.jpg --identity "shahrukh khan"
    python debug_face.py path/to/photo.jpg --run-label default --save-crop out.jpg
"""

import argparse

import cv2
import numpy as np

import config
from common import load_image, detect_faces, largest_or_best_face, align_face
from embedding import ArcFaceExtractor, get_normalized_embedding
from storage import Storage
from recommendation import cosine_distance, recommend_for_embedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--identity", type=str, default=None,
                         help="Also compare against every individual reference photo of this "
                              "identity (exact folder name under data/raw/), not just its centroid.")
    parser.add_argument("--run-label", type=str, default="default")
    parser.add_argument("--save-crop", type=str, default=None,
                         help="Save the aligned 112x112 crop that gets embedded, to inspect visually.")
    args = parser.parse_args()

    print(f"\n=== Loading {args.image_path} ===")
    img = load_image(args.image_path)
    print(f"Image shape: {img.shape}")

    faces = detect_faces(img)
    print(f"\n=== Detection ===")
    print(f"{len(faces)} face(s) detected.")
    for i, f in enumerate(faces):
        print(f"  face[{i}]  det_score={f.det_score:.4f}  bbox={[round(v, 1) for v in f.bbox.tolist()]}")

    if not faces:
        print("No face detected -- nothing further to check.")
        return

    face = largest_or_best_face(faces)  # always index 0: detect_faces() sorts by det_score descending
    print(f"\nUsing face[0] (det_score={face.det_score:.4f}) -- the highest-confidence detection.")
    if len(faces) > 1:
        print("*** MULTIPLE FACES DETECTED. If face[0] above isn't the person you meant, that's "
              "very likely the whole story -- the wrong face got aligned and embedded, and every "
              "distance/threshold number below is comparing the WRONG face to your clusters. ***")

    aligned = align_face(img, face)
    if args.save_crop:
        cv2.imwrite(args.save_crop, aligned)
        print(f"\nSaved the aligned crop to: {args.save_crop} -- open it and look at who's actually in it.")

    print("\n=== Embedding ===")
    extractor = ArcFaceExtractor()
    vec = get_normalized_embedding(extractor, aligned)
    print(f"Embedding shape: {vec.shape}, L2 norm: {np.linalg.norm(vec):.4f}")

    db = Storage()

    print(f"\n=== Distance to every existing cluster centroid (run_label='{args.run_label}') ===")
    centroids_df = db.load_centroids_df(args.run_label)
    if centroids_df.empty:
        print("No centroids found for this run_label.")
    else:
        rows = []
        for _, row in centroids_df.iterrows():
            d = cosine_distance(vec, row["centroid_vector"])
            rows.append((d, int(row["cluster_label"]), row["representative_identity"], int(row["n_members"])))
        rows.sort(key=lambda r: r[0])
        for d, label, identity, n in rows:
            flag = ""
            if d < config.THRESHOLDS["t1"]:
                flag = "  <-- within T1 (auto-merge zone)"
            elif d < config.THRESHOLDS["t2"]:
                flag = "  <-- within T2 (ask_user zone)"
            print(f"  cluster {label:>3}  ({identity:<20})  n_members={n:<4}  distance={d:.4f}{flag}")

    rec = recommend_for_embedding(vec, args.run_label, db)
    print(f"\n=== What Flow 1 would actually decide right now ===")
    print(f"  status: {rec['status']}")
    if rec["distance"] is not None:
        print(f"  nearest: cluster {rec['nearest_cluster_label']} ({rec['representative_identity']}), "
              f"distance {rec['distance']:.4f}")
    if rec.get("auto_merge_demoted"):
        print("  NOTE: demoted from auto_merge to ask_user by the per-cluster spread check "
              "(distance cleared global T1 but exceeded this cluster's own historical spread).")

    if args.identity:
        print(f"\n=== Distance to every individual reference photo of identity '{args.identity}' ===")
        all_df = db.load_embeddings_df()
        subset = all_df[all_df["identity"].str.lower() == args.identity.lower()]
        if subset.empty:
            print(f"No embeddings found where identity == '{args.identity}'. Identity matching here "
                  f"is exact against the data/raw/<identity>/ folder name -- check spelling/case.")
        else:
            dists = sorted(
                ((cosine_distance(vec, r["vector"]), r["image_path"]) for _, r in subset.iterrows()),
                key=lambda x: x[0],
            )
            for d, path in dists:
                print(f"  distance={d:.4f}   {path}")
            all_d = [d for d, _ in dists]
            print(f"\n  closest={min(all_d):.4f}  farthest={max(all_d):.4f}  mean={np.mean(all_d):.4f}")
            print("  If this new photo is far from EVERY reference photo of this identity (not just "
                  "the centroid), the centroid isn't the problem -- the new photo's pose/lighting/crop "
                  "is genuinely unusual relative to your whole reference set for this person.")

    db.close()


if __name__ == "__main__":
    main()
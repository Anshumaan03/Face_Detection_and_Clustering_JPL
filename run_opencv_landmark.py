import os
import cv2
import time
import json
from src.detection.Landmark_Detection.OpenCV_landmark import OpenCVLandmarkAligner

def main():
    # Enforce safe directory mapping based on your layout screenshot
    json_path = "/Users/anshumaansinghrathore/Desktop/Face Clustering/data/processed_opencv/opencv_detection_metadata.json"  
    raw_dir = "data/raw"
    output_dir = "data/aligned_opencv"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(json_path):
        print(f"[FATAL ERROR] Bounding box metadata JSON file missing at: {json_path}")
        return

    print("[OPENCV] Initializing core landmark tracking pipeline...")
    try:
        aligner = OpenCVLandmarkAligner(model_path="models/lbfmodel.yaml")
    except Exception as e:
        print(f"[FATAL ERROR] Engine initialization failed: {e}")
        return

    # Ingest pre-calculated bounding box metrics from Phase FR-2
    with open(json_path, "r") as f:
        detection_data = json.load(f)

    print(f"[OPENCV] Metadata loaded successfully. Parsing data nodes...\n" + "-"*75)
    opencv_metrics_registry = []

    # Stream keys directly from your JSON database structure
    for filename, face_data in detection_data.items():
        img_path = os.path.join(raw_dir, filename)
        
        # Verify the original source full-frame matrix asset exists
        if not os.path.exists(img_path):
            print(f"[WARN] Ingestion error: Source photo file missing at {img_path}. Skipping.")
            continue

        bgr_img = cv2.imread(img_path)
        if bgr_img is None:
            print(f"[WARN] Failed to read image matrix array for {filename}. Skipping.")
            continue

        base_name = os.path.splitext(filename)[0]
        
        # Track precise hardware processing clock cycles for benchmarking comparisons
        t0 = time.perf_counter()
        faces_aligned_count = 0

        try:
            # Parse individual structural coordinate objects mapping to this file
            for box in face_data.get("bounding_boxes", []):
                coords = box["coordinates"]
                x = coords["x"]
                y = coords["y"]
                w = coords["width"]
                h = coords["height"]
                box_idx = box["box_index"]

                # OpenCV Facemark LBF expects a clean list/tuple format: [x, y, w, h]
                opencv_bbox = [x, y, w, h]

                # Feed original uncropped picture array + rebuilt bounding box to aligner
                aligned_chip, _ = aligner.compute_alignment(bgr_img, opencv_bbox)

                if aligned_chip is not None:
                    # Write output asset down to your clean opencv sandbox target directory
                    out_path = f"{output_dir}/{base_name}_cv_face_{box_idx}.png"
                    cv2.imwrite(out_path, aligned_chip)
                    faces_aligned_count += 1

            latency = round((time.perf_counter() - t0) * 1000, 2)
            
            # Register structural diagnostic speeds for final analysis
            opencv_metrics_registry.append({
                "file": filename,
                "faces_aligned": faces_aligned_count,
                "latency_ms": latency
            })
            print(f"[SUCCESS] {filename} -> Standardized {faces_aligned_count} face chip(s) [Velocity: {latency}ms]")

        except Exception as e:
            print(f"[ERROR] Failed extracting features for {filename}: {e}")

    # Output benchmark records to root workspace
    with open("opencv_alignment_metrics.json", "w") as f:
        json.dump(opencv_metrics_registry, f, indent=2)

    print(f"\n[PIPELINE COMPLETE] Clean, straight face chips populated inside '{output_dir}/'.")

if __name__ == "__main__":
    main()
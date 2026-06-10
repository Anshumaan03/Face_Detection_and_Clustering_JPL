import os
import cv2
import json
import time
from detection.Image_Loader.Image_loader import ImageLoader
from detection.Face_Detection.Dlib_detector import DlibHOGDetector

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed_dlib"  # Targeted directly to your separate Dlib sandbox
METADATA_FILE = os.path.join(PROCESSED_DIR, "detection_metadata.json")

def main():
    print("=====================================================")
    print("[INFO] Initializing FR-2 Structured Dlib (HOG) Pipeline...")
    print("=====================================================\n")
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    loader = ImageLoader(target_dir=RAW_DIR)
    detector = DlibHOGDetector(upsample_num_times=1)
    
    metadata_registry = {}
    total_faces_found = 0
    processed_images_count = 0
    
    print("[INFO] Scanning stream and building HOG bounding box registry...")
    
    for payload in loader.load_valid_images():
        processed_images_count += 1
        
        # Adaptive unpacking wrapper for your loader dictionary
        if isinstance(payload, dict):
            img_matrix = payload.get('image') or payload.get('img') or payload.get('matrix') or list(payload.values())[0]
            filename = payload.get('filename') or payload.get('name') or payload.get('file') or list(payload.values())[1]
        else:
            img_matrix = payload[0]
            filename = payload[1]
            
        # Detect using Dlib HOG structural analysis
        faces_images, bounding_boxes = detector.detect_and_crop(img_matrix)

        
        # TIMER BLOCK: Capture exact hardware calculation latency
        start_time = time.perf_counter()
        faces_images, bounding_boxes = detector.detect_and_crop(img_matrix)
        latency = time.perf_counter() - start_time
  
        
        if len(faces_images) > 0:
            metadata_registry[filename] = {
                "total_faces_found": len(faces_images),
                "bounding_boxes": []
            }
            
            for idx, (face_img, box_coords) in enumerate(zip(faces_images, bounding_boxes)):
                total_faces_found += 1
                x, y, w, h = box_coords
                
                base_name, ext = os.path.splitext(filename)
                output_filename = f"crop_dlib_{base_name}_{idx}{ext}"
                output_path = os.path.join(PROCESSED_DIR, output_filename)
                
                # Save the high-quality HOG target slice to your dlib folder
                cv2.imwrite(output_path, face_img)
                
                box_entry = {
                    "box_index": idx,
                    "crop_filename": output_filename,
                    "coordinates": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    }
                }
                metadata_registry[filename]["bounding_boxes"].append(box_entry)
                
        if processed_images_count % 100 == 0:
            print(f"   ↳ HOG Sweep Log: Checked {processed_images_count} source files. Bounding boxes registered: {total_faces_found}")

    print(f"\n[INFO] Compiling master Dlib dataset coordinate registry map...")
    os.makedirs(PROCESSED_DIR, exist_ok=True) # if the directory is deleted then it recreates one

    with open(METADATA_FILE, "w") as json_file:
        json.dump(metadata_registry, json_file, indent=2)

    print("=====================================================")
    print("[STATUS] Pipeline Phase FR-2 Dlib HOG Execution Complete!")
    print(f"[STATUS] Total Sources Evaluated: {processed_images_count}")
    print(f"[STATUS] Total HOG Faces Logged: {total_faces_found}")
    print(f"[STATUS] HOG Bounding Box Registry Saved to: '{METADATA_FILE}'")
    print("=====================================================")

if __name__ == "__main__":
    main()
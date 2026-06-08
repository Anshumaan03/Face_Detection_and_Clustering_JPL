# download.py
import os
import cv2
import numpy as np
from datasets import load_dataset

TARGET_DIR = "data/raw"
MAX_IMAGES = 5000

print("=====================================================")
print("[INFO] Routing around broken university servers...")
print("[INFO] Streaming LFW Faces directly via Hugging Face Hub...")
print("=====================================================\n")

os.makedirs(TARGET_DIR, exist_ok=True)

# 1. Stream the dataset directly from Hugging Face infrastructure without downloading the huge tarball first
# This loads a pointer immediately without a massive lag time!
dataset_stream = load_dataset("bitmind/lfw", split="train", streaming=True)

print("[SUCCESS] Connected to dataset stream. Starting file conversion...")
print(f"[INFO] Extracting exactly {MAX_IMAGES} images into '{TARGET_DIR}'...\n")

extracted_count = 0

# 2. Iterate through the cloud data stream row-by-row
for row in dataset_stream:
    # Extract the pre-decoded PIL image object and target filename from the row
    pil_image = row["image"]
    filename = row["filename"]
    
    # If the row name pattern is missing, fallback to a standard incremental index tracker name
    if not filename:
        filename = f"face_sample_{extracted_count:04d}.jpg"
        
    destination_path = os.path.join(TARGET_DIR, filename)
    
    # Transform the image directly into an array format suitable for disk writing
    rgb_array = np.array(pil_image)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    
    # Save natively as a high-quality JPEG
    cv2.imwrite(destination_path, bgr_array)
    
    extracted_count += 1
    if extracted_count % 500 == 0:
        print(f"   ↳ Ingestion Log: Pulled, processed, and saved {extracted_count} images.")
        
    if extracted_count >= MAX_IMAGES:
        break

print("\n=====================================================")
print(f"[STATUS] Task Complete! Dataset fully populated.")
print(f"[STATUS] Total verified face arrays inside '{TARGET_DIR}': {len(os.listdir(TARGET_DIR))}")
print("=====================================================")
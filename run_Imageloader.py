# run_image_loader.py
import os
from src.detection.Image_Loader.Image_loader import ImageLoader

def run_pipeline_ingestion():
    # 1. Ensure raw landing directory structure exists
    os.makedirs("data/raw", exist_ok=True)
    
    # 2. Inject deliberate baseline bad data variations directly into the root 'raw' directory 
    # to confirm the gate defenses filter them safely out.
    with open("data/raw/.DS_Store", "w") as f:
        f.write("mac_os_ui_cache_trash")
        
    with open("data/raw/corrupted_empty_photo.jpg", "w") as f:
        pass # Zero-byte empty file

    # 3. Initialize the core pipeline engine
    loader = ImageLoader(target_dir="data/raw")
    success_count = 0
    
    print("▶️ Starting Image Loader Pipeline Stream...")
    
    # 4. Consume the generator data stream recursively
    for image_packet in loader.load_valid_images():
        success_count += 1
        print(f"   ↳ Ingested: {image_packet['filename']}")
        print(f"   ↳ Identity Label: {image_packet['identity_label']}")
        print(f"   ↳ Data Matrix Shape (H x W x C): {image_packet['pixel_grid'].shape}\n")
        
    print("-" * 50)
    print(f"[STATUS] Safely verified and passed {success_count} total image matrices.")

if __name__ == "__main__":
    run_pipeline_ingestion()
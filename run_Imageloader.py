# run_loader.py
import os
from detection.Image_Loader.Image_loader import ImageLoader

def run_pipeline_ingestion():
   
 # 1. Ensure raw landing directory exists
    os.makedirs("data/raw", exist_ok=True)
    
    # 2. Inject deliberate bad data variations to verify the gate defenses work
    with open("data/raw/.DS_Store", "w") as f:
        f.write("mac_os_ui_cache_trash")
        
    with open("data/raw/corrupted_empty_photo.jpg", "w") as f:
        pass # Zero-byte empty file

    # 3. Initialize the core pipeline engine
    loader = ImageLoader(target_dir="data/raw")
    success_count = 0
    
    # 4. Consume the generator data stream
    for image_packet in loader.load_valid_images():
        success_count += 1
        print(f"   ↳ Ingested: {image_packet['filename']}")
        print(f"   ↳ Data Matrix Shape (H x W x Channels): {image_packet['pixel_grid'].shape}\n")
        
   
    print(f"[STATUS] Safely verified and passed {success_count} total image matrices.")

if __name__ == "__main__":
    run_pipeline_ingestion()
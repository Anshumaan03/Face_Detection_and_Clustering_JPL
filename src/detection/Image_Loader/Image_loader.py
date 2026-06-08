# src/detection/loader.py
import os
import glob
import cv2
import numpy as np
from PIL import Image
import pillow_avif  # Automatically registers AVIF decoding capabilities into Pillow

class ImageLoader:
    def __init__(self, target_dir="data/raw"):
        """
        Initializes the loader with an expanded whitelist including modern web formats.
        """
        self.target_dir = target_dir
        # The complete format whitelist
        self.valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".avif")

    def load_valid_images(self):
        """
        Scans data/raw/, executes the combined gate, adaptively decodes 
        different image compression formats, and yields standard RGB payloads.
        """
        all_files = glob.glob(os.path.join(self.target_dir, "*"))
        
        if not all_files:
            print(f"[WARNING] No files found in '{self.target_dir}'.")
            return

        print(f"[INFO] ImageLoader scanning {len(all_files)} files in '{self.target_dir}'...\n")

        for file_path in all_files:
            filename = os.path.basename(file_path)
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if not os.path.exists(file_path):
                print(f"[SKIP] '{filename}' no longer exists on disk.")
                continue

            file_size_bytes = os.path.getsize(file_path)

            
            # 🔒 COMBINED SECURITY GATE (Checks Identity & Weight)
            
            if (file_extension not in self.valid_extensions) or (file_size_bytes == 0):
                print(f"[SKIP] '{filename}' rejected: Invalid format or file is completely empty.")
                continue

            
            # 🔄 ADAPTIVE DECODING PIPELINE (Unlocks the data matrix)
            
            if file_extension == ".avif":
                try:
                    # Use Pillow + AVIF plugin as the key to open the safe
                    pil_img = Image.open(file_path)
                    # Flatten the visual properties into a clean NumPy pixel grid
                    rgb_image = np.array(pil_img.convert("RGB"))
                except Exception as e:
                    print(f"[SKIP] '{filename}' failed to decode via AVIF subsystem. Error: {e}")
                    continue
            else:
                # Standard routing path for traditional formats (JPG, PNG, WEBP)
                bgr_image = cv2.imread(file_path)
                if bgr_image is None:
                    print(f"[SKIP] '{filename}' failed to decode pixel structure (Corrupted).")
                    continue
                # Fix color channel alignment from BGR to standard RGB
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


            image_payload = {
                "pixel_grid": rgb_image,
                "file_path": file_path,
                "filename": filename
            }

            print(f"[SUCCESS] '{filename}' successfully decoded, normalized, and loaded.")
            yield image_payload
import os
import glob
import cv2
import numpy as np
from PIL import Image
import pillow_avif  # Automatically registers AVIF decoding capabilities into Pillow

class ImageLoader:
    def __init__(self, target_dir="data/raw"):
        """
        Initializes the loader with a recursive scanner configuration.
        """
        self.target_dir = target_dir
        self.valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".avif")

    def load_valid_images(self):
        """
        Recursively walks into subfolders inside data/raw/, executes safety checks,
        extracts the folder name as the identity label, and yields standard RGB payloads.
        """
        # 🔍 RECURSIVE UPGRADE: '**/*' sweeps all subfolders and files deeply
        search_pattern = os.path.join(self.target_dir, "**", "*")
        all_paths = glob.glob(search_pattern, recursive=True)
        
        # Filter out paths that are just directory names, keeping only physical files
        all_files = [p for p in all_paths if os.path.isfile(p)]
        
        if not all_files:
            print(f"[WARNING] No valid physical files found in '{self.target_dir}'.")
            return

        print(f"[INFO] ImageLoader scanning {len(all_files)} total files recursively in '{self.target_dir}'...\n")

        for file_path in all_files:
            filename = os.path.basename(file_path)
            file_extension = os.path.splitext(file_path)[1].lower()
            
            # Extract the folder name directly above the file as the Identity Label
            # Example: data/raw/Virat Kohli/pic1.jpg -> parent folder name is 'Virat Kohli'
            parent_folder = os.path.basename(os.path.dirname(file_path))
            
            # Avoid processing files directly in raw root without an identity subfolder
            if parent_folder.lower() == "raw":
                identity_label = "Unassigned_Outlier"
            else:
                identity_label = parent_folder

            if not os.path.exists(file_path):
                print(f"[SKIP] '{filename}' no longer exists on disk.")
                continue

            file_size_bytes = os.path.getsize(file_path)

            # 🔒 COMBINED SECURITY GATE (Checks Identity & Weight)
            if (file_extension not in self.valid_extensions) or (file_size_bytes == 0):
                print(f"[SKIP] '{filename}' inside '{identity_label}' rejected: Invalid format or empty file.")
                continue

            # 🔄 ADAPTIVE DECODING PIPELINE (Unlocks the data matrix)
            if file_extension == ".avif":
                try:
                    pil_img = Image.open(file_path)
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

            # Package the structural details cleanly for subsequent components
            image_payload = {
                "pixel_grid": rgb_image,
                "file_path": file_path,
                "filename": filename,
                "identity_label": identity_label  # Added to feed MySQL identity assignments
            }

            print(f"[SUCCESS] '{filename}' from identity '{identity_label}' successfully decoded.")
            yield image_payload
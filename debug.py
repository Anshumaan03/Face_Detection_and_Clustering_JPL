import cv2
import json
import os
import numpy as np

# Load your files
json_path = "/Users/anshumaansinghrathore/Desktop/Face Clustering/data/processed_opencv/opencv_detection_metadata.json"
raw_dir = "data/raw"
model_path = "models/lbfmodel.yaml"

if not os.path.exists(json_path) or not os.path.exists(model_path):
    print("[ERROR] Missing JSON metadata or lbfmodel.yaml!")
    exit()

# 1. Initialize engine
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(model_path)

with open(json_path, "r") as f:
    data = json.load(f)

# Grab the very first image entry from your JSON
first_filename = list(data.keys())[0]
face_entry = data[first_filename]
coords = face_entry["bounding_boxes"][0]["coordinates"]

print(f"\n--- TESTING FILE: {first_filename} ---")
print(f"JSON Bounding Box Coordinates: {coords}")

# Load image
img = cv2.imread(os.path.join(raw_dir, first_filename))
if img is None:
    print("Could not load image file!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Format explicitly as an OpenCV-compatible 3D array element
faces_array = np.array([[coords["x"], coords["y"], coords["width"], coords["height"]]], dtype=np.int32)

# 2. Extract Landmarks
success, landmarks_list = facemark.fit(gray, faces_array)

print(f"Facemark Fit Success Flag: {success}")

if success and landmarks_list is not None and len(landmarks_list) > 0:
    points = landmarks_list[0][0]
    print(f"Total Landmark Points Returned: {len(points)}")
    print("\nFirst 5 Landmark Pixel Coordinates:")
    print(points[:5])
    
    # Check for wild out-of-bounds numbers
    h, w = img.shape[:2]
    print(f"\nOriginal Image Resolution: Width={w}, Height={h}")
else:
    print("[FAIL] Model returned no landmarks at all.")
# diagnose_missing.py
import cv2
import os
import glob
import config
from common import load_image, detect_faces

data_root = config.DATA_ROOT

for identity in sorted(os.listdir(data_root)):
    person_dir = os.path.join(data_root, identity)
    if not os.path.isdir(person_dir): continue
    
    paths = []
    for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"):
        paths.extend(glob.glob(os.path.join(person_dir, ext)))
    
    print(f"\n=== {identity} ({len(paths)} images) ===")
    for path in sorted(paths):
        try:
            img = load_image(path)
            faces = detect_faces(img)
            if not faces:
                print(f"  NO DETECTION: {os.path.basename(path)}")
                continue
            
            face = faces[0]
            x1,y1,x2,y2 = face.bbox
            w,h = x2-x1, y2-y1
            
            # check blur
            crop = img[int(y1):int(y2), int(x1):int(x2)]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            status = []
            if face.det_score < config.FACE_DET_MIN_SCORE: 
                status.append(f"LOW_SCORE={face.det_score:.3f}")
            if w < config.FACE_MIN_SIZE or h < config.FACE_MIN_SIZE: 
                status.append(f"SMALL={w:.0f}x{h:.0f}")
            if blur < config.FACE_MIN_BLUR_VAR: 
                status.append(f"BLURRY={blur:.1f}<{config.FACE_MIN_BLUR_VAR}")
            
            if status:
                print(f"  REJECTED ({', '.join(status)}): {os.path.basename(path)}")
            else:
                print(f"  OK (score={face.det_score:.3f}, blur={blur:.1f}): {os.path.basename(path)}")
        except Exception as e:
            print(f"  ERROR: {os.path.basename(path)} — {e}")
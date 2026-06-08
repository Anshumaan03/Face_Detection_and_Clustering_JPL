import cv2
import dlib
import numpy as np
import os

class DlibLandmarkAligner:
    def __init__(self, model_path="models/shape_predictor_68_face_landmarks.dat", target_size=(224, 224)):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[ERROR] Dlib weights file missing at: {model_path}")
        self.predictor = dlib.shape_predictor(model_path)
        self.target_w, self.target_h = target_size

    def compute_alignment(self, bgr_img, dlib_rect):
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        shape = self.predictor(gray, dlib_rect)
        
        coords = np.zeros((68, 2), dtype="int")
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # Defensive check: Ensure landmarks don't fall completely outside image dimensions
        h, w = bgr_img.shape[:2]
        if np.any(coords < 0) or np.any(coords[:, 0] >= w) or np.any(coords[:, 1] >= h):
            # Fallback: return a basic clean resize of the bounding box if landmarks clip out
            x1, y1, x2, y2 = max(0, dlib_rect.left()), max(0, dlib_rect.top()), min(w, dlib_rect.right()), min(h, dlib_rect.bottom())
            crop = bgr_img[y1:y2, x1:x2]
            if crop.size > 0:
                return cv2.resize(crop, (self.target_w, self.target_h)), coords
            return np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8), coords

        left_eye_center = np.mean(coords[36:42], axis=0)
        right_eye_center = np.mean(coords[42:48], axis=0)

        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))

        desired_dist = 0.3 * self.target_w
        current_dist = np.sqrt((dX ** 2) + (dY ** 2))
        scale = desired_dist / current_dist if current_dist != 0 else 1.0

        eyes_center = (float((left_eye_center[0] + right_eye_center[0]) / 2),
                       float((left_eye_center[1] + right_eye_center[1]) / 2))

        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        M[0, 2] += (self.target_w * 0.5 - eyes_center[0])
        M[1, 2] += (self.target_h * 0.35 - eyes_center[1])

        aligned_face = cv2.warpAffine(bgr_img, M, (self.target_w, self.target_h), flags=cv2.INTER_CUBIC)
        return aligned_face, coords
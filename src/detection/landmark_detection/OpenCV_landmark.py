import cv2
import numpy as np
import os

class OpenCVLandmarkAligner:
    def __init__(self, model_path="/Users/anshumaansinghrathore/Desktop/Face Clustering/models/lbfmodel 2.crdownload", target_size=(224, 224)):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[ERROR] OpenCV LBF weights missing at expected path: {model_path}")
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel(model_path)
        self.target_w, self.target_h = target_size

    def compute_alignment(self, bgr_img, opencv_bbox):
        h, w = bgr_img.shape[:2]
        x, y, box_w, box_h = opencv_bbox
        
        # Clip bounding box parameters to ensure they reside strictly within the image boundaries
        x_clean = max(0, min(x, w - 1))
        y_clean = max(0, min(y, h - 1))
        w_clean = max(1, min(box_w, w - x_clean))
        h_clean = max(1, min(box_h, h - y_clean))
        
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        faces_array = np.array([[x_clean, y_clean, w_clean, h_clean]], dtype=np.int32)
        
        success, landmarks_list = self.facemark.fit(gray, faces_array)
        if not success or landmarks_list is None or len(landmarks_list) == 0 or landmarks_list[0] is None:
            # Fallback crop if landmarks fail
            crop = bgr_img[y_clean:y_clean+h_clean, x_clean:x_clean+w_clean]
            if crop.size > 0:
                return cv2.resize(crop, (self.target_w, self.target_h)), None
            return None, None

        coords = landmarks_list[0][0].astype(int)

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
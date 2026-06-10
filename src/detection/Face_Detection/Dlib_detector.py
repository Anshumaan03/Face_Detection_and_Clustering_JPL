# src/detection/dlib_detector.py
import dlib
import cv2

class DlibHOGDetector:
    def __init__(self, upsample_num_times=1):
        """
        Initializes the Dlib HOG + Linear SVM Face Detector.
        
        upsample_num_times: How many times to upscale the image before scanning.
                            Higher numbers detect smaller faces but slow down processing.
        """
        # Load Dlib's built-in frontal face detector (HOG + Linear SVM)
        self.detector = dlib.get_frontal_face_detector()
        self.upsample_num_times = upsample_num_times

    def detect_and_crop(self, image_array):
        """
        Detects faces using HOG features and returns cropped face image matrices
        alongside their bounding box coordinate matrices.
        """
        if image_array is None:
            return [], []

        # Dlib natively processes standard RGB, but OpenCV operates in BGR.
        # Converting ensures Dlib interprets facial structures with optimal color alignment.
        rgb_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Scan the image. Returns a collection of dlib 'rect' objects
        rects = self.detector(rgb_img, self.upsample_num_times)
        
        cropped_faces = []
        bounding_boxes = []
        
        for rect in rects:
            # Extract standard boundary margins from Dlib's rectangle object
            x = rect.left()
            y = rect.top()
            w = rect.right() - rect.left()
            h = rect.bottom() - rect.top()
            
            # Constrain coordinates within original image boundaries to prevent crashing on edge overflows
            img_h, img_w = image_array.shape[:2]
            x_min = max(0, x)
            y_min = max(0, y)
            x_max = min(img_w, x + w)
            y_max = min(img_h, y + h)
            
            # Extract color crop matrix slice
            cropped_face = image_array[y_min:y_max, x_min:x_max]
            
            # Ensure the crop is valid and contains actual pixels before adding
            if cropped_face.size > 0:
                cropped_faces.append(cropped_face)
                bounding_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
                
        return cropped_faces, bounding_boxes
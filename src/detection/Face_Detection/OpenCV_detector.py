# src/detection/detector.py
import cv2
import numpy as np

class HaarFaceDetector:
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        
        #Initializes the Haar Cascade Face Detector.
        # scale_factor: How much the image size is reduced at each image scale.
        # min_neighbors: How many neighbors each candidate rectangle should have to retain it.
        # min_size: Minimum possible object size to detect.

        # Load OpenCV's official pre-trained Frontal Face Haar Cascade XML model
        self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(self.cascade_path)
        
        if self.detector.empty():
            raise IOError("[CRITICAL] Failed to load Haar Cascade XML configuration file.")
            
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

        # src/detection/detector.py
        #keep the rest of the class __init__ the same)

    def detect_and_crop(self, image_array):
        """
        Detects faces and returns both the image matrices and their spatial coordinates.
        """
        if image_array is None:
            return [], []

        gray_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        
        faces = self.detector.detectMultiScale(
            gray_img,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        cropped_faces = []
        # We keep 'faces' because it contains the raw list of [x, y, w, h]
        for (x, y, w, h) in faces:
            cropped_face = image_array[y:y+h, x:x+w]
            cropped_faces.append(cropped_face)
            
        # RETURN BOTH lists at the same time
        return cropped_faces, faces
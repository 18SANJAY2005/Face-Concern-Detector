import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from typing import Tuple, Optional


class FacePreprocessor:
  
    
    def __init__(self):
       
        self.detector = MTCNN()
    
    def detect_and_align_face(self, image_path: str) -> Tuple[np.ndarray, Optional[dict]]:
     
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
   
        results = self.detector.detect_faces(rgb_image)
        
        if not results:
            raise ValueError("No face detected in image")
        

        face = max(results, key=lambda x: x['confidence'])
        bounding_box = face['box']
        keypoints = face['keypoints']
        
  
        x, y, w, h = bounding_box
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(rgb_image.shape[1] - x, w + 2 * padding)
        h = min(rgb_image.shape[0] - y, h + 2 * padding)
        
        face_crop = rgb_image[y:y+h, x:x+w]
        
      
        landmarks = {
            'left_eye': keypoints['left_eye'],
            'right_eye': keypoints['right_eye'],
            'nose': keypoints['nose'],
            'mouth_left': keypoints['mouth_left'],
            'mouth_right': keypoints['mouth_right'],
            'bbox': (x, y, w, h)
        }
        
        return face_crop, landmarks
    
    def preprocess_for_model(self, face_image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
 
        resized = cv2.resize(face_image, target_size)
        
  
        normalized = resized.astype(np.float32) / 255.0
        
  
        if len(normalized.shape) == 3:
            normalized = np.transpose(normalized, (2, 0, 1))
        
        return normalized


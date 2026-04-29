import cv2
from typing import List, Any
from dataclasses import dataclass
from app.services.models import FaceBox

class FaceVisualizer:
    @staticmethod
    def draw_detections(image, faces: List[FaceBox]):
        """Draws a clean, minimalist targeting bracket around faces."""
        
        # Crisp, stark white
        WHITE = (255, 255, 255)
        
        for face in faces:
            x, y, w, h = face.x, face.y, face.w, face.h
            
            # --- 1. MINIMALIST CORNER BRACKETS ---
            # Instead of a full box, just draw the corners (15% of the width)
            cl = max(int(w * 0.15), 10) 
            thick = 2
            
            # Top-Left
            cv2.line(image, (x, y), (x + cl, y), WHITE, thick)
            cv2.line(image, (x, y), (x, y + cl), WHITE, thick)
            
            # Top-Right
            cv2.line(image, (x + w, y), (x + w - cl, y), WHITE, thick)
            cv2.line(image, (x + w, y), (x + w, y + cl), WHITE, thick)
            
            # Bottom-Left
            cv2.line(image, (x, y + h), (x + cl, y + h), WHITE, thick)
            cv2.line(image, (x, y + h), (x, y + h - cl), WHITE, thick)
            
            # Bottom-Right
            cv2.line(image, (x + w, y + h), (x + w - cl, y + h), WHITE, thick)
            cv2.line(image, (x + w, y + h), (x + w, y + h - cl), WHITE, thick)

            # --- 2. SUBTLE CENTER CROSSHAIR ---
            # A tiny, precise "+" in the exact center of the face
            cx, cy = x + w // 2, y + h // 2
            cv2.drawMarker(image, (cx, cy), WHITE, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

        return image
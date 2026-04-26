import os
import cv2
import urllib.request
from dataclasses import dataclass

@dataclass
class FaceDetectionResult:
    success: bool
    face_count: int
    error_message: str = ""

class FaceDetector:
    def __init__(self, min_confidence: float = 0.60, model_selection: int = 0):
        # 60% confidence is the sweet spot for YuNet
        self.min_confidence = min_confidence
        self.model_selection = model_selection 
        
        self.model_path = "face_detection_yunet_2023mar.onnx"
        self._ensure_model_exists()

    def _ensure_model_exists(self):
        """Automatically downloads the tiny 5MB YuNet ONNX model if missing."""
        if not os.path.exists(self.model_path):
            print(f"Downloading OpenCV YuNet model (One-time setup)...")
            # Official OpenCV Zoo repository link
            url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            urllib.request.urlretrieve(url, self.model_path)

    def _rotate_image(self, image, angle):
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def process_image(self, input_path: str, output_path: str) -> FaceDetectionResult:
        original_image = cv2.imread(input_path)

        if original_image is None:
            return FaceDetectionResult(
                success=False,
                face_count=0,
                error_message="Failed to read the uploaded image"
            )

        # 1. SPEED FIX: Pre-scale huge images to 800px max
        # YuNet is fast, but this ensures sub-second performance on giant 4K photos
        h, w = original_image.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            base_image = cv2.resize(original_image, (int(w * scale), int(h * scale)))
        else:
            base_image = original_image.copy()

        valid_boxes = []
        final_image = base_image

        # 2. ROTATION LOOP: Check 0, 90, 180, and 270 degrees
        for angle in [0, 90, 180, 270]:
            rotated_img = self._rotate_image(base_image, angle)
            curr_h, curr_w = rotated_img.shape[:2]
            
            # YuNet requires the image dimensions to be set during initialization
            detector = cv2.FaceDetectorYN.create(
                model=self.model_path,
                config="",
                input_size=(curr_w, curr_h),
                score_threshold=self.min_confidence,
                nms_threshold=0.3,
                top_k=5000
            )
            
            # Run the deep learning detector
            _, faces = detector.detect(rotated_img)
            
            current_valid_boxes = []
            
            if faces is not None:
                for face in faces:
                    # YuNet returns an array: [x, y, w, h, right_eye_x, right_eye_y... score]
                    x, y, w, h = list(map(int, face[:4]))
                    score = face[-1]
                    
                    if score >= self.min_confidence:
                        # Ensure boxes don't break outside image boundaries
                        x, y = max(0, x), max(0, y)
                        w, h = min(curr_w - x, w), min(curr_h - y, h)
                        
                        # Filter out glitchy zero-width/height boxes
                        if w > 0 and h > 0:
                            current_valid_boxes.append((x, y, w, h))

            # If we found faces, keep this rotation and stop checking!
            if current_valid_boxes:
                valid_boxes = current_valid_boxes
                final_image = rotated_img
                break

        if not valid_boxes:
            return FaceDetectionResult(
                success=False,
                face_count=0,
                error_message="No valid human face detected. Please upload a clear human portrait."
            )

        # Draw the bounding boxes on the properly rotated image
        for (x, y, w, h) in valid_boxes:
            cv2.rectangle(final_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Save the successfully rotated and processed image
        if not cv2.imwrite(output_path, final_image):
            return FaceDetectionResult(
                success=False,
                face_count=0,
                error_message="Failed to save processed image"
            )

        return FaceDetectionResult(success=True, face_count=len(valid_boxes))
import os
import cv2
import logging
import urllib.request
import threading
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# --- 1. CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DetectorConfig:
    min_confidence: float = 0.65
    max_image_size: int = 800
    side_view_ratio: float = 2.5
    min_face_area_ratio: float = 0.005  # 0.5% Noise Filter
    model_path: str = "face_detection_yunet_2023mar.onnx"
    model_url: str = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    score: float

@dataclass
class FaceDetectionResult:
    success: bool
    face_count: int
    error_message: str = ""

# --- 2. THE VISUALIZER (Separated Responsibility) ---
class FaceVisualizer:
    @staticmethod
    def draw_detections(image, faces: List[FaceBox]):
        """Pure function to handle drawing."""
        for face in faces:
            cv2.rectangle(image, (face.x, face.y), (face.x + face.w, face.y + face.h), (0, 255, 0), 2)
        return image

# --- 3. THE CORE DETECTOR ---
class FaceDetector:
    _download_lock = threading.Lock()

    def __init__(self, config: DetectorConfig = DetectorConfig()):
        self.cfg = config
        self._ensure_model_exists()

    def _ensure_model_exists(self):
        """Thread-safe download with logging."""
        with self._download_lock:
            if not os.path.exists(self.cfg.model_path):
                logger.info(f"Model missing. Downloading from {self.cfg.model_url}...")
                try:
                    urllib.request.urlretrieve(self.cfg.model_url, self.cfg.model_path)
                    logger.info("Model downloaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to download model: {e}")
                    raise RuntimeError("Could not initialize FaceDetector: Model download failed.")

    def _rotate_image(self, image, angle):
        if angle == 90: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180: return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def _is_side_view(self, face, face_width):
        """Landmark-based symmetry check."""
        re_x, le_x, nose_x = face[4], face[6], face[8]
        dist_r = abs(nose_x - re_x)
        dist_l = abs(nose_x - le_x)
        
        ratio = max(dist_r, dist_l) / (min(dist_r, dist_l) + 1e-6)
        eye_span_ratio = abs(re_x - le_x) / face_width
        
        return ratio > self.cfg.side_view_ratio or eye_span_ratio < 0.25

    def process_image(self, input_path: str, output_path: str) -> FaceDetectionResult:
        """ 
        Orchestration method: 
        Maintains your existing API while using the refactored components.
        """
        try:
            original_image = cv2.imread(input_path)
            if original_image is None:
                return FaceDetectionResult(False, 0, "Failed to read the uploaded image")

            # Performance scaling
            h, w = original_image.shape[:2]
            base_image = original_image
            if max(h, w) > self.cfg.max_image_size:
                scale = self.cfg.max_image_size / max(h, w)
                base_image = cv2.resize(original_image, (int(w * scale), int(h * scale)))

            valid_faces: List[FaceBox] = []
            final_oriented_image = base_image

            # Multi-angle processing
            for angle in [0, 90, 180, 270]:
                rotated_img = self._rotate_image(base_image, angle)
                curr_h, curr_w = rotated_img.shape[:2]
                
                detector = cv2.FaceDetectorYN.create(
                    model=self.cfg.model_path, config="", input_size=(curr_w, curr_h),
                    score_threshold=self.cfg.min_confidence
                )
                
                _, detections = detector.detect(rotated_img)
                
                if detections is not None:
                    current_pass_faces = []
                    for det in detections:
                        fx, fy, fw, fh = list(map(int, det[:4]))
                        score = det[-1]
                        
                        # Apply Filtering Logic
                        area_ratio = (fw * fh) / (curr_w * curr_h)
                        if score >= self.cfg.min_confidence and area_ratio >= self.cfg.min_face_area_ratio:
                            if not self._is_side_view(det, fw):
                                # Boundary protection
                                fx, fy = max(0, fx), max(0, fy)
                                fw, fh = min(curr_w - fx, fw), min(curr_h - fy, fh)
                                current_pass_faces.append(FaceBox(fx, fy, fw, fh, score))

                    if current_pass_faces:
                        valid_faces = current_pass_faces
                        final_oriented_image = rotated_img
                        break

            if not valid_faces:
                return FaceDetectionResult(False, 0, "Please upload a clear, front-facing human portrait.")

            # Separation of Concerns: Delegate Drawing to Visualizer
            final_image = FaceVisualizer.draw_detections(final_oriented_image, valid_faces)

            if not cv2.imwrite(output_path, final_image):
                logger.error(f"Failed to write image to {output_path}")
                return FaceDetectionResult(False, 0, "Internal Server Error: Could not save result.")

            return FaceDetectionResult(True, len(valid_faces))

        except Exception as e:
            logger.exception(f"Unexpected error during face detection: {e}")
            return FaceDetectionResult(False, 0, "An unexpected processing error occurred.")
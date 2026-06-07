import os
import cv2
import logging
import urllib.request
import threading
from dataclasses import dataclass
from typing import List

from app.services.face_visualizer import FaceVisualizer
from app.services.face_detection_result import FaceBox, FaceDetectionResult

# --- 1. CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DetectorConfig:
    min_confidence: float = 0.65
    max_image_size: int = 800
    side_view_ratio: float = 3.2
    min_eye_span_ratio: float = 0.16
    min_face_area_ratio: float = 0.004
    model_path: str = "face_detection_yunet_2023mar.onnx"
    model_url: str = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

# --- 2. THE CORE DETECTOR ---
class FaceDetector:
    _download_lock = threading.Lock()

    def __init__(self, config: DetectorConfig = DetectorConfig()):
        self.cfg = config
        self._ensure_model_exists()
        
        # Load the model into memory exactly ONCE when the server starts
        self.detector = cv2.FaceDetectorYN.create(
            model=self.cfg.model_path, 
            config="", 
            input_size=(320, 320), # Placeholder, updated dynamically per image
            score_threshold=self.cfg.min_confidence
        )

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
        """
        Rejects clear side-view faces while allowing slight tilt.

        Allowed:
        - front-facing faces
        - slightly angled faces
        - slight head tilt

        Rejected:
        - strong side profiles
        - faces where one eye is barely visible
        """

        re_x, le_x, nose_x = face[4], face[6], face[8]

        dist_r = abs(nose_x - re_x)
        dist_l = abs(nose_x - le_x)

        ratio = max(dist_r, dist_l) / (min(dist_r, dist_l) + 1e-6)
        eye_span_ratio = abs(re_x - le_x) / (face_width + 1e-6)

        # Very strong side profile: reject immediately
        if ratio > 4.2:
            return True

        # One eye span is too compressed: likely side view
        if eye_span_ratio < 0.13:
            return True

        # Moderate asymmetry + compressed eyes = side-ish enough to reject
        if ratio > self.cfg.side_view_ratio and eye_span_ratio < self.cfg.min_eye_span_ratio:
            return True

        return False
        
    def _save_face_crop(
        self,
        image,
        faces: List[FaceBox],
        crop_output_path: str,
        padding_ratio: float = 0.20
    ):
        """
        Saves a clean, unannotated crop of the dominant face region.

        If two dominant faces are present, it crops the union of both faces.
        This keeps training/inference focused on the face area instead of
        background, clothing, and scene composition.
        """

        if not crop_output_path or not faces:
            return None

        img_h, img_w = image.shape[:2]

        x1 = min(face.x for face in faces)
        y1 = min(face.y for face in faces)
        x2 = max(face.x + face.w for face in faces)
        y2 = max(face.y + face.h for face in faces)

        crop_w = x2 - x1
        crop_h = y2 - y1

        padding = int(padding_ratio * max(crop_w, crop_h))

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_w, x2 + padding)
        y2 = min(img_h, y2 + padding)

        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None

        if not cv2.imwrite(crop_output_path, face_crop):
            logger.error(f"Failed to write face crop to {crop_output_path}")
            return None

        return crop_output_path
    
    def process_image(
    self,
    input_path: str,
    output_path: str,
    crop_output_path: str = None
    ) -> FaceDetectionResult:
        
        """ 
        Orchestration method: Maintains existing API while using refactored components.
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

            # Local warning string for thread safety
            final_warning_msg = ""

            # Multi-angle processing — score ALL rotations, pick the best one.
            # YuNet returns noticeably lower confidence on upside-down/sideways faces,
            # so summing confidence across detections reliably identifies the correct orientation.
            best_score = -1.0
            best_faces: List[FaceBox] = []
            best_image = base_image

            for angle in [0, 90, 180, 270]:
                rotated_img = self._rotate_image(base_image, angle)
                curr_h, curr_w = rotated_img.shape[:2]

                # Dynamically update the input size for the pre-loaded model
                self.detector.setInputSize((curr_w, curr_h))
                _, detections = self.detector.detect(rotated_img)

                if detections is None:
                    continue

                current_pass_faces = []
                for det in detections:
                    fx, fy, fw, fh = list(map(int, det[:4]))
                    score = det[-1]

                    area_ratio = (fw * fh) / (curr_w * curr_h)
                    if score >= self.cfg.min_confidence and area_ratio >= self.cfg.min_face_area_ratio:
                        if not self._is_side_view(det, fw):
                            fx, fy = max(0, fx), max(0, fy)
                            fw, fh = min(curr_w - fx, fw), min(curr_h - fy, fh)
                            current_pass_faces.append(FaceBox(fx, fy, fw, fh, score))

                if not current_pass_faces:
                    continue

                # Sum of confidences = orientation score (higher = more correct orientation)
                orientation_score = sum(f.score for f in current_pass_faces)
                if orientation_score > best_score:
                    best_score = orientation_score
                    best_faces = current_pass_faces
                    best_image = rotated_img

            if best_faces:
                current_pass_faces = best_faces
                final_oriented_image = best_image

                # --- THE HEURISTIC AMBIGUITY ENGINE (DUO SUPPORT) ---

                # 1. Sort all found faces by area (largest first)
                current_pass_faces.sort(key=lambda f: f.w * f.h, reverse=True)

                # The largest face is always our baseline subject
                valid_subjects = [current_pass_faces[0]]
                main_area = current_pass_faces[0].w * current_pass_faces[0].h

                # 2. Check all other detected faces against the baseline area
                for face in current_pass_faces[1:]:
                    face_area = face.w * face.h
                    ratio = face_area / main_area

                    # If a background face is at least 30% of the area of the main face,
                    # it is considered a "Primary Subject", not just background noise.
                    if ratio >= 0.30:
                        valid_subjects.append(face)

                # 3. Enforce the Strict Security Thresholds
                if len(valid_subjects) > 2:
                    # REJECT: It's a crowd photo or chaotic background
                    return FaceDetectionResult(
                        False,
                        0,
                        "Image rejected: TrueImage supports one or two dominant faces. Small background face-like detections are ignored. Images with more than two dominant faces are rejected to avoid ambiguous analysis."
                    )

                # 4. Handle Messaging & Assignment
                if len(valid_subjects) == 2:
                    final_warning_msg = "Dual subjects detected. TrueImage will analyze both."
                elif len(current_pass_faces) > len(valid_subjects):
                    # It found tiny background faces and successfully ignored them
                    final_warning_msg = "Background artifacts detected. TrueImage isolated the primary subject for analysis."

                # Override the pass list with ONLY our valid, dominant subjects
                valid_faces = valid_subjects

            if not valid_faces:
                return FaceDetectionResult(False, 0, "Please upload a clear, front-facing human portrait.")

            # Save a clean face crop for AI model inference/training.
            # This is saved BEFORE drawing annotations so the model never sees green boxes.
            saved_crop_path = None

            if crop_output_path:
                saved_crop_path = self._save_face_crop(
                    image=final_oriented_image,
                    faces=valid_faces,
                    crop_output_path=crop_output_path,
                    padding_ratio=0.20
                )

            # Separation of Concerns: Delegate Drawing to Visualizer
            final_image = FaceVisualizer.draw_detections(final_oriented_image, valid_faces)

            if not cv2.imwrite(output_path, final_image):
                logger.error(f"Failed to write image to {output_path}")
                return FaceDetectionResult(False, 0, "Internal Server Error: Could not save result.")

            return FaceDetectionResult(
                success=True,
                face_count=len(valid_faces),
                error_message="",
                warning_message=final_warning_msg,
                faces=valid_faces,
                crop_path=saved_crop_path
            )
        except Exception as e:
            logger.exception(f"Unexpected error during face detection: {e}")
            return FaceDetectionResult(False, 0, "An unexpected processing error occurred.")
import os
import cv2
import logging
import urllib.request
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from app.services.face_visualizer import FaceVisualizer
from app.services.face_detection_result import FaceBox, FaceDetectionResult

# --- 1. CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectorConfig:
    # Bumped from 0.65 -> 0.70: works together with the geometry check below
    # rather than being the sole line of defense against false positives.
    min_confidence: float = 0.70
    max_image_size: int = 800
    side_view_ratio: float = 4.2
    min_eye_span_ratio: float = 0.12

    # Bumped from 0.004 -> 0.01: cuts out a lot of small-object false
    # positives (doorknobs, textures, distant background clutter) while still
    # allowing legitimate faces at a normal distance from camera.
    min_face_area_ratio: float = 0.01

    model_path: str = "face_detection_yunet_2023mar.onnx"
    model_url: str = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

    zero_degree_bonus: float = 0.10  # bias toward 0° to avoid micro-fluctuation flips

    # A rotated candidate must beat plain detection's own bar by at least this
    # much extra margin to be trusted — not an arbitrary high absolute score.
    # This distinguishes "a real face that just doesn't score amazingly at a
    # rotated angle" (small margin needed) from "a coincidental pattern match
    # on a rotated background" (usually sits right near min_confidence, with
    # near-zero margin above it).
    rotation_search_confidence_margin: float = 0.05

    # Geometry-plausibility bounds for landmark-based face verification.
    # See _has_plausible_face_geometry for what each guards against.
    min_eye_span_geometry_ratio: float = 0.25
    max_eye_span_geometry_ratio: float = 0.75
    min_nose_position_ratio: float = 0.25
    max_nose_position_ratio: float = 0.85


# `detect_fn` is injected into every strategy so detection logic lives in
# exactly one place (FaceDetector._detect_valid_faces), not duplicated per
# strategy. Each strategy returns (image, faces) if confident, else None to
# let the next strategy in the chain try.
DetectFn = Callable[["cv2.Mat"], List[FaceBox]]


# --- 2. ORIENTATION RESOLUTION (pluggable chain) ---
class OrientationStrategy(ABC):
    @abstractmethod
    def resolve(self, image, detect_fn: DetectFn) -> Optional[Tuple[object, List[FaceBox]]]:
        ...


class ExifOrientationResolver(OrientationStrategy):
    """
    First, cheapest link in the chain. cv2.imread() already applies EXIF
    orientation at load time, so `image` here has already been corrected
    if an EXIF tag existed. This strategy just checks: is a valid, upright
    face already detectable with no rotation search needed? If so, most
    camera-native uploads resolve here and skip the search entirely.
    """
    def resolve(self, image, detect_fn: DetectFn):
        faces = detect_fn(image)
        if faces:
            return image, faces
        return None


class FallbackRotationSearch(OrientationStrategy):
    """
    Handles the common real-world case: EXIF-stripped images (screenshots,
    re-compressed uploads, forwarded photos) where there's no metadata to
    rely on. Tries all 4 rotations and keeps whichever produces the
    best-scoring valid (non-side-view, geometrically-plausible) face.

    Accepting a rotation is a *stronger claim* than "the image is already
    fine", so the winning angle must clear base_confidence by an extra
    margin — not just hit an arbitrary absolute score. A genuine rotated
    face that scores a bit lower than usual (rotation interpolation costs
    some confidence) still clears this; a coincidental pattern match on a
    rotated background usually sits right at the floor with near-zero
    margin above it.
    """
    def __init__(self, rotate_fn: Callable[[object, int], object], base_confidence: float,
                 confidence_margin: float, zero_degree_bonus: float = 0.10):
        self._rotate = rotate_fn
        self._min_accept_confidence = base_confidence + confidence_margin
        self._zero_bonus = zero_degree_bonus

    def resolve(self, image, detect_fn: DetectFn):
        best_faces: List[FaceBox] = []
        best_image = image
        best_score = -1.0

        for angle in (0, 90, 180, 270):
            candidate = self._rotate(image, angle)
            faces = detect_fn(candidate)
            if not faces:
                continue

            top_score = max(f.score for f in faces)
            angle_score = top_score + (self._zero_bonus if angle == 0 else 0.0)

            if angle_score > best_score:
                best_score = angle_score
                best_faces = faces
                best_image = candidate

        if not best_faces or best_score < self._min_accept_confidence:
            return None

        return best_image, best_faces


class OrientationResolver:
    """
    Runs strategies in order, returning the first confident result.
    To add a smarter/lighter rotation classifier down the line, write a new
    OrientationStrategy and insert it into this list — FaceDetector doesn't
    need to change at all.
    """
    def __init__(self, strategies: List[OrientationStrategy]):
        self._strategies = strategies

    def resolve(self, image, detect_fn: DetectFn) -> Tuple[object, List[FaceBox]]:
        for strategy in self._strategies:
            result = strategy.resolve(image, detect_fn)
            if result is not None:
                return result
        return image, []


# --- 3. THE CORE DETECTOR ---
class FaceDetector:
    _download_lock = threading.Lock()

    def __init__(self, config: DetectorConfig = DetectorConfig()):
        self.cfg = config
        self._ensure_model_exists()

        self.detector = cv2.FaceDetectorYN.create(
            model=self.cfg.model_path,
            config="",
            input_size=(320, 320),
            score_threshold=self.cfg.min_confidence
        )

        self.orientation_resolver = OrientationResolver([
            ExifOrientationResolver(),
            FallbackRotationSearch(
                rotate_fn=self._rotate_image,
                base_confidence=self.cfg.min_confidence,
                confidence_margin=self.cfg.rotation_search_confidence_margin,
                zero_degree_bonus=self.cfg.zero_degree_bonus,
            ),
        ])

    def _ensure_model_exists(self):
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
        re_x, le_x, nose_x = face[4], face[6], face[8]
        dist_r = abs(nose_x - re_x)
        dist_l = abs(nose_x - le_x)
        ratio = max(dist_r, dist_l) / (min(dist_r, dist_l) + 1e-6)
        eye_span_ratio = abs(re_x - le_x) / (face_width + 1e-6)

        if ratio > 8.0 and eye_span_ratio < 0.04:
            return True
        if ratio > 6.0 and eye_span_ratio < 0.08:
            return True
        return False

    def _has_plausible_face_geometry(self, face, face_width, face_height):
        """
        YuNet returns landmarks for every detection above min_confidence, real
        or not — pareidolia (outlets, dolls, textures, some cartoon/animal
        faces) can score high enough to pass the confidence/area checks
        alone. This verifies the landmarks actually describe a human face
        layout, rather than trusting the score in isolation.
        """
        re_x, re_y, le_x, le_y = face[4], face[5], face[6], face[7]
        nose_x, nose_y = face[8], face[9]
        mouth_r_y, mouth_l_y = face[11], face[13]

        # Eyes above nose, nose above mouth (image y grows downward)
        eyes_y = (re_y + le_y) / 2.0
        mouth_y = (mouth_r_y + mouth_l_y) / 2.0
        if not (eyes_y < nose_y < mouth_y):
            return False

        # Interocular distance should be a plausible fraction of face width
        # for a real human face — not near-zero (collapsed) and not
        # implausibly wide.
        eye_span_ratio = abs(re_x - le_x) / (face_width + 1e-6)
        if not (self.cfg.min_eye_span_geometry_ratio <= eye_span_ratio <= self.cfg.max_eye_span_geometry_ratio):
            return False

        # Nose should sit at a plausible point between eyes and mouth
        # vertically, not squashed near either end.
        face_span = mouth_y - eyes_y
        if face_span <= 0:
            return False
        nose_position = (nose_y - eyes_y) / face_span
        if not (self.cfg.min_nose_position_ratio <= nose_position <= self.cfg.max_nose_position_ratio):
            return False

        return True

    def _detect_valid_faces(self, image) -> List[FaceBox]:
        """
        The single place raw YuNet detection happens. Every OrientationStrategy
        calls this via the injected `detect_fn` so there's no duplicated
        detect/filter logic between the EXIF check and the rotation search.
        """
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, detections = self.detector.detect(image)
        if detections is None:
            return []

        faces: List[FaceBox] = []
        for det in detections:
            fx, fy, fw, fh = list(map(int, det[:4]))
            score = det[-1]
            area_ratio = (fw * fh) / (w * h)

            if score >= self.cfg.min_confidence and area_ratio >= self.cfg.min_face_area_ratio:
                if not self._is_side_view(det, fw) and self._has_plausible_face_geometry(det, fw, fh):
                    fx, fy = max(0, fx), max(0, fy)
                    fw, fh = min(w - fx, fw), min(h - fy, fh)
                    faces.append(FaceBox(fx, fy, fw, fh, score))
        return faces

    def _save_face_crop(self, image, faces: List[FaceBox], crop_output_path: str, padding_ratio: float = 0.20):
        if not crop_output_path or not faces: return None

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

        if face_crop.size == 0: return None

        if not cv2.imwrite(crop_output_path, face_crop):
            logger.error(f"Failed to write face crop to {crop_output_path}")
            return None

        return crop_output_path

    def process_image(self, input_path: str, output_path: str, crop_output_path: str = None) -> FaceDetectionResult:
        try:
            # OpenCV applies EXIF orientation natively at load time by default.
            original_image = cv2.imread(input_path)
            if original_image is None:
                return FaceDetectionResult(False, 0, "Failed to read the uploaded image")

            h, w = original_image.shape[:2]
            base_image = original_image
            if max(h, w) > self.cfg.max_image_size:
                scale = self.cfg.max_image_size / max(h, w)
                base_image = cv2.resize(original_image, (int(w * scale), int(h * scale)))

            # Orientation resolution + detection happen together: whichever
            # strategy succeeds returns the faces it already found, so there's
            # no separate re-detection step afterward.
            final_oriented_image, current_pass_faces = self.orientation_resolver.resolve(
                base_image, detect_fn=self._detect_valid_faces
            )

            valid_faces: List[FaceBox] = []
            final_warning_msg = ""

            if current_pass_faces:
                # --- AMBIGUITY ENGINE (DUO SUPPORT) ---
                current_pass_faces.sort(key=lambda f: f.w * f.h, reverse=True)
                valid_subjects = [current_pass_faces[0]]
                main_area = current_pass_faces[0].w * current_pass_faces[0].h

                for face in current_pass_faces[1:]:
                    face_area = face.w * face.h
                    if (face_area / main_area) >= 0.30:
                        valid_subjects.append(face)

                if len(valid_subjects) > 2:
                    return FaceDetectionResult(
                        False, 0,
                        "Image rejected: TrueImage supports one or two dominant faces. Small background face-like detections are ignored. Images with more than two dominant faces are rejected to avoid ambiguous analysis."
                    )

                if len(valid_subjects) == 2:
                    final_warning_msg = "Dual subjects detected. TrueImage will analyze both."
                elif len(current_pass_faces) > len(valid_subjects):
                    final_warning_msg = "Background artifacts detected. TrueImage isolated the primary subject for analysis."

                valid_faces = valid_subjects

            if not valid_faces:
                return FaceDetectionResult(False, 0, "Please upload a clear, front-facing human portrait.")

            saved_crop_path = None
            if crop_output_path:
                saved_crop_path = self._save_face_crop(
                    image=final_oriented_image,
                    faces=valid_faces,
                    crop_output_path=crop_output_path,
                    padding_ratio=0.20
                )

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
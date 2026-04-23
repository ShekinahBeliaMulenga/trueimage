import cv2
import mediapipe as mp
from dataclasses import dataclass

@dataclass
class FaceDetectionResult:
    """Data Transfer Object for face detection outcomes."""
    success: bool
    face_count: int
    error_message: str = ""

class FaceDetector:
    """
    Service for detecting faces in an image using MediaPipe, 
    handling orientation, and drawing bounding boxes.
    """
    def __init__(self, min_confidence: float = 0.6, model_selection: int = 1):
        self.min_confidence = min_confidence
        self.model_selection = model_selection
        self.mp_face_detection = mp.solutions.face_detection

    def _rotate_image(self, image, angle):
        """Rotate image using OpenCV to handle incorrect EXIF orientations."""
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def process_image(self, input_path: str, output_path: str) -> FaceDetectionResult:
        """
        Reads, detects, annotates, and saves the image.
        Returns a FaceDetectionResult.
        """
        image = cv2.imread(input_path)
        if image is None:
            return FaceDetectionResult(
                success=False, face_count=0, error_message="Failed to read the uploaded image"
            )

        face_count = 0
        detected_results = None
        final_image = image

        # Initialize MediaPipe detector
        with self.mp_face_detection.FaceDetection(
            model_selection=self.model_selection,
            min_detection_confidence=self.min_confidence
        ) as detector:
            
            # Attempt detection at 4 different rotations
            for angle in [0, 90, 180, 270]:
                rotated_img = self._rotate_image(image, angle)
                rgb_image = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb_image)

                if results.detections:
                    detected_results = results.detections
                    final_image = rotated_img
                    break

        # If no faces were found after all rotations
        if not detected_results:
            return FaceDetectionResult(
                success=False, 
                face_count=0, 
                error_message="No face detected. Please upload a clear, front-facing portrait."
            )

        # Draw bounding boxes
        height, width, _ = final_image.shape
        for detection in detected_results:
            face_count += 1
            bbox = detection.location_data.relative_bounding_box

            x = max(0, int(bbox.xmin * width))
            y = max(0, int(bbox.ymin * height))
            w = int(bbox.width * width)
            h = int(bbox.height * height)
            x2 = min(width, x + w)
            y2 = min(height, y + h)

            cv2.rectangle(final_image, (x, y), (x2, y2), (0, 255, 0), 2)

        # Save annotated image
        if not cv2.imwrite(output_path, final_image):
            return FaceDetectionResult(
                success=False, face_count=0, error_message="Failed to save processed image"
            )

        return FaceDetectionResult(success=True, face_count=face_count)
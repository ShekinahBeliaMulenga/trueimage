from dataclasses import dataclass, field
from typing import List, Optional


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
    warning_message: str = ""
    faces: List[FaceBox] = field(default_factory=list)
    crop_path: Optional[str] = None
    # Origin (top-left) of the saved crop, in the ORIGINAL (pre-crop) image's
    # pixel coordinates. Needed to translate `faces` (also in original-image
    # coordinates) into crop-relative coordinates when annotating the crop
    # directly, without re-running detection on it. Both default to 0, which
    # is harmless when crop_path is None (there is nothing to translate).
    crop_offset_x: int = 0
    crop_offset_y: int = 0

    @property
    def bounding_box(self):
        """
        Returns the first dominant face box as (x, y, w, h).
        Kept for simple controller usage.
        """
        if not self.faces:
            return None

        face = self.faces[0]
        return face.x, face.y, face.w, face.h
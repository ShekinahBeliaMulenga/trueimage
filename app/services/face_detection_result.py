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
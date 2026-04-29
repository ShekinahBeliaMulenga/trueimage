from dataclasses import dataclass

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
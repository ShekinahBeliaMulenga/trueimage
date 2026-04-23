from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

from nudenet import NudeDetector


@dataclass
class ExplicitDetectionResult:
    verdict: str
    confidence_score: float
    message: str
    detections: List[Dict[str, Any]]


class ExplicitDetector:
    """
    Service for screening uploaded images for explicit or suggestive content.

    Design goals:
    - Keep model-specific labels internal
    - Expose only professional, user-friendly categories
    - Return a structured result suitable for controller integration
    """

    # Internal detector labels mapped into broad moderation groups.
    # These are intentionally kept private to avoid exposing raw labels to users.
    _EXPLICIT_LABELS = {
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_EXPOSED",
        "BUTTOCKS_EXPOSED",
    }

    _SUGGESTIVE_LABELS = {
        "FEMALE_BREAST_COVERED",
        "BUTTOCKS_COVERED",
        "BELLY_EXPOSED",
        "ARMPITS_EXPOSED",
        "FEET_EXPOSED",
    }

    def __init__(self, threshold: float = 0.50) -> None:
        """
        :param threshold: minimum confidence required for a detector output
                          to be considered during moderation.
        """
        self.threshold = threshold
        self.detector = NudeDetector()

    def predict(self, image_path: str) -> ExplicitDetectionResult:
        """
        Analyze an image and return a professional moderation result.

        Returns:
            ExplicitDetectionResult with:
            - verdict: SAFE / SUGGESTIVE / EXPLICIT
            - confidence_score: aggregated confidence for the detected category
            - message: clean user-facing message
            - detections: filtered raw detections for backend/internal use
        """
        raw_detections = self.detector.detect(image_path)

        filtered = [
            det for det in raw_detections
            if float(det.get("score", 0.0)) >= self.threshold
        ]

        explicit_scores: List[float] = []
        suggestive_scores: List[float] = []

        for det in filtered:
            label = det.get("class", "")
            score = float(det.get("score", 0.0))

            if label in self._EXPLICIT_LABELS:
                explicit_scores.append(score)
            elif label in self._SUGGESTIVE_LABELS:
                suggestive_scores.append(score)

        explicit_score = max(explicit_scores) if explicit_scores else 0.0
        suggestive_score = max(suggestive_scores) if suggestive_scores else 0.0

        if explicit_score >= 0.70:
            verdict = "EXPLICIT"
            confidence_score = explicit_score
            message = "Upload rejected: Image violates safety and content guidelines. Standard portraits only"
        elif explicit_score >= 0.40 or suggestive_score >= 0.60:
            verdict = "SUGGESTIVE"
            confidence_score = max(explicit_score, suggestive_score)
            message = "Upload flagged: Image contains restricted content. Please provide a clear, standard portrait"
        else:
            verdict = "SAFE"
            confidence_score = max(explicit_score, suggestive_score)
            message = "Safety scan complete. Content cleared for forensic analysis."

        return ExplicitDetectionResult(
            verdict=verdict,
            confidence_score=round(confidence_score, 4),
            message=message,
            detections=filtered
        )
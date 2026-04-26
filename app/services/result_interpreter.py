from dataclasses import dataclass


@dataclass
class Interpretation:
    label: str
    user_message: str
    confidence_score: float
    ai_probability: float


class ResultInterpreter:
    @staticmethod
    def analyze(probability: float) -> Interpretation:
        ai_probability = round(probability * 100, 1)

        if ai_probability <= 40:
            label = "REAL"
            user_message = "Neural scan complete. The image appears to be a real photograph."
            confidence_score = round((1 - probability) * 100, 1)

        elif ai_probability < 75:
            label = "UNCERTAIN"
            user_message = "The analysis is inconclusive. Manual review is recommended."
            confidence_score = ai_probability

        else:
            label = "AI-GENERATED"
            user_message = "Neural scan complete. Synthetic characteristics were detected in the image."
            confidence_score = ai_probability

        return Interpretation(
            label=label,
            user_message=user_message,
            confidence_score=confidence_score,
            ai_probability=ai_probability
        )
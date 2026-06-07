import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


@dataclass
class AIInferenceOutput:
    success: bool
    ai_probability: Optional[float] = None
    message: str = ""


class AIInferenceEngine:
    """
    Loads the trained TRUEIMAGE Keras model and returns the raw AI probability.

    Important:
    The model was trained with EfficientNetV2 include_preprocessing=True.
    Therefore, we feed RGB image pixels in normal 0-255 range.
    Do NOT divide by 255 here.
    """

    def __init__(self, model_path: str, image_size: tuple[int, int] = (224, 224)):
        self.model_path = model_path
        self.image_size = image_size
        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"AI model file not found: {self.model_path}")

        return tf.keras.models.load_model(self.model_path)

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            image = image.resize(self.image_size)

            image_array = np.array(image).astype("float32")

            # Shape becomes: (1, 224, 224, 3)
            image_array = np.expand_dims(image_array, axis=0)

            return image_array

    def predict_probability(self, image_path: str) -> AIInferenceOutput:
        try:
            image_batch = self._preprocess_image(image_path)

            prediction = self.model.predict(image_batch, verbose=0)

            ai_probability = float(prediction[0][0])
            ai_probability = max(0.0, min(1.0, ai_probability))

            return AIInferenceOutput(
                success=True,
                ai_probability=ai_probability,
                message="AI inference completed successfully."
            )

        except Exception as error:
            return AIInferenceOutput(
                success=False,
                message=f"AI inference failed: {str(error)}"
            )
import numpy as np
from PIL import Image

class ImageProcessor:

    def load_image(self, path):
        try:
            return Image.open(path)
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")

    def preprocess(self, image_path):
        try:
            # Load and convert to RGB
            image = Image.open(image_path).convert("RGB")

            # Resize to EfficientNetV2-S input size
            image = image.resize((224, 224))

            # Convert to numpy array and normalize to [0,1]
            image = np.array(image).astype(np.float32) / 255.0

            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            image = (image - mean) / std

            # Add batch dimension
            image = np.expand_dims(image, axis=0)

            return image

        except Exception as e:
            raise ValueError(f"Preprocessing failed: {str(e)}")
from pathlib import Path
import random
from PIL import Image, ImageOps, ImageEnhance, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parent.parent

AI_RAW_DIR = PROJECT_ROOT / "training" / "dataset" / "raw" / "ai_generated"
REAL_RAW_DIR = PROJECT_ROOT / "training" / "dataset" / "raw" / "real"

GEMINI_PREFIX = "gemini_"

GEMINI_AUGMENTATIONS_PER_IMAGE = 15
REAL_AUGMENTATION_COUNT = 150

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS


def open_rgb(path: Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def random_augment(image: Image.Image) -> Image.Image:
    """
    Creates a mild augmentation.
    We avoid extreme changes because the model should still see realistic faces.
    """

    # Random horizontal flip
    if random.random() < 0.5:
        image = ImageOps.mirror(image)

    # Slight brightness change
    brightness = random.uniform(0.85, 1.15)
    image = ImageEnhance.Brightness(image).enhance(brightness)

    # Slight contrast change
    contrast = random.uniform(0.85, 1.15)
    image = ImageEnhance.Contrast(image).enhance(contrast)

    # Slight color/saturation change
    color = random.uniform(0.90, 1.10)
    image = ImageEnhance.Color(image).enhance(color)

    # Slight sharpness variation
    sharpness = random.uniform(0.90, 1.20)
    image = ImageEnhance.Sharpness(image).enhance(sharpness)

    # Occasional tiny blur
    if random.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.6)))

    return image


def get_gemini_images():
    return [
        path for path in AI_RAW_DIR.iterdir()
        if is_image(path) and path.name.lower().startswith(GEMINI_PREFIX)
    ]


def get_real_images():
    return [
        path for path in REAL_RAW_DIR.iterdir()
        if is_image(path)
    ]


def augment_gemini_images():
    gemini_images = get_gemini_images()

    if not gemini_images:
        raise ValueError(
            f"No Gemini images found. Rename them to start with '{GEMINI_PREFIX}' inside {AI_RAW_DIR}"
        )

    print(f"Found Gemini seed images: {len(gemini_images)}")

    created = 0

    for source_path in gemini_images:
        base_image = open_rgb(source_path)

        for i in range(GEMINI_AUGMENTATIONS_PER_IMAGE):
            augmented = random_augment(base_image)

            output_name = f"gemini_aug_{source_path.stem}_{i + 1:03d}.jpg"
            output_path = AI_RAW_DIR / output_name

            if output_path.exists():
                continue

            augmented.save(output_path, format="JPEG", quality=95)
            created += 1

    print(f"Created Gemini AI augmentations: {created}")


def augment_real_images():
    real_images = get_real_images()

    if not real_images:
        raise ValueError(f"No real images found in {REAL_RAW_DIR}")

    selected = random.sample(real_images, min(REAL_AUGMENTATION_COUNT, len(real_images)))

    created = 0

    for index, source_path in enumerate(selected, start=1):
        base_image = open_rgb(source_path)
        augmented = random_augment(base_image)

        output_name = f"real_aug_balance_{index:04d}.jpg"
        output_path = REAL_RAW_DIR / output_name

        if output_path.exists():
            continue

        augmented.save(output_path, format="JPEG", quality=95)
        created += 1

    print(f"Created real balancing augmentations: {created}")


def count_images(folder: Path) -> int:
    return len([path for path in folder.iterdir() if is_image(path)])


def main():
    print("\nTRUEIMAGE Gemini Hard-Example Augmentation")
    print("------------------------------------------")

    before_ai = count_images(AI_RAW_DIR)
    before_real = count_images(REAL_RAW_DIR)

    print(f"Before real count: {before_real}")
    print(f"Before AI count  : {before_ai}")

    augment_gemini_images()
    augment_real_images()

    after_ai = count_images(AI_RAW_DIR)
    after_real = count_images(REAL_RAW_DIR)

    print("\nFinal raw dataset counts")
    print("------------------------")
    print(f"Real images        : {after_real}")
    print(f"AI-generated images: {after_ai}")


if __name__ == "__main__":
    
    main()
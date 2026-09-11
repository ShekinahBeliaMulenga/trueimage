from pathlib import Path
import random

from PIL import Image, ImageOps, ImageEnhance, ImageFilter


# =====================================================
# TRUEIMAGE HARD-EXAMPLE AI AUGMENTATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AI_RAW_DIR = PROJECT_ROOT / "training" / "dataset" / "raw" / "ai_generated"

GEMINI_PREFIX = "gemini_"
CHATGPT_PREFIX = "portrait_"

GEMINI_AUGMENTATIONS_PER_IMAGE = 15
CHATGPT_AUGMENTATIONS_PER_IMAGE = 15

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

RANDOM_SEED = 42


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def is_image(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in ALLOWED_EXTENSIONS
    )


def open_rgb(path: Path) -> Image.Image:
    """
    Opens an image, corrects EXIF orientation,
    and converts it to RGB.
    """

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


# =====================================================
# AUGMENTATION
# =====================================================

def random_augment(image: Image.Image) -> Image.Image:
    """
    Creates a mild realistic augmentation.

    The purpose is to increase variation while preserving
    the visual characteristics of AI-generated faces.
    """

    # Horizontal flip
    if random.random() < 0.5:
        image = ImageOps.mirror(image)

    # Slight brightness variation
    brightness = random.uniform(0.85, 1.15)
    image = ImageEnhance.Brightness(image).enhance(brightness)

    # Slight contrast variation
    contrast = random.uniform(0.85, 1.15)
    image = ImageEnhance.Contrast(image).enhance(contrast)

    # Slight saturation variation
    color = random.uniform(0.90, 1.10)
    image = ImageEnhance.Color(image).enhance(color)

    # Slight sharpness variation
    sharpness = random.uniform(0.90, 1.20)
    image = ImageEnhance.Sharpness(image).enhance(sharpness)

    # Occasional small blur
    if random.random() < 0.25:
        image = image.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(0.2, 0.6)
            )
        )

    return image


# =====================================================
# FIND HARD EXAMPLES
# =====================================================

def get_gemini_images():
    """
    Finds original Gemini images only.
    Augmented Gemini files are deliberately excluded.
    """

    return sorted(
        [
            path
            for path in AI_RAW_DIR.iterdir()
            if is_image(path)
            and path.name.lower().startswith(GEMINI_PREFIX)
            and not path.name.lower().startswith("gemini_aug_")
        ]
    )


def get_chatgpt_images():
    """
    Finds original ChatGPT images only.

    Your ChatGPT originals are expected to use:
        portrait_x
    """

    return sorted(
        [
            path
            for path in AI_RAW_DIR.iterdir()
            if is_image(path)
            and path.name.lower().startswith(CHATGPT_PREFIX)
            and not path.name.lower().startswith("chatgpt_aug_")
        ]
    )


# =====================================================
# AUGMENT GEMINI
# =====================================================

def augment_gemini_images():
    gemini_images = get_gemini_images()

    if not gemini_images:
        raise ValueError(
            f"No Gemini images found in:\n{AI_RAW_DIR}\n\n"
            f"Expected filenames beginning with '{GEMINI_PREFIX}'."
        )

    print(f"Gemini seed images found: {len(gemini_images)}")

    created = 0

    for source_path in gemini_images:

        base_image = open_rgb(source_path)

        for i in range(1, GEMINI_AUGMENTATIONS_PER_IMAGE + 1):

            output_name = (
                f"gemini_aug_"
                f"{source_path.stem}_"
                f"{i:03d}.jpg"
            )

            output_path = AI_RAW_DIR / output_name

            # Prevent duplicates if script is run again
            if output_path.exists():
                continue

            augmented = random_augment(base_image)

            augmented.save(
                output_path,
                format="JPEG",
                quality=95
            )

            created += 1

    print(f"Gemini augmentations created: {created}")


# =====================================================
# AUGMENT CHATGPT
# =====================================================

def augment_chatgpt_images():
    chatgpt_images = get_chatgpt_images()

    if not chatgpt_images:
        raise ValueError(
            f"No ChatGPT images found in:\n{AI_RAW_DIR}\n\n"
            f"Expected filenames beginning with '{CHATGPT_PREFIX}'."
        )

    print(f"ChatGPT seed images found: {len(chatgpt_images)}")

    created = 0

    for source_path in chatgpt_images:

        base_image = open_rgb(source_path)

        for i in range(1, CHATGPT_AUGMENTATIONS_PER_IMAGE + 1):

            output_name = (
                f"chatgpt_aug_"
                f"{source_path.stem}_"
                f"{i:03d}.jpg"
            )

            output_path = AI_RAW_DIR / output_name

            # Prevent duplicates if script is run again
            if output_path.exists():
                continue

            augmented = random_augment(base_image)

            augmented.save(
                output_path,
                format="JPEG",
                quality=95
            )

            created += 1

    print(f"ChatGPT augmentations created: {created}")


# =====================================================
# COUNT IMAGES
# =====================================================

def count_images(folder: Path) -> int:
    return len(
        [
            path
            for path in folder.iterdir()
            if is_image(path)
        ]
    )


# =====================================================
# MAIN
# =====================================================

def main():

    random.seed(RANDOM_SEED)

    print("\nTRUEIMAGE Hard-Example AI Augmentation")
    print("======================================")

    if not AI_RAW_DIR.exists():
        raise FileNotFoundError(
            f"AI dataset folder does not exist:\n{AI_RAW_DIR}"
        )

    before_ai = count_images(AI_RAW_DIR)

    print(f"AI images before augmentation: {before_ai}")
    print()

    augment_gemini_images()
    print()

    augment_chatgpt_images()
    print()

    after_ai = count_images(AI_RAW_DIR)

    print("--------------------------------------")
    print(f"AI images before: {before_ai}")
    print(f"AI images after : {after_ai}")
    print(f"AI images added : {after_ai - before_ai}")
    print("--------------------------------------")

    expected_added = (
        len(get_gemini_images()) * GEMINI_AUGMENTATIONS_PER_IMAGE
        + len(get_chatgpt_images()) * CHATGPT_AUGMENTATIONS_PER_IMAGE
    )

    print()
    print("Expected augmentation target:")
    print(f"Gemini  : 20 × {GEMINI_AUGMENTATIONS_PER_IMAGE} = 300")
    print(f"ChatGPT : 40 × {CHATGPT_AUGMENTATIONS_PER_IMAGE} = 600")
    print(f"Total   : 900")
    print()
    print(f"Expected AI total after augmentation: 3450")


if __name__ == "__main__":
    main()
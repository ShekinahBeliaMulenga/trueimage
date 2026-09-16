from pathlib import Path
import random

from PIL import (
    Image,
    ImageOps,
    ImageEnhance,
)


# =====================================================
# TRUEIMAGE HARD-EXAMPLE AI AUGMENTATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AI_RAW_DIR = (
    PROJECT_ROOT
    / "training"
    / "dataset"
    / "raw"
    / "ai_generated"
)

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
# HELPERS
# =====================================================

def is_image(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in ALLOWED_EXTENSIONS
    )


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


# =====================================================
# SAFE TRANSFORMATIONS (Preserve High-Frequency Detail)
# =====================================================

def horizontal_flip(image):
    return ImageOps.mirror(image)


def small_rotation(image, angle):
    return image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False
    )


def crop_resize(image, crop_fraction):
    width, height = image.size

    crop_width = int(width * crop_fraction)
    crop_height = int(height * crop_fraction)

    if crop_width >= width or crop_height >= height:
        return image

    max_left = width - crop_width
    max_top = height - crop_height

    left = random.randint(0, max_left)
    top = random.randint(0, max_top)

    cropped = image.crop(
        (left, top, left + crop_width, top + crop_height)
    )

    return cropped.resize((width, height), Image.Resampling.LANCZOS)


def centered_zoom(image, zoom):
    width, height = image.size

    crop_width = int(width / zoom)
    crop_height = int(height / zoom)

    left = (width - crop_width) // 2
    top = (height - crop_height) // 2

    cropped = image.crop(
        (left, top, left + crop_width, top + crop_height)
    )

    return cropped.resize((width, height), Image.Resampling.LANCZOS)


def small_translation(image, dx, dy):
    width, height = image.size
    return image.transform(
        (width, height),
        Image.Transform.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        resample=Image.Resampling.BICUBIC,
        fillcolor=None
    )


def adjust_brightness(image, factor):
    return ImageEnhance.Brightness(image).enhance(factor)


def adjust_contrast(image, factor):
    return ImageEnhance.Contrast(image).enhance(factor)


def adjust_saturation(image, factor):
    return ImageEnhance.Color(image).enhance(factor)


def adjust_sharpness(image, factor):
    return ImageEnhance.Sharpness(image).enhance(factor)


# =====================================================
# AUGMENTATION RECIPES (Destructive filters removed)
# =====================================================

def recipe_01(image):
    image = horizontal_flip(image)
    image = adjust_brightness(image, random.uniform(0.90, 1.10))
    return image

def recipe_02(image):
    image = small_rotation(image, random.uniform(-7, 7))
    image = adjust_contrast(image, random.uniform(0.90, 1.10))
    return image

def recipe_03(image):
    image = crop_resize(image, random.uniform(0.92, 0.97))
    image = adjust_saturation(image, random.uniform(0.92, 1.08))
    return image

def recipe_04(image):
    image = centered_zoom(image, random.uniform(1.03, 1.08))
    image = adjust_brightness(image, random.uniform(0.88, 1.12))
    return image

def recipe_05(image):
    image = horizontal_flip(image)
    image = small_rotation(image, random.uniform(-5, 5))
    image = adjust_contrast(image, random.uniform(0.90, 1.10))
    return image

def recipe_06(image):
    image = crop_resize(image, random.uniform(0.90, 0.96))
    image = adjust_brightness(image, random.uniform(0.90, 1.10))
    return image

def recipe_07(image):
    image = small_translation(image, random.randint(-8, 8), random.randint(-8, 8))
    image = adjust_contrast(image, random.uniform(0.88, 1.12))
    return image

def recipe_08(image):
    image = horizontal_flip(image)
    image = centered_zoom(image, random.uniform(1.02, 1.07))
    image = adjust_saturation(image, random.uniform(0.90, 1.10))
    return image

def recipe_09(image):
    image = small_rotation(image, random.uniform(-8, 8))
    image = adjust_sharpness(image, random.uniform(1.10, 1.30))
    return image

def recipe_10(image):
    image = horizontal_flip(image)
    image = small_translation(image, random.randint(-5, 5), random.randint(-5, 5))
    image = adjust_sharpness(image, random.uniform(0.85, 1.15))
    return image

def recipe_11(image):
    image = horizontal_flip(image)
    image = adjust_brightness(image, random.uniform(0.85, 1.15))
    image = adjust_saturation(image, random.uniform(0.92, 1.08))
    return image

def recipe_12(image):
    image = crop_resize(image, random.uniform(0.93, 0.98))
    image = small_rotation(image, random.uniform(-4, 4))
    image = adjust_contrast(image, random.uniform(0.90, 1.10))
    return image

def recipe_13(image):
    image = small_translation(image, random.randint(-6, 6), random.randint(-6, 6))
    image = adjust_brightness(image, random.uniform(0.90, 1.10))
    image = adjust_saturation(image, random.uniform(0.95, 1.15))
    return image

def recipe_14(image):
    image = horizontal_flip(image)
    image = centered_zoom(image, random.uniform(1.01, 1.05))
    image = adjust_contrast(image, random.uniform(0.95, 1.15))
    return image

def recipe_15(image):
    image = small_rotation(image, random.uniform(-6, 6))
    image = crop_resize(image, random.uniform(0.92, 0.98))
    image = adjust_sharpness(image, random.uniform(0.90, 1.20))
    return image


AUGMENTATION_RECIPES = [
    recipe_01, recipe_02, recipe_03, recipe_04, recipe_05,
    recipe_06, recipe_07, recipe_08, recipe_09, recipe_10,
    recipe_11, recipe_12, recipe_13, recipe_14, recipe_15,
]


# =====================================================
# FIND ORIGINAL IMAGES
# =====================================================

def get_gemini_images():
    return sorted([
        path for path in AI_RAW_DIR.iterdir()
        if is_image(path)
        and path.name.lower().startswith(GEMINI_PREFIX)
        and not path.name.lower().startswith("gemini_aug_")
    ])

def get_chatgpt_images():
    return sorted([
        path for path in AI_RAW_DIR.iterdir()
        if is_image(path)
        and path.name.lower().startswith(CHATGPT_PREFIX)
        and not path.name.lower().startswith("chatgpt_aug_")
    ])


# =====================================================
# AUGMENT
# =====================================================

def augment_gemini_images():
    gemini_images = get_gemini_images()
    if not gemini_images:
        raise ValueError(f"No Gemini images found in: {AI_RAW_DIR}")

    print(f"Gemini seed images found: {len(gemini_images)}")
    created = 0

    for source_path in gemini_images:
        base_image = open_rgb(source_path)
        for i, recipe in enumerate(AUGMENTATION_RECIPES, start=1):
            output_name = f"gemini_aug_{source_path.stem}_{i:03d}.jpg"
            output_path = AI_RAW_DIR / output_name

            if output_path.exists():
                continue

            augmented = recipe(base_image.copy())
            augmented.save(output_path, format="JPEG", quality=100) # Lossless as possible for JPG
            created += 1

    print(f"Gemini augmentations created: {created}")
    return created

def augment_chatgpt_images():
    chatgpt_images = get_chatgpt_images()
    if not chatgpt_images:
        raise ValueError(f"No ChatGPT images found in: {AI_RAW_DIR}")

    print(f"ChatGPT seed images found: {len(chatgpt_images)}")
    created = 0

    for source_path in chatgpt_images:
        base_image = open_rgb(source_path)
        for i, recipe in enumerate(AUGMENTATION_RECIPES, start=1):
            output_name = f"chatgpt_aug_{source_path.stem}_{i:03d}.jpg"
            output_path = AI_RAW_DIR / output_name

            if output_path.exists():
                continue

            augmented = recipe(base_image.copy())
            augmented.save(output_path, format="JPEG", quality=100) # Lossless as possible for JPG
            created += 1

    print(f"ChatGPT augmentations created: {created}")
    return created


def count_images(folder):
    return sum(1 for path in folder.iterdir() if is_image(path))


def main():
    random.seed(RANDOM_SEED)
    print("\nTRUEIMAGE Hard-Example AI Augmentation\n=======================================")

    if not AI_RAW_DIR.exists():
        raise FileNotFoundError(f"AI dataset folder does not exist: {AI_RAW_DIR}")

    before_ai = count_images(AI_RAW_DIR)
    print(f"AI images before augmentation: {before_ai}\n")

    gemini_created = augment_gemini_images()
    print()
    chatgpt_created = augment_chatgpt_images()
    print()

    after_ai = count_images(AI_RAW_DIR)
    expected_gemini = len(get_gemini_images()) * GEMINI_AUGMENTATIONS_PER_IMAGE
    expected_chatgpt = len(get_chatgpt_images()) * CHATGPT_AUGMENTATIONS_PER_IMAGE
    expected_total = expected_gemini + expected_chatgpt

    print("--------------------------------------")
    print(f"AI images before: {before_ai}")
    print(f"AI images after : {after_ai}")
    print(f"AI images added : {after_ai - before_ai}")
    print("--------------------------------------\n")

    if gemini_created == expected_gemini and chatgpt_created == expected_chatgpt:
        print("AUGMENTATION CHECK: PASSED")
    else:
        print("AUGMENTATION CHECK: NOT A FULL NEW GENERATION")


if __name__ == "__main__":
    main()
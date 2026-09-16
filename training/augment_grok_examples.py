from pathlib import Path
import random
import numpy as np
from PIL import (
    Image,
    ImageOps,
    ImageEnhance
)


# =====================================================
# TRUEIMAGE GROK HARD-EXAMPLE AUGMENTATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AI_RAW_DIR = (
    PROJECT_ROOT
    / "training"
    / "dataset"
    / "raw"
    / "ai_generated"
)

GROK_SEEDS = {
    "image_1",
    "image_2",
    "image_3",
    "image_4",
    "image_5",
}

AUGMENTATIONS_PER_IMAGE = 15

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
# INDIVIDUAL TRANSFORMATIONS
# =====================================================

def horizontal_flip(image):
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    return image

def random_crop_resize(image):
    width, height = image.size
    crop_fraction = random.uniform(0.88, 0.98)
    crop_width = int(width * crop_fraction)
    crop_height = int(height * crop_fraction)

    if crop_width >= width or crop_height >= height:
        return image

    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)

    image = image.crop((left, top, left + crop_width, top + crop_height))
    return image.resize((width, height), Image.Resampling.LANCZOS)

def small_rotation(image):
    angle = random.uniform(-10, 10)
    return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)

def small_zoom(image):
    width, height = image.size
    zoom = random.uniform(1.02, 1.10)
    crop_width = int(width / zoom)
    crop_height = int(height / zoom)

    left = (width - crop_width) // 2
    top = (height - crop_height) // 2

    image = image.crop((left, top, left + crop_width, top + crop_height))
    return image.resize((width, height), Image.Resampling.LANCZOS)

def random_brightness(image):
    return ImageEnhance.Brightness(image).enhance(random.uniform(0.80, 1.20))

def random_contrast(image):
    return ImageEnhance.Contrast(image).enhance(random.uniform(0.80, 1.20))

def random_saturation(image):
    return ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))

def random_sharpness(image):
    return ImageEnhance.Sharpness(image).enhance(random.uniform(0.75, 1.30))

def vectorized_mild_noise(image):
    """
    Replaces the extremely slow pure-Python loop with NumPy.
    Runs almost instantly.
    """
    if random.random() >= 0.20:
        return image

    img_array = np.array(image, dtype=np.int16)
    noise_strength = random.randint(2, 6)
    
    # Generate random noise array matching the image shape
    noise = np.random.randint(-noise_strength, noise_strength + 1, img_array.shape, dtype=np.int16)
    
    # Add noise, clip to valid RGB bounds, and convert back to uint8
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


# =====================================================
# COMBINED AUGMENTATION
# =====================================================

def random_augment(image):
    image = image.copy()

    # Geometric variation
    if random.random() < 0.60:
        image = horizontal_flip(image)
    if random.random() < 0.65:
        image = random_crop_resize(image)
    if random.random() < 0.50:
        image = small_rotation(image)
    if random.random() < 0.45:
        image = small_zoom(image)

    # Appearance variation
    if random.random() < 0.75:
        image = random_brightness(image)
    if random.random() < 0.70:
        image = random_contrast(image)
    if random.random() < 0.55:
        image = random_saturation(image)
    if random.random() < 0.40:
        image = random_sharpness(image)

    # Image-quality variation
    image = vectorized_mild_noise(image)
    
    # Removed jpeg compression, blur, and scale to preserve AI artifacts

    return image


# =====================================================
# FIND GROK SEEDS
# =====================================================

def get_grok_images():
    found = []
    for path in AI_RAW_DIR.iterdir():
        if not is_image(path):
            continue
        if path.stem.lower() in GROK_SEEDS:
            found.append(path)
    return sorted(found, key=lambda p: p.stem.lower())


# =====================================================
# AUGMENT
# =====================================================

def augment_grok_images():
    grok_images = get_grok_images()
    if len(grok_images) != 5:
        found = [path.name for path in grok_images]
        raise ValueError(
            "Expected exactly these five Grok originals:\n"
            "image_1.jpg\nimage_2.jpg\nimage_3.jpg\nimage_4.jpg\nimage_5.jpg\n\n"
            f"Found: {found}"
        )

    print(f"Grok seed images found: {len(grok_images)}\n")
    total_created = 0

    for source_path in grok_images:
        print(f"Creating augmentations for {source_path.name}")
        base_image = open_rgb(source_path)
        created_for_source = 0

        for i in range(1, AUGMENTATIONS_PER_IMAGE + 1):
            output_name = f"grok_aug_{source_path.stem}_{i:03d}.jpg"
            output_path = AI_RAW_DIR / output_name

            if output_path.exists():
                continue

            augmented = random_augment(base_image)
            # Save at 100 quality to avoid masking AI artifacts
            augmented.save(output_path, format="JPEG", quality=100) 

            created_for_source += 1
            total_created += 1

        print(f"  Created: {created_for_source}")

    return total_created


# =====================================================
# MAIN
# =====================================================

def count_images(folder):
    return sum(1 for path in folder.iterdir() if is_image(path))


def main():
    random.seed(RANDOM_SEED)
    print("\nTRUEIMAGE Grok Augmentation\n============================")

    if not AI_RAW_DIR.exists():
        raise FileNotFoundError(f"Missing directory:\n{AI_RAW_DIR}")

    before = count_images(AI_RAW_DIR)
    print(f"AI images before augmentation: {before}\n")

    created = augment_grok_images()
    after = count_images(AI_RAW_DIR)
    expected = 5 * AUGMENTATIONS_PER_IMAGE

    print("\n--------------------------------")
    print(f"AI images before: {before}")
    print(f"AI images after : {after}")
    print(f"AI images added : {after - before}")
    print("--------------------------------\n")
    print(f"Expected new augmentations: {expected}")
    print(f"Created this run: {created}")

    if after - before == expected:
        print("\nAUGMENTATION CHECK: PASSED")
    else:
        print("\nAUGMENTATION CHECK: FAILED")


if __name__ == "__main__":
    main()
import os
import random
import shutil
from pathlib import Path
from PIL import Image, ImageOps


# =====================================================
# TRUEIMAGE DATASET PREPARATION SCRIPT
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATASET_DIR = PROJECT_ROOT / "training" / "dataset" / "raw"
PROCESSED_DATASET_DIR = PROJECT_ROOT / "training" / "dataset" / "processed"

CLASSES = ["real", "ai_generated"]

IMAGE_SIZE = (224, 224)

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def reset_processed_folder():
    """
    Deletes the old processed dataset and recreates a clean folder structure.
    """

    if PROCESSED_DATASET_DIR.exists():
        shutil.rmtree(PROCESSED_DATASET_DIR)

    for split in ["train", "validation", "test"]:
        for class_name in CLASSES:
            folder = PROCESSED_DATASET_DIR / split / class_name
            folder.mkdir(parents=True, exist_ok=True)


def is_valid_image(file_path: Path) -> bool:
    """
    Checks if the file is a readable image.
    """

    try:
        with Image.open(file_path) as image:
            image.verify()
        return True
    except Exception:
        return False


def preprocess_and_save_image(source_path: Path, destination_path: Path):
    """
    Opens an image, converts it to RGB, resizes it to 224x224,
    and saves it as JPEG.
    """

    with Image.open(source_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)

        destination_path = destination_path.with_suffix(".jpg")
        image.save(destination_path, format="JPEG", quality=95)


def collect_images(class_name: str):
    """
    Collects all valid image paths for one class.
    """

    class_folder = RAW_DATASET_DIR / class_name

    if not class_folder.exists():
        raise FileNotFoundError(f"Missing folder: {class_folder}")

    image_paths = []

    for file_path in class_folder.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS:
            if is_valid_image(file_path):
                image_paths.append(file_path)
            else:
                print(f"[SKIPPED] Corrupted image: {file_path}")

    return image_paths


def split_dataset(image_paths):
    """
    Splits images into train, validation, and test sets.
    """

    random.shuffle(image_paths)

    total = len(image_paths)
    train_end = int(total * TRAIN_RATIO)
    validation_end = train_end + int(total * VALIDATION_RATIO)

    train_files = image_paths[:train_end]
    validation_files = image_paths[train_end:validation_end]
    test_files = image_paths[validation_end:]

    return train_files, validation_files, test_files


def process_split(files, split_name: str, class_name: str):
    """
    Saves processed images into the correct split/class folder.
    """

    destination_folder = PROCESSED_DATASET_DIR / split_name / class_name

    for index, source_path in enumerate(files, start=1):
        new_filename = f"{class_name}_{index:05d}.jpg"
        destination_path = destination_folder / new_filename

        try:
            preprocess_and_save_image(source_path, destination_path)
        except Exception as error:
            print(f"[FAILED] {source_path} -> {error}")


def prepare_dataset():
    """
    Main dataset preparation pipeline.
    """

    random.seed(RANDOM_SEED)

    print("\nTRUEIMAGE Dataset Preparation Started")
    print("------------------------------------")

    reset_processed_folder()

    for class_name in CLASSES:
        print(f"\nProcessing class: {class_name}")

        image_paths = collect_images(class_name)

        if len(image_paths) == 0:
            raise ValueError(f"No valid images found for class: {class_name}")

        print(f"Valid images found: {len(image_paths)}")

        train_files, validation_files, test_files = split_dataset(image_paths)

        print(f"Train: {len(train_files)}")
        print(f"Validation: {len(validation_files)}")
        print(f"Test: {len(test_files)}")

        process_split(train_files, "train", class_name)
        process_split(validation_files, "validation", class_name)
        process_split(test_files, "test", class_name)

    print("\nDataset preparation complete.")
    print(f"Processed dataset saved to: {PROCESSED_DATASET_DIR}")


if __name__ == "__main__":
    prepare_dataset()
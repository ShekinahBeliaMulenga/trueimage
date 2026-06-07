from pathlib import Path
from datasets import load_dataset
from PIL import Image, ImageOps
from tqdm import tqdm
import cv2
import numpy as np


# =====================================================
# TRUEIMAGE - OpenFake Face-Only Subset Downloader
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGET_DIR = PROJECT_ROOT / "training" / "dataset" / "raw" / "ai_generated"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COUNT = 500

FACE_KEYWORDS = [
    "face",
    "portrait",
    "person",
    "man",
    "woman",
    "human",
    "selfie",
    "headshot",
    "close-up",
    "close up",
    "profile picture",
    "passport photo",
]

PREFERRED_GENERATORS = [
    "flux",
    "midjourney",
    "gpt",
    "dall",
    "dalle",
    "dall-e",
    "stable",
    "diffusion",
    "imagen",
    "ideogram",
    "recraft",
]


def get_text_field(example: dict) -> str:
    """
    OpenFake metadata may use different text fields depending on split/version.
    This function safely checks common fields.
    """

    possible_fields = [
        "prompt",
        "caption",
        "text",
        "description",
        "query",
    ]

    parts = []

    for field in possible_fields:
        value = example.get(field)
        if value:
            parts.append(str(value))

    return " ".join(parts).lower()


def get_model_name(example: dict) -> str:
    """
    Safely reads the generator/model name from common metadata fields.
    """

    possible_fields = [
        "model",
        "model_name",
        "model_family",
        "generator",
        "source",
    ]

    for field in possible_fields:
        value = example.get(field)
        if value:
            return str(value).lower()

    return "unknown"


def is_fake_label(example: dict) -> bool:
    """
    Handles both possible label formats:
    - string label: 'fake'
    - numeric label: 1
    """

    label = example.get("label")

    if isinstance(label, str):
        return label.lower() == "fake"

    if isinstance(label, int):
        return label == 1

    return False


def prompt_suggests_face(example: dict) -> bool:
    text = get_text_field(example)
    return any(keyword in text for keyword in FACE_KEYWORDS)


def model_is_preferred(example: dict) -> bool:
    model_name = get_model_name(example)
    return any(keyword in model_name for keyword in PREFERRED_GENERATORS)


def pil_to_cv2(image: Image.Image):
    image = image.convert("RGB")
    array = np.array(image)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def has_clear_face(image: Image.Image) -> bool:
    """
    Uses OpenCV Haar face detector as a lightweight filter.
    This prevents saving random non-face images from OpenFake.
    """

    cv_image = pil_to_cv2(image)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    return len(faces) >= 1


def clean_filename(text: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in ["-", "_"] else "_"
        for char in text.lower()
    )
    return safe[:40] if safe else "unknown"


def save_image(image: Image.Image, model_name: str, index: int):
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    filename = f"openfake_face_{clean_filename(model_name)}_{index:05d}.jpg"
    output_path = TARGET_DIR / filename

    image.save(output_path, format="JPEG", quality=95)


def stream_openfake():
    """
    Uses streaming=True so the full OpenFake dataset is not downloaded.
    """

    return load_dataset(
        "ComplexDataLab/OpenFake",
        "core",
        split="test",
        streaming=True
    )


def main():
    print("\nTRUEIMAGE OpenFake Face Subset Downloader")
    print("-----------------------------------------")
    print(f"Target folder: {TARGET_DIR}")
    print(f"Target count : {TARGET_COUNT}")

    saved = 0
    scanned = 0
    prompt_matches = 0
    face_matches = 0

    dataset = stream_openfake()

    # First pass: prefer modern generator names
    print("\nPass 1: preferred modern generators")

    for example in tqdm(dataset):
        scanned += 1

        if not is_fake_label(example):
            continue

        if not prompt_suggests_face(example):
            continue

        prompt_matches += 1

        if not model_is_preferred(example):
            continue

        try:
            image = example["image"]

            if not isinstance(image, Image.Image):
                image = Image.open(image)

            if not has_clear_face(image):
                continue

            face_matches += 1
            saved += 1

            model_name = get_model_name(example)
            save_image(image, model_name, saved)

            print(f"[{saved}/{TARGET_COUNT}] saved from model: {model_name}")

            if saved >= TARGET_COUNT:
                break

        except Exception as error:
            print(f"[SKIPPED] {error}")

    # Fallback pass if preferred models were not enough
    if saved < TARGET_COUNT:
        print("\nPass 2: fallback fake face images")

        dataset = stream_openfake()

        for example in tqdm(dataset):
            if not is_fake_label(example):
                continue

            if not prompt_suggests_face(example):
                continue

            try:
                image = example["image"]

                if not isinstance(image, Image.Image):
                    image = Image.open(image)

                if not has_clear_face(image):
                    continue

                saved += 1

                model_name = get_model_name(example)
                save_image(image, model_name, saved)

                print(f"[{saved}/{TARGET_COUNT}] saved from model: {model_name}")

                if saved >= TARGET_COUNT:
                    break

            except Exception as error:
                print(f"[SKIPPED] {error}")

    print("\nDone.")
    print(f"Scanned rows          : {scanned}")
    print(f"Prompt matches        : {prompt_matches}")
    print(f"Face-detected matches : {face_matches}")
    print(f"Images saved          : {saved}")
    print(f"Saved to              : {TARGET_DIR}")


if __name__ == "__main__":
    main()
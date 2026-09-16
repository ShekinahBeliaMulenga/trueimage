import concurrent.futures
import math
import random
import shutil
from pathlib import Path
from PIL import Image, ImageFile, ImageOps

# Allow PIL to handle truncated files gracefully
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =====================================================
# TRUEIMAGE DATASET PREPARATION
# Fully data-driven, group-safe 70/15/15 split
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

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# =====================================================
# RESET PROCESSED DATASET
# =====================================================

def reset_processed_folder():
    if PROCESSED_DATASET_DIR.exists():
        shutil.rmtree(PROCESSED_DATASET_DIR)

    for split in ("train", "validation", "test"):
        for class_name in CLASSES:
            (
                PROCESSED_DATASET_DIR
                / split
                / class_name
            ).mkdir(parents=True, exist_ok=True)


# =====================================================
# IMAGE VALIDATION
# =====================================================

def is_valid_image(file_path: Path) -> bool:
    try:
        with Image.open(file_path) as image:
            image.verify()
        return True
    except Exception:
        return False


# =====================================================
# PREPROCESSING
# =====================================================

def preprocess_and_save_image(
    source_path: Path,
    destination_path: Path,
):
    with Image.open(source_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS,
        )

        destination_path = destination_path.with_suffix(".jpg")

        image.save(
            destination_path,
            format="JPEG",
            quality=95,
        )


# =====================================================
# COLLECT IMAGES
# =====================================================

def collect_images(class_name: str):
    class_folder = RAW_DATASET_DIR / class_name

    if not class_folder.exists():
        raise FileNotFoundError(
            f"Missing folder: {class_folder}"
        )

    image_paths = []

    for file_path in class_folder.rglob("*"):
        if (
            file_path.is_file()
            and file_path.suffix.lower() in ALLOWED_EXTENSIONS
        ):
            if is_valid_image(file_path):
                image_paths.append(file_path)
            else:
                print(f"[SKIPPED] Corrupted image: {file_path}")

    return sorted(
        image_paths,
        key=lambda path: path.name.lower(),
    )


# =====================================================
# GROUP IDENTIFICATION
# =====================================================

def get_group_key(file_path: Path) -> str:
    """
    Returns a stable group key for an original image
    and all of its known augmentations.

    Supported families:
        gemini_aug_gemini_01_001 -> gemini_01
        chatgpt_aug_portrait_01_001 -> portrait_01
        grok_aug_image_1_001 -> image_1

    Originals are mapped to their own key.
    Everything else is treated as an independent singleton group.
    """

    stem = file_path.stem.lower()

    if stem.startswith("gemini_aug_"):
        base = stem[len("gemini_aug_"):]
        return base.rsplit("_", 1)[0] if "_" in base else base

    if stem.startswith("chatgpt_aug_"):
        base = stem[len("chatgpt_aug_"):]
        return base.rsplit("_", 1)[0] if "_" in base else base

    if stem.startswith("grok_aug_"):
        base = stem[len("grok_aug_"):]
        return base.rsplit("_", 1)[0] if "_" in base else base

    if stem.startswith("gemini_"):
        return stem

    if stem.startswith("portrait_"):
        return stem

    if stem in {
        "image_1",
        "image_2",
        "image_3",
        "image_4",
        "image_5",
    }:
        return stem

    return stem


# =====================================================
# SOURCE IDENTIFICATION
# =====================================================

def get_source_type(group_key: str) -> str:
    key = group_key.lower()

    if key.startswith("gemini_"):
        return "gemini"

    if key.startswith("portrait_"):
        return "chatgpt"

    if key.startswith("image_"):
        return "grok"

    return "other"


# =====================================================
# BUILD GROUPS
# =====================================================

def build_groups(image_paths):
    groups = {}

    for image_path in image_paths:
        key = get_group_key(image_path)
        groups.setdefault(key, []).append(image_path)

    return groups


# =====================================================
# DATA-DRIVEN TARGET CALCULATION
# =====================================================

def calculate_split_targets(per_class_count: int):
    """
    Returns integer split targets whose sum equals the
    available balanced class count.

    The largest-remainder method is used so that the
    three targets remain as close as possible to the
    requested ratios.
    """

    raw = {
        "train": per_class_count * TRAIN_RATIO,
        "validation": per_class_count * VALIDATION_RATIO,
        "test": per_class_count * TEST_RATIO,
    }

    targets = {
        split: math.floor(value)
        for split, value in raw.items()
    }

    remaining = (
        per_class_count
        - sum(targets.values())
    )

    fractions = sorted(
        (
            (raw[split] - targets[split], split)
            for split in raw
        ),
        reverse=True,
    )

    for _, split in fractions[:remaining]:
        targets[split] += 1

    return targets


# =====================================================
# GENERATOR FAMILY DISCOVERY
# =====================================================

def discover_generator_families(image_paths):
    """
    Discovers generator families directly from the filesystem.
    No fixed number of Gemini, ChatGPT, or Grok families is assumed.
    """

    groups = build_groups(image_paths)

    generators = {
        "gemini": [],
        "chatgpt": [],
        "grok": [],
    }

    for group_key, members in groups.items():
        source = get_source_type(group_key)

        if source in generators:
            generators[source].append(
                (group_key, members)
            )

    return generators


# =====================================================
# GENERATOR COVERAGE
# =====================================================

def allocate_generator_families(
    image_paths,
    split_targets,
):
    """
    Allocates complete generator families first.

    Every discovered generator is represented in every
    split whenever that generator has at least three
    complete families available.

    When a generator has fewer than three families, it is
    represented in as many distinct splits as possible,
    without ever splitting a family.
    """

    generators = discover_generator_families(
        image_paths
    )

    split_groups = {
        "train": [],
        "validation": [],
        "test": [],
    }

    rng = random.Random(
        RANDOM_SEED
    )

    # -------------------------------------------------
    # Allocate each generator independently.
    # -------------------------------------------------

    for source, families in generators.items():

        families = list(families)
        rng.shuffle(families)

        if not families:
            continue

        split_order = [
            "train",
            "validation",
            "test",
        ]

        # Prefer coverage across all splits.
        number_of_splits = min(
            len(families),
            len(split_order),
        )

        coverage_splits = split_order[
            :number_of_splits
        ]

        # Put one family in each coverage split first.
        selected = families[:number_of_splits]

        used = number_of_splits

        for split, family in zip(
            coverage_splits,
            selected,
        ):
            split_groups[split].append(
                family
            )

        # Remaining families are assigned one at a time
        # to the split with the most remaining capacity.
        remaining = families[used:]

        for family in remaining:
            family_size = len(family[1])

            possible_splits = sorted(
                split_order,
                key=lambda split: (
                    split_targets[split]
                    - count_group_images(
                        split_groups[split]
                    )
                ),
                reverse=True,
            )

            chosen = None

            for split in possible_splits:
                current = count_group_images(
                    split_groups[split]
                )

                if (
                    current + family_size
                    <= split_targets[split]
                ):
                    chosen = split
                    break

            if chosen is None:
                raise RuntimeError(
                    f"Unable to place generator family "
                    f"{family[0]} without exceeding "
                    f"split targets."
                )

            split_groups[chosen].append(
                family
            )

    return split_groups


# =====================================================
# COUNT GROUP IMAGES
# =====================================================

def count_group_images(groups):
    return sum(
        len(members)
        for _, members in groups
    )


# =====================================================
# ALLOCATE SINGLETONS
# =====================================================

def allocate_singletons_to_targets(
    singleton_groups,
    existing_groups,
    split_targets,
):
    """
    Fill the remaining split capacity with singleton groups.
    Singleton groups are independent, so they may be assigned individually.
    """

    rng = random.Random(
        RANDOM_SEED + 100
    )

    singletons = list(
        singleton_groups
    )

    rng.shuffle(singletons)

    result = {
        split: list(existing_groups[split])
        for split in (
            "train",
            "validation",
            "test",
        )
    }

    for group in singletons:

        group_size = len(group[1])

        capacities = {
            split: (
                split_targets[split]
                - count_group_images(
                    result[split]
                )
            )
            for split in result
        }

        possible = [
            split
            for split, capacity in capacities.items()
            if capacity >= group_size
        ]

        if not possible:
            raise RuntimeError(
                "Unable to allocate remaining singleton "
                f"group '{group[0]}' without exceeding "
                "split targets."
            )

        chosen = max(
            possible,
            key=lambda split: capacities[split],
        )

        result[chosen].append(group)

    return result


# =====================================================
# BUILD FINAL SPLIT
# =====================================================

def build_ai_split(
    image_paths,
    split_targets,
):
    groups = build_groups(image_paths)

    multi_groups = [
        (key, members)
        for key, members in groups.items()
        if len(members) > 1
    ]

    singleton_groups = [
        (key, members)
        for key, members in groups.items()
        if len(members) == 1
    ]

    generator_groups = allocate_generator_families(
        image_paths,
        split_targets,
    )

    # Keys assigned via generator allocation
    generator_keys = {
        key
        for split_groups in generator_groups.values()
        for key, _ in split_groups
    }

    # Remove assigned generator keys from singletons
    remaining_singletons = [
        group
        for group in singleton_groups
        if group[0] not in generator_keys
    ]

    # Filter out multi-image groups that were already allocated as generator families
    unassigned_multi = [
        group
        for group in multi_groups
        if group[0] not in generator_keys
    ]

    # Place any unknown multi-image groups by capacity.
    groups_for_capacity = {
        split: list(generator_groups[split])
        for split in ("train", "validation", "test")
    }

    rng = random.Random(
        RANDOM_SEED + 150
    )
    rng.shuffle(unassigned_multi)

    for group in unassigned_multi:

        size = len(group[1])

        possible = [
            split
            for split in groups_for_capacity
            if (
                count_group_images(
                    groups_for_capacity[split]
                )
                + size
                <= split_targets[split]
            )
        ]

        if not possible:
            raise RuntimeError(
                f"Unable to allocate group "
                f"{group[0]} without exceeding "
                "split target."
            )

        chosen = max(
            possible,
            key=lambda split: (
                split_targets[split]
                - count_group_images(
                    groups_for_capacity[split]
                )
            ),
        )

        groups_for_capacity[chosen].append(
            group
        )

    final_groups = allocate_singletons_to_targets(
        remaining_singletons,
        groups_for_capacity,
        split_targets,
    )

    files = {}

    for split in (
        "train",
        "validation",
        "test",
    ):

        split_files = []

        for _, members in final_groups[split]:
            split_files.extend(members)

        if len(split_files) != split_targets[split]:
            raise RuntimeError(
                f"AI {split} count is "
                f"{len(split_files)}; expected "
                f"{split_targets[split]}."
            )

        files[split] = split_files

    # -------------------------------------------------
    # Group leakage check
    # -------------------------------------------------

    group_sets = {
        split: {
            get_group_key(path)
            for path in files[split]
        }
        for split in files
    }

    if (
        group_sets["train"]
        & group_sets["validation"]
    ):
        raise RuntimeError(
            "GROUP LEAKAGE detected between "
            "train and validation."
        )

    if (
        group_sets["train"]
        & group_sets["test"]
    ):
        raise RuntimeError(
            "GROUP LEAKAGE detected between "
            "train and test."
        )

    if (
        group_sets["validation"]
        & group_sets["test"]
    ):
        raise RuntimeError(
            "GROUP LEAKAGE detected between "
            "validation and test."
        )

    # -------------------------------------------------
    # Generator representation
    # -------------------------------------------------

    discovered_generators = {
        source
        for source, families
        in discover_generator_families(image_paths).items()
        if families
    }

    generator_presence = {}

    for split in files:

        present = {
            get_source_type(
                get_group_key(path)
            )
            for path in files[split]
        }

        generator_presence[split] = {
            source: (
                source in present
            )
            for source in discovered_generators
        }

    # If at least three families of a generator exist,
    # it must appear in every split.
    generator_family_counts = {
        source: len(families)
        for source, families
        in discover_generator_families(
            image_paths
        ).items()
    }

    for source, family_count in generator_family_counts.items():

        if family_count >= 3:

            for split in files:

                if not generator_presence[
                    split
                ].get(source, False):

                    raise RuntimeError(
                        f"{source} is missing from "
                        f"{split}."
                    )

    # At least one generator family goes to training
    # whenever that generator exists.
    for source, families in discover_generator_families(
        image_paths
    ).items():

        if not families:
            continue

        if not any(
            get_source_type(
                get_group_key(path)
            ) == source
            for path in files["train"]
        ):
            raise RuntimeError(
                f"{source} is missing from training."
            )

    return files, generator_presence


# =====================================================
# BUILD REAL SPLIT
# =====================================================

def build_real_split(
    image_paths,
    split_targets,
):
    """
    Real images are treated as independent groups.
    """

    files = list(image_paths)

    rng = random.Random(
        RANDOM_SEED + 300
    )

    rng.shuffle(files)

    train_end = split_targets["train"]

    validation_end = (
        train_end
        + split_targets["validation"]
    )

    return {
        "train": files[:train_end],
        "validation": files[
            train_end:validation_end
        ],
        "test": files[
            validation_end:
        ],
    }


# =====================================================
# PROCESS SPLIT (Parallelized)
# =====================================================

def _preprocess_worker(task):
    source_path, destination_path = task
    preprocess_and_save_image(source_path, destination_path)


def process_split(
    files,
    split_name,
    class_name,
):
    destination_folder = (
        PROCESSED_DATASET_DIR
        / split_name
        / class_name
    )

    tasks = [
        (
            source_path,
            destination_folder / f"{class_name}_{index:05d}.jpg"
        )
        for index, source_path in enumerate(files, start=1)
    ]

    try:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            list(executor.map(_preprocess_worker, tasks))
    except Exception as error:
        raise RuntimeError(
            f"Failed to process split {split_name}/{class_name}: {error}"
        )


# =====================================================
# VERIFY PROCESSED DATASET
# =====================================================

def verify_processed_dataset(
    split_targets,
):
    print()
    print("PROCESSED DATASET VERIFICATION")
    print("------------------------------------")

    expected_total = (
        sum(split_targets.values())
        * len(CLASSES)
    )

    total = 0

    for split in (
        "train",
        "validation",
        "test",
    ):

        print()
        print(split.upper())

        for class_name in CLASSES:

            folder = (
                PROCESSED_DATASET_DIR
                / split
                / class_name
            )

            actual = sum(
                1
                for file in folder.iterdir()
                if file.is_file()
            )

            expected = split_targets[split]

            print(
                f"  {class_name:<15}"
                f"{actual:>5} / {expected}"
            )

            if actual != expected:
                raise RuntimeError(
                    f"{split}/{class_name}: "
                    f"found {actual}, expected {expected}."
                )

            total += actual

    if total != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} processed "
            f"images, found {total}."
        )

    print()
    print("All processed counts: PASSED")


# =====================================================
# MAIN
# =====================================================

def prepare_dataset():

    random.seed(RANDOM_SEED)

    print()
    print(
        "TRUEIMAGE Dataset Preparation Started"
    )
    print("-------------------------------------")

    reset_processed_folder()

    datasets = {}

    # -------------------------------------------------
    # Discover images
    # -------------------------------------------------

    for class_name in CLASSES:

        print()
        print(
            f"Processing class: {class_name}"
        )

        image_paths = collect_images(
            class_name
        )

        if not image_paths:
            raise ValueError(
                f"No valid images found for "
                f"class: {class_name}"
            )

        print(
            f"Valid images found: "
            f"{len(image_paths)}"
        )

        datasets[class_name] = image_paths

    # -------------------------------------------------
    # Balance automatically
    # -------------------------------------------------

    available_real = len(
        datasets["real"]
    )

    available_ai = len(
        datasets["ai_generated"]
    )

    target_per_class = min(
        available_real,
        available_ai,
    )

    if target_per_class == 0:
        raise RuntimeError(
            "One or both classes contain no images."
        )

    print()
    print("RAW DATASET")
    print("------------------------------------")
    print(
        f"Real available: "
        f"{available_real}"
    )
    print(
        f"AI available:   "
        f"{available_ai}"
    )
    print(
        f"Balanced target per class: "
        f"{target_per_class}"
    )

    # -------------------------------------------------
    # Deterministic subset balancing
    # -------------------------------------------------

    if available_real != target_per_class:

        print()
        print(
            f"Reducing REAL class from "
            f"{available_real} to "
            f"{target_per_class}..."
        )

        real_paths = list(
            datasets["real"]
        )

        rng = random.Random(
            RANDOM_SEED + 400
        )

        rng.shuffle(real_paths)

        datasets["real"] = real_paths[
            :target_per_class
        ]

    if available_ai != target_per_class:

        print()
        print(
            f"Selecting {target_per_class} "
            "AI images group-safely..."
        )

        ai_groups = build_groups(
            datasets["ai_generated"]
        )

        rng = random.Random(
            RANDOM_SEED + 500
        )

        grouped = list(
            ai_groups.items()
        )

        rng.shuffle(grouped)

        selected_ai = []
        selected_count = 0

        remaining = []

        for group_key, members in grouped:

            if (
                selected_count
                + len(members)
                <= target_per_class
            ):

                selected_ai.extend(
                    members
                )

                selected_count += len(members)

            else:
                remaining.append(
                    (group_key, members)
                )

        if selected_count < target_per_class:

            needed = (
                target_per_class
                - selected_count
            )

            singleton_candidates = [
                members[0]
                for _, members in remaining
                if len(members) == 1
            ]

            if len(singleton_candidates) < needed:
                raise RuntimeError(
                    "Cannot create an exactly balanced "
                    "AI subset without splitting an "
                    "augmentation family."
                )

            rng.shuffle(
                singleton_candidates
            )

            selected_ai.extend(
                singleton_candidates[:needed]
            )

        if len(selected_ai) != target_per_class:
            raise RuntimeError(
                "AI balancing produced an "
                "incorrect image count."
            )

        datasets["ai_generated"] = selected_ai

    # -------------------------------------------------
    # Final split targets
    # -------------------------------------------------

    split_targets = calculate_split_targets(
        target_per_class
    )

    print()
    print("FINAL SPLIT TARGETS PER CLASS")
    print("------------------------------------")

    print(
        f"Train:       {split_targets['train']}"
    )
    print(
        f"Validation:  {split_targets['validation']}"
    )
    print(
        f"Test:        {split_targets['test']}"
    )

    if sum(split_targets.values()) != target_per_class:
        raise RuntimeError(
            "Split targets do not sum to the "
            "balanced class size."
        )

    # -------------------------------------------------
    # Split real
    # -------------------------------------------------

    print()
    print("Creating REAL split...")

    real_files = build_real_split(
        datasets["real"],
        split_targets,
    )

    # -------------------------------------------------
    # Split AI
    # -------------------------------------------------

    print()
    print("Creating AI split...")

    ai_files, generator_presence = build_ai_split(
        datasets["ai_generated"],
        split_targets,
    )

    # -------------------------------------------------
    # Report AI source coverage
    # -------------------------------------------------

    print()
    print("GENERATOR REPRESENTATION")
    print("------------------------------------")

    for split in (
        "train",
        "validation",
        "test",
    ):

        parts = []

        for source, present in sorted(
            generator_presence[split].items()
        ):

            parts.append(
                f"{source}={'YES' if present else 'NO'}"
            )

        print(
            f"{split.capitalize():<12}"
            + "  "
            + "  ".join(parts)
        )

    # -------------------------------------------------
    # Process real
    # -------------------------------------------------

    print()
    print("Processing REAL training images...")
    process_split(
        real_files["train"],
        "train",
        "real",
    )

    print(
        "Processing REAL validation images..."
    )
    process_split(
        real_files["validation"],
        "validation",
        "real",
    )

    print("Processing REAL test images...")
    process_split(
        real_files["test"],
        "test",
        "real",
    )

    # -------------------------------------------------
    # Process AI
    # -------------------------------------------------

    print()
    print("Processing AI training images...")
    process_split(
        ai_files["train"],
        "train",
        "ai_generated",
    )

    print(
        "Processing AI validation images..."
    )
    process_split(
        ai_files["validation"],
        "validation",
        "ai_generated",
    )

    print("Processing AI test images...")
    process_split(
        ai_files["test"],
        "test",
        "ai_generated",
    )

    # -------------------------------------------------
    # Verify
    # -------------------------------------------------

    verify_processed_dataset(
        split_targets
    )

    # -------------------------------------------------
    # Final report
    # -------------------------------------------------

    total_per_split = {
        split: split_targets[split] * 2
        for split in split_targets
    }

    print()
    print(
        "===================================="
    )
    print(
        "DATASET PREPARATION COMPLETE"
    )
    print(
        "===================================="
    )

    print()
    print("TRAIN:")
    print(
        f"  Real:          "
        f"{split_targets['train']}"
    )
    print(
        f"  AI-generated:  "
        f"{split_targets['train']}"
    )
    print(
        f"  Total:         "
        f"{total_per_split['train']}"
    )

    print()
    print("VALIDATION:")
    print(
        f"  Real:          "
        f"{split_targets['validation']}"
    )
    print(
        f"  AI-generated:  "
        f"{split_targets['validation']}"
    )
    print(
        f"  Total:         "
        f"{total_per_split['validation']}"
    )

    print()
    print("TEST:")
    print(
        f"  Real:          "
        f"{split_targets['test']}"
    )
    print(
        f"  AI-generated:  "
        f"{split_targets['test']}"
    )
    print(
        f"  Total:         "
        f"{total_per_split['test']}"
    )

    print()
    print("TOTAL:")
    print(
        f"  Real:          "
        f"{target_per_class}"
    )
    print(
        f"  AI-generated:  "
        f"{target_per_class}"
    )
    print(
        f"  Total:         "
        f"{target_per_class * 2}"
    )

    print()
    print("Checks:")
    print("  Class balance:             PASSED")
    print("  70/15/15 split:            PASSED")
    print("  Generator representation:  PASSED")
    print("  Group leakage:             PASSED")
    print("  Processed counts:          PASSED")

    print()
    print(
        f"Processed dataset saved to:\n"
        f"{PROCESSED_DATASET_DIR}"
    )


if __name__ == "__main__":
    prepare_dataset()
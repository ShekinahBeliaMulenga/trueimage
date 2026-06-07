from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_ROOT / "training" / "dataset" / "processed" / "test"
MODEL_PATH = PROJECT_ROOT / "app" / "models" / "trueimage_model.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def load_test_dataset():
    return tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=["real", "ai_generated"],
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


def main():
    print("\nTRUEIMAGE Threshold Analysis")
    print("----------------------------")

    model = tf.keras.models.load_model(MODEL_PATH)
    test_dataset = load_test_dataset()

    y_true = []
    y_probability = []

    for images, labels in test_dataset:
        probabilities = model.predict(images, verbose=0)

        y_true.extend(labels.numpy().flatten())
        y_probability.extend(probabilities.flatten())

    y_true = np.array(y_true).astype(int)
    y_probability = np.array(y_probability)

    auc = roc_auc_score(y_true, y_probability)

    print("\nClass mapping:")
    print("real          -> 0")
    print("ai_generated  -> 1")

    print(f"\nROC-AUC: {auc:.4f}")

    print("\nThreshold Results")
    print("-----------------")

    for threshold in THRESHOLDS:
        y_predicted = (y_probability >= threshold).astype(int)

        accuracy = accuracy_score(y_true, y_predicted)
        precision = precision_score(y_true, y_predicted, zero_division=0)
        recall = recall_score(y_true, y_predicted, zero_division=0)
        f1 = f1_score(y_true, y_predicted, zero_division=0)

        matrix = confusion_matrix(y_true, y_predicted)

        tn, fp, fn, tp = matrix.ravel()

        print(f"\nThreshold: {threshold:.2f}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"False Positives real -> AI : {fp}")
        print(f"False Negatives AI -> real : {fn}")
        print(matrix)


if __name__ == "__main__":
    main()
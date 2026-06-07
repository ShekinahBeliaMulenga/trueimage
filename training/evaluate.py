from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =====================================================
# TRUEIMAGE MODEL EVALUATION SCRIPT
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_ROOT / "training" / "dataset" / "processed" / "test"
MODEL_PATH = PROJECT_ROOT / "app" / "models" / "trueimage_model.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


def load_test_dataset():
    """
    Loads the test dataset.

    Class mapping:
        real          -> 0
        ai_generated  -> 1
    """

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=["real", "ai_generated"],
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return test_dataset


def evaluate_model():
    print("\nTRUEIMAGE Model Evaluation Started")
    print("----------------------------------")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

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

    y_predicted = (y_probability >= 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_predicted)
    precision = precision_score(y_true, y_predicted, zero_division=0)
    recall = recall_score(y_true, y_predicted, zero_division=0)
    f1 = f1_score(y_true, y_predicted, zero_division=0)
    auc = roc_auc_score(y_true, y_probability)

    print("\nClass mapping:")
    print("real          -> 0")
    print("ai_generated  -> 1")

    print("\nOverall Test Metrics")
    print("--------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\nConfusion Matrix")
    print("----------------")
    print(confusion_matrix(y_true, y_predicted))

    print("\nClassification Report")
    print("---------------------")
    print(
        classification_report(
            y_true,
            y_predicted,
            target_names=["real", "ai_generated"],
            zero_division=0
        )
    )

    print("\nInterpretation")
    print("--------------")
    print("False Positive: real image predicted as AI-generated.")
    print("False Negative: AI-generated image predicted as real.")


if __name__ == "__main__":
    evaluate_model()
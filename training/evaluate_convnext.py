from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# =====================================================
# TRUEIMAGE CONVNEXT-TINY MODEL EVALUATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "training" / "dataset" / "processed"
TEST_DIR = DATASET_DIR / "test"

MODEL_PATH = (
    PROJECT_ROOT
    / "app"
    / "models"
    / "trueimage_model_convnexttiny.keras"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


# =====================================================
# LOAD TEST DATASET
# =====================================================

def load_test_dataset():

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=[
            "real",
            "ai_generated"
        ],
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return test_dataset


# =====================================================
# EVALUATE MODEL
# =====================================================

def evaluate_model():

    print("\nTRUEIMAGE ConvNeXt-Tiny Model Evaluation")
    print("-----------------------------------------")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ConvNeXt-Tiny model not found:\n{MODEL_PATH}"
        )

    test_dataset = load_test_dataset()

    print("\nClass mapping:")
    print("real          -> 0")
    print("ai_generated  -> 1")

    # compile=False is important because the training model
    # contains the custom FBetaScore metric.
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    y_true = []
    y_prob = []

    for images, labels in test_dataset:

        probabilities = model.predict(
            images,
            verbose=0
        )

        y_true.extend(
            labels.numpy()
            .flatten()
            .tolist()
        )

        y_prob.extend(
            probabilities
            .flatten()
            .tolist()
        )

    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)

    # Default threshold
    threshold = 0.50

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    # =================================================
    # METRICS
    # =================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

    # =================================================
    # RESULTS
    # =================================================

    print("\nOverall Test Metrics")
    print("--------------------")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    # =================================================
    # CONFUSION MATRIX
    # =================================================

    print("\nConfusion Matrix")
    print("----------------")

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print(cm)

    # =================================================
    # CLASSIFICATION REPORT
    # =================================================

    print("\nClassification Report")
    print("---------------------")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "real",
                "ai_generated"
            ],
            zero_division=0
        )
    )

    # =================================================
    # FALSE POSITIVE / NEGATIVE COUNTS
    # =================================================

    true_negative, false_positive, false_negative, true_positive = cm.ravel()

    print("\nError Analysis")
    print("--------------")

    print(
        f"False Positives "
        f"(real -> AI): {false_positive}"
    )

    print(
        f"False Negatives "
        f"(AI -> real): {false_negative}"
    )

    print(
        f"\nTest samples: {len(y_true)}"
    )

    print(
        f"Threshold used: {threshold:.2f}"
    )


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    evaluate_model()
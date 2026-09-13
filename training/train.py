from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report


# =====================================================
# TRUEIMAGE MODEL TRAINING SCRIPT - FALSE-POSITIVE FOCUSED VERSION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "training" / "dataset" / "processed"
TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"

MODEL_OUTPUT_DIR = PROJECT_ROOT / "app" / "models"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = MODEL_OUTPUT_DIR / "trueimage_model_effv2s.keras"
BEST_WEIGHTS_PATH = MODEL_OUTPUT_DIR / "trueimage_best.weights.h5"
THRESHOLD_PATH = MODEL_OUTPUT_DIR / "trueimage_threshold.txt"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

PHASE_1_EPOCHS = 10
PHASE_2_EPOCHS = 10

# --- False-positive controls -------------------------------------------
# Multiply the "real" class weight beyond plain class balancing to make
# the model extra cautious about calling something ai_generated.
# 1.0 = pure balancing based on folder counts. Try 1.3-1.8 if FPs persist.
REAL_CLASS_WEIGHT_BOOST = 1.3

# Beta < 1 weights precision higher than recall in the metric used for
# checkpointing / early stopping. 0.5 is a reasonable "precision matters
# roughly 2x more than recall" setting.
FBETA_BETA = 0.5

# After training, search for the smallest threshold that hits this
# precision on the validation set (precision here = of everything the
# model calls ai_generated, what fraction actually is). Raise this if
# you're still seeing too many false positives after retraining.
TARGET_PRECISION = 0.97

# Toggle on if you suspect your real/ai_generated images differ in
# compression signature (see note in load_datasets docstring below).
ADD_COMPRESSION_AUGMENTATION = False


@tf.keras.utils.register_keras_serializable(package="trueimage")
class FBetaScore(tf.keras.metrics.Metric):
    """F-beta combining precision and recall, weighted toward precision when beta < 1."""

    def __init__(self, beta=0.5, threshold=0.5, name="f_beta", **kwargs):
        super().__init__(name=name, **kwargs)
        self.beta = beta
        self.threshold = threshold
        self.precision = tf.keras.metrics.Precision(thresholds=threshold)
        self.recall = tf.keras.metrics.Recall(thresholds=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        beta_sq = self.beta ** 2
        return (1 + beta_sq) * p * r / (beta_sq * p + r + tf.keras.backend.epsilon())

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

    def get_config(self):
        config = super().get_config()
        config.update({"beta": self.beta, "threshold": self.threshold})
        return config


def load_datasets():
    """
    NOTE ON FALSE POSITIVES: before touching any code, spot-check whether
    your `real` and `ai_generated` folders differ in anything OTHER than
    the thing you want the model to learn - e.g. average resolution,
    JPEG quality, presence of EXIF data, aspect ratio, or watermarks.
    If the "real" folder is a narrow, uniform source (e.g. one stock
    dataset) it will not generalize to the diverse real photos you see
    at inference time, and that alone can produce a high FP rate no
    matter how the model is trained.
    """
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=["real", "ai_generated"],
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=["real", "ai_generated"],
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_dataset, validation_dataset


def _compression_jitter(images, labels):
    """Randomly re-compress each image in the batch as JPEG (quality 35-100).

    Helps the model stop treating compression artifacts as a tell for
    'ai_generated' when the real-world real photos it sees have been through
    multiple rounds of social-media recompression. Only apply to training data.
    """
    def _apply(img):
        img_uint8 = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)
        img_uint8 = tf.image.random_jpeg_quality(img_uint8, 35, 100)
        return tf.cast(img_uint8, tf.float32)

    images = tf.map_fn(_apply, images, fn_output_signature=tf.float32)
    return images, labels


def optimize_dataset(dataset, augment_compression=False):
    dataset = dataset.cache()
    if augment_compression:
        dataset = dataset.map(_compression_jitter, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


def compute_class_weights():
    """Balance classes by folder count, then optionally boost the 'real'
    weight further so misclassifying a real image as ai_generated is
    penalized harder than the reverse mistake."""
    n_real = sum(1 for _ in (TRAIN_DIR / "real").iterdir())
    n_ai = sum(1 for _ in (TRAIN_DIR / "ai_generated").iterdir())
    total = n_real + n_ai

    weight_real = (total / (2.0 * n_real)) * REAL_CLASS_WEIGHT_BOOST
    weight_ai = total / (2.0 * n_ai)

    print(f"\nTraining set counts -> real: {n_real}, ai_generated: {n_ai}")
    print(f"Class weights -> real: {weight_real:.3f} (boost x{REAL_CLASS_WEIGHT_BOOST}), "
          f"ai_generated: {weight_ai:.3f}")

    return {0: weight_real, 1: weight_ai}


def build_model():
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.12),
            tf.keras.layers.RandomBrightness(0.08),
        ],
        name="data_augmentation"
    )

    base_model = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        include_preprocessing=True
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_image")

    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)

    x = tf.keras.layers.BatchNormalization(name="batch_normalization")(x)
    x = tf.keras.layers.Dropout(0.40, name="dropout_1")(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="dense_features"
    )(x)

    x = tf.keras.layers.Dropout(0.30, name="dropout_2")(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="ai_probability"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="trueimage_detector")

    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            FBetaScore(beta=FBETA_BETA, name="f_beta"),
        ]
    )


def evaluate_and_tune_threshold(model, validation_dataset, target_precision=TARGET_PRECISION):
    """Scan validation predictions to find a threshold hitting target_precision
    on the ai_generated class, then print a confusion matrix and full report
    at that threshold so you can see the false-positive rate directly."""
    y_true, y_pred_probs = [], []

    for images, labels in validation_dataset:
        probs = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten().tolist())
        y_pred_probs.extend(probs.flatten().tolist())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)

    # Confusion matrix / report at the default 0.5 threshold, for comparison
    default_preds = (y_pred_probs >= 0.5).astype(int)
    print("\n--- At default 0.5 threshold ---")
    print(confusion_matrix(y_true, default_preds))
    print(classification_report(y_true, default_preds, target_names=["real", "ai_generated"]))

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_probs)

    best_threshold = 0.5
    for p, t in zip(precisions[:-1], thresholds):
        if p >= target_precision:
            best_threshold = t
            break
    else:
        print(f"\nWARNING: no threshold reached {target_precision:.0%} precision on this "
              f"validation set. Using the highest-precision threshold found instead. "
              f"This usually means the dataset issue in load_datasets()'s docstring, "
              f"not the training loop, is the bottleneck.")
        best_idx = int(np.argmax(precisions[:-1])) if len(precisions) > 1 else 0
        best_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5

    tuned_preds = (y_pred_probs >= best_threshold).astype(int)
    print(f"\n--- At tuned threshold {best_threshold:.3f} (target precision {target_precision:.0%}) ---")
    print(confusion_matrix(y_true, tuned_preds))
    print(classification_report(y_true, tuned_preds, target_names=["real", "ai_generated"]))

    THRESHOLD_PATH.write_text(str(float(best_threshold)))
    print(f"\nSaved tuned threshold to {THRESHOLD_PATH} - use this at inference time "
          f"instead of the default 0.5 cutoff.")

    return best_threshold


def train_model():
    print("\nTRUEIMAGE Improved Model Training Started")
    print("----------------------------------------")

    train_dataset, validation_dataset = load_datasets()

    print("\nClass mapping:")
    print("real          -> 0")
    print("ai_generated  -> 1")

    class_weights = compute_class_weights()

    train_dataset = optimize_dataset(train_dataset, augment_compression=ADD_COMPRESSION_AUGMENTATION)
    validation_dataset = optimize_dataset(validation_dataset, augment_compression=False)

    model, base_model = build_model()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_WEIGHTS_PATH,
            monitor="val_f_beta",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_f_beta",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # =================================================
    # PHASE 1: Train only the classification head
    # =================================================
    print("\nPHASE 1: Training classifier head")
    print("---------------------------------")

    base_model.trainable = False
    compile_model(model, learning_rate=0.0003)

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=PHASE_1_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # =================================================
    # PHASE 2: Fine-tune upper layers of EfficientNet
    # =================================================
    print("\nPHASE 2: Fine-tuning top EfficientNet layers")
    print("--------------------------------------------")

    base_model.trainable = True

    # Freeze most layers, fine-tune only the last layers
    for layer in base_model.layers[:-25]:
        layer.trainable = False

    compile_model(model, learning_rate=0.00003)

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        initial_epoch=PHASE_1_EPOCHS,
        epochs=PHASE_1_EPOCHS + PHASE_2_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    if BEST_WEIGHTS_PATH.exists():
        model.load_weights(BEST_WEIGHTS_PATH)

    model.save(MODEL_SAVE_PATH, include_optimizer=False)

    print("\nImproved training complete.")
    print(f"Best weights saved to: {BEST_WEIGHTS_PATH}")
    print(f"Final model saved to: {MODEL_SAVE_PATH}")

    print("\nRunning post-training threshold tuning on validation set...")
    evaluate_and_tune_threshold(model, validation_dataset)


if __name__ == "__main__":
    train_model()
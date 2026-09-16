from pathlib import Path

import numpy as np

import os

# Disable TensorFlow/XLA automatic JIT before importing TensorFlow.
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_cpu_enable_fast_math=false"

import tensorflow as tf

# Explicitly disable JIT/XLA at the TensorFlow runtime level.
tf.config.optimizer.set_jit(False)

from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    classification_report
)


# =====================================================
# TRUEIMAGE CONVNEXT-TINY TRAINING
# FALSE-POSITIVE FOCUSED VERSION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "training" / "dataset" / "processed"
TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"

MODEL_OUTPUT_DIR = PROJECT_ROOT / "app" / "models"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = MODEL_OUTPUT_DIR / "trueimage_model_convnexttiny.keras"
BEST_WEIGHTS_PATH = MODEL_OUTPUT_DIR / "trueimage_convnexttiny_best.weights.h5"
THRESHOLD_PATH = MODEL_OUTPUT_DIR / "trueimage_convnexttiny_threshold.txt"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

PHASE_1_EPOCHS = 10
PHASE_2_EPOCHS = 10


# =====================================================
# FALSE-POSITIVE CONTROLS
# =====================================================

# Gives the REAL class extra importance.
# real = 0
# ai_generated = 1
REAL_CLASS_WEIGHT_BOOST = 1.3

# F-beta with beta < 1 gives more importance to precision.
FBETA_BETA = 0.5

# Validation threshold tuning target.
TARGET_PRECISION = 0.97

# Leave disabled initially.
ADD_COMPRESSION_AUGMENTATION = False


# =====================================================
# F-BETA METRIC
# =====================================================

@tf.keras.utils.register_keras_serializable(package="trueimage")
class FBetaScore(tf.keras.metrics.Metric):
    """
    F-beta metric.

    beta < 1 gives more importance to precision than recall.
    """

    def __init__(
        self,
        beta=0.5,
        threshold=0.5,
        name="f_beta",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.beta = beta
        self.threshold = threshold

        self.precision = tf.keras.metrics.Precision(
            thresholds=threshold
        )

        self.recall = tf.keras.metrics.Recall(
            thresholds=threshold
        )

    def update_state(
        self,
        y_true,
        y_pred,
        sample_weight=None
    ):
        self.precision.update_state(
            y_true,
            y_pred,
            sample_weight
        )

        self.recall.update_state(
            y_true,
            y_pred,
            sample_weight
        )

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()

        beta_squared = self.beta ** 2

        return (
            (1 + beta_squared)
            * precision
            * recall
            / (
                beta_squared * precision
                + recall
                + tf.keras.backend.epsilon()
            )
        )

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

    def get_config(self):
        config = super().get_config()

        config.update({
            "beta": self.beta,
            "threshold": self.threshold
        })

        return config


# =====================================================
# DATASET LOADING
# =====================================================

def load_datasets():
    """
    Loads the processed dataset.

    IMPORTANT:
    real          -> 0
    ai_generated  -> 1
    """

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=[
            "real",
            "ai_generated"
        ],
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
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

    return (
        train_dataset,
        validation_dataset
    )


# =====================================================
# OPTIONAL COMPRESSION AUGMENTATION
# =====================================================

def _compression_jitter(images, labels):
    """
    Applies random JPEG compression.

    Disabled by default.
    """

    def _apply(image):

        image_uint8 = tf.cast(
            tf.clip_by_value(
                image,
                0,
                255
            ),
            tf.uint8
        )

        image_uint8 = tf.image.random_jpeg_quality(
            image_uint8,
            35,
            100
        )

        return tf.cast(
            image_uint8,
            tf.float32
        )

    images = tf.map_fn(
        _apply,
        images,
        fn_output_signature=tf.float32
    )

    return images, labels


# =====================================================
# DATASET OPTIMIZATION
# =====================================================

def optimize_dataset(
    dataset,
    augment_compression=False
):
    """
    Prepares the input pipeline without caching the entire
    decoded dataset in RAM.

    This is intentionally conservative for CPU-only
    training on native Windows.
    """

    if augment_compression:

        dataset = dataset.map(
            _compression_jitter,
            num_parallel_calls=1
        )

    return dataset.prefetch(
        buffer_size=1
    )


# =====================================================
# CLASS WEIGHTS
# =====================================================

def compute_class_weights():

    real_dir = TRAIN_DIR / "real"
    ai_dir = TRAIN_DIR / "ai_generated"

    n_real = sum(
        1
        for path in real_dir.iterdir()
        if path.is_file()
    )

    n_ai = sum(
        1
        for path in ai_dir.iterdir()
        if path.is_file()
    )

    total = n_real + n_ai

    weight_real = (
        total / (2.0 * n_real)
    ) * REAL_CLASS_WEIGHT_BOOST

    weight_ai = (
        total / (2.0 * n_ai)
    )

    print(
        f"\nTraining set counts -> "
        f"real: {n_real}, "
        f"ai_generated: {n_ai}"
    )

    print(
        f"Class weights -> "
        f"real: {weight_real:.3f} "
        f"(boost x{REAL_CLASS_WEIGHT_BOOST}), "
        f"ai_generated: {weight_ai:.3f}"
    )

    return {
        0: weight_real,
        1: weight_ai
    }


# =====================================================
# CONVNEXT-TINY MODEL
# =====================================================

def build_model():

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(
                "horizontal"
            ),

            tf.keras.layers.RandomRotation(
                0.05
            ),

            tf.keras.layers.RandomZoom(
                0.10
            ),

            tf.keras.layers.RandomContrast(
                0.12
            ),

            tf.keras.layers.RandomBrightness(
                0.08
            ),
        ],
        name="data_augmentation"
    )

    # =================================================
    # CONVNEXT-TINY
    # =================================================

    base_model = tf.keras.applications.ConvNeXtTiny(
        include_top=False,
        weights="imagenet",
        input_shape=(
            224,
            224,
            3
        ),
        include_preprocessing=True
    )

    base_model.trainable = False

    # =================================================
    # CLASSIFICATION HEAD
    # =================================================

    inputs = tf.keras.Input(
        shape=(
            224,
            224,
            3
        ),
        name="input_image"
    )

    x = data_augmentation(
        inputs
    )

    x = base_model(
        x,
        training=False
    )

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="batch_normalization"
    )(x)

    x = tf.keras.layers.Dropout(
        0.40,
        name="dropout_1"
    )(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(
            0.001
        ),
        name="dense_features"
    )(x)

    x = tf.keras.layers.Dropout(
        0.30,
        name="dropout_2"
    )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="ai_probability"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
        name="trueimage_convnexttiny_detector"
    )

    return (
        model,
        base_model
    )


# =====================================================
# COMPILE MODEL
# =====================================================

def compile_model(
    model,
    learning_rate
):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),

        # Explicitly disable XLA/JIT compilation.
        jit_compile=False,

        loss="binary_crossentropy",

        metrics=[
            "accuracy",

            tf.keras.metrics.Precision(
                name="precision"
            ),

            tf.keras.metrics.Recall(
                name="recall"
            ),

            tf.keras.metrics.AUC(
                name="auc"
            ),

            FBetaScore(
                beta=FBETA_BETA,
                name="f_beta"
            )
        ]
    )


# =====================================================
# VALIDATION THRESHOLD TUNING
# =====================================================

def evaluate_and_tune_threshold(
    model,
    validation_dataset,
    target_precision=TARGET_PRECISION
):

    """
    Finds the smallest validation threshold that
    achieves the requested AI precision.
    """

    y_true = []
    y_pred_probs = []

    for images, labels in validation_dataset:

        probabilities = model.predict(
            images,
            verbose=0
        )

        y_true.extend(
            labels.numpy()
            .flatten()
            .tolist()
        )

        y_pred_probs.extend(
            probabilities
            .flatten()
            .tolist()
        )

    y_true = np.array(
        y_true
    )

    y_pred_probs = np.array(
        y_pred_probs
    )

    # -------------------------------------------------
    # DEFAULT THRESHOLD
    # -------------------------------------------------

    default_predictions = (
        y_pred_probs >= 0.5
    ).astype(int)

    print(
        "\n--- At default 0.5 threshold ---"
    )

    print(
        confusion_matrix(
            y_true,
            default_predictions
        )
    )

    print(
        classification_report(
            y_true,
            default_predictions,
            target_names=[
                "real",
                "ai_generated"
            ]
        )
    )

    # -------------------------------------------------
    # PRECISION-RECALL CURVE
    # -------------------------------------------------

    precisions, recalls, thresholds = (
        precision_recall_curve(
            y_true,
            y_pred_probs
        )
    )

    best_threshold = 0.5

    found_target = False

    for precision, threshold in zip(
        precisions[:-1],
        thresholds
    ):

        if precision >= target_precision:

            best_threshold = threshold
            found_target = True

            break

    if not found_target:

        print(
            f"\nWARNING: no threshold reached "
            f"{target_precision:.0%} precision "
            f"on this validation set."
        )

        if len(precisions) > 1:

            best_index = int(
                np.argmax(
                    precisions[:-1]
                )
            )

            if len(thresholds) > 0:

                best_threshold = (
                    thresholds[
                        best_index
                    ]
                )

    # -------------------------------------------------
    # TUNED THRESHOLD
    # -------------------------------------------------

    tuned_predictions = (
        y_pred_probs >= best_threshold
    ).astype(int)

    print(
        f"\n--- At tuned threshold "
        f"{best_threshold:.3f} "
        f"(target precision "
        f"{target_precision:.0%}) ---"
    )

    print(
        confusion_matrix(
            y_true,
            tuned_predictions
        )
    )

    print(
        classification_report(
            y_true,
            tuned_predictions,
            target_names=[
                "real",
                "ai_generated"
            ]
        )
    )

    # -------------------------------------------------
    # SAVE THRESHOLD
    # -------------------------------------------------

    THRESHOLD_PATH.write_text(
        str(
            float(
                best_threshold
            )
        ),
        encoding="utf-8"
    )

    print(
        f"\nSaved tuned threshold to "
        f"{THRESHOLD_PATH}"
    )

    print(
        "Use this threshold at inference "
        "instead of the default 0.5 cutoff."
    )

    return best_threshold


# =====================================================
# MAIN TRAINING
# =====================================================

def train_model():

    print(
        "\nTRUEIMAGE ConvNeXt-Tiny "
        "Model Training Started"
    )

    print(
        "========================================"
    )

    print(
        "Runtime safeguards: XLA/JIT disabled, "
        "dataset RAM cache disabled"
    )

    train_dataset, validation_dataset = (
        load_datasets()
    )

    print(
        "\nClass mapping:"
    )

    print(
        "real          -> 0"
    )

    print(
        "ai_generated  -> 1"
    )

    # -------------------------------------------------
    # CLASS WEIGHTS
    # -------------------------------------------------

    class_weights = (
        compute_class_weights()
    )

    # -------------------------------------------------
    # DATASET OPTIMIZATION
    # -------------------------------------------------

    train_dataset = optimize_dataset(
        train_dataset,
        augment_compression=(
            ADD_COMPRESSION_AUGMENTATION
        )
    )

    validation_dataset = optimize_dataset(
        validation_dataset,
        augment_compression=False
    )

    # -------------------------------------------------
    # MODEL
    # -------------------------------------------------

    model, base_model = build_model()

    # -------------------------------------------------
    # CALLBACKS
    # -------------------------------------------------

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
    # PHASE 1
    # TRAIN CLASSIFICATION HEAD
    # =================================================

    print(
        "\nPHASE 1: Training classifier head"
    )

    print(
        "---------------------------------"
    )

    base_model.trainable = False

    compile_model(
        model,
        learning_rate=0.0003
    )

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=PHASE_1_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # =================================================
    # PHASE 2
    # FINE-TUNE LAST CONVNEXT LAYERS
    # =================================================

    print(
        "\nPHASE 2: Fine-tuning top ConvNeXt layers"
    )

    print(
        "--------------------------------------------"
    )

    base_model.trainable = True

    # Freeze most layers.
    # Fine-tune only the last 25 layers.
    for layer in base_model.layers[:-25]:

        layer.trainable = False

    compile_model(
        model,
        learning_rate=0.00003
    )

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        initial_epoch=PHASE_1_EPOCHS,
        epochs=(
            PHASE_1_EPOCHS
            + PHASE_2_EPOCHS
        ),
        class_weight=class_weights,
        callbacks=callbacks
    )

    # -------------------------------------------------
    # LOAD BEST WEIGHTS
    # -------------------------------------------------

    if BEST_WEIGHTS_PATH.exists():

        model.load_weights(
            BEST_WEIGHTS_PATH
        )

    # -------------------------------------------------
    # SAVE FINAL MODEL
    # -------------------------------------------------

    model.save(
        MODEL_SAVE_PATH,
        include_optimizer=False
    )

    print(
        "\nConvNeXt-Tiny training complete."
    )

    print(
        f"Best weights saved to: "
        f"{BEST_WEIGHTS_PATH}"
    )

    print(
        f"Final model saved to: "
        f"{MODEL_SAVE_PATH}"
    )

    # -------------------------------------------------
    # POST-TRAINING THRESHOLD TUNING
    # -------------------------------------------------

    print(
        "\nRunning post-training "
        "threshold tuning..."
    )

    evaluate_and_tune_threshold(
        model,
        validation_dataset
    )


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    
    train_model()
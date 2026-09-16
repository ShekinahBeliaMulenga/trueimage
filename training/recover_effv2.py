from pathlib import Path
import tensorflow as tf


# =====================================================
# TRUEIMAGE - RECOVER BEST EFFICIENTNETV2-S MODEL
# NO TRAINING
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_OUTPUT_DIR = (
    PROJECT_ROOT / "app" / "models"
)

BEST_WEIGHTS_PATH = (
    MODEL_OUTPUT_DIR
    / "trueimage_best.weights.h5"
)

MODEL_SAVE_PATH = (
    MODEL_OUTPUT_DIR
    / "trueimage_model_effv2s.keras"
)

IMAGE_SIZE = (224, 224)


# =====================================================
# BUILD EXACT MODEL ARCHITECTURE
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

    base_model = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        include_preprocessing=True
    )

    base_model.trainable = True

    # Same fine-tuning configuration used during training
    for layer in base_model.layers[:-25]:
        layer.trainable = False

    inputs = tf.keras.Input(
        shape=(224, 224, 3),
        name="input_image"
    )

    x = data_augmentation(inputs)

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
        name="ai_probability",
        dtype="float32"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
        name="trueimage_detector"
    )

    return model


# =====================================================
# RECOVER
# =====================================================

def main():

    print()
    print(
        "TRUEIMAGE EfficientNetV2-S Recovery"
    )
    print(
        "===================================="
    )

    if not BEST_WEIGHTS_PATH.exists():

        raise FileNotFoundError(
            f"Best weights not found:\n"
            f"{BEST_WEIGHTS_PATH}"
        )

    print()
    print(
        f"Loading best weights:\n"
        f"{BEST_WEIGHTS_PATH}"
    )

    model = build_model()

    # Load the weights saved at the best epoch.
    model.load_weights(
        BEST_WEIGHTS_PATH
    )

    print(
        "Best weights loaded successfully."
    )

    # Save the recovered model without
    # optimizer/training state.
    model.save(
        str(MODEL_SAVE_PATH),
        include_optimizer=False
    )

    print()
    print(
        "Model recovered successfully."
    )

    print(
        f"Model saved to:\n"
        f"{MODEL_SAVE_PATH}"
    )

    print()
    print(
        "NO TRAINING WAS PERFORMED."
    )


if __name__ == "__main__":
    main()
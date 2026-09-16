from pathlib import Path
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "training" / "dataset" / "processed"
TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"

MODEL_OUTPUT_DIR = PROJECT_ROOT / "app" / "models"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = MODEL_OUTPUT_DIR / "trueimage_efficientnet_b0.keras"
BEST_WEIGHTS_PATH = MODEL_OUTPUT_DIR / "trueimage_efficientnet_b0_best.weights.h5"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32  # Bumped up for stability
SEED = 42

PHASE_1_EPOCHS = 10
PHASE_2_EPOCHS = 10


def load_datasets():
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=["real", "ai_generated"],
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=["real", "ai_generated"],
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_dataset, validation_dataset


def optimize_dataset(dataset):
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


def build_model():
    # Minimal augmentations to preserve artifact signals
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.02),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_image")
    x = data_augmentation(inputs)

    # CRITICAL: EfficientNet scaling normalization
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(0.40, name="dropout_1")(x)
    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="dense_features",
    )(x)
    x = tf.keras.layers.Dropout(0.30, name="dropout_2")(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="ai_probability")(x)

    model = tf.keras.Model(
        inputs, outputs, name="trueimage_efficientnet_b0_detector"
    )

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
        ],
    )


def train_model():
    train_dataset, validation_dataset = load_datasets()
    train_dataset = optimize_dataset(train_dataset)
    validation_dataset = optimize_dataset(validation_dataset)

    model, base_model = build_model()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_WEIGHTS_PATH,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=4, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1
        ),
    ]

    # --- PHASE 1 ---
    print("\nPHASE 1: Training classifier head...")
    compile_model(model, learning_rate=3e-4)
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=PHASE_1_EPOCHS,
        callbacks=callbacks,
    )

    # --- PHASE 2 ---
    print("\nPHASE 2: Fine-tuning top EfficientNet-B0 layers...")
    base_model.trainable = True

    # Freeze all layers except last 25, while keeping BatchNorm frozen
    for layer in base_model.layers[:-25]:
        layer.trainable = False

    for layer in base_model.layers[-25:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, learning_rate=3e-5)
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        initial_epoch=PHASE_1_EPOCHS,
        epochs=PHASE_1_EPOCHS + PHASE_2_EPOCHS,
        callbacks=callbacks,
    )

    if BEST_WEIGHTS_PATH.exists():
        model.load_weights(BEST_WEIGHTS_PATH)

    model.save(MODEL_SAVE_PATH, include_optimizer=False)
    print("\nTraining complete.")


if __name__ == "__main__":
    train_model()
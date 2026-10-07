import os
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (
    Input,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

try:
    from config import (
        TRAIN_DIR,
        VAL_DIR,
        MODEL_PATH,
        IMG_SIZE,
        BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        CLASS_NAMES
    )
except ImportError:
    from src.config import (
        TRAIN_DIR,
        VAL_DIR,
        MODEL_PATH,
        IMG_SIZE,
        BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        CLASS_NAMES
    )


def build_model():
    """Build the MobileNetV2 binary classification model."""

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )

    # Freeze pretrained MobileNetV2 layers initially
    base_model.trainable = False

    inputs = Input(shape=(*IMG_SIZE, 3))

    # Data augmentation
    x = RandomFlip("horizontal")(inputs)
    x = RandomRotation(0.1)(x)
    x = RandomZoom(0.1)(x)

    # MobileNetV2 preprocessing is INSIDE the model
    x = preprocess_input(x)

    x = base_model(x, training=False)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    # Binary classification:
    # 0 = with_helmet
    # 1 = without_helmet
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def create_generators():
    """Create training and validation data generators."""

    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # No rescale here because preprocess_input is inside the model
    val_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=CLASS_NAMES,
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=CLASS_NAMES,
        shuffle=False
    )

    print("\nClass mapping:")
    print(train_gen.class_indices)

    return train_gen, val_gen


def plot_history(history):
    """Plot and save training/validation accuracy and loss."""

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(1, len(accuracy) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label="Training Accuracy")
    plt.plot(epochs_range, val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.tight_layout()

    curves_path = os.path.join(
        os.path.dirname(MODEL_PATH),
        "training_curves.png"
    )

    plt.savefig(curves_path, dpi=150)
    plt.show()

    print(f"\nTraining curves saved to: {curves_path}")


def main():
    print("Building MobileNetV2 model...")

    model = build_model()
    model.summary()

    print("\nCreating data generators...")

    train_gen, val_gen = create_generators()

    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\nStarting training...")

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks
    )

    plot_history(history)

    print("\nBest model saved to:")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()
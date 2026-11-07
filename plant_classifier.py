import os
import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tqdm import tqdm
import matplotlib.pyplot as plt

# Update these paths for your local dataset.
DATA_ROOT = os.environ.get("PLANT_DATA_ROOT", "./dataset-for-task2")
TRAIN_PATH = os.path.join(DATA_ROOT, "train", "*", "*.png")
TEST_PATH = os.path.join(DATA_ROOT, "test", "*.png")
TEMPLATE_CSV = os.environ.get("PLANT_TEMPLATE_CSV", "./submission-for-task2.csv")
OUTPUT_DIR = os.environ.get("PLANT_OUTPUT_DIR", "./outputs")

IMAGE_SIZE = 192
NUM_CLASSES = 5
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
LR_INIT = 1.5e-4


def read_img(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Unable to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))


def create_model():
    base_model = EfficientNetB2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        weights="imagenet",
        include_top=False,
    )
    base_model.trainable = True
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.55),
        Dense(512, activation="relu"),
        Dropout(0.45),
        Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=LR_INIT),
        loss=CategoricalCrossentropy(label_smoothing=0.12),
        metrics=["accuracy"],
    )
    return model


def tta_predict(model, x_test, n_tta=8):
    preds = []
    tta_gen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )
    for i in range(n_tta):
        if i == 0:
            batch = x_test
        else:
            batch = next(tta_gen.flow(
                x_test, batch_size=len(x_test), shuffle=False
            ))
        preds.append(model.predict(batch, verbose=0))
    return np.mean(preds, axis=0)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_paths = glob.glob(TRAIN_PATH)
    if not train_paths:
        raise FileNotFoundError(f"No training PNG files found using: {TRAIN_PATH}")

    x_train, labels = [], []
    print(f"Training images: {len(train_paths)}")
    for path in tqdm(train_paths, desc="Loading training images"):
        x_train.append(read_img(path))
        labels.append(os.path.basename(os.path.dirname(path)))
    x_train = preprocess_input(np.asarray(x_train))

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    if len(label_encoder.classes_) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, got {len(label_encoder.classes_)}: "
            f"{list(label_encoder.classes_)}"
        )
    y_onehot = to_categorical(y_encoded, num_classes=NUM_CLASSES)
    print(f"Classes: {list(label_encoder.classes_)}")

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train,
        y_onehot,
        test_size=0.12,
        random_state=RANDOM_SEED,
        stratify=y_encoded,
    )

    train_gen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.25,
        height_shift_range=0.25,
        zoom_range=0.25,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.75, 1.25],
    )
    train_gen.fit(x_tr)

    model = create_model()
    model.summary()
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy", patience=15, restore_best_weights=True,
            mode="max",
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.45, patience=6, min_lr=5e-7,
        ),
        ModelCheckpoint(
            os.path.join(OUTPUT_DIR, "best_model.keras"),
            monitor="val_accuracy", save_best_only=True, mode="max",
        ),
    ]

    history = model.fit(
        train_gen.flow(x_tr, y_tr, batch_size=BATCH_SIZE, seed=RANDOM_SEED),
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    for metric, title, filename in [
        ("accuracy", "Accuracy", "accuracy.png"),
        ("loss", "Loss", "loss.png"),
    ]:
        plt.figure()
        plt.plot(history.history[metric], label=f"train {metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"validation {metric}")
        plt.legend()
        plt.title(title)
        plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches="tight")
        plt.close()

    test_paths = glob.glob(TEST_PATH)
    if not test_paths:
        raise FileNotFoundError(f"No test PNG files found using: {TEST_PATH}")
    template = pd.read_csv(TEMPLATE_CSV)
    if "ID" not in template.columns or "Category" not in template.columns:
        raise ValueError("Submission template must have ID and Category columns")
    test_ids = [os.path.basename(p) for p in test_paths]
    if len(test_paths) != len(template) or set(test_ids) != set(template["ID"]):
        raise ValueError("Test image IDs/count do not match the submission template")

    x_test = preprocess_input(np.asarray([
        read_img(path) for path in tqdm(test_paths, desc="Loading test images")
    ]))
    test_prob = tta_predict(model, x_test, n_tta=8)
    pred_idx = np.argmax(test_prob, axis=1)
    pred_categories = label_encoder.inverse_transform(pred_idx)

    output = pd.DataFrame({"ID": test_ids, "Category": pred_categories})
    output = output.set_index("ID").loc[template["ID"]].reset_index()
    output.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
    print(f"Submission saved to {os.path.join(OUTPUT_DIR, 'submission.csv')}")


if __name__ == "__main__":
    main()

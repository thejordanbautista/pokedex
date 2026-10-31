#!/usr/bin/env python3
"""Fine-tune MobileNetV3 on a folder of labelled photos and export an int8 TFLite model.

Run on a laptop with a GPU or in Google Colab, not on the Pi:

    pip install tensorflow
    python scripts/train_classifier.py photos/ --out models/pokemon

`photos/` has one sub-folder per class, named exactly as the dex entry (or one of its labels):

    photos/Pikachu/*.jpg
    photos/Bulbasaur/*.jpg

Tips: 50+ photos per class, taken with the Pokédex's own camera, in the lighting you'll use,
with cards, plushies and figures from several angles. Add a "background" class of random scenes
so the model has somewhere to put "nothing here".

Writes <out>.tflite and <out>_labels.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path, help="folder with one sub-folder per class")
    p.add_argument("--out", type=Path, default=Path("models/pokemon"))
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--fine-tune-epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--float", action="store_true", help="export float32 instead of int8")
    args = p.parse_args()

    import numpy as np
    import tensorflow as tf

    size = (args.size, args.size)
    train = tf.keras.utils.image_dataset_from_directory(
        args.data, validation_split=0.2, subset="training", seed=42, image_size=size, batch_size=args.batch)
    val = tf.keras.utils.image_dataset_from_directory(
        args.data, validation_split=0.2, subset="validation", seed=42, image_size=size, batch_size=args.batch)
    classes = train.class_names
    print(f"{len(classes)} classes: {classes[:10]}{' ...' if len(classes) > 10 else ''}")

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
    ])
    # The model's input is pixels scaled to [-1, 1], which is what pokedex/classifier.py feeds
    # float models. For the int8 export, uint8 pixels 0-255 map onto that range automatically.
    to_unit = lambda x: x / 127.5 - 1.0  # noqa: E731
    train = train.map(lambda x, y: (to_unit(augment(x, training=True)), y)).prefetch(tf.data.AUTOTUNE)
    val = val.map(lambda x, y: (to_unit(x), y)).prefetch(tf.data.AUTOTUNE)

    # MobileNetV3 has built-in [0, 255] preprocessing, so scale back up inside the model.
    base = tf.keras.applications.MobileNetV3Small(input_shape=(*size, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = tf.keras.Input((*size, 3))
    x = tf.keras.layers.Rescaling(127.5, offset=127.5)(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(classes), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", ["accuracy"])
    model.fit(train, validation_data=val, epochs=args.epochs)

    base.trainable = True
    for layer in base.layers[: len(base.layers) * 2 // 3]:
        layer.trainable = False
    model.compile(tf.keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy", ["accuracy"])
    model.fit(train, validation_data=val, epochs=args.fine_tune_epochs)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if not args.float:
        def representative():
            for images, _ in val.take(20):
                for img in images:
                    yield [np.expand_dims(img.numpy().astype(np.float32), 0)]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8   # raw 0-255 pixels in
        converter.inference_output_type = tf.uint8  # quantized probabilities out
    tflite = converter.convert()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model_path = args.out.with_suffix(".tflite")
    model_path.write_bytes(tflite)
    labels_path = args.out.parent / f"{args.out.name}_labels.txt"
    labels_path.write_text("\n".join(classes) + "\n", encoding="utf-8")
    print(f"Wrote {model_path} ({len(tflite) / 1e6:.1f} MB) and {labels_path}")


if __name__ == "__main__":
    main()

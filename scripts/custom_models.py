import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

from tensorflow.keras import layers, models, Model
from tensorflow.python.keras.callbacks import (
    TensorBoard,
    EarlyStopping,
    ModelCheckpoint,
)
from tensorflow.keras.models import Sequential


import keras_cv


def rebuild_top(model_base, kind="cla") -> Sequential:
    """Rebuild top of a pre-trained model to make it suitable for classification or regression."""

    assert kind in ["cla", "reg"], "kind must be either cla or reg"

    model = tf.keras.Sequential()

    model.add(model_base)

    # Rebuild top
    # FIXME: en el codigo original de keras, esto es un Conv2D-relu-dropout-conv2d-flatten-dense ¿esta bien como lo armé?
    #   based on: https://stackoverflow.com/questions/54537674/modify-resnet50-output-layer-for-regression?rq=3
    model.add(layers.Flatten())

    if kind == "cla":
        # Add fully conected layers
        # model.add(layers.Dense(2048, name="fc1", activation="relu"))
        #         model.add(layers.Dense(2048, name="fc1", activation="relu"))
        model.add(layers.Dense(10, name="predictions", activation="softmax"))
    if kind == "reg":
        model.add(layers.Dense(1, name="predictions", activation="linear"))

    return model


def YOLOv8(n_classes=2, freeze_layers=10) -> Sequential:
    """https://keras.io/api/applications/mobilenet_v3/#mobilenetv3small-function"""

    # backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    #     "yolo_v8_s_backbone_coco"  # We will use yolov8 small backbone with coco weights
    # )

    # yolo = keras_cv.models.YOLOV8Detector(
    #     num_classes=n_classes,
    #     bounding_box_format="xyxy",
    #     backbone=backbone,
    #     fpn_depth=1,
    # )
    model = keras_cv.models.YOLOV8Detector(
        num_classes=n_classes,
        bounding_box_format="xyxy",
        backbone=keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_s_backbone_coco"),
        fpn_depth=2,
    )
    for layer in model.layers[:freeze_layers]:
        layer.trainable = False

    return model


def small_cnn(resizing_size=200) -> Sequential:
    """layer normalization entre cada capa y su activación. Batch norm no funca
    porque uso batches de 1, se supone que no funciona bien para muestras de
    menos de 32 (demasiada varianza en las estadísticas de cada batch).

    'there are strong theoretical reasons against it, and multiple publications
    have shown BN performance degrade for batch_size under 32, and severely for <=8.
    In a nutshell, batch statistics "averaged" over a single sample vary greatly
    sample-to-sample (high variance), and BN mechanisms don't work as intended'
    (https://stackoverflow.com/questions/59648509/batch-normalization-when-batch-size-1).

    Layer normalization is independent of the batch size, so it can be applied to
    batches with smaller sizes as well.
    (https://www.pinecone.io/learn/batch-layer-normalization/)"""

    model = models.Sequential()

    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation="linear",
            input_shape=(resizing_size, resizing_size, 4),
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="linear"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation="linear"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="linear"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    # model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="linear"))

    return model

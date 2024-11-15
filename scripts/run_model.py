##############      Configuración      ##############
import os

# REPO = r"/mnt/d/Becas y Proyectos/EY Challenge 2024/EY24"
REPO = r"C:/Users/Usuario/OneDrive - Royal Holloway University of London/EY Deep Learning/ey_deep_learning/EY24"
assert os.path.isdir(
    REPO
), "No existe el repositorio. Revisar la variable REPO del codigo run_model"

PATH_DATAIN = rf"{REPO}/data/data_in"
PATH_DATAOUT = rf"{REPO}/data/data_out"
PATH_SCRIPTS = rf"{REPO}/scripts"
PATH_LOGS = rf"{REPO}/logs"
PATH_OUTPUTS = rf"{REPO}/outputs"

for folder in [PATH_DATAIN, PATH_DATAOUT, PATH_SCRIPTS, PATH_LOGS, PATH_OUTPUTS]:
    os.makedirs(folder, exist_ok=True)

###############################################

import custom_models
import utils
import plot_tools

import os
import sys
import scipy
import random
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from typing import Iterator, List, Union, Tuple, Any
from datetime import datetime
from sklearn.model_selection import train_test_split
from shapely.geometry import Polygon

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

from tensorflow.keras import layers, models, Model
from tensorflow.keras.callbacks import (
    TensorBoard,
    EarlyStopping,
    ModelCheckpoint,
    CSVLogger,
)
from tensorflow.keras.models import Sequential

import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization

pd.set_option("display.max_columns", None)

# the next 3 lines of code are for my machine and setup due to https://github.com/tensorflow/tensorflow/issues/43174
try:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("No GPU set. Is the GPU already initialized?")


PR_DS_POST = xr.open_dataset(rf"{REPO}/data/data_in/Post_Event_San_Juan.tif")
PR_DS_PRE = xr.open_dataset(rf"{REPO}/data/data_in/Pre_Event_San_Juan.tif")

PR_DS_POST_POLYGON = utils.get_dataset_extent(PR_DS_POST)
PR_DS_PRE_POLYGON = utils.get_dataset_extent(PR_DS_PRE)


def generate_savename(model_name, sample_size, variable, extra):

    savename = f"{model_name}_sample{sample_size}_{variable}{extra}"

    return savename


def create_datasets(
    AREAS_TO_SAMPLE,
    building_gdf,
    image_size,
    variable,
    train_size,
    savename="",
    save_examples=True,
):
    # Trying to replicate these good practices from here: https://cs230.stanford.edu/blog/datapipeline/#best-practices

    # Based on: https://medium.com/@acordier/tf-data-dataset-generators-with-parallelization-the-easy-way-b5c5f7d2a18
    def get_data(load=False):
        # Decoding from the EagerTensor object. Extracts the number/value from the tensor
        #   example: <tf.Tensor: shape=(), dtype=uint8, numpy=20> -> 20

        # initialize iterators & params
        bboxs = np.zeros(shape=(0, 4))
        image = np.zeros(shape=(3, 0, 0))
        img_correct_shape = (3, image_size, image_size)

        all_conditions_met = False
        # Iterate until the image has the correct shape (when selecting borders)
        while all_conditions_met is False:  # or (has_damage is False):

            # Reset conditions
            img_has_correct_shape = False
            img_has_buildings = False
            img_has_damaged_buildings = False

            # Sample a random area to generate the image
            polygon = AREAS_TO_SAMPLE.sample(1).geometry.values[0]

            # Generate the image
            image, boundaries = utils.stacked_image_from_census_tract(
                dataset=PR_DS_POST,
                polygon=polygon,
                img_size=image_size,
                n_bands=3,
                stacked_images=[1],
            )
            if boundaries is not None:
                im_classes, bboxs = utils.get_image_classes_and_boxes(
                    building_gdf, boundaries, image_size, variable
                )

                img_has_correct_shape = image.shape == img_correct_shape
                img_has_buildings = bboxs.shape[0] != 0
                img_has_damaged_buildings = utils.assess_image_damage(im_classes)

            all_conditions_met = all(
                [img_has_correct_shape, img_has_buildings, img_has_damaged_buildings]
            )

        # Reduce quality and process image
        image = utils.process_image(image, resizing_size=image_size)

        # Augment dataset
        image = utils.augment_image(image, bboxs)

        return image, im_classes, bboxs

    # Pack the dataset into a dictionary with desired types
    def pack_features_vector(image, label, bbox):

        return image, {"boxes": bbox, "classes": label}

    ### Generate Datasets
    # Split the data
    val_size = int(train_size * 0.1)
    print()
    print(f"Train size: {train_size} images")
    print(f"Validation size: {val_size} images")

    ## TRAIN ##
    # Generator for the index
    train_dataset = tf.data.Dataset.from_generator(
        lambda: list(range(train_size)),  # The index generator,
        tf.uint32,
    )  # Creates a dataset with only the indexes (0, 1, 2, 3, etc.)

    train_dataset = train_dataset.map(
        lambda i: tf.py_function(  # The actual data generator. Passes the index to the function that will process the data.
            func=get_data,
            inp=[i],
            Tout=[tf.uint8, tf.uint8, tf.float64],  # image, classes, bbox
        ),
    )
    train_dataset = train_dataset.map(
        lambda image, classes, bbox: (
            tf.ensure_shape(image, (image_size, image_size, 3)),
            tf.ensure_shape(classes, [None]),  # Modify the shape according to your data
            tf.ensure_shape(bbox, [None, 4]),  # Modify the shape according to your data
        )
    )

    train_dataset = train_dataset.map(pack_features_vector)
    train_dataset = train_dataset.ragged_batch(4, drop_remainder=True).prefetch(
        tf.data.AUTOTUNE
    )

    ## VAL ##
    # Generator for the index
    val_dataset = tf.data.Dataset.from_generator(
        lambda: list(range(val_size)),  # The index generator,
        tf.uint32,
    )  # Creates a dataset with only the indexes (0, 1, 2, 3, etc.)

    val_dataset = val_dataset.map(
        lambda i: tf.py_function(  # The actual data generator. Passes the index to the function that will process the data.
            func=get_data,
            inp=[i],
            Tout=[tf.uint8, tf.uint8, tf.float64],  # image, classes, bbox
        ),
    )
    val_dataset = val_dataset.map(
        lambda image, classes, bbox: (
            tf.ensure_shape(image, (image_size, image_size, 3)),
            tf.ensure_shape(classes, [None]),  # Modify the shape according to your data
            tf.ensure_shape(bbox, [None, 4]),  # Modify the shape according to your data
        )
    )

    val_dataset = val_dataset.map(pack_features_vector)
    val_dataset = val_dataset.ragged_batch(8, drop_remainder=True)

    if save_examples == True:
        print("saving train/test examples")

        plot_tools.visualize_dataset(
            f"{savename}_train_", train_dataset, (0, 255), 2, 2
        )
        plot_tools.visualize_dataset(f"{savename}_val_", val_dataset, (0, 255), 2, 2)
        i = 0
        for x in train_dataset.take(1):
            np.save(
                f"{PATH_OUTPUTS}/{savename}_train_example_{i}_imgs", tfds.as_numpy(x)[0]
            )
            np.save(
                f"{PATH_OUTPUTS}/{savename}_train_example_{i}_classes",
                tfds.as_numpy(x)[1]["classes"],
            )
            np.save(
                f"{PATH_OUTPUTS}/{savename}_train_example_{i}_bbox",
                tfds.as_numpy(x)[1]["boxes"],
            )
            i += 1

        i = 0
        for x in val_dataset.take(1):
            np.save(
                f"{PATH_OUTPUTS}/{savename}_val_example_{i}_imgs", tfds.as_numpy(x)[0]
            )
            np.save(
                f"{PATH_OUTPUTS}/{savename}_val_example_{i}_classes",
                tfds.as_numpy(x)[1]["classes"],
            )
            np.save(
                f"{PATH_OUTPUTS}/{savename}_val_example_{i}_bbox",
                tfds.as_numpy(x)[1]["boxes"],
            )

            i += 1

    print()
    print("Dataset generado!")

    return train_dataset, val_dataset


def get_callbacks(
    savename,
    val_dataset,
    logdir=None,
) -> List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]:
    """Accepts the model name as a string and returns multiple callbacks for training the keras model.

    Parameters
    ----------
    model_name : str
        The name of the model as a string.

    Returns
    -------
    List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]
        A list of multiple keras callbacks.
    """

    class CheckpointModel(tf.keras.callbacks.Callback):
        def __init__(self, log_dir):
            super(CheckpointModel, self).__init__()
            self.log_dir = log_dir

        def on_epoch_end(self, epoch, logs=None):
            # Save model
            os.makedirs(f"{PATH_DATAOUT}/models_by_epoch/{savename}", exist_ok=True)
            self.model.save(
                f"{PATH_DATAOUT}/models_by_epoch/{savename}/{savename}_{epoch}",
                include_optimizer=True,
            )
            generate_predictions(savename, model=self.model, epoch=epoch)

    tensorboard_callback = TensorBoard(
        log_dir=logdir, histogram_freq=1  # , profile_batch="100,200"
    )
    # use tensorboard --logdir logs/scalars in your command line to startup tensorboard with the correct logs

    # Create an instance of your custom callback
    checkpoint_model = CheckpointModel(log_dir=logdir)

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0,  # the training is terminated as soon as the performance measure gets worse from one epoch to the next
        start_from_epoch=50,
        patience=30,  # amount of epochs with no improvements until the model stops
        verbose=2,
        mode="auto",  # the model is stopped when the quantity monitored has stopped decreasing
        restore_best_weights=True,  # restore the best model with the lowest validation error
    )
    # reduce_lr = keras.callbacks.ReduceLROnPlateau(
    #     monitor="val_loss", factor=0.2, patience=10, min_lr=0.0000001
    # )
    model_checkpoint_callback = ModelCheckpoint(
        f"{PATH_DATAOUT}/models/{savename}",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,  # save the best model
        mode="auto",
        save_freq="epoch",  # save every epoch
    )
    csv_logger = CSVLogger(
        f"{PATH_DATAOUT}/models_by_epoch/{savename}/{savename}_history.csv", append=True
    )

    return [
        # coco_metrics_callback,  # Mean Average Precision (mAP) metric
        early_stopping_callback,
        # reduce_lr,
        model_checkpoint_callback,  # Save best model
        checkpoint_model,  # Save model every 10 epochs
        csv_logger,  # Save history in csv
        tensorboard_callback,  # Save history in tensorboard
    ]


def run_model(
    model_function: Model,
    train_dataset: Iterator,
    val_dataset: Iterator,
    epochs: int,
    callbacks: List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]],
    box_loss,
    class_loss,
    savename: str = "",
):
    """This function runs a keras model with the Ranger optimizer and multiple callbacks. The model is evaluated within
    training through the validation generator and afterwards one final time on the test generator.

    Parameters
    ----------
    model_function : Model
        Keras model fun44ction like small_cnn()  or adapt_efficient_net().
    lr : float4
        Learning rate.
    train_dataset : Iterator
        tensorflow dataset for the training data.
    test_dataset : Iterator
        tesorflow dataset for the test data.
    loss: str
        Loss function.
    metrics: List[str]
        List of metrics to be used.

    Returns
    -------
    History
        The history of the keras model as a History object. To access it as a Dict, use history.history.
    """
    import keras.backend as K

    def get_last_trained_epoch(savename):
        try:
            files = os.listdir(f"{PATH_DATAOUT}/models_by_epoch/{savename}")
            epochs = [file.split("_")[-1] for file in files]
            epochs = [int(epoch) for epoch in epochs if epoch.isdigit()]
            initial_epoch = max(epochs)
        except:
            os.makedirs(f"{PATH_DATAOUT}/models_by_epoch/{savename}", exist_ok=True)
            print("Model not found, running from begining")
            initial_epoch = None

        return initial_epoch

    initial_epoch = get_last_trained_epoch(savename)

    if initial_epoch is None:
        # constructs the model and compiles it
        model = model_function
        model.summary()

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, global_clipnorm=10)

        model.compile(
            optimizer=optimizer,
            classification_loss=class_loss,  # "focal",
            box_loss=box_loss,  # "SmoothL1",
        )
        initial_epoch = 0

    else:
        print("Restoring model...")
        model_path = (
            f"{PATH_DATAOUT}/models_by_epoch/{savename}/{savename}_{initial_epoch}"
        )
        model = keras.models.load_model(
            model_path,
        )  # load the model from file
        initial_epoch = initial_epoch + 1

    history = model.fit(
        train_dataset,
        epochs=epochs,
        initial_epoch=initial_epoch,
        validation_data=val_dataset,
        callbacks=callbacks,
        workers=2,  # adjust this according to the number of CPU cores of your machine
    )

    return model, history  # type: ignore


def set_model_and_loss_function(model_name: str, n_classes: int):
    # Diccionario de modelos
    get_model_from_name = {
        # FIXME: Agregar este modelo que no existe hoy
        "YOLOv8": custom_models.YOLOv8(n_classes, freeze_layers=25),
        "RetinaNet": custom_models.RetinaNet(n_classes, freeze_layers=3),
    }

    # Validación de parámetros
    assert (
        model_name in get_model_from_name.keys()
    ), "model_name must be one of the following: " + str(
        list(get_model_from_name.keys())
    )

    # Get model
    model = get_model_from_name[model_name]

    # Set loss and metrics
    if model_name == "YOLOv8":
        class_loss = "binary_crossentropy"
        box_loss = "ciou"
    elif model_name == "RetinaNet":
        class_loss = "focal"
        box_loss = "SmoothL1"

    return model, class_loss, box_loss


def generate_predictions(savename, model=None, epoch="final"):
    import cv2

    submit_folder = rf"{PATH_DATAIN}/Submission data"

    if model is None:
        model = custom_models.YOLOv8(2)
        model.load_weights(rf"{PATH_DATAOUT}/models/{savename}").expect_partial()

    images_files = os.listdir(submit_folder)
    images = []
    for image_file in images_files:
        image_path = os.path.join(submit_folder, image_file)
        image = cv2.imread(image_path)  # Read the image
        images += [image]
    images = np.stack(images)
    images_ds = tf.data.Dataset.from_tensor_slices(images)
    images_ds = images_ds.ragged_batch(4)

    # Make prediction
    predictions = model.predict(images_ds)

    plot_tools.visualize_predictions(savename, images, predictions, 4, 3)

    out = f"{PATH_OUTPUTS}/{savename}_unformatted_submitions_{epoch}.npy"
    np.save(out, predictions)
    print("Se creó: ", out)


def run(
    variable: str,
    model_name: str,
    weights=None,
    image_size=512,
    train_size=1000,
    n_epochs=100,
    extra="",
):
    """Run all the code of this file.

    Parameters
    ----------
    small_sample : bool, optional
        If you just want to check if the code is working, set small_sample to True, by default False
    """

    savename = generate_savename(model_name, train_size, variable, extra)
    log_dir = f"{PATH_LOGS}/{savename}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    ## Set Model & loss function
    model, class_loss, box_loss = set_model_and_loss_function(
        model_name=model_name,
        n_classes=2,
    )

    building_gdf = gpd.read_parquet(
        rf"{REPO}/data/data_out/building_footprint_cropped_{variable}.parquet"
    )

    # Transform the building footprint into a gdf with areas to sample
    labeled_data_region = building_gdf.dissolve().geometry[0]
    relevant_area = building_gdf[building_gdf[variable] == 1]
    relevant_area.geometry = relevant_area.buffer(115)  # Aprox 115mts
    roi = relevant_area.dissolve().geometry[0]
    AREAS_TO_SAMPLE = utils.multi_polygon_to_geodataframe(roi)

    ## Transform dataframes into datagenerators:
    #    instead of iterating over census tracts (dataframes), we will generate one (or more) images per census tract
    print("Setting up data generators...")
    train_dataset, val_dataset = create_datasets(
        AREAS_TO_SAMPLE,
        building_gdf,
        image_size,
        variable,
        train_size=train_size,
        savename=savename,
        save_examples=True,
    )

    # Get tensorboard callbacks and set the custom test loss computation
    #   at the end of each epoch
    callbacks = get_callbacks(
        savename=savename,
        val_dataset=val_dataset,
        logdir=log_dir,
    )

    # Run model
    model, history = run_model(
        model_function=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=n_epochs,
        callbacks=callbacks,
        box_loss=box_loss,
        class_loss=class_loss,
        savename=savename,
    )
    print("Fin del entrenamiento. Generando predicciones...")

    generate_predictions(savename)
    # raise SystemExit


if __name__ == "__main__":

    variable = "damaged"  # "damaged" or "destroyed"
    image_size = 512  # submittion size
    train_size = 10000
    model = "RetinaNet"
    extra = "_fine_tuning_weighted"
    weights = None

    # Train the Model
    run(
        variable=variable,
        model_name=model,
        weights=weights,
        image_size=image_size,
        train_size=train_size,
        n_epochs=200,
        extra=extra,
    )

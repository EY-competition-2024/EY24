##############      Configuración      ##############
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from dotenv import dotenv_values

pd.set_option("display.max_columns", None)
envpath = "/mnt/d/Maestría/Tesis/Repo/scripts/globals.env"
if os.path.isfile(envpath):
    env = dotenv_values(envpath)
else:
    env = dotenv_values(r"D:/Maestría/Tesis/Repo/scripts/globals_win.env")

path_proyecto = env["PATH_PROYECTO"]
path_datain = env["PATH_DATAIN"]
path_dataout = env["PATH_DATAOUT"]
path_scripts = env["PATH_SCRIPTS"]
path_satelites = env["PATH_SATELITES"]
path_logs = env["PATH_LOGS"]
path_outputs = env["PATH_OUTPUTS"]
path_imgs = env["PATH_IMGS"]
# path_programas  = globales[7]
###############################################

import true_metrics
import custom_models
import build_dataset
import grid_predictions
import utils

import os
import sys
import scipy
import random
import pandas as pd
import xarray as xr
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
import cv2
import skimage

# the next 3 lines of code are for my machine and setup due to https://github.com/tensorflow/tensorflow/issues/43174
try:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("No GPU set. Is the GPU already initialized?")


REPO = r"D:\Becas y Proyectos\EY Challenge 2024\EY24"
assert os.path.isdir(REPO), "No existe el repositorio. Revisar la variable REPO del codigo run_model"

PR_DS_POST = xr.open_dataset(
    rf"{REPO}\data\data_in\Post_Event_San_Juan.tif"
)
PR_DS_PRE = xr.open_dataset(
    rf"{REPO}\data\data_in\Pre_Event_San_Juan.tif"
)

BUILDING_GDF = gpd.read_file(
    rf"{REPO}\data\data_in\Buildins Footprint ROI\building_footprint_roi_challenge.shp"
)
BUILDING_GDF.to_crs(epsg=32619, inplace=True)

PR_DS_POST_POLYGON = utils.get_dataset_extent(PR_DS_POST)
PR_DS_PRE_POLYGON = utils.get_dataset_extent(PR_DS_PRE)

def generate_savename(model_name, image_size, sample_size, extra):

    savename = f"{model_name}_size{image_size}_sample{sample_size}{extra}"

    return savename


def create_datasets(
    image_size,
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
        iteration = 0
        image = np.zeros(shape=(3, 0, 0))
        img_correct_shape = (3, image_size, image_size)

        while image.shape != img_correct_shape:
            # Iterate until the image has the correct shape (when selecting borders)

            # Generate the image
            image, boundaries = utils.stacked_image_from_census_tract(
                dataset=PR_DS_POST,  # FIXME: aca le pódría agregar un polígono de la extension de todo PR y creo que estamos, solo samplea de ahí
                polygon=PR_DS_POST_POLYGON,  # IDEM ACA, cambiar a globales
                img_size=image_size,
                n_bands=3,
                stacked_images=[1],
            )

        # FIXME: armar estas funciones
        im_classes, bboxs = utils.get_image_classes_and_boxes(BUILDING_GDF, boundaries)
        
        # FIXME: verificar si esto hace falta o no
        # Reduce quality and process image
        image = utils.process_image(image, resizing_size=image_size)

        # Augment dataset
        if type == "train":
            # FIXME: revisar como aumentar
            image = utils.augment_image(image)
            # image = image

        # np.save(fr"/mnt/d/Maestría/Tesis/Repo/data/data_out/test_arrays/img_{i}_{df_subset.iloc[i].link}.npy", image)
        return image, im_classes, bboxs

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

    train_dataset = train_dataset.shuffle(
        buffer_size=int(train_size),
        seed=825,
        reshuffle_each_iteration=True,
    )

    train_dataset = train_dataset.map(
        lambda i: tf.py_function(  # The actual data generator. Passes the index to the function that will process the data.
            func=get_data,
            inp=[i],
            Tout=[tf.uint8, tf.uint16, tf.float32],  # image, classes, bbox
        ),
    )

    train_dataset = train_dataset.batch(64).prefetch(tf.data.AUTOTUNE)

    ## VAL ##
    # Generator for the index
    val_dataset = tf.data.Dataset.from_generator(
        lambda: list(range(val_size)),  # The index generator,
        tf.uint32,
    )  # Creates a dataset with only the indexes (0, 1, 2, 3, etc.)
    val_dataset = val_dataset.shuffle(
        buffer_size=int(val_size),
        seed=825,
        reshuffle_each_iteration=True,
    )
    val_dataset = val_dataset.map(
        lambda i: tf.py_function(  # The actual data generator. Passes the index to the function that will process the data.
            func=get_data,
            inp=[i],
            Tout=[tf.uint8, tf.uint16, tf.float32],  # image, classes, bbox
        ),
    )
    val_dataset = val_dataset.batch(128)

    if save_examples == True:
        print("saving train/test examples")

        i = 0
        for x in train_dataset.take(5):
            np.save(
                f"{path_outputs}/{savename}_train_example_{i}_imgs", tfds.as_numpy(x)[0]
            )
            np.save(
                f"{path_outputs}/{savename}_train_example_{i}_classes",
                tfds.as_numpy(x)[1],
            )
            np.save(
                f"{path_outputs}/{savename}_train_example_{i}_bbox", tfds.as_numpy(x)[2]
            )
            i += 1

        i = 0
        for x in val_dataset.take(5):
            np.save(
                f"{path_outputs}/{savename}_val_example_{i}_imgs", tfds.as_numpy(x)[0]
            )
            np.save(
                f"{path_outputs}/{savename}_val_example_{i}_classes",
                tfds.as_numpy(x)[1],
            )
            np.save(
                f"{path_outputs}/{savename}_val_example_{i}_bbox", tfds.as_numpy(x)[2]
            )

            i += 1

    print()
    print("Dataset generado!")

    return train_dataset, val_dataset


def get_callbacks(
    savename,
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

    class CustomLossCallback(tf.keras.callbacks.Callback):
        def __init__(self, log_dir):
            super(CustomLossCallback, self).__init__()
            self.log_dir = log_dir

        def on_epoch_end(self, epoch, logs=None):
            # Save model
            if epoch % 10 == 0:
                os.makedirs(f"{path_dataout}/models_by_epoch/{savename}", exist_ok=True)
                self.model.save(
                    f"{path_dataout}/models_by_epoch/{savename}/{savename}_{epoch}",
                    include_optimizer=True,
                )

    tensorboard_callback = TensorBoard(
        log_dir=logdir, histogram_freq=1  # , profile_batch="100,200"
    )
    # use tensorboard --logdir logs/scalars in your command line to startup tensorboard with the correct logs

    # Create an instance of your custom callback
    custom_loss_callback = CustomLossCallback(log_dir=logdir)

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0,  # the training is terminated as soon as the performance measure gets worse from one epoch to the next
        start_from_epoch=50,
        patience=50,  # amount of epochs with no improvements until the model stops
        verbose=2,
        mode="auto",  # the model is stopped when the quantity monitored has stopped decreasing
        restore_best_weights=True,  # restore the best model with the lowest validation error
    )
    # reduce_lr = keras.callbacks.ReduceLROnPlateau(
    #     monitor="val_loss", factor=0.2, patience=10, min_lr=0.0000001
    # )
    model_checkpoint_callback = ModelCheckpoint(
        f"{path_dataout}/models/{savename}",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,  # save the best model
        mode="auto",
        save_freq="epoch",  # save every epoch
    )
    csv_logger = CSVLogger(
        f"{path_dataout}/models_by_epoch/{savename}/{savename}_history.csv", append=True
    )

    return [
        tensorboard_callback,
        # reduce_lr,
        early_stopping_callback,
        model_checkpoint_callback,
        csv_logger,
        custom_loss_callback,
    ]


def run_model(
    model_function: Model,
    lr: float,
    train_dataset: Iterator,
    val_dataset: Iterator,
    loss: str,
    epochs: int,
    metrics: List[str],
    callbacks: List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]],
    savename: str = "",
):
    """This function runs a keras model with the Ranger optimizer and multiple callbacks. The model is evaluated within
    training through the validation generator and afterwards one final time on the test generator.

    Parameters
    ----------
    model_function : Model
        Keras model function like small_cnn()  or adapt_efficient_net().
    lr : float
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

    def get_last_trained_epoch(savename):
        try:
            files = os.listdir(f"{path_dataout}/models_by_epoch/{savename}")
            epochs = [file.split("_")[-1] for file in files]
            epochs = [int(epoch) for epoch in epochs if epoch.isdigit()]
            initial_epoch = max(epochs)
        except:
            os.makedirs(f"{path_dataout}/models_by_epoch/{savename}")
            print("Model not found, running from begining")
            initial_epoch = None

        return initial_epoch

    initial_epoch = get_last_trained_epoch(savename)

    if initial_epoch is None:
        # constructs the model and compiles it
        model = model_function
        model.summary()

        model.compile(
            optimizer="nadam",
            classification_loss="binary_crossentropy",
            box_loss="ciou",
        )
        initial_epoch = 0

    else:
        print("Restoring model...")
        try:
            model_path = (
                f"{path_dataout}/models_by_epoch/{savename}/{savename}_{initial_epoch}"
            )
            model = keras.models.load_model(model_path)  # load the model from file
        except:
            initial_epoch -= 1
            model_path = (
                f"{path_dataout}/models_by_epoch/{savename}/{savename}_{initial_epoch}"
            )
            model = keras.models.load_model(model_path)  # load the model from file
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
        "YOLOv8": custom_models.YOLOv8(n_classes),  # kind=kind),
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
    # FIXME: elegir esto, ver que mierda se usa
    loss = keras.losses.CategoricalCrossentropy()
    metrics = [
        keras.metrics.CategoricalAccuracy(),
        keras.metrics.CategoricalCrossentropy(),
    ]

    return model, loss, metrics


def run(
    model_name: str,
    weights=None,
    image_size=512,
    train_size=10000,
    n_epochs=100,
    extra="",
):
    """Run all the code of this file.

    Parameters
    ----------
    small_sample : bool, optional
        If you just want to check if the code is working, set small_sample to True, by default False
    """

    savename = generate_savename(model_name, image_size, train_size, extra)
    log_dir = f"{path_logs}/{savename}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    ## Set Model & loss function
    model, loss, metrics = set_model_and_loss_function(
        model_name=model_name,
        bands=3,
        resizing_size=image_size,
        weights=weights,
    )

    ## Transform dataframes into datagenerators:
    #    instead of iterating over census tracts (dataframes), we will generate one (or more) images per census tract
    print("Setting up data generators...")
    train_dataset, val_dataset = create_datasets(
        image_size,
        train_size=train_size,
        savename=savename,
        save_examples=True,
    )

    # Get tensorboard callbacks and set the custom test loss computation
    #   at the end of each epoch
    callbacks = get_callbacks(
        savename=savename,
        logdir=log_dir,
    )

    # Run model
    model, history = run_model(
        model_function=model,
        lr=0.0001 * 5,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss=loss,
        metrics=metrics,
        callbacks=callbacks,
        epochs=n_epochs,
        savename=savename,
    )
    print("Fin del entrenamiento")
    # raise SystemExit


if __name__ == "__main__":

    image_size = 640  # YOLO Default
    train_size = 10_000
    model = "YOLOv8"
    extra = ""
    weights = None

    # Train the Model
    run(
        weights=weights,
        image_size=image_size,
        train_size=train_size,
        n_epochs=200,
        extra=extra,
    )

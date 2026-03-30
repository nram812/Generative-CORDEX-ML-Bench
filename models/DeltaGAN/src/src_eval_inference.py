import xarray as xr
import sys
import matplotlib.pyplot as plt
import os
import tqdm
from tqdm import tqdm
import tqdm
import tensorflow as tf
import numpy as np
from functools import partial

AUTOTUNE = tf.data.experimental.AUTOTUNE
from dask.diagnostics import ProgressBar
import json
import pandas as pd
sys.path.append(r'/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/models/DeltaGAN')
from src.layers import *
from src.models import *
from src.gan import *
from src.process_input_training_data import *
from tensorflow.keras import layers
from src.process_input_training_data import *

def create_output(X, y):
    y = y.isel(time=0).drop("time")
    y = y.expand_dims({"time": X.time.size})
    y['time'] = (('time'), X.time.to_index())
    return y
# changed activation function to hyperbolic tangent

# class BicubicUpSampling2D(tf.keras.layers.Layer):
#     def __init__(self, size, **kwargs):
#         super(BicubicUpSampling2D, self).__init__(**kwargs)
#         self.size = size
#
#     def call(self, inputs):
#         # Use tf.shape for dynamic dims — safe during both training and loading
#         new_h = tf.shape(inputs)[1] * self.size[0]
#         new_w = tf.shape(inputs)[2] * self.size[1]
#         return tf.image.resize(inputs, [new_h, new_w],
#                                method=tf.image.ResizeMethod.BILINEAR)
#
#     def compute_output_shape(self, input_shape):
#         # Allows Keras to trace output shape statically during model reconstruction
#         h = input_shape[1] * self.size[0] if input_shape[1] is not None else None
#         w = input_shape[2] * self.size[1] if input_shape[2] is not None else None
#         return (input_shape[0], h, w, input_shape[3])
#
#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({'size': self.size})
#         return config


def load_model_cascade(model_name, epoch=295, model_dir=None):
    # Load config first — needed to rebuild architecture
    with open(f'{model_dir}/{model_name}/config_info.json') as f:
        config = json.load(f)

    input_shape = config["input_shape"]
    output_shape = config["output_shape"]
    n_filters = config["n_filters"]
    n_channels = config["n_input_channels"]
    n_output_channels = config["n_output_channels"]
    orog_bool = config["orog_fields"] == "orog"
    output_varname = config["output_varname"]

    # Rebuild generator exactly as in training
    gan = res_gan(
        input_shape, output_shape, n_filters[:], n_channels, n_output_channels,
        final_activation='linear',
        orog_predictor=orog_bool,
        temp_conditioning=False,
        varname=output_varname
    )

    # Rebuild unet exactly as in training — final_activation depends on varname
    if output_varname == "pr":
        final_activation_unet = tf.keras.layers.LeakyReLU(0.01)
    else:
        final_activation_unet = 'linear'

    unet_model = unet(
        input_shape, output_shape, n_filters[:], n_channels, n_output_channels,
        final_activation=final_activation_unet,
        orog_predictor=orog_bool,
        temp_conditioning=False,
        varname=output_varname
    )

    # Load weights by name — works on full model.save() .h5 files
    gan.load_weights(
        f'{model_dir}/{model_name}/generator_epoch_{epoch}.h5'
    )
    unet_model.load_weights(
        f'{model_dir}/{model_name}/unet_epoch_{epoch}.h5'
    )

    return gan, unet_model, config["ad_loss_factor"]

# def load_model_cascade(model_name, epoch =295, model_dir = None):
#
#     custom_objects = {"BicubicUpSampling2D": BicubicUpSampling2D,
#                                                      "SEBlock":SEBlock, "SelfAttention2D": SelfAttention2D,
#                       "CBAMBlock":CBAMBlock, "TimeFilmLayer":TimeFilmLayer,
#                       "LeakyReLU": tf.keras.layers.LeakyReLU}
#     gan = tf.keras.models.load_model(f'{model_dir}/{model_name}/generator_epoch_{epoch}.h5',
#                                      custom_objects=custom_objects,
#                                      compile=False)
#     with open(f'{model_dir}/{model_name}/config_info.json') as f:
#         config = json.load(f)
#
#         unet = tf.keras.models.load_model(f'{model_dir}/{model_name}/unet_epoch_{epoch}.h5',
#                                           custom_objects=custom_objects, compile=False)
#
#     return gan, unet, config["ad_loss_factor"]


def load_and_normalize_topography_data(filepath):
    # Load the dataset
    topography_data = xr.open_dataset(filepath)

    # Extract variables
    vegt = topography_data.vegt
    orog = topography_data.orog
    he = topography_data.he

    # Print maximum values
    print(f"Max orog: {orog.max().values}, Max he: {he.max().values}, Max vegt: {vegt.max().values}")

    # Normalize the data to the range [0, 1]
    vegt = (vegt - vegt.min()) / (vegt.max() - vegt.min())
    orog = (orog - orog.min()) / (orog.max() - orog.min())
    he = (he - he.min()) / (he.max() - he.min())

    return vegt, orog, he


def normalize_and_stack(concat_dataset, means_filepath, stds_filepath, variables):
    """
    Normalizes specified variables in a dataset with given mean and standard deviation,
    then stacks them along a new 'channel' dimension.

    Parameters:
    concat_dataset (xarray.Dataset): Dataset to normalize.
    means_filepath (str): File path to the dataset containing mean values.
    stds_filepath (str): File path to the dataset containing standard deviation values.
    variables (list): List of variable names to normalize and stack.

    Returns:
    xarray.Dataset: The normalized and stacked dataset.
    """

    # Load mean and standard deviation datasets
    means = xr.open_dataset(means_filepath)
    stds = xr.open_dataset(stds_filepath)

    # Normalize the dataset
    X_norm = (concat_dataset[variables] - means[variables]) / stds[variables]
    X_norm['time'] = pd.to_datetime(X_norm.time.dt.strftime("%Y-%m-%d"))

    # Stack the variables along a new 'channel' dimension
    stacked_X = xr.concat([X_norm[varname] for varname in variables], dim="channel")
    stacked_X['channel'] = (('channel'), variables)
    stacked_X = stacked_X.transpose("time", "lat", "lon", "channel")

    return stacked_X


def expand_conditional_inputs(X, batch_size):
    expanded_image = tf.expand_dims(X, axis=0)  # Shape: (1, 172, 179)

    # Repeat the image to match the desired batch size
    expanded_image = tf.repeat(expanded_image, repeats=batch_size, axis=0)  # Shape: (batch_size, 172, 179)

    # Create a new axis (1) on the last axis
    expanded_image = tf.expand_dims(expanded_image, axis=-1)
    return expanded_image


@tf.function
def predict_batch_residual(gan, unet, latent_vectors, data_batch, orog, time_of_year, config):

    unet_args = [data_batch, orog, time_of_year] if config['orog_fields'] == "orog" else [data_batch, time_of_year]
    intermediate = unet(unet_args, training=False)
    intermediate = tf.clip_by_value(intermediate, clip_value_min=-6.8, clip_value_max=6)

    gan_args = [latent_vectors[0], latent_vectors[1], intermediate, data_batch, orog, time_of_year] \
        if config['orog_fields'] == "orog" else [latent_vectors[0], latent_vectors[1], intermediate, data_batch, time_of_year]
    gan_residual = gan(gan_args,
                     training=False)  # + intermediate
    gan_prediction = gan_residual + intermediate

    return gan_prediction, intermediate



def predict_parallel_resid(model, unet, inputs, output_shape, batch_size, orog_vector, time_of_year, means_output, stds_output, config = None):
    n_iterations = inputs.shape[0] // batch_size
    remainder = inputs.shape[0] - n_iterations * batch_size
    output_shape_gan = output_shape
    output_shape_unet = output_shape_gan.copy()
    dset_gan = []
    dset_unet = []
    with tqdm.tqdm(total=n_iterations, desc="Predicting", unit="batch") as pbar:
        for i in range(n_iterations):
            data_batch = inputs[i * batch_size: (i + 1) * batch_size]
            time_of_year_batch = time_of_year[i * batch_size: (i + 1) * batch_size]
            random_latent_vectors1 = tf.random.normal(shape=(batch_size,) + tuple(model.inputs[0].shape[1:]))
            random_latent_vectors2 = tf.random.normal(shape=(batch_size,) + tuple(model.inputs[1].shape[1:]))
            orog = expand_conditional_inputs(orog_vector, batch_size)
            output_gan, output_unet = predict_batch_residual(model, unet, [random_latent_vectors1, random_latent_vectors2],
                                                              data_batch, orog, time_of_year_batch, config)
            if config["output_varname"] == "pr":
                dset_gan+=tf.clip_by_value(output_gan, clip_value_min=-6.8, clip_value_max=6.8).numpy()[:, :, :, 0].tolist()
            else:
                dset_gan += output_gan.numpy()[:, :, :,  0].tolist()

            dset_unet+=output_unet.numpy()[:, :, :, 0].tolist()
            pbar.update(1)  # Update the progress bar

        if remainder != 0:
            random_latent_vectors1 = tf.random.normal(shape=(batch_size,) + tuple(model.inputs[0].shape[1:]))
            random_latent_vectors2 = tf.random.normal(shape=(batch_size,) + tuple(model.inputs[1].shape[1:]))
            # random_latent_vectors1 = tf.repeat(random_latent_vectors1, repeats=batch_size, axis=0)
            orog = expand_conditional_inputs(orog_vector, remainder)
            output_gan, output_unet = predict_batch_residual(model, unet, [random_latent_vectors1[:remainder], random_latent_vectors2[:remainder]],
                                                             inputs[inputs.shape[0] - remainder:], orog, time_of_year[inputs.shape[0] - remainder:],
                                                             config)

            if config["output_varname"] == "pr":
                dset_gan += tf.clip_by_value(output_gan, clip_value_min=-6.8, clip_value_max=6.8).numpy()[:, :, :,
                            0].tolist()
                dset_unet += output_unet.numpy()[:, :, :, 0].tolist()
            else:
                dset_gan += output_gan.numpy()[:, :, :, 0].tolist()
                dset_unet += output_unet.numpy()[:, :, :, 0].tolist()
            pbar.update(1)  # Update the progress bar


    output_shape_gan[config['output_varname']].values = dset_gan
    output_shape_unet[config['output_varname']].values = dset_unet
    if config["output_varname"] == "tasmax":
        output_shape_gan[config['output_varname']] = output_shape_gan[config['output_varname']] * stds_output[config['output_varname']] + means_output[config['output_varname']]
        output_shape_unet[config['output_varname']] = output_shape_unet[config['output_varname']] * stds_output[
            config['output_varname']] + means_output[config['output_varname']]
    else:
        output_shape_gan[config['output_varname']]  = np.exp( output_shape_gan[config['output_varname']] ) - 1
        output_shape_unet[config['output_varname']] = np.exp(output_shape_unet[config['output_varname']]) - 1
    return output_shape_gan, output_shape_unet






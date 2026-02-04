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
sys.path.append(os.getcwd())
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

def load_model_cascade(model_name, epoch =295, model_dir = None):
    custom_objects = {"BicubicUpSampling2D": BicubicUpSampling2D,
                                                     "SEBlock":SEBlock, "SelfAttention2D": SelfAttention2D,
                      "CBAMBlock":CBAMBlock, "TimeFilmLayer":TimeFilmLayer,
                      "LeakyReLU": tf.keras.layers.LeakyReLU}
    gan = tf.keras.models.load_model(f'{model_dir}/{model_name}/generator_epoch_{epoch}.h5',
                                     custom_objects=custom_objects,
                                     compile=False)
    with open(f'{model_dir}/{model_name}/config_info.json') as f:
        config = json.load(f)

        unet = tf.keras.models.load_model(f'{model_dir}/{model_name}/unet_epoch_{epoch}.h5',
                                          custom_objects=custom_objects, compile=False)

    return gan, unet, config["ad_loss_factor"]


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
    X_norm = (concat_dataset[variables] - means) / stds
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
            dset_gan+=output_gan.numpy()[:, :, :, 0].tolist()
            dset_unet+=output_unet.numpy()[:, :, :, 0].tolist()
            pbar.update(1)  # Update the progress bar

        if remainder != 0:
            random_latent_vectors1 = tf.random.normal(shape=(batch_size,) + tuple(model.inputs[0].shape[1:]))
            random_latent_vectors2 = tf.random.normal(shape=(batch_size,) + tuple(model.inputs[1].shape[1:]))
            # random_latent_vectors1 = tf.repeat(random_latent_vectors1, repeats=batch_size, axis=0)
            orog = expand_conditional_inputs(orog_vector, remainder)
            output_gan, output_unet = predict_batch_residual(model, unet, [random_latent_vectors1[:remainder], random_latent_vectors2[:remainder]],
                                                             inputs[inputs.shape[0] - remainder:], orog, time_of_year[inputs.shape[0] - remainder:], config)
            dset_gan+=output_gan.numpy()[:, :, :, 0].tolist()
            dset_unet+=output_unet.numpy()[:, :, :, 0].tolist()
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




def run_experiments(experiments, epoch_list, model_dir,
                    input_predictors, common_times, output_shape, orog, he, vegt, n_members, batch_size=64):
    """
    Runs inference on some predictor fields in/ out-of-sample


    experiments: list of experiment names in the model_dir folder
    input_predictors: stacked netcdf with dims (time, lat, lon, channel) and normalized data
    common_times: common_times between output_shape data and input_predictors
    output_shape: a netcdf (y_true) that is the same shape as the output prediction, it contains the time metadata
    orog, he, vegt: auxiliary files from ccam
    n_member: the number of ensemble members

    """

    # if the epoch list is only a float convert to a list
    if isinstance(epoch_list, int):
        epoch_list = [epoch_list] * len(experiments)

    # creating empty lists to save outputs
    dsets = []
    lambda_var = []
    for i, experiment in enumerate(experiments):
        if 'cascade' in experiment:
            gan, unet, lambdas = load_model_cascade(experiment,
                                                    epoch_list[i], model_dir)
            if i == 0:
                # first instance is always a unet model
                lambdas = 0.0
                preds = xr.concat([predict_parallel_resid(gan, unet,
                                                          input_predictors.sel(time=common_times).values,
                                                          output_shape.sel(time=common_times),
                                                          batch_size, orog, he, vegt, model_type='unet')
                                   for i in range(n_members)],
                                  dim="member")
            else:
                # do not change lambdas value otherwise
                lambdas = lambdas
                preds = xr.concat([predict_parallel_resid(gan, unet,
                                                          input_predictors.sel(time=common_times).values,
                                                          output_shape.sel(time=common_times),
                                                          batch_size, orog, he, vegt, model_type='GAN')
                                   for i in range(n_members)],
                                  dim="member")


        else:
            gan, lambdas = load_model_reg(experiment,
                                          epoch_list[i], model_dir)
            preds = xr.concat([predict_parallel_v1(gan,
                                                   input_predictors.sel(time=common_times).values,
                                                   output_shape.sel(time=common_times),
                                                   batch_size, orog, he, vegt, model_type='GAN')
                               for i in range(n_members)],
                              dim="member")
        lambda_var.append(lambdas)
        dsets.append(preds)
    dsets = xr.concat(dsets, dim="experiment")
    dsets['experiment'] = (('experiment'), lambda_var)
    dsets = dsets.reindex(experiment=sorted(dsets.experiment.values))
    return dsets




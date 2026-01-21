import xarray as xr
import numpy as np
import pandas as pd
import os
from dask.diagnostics import ProgressBar
import tensorflow as tf
import glob

def prepare_training_data(config, X, y, means, stds, match_index=True):
    """
    Normalize and stack training data features into a single dimension.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing experimental setup, must include 'var_names' key
        with list of variable names to process
    X : xarray.Dataset
        Training predictor data with variables specified in config['var_names']
    y : xarray.Dataset
        Training target data
    means : xarray.Dataset
        Mean values for normalization, with same variables as X
    stds : xarray.Dataset
        Standard deviation values for normalization, with same variables as X
    match_index : bool, optional
        If True, align X and y by time intersection. Default is True

    Returns
    -------
    stacked_X : xarray.DataArray
        Normalized predictor data with features stacked along 'channel' dimension
    y : xarray.Dataset
        Target data, potentially time-aligned if match_index=True
    """
    list_of_vars = config["var_names"]

    # Normalize data using provided means and stds
    X_norm = (X[list_of_vars] - means[list_of_vars].mean()) / stds[list_of_vars].mean()

    # Stack features along channel dimension
    stacked_X = xr.concat([X_norm[varname] for varname in list_of_vars], dim="channel")
    stacked_X['channel'] = (('channel'), list_of_vars)

    # Optionally align X and y by time
    if match_index:
        times = stacked_X.time.to_index().intersection(y.time.to_index())
        stacked_X = stacked_X.sel(time=times)
        y = y.sel(time=times)

    return stacked_X, y


def prepare_static_fields(config):
    """
    Load and normalize static topography data.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing 'static_predictors' key with path to
        topography dataset

    Returns
    -------
    orog : xarray.DataArray
        Orography (topography) data normalized to range [0, 1]
    """
    topography_data = xr.open_dataset(config["static_predictors"])
    orog = topography_data.orog

    # Min-max normalization to [0, 1]
    orog = (orog - orog.min()) / (orog.max() - orog.min())

    return orog


def preprocess_input_data(config, match_index=True):
    """
    Load and preprocess all training data including predictors, targets, and static fields.

    Computes or loads normalization statistics (means and standard deviations) for both
    input and output variables. Creates necessary directories and saves statistics if
    they don't already exist.

    Parameters
    ----------
    config : dict
        Configuration dictionary with the following required keys:
        - 'train_x': path to predictor training data
        - 'train_y': path to target training data
        - 'static_predictors': path to static fields (topography)
        - 'output_folder': directory for saving outputs
        - 'model_name': name of the model
        Optional keys (created if missing):
        - 'mean': path to predictor means
        - 'std': path to predictor standard deviations
        - 'means_output': path to target means
        - 'stds_output': path to target standard deviations
    match_index : bool, optional
        If True, align predictors and targets by time intersection. Default is True

    Returns
    -------
    stacked_X : xarray.DataArray
        Normalized and stacked predictor data
    y : xarray.Dataset
        Target data with boundary coordinates removed
    orog : xarray.DataArray
        Normalized orography data
    config : dict
        Updated configuration dictionary with paths to saved statistics
    """
    orog = prepare_static_fields(config)
    X = xr.open_dataset(config["train_x"])
    y = xr.open_dataset(config["train_y"])
    if not os.path.exists(f'{config["output_folder"]}/{config["model_name"]}'):
        os.makedirs(f'{config["output_folder"]}/{config["model_name"]}')
    if os.path.exists(f'{config["output_folder"]}/{config["model_name"]}'):
        try:
            folder = f'{config["output_folder"]}/{config["model_name"]}'
            for nc_file in glob.glob(f'{config["output_folder"]}/{config["model_name"]}/*.nc'):
                os.remove(nc_file)
                print(f"Removed: {nc_file}")
        except:
            print('no files removed')


    # Load or compute predictor normalization statistics
    try:
        print("Loading existing means and standard deviations")

        means = xr.open_dataset(config["mean"])
        stds = xr.open_dataset(config["std"])
    except:
        print("Computing means and standard deviations")

        means = X.mean(["time"])
        means.to_netcdf(f'{config["output_folder"]}/{config["model_name"]}/predictor_means.nc')
        stds = X.std(["time"])
        stds.to_netcdf(f'{config["output_folder"]}/{config["model_name"]}/predictor_stds.nc')
        config["mean"] = f'{config["output_folder"]}/{config["model_name"]}/predictor_means.nc'
        config["std"] = f'{config["output_folder"]}/{config["model_name"]}/predictor_stds.nc'

    # Load or compute target normalization statistics
    try:
        stds_output = xr.open_dataset(config["stds_output"])
        means_output = xr.open_dataset(config["means_output"])
    except:
        means_output = y.mean("time")
        means_output.to_netcdf(f'{config["output_folder"]}/{config["model_name"]}/target_means.nc')
        stds_output = y.std("time")
        stds_output.to_netcdf(f'{config["output_folder"]}/{config["model_name"]}/target_stds.nc')
        config["means_output"] = f'{config["output_folder"]}/{config["model_name"]}/target_means.nc'
        config["stds_output"] = f'{config["output_folder"]}/{config["model_name"]}/target_stds.nc'

    # Remove boundary coordinate variables if present
    try:
        y = y.drop_vars(["lat_bnds", "lon_bnds"])
    except:
        pass

    # Prepare the training data
    stacked_X, y = prepare_training_data(config, X, y, means, stds, match_index=match_index)

    return stacked_X, y, orog, config


def create_dataset(y, X, eval_times):
    """
    Convert xarray data to TensorFlow dataset format.

    Parameters
    ----------
    y : xarray.Dataset
        Target data containing 'pr' variable
    X : xarray.DataArray
        Predictor data with time coordinate

    Returns
    -------
    tf.data.Dataset
        TensorFlow dataset with tuples of (outputs, inputs) where:
        - outputs: dict with 'pr' key containing target precipitation values
        - inputs: dict with 'X' (predictor values) and 'time_of_year' (day of year)
    """
    output_vars = {
        'pr': tf.convert_to_tensor(y[:eval_times].values, dtype=tf.float32),
    }

    X_tensor = {
        "X": tf.convert_to_tensor(X.values[:eval_times], dtype=tf.float32),
        "time_of_year": tf.convert_to_tensor(X.time.dt.dayofyear.values[:eval_times], dtype=tf.float32)
    }

    return tf.data.Dataset.from_tensor_slices((output_vars, X_tensor))
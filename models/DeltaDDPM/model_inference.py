import os
import sys
import glob
import json
import datetime
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.distribute import MirroredStrategy
from dask.diagnostics import ProgressBar

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

# Configuration
repo_dirs = r'//esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/models/DeltaDDPM'
os.chdir(repo_dirs)
sys.path.append(repo_dirs)

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Custom modules
from src.layers import *
from src.models_dm import *
from src.dm import *
from src.process_input_training_data import *
from src.src_eval_inference import *





# Load configuration
config_file = r'/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep/SA/models/DM_model_test_SA_0.011000-0.0001-0.02_pr_SA_orogorog/config_info.json'\
              #r'/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep/SA/models/GAN_Final_hist_future_Learning_decay_0.01_pr_SA_orogorog/config_info.json'#sys.argv[-1]

with open(config_file, 'r') as f:
    config = json.load(f)


def load_model_dm(model_name, epoch =295, model_dir = None):
    custom_objects = {"BicubicUpSampling2D": BicubicUpSampling2D,
                                                     "SEBlock":SEBlock, "FiLMResidual":FiLMResidual,"SelfAttention2D": SelfAttention2D,"SinusoidalTimeEmbedding":SinusoidalTimeEmbedding, "CBAMBlock":CBAMBlock, "TimeFilmLayer":TimeFilmLayer, "LeakyReLU": tf.keras.layers.LeakyReLU}
    gan = tf.keras.models.load_model(f'{model_dir}/{model_name}/ema_generator_epoch_{epoch}.h5',
                                     custom_objects=custom_objects,
                                     compile=False)
    with open(f'{model_dir}/{model_name}/config_info.json') as f:
        config = json.load(f)

        unet = tf.keras.models.load_model(f'{model_dir}/{model_name}/unet_epoch_{epoch}.h5',
                                          custom_objects=custom_objects, compile=False)

    return gan, unet, config["ad_loss_factor"]
#@tf.function
def predict_batch_residual_diffusion(diffusion_model, unet, scheduler, data_batch, orog, time_of_year,
                                     config, num_inference_steps=100, seed=None):
    """
    Predict a batch using diffusion model with DDIM sampling.

    Args:
        diffusion_model: The trained diffusion model (ema_diffusion)
        unet: The U-Net backbone model
        scheduler: The noise scheduler
        data_batch: Input data batch
        orog: Orography data
        time_of_year: Time of year information
        config: Configuration dictionary
        num_inference_steps: Number of DDIM sampling steps (default 100)
        seed: Random seed for reproducibility

    Returns:
        gan_prediction: Final diffusion model prediction
        intermediate: U-Net intermediate prediction
    """
    batch_size = tf.shape(data_batch)[0]

    # Get U-Net intermediate prediction
    unet_args = [data_batch, orog, time_of_year] if config['orog_fields'] == "orog" else [data_batch, time_of_year]
    intermediate = unet(unet_args, training=False)

    # Initialize random noise
    if seed is not None:
        tf.random.set_seed(seed)
    residual_pred = tf.random.normal(shape=(batch_size, 128, 128, 1))  # Adjust shape as needed

    # DDIM sampling with fewer steps
    timesteps = tf.cast(tf.linspace(scheduler.timesteps - 1, 0, num_inference_steps), tf.int32)

    for i in tf.range(num_inference_steps - 1):
        t = timesteps[i]
        t_next = timesteps[i + 1]

        t_tensor = tf.fill([batch_size, 1], t)

        # Prepare diffusion model arguments
        dm_args = [residual_pred, t_tensor, data_batch, orog, intermediate, time_of_year] \
            if config['orog_fields'] == "orog" else [residual_pred, t_tensor, data_batch, intermediate, time_of_year]

        # Predict noise using the diffusion model
        eps_t = diffusion_model(dm_args, training=False)

        # Extract scheduler values
        alpha_bar_t = tf.gather(scheduler.alpha_bar, t)
        alpha_bar_next = tf.gather(scheduler.alpha_bar, t_next)

        sqrt_alpha_bar_t = tf.sqrt(alpha_bar_t)
        sqrt_1m_alpha_bar_t = tf.sqrt(1.0 - alpha_bar_t)
        sqrt_alpha_bar_next = tf.sqrt(alpha_bar_next)
        sqrt_1m_alpha_bar_next = tf.sqrt(1.0 - alpha_bar_next)

        # Estimate x0 deterministically
        x0 = (residual_pred - sqrt_1m_alpha_bar_t * eps_t) / sqrt_alpha_bar_t
        x0 = tf.clip_by_value(x0, -5.0, 5.0)  # Optional clipping

        # DDIM deterministic update (no noise injection)
        residual_pred = sqrt_alpha_bar_next * x0 + sqrt_1m_alpha_bar_next * eps_t

    # Final prediction is intermediate + residual
    diffusion_prediction = residual_pred + intermediate

    return diffusion_prediction, intermediate


def predict_parallel_resid_diffusion(diffusion_model, unet, scheduler, inputs, output_shape, batch_size,
                                     orog_vector, time_of_year, means_output, stds_output, config=None,
                                     num_inference_steps=100, seed=None):
    """
    Parallel prediction using diffusion model.

    Args:
        diffusion_model: The trained diffusion model (ema_diffusion)
        unet: The U-Net backbone model
        scheduler: The noise scheduler
        inputs: Input data
        output_shape: Template for output shape
        batch_size: Batch size for processing
        orog_vector: Orography vector
        time_of_year: Time of year data
        means_output: Output means for denormalization
        stds_output: Output stds for denormalization
        config: Configuration dictionary
        num_inference_steps: Number of DDIM sampling steps
        seed: Random seed for reproducibility

    Returns:
        output_shape_diffusion: Diffusion model predictions
        output_shape_unet: U-Net predictions
    """
    n_iterations = inputs.shape[0] // batch_size
    remainder = inputs.shape[0] - n_iterations * batch_size
    output_shape_diffusion = output_shape.copy()
    output_shape_unet = output_shape.copy()
    dset_diffusion = []
    dset_unet = []

    with tqdm.tqdm(total=n_iterations + (1 if remainder > 0 else 0), desc="Predicting", unit="batch") as pbar:
        for i in range(n_iterations):
            data_batch = inputs[i * batch_size: (i + 1) * batch_size]
            time_of_year_batch = time_of_year[i * batch_size: (i + 1) * batch_size]
            orog = expand_conditional_inputs(orog_vector, batch_size)

            # Use different seed for each batch if seed is provided
            batch_seed = seed + i if seed is not None else None

            output_diffusion, output_unet = predict_batch_residual_diffusion(
                diffusion_model, unet, scheduler, data_batch, orog, time_of_year_batch,
                config, num_inference_steps, batch_seed
            )

            dset_diffusion += output_diffusion.numpy()[:, :, :, 0].tolist()
            dset_unet += output_unet.numpy()[:, :, :, 0].tolist()
            pbar.update(1)

        # Handle remainder
        if remainder != 0:
            orog = expand_conditional_inputs(orog_vector, remainder)
            batch_seed = seed + n_iterations if seed is not None else None

            output_diffusion, output_unet = predict_batch_residual_diffusion(
                diffusion_model, unet, scheduler,
                inputs[inputs.shape[0] - remainder:],
                orog,
                time_of_year[inputs.shape[0] - remainder:],
                config,
                num_inference_steps,
                batch_seed
            )

            dset_diffusion += output_diffusion.numpy()[:, :, :, 0].tolist()
            dset_unet += output_unet.numpy()[:, :, :, 0].tolist()
            pbar.update(1)

    # Assign predictions to output shapes
    output_shape_diffusion[config['output_varname']].values = dset_diffusion
    output_shape_unet[config['output_varname']].values = dset_unet

    # Denormalize based on variable type
    if config["output_varname"] == "tasmax":
        output_shape_diffusion[config['output_varname']] = (
                output_shape_diffusion[config['output_varname']] * stds_output[config['output_varname']] +
                means_output[config['output_varname']]
        )
        output_shape_unet[config['output_varname']] = (
                output_shape_unet[config['output_varname']] * stds_output[config['output_varname']] +
                means_output[config['output_varname']]
        )
    else:  # Precipitation
        output_shape_diffusion[config['output_varname']] = np.exp(output_shape_diffusion[config['output_varname']]) - 1
        output_shape_unet[config['output_varname']] = np.exp(output_shape_unet[config['output_varname']]) - 1

    return output_shape_diffusion, output_shape_unet

# Transpose orography to standard format
domain = config["region"]
base_path = "/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep"
test_file_path = f"{base_path}/{domain}/test"
output_path_predictions_gan = r'/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/predictions/DeltaGAN'
output_path_predictions_unet = r'/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/predictions/RegressUNet'
# Get test files
files = glob.glob(f"{test_file_path}/*/predictors/*/*.nc", recursive=True)
# Determine experiment type
experiment = "Emul_hist_future" if "future" in config["experiment"] else "ESD_pseudo_reality"
domain_name = f"{domain}_Domain"
scheduler = DiffusionSchedule(timesteps=config["dm_timesteps"],
                              beta_start=config["dm_beta_start"], beta_end=config["dm_beta_end"])
# Process each file
for file in files:
    filename = file.split('/')[-1]
    output_filename = f'Predictions_pr_tasmax_{filename}'

    output_path_gan = file.replace(test_file_path, f'{output_path_predictions_gan}/{domain_name}/{experiment}') \
        .replace('predictors/', '') \
        .replace(filename, output_filename)
    output_path_unet = file.replace(test_file_path, f'{output_path_predictions_unet}/{domain_name}/{experiment}') \
        .replace('predictors/', '') \
        .replace(filename, output_filename)
    # Two paths, one for the GAN, and one for the U-Net.

    Path(output_path_unet).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path_gan).parent.mkdir(parents=True, exist_ok=True)
    # Preprocess and predict
    stacked_X, y, orog, config, means_output, stds_output = preprocess_inference_data(
        config, file, domain_name, experiment
    )
    try:
        orog = orog.transpose("y", "x")
    except:
        orog = orog.transpose("lat", "lon")

    gan, unet_model, ad_loss = load_model_cascade(
        config["model_name"],
        epoch=50,
        model_dir=config["output_folder"]
    )
    dm, unet_model_diff, ad_loss = load_model_dm(
        config["model_name"],
        epoch=265,
        model_dir=config["output_folder"]
    )
    n_times =365
    z1 = []
    for i in range(1):
        dm_preds, unet_preds = predict_parallel_resid_diffusion(dm, unet_model_diff, scheduler,stacked_X.isel(time = slice(0, n_times)).values,
                                   y.isel(time = slice(0, n_times)), 64, orog.values,
                                   stacked_X.isel(time = slice(0, n_times)).time.dt.dayofyear.values, means_output,
                               stds_output, config = config, num_inference_steps=25)
        z1.append(dm)
#gan_outputs = xr.concat(z1, dim ="member")
training_distribution = xr.open_dataset(config["train_y"])

fig, ax = plt.subplots(1,3, figsize = (18, 6))
dm_preds.mean("time").pr.plot(ax = ax[0], vmin =0, vmax =10)
unet_preds.mean("time").pr.plot(ax = ax[1], vmin =0, vmax =10)
training.sel(time = unet_preds.time).mean("time").pr.plot(ax = ax[2], vmin =0, vmax =10)
fig.show()

fig, ax = plt.subplots()
ax.hist(dm_preds.pr.values.ravel(), histtype ='step', color ='r', bins = np.arange(0,800,10))
ax.hist(training.pr.sel(time = unet_preds.time).values.ravel(), histtype ='step', color ='k', bins = np.arange(0,800,10))
ax.hist(unet_preds.pr.sel(time = unet_preds.time).values.ravel(), histtype ='step', color ='b', bins = np.arange(0,800,10))
ax.hist(training_distribution.pr.sel(time = "1961").values.ravel(), histtype ='step', color ='orange', bins = np.arange(0,800,10))
ax.set_yscale('log')
fig.show()

#gan_outputs.isel(time =0).pr.plot(col ="member", col_wrap =2, vmin =0, cmap ='viridis')


training = xr.open_dataset(r'/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep/SA/test/mid_century/target/pr_tasmax_NorESM2-MM_2041-2060.nc')
fig, ax = plt.subplots(1,3, figsize = (18, 6))
vmax=10
gan_outputs.pr.mean(["time"]).isel(member =0).plot(cmap ='BrBG', vmin =0, vmax =vmax,ax = ax[0])
unet.pr.mean("time").plot(cmap ='BrBG', vmin =0, vmax =vmax,ax = ax[1])#i
training.pr.sel(time = unet.time).mean("time").plot(cmap ='BrBG', vmin =0, vmax =vmax,ax = ax[2])
fig.show()
z1 = abs(unet.pr.mean("time") - training.pr.sel(time = unet.time).mean("time"))
abs(z1).mean()


z1 = abs(gan_outputs.isel(member =0).pr.mean("time") - training.pr.sel(time = unet.time).mean("time"))
abs(z1).mean()

z1 = abs(gan_outputs.mean("member").pr.mean("time") - training.pr.sel(time = unet.time).mean("time"))
abs(z1).mean()






if output_varname == "pr":
    config["delta"] = 1
    conversion_factor = 1
    config['conversion_factor'] = conversion_factor
    y[output_varname] = np.log(y[output_varname]* conversion_factor + 1)
elif output_varname == "tasmax":
    y[output_varname]=(y[output_varname] - output_means[output_varname])/ output_stds[output_varname]
    # normalize but preserve spatial gradients.
# the above doesn't conserve spatial gradients in temperature, but argubly it makes the problem easier?

common_times = stacked_X.time.to_index().intersection(y.time.to_index())
stacked_X = stacked_X.sel(time=common_times)
y = y.sel(time=common_times)

try:
    try:
        y = y[[output_varname]].transpose("time", "lat", "lon")
        stacked_X = stacked_X.transpose("time", "lat", "lon","channel")
    except:
        try:
            # This is mostly an error message for the SA region.
            y = y[[output_varname]].drop("bnds").transpose("time", "lat", "lon")
            stacked_X = stacked_X.drop("bnds").transpose("time", "lat", "lon","channel")
        except:
            y = y[[output_varname]].drop("bnds").transpose("time", "lat", "lon")
            stacked_X = stacked_X.transpose("time", "lat", "lon","channel")
except:
    y = y[[output_varname]].transpose("time", "y", "x")
    stacked_X = stacked_X.transpose("time", "lat", "lon","channel")
# rounding to three decimal places
with ProgressBar():
    y = y.load()
    stacked_X = stacked_X.load()

if output_varname == "pr":
    final_activation_unet = tf.keras.layers.LeakyReLU(0.01)
else:
    final_activation_unet = 'linear'


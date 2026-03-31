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
import pandas as pd

AUTOTUNE = tf.data.experimental.AUTOTUNE
from dask.diagnostics import ProgressBar
import json
import pandas as pd
sys.path.append(r'/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/models/FlowMatching')
from src.layers import *
from tensorflow.keras import layers
from src.dm_updated_v3 import *
sys.path.append(r'/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/models/DeltaGAN')
from src.process_input_training_data import *




import os
import json
import numpy as np
import tensorflow as tf
import tqdm
from pathlib import Path


# ============================================================
# MODEL LOADER
# ============================================================
import os
import glob
import json
import numpy as np
import xarray as xr
import tqdm
import tensorflow as tf


def create_output(X, y):
    y = y.isel(time=0).drop("time")
    y = y.expand_dims({"time": X.time.size})
    y['time'] = (('time'), X.time.to_index())
    return y
# changed activation function to hyperbolic tangent


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


# ============================================================
# ODE SOLVERS
# Each solver takes the same signature:
#   (flow_net, x_t, t_val, dt, b_size, x_lr, orog, doy, use_orog)
# and returns the updated x_t
# ============================================================



# ============================================================
# MODEL LOADER
# ============================================================

def load_model_cascade(config, epoch=295):
    """
    Build flow UNet from config, load EMA weights and sigma_z.
    Returns (model, sigma_z).
    """
    model_name = config["model_name"]
    model_dir  = config["output_folder"]
    out_dir    = f"{model_dir}/{model_name}"

    if epoch < 10:
        epoch_str = f"000{epoch}"
    elif epoch < 100:
        epoch_str = f"00{epoch}"
    elif epoch < 1000:
        epoch_str = f"0{epoch}"
    else:
        epoch_str = str(epoch)

    output_varname   = config['output_varname']
    final_activation = tf.keras.layers.LeakyReLU(0.1) if output_varname == "pr" else "linear"
    orog_bool        = config.get('orog_fields', 'None') == 'orog'

    model = build_flow_unet(
        input_size       = config["input_shape"],
        resize_output    = config["output_shape"],
        num_filters      = config["n_filters"],
        num_channels     = config["n_input_channels"],
        orog_predictor   = orog_bool,
        varname          = output_varname,
        final_activation = final_activation
    )

    weights_path = f'{out_dir}/flow_net_epoch{epoch_str}.weights.h5'
    model.load_weights(weights_path)
    print(f"Loaded weights : {weights_path}")
    sigma_value = pd.read_csv(f'{out_dir}/training_history.csv', index_col =0)
    sigma_z = sigma_value.loc[epoch -1]['sigma_z']
    return model, sigma_z


import tensorflow as tf
import numpy as np
import tqdm

import tensorflow as tf
import numpy as np
import tqdm

def predict_parallel_flow(
        model,
        sigma_z,
        inputs,
        output_xr,
        batch_size,
        orog_vector,
        time_of_year,
        means,
        stds,
        config,
):
    """
    AB5 (5th order) explicit multistep method + DPM-style correctors.
    - 1 model call per step
    - Warms up using Euler -> AB2 -> AB3 -> AB4
    """

    varname       = config['output_varname']
    use_orog      = len(model.inputs) == 5
    num_steps     = 60
    t_embed_scale = float(config.get('t_embed_scale', 1000.0))
    H, W          = output_xr[varname].shape[1:3]

    dt    = 1.0 / num_steps
    times = tf.cast(tf.linspace(0.0, 1.0 - dt, num_steps), tf.float32)

    mean_val = means[varname].values# if hasattr(means[varname], 'values') else means[varname]
    std_val  = stds[varname].values#  if hasattr(stds[varname],  'values') else stds[varname]

    n_samples = inputs.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    results   = []

    print(f"sigma_z={sigma_z:.4f} | steps={num_steps} | use_orog={use_orog}")

    with tqdm.tqdm(total=n_batches, desc=f"Flow: {varname}") as pbar:
        for i_batch in range(n_batches):

            start = i_batch * batch_size
            end   = min(start + batch_size, n_samples)
            bsz   = end - start

            # Inputs
            x_lr = tf.cast(inputs[start:end], tf.float32)

            doy_batch = tf.cast(
                tf.expand_dims(time_of_year[start:end], -1),
                tf.float32
            )

            orog_batch = tf.cast(
                tf.repeat(tf.expand_dims(tf.expand_dims(orog_vector, 0), -1), bsz, axis=0),
                tf.float32
            )

            # Initial noise
            #x_t = sigma_z * tf.random.normal(shape=(bsz, H, W, 1))
            noise = tf.random.normal(shape=(bsz, H, W, 1))
            # Clip to 3 standard deviations to remove extreme outliers
            if config["output_varname"] == "pr":
                noise = tf.clip_by_value(noise, -2.25, 2.25)
            x_t = sigma_z * noise

            # Velocity history (store last 4 for AB5)
            v_hist = []

            for i, t_val in enumerate(times):

                t_net = tf.cast(
                    tf.fill((bsz, 1), t_val * t_embed_scale),
                    tf.int32
                )

                def _inp(xt):
                    return ([xt, t_net, x_lr, orog_batch, doy_batch]
                            if use_orog else
                            [xt, t_net, x_lr, doy_batch])

                v_i = model(_inp(x_t), training=False)

                # --- Multistep integration (Warmup to AB5) ---
                if i == 0:
                    # Euler (AB1)
                    x_t = x_t + dt * v_i

                elif i == 1:
                    # AB2
                    x_t = x_t + dt * (1.5 * v_i - 0.5 * v_hist[-1])

                    # small corrector (DPM-style)
                    x_t = x_t + 0.5 * dt * (v_i - v_hist[-1])

                elif i == 2:
                    # AB3
                    x_t = x_t + (dt / 12.0) * (
                        23.0 * v_i
                        - 16.0 * v_hist[-1]
                        + 5.0 * v_hist[-2]
                    )

                    # stabilises high-order drift using recent curvature
                    correction = (v_i - 2.0 * v_hist[-1] + v_hist[-2])
                    x_t = x_t + (dt / 6.0) * correction

                elif i == 3:
                    # AB4
                    x_t = x_t + (dt / 24.0) * (
                        55.0 * v_i
                        - 59.0 * v_hist[-1]
                        + 37.0 * v_hist[-2]
                        - 9.0 * v_hist[-3]
                    )

                    # 3rd-order DPM-style corrector
                    correction = (v_i - 3.0 * v_hist[-1] + 3.0 * v_hist[-2] - v_hist[-3])
                    x_t = x_t + (dt / 24.0) * correction

                else:
                    # AB5
                    x_t = x_t + (dt / 720.0) * (
                        1901.0 * v_i
                        - 2774.0 * v_hist[-1]
                        + 2616.0 * v_hist[-2]
                        - 1274.0 * v_hist[-3]
                        + 251.0 * v_hist[-4]
                    )

                    # 4th-order DPM-style corrector
                    correction = (v_i - 4.0 * v_hist[-1] + 6.0 * v_hist[-2] - 4.0 * v_hist[-3] + v_hist[-4])
                    x_t = x_t + (dt / 120.0) * correction

                # Update history
                v_hist.append(v_i)
                # Keep only the last 4 velocities for memory efficiency
                if len(v_hist) > 4:
                    v_hist.pop(0)

            results.extend(x_t.numpy()[:, :, :, 0].tolist())
            pbar.update(1)

    final_arr = np.array(results)
    print(f"Raw output range: [{final_arr.min():.3f}, {final_arr.max():.3f}]")

    if varname == "pr":
        final_arr = np.maximum(np.exp(final_arr) - 1.0, 0.0)
    else:
        final_arr = final_arr * std_val + mean_val

    output_xr[varname].values = final_arr
    return output_xr

#
# def predict_parallel_flow(
#         model,
#         sigma_z,
#         inputs,
#         output_xr,
#         batch_size,
#         orog_vector,
#         time_of_year,
#         means,
#         stds,
#         config,
# ):
#     """
#     AB3 (3rd order) + lightweight multistep corrector.
#     - 1 model call per step
#     - more stable than plain AB3
#     """
#
#     varname       = config['output_varname']
#     use_orog      = len(model.inputs) == 5
#     num_steps     = 50
#     t_embed_scale = float(config.get('t_embed_scale', 1000.0))
#     H, W          = output_xr[varname].shape[1:3]
#
#     dt    = 1.0 / num_steps
#     times = tf.cast(tf.linspace(0.0, 1.0 - dt, num_steps), tf.float32)
#
#     mean_val = means[varname].values if hasattr(means[varname], 'values') else means[varname]
#     std_val  = stds[varname].values  if hasattr(stds[varname],  'values') else stds[varname]
#
#     n_samples = inputs.shape[0]
#     n_batches = (n_samples + batch_size - 1) // batch_size
#     results   = []
#
#     print(f"sigma_z={sigma_z:.4f} | steps={num_steps} | use_orog={use_orog}")
#
#     with tqdm.tqdm(total=n_batches, desc=f"Flow: {varname}") as pbar:
#         for i_batch in range(n_batches):
#
#             start = i_batch * batch_size
#             end   = min(start + batch_size, n_samples)
#             bsz   = end - start
#
#             # Inputs
#             x_lr = tf.cast(inputs[start:end], tf.float32)
#
#             doy_batch = tf.cast(
#                 tf.expand_dims(time_of_year[start:end], -1),
#                 tf.float32
#             )
#
#             orog_batch = tf.cast(
#                 tf.repeat(tf.expand_dims(tf.expand_dims(orog_vector, 0), -1), bsz, axis=0),
#                 tf.float32
#             )
#
#             # Initial noise
#             x_t = sigma_z * tf.random.normal(shape=(bsz, H, W, 1))
#
#             # Velocity history (store last 2)
#             v_hist = []
#
#             for i, t_val in enumerate(times):
#
#                 t_net = tf.cast(
#                     tf.fill((bsz, 1), t_val * t_embed_scale),
#                     tf.int32
#                 )
#
#                 def _inp(xt):
#                     return ([xt, t_net, x_lr, orog_batch, doy_batch]
#                             if use_orog else
#                             [xt, t_net, x_lr, doy_batch])
#
#                 v_i = model(_inp(x_t), training=False)
#
#                 # --- Multistep integration ---
#                 if i == 0:
#                     # Euler warmup
#                     x_t = x_t + dt * v_i
#
#                 elif i == 1:
#                     # AB2
#                     x_t = x_t + dt * (1.5 * v_i - 0.5 * v_hist[-1])
#
#                     # small corrector (DPM-style)
#                     x_t = x_t + 0.5 * dt * (v_i - v_hist[-1])
#
#                 else:
#                     # AB3
#                     x_t = x_t + dt * (
#                         (23/12) * v_i
#                         - (16/12) * v_hist[-1]
#                         + (5/12)  * v_hist[-2]
#                     )
#
#                     # --- Corrector (key addition) ---
#                     # stabilises high-order drift using recent curvature
#                     correction = (v_i - 2 * v_hist[-1] + v_hist[-2])
#                     x_t = x_t + (dt / 6.0) * correction
#
#                 # Update history
#                 v_hist.append(v_i)
#                 if len(v_hist) > 2:
#                     v_hist.pop(0)
#
#             results.extend(x_t.numpy()[:, :, :, 0].tolist())
#             pbar.update(1)
#
#     final_arr = np.array(results)
#     print(f"Raw output range: [{final_arr.min():.3f}, {final_arr.max():.3f}]")
#
#     if varname == "pr":
#         final_arr = np.maximum(np.exp(final_arr) - 1.0, 0.0)
#     else:
#         final_arr = final_arr * std_val + mean_val
#
#     output_xr[varname].values = final_arr
#     return output_xr
#

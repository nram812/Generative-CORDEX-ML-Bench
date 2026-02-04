"""
Inference script for DeltaDDPM and RegressUNet models on CORDEX climate data - TEST SET.

This script processes test data through trained diffusion and U-Net models for multiple
regions, experiment types, and orography configurations.
"""

import os
import sys
import glob
import json
from pathlib import Path

import numpy as np
import xarray as xr
import tensorflow as tf
import tqdm

# Configuration
REPO_DIR = r'//esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/models/DeltaDDPM'
os.chdir(REPO_DIR)
sys.path.append(REPO_DIR)

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Custom modules
from src.layers import *
from src.models_dm import *
from src.dm import *
from src.process_input_training_data import *
from src.src_eval_inference import *

# Configuration
REGIONS = ["NZ", "ALPS", "SA"]
EXPERIMENT_TYPES = ["ESD", "Hist_future"]
OROG_TYPES = ["orog", "no_orog"]
BASE_PATH = "/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep"


# Model loading epoch and parameters
MODEL_EPOCH = 250
BATCH_SIZE = 64
NUM_INFERENCE_STEPS = 40  # DDIM sampling steps
PREDICTIONS_BASE = f"/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/predictions_test_dm_{MODEL_EPOCH}"
n_years_test = 3
if not os.path.exists(PREDICTIONS_BASE):
    os.makedirs(PREDICTIONS_BASE)


def load_model_dm(model_name, epoch=MODEL_EPOCH, model_dir=None):
    """
    Load diffusion model and U-Net.

    Args:
        model_name: Name of the model
        epoch: Epoch number to load
        model_dir: Directory containing the model

    Returns:
        tuple: (diffusion_model, unet, ad_loss_factor)
    """
    custom_objects = {
        "BicubicUpSampling2D": BicubicUpSampling2D,
        "SEBlock": SEBlock,
        "FiLMResidual": FiLMResidual,
        "SelfAttention2D": SelfAttention2D,
        "SinusoidalTimeEmbedding": SinusoidalTimeEmbedding,
        "CBAMBlock": CBAMBlock,
        "TimeFilmLayer": TimeFilmLayer,
        "LeakyReLU": tf.keras.layers.LeakyReLU
    }

    diffusion_model = tf.keras.models.load_model(
        f'{model_dir}/{model_name}/ema_generator_epoch_{epoch}.h5',
        custom_objects=custom_objects,
        compile=False
    )

    with open(f'{model_dir}/{model_name}/config_info.json') as f:
        config = json.load(f)

    unet = tf.keras.models.load_model(
        f'{model_dir}/{model_name}/unet_epoch_{epoch}.h5',
        custom_objects=custom_objects,
        compile=False
    )

    return diffusion_model, unet, config["ad_loss_factor"]


def predict_batch_residual_diffusion(diffusion_model, unet, scheduler, data_batch, orog, time_of_year,
                                     config, num_inference_steps=NUM_INFERENCE_STEPS, seed=None):
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
        tuple: (diffusion_prediction, intermediate)
    """
    batch_size = tf.shape(data_batch)[0]

    # Get U-Net intermediate prediction
    unet_args = [data_batch, orog, time_of_year] if config['orog_fields'] == "orog" else [data_batch, time_of_year]
    intermediate = unet(unet_args, training=False)

    # Initialize random noise
    if seed is not None:
        tf.random.set_seed(seed)
    residual_pred = tf.random.normal(shape=(batch_size, 128, 128, 1))

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
        x0 = tf.clip_by_value(x0, -5.0, 5.0)

        # DDIM deterministic update (no noise injection)
        residual_pred = sqrt_alpha_bar_next * x0 + sqrt_1m_alpha_bar_next * eps_t

    # Final prediction is intermediate + residual
    diffusion_prediction = residual_pred + intermediate

    return diffusion_prediction, intermediate


def predict_parallel_resid_diffusion(diffusion_model, unet, scheduler, inputs, output_shape, batch_size,
                                     orog_vector, time_of_year, means_output, stds_output, config=None,
                                     num_inference_steps=NUM_INFERENCE_STEPS, seed=None):
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
        tuple: (output_shape_diffusion, output_shape_unet)
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


def get_config_file_path(region, experiment_type, variable, orog_flag):
    """
    Construct the configuration file path for a given setup.

    Args:
        region: Region code (NZ, ALPS, SA)
        experiment_type: Either 'ESD' or 'Hist_future'
        variable: Climate variable ('pr' or 'tasmax')
        orog_flag: Orography flag ('orog' or 'None')

    Returns:
        str: Path to configuration file
    """
    if experiment_type == "ESD":
        experiment_str = "ESD"
    else:
        experiment_str = "hist_future"

    return (f'{BASE_PATH}/{region}/models/'
            f'DM_model_Final_{experiment_str}_1000-0.0001-0.02_{variable}_{region}_orog{orog_flag}/'
            f'config_info.json')


def get_experiment_name(config, orog_flag):
    """
    Determine experiment name from config and orography flag.

    Args:
        config: Configuration dictionary
        orog_flag: Orography flag ('orog' or 'None')

    Returns:
        str: Experiment name
    """
    if "future" in config["experiment"]:
        base_name = "Emul_hist_future"
    else:
        base_name = "ESD_pseudo_reality"

    suffix = "_orog" if orog_flag == 'orog' else "_no_orog"
    return f"{base_name}{suffix}"


def process_file(file_path, config_pr, config_tasmax, region, orog_flag,
                 output_path_diffusion_base, output_path_unet_base, test_file_path, n_members=5):
    """
    Process a single test file through both diffusion and U-Net models.

    Args:
        file_path: Path to input file
        config_pr: Configuration for precipitation model
        config_tasmax: Configuration for temperature model
        region: Region code
        orog_flag: Orography flag
        output_path_diffusion_base: Base output path for diffusion predictions
        output_path_unet_base: Base output path for U-Net predictions
        test_file_path: Base path for test files
        n_members: Number of ensemble members to generate
    """
    filename = file_path.split('/')[-1]
    output_filename = f'Predictions_pr_tasmax_test_set.nc'

    domain_name = f"{region}_Domain"
    experiment = get_experiment_name(config_tasmax, orog_flag)

    # Construct output paths
    output_path_diffusion = f'{output_path_diffusion_base}/{domain_name}/{experiment}/{output_filename}'
    output_path_unet = f'{output_path_unet_base}/{domain_name}/{experiment}/{output_filename}'

    # Create output directories
    Path(output_path_diffusion).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path_unet).parent.mkdir(parents=True, exist_ok=True)

    # Preprocess data
    print(f"Processing: {filename}")
    stacked_X, y, orog, config_tasmax, means_output, stds_output = preprocess_inference_data(
        config_tasmax, file_path, domain_name, experiment
    )

    # Select only the last n_years_test of data (test set)
    stacked_X = stacked_X.isel(time=slice(-365 * n_years_test, None))
    y = y.isel(time=slice(-365 * n_years_test, None))

    # Transpose orography to standard format
    try:
        orog = orog.transpose("y", "x")
    except:
        orog = orog.transpose("lat", "lon")

    # Initialize schedulers
    scheduler_pr = DiffusionSchedule(
        timesteps=config_pr["dm_timesteps"],
        beta_start=config_pr["dm_beta_start"],
        beta_end=config_pr["dm_beta_end"]
    )

    scheduler_tasmax = DiffusionSchedule(
        timesteps=config_tasmax["dm_timesteps"],
        beta_start=config_tasmax["dm_beta_start"],
        beta_end=config_tasmax["dm_beta_end"]
    )

    # Load models
    print("Loading precipitation diffusion model...")
    dm_pr, unet_model_pr, _ = load_model_dm(
        config_pr["model_name"],
        epoch=MODEL_EPOCH,
        model_dir=config_pr["output_folder"]
    )

    print("Loading temperature diffusion model...")
    dm_tasmax, unet_model_tasmax, _ = load_model_dm(
        config_tasmax["model_name"],
        epoch=MODEL_EPOCH,
        model_dir=config_tasmax["output_folder"]
    )

    # Make predictions for multiple ensemble members
    diffusion_preds = []

    for member_idx in range(n_members):
        print(f"Generating ensemble member {member_idx + 1}/{n_members}...")

        print("Generating precipitation predictions...")
        diffusion_preds_pr, unet_preds_pr = predict_parallel_resid_diffusion(
            dm_pr, unet_model_pr, scheduler_pr, stacked_X.values, y, BATCH_SIZE,
            orog.values, stacked_X.time.dt.dayofyear.values, means_output, stds_output,
            config=config_pr, num_inference_steps=NUM_INFERENCE_STEPS, seed=member_idx * 1000
        )

        print("Generating temperature predictions...")
        diffusion_preds_tasmax, unet_preds_tasmax = predict_parallel_resid_diffusion(
            dm_tasmax, unet_model_tasmax, scheduler_tasmax, stacked_X.values, y, BATCH_SIZE,
            orog.values, stacked_X.time.dt.dayofyear.values, means_output, stds_output,
            config=config_tasmax, num_inference_steps=NUM_INFERENCE_STEPS, seed=member_idx * 1000
        )

        # Merge predictions
        merged_preds_diffusion = xr.merge([
            diffusion_preds_pr[['pr']],
            diffusion_preds_tasmax[['tasmax']]
        ])

        diffusion_preds.append(merged_preds_diffusion)

    # Combine ensemble members
    diffusion_preds = xr.concat(diffusion_preds, dim="member")
    diffusion_preds['member'] = (('member'), np.arange(n_members))

    # Save U-Net predictions (only need one set)
    merged_preds_unet = xr.merge([
        unet_preds_pr[['pr']],
        unet_preds_tasmax[['tasmax']]
    ])

    # Save predictions
    print("Saving predictions...")
    diffusion_preds.to_netcdf(output_path_diffusion)
    merged_preds_unet.to_netcdf(output_path_unet)

    print(f"Completed: {output_filename}\n")


def main():
    """Main execution function."""

    for orog_type in OROG_TYPES:
        output_path_diffusion_base = f'{PREDICTIONS_BASE}/DeltaDDPM_{orog_type}'
        output_path_unet_base = f'{PREDICTIONS_BASE}/RegressUNet_DM_{orog_type}'

        for experiment_type in EXPERIMENT_TYPES:
            for region in REGIONS:
                print(f"\n{'=' * 80}")
                print(f"Processing: {region} - {experiment_type} - {orog_type}")
                print(f"{'=' * 80}\n")

                orog_flag = 'orog' if orog_type == "orog" else 'None'

                # Load configurations
                config_file_pr = get_config_file_path(
                    region, experiment_type, 'pr', orog_flag
                )
                config_file_tasmax = get_config_file_path(
                    region, experiment_type, 'tasmax', orog_flag
                )

                try:
                    with open(config_file_pr, 'r') as f:
                        config_pr = json.load(f)
                    with open(config_file_tasmax, 'r') as f:
                        config_tasmax = json.load(f)
                except FileNotFoundError as e:
                    print(f"Warning: Config file not found - {e}")
                    print("Skipping this configuration...\n")
                    pass

                print(f"Model: {config_tasmax['model_name']}")

                # Get test file from train_x (as in GAN script)
                test_file_path = f"{BASE_PATH}/{region}/test"
                file = config_tasmax["train_x"]

                print(f"Using file: {file}")

                try:
                    process_file(
                        file, config_pr, config_tasmax, region, orog_flag,
                        output_path_diffusion_base, output_path_unet_base,
                        test_file_path
                    )
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    print("Continuing with next configuration...\n")
                    pass

    print("\n" + "=" * 80)
    print("All processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
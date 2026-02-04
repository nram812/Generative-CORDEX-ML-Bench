"""
Inference script for DeltaGAN and RegressUNet models on CORDEX climate data.

This script processes test data through trained GAN and U-Net models for multiple
regions, experiment types, and orography configurations.
"""

import os
import sys
import glob
import json
from pathlib import Path

import xarray as xr
import tensorflow as tf

# Configuration
REPO_DIR = r'//esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/models/DeltaGAN'
os.chdir(REPO_DIR)
sys.path.append(REPO_DIR)

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Custom modules
from src.layers import *
from src.models import *
from src.gan import *
from src.process_input_training_data import *
from src.src_eval_inference import *

# Configuration
REGIONS = ["NZ", "ALPS", "SA"]
EXPERIMENT_TYPES = ["ESD", "Hist_future"]
OROG_TYPES = ["orog", "no_orog"]
BASE_PATH = "/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep"
PREDICTIONS_BASE = "/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/predictions_245_epoch"

# Model loading epoch
MODEL_EPOCH = 245
BATCH_SIZE = 128
n_years_test =3

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
            f'GAN_Final_{experiment_str}_Learning_decay_0.01_{variable}_{region}_orog{orog_flag}/'
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
                 output_path_gan_base, output_path_unet_base, test_file_path,n_members =5):
    """
    Process a single test file through both GAN and U-Net models.

    Args:
        file_path: Path to input file
        config_pr: Configuration for precipitation model
        config_tasmax: Configuration for temperature model
        region: Region code
        orog_flag: Orography flag
        output_path_gan_base: Base output path for GAN predictions
        output_path_unet_base: Base output path for U-Net predictions
        test_file_path: Base path for test files
    """
    filename = file_path.split('/')[-1]
    output_filename = f'Predictions_pr_tasmax_test_set.nc'

    domain_name = f"{region}_Domain"
    experiment = get_experiment_name(config_tasmax, orog_flag)

    # Construct output paths
    output_path_gan = f'{output_path_gan_base}/{domain_name}/{experiment}/{output_filename}'

    output_path_unet = f'{output_path_unet_base}/{domain_name}/{experiment}/{output_filename}'

    # Create output directories
    Path(output_path_gan).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path_unet).parent.mkdir(parents=True, exist_ok=True)

    # Preprocess data
    print(f"Processing: {filename}")
    stacked_X, y, orog, config_tasmax, means_output, stds_output = preprocess_inference_data(
        config_tasmax, file_path, domain_name, experiment
    )
    stacked_X = stacked_X.isel(time = slice(-365*n_years_test, None))
    y = y.isel(time = slice(-365*n_years_test, None))


    # Transpose orography to standard format
    try:
        orog = orog.transpose("y", "x")
    except:
        orog = orog.transpose("lat", "lon")

    # Load models
    print("Loading precipitation model...")
    gan_pr, unet_model_pr, _ = load_model_cascade(
        config_pr["model_name"],
        epoch=MODEL_EPOCH,
        model_dir=config_pr["output_folder"]
    )

    print("Loading temperature model...")
    gan_tasmax, unet_model_tasmax, _ = load_model_cascade(
        config_tasmax["model_name"],
        epoch=MODEL_EPOCH,
        model_dir=config_tasmax["output_folder"]
    )

    # Make predictions
    # ERROR FIX: Removed undefined 'n_times' variable
    print("Generating precipitation predictions...")

    gan_preds = []
    for _ in range(n_members):

        gan_preds_pr, unet_preds_pr = predict_parallel_resid(
            gan_pr, unet_model_pr, stacked_X.values, y, BATCH_SIZE, orog.values,
            stacked_X.time.dt.dayofyear.values, means_output, stds_output,
            config=config_pr
        )

        print("Generating temperature predictions...")
        gan_preds_tasmax, unet_preds_tasmax = predict_parallel_resid(
            gan_tasmax, unet_model_tasmax, stacked_X.values, y, BATCH_SIZE, orog.values,
            stacked_X.time.dt.dayofyear.values, means_output, stds_output,
            config=config_tasmax
        )

        # Merge and save predictions
        print("Saving predictions...")
        merged_preds_gan = xr.merge([
            gan_preds_pr[['pr']],
            gan_preds_tasmax[['tasmax']]
        ])
        merged_preds_unet = xr.merge([
            unet_preds_pr[['pr']],
            unet_preds_tasmax[['tasmax']]
        ])
        gan_preds.append(merged_preds_gan)
        #unet_preds.append(merged_preds_unet)
    gan_preds = xr.concat(gan_preds, dim = "member")
    gan_preds['member'] = (('member'), np.arange(n_members))

    gan_preds.to_netcdf(output_path_gan)
    merged_preds_unet.to_netcdf(output_path_unet)

    print(f"Completed: {output_filename}\n")


def main():
    """Main execution function."""

    for orog_type in OROG_TYPES:
        output_path_gan_base = f'{PREDICTIONS_BASE}/DeltaGAN_{orog_type}'
        output_path_unet_base = f'{PREDICTIONS_BASE}/RegressUNet_{orog_type}'

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

                # Get test files
                test_file_path = f"{BASE_PATH}/{region}/test"
                file = config_tasmax["train_x"]
                print("file")
                try:
                    process_file(
                        file, config_pr, config_tasmax, region, orog_flag,
                        output_path_gan_base, output_path_unet_base,
                        test_file_path
                    )
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    print("Continuing with next file...\n")
                    continue

    print("\n" + "=" * 80)
    print("All processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
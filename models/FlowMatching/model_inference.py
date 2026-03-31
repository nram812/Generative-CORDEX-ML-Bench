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

import numpy as np
import xarray as xr
import tensorflow as tf

REPO_DIR = r'//esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/models/FlowMatching'
os.chdir(REPO_DIR)
sys.path.append(REPO_DIR)

AUTOTUNE = tf.data.experimental.AUTOTUNE

from src.layers import *
from src.models import *
from src.process_input_training_data import *
from src.src_eval_inference import *

REGIONS         = [sys.argv[-2]]#["NZ", "ALPS", "SA"]
EXPERIMENT_TYPES = [sys.argv[-1]]#"ESD", "Hist_future"]
MODEL_EPOCH     = 495
BATCH_SIZE      = 128
OROG_TYPES      = ["orog"]
BASE_PATH       = "/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep"
PREDICTIONS_BASE = f"/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/OUTPUT/FullDataset/Flow_{MODEL_EPOCH}_V4"

if not os.path.exists(PREDICTIONS_BASE):
    os.makedirs(PREDICTIONS_BASE)


def get_config_file_path(region, experiment_type, variable, orog_flag):
    if experiment_type == "ESD":
        experiment_str = "ESD"
    else:
        experiment_str = "hist_future"

    # BUG 1 FIX: added missing closing parenthesis
    return (
        f'{BASE_PATH}/{region}/models/'
        f'FlowMatchingFinal_v2_Final_{experiment_str}_0.01_{variable}_{region}_orog{orog_flag}/'
        f'config_info.json'
    )


def get_experiment_name(config, orog_flag):
    if "future" in config["experiment"]:
        base_name = "Emul_hist_future"
    else:
        base_name = "ESD_pseudo_reality"

    suffix = "_orog" if orog_flag == 'orog' else "_no_orog"
    return f"{base_name}{suffix}"


def process_file(file_path, config_pr, config_tasmax, region, orog_flag,
                 output_path_diffusion_base, test_file_path,
                 n_members=5, solver="euler", MODEL_EPOCH = MODEL_EPOCH, BATCH_SIZE = BATCH_SIZE):
    """
    Process a single test file through both flow matching models
    (precipitation and temperature), generate ensemble members,
    and save merged predictions.
    """
    filename  = file_path.split('/')[-1]
    output_filename = f'Predictions_pr_tasmax_{filename}'
    domain_name = f"{region}_Domain"
    experiment  = get_experiment_name(config_tasmax, orog_flag)

    # Output path (flow predictions only — UNet removed)
    output_path_diffusion = file_path.replace(
        test_file_path,
        f'{output_path_diffusion_base}/{domain_name}/{experiment}'
    ).replace('predictors/', '').replace(filename, output_filename)
    Path(output_path_diffusion).parent.mkdir(parents=True, exist_ok=True)

    # Preprocess
    print(f"Processing: {filename}")
    stacked_X, y, orog, config_tasmax, means_output, stds_output = preprocess_inference_data(
        config_tasmax, file_path, domain_name, experiment)
    try:
        orog = orog.transpose("y", "x")
    except Exception:
        orog = orog.transpose("lat", "lon")

    # Load models
    print("Loading precipitation model...")
    flow_model_pr, sigma_z_pr = load_model_cascade(config_pr,
        epoch=MODEL_EPOCH
    )
    print("Loading temperature model...")
    flow_model_tasmax, sigma_z_tasmax = load_model_cascade(config_tasmax,
                                           epoch=MODEL_EPOCH
    )
    time_of_year_values = stacked_X.time.dt.dayofyear.values
    config_pr["num_ode_steps"] = 30
    config_tasmax["num_ode_steps"] = 30
    gan_preds = []
    for iii in range(n_members):
        print(f"Generating ensemble member {iii + 1}/{n_members}...")

        flow_preds_pr = predict_parallel_flow(
            model=flow_model_pr,
            inputs=stacked_X.values,
            sigma_z=sigma_z_pr,
            output_xr=y.copy(),
            batch_size=BATCH_SIZE,
            orog_vector=orog.values,
            time_of_year=time_of_year_values,
            means=means_output,
            stds=stds_output,
            config=config_pr
        )

        print("  -> Temperature...")
        flow_preds_tasmax = predict_parallel_flow(
            model=flow_model_tasmax,
            inputs=stacked_X.values,
            sigma_z=sigma_z_tasmax,
            output_xr=y.copy(),
            batch_size=BATCH_SIZE,
            orog_vector=orog.values,
            time_of_year=time_of_year_values,
            means=means_output,
            stds=stds_output,
            config=config_tasmax
        )
        merged = xr.merge([
            flow_preds_pr[['pr']],
            flow_preds_tasmax[['tasmax']]
        ])
        gan_preds.append(merged)

    # Concatenate ensemble members
    gan_preds = xr.concat(gan_preds, dim="member")
    gan_preds['member'] = (('member',), np.arange(n_members))
    gan_preds = gan_preds.astype('float32')

    # BUG 3 FIX: pass encoding to to_netcdf
    encoding = {var: {'zlib': True, 'complevel': 5} for var in gan_preds.data_vars}
    print(f"Saving predictions to {output_path_diffusion} ...")
    gan_preds.to_netcdf(output_path_diffusion, encoding=encoding)

    # BUG 2 & 4 FIX: removed merged_preds_unet and output_path_unet
    # (no UNet in this pipeline — flow models only)

    print(f"Completed: {output_filename}\n")

def main():
    """Main execution function."""

    for orog_type in OROG_TYPES:
        output_path_gan_base = f'{PREDICTIONS_BASE}/FlowMatching_{orog_type}'

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
                files = glob.glob(
                    f"{test_file_path}/*/predictors/*/*.nc",
                    recursive=True
                )

                if not files:
                    print(f"Warning: No test files found in {test_file_path}")
                    pass

                print(f"Found {len(files)} test file(s)")

                # Process each file
                for file in files:
                    print("running file", file)
                    try:
                        process_file(
                            file, config_pr, config_tasmax, region, orog_flag,
                            output_path_gan_base,
                            test_file_path
                        )
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
                        print("Continuing with next file...\n")
                        pass

    print("\n" + "=" * 80)
    print("All processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
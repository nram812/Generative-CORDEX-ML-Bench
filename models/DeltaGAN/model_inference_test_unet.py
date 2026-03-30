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
version = "test"
if version == "test":
    REGIONS = ["NZ"]  #
    EXPERIMENT_TYPES = ["Emul_hist_future"]
    MODEL_EPOCH = 95
else:
    REGIONS = ["NZ", "ALPS", "SA"]
    EXPERIMENT_TYPES = ["ESD", "Hist_future"]
    MODEL_EPOCH = 115



BATCH_SIZE = 128
OROG_TYPES = ["orog"]#["no_orog", "orog"]#, "no_orog"]
BASE_PATH = "/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep"
PREDICTIONS_BASE = f"/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/OUTPUT/FullDataset/GAN_{MODEL_EPOCH}_V4"
if not os.path.exists(PREDICTIONS_BASE):
    os.makedirs(PREDICTIONS_BASE)
custom_override = r'//esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep/NZ/models/GAN_Final_hist_future_Learning_decay_LittleReluconfig_0.01_tasmax_NZ_orogorog/config_info.json'
# Model loading epoch and parameters


def get_config_file_path(region, experiment_type, variable, orog_flag, custom_config_override = custom_override):
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
    if custom_config_override is not None:
        return custom_config_override
    else:
        return (f'{BASE_PATH}/{region}/models/'
            f'GAN_{version}_{experiment_str}_Learning_decay_0.01_{variable}_{region}_orog{orog_flag}/'
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
                 output_path_diffusion_base, output_path_unet_base, test_file_path,
                 n_members=1, temp_conditioning=None):
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
        temp_conditioning: Temperature conditioning type
    """
    try:
        temp_conditioning = config_tasmax.get("temp_conditioning", "None")
    except:
        temp_conditioning = "None"

    filename = file_path.split('/')[-1]
    output_filename = f'Predictions_pr_tasmax_{filename}'

    domain_name = f"{region}_Domain"
    experiment = get_experiment_name(config_tasmax, orog_flag)

    # Construct output paths
    output_path_diffusion = file_path.replace(
        test_file_path,
        f'{output_path_diffusion_base}/{domain_name}/{experiment}'
    ).replace('predictors/', '').replace(filename, output_filename)

    output_path_unet = file_path.replace(
        test_file_path,
        f'{output_path_unet_base}/{domain_name}/{experiment}'
    ).replace('predictors/', '').replace(filename, output_filename)

    # Create output directories
    Path(output_path_diffusion).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path_unet).parent.mkdir(parents=True, exist_ok=True)

    # Preprocess data
    print(f"Processing: {filename}")
    stacked_X, y, orog, config_tasmax, means_output, stds_output = preprocess_inference_data(
        config_tasmax, file_path, domain_name, experiment
    )
    print(y.time, stacked_X.time)
    stacked_X = stacked_X.isel(time = slice(-700,None))
    y = y.isel(time = slice(-700,None))

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
    for iii in range(n_members):
        print(f"Generating ensemble member {iii + 1}/{n_members}...")

        # Prepare time of year values
        #if temp_conditioning == "None":
        time_of_year_values = stacked_X.time.dt.dayofyear.values
        # else:
        #     try:
        #         time_of_year_values = stacked_X.sel(channel='t_850').mean(["lat", "lon"]).values
        #     except:
        #         time_of_year_values = stacked_X.sel(channel='t_850').mean(["y", "x"]).values

        print("Generating temperature predictions...")
        gan_preds_tasmax, unet_preds_tasmax = predict_parallel_resid(
            gan_tasmax, unet_model_tasmax, stacked_X.values, y[['tasmax']], BATCH_SIZE, orog.values,
            time_of_year_values, means_output, stds_output,
            config=config_tasmax)

        # Merge and save predictions
        print("Saving predictions...")
        merged_preds_gan = gan_preds_tasmax[['tasmax']]
        #merged_preds_gan = merged_preds_gan.astype('float32')
        if iii ==0:
            merged_preds_unet = unet_preds_tasmax[['tasmax']]

        gan_preds.append(merged_preds_gan)

    gan_preds = xr.concat(gan_preds, dim = "member")
    gan_preds['member'] = (('member'), np.arange(n_members))
    gan_preds = gan_preds.astype('float32')
    merged_preds_unet = merged_preds_unet.astype('float32')
    print("Saving predictions...")
    #gan_preds.to_netcdf(output_path_diffusion)

    encoding_unet = {var: {'zlib': True, 'complevel': 5} for var in merged_preds_unet.data_vars}
    #merged_preds_unet.to_netcdf(output_path_unet)

    print(f"Completed: {output_filename}\n")
    return merged_preds_unet

@tf.function
def predict_batch_residual(gan, unet, latent_vectors, data_batch, orog, time_of_year, config):

    unet_args = [data_batch, orog, time_of_year] if config['orog_fields'] == "orog" else [data_batch, time_of_year]
    intermediate = unet(unet_args, training=False)
    #intermediate = tf.clip_by_value(intermediate, clip_value_min=-6.8, clip_value_max=6)

    # gan_args = [latent_vectors[0], latent_vectors[1], intermediate, data_batch, orog, time_of_year] \
    #     if config['orog_fields'] == "orog" else [latent_vectors[0], latent_vectors[1], intermediate, data_batch, time_of_year]
    # gan_residual = gan(gan_args,
    #                  training=False)  # + intermediate
    # gan_prediction = gan_residual + intermediate

    return intermediate, intermediate



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
        output_shape_gan[config['output_varname']] = (output_shape_gan[config['output_varname']]-8) * stds_output[config['output_varname']].mean() + means_output[config['output_varname']].mean()
        output_shape_unet[config['output_varname']] = (output_shape_unet[config['output_varname']]-8) * stds_output[
            config['output_varname']].mean() + means_output[config['output_varname']].mean()
    else:
        output_shape_gan[config['output_varname']]  = np.exp( output_shape_gan[config['output_varname']] ) - 1
        output_shape_unet[config['output_varname']] = np.exp(output_shape_unet[config['output_varname']]) - 1
    return output_shape_gan, output_shape_unet

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
                        test_preds = process_file(
                            file, config_pr, config_tasmax, region, orog_flag,
                            output_path_gan_base, output_path_unet_base,
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
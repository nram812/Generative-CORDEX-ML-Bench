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
repo_dirs = r'//esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/models/DeltaGAN'
os.chdir(repo_dirs)
sys.path.append(repo_dirs)

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Custom modules
from src.layers import *
from src.models import *
from src.gan import *
from src.process_input_training_data import *
from src.src_eval_inference import *

import glob
config_files = glob.glob(r'/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep/*/models/GAN*/config_info.json', recursive=True)
regions = ["NZ", "ALPS","SA"] # ADD SA
experiment_types = ["ESD", "Hist_future"]
orog_types = ["orog","no_orog"]
base_path = "/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep"
output_path_predictions_gan = f'/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/predictions/DeltaGAN'
output_path_predictions_unet = f'/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/predictions/RegressUNet'
for region in regions:
    for experiment_type in experiment_types:
        for orog_type in orog_types:
            orog_flag = 'orog' if orog_type == "orog" else 'None'
            if experiment_type =="ESD":
                config_file_pr = f'/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep/{region}/models/GAN_Final_ESD_0.01_pr_{region}_orog{orog_flag}/config_info.json'
                config_file_tasmax = f'/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep/{region}/models/GAN_Final_ESD_0.01_tasmax_{region}_orog{orog_flag}/config_info.json'


            else:
                config_file_tasmax = f'/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep/{region}/models/GAN_Final_hist_future_0.01_tasmax_{region}_orog{orog_flag}/config_info.json'
                config_file_pr = f'/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep/{region}/models/GAN_Final_hist_future_0.01_pr_{region}_orog{orog_flag}/config_info.json'

            with open(config_file_pr, 'r') as f:
                config_pr = json.load(f)
            with open(config_file_tasmax, 'r') as f:
                config_tasmax = json.load(f)
            print(config_tasmax["model_name"])

            # Transpose orography to standard format
            domain = config_tasmax["region"]
            test_file_path = f"{base_path}/{region}/test"

            # Get test files
            files = glob.glob(f"{test_file_path}/*/predictors/*/*.nc", recursive=True)
            print(files)
            # Determine experiment type
            experiment = "Emul_hist_future" if "future" in config_tasmax["experiment"] else "ESD_pseudo_reality"
            # How to do orog vs no orog submissions
            experiment = f'{experiment}_orog' if orog_flag == 'orog' else f'{experiment}_no_orog'
            domain_name = f"{region}_Domain"

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
                stacked_X, y, orog, config_tasmax, means_output, stds_output = preprocess_inference_data(
                    config_tasmax, file, domain_name, experiment
                )
                try:
                    orog = orog.transpose("y", "x")
                except:
                    orog = orog.transpose("lat", "lon")

                gan_pr, unet_model_pr, ad_loss = load_model_cascade(
                    config_pr["model_name"],
                    epoch=200,
                    model_dir=config_pr["output_folder"]
                )
                gan_tasmax, unet_model_tasmax, ad_loss = load_model_cascade(
                    config_tasmax["model_name"],
                    epoch=200,
                    model_dir=config_tasmax["output_folder"]
                )

                n_times = 365
                gan_preds_pr, unet_preds_pr = predict_parallel_resid(gan_pr, unet_model_pr, stacked_X.isel(time = slice(0, n_times)),
                                               y.isel(time = slice(0, n_times)), 128, orog,
                                               stacked_X.isel(time = slice(0, n_times)).time.dt.dayofyear.values, means_output,
                                           stds_output, config = config_pr)
                gan_preds_tasmax, unet_preds_tasmax = predict_parallel_resid(gan_tasmax, unet_model_tasmax, stacked_X.isel(time = slice(0, n_times)),
                                               y.isel(time = slice(0, n_times)), 128, orog,
                                               stacked_X.isel(time = slice(0, n_times)).time.dt.dayofyear.values, means_output,
                                           stds_output, config = config_tasmax)
                merged_preds_gan = xr.merge([gan_preds_pr[['pr']],gan_preds_tasmax[['tasmax']]])
                merged_preds_unet = xr.merge([unet_preds_pr[['pr']], unet_preds_tasmax[['tasmax']]])
                merged_preds_gan.to_netcdf(output_path_gan)
                merged_preds_unet.to_netcdf(output_path_unet)











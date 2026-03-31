import xarray as xr
import sys
import matplotlib.pyplot as plt
import os
import datetime
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from tensorflow.keras.callbacks import Callback
import numpy as np
import tensorflow as tf
from functools import partial
repo_dirs = r'/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/models/FlowMatching'
os.chdir(repo_dirs)
AUTOTUNE = tf.data.experimental.AUTOTUNE
from dask.diagnostics import ProgressBar
import pandas as pd
import tensorflow.keras.layers as layers
import json
from tensorflow.keras.optimizers import Adam
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras import layers
import sys

"""TODO REMOVE TV LOSS and use updated code provided"""

config_file = r'//esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep/SA/models/FlowMatchingFinal_v2_Final_hist_future_0.01_tasmax_SA_orogorog/config_info.json'

with open(config_file, 'r') as f:
    config = json.load(f)
temp_conditioning = False




sys.path.append(repo_dirs)
from src.layers import *
#from src.models import *
from src.dm_updated_v3 import *
from src.process_input_training_data import *

# ── Import flow matching classes ────────────────────────────────────────────
# These are defined in the flow matching module; adjust import path as needed


stacked_X, y, orog, config = preprocess_input_data(config)
output_means = xr.open_dataset(config["means_output"])
std_means = xr.open_dataset(config["stds_output"])

stacked_X_test = stacked_X.isel(time = slice(0,15)).transpose("time","lat","lon","channel")
y_test = y.isel(time = slice(0,15))
MODEL_EPOCH =300
flow_model_pr, sigma_z = load_model_cascade(config,
                                   epoch=MODEL_EPOCH
                                   )
SOLVERS    = {"euler": _euler_solve, "heun": _heun_solve, "dpm2m": _dpm2m_solve}
SOLVER_NFE = {"euler": 1, "heun": 2, "dpm2m": 1}


flow_preds_pr = predict_parallel_flow(
    model=flow_model_pr,
    inputs=stacked_X_test.values,
    sigma_z =sigma_z,
    output_xr=y_test.copy(),
    batch_size=BATCH_SIZE,
    orog_vector=orog.values,
    time_of_year=stacked_X_test.time.dt.dayofyear,
    means=output_means,
    stds=std_means,
    config=config
)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1,)
ax.hist(flow_preds_pr.pr.values.ravel(), histtype ='step',
        bins = np.arange(0,1000,10), color ='r',
        label ='Prediction')
ax.hist(y_test.pr.values.ravel(), histtype ='step',
        bins = np.arange(0,1000,10), color ='b',
        label ='Prediction')
ax.set_yscale('log')
ax.legend()
fig.show()


import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1,)
ax.hist(flow_preds_pr.tasmax.values.ravel(), histtype ='step',
        bins = np.arange(270,320,1), color ='r',
        label ='Prediction')
ax.hist(y_test.tasmax.values.ravel(), histtype ='step',
        bins = np.arange(270,320,1), color ='b',
        label ='Prediction')
#ax.set_yscale('log')
ax.legend()
fig.show()



fig, ax = plt.subplots(1,2)
flow_preds_pr.pr.max("time").plot(ax = ax[0], vmax =310)
y_test.pr.max("time").plot(ax = ax[1])
fig.show()

fig, ax = plt.subplots(1,2)
flow_preds_pr.tasmax.isel(time =3).plot(ax = ax[0], vmax =320, vmin =280)
y_test.tasmax.isel(time =3).plot(ax = ax[1], vmax =320, vmin =280)
fig.show()


# print("  -> Temperature...")
# flow_preds_tasmax = predict_parallel_flow(
#     model_obj=flow_model_tasmax,
#     inputs=stacked_X.values,
#     output_xr=y.copy(),
#     batch_size=BATCH_SIZE,
#     orog_vector=orog.values,
#     time_of_year=time_of_year_values,
#     means=means_output,
#     stds=stds_output,
#     config=config_tasmax,
#     solver=solver
# )


def process_file(file_path, config_pr, config_tasmax, region, orog_flag,
                 output_path_diffusion_base, test_file_path,
                 n_members=1, solver="euler", MODEL_EPOCH = MODEL_EPOCH, BATCH_SIZE = BATCH_SIZE):
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
        config_tasmax, file_path, domain_name, experiment
    )
    stacked_X = stacked_X.isel(time=slice(0, 5))
    y         = y.isel(time=slice(0, 5))

    try:
        orog = orog.transpose("y", "x")
    except Exception:
        orog = orog.transpose("lat", "lon")

    # Load models
    print("Loading precipitation model...")
    flow_model_pr = load_model_cascade(config_pr,
        epoch=MODEL_EPOCH
    )
    print("Loading temperature model...")
    flow_model_tasmax = load_model_cascade(config_tasmax,
                                           epoch=MODEL_EPOCH
    )

    # Time conditioning
    time_of_year_values = stacked_X.time.dt.dayofyear.values
    config_pr["num_ode_steps"] =50
    config_tasmax["num_ode_steps"] = 50
    # Ensemble loop
    gan_preds = []
    for iii in range(n_members):
        print(f"Generating ensemble member {iii + 1}/{n_members}...")

        print("  -> Precipitation...")
        flow_preds_pr = predict_parallel_flow(
            model_obj    = flow_model_pr,
            inputs       = stacked_X.values,
            output_xr    = y.copy(),
            batch_size   = BATCH_SIZE,
            orog_vector  = orog.values,
            time_of_year = time_of_year_values,
            means        = means_output,
            stds         = stds_output,
            config       = config_pr,
            solver       = solver
        )

        print("  -> Temperature...")
        flow_preds_tasmax = predict_parallel_flow(
            model_obj    = flow_model_tasmax,
            inputs       = stacked_X.values,
            output_xr    = y.copy(),
            batch_size   = BATCH_SIZE,
            orog_vector  = orog.values,
            time_of_year = time_of_year_values,
            means        = means_output,
            stds         = stds_output,
            config       = config_tasmax,
            solver       = solver
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
"""Reminder about stacked_X, and deo steps in function
I need to check sigma_z
"""
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
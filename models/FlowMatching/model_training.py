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


config_file = sys.argv[-1]

with open(config_file, 'r') as f:
    config = json.load(f)
temp_conditioning = False

decay = 'Exponential'
config["decay_type"] = decay
input_shape = config["input_shape"]
output_shape = config["output_shape"]
n_filters = config["n_filters"]
kernel_size = config["kernel_size"]
n_channels = config["n_input_channels"]
n_output_channels = config["n_output_channels"]
orog_predictor = config["orog_fields"]
output_varname = config['output_varname']

# ── Flow matching specific config ──────────────────────────────────────────
config["batch_size"] = 16          # same as GAN
config["epochs"] = 500         # same as GAN
config["ema_decay"] = 0.999
config["beta_ema_sigma"] = 0.01
config["lambda_reg"] = 10.25
config["lambda_reg"] = 0.1
config["num_ode_steps"] = 50
config["t_embed_scale"] = 1000.0
if output_varname == "pr":
    config["student_t_df"] = 40.0
else:
    config["student_t_df"] = 40.0
config["intensity_weight"] = 0.25       # same as GAN itensity_weight

config["model_name"] = config["model_name"].replace("GAN", "FlowMatchingFinal_v3")
if orog_predictor == "orog":
    orog_bool = True
else:
    orog_bool = False
print("orog_bool is ", orog_bool)
BATCH_SIZE = config["batch_size"]
config["model_name"] = (
    config["model_name"] + "_" + str(config["ad_loss_factor"]) + "_"
    + str(config["output_varname"]) + "_" + str(config["region"]) + "_" + "orog" + orog_predictor
)
if not os.path.exists(f'{config["output_folder"]}/{config["model_name"]}'):
    os.makedirs(f'{config["output_folder"]}/{config["model_name"]}')

sys.path.append(repo_dirs)
from src.layers import *
from src.models import *
from src.dm_updated_v3 import *
from src.process_input_training_data import *

# ── Import flow matching classes ────────────────────────────────────────────
# These are defined in the flow matching module; adjust import path as needed


stacked_X, y, orog, config = preprocess_input_data(config)

try:
    orog = orog.transpose("y", "x")
except:
    orog = orog.transpose("lat", "lon")
output_means = xr.open_dataset(config["means_output"])
output_stds  = xr.open_dataset(config["stds_output"])

# ── Normalisation: identical to GAN script ──────────────────────────────────
if output_varname == "pr":
    config["delta"]            = 1
    conversion_factor          = 1
    config['conversion_factor'] = conversion_factor
    y[output_varname] = np.log(y[output_varname] * conversion_factor + config["delta"])
elif output_varname == "tasmax":
    z_score = (y[output_varname] - output_means[output_varname]) / output_stds[output_varname]
    y[output_varname] = z_score

common_times = stacked_X.time.to_index().intersection(y.time.to_index())
stacked_X = stacked_X.sel(time=common_times)
y = y.sel(time=common_times)

try:
    try:
        y = y[[output_varname]].transpose("time", "lat", "lon")
        stacked_X = stacked_X.transpose("time", "lat", "lon", "channel")
    except:
        try:
            y = y[[output_varname]].drop("bnds").transpose("time", "lat", "lon")
            stacked_X = stacked_X.drop("bnds").transpose("time", "lat", "lon", "channel")
        except:
            y = y[[output_varname]].drop("bnds").transpose("time", "lat", "lon")
            stacked_X = stacked_X.transpose("time", "lat", "lon", "channel")
except:
    y  = y[[output_varname]].transpose("time", "y", "x")
    stacked_X = stacked_X.transpose("time", "lat", "lon", "channel")

with ProgressBar():
    y         = y.load()
    stacked_X = stacked_X.load()
if output_varname == "pr":
    final_activation = tf.keras.layers.LeakyReLU(0.1)
else:
    final_activation = "linear"
# ── Build flow net (replaces generator + unet_model + d_model) ─────────────
flow_net = build_flow_unet(
    input_size=input_shape,
    resize_output=output_shape,
    num_filters=n_filters[:],
    num_channels=n_channels,
    orog_predictor=orog_bool,varname = output_varname, final_activation = final_activation
)
ema_flow_net = build_flow_unet(
    input_size=input_shape,
    resize_output=output_shape,
    num_filters=n_filters[:],
    num_channels=n_channels,
    orog_predictor=orog_bool, varname = output_varname, final_activation = final_activation
)

ema_flow_net.set_weights(flow_net.get_weights())
flow_net.summary()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# ── Training / validation split: identical to GAN script ───────────────────
start_time_init = stacked_X.time.min()
end_time        = stacked_X.time.max()
start_time      = start_time_init + pd.Timedelta(days=((365 * 3) // BATCH_SIZE) * BATCH_SIZE - 1)

total_size = stacked_X.sel(time=slice(start_time, end_time)).time.size
BATCH_SIZE = int(BATCH_SIZE)
eval_times = (BATCH_SIZE * ((total_size) // BATCH_SIZE))

# ── Checkpoints (replaces generator/discriminator/unet checkpoints) ─────────
flow_checkpoint = ModelCheckpointCallback(
    model_ref=flow_net,
    filepath=f'{config["output_folder"]}/{config["model_name"]}/flow_net',
    period=5
)
ema_flow_checkpoint = ModelCheckpointCallback(
    model_ref=ema_flow_net,
    filepath=f'{config["output_folder"]}/{config["model_name"]}/ema_flow_net',
    period=5
)

# ── Learning rate schedule: identical to GAN script ────────────────────────
config["learning_rate_unet"] = 0.00003
config["decay_rate"]         = 0.996

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    config["learning_rate_unet"],
    decay_steps=config["decay_steps"],
    decay_rate=config["decay_rate_gan"]
)

# Single optimiser for the flow net (replaces generator + unet optimisers)
fm_optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule,
    beta_1=config["beta_1"],
    beta_2=config["beta_2"]
)

# ── Dataset: identical to GAN script ───────────────────────────────────────
data = create_dataset(
    y[output_varname].sel(time=slice(start_time, end_time)),
    stacked_X.sel(time=slice(start_time, end_time)),
    eval_times,
    temp_conditioning=temp_conditioning
)
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
data = data.with_options(options)
data = data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
data = data.shuffle(16)

end_time_val   = start_time_init + pd.Timedelta(days=((365 * 3) // BATCH_SIZE) * BATCH_SIZE - 1)
start_time_val = start_time_init

val_stacked_X  = stacked_X.sel(time=slice(start_time_val, end_time_val))
val_y          = y[output_varname].sel(time=slice(start_time_val, end_time_val))
val_total_size = int(val_stacked_X.time.size)
val_eval_times = (BATCH_SIZE * ((val_total_size) // BATCH_SIZE))
val_BATCH_SIZE = int(BATCH_SIZE)
print("Val batch size:", val_BATCH_SIZE)

val_data = create_dataset(val_y, val_stacked_X, val_eval_times, temp_conditioning=temp_conditioning)
options  = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
val_data = val_data.with_options(options)
val_data = val_data.batch(val_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ── Flow matching model (replaces WGAN_Cascaded_IP) ─────────────────────────
schedule = FlowSchedule(
    num_ode_steps=config["num_ode_steps"],
    t_embed_scale=config["t_embed_scale"],
)

fm_model = SingleStageFlowMatching(
    flow_net=flow_net,
    ema_flow_net=ema_flow_net,
    schedule=schedule,
    orog=tf.convert_to_tensor(orog.values, 'float32'),
    varname=output_varname,
    ema_decay=config["ema_decay"],
    lambda_reg=config["lambda_reg"],
    beta_ema_sigma=config["beta_ema_sigma"],
    use_gan_loss_constraints=False,
    orog_bool=orog_bool,
    intensity_weight=config["intensity_weight"],
    student_t_df=config["student_t_df"],
)

# ── Prediction callback (replaces PredictionCallback) ───────────────────────
call_backargs = [
    stacked_X.sel(time=slice(start_time_val, end_time)).isel(time=slice(0, 30)),
    stacked_X.sel(time=slice(start_time_val, end_time)).isel(time=slice(0, 30)).time.dt.dayofyear
]

prediction_callback = PredictionCallbackFlow(
    flow_model_obj=fm_model,
    x_input=call_backargs,
    y_input=y[output_varname].sel(time=slice(start_time_val, end_time)).isel(time=slice(0, 30)),
    save_dir=f'{config["output_folder"]}/{config["model_name"]}',
    batch_size=30,
    orog=orog.values,
    output_mean=output_means[output_varname].values,
    output_std=output_stds[output_varname].values,
    varname=output_varname,
    orog_bool=orog_bool,
    plot_every=10,
    n_panels=15,
    student_t_df=config["student_t_df"],
)

fm_model.compile(
    fm_optimizer=fm_optimizer,
    loss_fn=tf.keras.losses.mean_squared_error
)

with open(f'{config["output_folder"]}/{config["model_name"]}/config_info.json', 'w') as f:
    json.dump(config, f)

history = fm_model.fit(
    data, batch_size=BATCH_SIZE, epochs=config["epochs"], verbose=2, shuffle=True,
    callbacks=[flow_checkpoint, ema_flow_checkpoint, prediction_callback],
    validation_data=val_data
)

history = pd.DataFrame(history.history)
history.to_csv(f'{config["output_folder"]}/{config["model_name"]}/training_history.csv')
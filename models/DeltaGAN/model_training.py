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
repo_dirs =r'//esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/models/DeltaGAN'
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

config_file = sys.argv[-1]#r'/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep/ALPS/configs/Emul_hist_future/ALPS_hist_future_tasmax_orog.json'#sys.argv[-1]
with open(config_file, 'r') as f:
    config = json.load(f)
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
config["itensity_weight"] = 4.25
config["batch_size"] = 16
config["epochs"] = 250
config["av_int_weight"] = 1
config["model_name"] = config["model_name"] + "_Learning_decay"
if orog_predictor == "orog":
    orog_bool = True
else:
    orog_bool = False
print("orog_bool is ", orog_bool)
BATCH_SIZE = config["batch_size"]  # config["batch_size"]
init_weights = True
config["model_name"] = config["model_name"] + "_" + str(config["ad_loss_factor"]) + "_"+ str(config["output_varname"]) + "_"+  str(config["region"]) + "_" + "orog" + orog_predictor
if not os.path.exists(f'{config["output_folder"]}/{config["model_name"]}'):
    os.makedirs(f'{config["output_folder"]}/{config["model_name"]}')
# custom modules
sys.path.append(repo_dirs)
from src.layers import *
from src.models import *
from src.gan import *
from src.process_input_training_data import *

stacked_X, y, orog, config = preprocess_input_data(config)

# Config file has been updated
try:
    orog = orog.transpose("y", "x")
except:
    orog = orog.transpose("lat","lon")
output_means = xr.open_dataset(config["means_output"])
output_stds = xr.open_dataset(config["stds_output"])

if output_varname == "pr":
    config["delta"] = 1
    conversion_factor = 1
    config['conversion_factor'] = conversion_factor
    y[output_varname] = np.log(y[output_varname]* conversion_factor + config["delta"])
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


n_filters = n_filters  # + [512]
generator = res_gan(input_shape, output_shape, n_filters[:], n_channels, n_output_channels,
                                          final_activation='linear', orog_predictor = orog_bool)
if output_varname == "pr":
    unet_model = unet(input_shape, output_shape, n_filters[:], n_channels, n_output_channels,
                                          final_activation=tf.keras.layers.LeakyReLU(0.01),
                           orog_predictor = orog_bool)
else:
    unet_model = unet(input_shape, output_shape, n_filters[:], n_channels, n_output_channels,
                                          final_activation='linear', orog_predictor = orog_bool)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
unet_model.summary()
generator.summary()

noise_dim = [tuple(generator.inputs[i].shape[1:]) for i in range(len(generator.inputs) - 1)]
d_model = critic(tuple(output_shape) + (n_output_channels,),
                                     tuple(input_shape) + (n_channels,), conditioning = False)
d_model.summary()
learning_rate_adapted = True

start_time= stacked_X.time.min()  # Last time step
end_time_init = stacked_X.time.max()  # Last time step
end_time = end_time_init - pd.Timedelta(days = ((365*3)//BATCH_SIZE) *BATCH_SIZE-1 )# Start of last 2 years

total_size = stacked_X.sel(time=slice(start_time, end_time)).time.size
BATCH_SIZE = int(BATCH_SIZE)

eval_times = (BATCH_SIZE * ((total_size) // BATCH_SIZE)) # an updated eval_times
generator_checkpoint = GeneratorCheckpoint(
    generator=generator,
    filepath=f'{config["output_folder"]}/{config["model_name"]}/generator',
    period=5  # Save every 5 epochs
)

discriminator_checkpoint = DiscriminatorCheckpoint(
    discriminator=d_model,
    filepath=f'{config["output_folder"]}/{config["model_name"]}/discriminator',
    period=5  # Save every 5 epochs
)

unet_checkpoint = DiscriminatorCheckpoint(
    discriminator=unet_model,
    filepath=f'{config["output_folder"]}/{config["model_name"]}/unet',
    period=5  # Save every 5 epochs
)
n_cycles = 10
steps_per_epoch = (eval_times // BATCH_SIZE)
warmup_epochs = 1
warmup_steps = warmup_epochs * steps_per_epoch
total_steps = steps_per_epoch * config["epochs"]

if decay =='Cosine':
    # Example usage in your optimizer config (without warmup, as in your original code)
    lr_schedule = CosineAnnealingLR(
        initial_learning_rate=1e-5,
        decay_steps= total_steps//n_cycles, warmup_target=7e-5,  # Peak learning rate
        warmup_steps=warmup_steps,
        alpha=0.1  # Don't decay all the way to 0
    )

    lr_schedule_gan = CosineAnnealingLR(
        initial_learning_rate=1e-5,
        decay_steps=total_steps//n_cycles, warmup_target=7e-5,  # Peak learning rate
        warmup_steps=warmup_steps,
        alpha=0.1  # Don't decay all the way to 0
    )
else:
    config["learning_rate_unet"] = 0.00015
    config["learning_rate_unet"] = 0.00015
    config["decay_rate"] = 0.995
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        config["learning_rate_unet"], decay_steps=config["decay_steps"], decay_rate=config["decay_rate_gan"] **2)

    lr_schedule_gan = tf.keras.optimizers.schedules.ExponentialDecay(
        config["learning_rate"], decay_steps=config["decay_steps"], decay_rate=config["decay_rate_gan"])

generator_optimizer = keras.optimizers.Adam(
    learning_rate=lr_schedule_gan, beta_1=config["beta_1"], beta_2=config["beta_2"])

discriminator_optimizer = keras.optimizers.Adam(
    learning_rate=config["learning_rate"], beta_1=config["beta_1"], beta_2=config["beta_2"])
unet_optimizer = keras.optimizers.Adam(
    learning_rate=lr_schedule, beta_1=config["beta_1"], beta_2=config["beta_2"])

data = create_dataset(y[output_varname].sel(time=slice(start_time, end_time)),
                      stacked_X.sel(time=slice(start_time, end_time)), eval_times)
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
data = data.with_options(options)

data = data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
data = data.shuffle(16)

# For calendar-aware date offsets
# Compute the 2-year validation window (assumes 'time' is a datetime/cftime coordinate)
end_time = stacked_X.time.max()  # Last time step
start_time_val = end_time - pd.Timedelta(days = ((365*3)//BATCH_SIZE) *BATCH_SIZE-1 )# Start of last 2 years

# Slice the datasets to the validation period
val_stacked_X = stacked_X.sel(time=slice(start_time_val, end_time))
val_y = y[output_varname].sel(time=slice(start_time_val, end_time))


# Compute validation size (full window)
val_total_size = int(val_stacked_X.time.size)
val_eval_times = (BATCH_SIZE * ((val_total_size) // BATCH_SIZE))  # Use all steps for validation

# Adjust BATCH_SIZE consistently with training (if needed for GCMs)
val_BATCH_SIZE = int(BATCH_SIZE)  # Reuse global BATCH_SIZE
print("Val batch size:", val_BATCH_SIZE)

# Create the validation dataset (first val_eval_times steps of the sliced data)
val_data = create_dataset(val_y, val_stacked_X, val_eval_times)

# Apply distributed sharding options
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
val_data = val_data.with_options(options)

# Batch and prefetch (no shuffle for validation to maintain temporal order)
val_data = val_data.batch(val_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# Note: No .shuffle() here—validation should be deterministic



try:
    av_int_weight = config["av_int_weight"]

except:
    av_int_weight = 0.0
    config["av_int_weight"] = av_int_weight

wgan = WGAN_Cascaded_IP(discriminator=d_model,
                        generator=generator,
                        latent_dim=noise_dim,
                        discriminator_extra_steps=config["discrim_steps"],
                        ad_loss_factor=config["ad_loss_factor"],
                        orog=tf.convert_to_tensor(orog.values, 'float32'),
                        gp_weight=config["gp_weight"],
                        unet=unet_model,
                        train_unet=True,
                        intensity_weight=config["itensity_weight"],
                        average_intensity_weight=av_int_weight,
                        varname=output_varname, orog_bool = orog_bool)
prediction_callback = PredictionCallback(unet_model, generator, wgan, [stacked_X.sel(time=slice(start_time_val, end_time)).isel(time=slice(0, 30)), stacked_X.sel(time=slice(start_time_val, end_time)).isel(time=slice(0, 30)).time.dt.dayofyear],
                                         y[output_varname].sel(time=slice(start_time_val, end_time)).isel(time=slice(0, 30)),
                                         orog=orog.values,
                                         save_dir=f'{config["output_folder"]}/{config["model_name"]}',
                                         output_mean=output_means[output_varname].values,
                                         output_std=output_stds[output_varname].values,
                                         varname=output_varname, orog_bool = orog_bool)
# Compile the WGAN model.
wgan.compile(d_optimizer=discriminator_optimizer,
             g_optimizer=generator_optimizer,
             g_loss_fn=generator_loss,
             d_loss_fn=discriminator_loss,
             u_optimizer=unet_optimizer,
             u_loss_fn=tf.keras.losses.mean_squared_error)

with open(f'{config["output_folder"]}/{config["model_name"]}/config_info.json', 'w') as f:
    json.dump(config, f)

history = wgan.fit(data, batch_size=BATCH_SIZE, epochs=config["epochs"], verbose=1, shuffle=True,
         callbacks=[generator_checkpoint, discriminator_checkpoint, unet_checkpoint, prediction_callback, tensorboard_callback],
         validation_data = val_data)

import pandas as pd

history = pd.DataFrame(history.history)
history.to_csv(f'{config["output_folder"]}/{config["model_name"]}/training_history.csv')

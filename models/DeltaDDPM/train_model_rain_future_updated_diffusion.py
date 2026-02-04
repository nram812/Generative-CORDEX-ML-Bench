import xarray as xr
import sys
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from tensorflow.keras.callbacks import Callback
import numpy as np
import tensorflow as tf
from functools import partial
from tqdm import tqdm

AUTOTUNE = tf.data.experimental.AUTOTUNE
from dask.diagnostics import ProgressBar
import pandas as pd
import tensorflow.keras.layers as layers
import json
from tensorflow.keras.optimizers import Adam
from tensorflow._api.v2.distribute import MirroredStrategy
from tensorflow.keras import layers
import datetime


config_file = sys.argv[-1]#r'//esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/DATA_prep/SA/configs/Emul_hist_future/SA_hist_future_pr_orog.json'#sys.argv[-1]
with open(config_file, 'r') as f:
    config = json.load(f)
config["model_name"] = config["model_name"].replace('GAN', 'DM_model')
config["dm_timesteps"] = 1000
config["dm_beta_start"] = 1e-4
config["dm_beta_end"] =0.02
config["dm_ema_decay"] =0.985
input_shape = config["input_shape"]
output_shape = config["output_shape"]
n_filters = config["n_filters"]
kernel_size = config["kernel_size"]
n_channels = config["n_input_channels"]
n_output_channels = config["n_output_channels"]
orog_predictor = config["orog_fields"]
output_varname = config['output_varname']
config["itensity_weight"] = 4.25
config["batch_size"] = 8
config["epochs"] = 500
config["av_int_weight"] = 1
if orog_predictor == "orog":
    orog_bool = True
else:
    orog_bool = False
print("orog_bool is ", orog_bool)
BATCH_SIZE = config["batch_size"]  # config["batch_size"]
init_weights = True
dm_info = f'{config["dm_timesteps"]}-{config["dm_beta_start"]}-{config["dm_beta_end"]}'
config["model_name"] = config["model_name"] + "_" + dm_info  + "_"+ str(config["output_varname"]) + "_"+  str(config["region"]) + "_" + "orog" + orog_predictor
if not os.path.exists(f'{config["output_folder"]}/{config["model_name"]}'):
    os.makedirs(f'{config["output_folder"]}/{config["model_name"]}')

# creating a path to store the model outputs
if not os.path.exists(f'{config["output_folder"]}/{config["model_name"]}'):
    os.makedirs(f'{config["output_folder"]}/{config["model_name"]}')
# custom modules
sys.path.append(r'//esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/models/DeltaDDPM')
from src.layers import *
from src.gan import GeneratorCheckpoint, DiscriminatorCheckpoint
from src.process_input_training_data import *
from src.models_dm import *
from src.dm import ResidualDiffusion, DiffusionSchedule, EDMSchedule, PredictionCallbackDiffusion
from src.models_dm import build_diffusion_unet

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


n_filters = n_filters#+ [512]

print("OUTPUT SHAPE===================================")
print(f"{input_shape=}  |  {output_shape=}")


generator = build_diffusion_unet(input_shape, output_shape, n_filters[:], n_channels, orog_predictor =orog_bool)
ema_generator = build_diffusion_unet(input_shape, output_shape, n_filters[:], n_channels, orog_predictor =orog_bool)

if output_varname == "pr":
    unet_model = unet(input_shape, output_shape, n_filters[:], n_channels, n_output_channels,
                      final_activation=tf.keras.layers.LeakyReLU(0.01),
                      orog_predictor=orog_bool)
else:
    unet_model = unet(input_shape, output_shape, n_filters[:], n_channels, n_output_channels,
                      final_activation='linear',
                      orog_predictor=orog_bool)

ema_generator.set_weights(generator.get_weights())

unet_model.summary()
generator.summary()

noise_dim = [tuple(generator.inputs[i].shape[1:]) for i in range(len(generator.inputs) - 1)]

learning_rate_adapted = True
generator_checkpoint = GeneratorCheckpoint(
    generator=generator,
    filepath=f'{config["output_folder"]}/{config["model_name"]}/generator',
    period=5  # Save every 5 epochs
)
ema_generator_checkpoint = GeneratorCheckpoint(
    generator=ema_generator,
    filepath=f'{config["output_folder"]}/{config["model_name"]}/ema_generator',
    period=5  # Save every 5 epochs
)

unet_checkpoint = DiscriminatorCheckpoint(
    discriminator=unet_model,
    filepath=f'{config["output_folder"]}/{config["model_name"]}/unet',
    period=5  # Save every 5 epochs
)



learning_rate_adapted = True

start_time= stacked_X.time.min()  # Last time step
end_time_init = stacked_X.time.max()  # Last time step
end_time = end_time_init - pd.Timedelta(days = ((365*3)//BATCH_SIZE) *BATCH_SIZE-1 )# Start of last 2 years

total_size = stacked_X.sel(time=slice(start_time, end_time)).time.size
BATCH_SIZE = int(BATCH_SIZE)

eval_times = (BATCH_SIZE * ((total_size) // BATCH_SIZE)) # an updated eval_times

# Example usage in your optimizer config (without warmup, as in your original code)


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

"""7e-6 maybe too slow of a learning rate, but have too few cycles appaears to screw things up"""
n_cycles = 10
steps_per_epoch = (eval_times // BATCH_SIZE)
warmup_epochs = 1
warmup_steps = warmup_epochs * steps_per_epoch
total_steps = steps_per_epoch * config["epochs"]
decay ="normal"
# Example usage in your optimizer config (without warmup, as in your original code)
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
    config["learning_rate_unet"] = 0.00008
    config["learning_rate_unet"] = 0.00008
    config["decay_rate"] = 0.995
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        config["learning_rate_unet"], decay_steps=config["decay_steps"], decay_rate=config["decay_rate_gan"])

    lr_schedule_gan = tf.keras.optimizers.schedules.ExponentialDecay(
        config["learning_rate"], decay_steps=config["decay_steps"], decay_rate=config["decay_rate_gan"])



generator_optimizer = keras.optimizers.Adam(
    learning_rate=lr_schedule_gan, beta_1=config["beta_1"], beta_2=config["beta_2"])

unet_optimizer = keras.optimizers.Adam(
    learning_rate=lr_schedule, beta_1=config["beta_1"], beta_2=config["beta_2"])

scheduler = DiffusionSchedule(timesteps=config["dm_timesteps"],
                              beta_start=config["dm_beta_start"], beta_end=config["dm_beta_end"])
if config.get("diffusion_type") == "EDM":
    scheduler = EDMSchedule()


class ResidualDiffusion(tf.keras.Model):
    def __init__(self, diffusion=None,
                 ema_diffusion=None,
                 scheduler=None,
                 gp_weight=10.0,
                 orog=None, unet=None, train_unet=True, varname="pr",
                 ema_decay=None,
                 use_gan_loss_constraints=None, orog_bool=True):
        super(ResidualDiffusion, self).__init__()

        self.diffusion = diffusion
        self.ema_diffusion = ema_diffusion
        self.scheduler = scheduler
        self.gp_weight = gp_weight
        self.orog = orog
        self.unet = unet
        self.orog_bool = orog_bool
        self.train_unet = train_unet
        self.varname = varname
        self.ema_decay = ema_decay
        self.use_gan_loss_constraints = use_gan_loss_constraints

        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.u_loss_tracker = tf.keras.metrics.Mean(name="unet_loss")

        if self.use_gan_loss_constraints:
            self.gan_mae_tracker = tf.keras.metrics.Mean(name="gan_mae")
            self.max_iten_pred_tracker = tf.keras.metrics.Mean(name="max_iten_pred")
            self.max_iten_true_tracker = tf.keras.metrics.Mean(name="max_iten_true")

    def compile(self, dm_optimizer, u_optimizer, u_loss_fn):
        super(ResidualDiffusion, self).compile()
        self.dm_optimizer = dm_optimizer
        self.u_optimizer = u_optimizer
        self.u_loss_fn = u_loss_fn

    @staticmethod
    def expand_conditional_inputs(X, batch_size):
        expanded_image = tf.expand_dims(X, axis=0)  # Shape: (1, 172, 179)

        # Repeat the image to match the desired batch size
        expanded_image = tf.repeat(expanded_image, repeats=batch_size, axis=0)  # Shape: (batch_size, 172, 179)

        # Create a new axis (1) on the last axis
        expanded_image = tf.expand_dims(expanded_image, axis=-1)
        return expanded_image

    @staticmethod
    def process_real_images(real_images_obj):
        output_vars, averages = real_images_obj  # Unpack the input
        # Extract relevant variables from the output_vars dictionary
        real_images = output_vars['pr']

        average = averages["X"]
        time_of_year = averages["time_of_year"]
        return real_images, average, time_of_year

    def train_step(self, real_images):
        real_images, average, time_of_year = self.process_real_images(real_images)
        real_images = tf.expand_dims(real_images, axis=-1)
        batch_size = tf.shape(real_images)[0]
        orog_vector = self.expand_conditional_inputs(self.orog, batch_size)
        # Prepare conditional arguments
        if self.orog_bool:
            unet_args = [average, orog_vector, time_of_year]
        else:
            unet_args = [average, time_of_year]
        # make sure the auxiliary inputs are the same shape as the training batch
        # if the U-Net is trained, apply gradients otherwise only use inference mode from the U-Net
        # ============================================================================
        # PHASE 1: Train U-Net (Initial Prediction Network)
        # ============================================================================
        if self.train_unet:
            with tf.GradientTape() as tape:
                init_prediction = self.unet(unet_args, training=True)
                real_rain = real_images
                pred_rain = init_prediction
                mae_unet = self.u_loss_fn(real_rain, pred_rain)
            u_gradient = tape.gradient(mae_unet, self.unet.trainable_variables)
            self.u_optimizer.apply_gradients(zip(u_gradient, self.unet.trainable_variables))
        else:
            # Alternative path (not typically used based on code structure)
            with tf.GradientTape() as tape:
                init_prediction = self.unet(unet_args, training=True)
                mae_unet = self.u_loss_fn(real_images[:, :, :], init_prediction)

        # Diffusion model
        noise = tf.random.normal(shape=tf.shape(real_images))

        t = tf.random.uniform(shape=(batch_size, 1), minval=0,
                              maxval=self.scheduler.timesteps, dtype=tf.int32)
        sqrt_alpha_bar_t = tf.gather(self.scheduler.sqrt_alpha_bar, t)
        sqrt_alpha_bar_t = tf.reshape(sqrt_alpha_bar_t, (batch_size, 1, 1, 1))
        sqrt_one_minus_alpha_bar_t = tf.gather(self.scheduler.sqrt_one_minus_alpha_bar, t)
        sqrt_one_minus_alpha_bar_t = tf.reshape(sqrt_one_minus_alpha_bar_t, (batch_size, 1, 1, 1))

        with tf.GradientTape() as tape:
            init_prediction_unet = self.unet(unet_args, training=True)
            residual_gt = (real_images - init_prediction_unet)

            residual_noisy = sqrt_alpha_bar_t * residual_gt + sqrt_one_minus_alpha_bar_t * noise
            if self.orog_bool:
                noise_pred = self.diffusion(
                    [residual_noisy, t, average, orog_vector, init_prediction_unet, time_of_year], training=True)
            else:
                noise_pred = self.diffusion([residual_noisy, t, average, init_prediction_unet, time_of_year],
                                            training=True)
            dm_loss = self.u_loss_fn(noise, noise_pred)

            # reconstruct residual_gt (x0) using predicted noise (eps_theta(xt))
            # eq. 15 from Ho et al 2020
            # residual_pred = (residual_noisy - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t # TODO: remove constraint

            # or try see if this works better: need to grab the other params from scheduler
            # residual_pred = 1.0 / tf.sqrt(alpha_t) * (residual_noisy - (beta_t / tf.sqrt(1 - alpha_bar_t)) * noise_pred)

        dm_gradient = tape.gradient(dm_loss, self.diffusion.trainable_variables)
        self.dm_optimizer.apply_gradients(zip(dm_gradient, self.diffusion.trainable_variables))

        for weight, ema_weight in zip(self.diffusion.weights, self.ema_diffusion.weights):
            ema_weight.assign(self.ema_decay * ema_weight + (1 - self.ema_decay) * weight)

        self.g_loss_tracker.update_state(dm_loss)
        self.u_loss_tracker.update_state(mae_unet)

        if not self.use_gan_loss_constraints:
            return {
                "g_loss": self.g_loss_tracker.result(),
                "unet_loss": self.u_loss_tracker.result(),
            }

        self.gan_mae_tracker.update_state(mae)
        self.max_iten_pred_tracker.update_state(tf.math.exp(maximum_intensity_predicted))
        self.max_iten_true_tracker.update_state(tf.math.exp(maximum_intensity))

        return {
            "g_loss": self.g_loss_tracker.result(),
            "unet_loss": self.u_loss_tracker.result(),
            "gan_mae": self.gan_mae_tracker.result(),
            "max_iten_pred": self.max_iten_pred_tracker.result(),
            "max_iten_true": self.max_iten_true_tracker.result(),
        }

    def test_step(self, real_images):
        """
        Validation/test step for the ResidualDiffusion model.
        Evaluates U-Net and diffusion model performance without updating weights.
        """
        real_images, average, time_of_year = self.process_real_images(real_images)
        real_images = tf.expand_dims(real_images, axis=-1)
        batch_size = tf.shape(real_images)[0]
        orog_vector = self.expand_conditional_inputs(self.orog, batch_size)

        # Prepare conditional arguments
        if self.orog_bool:
            unet_args = [average, orog_vector, time_of_year]
        else:
            unet_args = [average, time_of_year]

        # ============================================================================
        # Evaluate U-Net (Initial Prediction Network)
        # ============================================================================
        init_prediction = self.unet(unet_args, training=False)
        mae_unet = self.u_loss_fn(real_images, init_prediction)

        # ============================================================================
        # Evaluate Diffusion Model
        # ============================================================================
        noise = tf.random.normal(shape=tf.shape(real_images))

        t = tf.random.uniform(
            shape=(batch_size, 1),
            minval=0,
            maxval=self.scheduler.timesteps,
            dtype=tf.int32
        )

        sqrt_alpha_bar_t = tf.gather(self.scheduler.sqrt_alpha_bar, t)
        sqrt_alpha_bar_t = tf.reshape(sqrt_alpha_bar_t, (batch_size, 1, 1, 1))
        sqrt_one_minus_alpha_bar_t = tf.gather(self.scheduler.sqrt_one_minus_alpha_bar, t)
        sqrt_one_minus_alpha_bar_t = tf.reshape(sqrt_one_minus_alpha_bar_t, (batch_size, 1, 1, 1))

        # Compute residual
        residual_gt = real_images - init_prediction
        residual_noisy = sqrt_alpha_bar_t * residual_gt + sqrt_one_minus_alpha_bar_t * noise

        # Predict noise using EMA diffusion model for better stability
        if self.orog_bool:
            noise_pred = self.ema_diffusion(
                [residual_noisy, t, average, orog_vector, init_prediction, time_of_year],
                training=False
            )
        else:
            noise_pred = self.ema_diffusion(
                [residual_noisy, t, average, init_prediction, time_of_year],
                training=False
            )

        dm_loss = self.u_loss_fn(noise, noise_pred)

        # Update metrics
        self.g_loss_tracker.update_state(dm_loss)
        self.u_loss_tracker.update_state(mae_unet)

        if not self.use_gan_loss_constraints:
            return {
                "g_loss": self.g_loss_tracker.result(),
                "unet_loss": self.u_loss_tracker.result(),
            }

        # Optional: Add GAN constraint metrics if needed
        # Note: You'll need to compute mae, maximum_intensity_predicted, maximum_intensity
        # similar to train_step if use_gan_loss_constraints is True

        self.gan_mae_tracker.update_state(mae)
        self.max_iten_pred_tracker.update_state(tf.math.exp(maximum_intensity_predicted))
        self.max_iten_true_tracker.update_state(tf.math.exp(maximum_intensity))

        return {
            "g_loss": self.g_loss_tracker.result(),
            "unet_loss": self.u_loss_tracker.result(),
            "gan_mae": self.gan_mae_tracker.result(),
            "max_iten_pred": self.max_iten_pred_tracker.result(),
            "max_iten_true": self.max_iten_true_tracker.result(),
        }

    @property
    def metrics(self):
        if not self.use_gan_loss_constraints:
            return [
                self.g_loss_tracker,
                self.u_loss_tracker,
            ]
        return [
            self.g_loss_tracker,
            self.u_loss_tracker,
            self.gan_mae_tracker,
            self.max_iten_pred_tracker,
            self.max_iten_true_tracker,
        ]


class PredictionCallbackDiffusion(tf.keras.callbacks.Callback):
    def __init__(self, unet, diffusion, ema_diffusion, scheduler, model_object, x_input, y_input, save_dir=None,
                 batch_size=30, orog=None,
                 output_mean=None, output_std=None, varname="pr", orog_bool=True):
        """
        Args:
            model: The trained model.
            sample_input: A sample input tensor to generate predictions.
            save_dir: Directory where prediction images will be saved.
        """
        super(PredictionCallbackDiffusion, self).__init__()
        self.unet = unet
        self.diffusion = diffusion
        self.ema_diffusion = ema_diffusion
        self.scheduler = scheduler
        self.model_object = model_object
        self.x_input = x_input
        self.orog_bool = orog_bool
        self.y_input = y_input
        self.x_input_tensor1 = tf.convert_to_tensor(x_input[0].values[:batch_size])
        self.x_input_tensor2 = tf.convert_to_tensor(x_input[1].values[:batch_size])
        self.y_input_tensor = tf.convert_to_tensor(y_input)
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.orog = orog
        self.orog_tensor = tf.convert_to_tensor(orog)
        self.output_mean = output_mean
        self.output_std = output_std
        self.varname = varname

        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """Runs at the end of each epoch to save a prediction image."""
        # Generate prediction
        if epoch % 3 == 0:
            tf.random.set_seed(16)
            orog_vector = self.model_object.expand_conditional_inputs(self.orog_tensor, self.batch_size)
            # average_combined, orog_vector,time_of_year_combined, spatial_means_combined,
            #                 spatial_stds_combined
            unet_args = [self.x_input_tensor1[:self.batch_size], orog_vector[:self.batch_size],
                         self.x_input_tensor2[:self.batch_size]] \
                if self.orog_bool else [self.x_input_tensor1[:self.batch_size],
                                        self.x_input_tensor2[:self.batch_size]]
            # TODO: list:
            #   FIXME: stop rescaling noise in diffusion model
            unet_prediction = self.unet(unet_args)

            tf.random.set_seed(16)
            residual_pred = tf.random.normal(shape=(self.batch_size, 128, 128, 1))  # FIXME: get correct shape

            # DDIM sampling with fewer steps
            num_inference_steps = 25# Much faster than 1000 steps
            timesteps = tf.cast(tf.linspace(self.scheduler.timesteps - 1, 0, num_inference_steps), tf.int32)

            for i in tqdm(range(num_inference_steps - 1), desc="Sampling timesteps"):
                t = timesteps[i]
                t_next = timesteps[i + 1]

                t_tensor = tf.fill((self.batch_size, 1), t)
                dm_args = [residual_pred[:self.batch_size], t_tensor[:self.batch_size],
                           self.x_input_tensor1[:self.batch_size],
                           orog_vector[:self.batch_size], unet_prediction[:self.batch_size],
                           self.x_input_tensor2[:self.batch_size]] \
                    if self.orog_bool else [residual_pred[:self.batch_size], t_tensor[:self.batch_size],
                                            self.x_input_tensor1[:self.batch_size],
                                            unet_prediction[:self.batch_size],
                                            self.x_input_tensor2[:self.batch_size]]

                # Predict noise using the model
                eps_t = self.ema_diffusion.predict(dm_args, verbose=0)

                # Extract scheduler values
                alpha_bar_t = self.scheduler.alpha_bar[t]
                alpha_bar_next = self.scheduler.alpha_bar[t_next]

                sqrt_alpha_bar_t = tf.sqrt(alpha_bar_t)
                sqrt_1m_alpha_bar_t = tf.sqrt(1.0 - alpha_bar_t)
                sqrt_alpha_bar_next = tf.sqrt(alpha_bar_next)
                sqrt_1m_alpha_bar_next = tf.sqrt(1.0 - alpha_bar_next)

                # Estimate x0 deterministically
                x0 = (residual_pred - sqrt_1m_alpha_bar_t * eps_t) / sqrt_alpha_bar_t
                x0 = tf.clip_by_value(x0, -5.0, 5.0)  # Optional clipping

                # DDIM deterministic update (no noise injection)
                residual_pred = sqrt_alpha_bar_next * x0 + sqrt_1m_alpha_bar_next * eps_t
                #
            if self.varname == "pr":

                unet_final = tf.math.exp(unet_prediction) - 1
                gan_final = tf.math.exp(unet_prediction + residual_pred) - 1
            else:
                unet_final = tf.squeeze(unet_prediction) * self.output_std + self.output_mean
                gan_final = tf.squeeze(unet_prediction + residual_pred) * self.output_std + self.output_mean

            y_copy = self.y_input.copy()
            y_2 = self.y_input.copy()
            y_copy.values = tf.squeeze(unet_final)
            y_2.values = tf.squeeze(gan_final)
            y_2 = y_2.where(y_2 > 0.5, 0.0)
            y_copy = y_copy.where(y_2 > 0.5, 0.0)
            boundaries2 = [0, 5, 12.5, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 125, 150, 200, 250]
            colors2 = [[0.000, 0.000, 0.000, 0.000], [0.875, 0.875, 0.875, 0.784], \
                       [0.761, 0.761, 0.761, 1.000], [0.639, 0.886, 0.871, 1.000], [0.388, 0.773, 0.616, 1.000], \
                       [0.000, 0.392, 0.392, 0.588], [0.000, 0.576, 0.576, 0.667], [0.000, 0.792, 0.792, 0.745], \
                       [0.000, 0.855, 0.855, 0.863], [0.212, 1.000, 1.000, 1.000], [0.953, 0.855, 0.992, 1.000], \
                       [0.918, 0.765, 0.992, 1.000], [0.918, 0.612, 1.000, 1.000], [0.878, 0.431, 1.000, 1.000], \
                       [0.886, 0.349, 1.000, 1.000], [0.651, 0.004, 0.788, 1.000], [0.357, 0.008, 0.431, 1.000], \
                       [0.180, 0.000, 0.224, 1.000]]
            # abbreviated for clarity

            # Create the colormap using ListedColormap
            cmap = 'viridis'#mcolors.ListedColormap(colors2)
            #norm = mcolors.BoundaryNorm(boundaries2, cmap.N)

            for i in range(8):
                print(i)
                fig, ax = plt.subplots(1, 3, figsize=(16, 6))
                print(y_2, y_2.max())
                if self.varname == "pr":
                    y_copy.isel(time=i).plot(ax=ax[0], cmap=cmap, vmax =100)
                    y_2.isel(time=i).plot(ax=ax[1], cmap=cmap, vmax =100)
                    (np.exp(self.y_input.isel(time=i)) - 1).plot(ax=ax[2],
                                                                          cmap=cmap, vmax =100)
                else:
                    true = self.y_input.isel(time=i) * self.output_std + self.output_mean
                    min_t = true.values.min()
                    max_t = true.values.max()
                    levels = np.arange(min_t, max_t, 0.5)
                    y_copy.isel(time=i).plot(ax=ax[0], cmap='RdBu_r', levels=levels)
                    y_2.isel(time=i).plot(ax=ax[1], cmap='RdBu_r', levels=levels)
                    true.plot(ax=ax[2], cmap='RdBu_r', levels=levels)
                # ax[0].coastlines('10m')
                # ax[1].coastlines('10m')
                # ax[2].coastlines('10m')
                ax[0].set_title('Unet')
                ax[1].set_title('DM')
                ax[2].set_title('GT')
                # Save the figure
                filename = os.path.join(self.save_dir, f"epoch_{epoch + 1}_{i}.png")
                plt.savefig(filename, bbox_inches="tight", dpi=200)
                plt.close()

            print(f"Saved prediction image to {filename}")
diffusion_model = ResidualDiffusion(diffusion=generator,
                         ema_diffusion=ema_generator,
                         scheduler=scheduler,
                                     orog=tf.convert_to_tensor(orog.values, 'float32'),
                                     unet=unet_model,
                                     train_unet=True, varname = output_varname,
                                     ema_decay=config["dm_ema_decay"],
                                     use_gan_loss_constraints=config.get("use_gan_loss_constraints", False),
                                    orog_bool = orog_bool)
prediction_callback = PredictionCallbackDiffusion(unet_model, generator, ema_generator, scheduler, diffusion_model, [stacked_X.sel(time=slice(start_time_val, end_time)).isel(time=slice(0, 30)),
                                                                                                                     stacked_X.sel(time=slice(start_time_val, end_time)).isel(time=slice(0, 30)).time.dt.dayofyear],
                                         y[output_varname].sel(time=slice(start_time_val, end_time)).isel(time=slice(0, 30)), orog = orog.values,
                                         save_dir = f'{config["output_folder"]}/{config["model_name"]}',
                                         output_mean =output_means[output_varname].values, output_std = output_stds[output_varname].values, varname = output_varname,
                                                  orog_bool = orog_bool)
# Compile the diffusion model.
diffusion_model.compile(dm_optimizer=generator_optimizer,
             u_optimizer=unet_optimizer,
             u_loss_fn=tf.keras.losses.mean_squared_error)

with open(f'{config["output_folder"]}/{config["model_name"]}/config_info.json', 'w') as f:
    json.dump(config, f)

history = diffusion_model.fit(data, batch_size=BATCH_SIZE, epochs=config["epochs"], verbose=1, shuffle=True,
         callbacks=[generator_checkpoint, ema_generator_checkpoint, unet_checkpoint, prediction_callback],
                    validation_data=val_data) #prediction_callback config["dm_ema_decay"]
import pandas as pd
history = pd.DataFrame(history.history)
history.to_csv(f'{config["output_folder"]}/{config["model_name"]}/training_loss.csv')

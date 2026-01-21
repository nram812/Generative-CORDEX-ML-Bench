import logging
import tensorflow as tf
import numpy as np
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import math

class PredictionCallbackDiffusion(tf.keras.callbacks.Callback):
    def __init__(self, unet, diffusion, ema_diffusion, scheduler, model_object, x_input, y_input, save_dir=None, batch_size =30, orog =None,
                                             output_mean =None, output_std = None, varname = "pr", orog_bool = True):
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
        self.x_input_tensor = tf.convert_to_tensor(x_input[0].values[:batch_size])
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
            #average_combined, orog_vector,time_of_year_combined, spatial_means_combined,
            #                 spatial_stds_combined
            unet_args = [self.x_input_tensor[0][:self.batch_size], orog_vector[:self.batch_size],  self.x_input_tensor[1][:self.batch_size]] \
                if self.orog_bool else [ self.x_input_tensor[0][:self.batch_size],  self.x_input_tensor[1][:self.batch_size]]
            # TODO: list:
            #   FIXME: stop rescaling noise in diffusion model
            unet_prediction = self.unet(unet_args)

            tf.random.set_seed(16)
            residual_pred = tf.random.normal(shape=(self.batch_size, 172, 179, 1)) #FIXME: get correct shape
            for t in reversed(range(self.scheduler.timesteps)):
                t_tensor = tf.fill((self.batch_size, 1), t)
                dm_args = [residual_pred[:self.batch_size], t_tensor[:self.batch_size], self.x_input_tensor[0][:self.batch_size],
                           orog_vector[:self.batch_size], unet_prediction[:self.batch_size], self.x_input_tensor[1][:self.batch_size]] \
                    if self.orog_bool else [residual_pred[:self.batch_size], t_tensor[:self.batch_size],
                                            self.x_input_tensor[0][:self.batch_size], unet_prediction[:self.batch_size][:self.batch_size],
                                            self.x_input_tensor[1]]

                eps_theta = self.ema_diffusion.predict(dm_args, verbose=0)

                beta_t = self.scheduler.beta[t]
                alpha_t = self.scheduler.alpha[t]
                alpha_bar_t = self.scheduler.alpha_bar[t]

                residual_pred = 1.0 / tf.sqrt(alpha_t) * (residual_pred - (beta_t / tf.sqrt(1 - alpha_bar_t)) * eps_theta)

                if t > 0:
                    eps = tf.random.normal(shape=(self.batch_size, 128, 128, 1)) # FIXME: get correct shape
                    residual_pred += tf.sqrt(beta_t) * eps




            if self.varname == "pr":

                unet_final = tf.math.exp(unet_prediction) -1
                gan_final = tf.math.exp(unet_prediction + residual_pred)-1
            else:
                unet_final = tf.squeeze(unet_prediction) * self.output_std + self.output_mean
                gan_final = tf.squeeze(unet_prediction + residual_pred) * self.output_std + self.output_mean

            y_copy = self.y_input.copy()
            y_2 = self.y_input.copy()
            y_copy.values = tf.squeeze(unet_final)
            y_2.values = tf.squeeze(gan_final)
            y_2 = y_2.where(y_2>0.5, 0.0)
            y_copy = y_copy.where(y_2>0.5, 0.0)
            boundaries2 = [0, 5,12.5, 15, 20, 25,30, 35, 40, 50, 60, 70, 80, 100, 125, 150, 200, 250]
            colors2 = [[0.000, 0.000, 0.000, 0.000], [0.875, 0.875, 0.875, 0.784],\
                      [0.761, 0.761, 0.761, 1.000], [0.639, 0.886, 0.871, 1.000], [0.388, 0.773, 0.616, 1.000],\
                      [0.000, 0.392, 0.392, 0.588], [0.000, 0.576, 0.576, 0.667], [0.000, 0.792, 0.792, 0.745],\
                      [0.000, 0.855, 0.855, 0.863], [0.212, 1.000, 1.000, 1.000], [0.953, 0.855, 0.992, 1.000],\
                      [0.918, 0.765, 0.992, 1.000], [0.918, 0.612, 1.000, 1.000], [0.878, 0.431, 1.000, 1.000],\
                      [0.886, 0.349, 1.000, 1.000], [0.651, 0.004, 0.788, 1.000], [0.357, 0.008, 0.431, 1.000],\
                      [0.180, 0.000, 0.224, 1.000]]
            #reviated for clarity

            # Create the colormap using ListedColormap
            cmap = mcolors.ListedColormap(colors2)
            norm = mcolors.BoundaryNorm(boundaries2, cmap.N)

            for i in range(8):
                print(i)
                fig, ax = plt.subplots(1, 3, figsize = (16, 6), subplot_kw = dict(projection = ccrs.PlateCarree(central_longitude =171.77)))
                if self.varname =="pr":
                    y_copy.isel(time =i).plot.contourf(ax = ax[0], transform = ccrs.PlateCarree(), cmap = cmap, norm = norm)
                    y_2.isel(time =i).plot.contourf(ax = ax[1], transform = ccrs.PlateCarree(), cmap = cmap, norm = norm)
                    (np.exp(self.y_input.isel(time =i))-1) .plot.contourf(ax = ax[2], transform = ccrs.PlateCarree(), cmap = cmap, norm = norm)
                else:
                    true = self.y_input.isel(time =i) * self.output_std + self.output_mean
                    min_t = true.values.min()
                    max_t = true.values.max()
                    levels = np.arange(min_t, max_t, 0.5)
                    y_copy.isel(time =i).plot(ax = ax[0], transform = ccrs.PlateCarree(), cmap ='RdBu_r', levels = levels)
                    y_2.isel(time =i).plot(ax = ax[1], transform = ccrs.PlateCarree(), cmap ='RdBu_r', levels = levels)
                    true.plot(ax = ax[2], transform = ccrs.PlateCarree(), cmap ='RdBu_r', levels = levels)
                ax[0].coastlines('10m')
                ax[1].coastlines('10m')
                ax[2].coastlines('10m')
                ax[0].set_title('Unet')
                ax[1].set_title('DM')
                ax[2].set_title('GT')
                # Save the figure
                filename = os.path.join(self.save_dir, f"epoch_{epoch+1}_{i}.png")
                plt.savefig(filename, bbox_inches="tight", dpi =200)
                plt.close()

            print(f"Saved prediction image to {filename}")


class DiffusionSchedule:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        beta = np.linspace(beta_start, beta_end, timesteps)
        self.beta = tf.constant(beta, dtype=tf.float32)
        self.alpha = 1 - self.beta
        self.alpha_bar = tf.math.cumprod(self.alpha)
        self.sqrt_alpha_bar = tf.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = tf.sqrt(1 - self.alpha_bar)


class EDMSchedule:
    #def __init__(self, timesteps=100, sigma_min=0.02, sigma_max=50.0, rho=7.0):
    def __init__(self, timesteps=18, sigma_min=0.002, sigma_max=800.0, rho=7.0):
        """        
        For i = 0,..., timesteps - 1
            sigma_i = ( sigma_max^(1/rho) + (i/(timesteps-1)) * (sigma_min^(1/rho) - sigma_max^(1/rho)) )^rho
        """
        self.timesteps = timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        sigmas = []
        for i in range(timesteps):
            fraction = i / (timesteps - 1)
            sigma_i = (sigma_max**(1/rho) + fraction * (sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
            sigmas.append(sigma_i)
        self.sigmas = tf.constant(sigmas, dtype=tf.float32)


class ResidualDiffusion(tf.keras.Model):
    def __init__(self, diffusion=None,
                 ema_diffusion=None,
                 scheduler=None,
                 gp_weight=10.0,
                 orog=None, unet=None, train_unet=True, varname="pr",
                 ema_decay=None,
                 use_gan_loss_constraints=None, orog_bool = True):
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
                noise_pred = self.diffusion([residual_noisy, t, average, orog_vector, init_prediction_unet, time_of_year], training=True)
            else:
                noise_pred = self.diffusion([residual_noisy, t, average, init_prediction_unet, time_of_year],
                                            training=True)
            dm_loss = self.u_loss_fn(noise, noise_pred)
                
                # reconstruct residual_gt (x0) using predicted noise (eps_theta(xt))
                # eq. 15 from Ho et al 2020
                #residual_pred = (residual_noisy - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t # TODO: remove constraint

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

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow
import xarray as xr
from dask.diagnostics import ProgressBar
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors

class GeneratorCheckpoint(Callback):
    def __init__(self, generator, filepath, period):
        super().__init__()
        self.generator = generator
        self.filepath = filepath
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            self.generator.save(f"{self.filepath}_epoch_{epoch + 1}.h5")


class DiscriminatorCheckpoint(Callback):
    def __init__(self, discriminator, filepath, period):
        super().__init__()
        self.discriminator = discriminator
        self.filepath = filepath
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            self.discriminator.save(f"{self.filepath}_epoch_{epoch + 1}.h5")


def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


class PredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, unet, generator, wgan_object,x_input, y_input, save_dir=None, batch_size =30, orog =None,
                                             output_mean =None, output_std = None, varname = "pr", orog_bool = True):
        """
        Args:
            model: The trained model.
            sample_input: A sample input tensor to generate predictions.
            save_dir: Directory where prediction images will be saved.
        """
        super(PredictionCallback, self).__init__()
        self.unet = unet
        self.generator = generator
        self.wgan = wgan_object
        self.x_input = x_input
        self.y_input = y_input
        self.save_dir = save_dir
        self.batch_size =batch_size
        self.orog = orog
        self.output_mean = output_mean
        self.output_std = output_std
        self.varname = varname
        self.orog_bool = orog_bool

        
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """Runs at the end of each epoch to save a prediction image."""
        # Generate prediction
        if epoch % 3 == 0:
            tf.random.set_seed(16)
            random_latent_vectors = tf.random.normal(
                shape=(self.batch_size,) + self.wgan.latent_dim[0]
            )
            random_latent_vectors1 = tf.random.normal(
                shape=(self.batch_size,) + self.wgan.latent_dim[1]
            )
            orog_vector = self.wgan.expand_conditional_inputs(self.orog, self.batch_size)
            unet_args = [self.x_input[0].values[:self.batch_size], orog_vector, self.x_input[1].values[:self.batch_size]] \
                if self.orog_bool else [self.x_input[0].values[:self.batch_size], self.x_input[1].values[:self.batch_size]]
            gan_end_args = unet_args

            unet_prediction = self.unet.predict(unet_args, verbose=0)

            gan_prediction = self.generator.predict([random_latent_vectors,random_latent_vectors1,unet_prediction] + gan_end_args)
            if self.varname == "pr":

                unet_final = tf.math.exp(unet_prediction) -1
                gan_final = tf.math.exp(unet_prediction + gan_prediction)-1
            else:
                unet_final = tf.squeeze(unet_prediction) * self.output_std + self.output_mean
                gan_final = tf.squeeze(unet_prediction + gan_prediction) * self.output_std + self.output_mean

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
                fig, ax = plt.subplots(1, 3, figsize = (16, 6))#, subplot_kw = dict(projection = ccrs.PlateCarree(central_longitude =171.77)))
                if self.varname =="pr":
                    y_copy.isel(time =i).plot(ax = ax[0], cmap='viridis', vmin =0, vmax =100)#transform = ccrs.PlateCarree(),
                    y_2.isel(time =i).plot(ax = ax[1], cmap='viridis', vmin =0, vmax =100)
                    (np.exp(self.y_input.isel(time =i))-1).plot(ax = ax[2], cmap='viridis', vmin =0, vmax =100)#transform = ccrs.PlateCarree(),
                else:
                    true = self.y_input.isel(time =i) * self.output_std + self.output_mean
                    min_t = true.values.min()
                    max_t = true.values.max()
                    levels = np.arange(min_t, max_t, 0.5)
                    y_copy.isel(time =i).plot.contourf(ax = ax[0], cmap ='RdBu_r', levels = levels)
                    y_2.isel(time =i).plot.contourf(ax = ax[1],  cmap ='RdBu_r', levels = levels)
                    true.plot.contourf(ax = ax[2],  cmap ='RdBu_r', levels = levels)
                # ax[0].coastlines('10m')
                # ax[1].coastlines('10m')
                # ax[2].coastlines('10m')
                ax[0].set_title('Unet')
                ax[1].set_title('GAN')
                ax[2].set_title('GT')
                # Save the figure
                filename = os.path.join(self.save_dir, f"epoch_{epoch+1}_{i}.png")
                plt.savefig(filename, bbox_inches="tight", dpi =200)
                plt.close()

            print(f"Saved prediction image to {filename}")


def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


class WGAN_Cascaded_IP(keras.Model):
    """
    A residual GAN to downscale precipitatoin, this GAN incorparates an Intensity Constraint
    """

    def __init__(self, discriminator, generator, latent_dim,
                 discriminator_extra_steps=3, gp_weight=10.0, ad_loss_factor=1e-3,orog=None, unet=None, train_unet=True,
                 intensity_weight = 1, average_intensity_weight =0.0, varname = "pr", orog_bool=True):
        super(WGAN_Cascaded_IP, self).__init__()

        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.ad_loss_factor = ad_loss_factor
        self.orog = orog
        self.unet = unet
        self.train_unet = train_unet
        self.intensity_weight = intensity_weight
        self.average_itensity_weight = average_intensity_weight
        self.varname = varname
        self.orog_bool = orog_bool

    def compile(self, d_optimizer, g_optimizer, d_loss_fn,
                g_loss_fn, u_loss_fn, u_optimizer):
        super(WGAN_Cascaded_IP, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.u_loss_fn = u_loss_fn
        self.u_optimizer = u_optimizer

    def gradient_penalty(self, batch_size, real_images, fake_images, average):
        """
        need to modify
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator([interpolated, average],
                                      training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

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
        """Train step for GAN with U-Net initialization."""
        # Process inputs
        real_images, average, time_of_year = self.process_real_images(real_images)
        real_images = tf.expand_dims(real_images, axis=-1)
        batch_size = tf.shape(real_images)[0]
        orog_vector = self.expand_conditional_inputs(self.orog, batch_size)
        # Prepare conditional arguments
        if self.orog_bool:
            unet_args = [average, orog_vector, time_of_year]
        else:
            unet_args = [average, time_of_year]
        gan_end_args = unet_args  # Same arguments for GAN
        # Create orography mask (land vs ocean)
        orog_mask = tf.cast(tf.squeeze(orog_vector) > 0.001, 'float32')
        # ============================================================================
        # PHASE 1: Train U-Net (Initial Prediction Network)
        # ============================================================================
        if self.train_unet:
            with tf.GradientTape() as tape:
                init_prediction = self.unet(unet_args, training=True)
                real_rain = real_images
                pred_rain = init_prediction
                TOTALLOSS = self.u_loss_fn(real_rain, pred_rain)
            u_gradient = tape.gradient(TOTALLOSS, self.unet.trainable_variables)
            self.u_optimizer.apply_gradients(zip(u_gradient, self.unet.trainable_variables))
        else:
            # Alternative path (not typically used based on code structure)
            with tf.GradientTape() as tape:
                init_prediction = self.unet(unet_args, training=True)
                mae_unet = self.u_loss_fn(real_images[:, :, :], init_prediction)
        # ============================================================================
        # PHASE 2: Train Discriminator
        # ============================================================================
        for _ in range(self.d_steps):
            latent_low = tf.random.normal(shape=(batch_size,) + self.latent_dim[0])
            latent_abstract = tf.random.normal(shape=(batch_size,) + self.latent_dim[1])

            with tf.GradientTape() as tape:
                # Generate predictions and residuals
                init_prediction_unet = self.unet(unet_args, training=True)
                residual_gt = real_images - init_prediction_unet

                # Generate fake residuals
                gen_inputs = [latent_low, latent_abstract, init_prediction_unet] + gan_end_args
                fake_residuals = self.generator(gen_inputs, training=True)

                # Discriminator predictions
                fake_logits = self.discriminator([fake_residuals, average], training=True)
                real_logits = self.discriminator([residual_gt, average], training=True)

                # Discriminator loss with gradient penalty
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                gp = self.gradient_penalty(batch_size, residual_gt, fake_residuals, average)
                d_loss = d_cost + gp * self.gp_weight

            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        # ============================================================================
        # PHASE 3: Train Generator
        # ============================================================================
        ensemble_size = 3

        # Generate ensemble latent vectors
        with tf.GradientTape() as tape:
            latent_low_ensemble = tf.random.normal(shape=(ensemble_size, batch_size) + self.latent_dim[0])
            latent_high_ensemble = tf.random.normal(shape=(ensemble_size, batch_size) + self.latent_dim[1])

            # Get U-Net prediction and ground truth residuals
            init_prediction_unet = self.unet(unet_args, training=True)
            residual_gt = real_images - init_prediction_unet

            # Generate ensemble of residual predictions
            generated_residuals = tf.stack([
                self.generator(
                    [latent_low_ensemble[i], latent_high_ensemble[i], init_prediction_unet] + gan_end_args,
                    training=True
                )
                for i in range(ensemble_size)
            ], axis=0)

            # Mean residual prediction across ensemble
            mean_residual = tf.reduce_mean(generated_residuals, axis=0)

            # Discriminator scores for ensemble members
            gen_logits = self.discriminator([generated_residuals[0], average], training=True)
            # --- Loss 1: Residual MAE (land-weighted) ---
            residual_gt_squeezed = residual_gt
            mean_residual_squeezed = mean_residual

            TOTALLOSS_gan = self.u_loss_fn(
                residual_gt_squeezed,
                mean_residual_squeezed
            )

            # --- Loss 2: Intensity Matching ---
            # Full predictions (U-Net + residuals)
            full_predictions = generated_residuals + init_prediction_unet
            if self.varname == "pr":
                # Precipitation-specific: max pooling for extreme precipitation
                max_real = tf.nn.max_pool2d(real_images, ksize=(20, 20), strides=10, padding="SAME")
                max_pred_ensemble = tf.stack([
                    tf.nn.max_pool2d(full_predictions[i], ksize=(20, 20), strides=10, padding="SAME")
                    for i in range(ensemble_size)
                ], axis=0)
                max_pred = tf.reduce_mean(max_pred_ensemble, axis=0)
                max_intensity_loss = tf.reduce_mean(tf.square(max_real - max_pred))

                # Average intensity
                avg_real = tf.reduce_mean(real_images, axis=[-1, -4])
                avg_pred = tf.reduce_mean(tf.stack([
                    tf.reduce_mean(full_predictions[i], axis=[-1, -4])
                    for i in range(ensemble_size)
                ], axis=0))
                avg_intensity_loss = tf.reduce_mean(tf.square(avg_real - avg_pred))
            else:
                # Other variables: min/max intensity
                max_real = tf.reduce_max(real_images, axis=[-1, -2, -3])
                max_pred = tf.reduce_max(full_predictions, axis=[0, -1, -2, -3])
                max_loss = tf.reduce_mean(tf.square(max_real - max_pred))

                min_real = tf.reduce_min(real_images, axis=[-1, -2, -3])
                min_pred = tf.reduce_min(full_predictions, axis=[0, -1, -2, -3])
                min_loss = tf.reduce_mean(tf.square(min_real - min_pred))

                max_intensity_loss = 0.5 * (min_loss + max_loss)

                # Average intensity
                avg_real = tf.reduce_mean(real_images, axis=[-1, -4])
                avg_pred = tf.reduce_mean(full_predictions, axis=[-1, -4])
                avg_intensity_loss = tf.reduce_mean(tf.square(avg_real - avg_pred))

            # --- Loss 3: Adversarial Loss ---
            adv_loss = self.ad_loss_factor * self.g_loss_fn(gen_logits)

            # --- Total Generator Loss ---
            g_loss = (adv_loss +
                      TOTALLOSS_gan +
                      self.average_itensity_weight * avg_intensity_loss +
                      self.intensity_weight * max_intensity_loss)

        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        # ============================================================================
        # Return Metrics
        # ============================================================================
        # Get max intensity predictions for logging
        if self.varname == "pr":
            max_pred_log = tf.reduce_max(max_pred_ensemble, axis=0)
        else:
            max_pred_log = max_pred

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "adv_loss": adv_loss,
            "unet_loss": TOTALLOSS,
            "gan_mae": TOTALLOSS_gan,
            "max_iten_pred": max_pred_log,
            "max_iten_true": max_real
        }

    def test_step(self, real_images):
        """
        Validation step: Computes losses and metrics on validation data without training.
        Processes inputs identically to train_step but skips optimization.
        """
        # Process inputs (same as train_step)
        real_images, average, time_of_year = self.process_real_images(real_images)
        batch_size = tf.shape(real_images)[0]
        real_images = tf.expand_dims(real_images, axis=-1)
        orog_vector = self.expand_conditional_inputs(self.orog, batch_size)

        # Prepare conditional arguments (same as train_step)
        unet_args = [average, orog_vector, time_of_year] if self.orog_bool else [average, time_of_year]
        gan_end_args = unet_args

        # Create orography mask (land vs ocean)
        orog_mask = tf.cast(tf.squeeze(orog_vector) > 0.001, 'float32')

        # ============================================================================
        # U-Net Forward Pass
        # ============================================================================
        init_prediction = self.unet(unet_args, training=False)

        # Compute U-Net loss (weighted land/ocean reconstruction)
        real_rain = real_images
        pred_rain = init_prediction

        total_loss_val = self.u_loss_fn(real_rain, pred_rain)

        # Compute ground-truth residuals
        residual_gt = real_images - init_prediction
        print("upto here 0")
        # ============================================================================
        # Discriminator Metrics (Single Sample)
        # ============================================================================
        latent_low = tf.random.normal(shape=(batch_size,) + self.latent_dim[0])
        latent_high = tf.random.normal(shape=(batch_size,) + self.latent_dim[1])

        gen_inputs = [latent_low, latent_high, init_prediction] + gan_end_args
        fake_residuals = self.generator(gen_inputs, training=False)

        # Discriminator forward passes
        fake_logits = self.discriminator([fake_residuals, average], training=False)
        real_logits = self.discriminator([residual_gt, average], training=False)
        print("upto here 1")
        # Discriminator loss
        d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
        gp = self.gradient_penalty(batch_size, residual_gt, fake_residuals, average)
        d_loss = d_cost + gp * self.gp_weight

        # ============================================================================
        # Generator Metrics (one-member)
        # ===========================================================================

        # Generate ensemble latent vectors

        # Generate ensemble of residual predictions
        generated_residuals = self.generator(gen_inputs,
                training=False
            )

        # First sample for intensity/adversarial metrics
        first_residual = generated_residuals
        # Mean residual for reconstruction metrics
        mean_residual = first_residual
        # --- Loss 1: Residual MAE (land-weighted) ---
        residual_gt_squeezed = residual_gt
        mean_residual_squeezed = mean_residual

        # loss_ocean_gan = self.u_loss_fn(
        #     residual_gt_squeezed * (1 - orog_mask),
        #     mean_residual_squeezed * (1 - orog_mask)
        # )
        # loss_land_gan = self.u_loss_fn(
        #     residual_gt_squeezed * orog_mask,
        #     mean_residual_squeezed * orog_mask
        # )
        total_loss_gan_val = self.u_loss_fn(
            residual_gt_squeezed,
            mean_residual_squeezed
        )

        print("upto here -1")
        # --- Loss 2: Intensity Matching ---
        # Full predictions (U-Net + residuals)
        first_full_pred = first_residual + init_prediction

        if self.varname == "pr":
            # Precipitation-specific: max pooling for extreme precipitation
            max_real = tf.nn.max_pool2d(real_images, ksize=(20, 20), strides=10, padding="SAME")
            max_pred = tf.nn.max_pool2d(first_full_pred, ksize=(20, 20), strides=10, padding="SAME")
            max_intensity_loss = tf.reduce_mean(tf.square(max_real - max_pred))

            # Average intensity (spatial mean)
            avg_real = tf.reduce_mean(real_images, axis=[1, 2])
            avg_pred = tf.reduce_mean(first_full_pred, axis=[1, 2])
            avg_intensity_loss = tf.reduce_mean(tf.square(avg_real - avg_pred))
        else:
            # Other variables: min/max intensity
            max_real = tf.reduce_max(real_images, axis=[1, 2, 3])
            max_pred = tf.reduce_max(first_full_pred, axis=[1, 2, 3])
            max_loss = tf.reduce_mean(tf.square(max_real - max_pred))

            min_real = tf.reduce_min(real_images, axis=[1, 2, 3])
            min_pred = tf.reduce_min(first_full_pred, axis=[1, 2, 3])
            min_loss = tf.reduce_mean(tf.square(min_real - min_pred))

            max_intensity_loss = 0.5 * (min_loss + max_loss)

            # Average intensity (spatial mean)
            avg_real = tf.reduce_mean(real_images, axis=[1, 2])
            avg_pred = tf.reduce_mean(first_full_pred, axis=[1, 2])
            avg_intensity_loss = tf.reduce_mean(tf.square(avg_real - avg_pred))

        # --- Loss 3: Adversarial Loss ---
        adv_loss = self.ad_loss_factor * self.g_loss_fn(fake_logits)

        # --- Total Generator Loss ---
        g_loss = (adv_loss +
                  total_loss_gan_val +
                  self.average_itensity_weight * avg_intensity_loss +
                  self.intensity_weight * max_intensity_loss)

        # ============================================================================
        # Return Metrics
        # ============================================================================
        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "adv_loss": adv_loss,
            "unet_loss":  total_loss_val,
            "gan_mae": total_loss_gan_val,
        }


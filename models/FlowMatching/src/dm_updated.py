"""
Single-Stage Flow Matching for Precipitation Downscaling
with Heavy-Tailed Student-t Prior (t-Flow)

References:
  - Lipman et al. 2022 "Flow Matching for Generative Modeling"
  - Liu et al. 2022 "Rectified Flow"
  - Pandey et al. 2024 "Heavy-Tailed Diffusion Models" (t-Flow)

Changes vs previous version:
  - Cosine interpolant replaces t² (stable nonlinear schedule, no singularity)
  - Analytical cosine velocity target (no 1/(1-t) divergence)
  - t sampled in [0, 0.95] to avoid noisy near-terminal region
  - Quantile regression loss (q=0.95) to penalise extreme overestimation
  - Total variation loss on predicted x1 to reduce spatial graininess
  - sigma_z clipping retained and tightened
  - Heun (RK2) ODE sampler with velocity clipping retained
  - num_ode_steps bumped to 100 in sampler for smoother inference
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tqdm import tqdm
import sys
sys.path.append(r'/esi/project/niwa00018/rampaln/PUBLICATIONS/2026/CORDEX_ML_TF/models/FlowMatching')
from src.layers import *


# =============================================================================
# Student-t Sampling
# =============================================================================

def sample_student_t(shape, df, sigma_z) -> tf.Tensor:
    df_f       = tf.cast(df, tf.float32)
    z          = tf.random.normal(shape=shape)
    batch_size = shape[0] if isinstance(shape, (list, tuple)) else shape[0]
    chi2_shape = tf.stack([batch_size, 1, 1, 1])
    v          = tf.random.gamma(shape=chi2_shape, alpha=df_f / 2.0, beta=0.5)
    return sigma_z * z / tf.sqrt(v / df_f)


# =============================================================================
# Flow Schedule  —  cosine interpolant (stable, nonlinear, no singularity)
# =============================================================================

class FlowSchedule:
    """
    Cosine interpolant schedule.

    Interpolant:   x_t = (1 - alpha(t)) * x0 + alpha(t) * x1
                   alpha(t) = (1 - cos(pi * t)) / 2

    Velocity target is the analytical time-derivative:
                   v*(t) = d/dt x_t = dalpha/dt * (x1 - x0)
                   dalpha/dt = (pi/2) * sin(pi * t)

    This completely avoids the 1/(1-t) singularity of the previous
    time-aware version while still giving a nonlinear (slow start/end,
    fast middle) schedule that helps extremes.
    """

    def __init__(self, num_ode_steps: int = 100, t_embed_scale: float = 1000.0):
        self.num_ode_steps = num_ode_steps   # bumped to 100 for smoother inference
        self.t_embed_scale = t_embed_scale

    def sample_t(self, batch_size: int) -> tf.Tensor:
        # Clamp away from t=1 — velocity targets are noisiest near terminal time
        t = tf.random.uniform(shape=(batch_size,), minval=0.0, maxval=0.95)
        return tf.reshape(t, (batch_size, 1, 1, 1))

    def interpolate(self, x0: tf.Tensor, x1: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        alpha = (1.0 - tf.cos(np.pi * t)) / 2.0
        return (1.0 - alpha) * x0 + alpha * x1

    def velocity_target(self, x0: tf.Tensor, x1: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """Analytical derivative of the cosine interpolant — no singularity."""
        dalpha_dt = (np.pi / 2.0) * tf.sin(np.pi * t)
        return dalpha_dt * (x1 - x0)

    def t_for_network(self, t: tf.Tensor, batch_size: int) -> tf.Tensor:
        return tf.reshape(t, (batch_size, 1)) * self.t_embed_scale

    def ode_times(self) -> tf.Tensor:
        dt = 1.0 / self.num_ode_steps
        return tf.cast(tf.linspace(0.0, 1.0 - dt, self.num_ode_steps), tf.float32)


# =============================================================================
# Spatial losses
# =============================================================================

def tv_loss(x: tf.Tensor) -> tf.Tensor:
    """
    Total variation loss on predicted x1 to suppress spatial graininess.
    Applied to the predicted image rather than the velocity field so that
    it penalises checkerboard artefacts without over-smoothing fine structure.
    """
    dy = x[:, 1:, :, :] - x[:, :-1, :, :]
    dx = x[:, :, 1:, :] - x[:, :, :-1, :]
    return tf.reduce_mean(tf.abs(dy)) + tf.reduce_mean(tf.abs(dx))


def quantile_loss(y_true: tf.Tensor, y_pred: tf.Tensor, q: float = 0.95) -> tf.Tensor:
    """
    Quantile (pinball) regression loss at quantile q.
    Penalises over-prediction of extremes, which directly addresses the
    tail overestimation visible in the histogram.
    """
    residual = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q * residual, (q - 1.0) * residual))


# =============================================================================
# Flow UNet  (unchanged from previous version)
# =============================================================================

def build_flow_unet(
        input_size,
        resize_output,
        num_filters,
        num_channels,
        time_embed_dim: int = 256,
        orog_predictor: bool = True,
        varname: str = "pr",
        final_activation="linear"
) -> tf.keras.Model:
    """
    Velocity-predicting UNet for single-stage flow matching.
    Architecture structure, attention types, and res_block chains
    are matched to the res_gan architecture.
    """

    # ── Inputs ────────────────────────────────────────────────────────────────
    inp_x_t      = tf.keras.Input(shape=[resize_output[0], resize_output[1], 1], name="x_t")
    inp_lr       = tf.keras.Input(shape=[input_size[0], input_size[1], num_channels], name="x_lr")
    inp_t        = tf.keras.Input((), dtype=tf.int32, name="timestep")
    time_of_year = tf.keras.Input(shape=[1], name="time_input")

    if orog_predictor:
        inp_static_hr = tf.keras.Input(
            shape=[resize_output[0], resize_output[1], 1], name="static_hr"
        )

    # ── Timestep embedding ───────────────────────────────────────────────────
    temb = SinusoidalTimeEmbedding(embed_dim=32)(inp_t)
    temb = tf.keras.layers.Dense(time_embed_dim, activation="gelu")(temb)

    # ── Seasonal conditioning (FiLM) ─────────────────────────────────────────
    conditioned_lr = TimeFilmLayer(
        num_channels=num_channels, hidden_dim=128,
        use_sinusoidal=True, max_period=365,
        name='time_film_lr', varname=varname
    )([inp_lr, time_of_year])

    conditioned_x_t = TimeFilmLayer(
        num_channels=1, hidden_dim=32,
        use_sinusoidal=True, max_period=365,
        name='time_film_xt', varname=varname
    )([inp_x_t, time_of_year])

    # ── HR encoder ───────────────────────────────────────────────────────────
    if orog_predictor:
        concat_hr = tf.keras.layers.Concatenate(axis=-1)([conditioned_x_t, inp_static_hr])
    else:
        concat_hr = conditioned_x_t

    x, temp1 = down_block(concat_hr, num_filters[2], kernel_size=5, i=4,
                          use_pool=False, attn_type="channel", varname=varname)
    x = FiLMResidual(num_filters[2])(x, temb)

    x, temp2 = down_block(x, num_filters[1], kernel_size=3, i=1,
                          attn_type="cbam", varname=varname)
    x = FiLMResidual(num_filters[1])(x, temb)

    x, temp3 = down_block(x, num_filters[2], kernel_size=3, i=2,
                          attn_type="None", varname=varname)
    x = FiLMResidual(num_filters[2])(x, temb)

    # ── LR branch ────────────────────────────────────────────────────────────
    x1 = res_block_initial(conditioned_lr,  [num_filters[2] * 2], 3, [1, 1], "lr_block1", attn_type="None", varname=varname)
    x1 = res_block_initial(x1, [num_filters[2] * 2], 3, [1, 1], "lr_block2", attn_type="None", varname=varname)
    x1 = res_block_initial(x1, [64],  3, [1, 1], "lr_block3", attn_type="None", varname=varname)
    x1 = res_block_initial(x1, [128], 5, [1, 1], "lr_block4", attn_type="None", varname=varname)

    # ── Bottleneck ───────────────────────────────────────────────────────────
    concat_scales = tf.keras.layers.Concatenate(axis=-1)([x1, x])
    x = res_block_initial(concat_scales, [256], 3, [1, 1], "merge_block",
                          attn_type="self", varname=varname)
    x = FiLMResidual(256)(x, temb)

    # ── Decoder ──────────────────────────────────────────────────────────────
    x = up_block(x, temp3, kernel_size=3, filters=num_filters[2], i=0,
                 concat=True, attn_type="None", varname=varname)
    x = FiLMResidual(num_filters[2])(x, temb)

    x = up_block(x, temp2, kernel_size=3, filters=num_filters[1], i=2,
                 concat=True, attn_type="cbam", varname=varname)
    x = FiLMResidual(num_filters[1])(x, temb)

    x = up_block(x, temp1, kernel_size=5, filters=num_filters[0], i=3,
                 concat=True, attn_type="cbam", varname=varname)
    x = FiLMResidual(num_filters[0])(x, temb)

    # ── Output convolutions ───────────────────────────────────────────────────
    output = TimeFilmLayer(
        num_channels=num_filters[0], hidden_dim=128,
        use_sinusoidal=True, max_period=365, name='time_film_out'
    )([x, time_of_year])

    output = res_block_initial(output, [32], 5, [1, 1], "output_conv",
                               attn_type="channel", varname=varname)

    output = tf.keras.layers.Conv2D(16, 5, activation=final_activation, padding='same')(output)
    out    = tf.keras.layers.Conv2D(1,  1, activation=final_activation, padding='same',
                                    name="velocity")(output)

    # ── Assembly ─────────────────────────────────────────────────────────────
    input_layers = ([inp_x_t, inp_t, inp_lr, inp_static_hr, time_of_year]
                    if orog_predictor else
                    [inp_x_t, inp_t, inp_lr, time_of_year])

    return tf.keras.Model(inputs=input_layers, outputs=out,
                          name="single_stage_flow_unet")


# =============================================================================
# ODE Sampler  —  Heun (RK2) with velocity clipping
# =============================================================================

def sample_ode(
    flow_net,
    average,
    orog_vector,
    time_of_year,
    schedule: FlowSchedule,
    sigma_z: float,
    batch_size: int,
    spatial_shape: tuple,
    orog_bool: bool = True,
    student_t_df: float = 15.0,
) -> tf.Tensor:
    """
    Heun (RK2) integrator with tanh velocity clipping.
    Uses num_ode_steps=100 (set on schedule) for smooth integration.
    """
    H, W  = spatial_shape
    x_t   = sample_student_t((batch_size, H, W, 1), student_t_df, sigma_z)
    dt    = 1.0 / schedule.num_ode_steps
    times = schedule.ode_times()

    for i in tqdm(range(schedule.num_ode_steps), desc="Flow ODE"):
        t_val = times[i]
        t_net = tf.fill((batch_size, 1), t_val * schedule.t_embed_scale)

        def _inputs(xt):
            return ([xt, t_net, average, orog_vector, time_of_year]
                    if orog_bool else [xt, t_net, average, time_of_year])

        # First Heun step
        v1 = flow_net(_inputs(x_t), training=False)
        v1 = 5.0 * tf.tanh(v1 / 5.0)
        x_mid = x_t + v1 * dt

        # Second Heun step
        v2 = flow_net(_inputs(x_mid), training=False)
        v2 = 5.0 * tf.tanh(v2 / 5.0)

        x_t = x_t + 0.5 * (v1 + v2) * dt

    return x_t


# =============================================================================
# Single-Stage Flow Matching Model
# =============================================================================

class SingleStageFlowMatching(tf.keras.Model):
    """
    Loss = flow_matching + 0.5 * terminal_consistency
           + lambda_reg * normalisation_reg
           + tv_weight * total_variation          (← reduces graininess)
           + quantile_weight * quantile_loss      (← reduces extreme overestimation)
    """

    def __init__(
        self,
        flow_net,
        ema_flow_net,
        schedule: FlowSchedule,
        orog,
        varname: str = "pr",
        ema_decay: float = 0.985,
        lambda_reg: float = 0.25,
        beta_ema_sigma: float = 0.01,
        use_gan_loss_constraints: bool = False,
        orog_bool: bool = True,
        intensity_weight: float = 4.25,
        student_t_df: float = 15.0,   # raised from 5–6; reduces heavy-tail noise
        tv_weight: float = 0.1,       # total variation on predicted x1, orginally 0.5
        quantile_weight: float = 0.15,  # quantile regression at q=0.95
        quantile_q: float = 0.99,
    ):
        super().__init__()
        self.flow_net                 = flow_net
        self.ema_flow_net             = ema_flow_net
        self.schedule                 = schedule
        self.orog                     = orog
        self.varname                  = varname
        self.ema_decay                = ema_decay
        self.lambda_reg               = lambda_reg
        self.beta_ema_sigma           = beta_ema_sigma
        self.use_gan_loss_constraints = use_gan_loss_constraints
        self.orog_bool                = orog_bool
        self.intensity_weight         = intensity_weight
        self.student_t_df             = tf.constant(student_t_df, dtype=tf.float32)
        self.tv_weight                = tv_weight
        self.quantile_weight          = quantile_weight
        self.quantile_q               = quantile_q

        self.sigma_z = tf.Variable(
            initial_value=1.0, trainable=False, dtype=tf.float32, name="sigma_z"
        )
        self.flow_loss_tracker        = tf.keras.metrics.Mean(name="flow_loss")
        self.val_flow_loss_tracker    = tf.keras.metrics.Mean(name="val_flow_loss")
        self.tv_loss_tracker          = tf.keras.metrics.Mean(name="tv_loss")
        self.quantile_loss_tracker    = tf.keras.metrics.Mean(name="quantile_loss")

        if self.use_gan_loss_constraints:
            self.intensity_tracker     = tf.keras.metrics.Mean(name="intensity_loss")
            self.val_intensity_tracker = tf.keras.metrics.Mean(name="val_intensity_loss")

    def compile(self, fm_optimizer, loss_fn):
        super().compile()
        self.fm_optimizer = fm_optimizer
        self.loss_fn      = loss_fn

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _expand_orog(orog, batch_size):
        x = tf.expand_dims(orog, axis=0)
        x = tf.repeat(x, repeats=batch_size, axis=0)
        return tf.expand_dims(x, axis=-1)

    @staticmethod
    def _unpack_batch(batch):
        output_vars, averages = batch
        real_images  = output_vars["pr"]
        average      = averages["X"]
        time_of_year = averages["time_of_year"]
        return real_images, average, time_of_year

    def _flow_args(self, x_t, t_net, average, orog_vector, time_of_year):
        if self.orog_bool:
            return [x_t, t_net, average, orog_vector, time_of_year]
        return [x_t, t_net, average, time_of_year]

    def _sample_x0(self, shape) -> tf.Tensor:
        return sample_student_t(shape, self.student_t_df, self.sigma_z)

    def _update_sigma_z(self, real_images):
        """EMA update with tight clipping to prevent sigma_z explosion."""
        std = tf.math.reduce_std(real_images)
        std = tf.clip_by_value(std, 0.3, 2.0)
        self.sigma_z.assign(
            (1.0 - self.beta_ema_sigma) * self.sigma_z + self.beta_ema_sigma * std
        )

    # ── Training step ─────────────────────────────────────────────────────────

    def train_step(self, batch):
        real_images, average, time_of_year = self._unpack_batch(batch)
        real_images = tf.expand_dims(real_images, axis=-1)
        batch_size  = tf.shape(real_images)[0]
        orog_vector = self._expand_orog(self.orog, batch_size)
        self._update_sigma_z(real_images)

        with tf.GradientTape() as tape:
            x1 = real_images
            x0 = self._sample_x0(tf.shape(x1))

            # Cosine schedule — t clamped to [0, 0.95] in sample_t
            t     = self.schedule.sample_t(batch_size)
            t_net = self.schedule.t_for_network(t, batch_size)
            x_t   = self.schedule.interpolate(x0, x1, t)

            # Analytical velocity target (no singularity)
            v_target = self.schedule.velocity_target(x0, x1, t)

            v_pred = self.flow_net(
                self._flow_args(x_t, t_net, average, orog_vector, time_of_year),
                training=True
            )
            # Velocity clipping
            v_pred = 5.0 * tf.tanh(v_pred / 5.0)

            # ── Loss components ───────────────────────────────────────────────

            # 1. Flow matching (time-weighted MSE)
            weight   = 1.0 / (t + 0.05)
            fm_loss  = tf.reduce_mean(weight * tf.square(v_target - v_pred))

            # 2. Terminal consistency — predicted x1 should match real x1
            #    Use cosine alpha to recover x1: x1_pred = (x_t - (1-alpha)*x0) / alpha
            #    Simpler: x1_pred ≈ x_t + v_pred * (1 - alpha(t))
            alpha    = (1.0 - tf.cos(np.pi * t)) / 2.0
            x1_pred  = x_t + v_pred * (1.0 - alpha)
            term_loss = tf.reduce_mean(tf.square(x1_pred - x1))

            # 3. Light normalisation regularisation
            reg_loss = self.lambda_reg * tf.reduce_mean(
                tf.square(x1 / (self.sigma_z + 1e-6))
            )

            # 4. Total variation on predicted x1 — suppresses spatial graininess
            tv = tv_loss(x1_pred)

            # 5. Quantile regression — penalises extreme overestimation
            ql = quantile_loss(x1, x1_pred, q=self.quantile_q)

            total_loss = (fm_loss
                          + 0.5  * term_loss
                          + reg_loss
                          + self.tv_weight       * tv
                          + self.quantile_weight * ql)

            if self.use_gan_loss_constraints:
                max_pred       = tf.reduce_max(x1_pred, axis=[1, 2])
                max_true       = tf.reduce_max(real_images, axis=[1, 2])
                intensity_loss = tf.reduce_mean(tf.abs(max_pred - max_true))
                total_loss    += self.intensity_weight * intensity_loss * tf.reduce_mean(t)

        grads = tape.gradient(total_loss, self.flow_net.trainable_variables)
        self.fm_optimizer.apply_gradients(zip(grads, self.flow_net.trainable_variables))

        # EMA update
        for w, ema_w in zip(self.flow_net.weights, self.ema_flow_net.weights):
            ema_w.assign(self.ema_decay * ema_w + (1.0 - self.ema_decay) * w)

        self.flow_loss_tracker.update_state(fm_loss)
        self.tv_loss_tracker.update_state(tv)
        self.quantile_loss_tracker.update_state(ql)

        results = {
            "flow_loss":     self.flow_loss_tracker.result(),
            "term_loss":     term_loss,
            "tv_loss":       self.tv_loss_tracker.result(),
            "quantile_loss": self.quantile_loss_tracker.result(),
            "sigma_z":       self.sigma_z,
        }
        if self.use_gan_loss_constraints:
            self.intensity_tracker.update_state(intensity_loss)
            results["intensity_loss"] = self.intensity_tracker.result()
        return results

    # ── Validation step ───────────────────────────────────────────────────────

    def test_step(self, batch):
        real_images, average, time_of_year = self._unpack_batch(batch)
        real_images = tf.expand_dims(real_images, axis=-1)
        batch_size  = tf.shape(real_images)[0]
        orog_vector = self._expand_orog(self.orog, batch_size)

        x1 = real_images
        x0 = self._sample_x0(tf.shape(x1))

        t     = self.schedule.sample_t(batch_size)
        t_net = self.schedule.t_for_network(t, batch_size)
        x_t   = self.schedule.interpolate(x0, x1, t)

        v_target = self.schedule.velocity_target(x0, x1, t)
        v_pred   = self.ema_flow_net(
            self._flow_args(x_t, t_net, average, orog_vector, time_of_year),
            training=False
        )
        v_pred = 5.0 * tf.tanh(v_pred / 5.0)

        weight    = 1.0 / (t + 0.05)
        fm_loss   = tf.reduce_mean(weight * tf.square(v_target - v_pred))

        alpha     = (1.0 - tf.cos(np.pi * t)) / 2.0
        x1_pred   = x_t + v_pred * (1.0 - alpha)
        term_loss = tf.reduce_mean(tf.square(x1_pred - x1))
        tv        = tv_loss(x1_pred)
        ql        = quantile_loss(x1, x1_pred, q=self.quantile_q)

        total_loss = (fm_loss
                      + 0.5  * term_loss
                      + self.tv_weight       * tv
                      + self.quantile_weight * ql)

        self.val_flow_loss_tracker.update_state(total_loss)

        results = {
            "val_flow_loss":  self.val_flow_loss_tracker.result(),
            "val_term_loss":  term_loss,
            "val_tv_loss":    tv,
            "val_quant_loss": ql,
        }

        if self.use_gan_loss_constraints:
            max_pred       = tf.reduce_max(x1_pred, axis=[1, 2])
            max_true       = tf.reduce_max(real_images, axis=[1, 2])
            intensity_loss = tf.reduce_mean(tf.abs(max_pred - max_true))
            self.val_intensity_tracker.update_state(intensity_loss)
            results["val_intensity_loss"] = self.val_intensity_tracker.result()

        return results

    @property
    def metrics(self):
        m = [self.flow_loss_tracker, self.val_flow_loss_tracker,
             self.tv_loss_tracker, self.quantile_loss_tracker]
        if self.use_gan_loss_constraints:
            m += [self.intensity_tracker, self.val_intensity_tracker]
        return m


# =============================================================================
# Prediction Callback
# =============================================================================

class PredictionCallbackFlow(tf.keras.callbacks.Callback):

    def __init__(
            self, flow_model_obj, x_input, y_input, save_dir,
            batch_size=16, orog=None, output_mean=None, output_std=None,
            varname="pr", orog_bool=True, plot_every=5, n_panels=8,
            student_t_df=15.0,
    ):
        super().__init__()
        self.fm_obj      = flow_model_obj
        self.flow_net    = flow_model_obj.ema_flow_net
        self.schedule    = flow_model_obj.schedule
        self.orog_bool   = orog_bool
        self.save_dir    = save_dir
        self.batch_size  = batch_size
        self.output_mean = output_mean
        self.output_std  = output_std
        self.varname     = varname
        self.plot_every  = plot_every
        self.n_panels    = n_panels
        self.student_t_df = student_t_df

        self.x1_tensor   = tf.cast(x_input[0].values[:batch_size], tf.float32)
        self.doy_tensor  = tf.cast(x_input[1].values[:batch_size], tf.float32)
        self.y_tensor    = tf.cast(y_input.values[:batch_size], tf.float32)
        self.orog_tensor = tf.cast(orog, tf.float32)

        os.makedirs(save_dir, exist_ok=True)

    def _sample_heun(self, H, W):
        """Heun (RK2) sampler with velocity clipping."""
        x_t = sample_student_t(
            (self.batch_size, H, W, 1),
            self.student_t_df,
            float(self.fm_obj.sigma_z.numpy())
        )
        dt          = 1.0 / self.schedule.num_ode_steps
        times       = self.schedule.ode_times()
        orog_vector = SingleStageFlowMatching._expand_orog(
            self.orog_tensor, self.batch_size
        )

        for t_val in times:
            t_net = tf.fill((self.batch_size, 1), t_val * self.schedule.t_embed_scale)

            def _inp(xt):
                return ([xt, t_net, self.x1_tensor, orog_vector, self.doy_tensor]
                        if self.orog_bool else
                        [xt, t_net, self.x1_tensor, self.doy_tensor])

            v1    = 5.0 * tf.tanh(self.flow_net(_inp(x_t),           training=False) / 5.0)
            x_mid = x_t + v1 * dt
            v2    = 5.0 * tf.tanh(self.flow_net(_inp(x_mid),         training=False) / 5.0)
            x_t   = x_t + 0.5 * (v1 + v2) * dt

        return x_t

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.plot_every != 0:
            return

        tf.random.set_seed(42)
        H, W      = self.y_tensor.shape[1], self.y_tensor.shape[2]
        flow_pred = self._sample_heun(H, W)

        # Inverse transform (clamp to ≥ 0 for precipitation)
        if self.varname == "pr":
            fm_out   = tf.maximum(tf.math.exp(flow_pred) - 1.0, 0.0)
            true_out = tf.maximum(
                tf.math.exp(tf.expand_dims(self.y_tensor, -1)) - 1.0, 0.0
            )
        else:
            fm_out   = tf.squeeze(flow_pred) * self.output_std + self.output_mean
            true_out = self.y_tensor * self.output_std + self.output_mean

        for i in range(min(self.n_panels, self.batch_size)):
            fig, axes2 = plt.subplots(1, 3, figsize=(14, 5))
            axes = axes2[:2]

            if self.varname == "pr":
                for ax, data, title in zip(
                    axes,
                    [fm_out[i, ..., 0], true_out[i, ..., 0]],
                    [f"t-Flow df={self.student_t_df:.0f}", "Ground Truth"]
                ):
                    ax.imshow(data.numpy(), cmap="viridis",
                              vmin=0, vmax=100, origin="upper")
                    ax.set_title(title)
                    ax.axis("off")
            else:
                vmin = float(tf.reduce_min(true_out[i]).numpy())
                vmax = float(tf.reduce_max(true_out[i]).numpy())
                for ax, data, title in zip(
                    axes,
                    [fm_out[i], true_out[i]],
                    [f"t-Flow df={self.student_t_df:.0f}", "Ground Truth"]
                ):
                    ax.imshow(data.numpy(), cmap="RdBu_r",
                              vmin=vmin, vmax=vmax, origin="upper")
                    ax.set_title(title)
                    ax.axis("off")

            if self.varname == "pr":
                axes2[-1].hist(
                    np.clip(fm_out.numpy().ravel(), 0, 500),
                    bins=np.arange(0, 500, 10),
                    histtype='step', color='r', label="Prediction"
                )
                axes2[-1].hist(
                    np.clip(true_out.numpy().ravel(), 0, 500),
                    bins=np.arange(0, 500, 10),
                    histtype='step', color='k', label="GT"
                )
                axes2[-1].set_yscale('log')
                axes2[-1].legend()
            else:
                axes2[-1].hist(fm_out.numpy().ravel(),  bins=50,
                               histtype='step', color='r', label="Prediction")
                axes2[-1].hist(true_out.numpy().ravel(), bins=50,
                               histtype='step', color='k', label="GT")
                axes2[-1].legend()

            fname = os.path.join(
                self.save_dir, f"epoch_{epoch + 1:04d}_sample_{i:02d}.png"
            )
            plt.tight_layout()
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"\n[FlowCallback] Epoch {epoch + 1}: saved panels → {self.save_dir}")


# =============================================================================
# Checkpoint Callback
# =============================================================================

class ModelCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_ref, filepath: str, period: int = 5):
        super().__init__()
        self.model_ref = model_ref
        self.filepath  = filepath
        self.period    = period

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            self.model_ref.save_weights(
                f"{self.filepath}_epoch{epoch + 1:04d}.weights.h5"
            )
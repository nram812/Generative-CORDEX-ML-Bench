import tensorflow as tf
import numpy as np


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow
import xarray as xr
from dask.diagnostics import ProgressBar
from tensorflow.keras.callbacks import Callback
import numpy as np
import pandas as pd
import math

class BicubicUpSampling2D(tf.keras.layers.Layer):
    def __init__(self, size, **kwargs):
        super(BicubicUpSampling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs):
        return tf.image.resize(inputs, [int(inputs.shape[1] * self.size[0]), int(inputs.shape[2] * self.size[1])],
                               method=tf.image.ResizeMethod.BILINEAR)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'size': self.size
        })
        return config

class CosineAnnealingLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Cosine Annealing Learning Rate Schedule with optional warmup.

    Args:
        initial_learning_rate: Starting learning rate (or warmup start if warmup is used)
        decay_steps: Number of steps for the cosine decay phase
        alpha: Minimum learning rate as fraction of max_lr (default: 0.0)
        warmup_target: Target learning rate after warmup (if None, no warmup)
        warmup_steps: Number of warmup steps
    """

    def __init__(
            self,
            initial_learning_rate,
            decay_steps,
            alpha=0.0,
            warmup_target=None,
            warmup_steps=0,
            name="CosineAnnealingLR"
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.name = name
        self.warmup_steps = warmup_steps
        self.warmup_target = warmup_target

        if self.decay_steps <= 0:
            raise ValueError(f"decay_steps must be > 0, got {self.decay_steps}")

    def __call__(self, step):
        with tf.name_scope(self.name):
            # Convert to tensor
            initial_lr = tf.cast(self.initial_learning_rate, tf.float32)
            step = tf.cast(step, tf.float32)
            decay_steps = tf.cast(self.decay_steps, tf.float32)

            if self.warmup_target is not None:
                # With warmup
                warmup_target = tf.cast(self.warmup_target, tf.float32)
                warmup_steps = tf.cast(self.warmup_steps, tf.float32)

                # Warmup phase: linear increase
                warmup_lr = initial_lr + (warmup_target - initial_lr) * (step / warmup_steps)

                # Decay phase: cosine annealing from warmup_target
                decay_step = step - warmup_steps
                cosine_decay = 0.5 * (1.0 + tf.cos(
                    math.pi * tf.minimum(decay_step, decay_steps) / decay_steps
                ))
                decayed_lr = (warmup_target - warmup_target * self.alpha) * cosine_decay + warmup_target * self.alpha

                # Choose based on current step
                lr = tf.cond(
                    step < warmup_steps,
                    lambda: warmup_lr,
                    lambda: decayed_lr
                )
            else:
                # No warmup: decay from initial_lr
                cosine_decay = 0.5 * (1.0 + tf.cos(
                    math.pi * tf.minimum(step, decay_steps) / decay_steps
                ))
                lr = (initial_lr - initial_lr * self.alpha) * cosine_decay + initial_lr * self.alpha

            return lr

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "warmup_target": self.warmup_target,
            "warmup_steps": self.warmup_steps,
            "name": self.name,
        }


class SEBlock(tf.keras.layers.Layer):
    """Squeeze-and-Excitation block"""

    def __init__(self, ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]

        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(channels // self.ratio, activation='relu')
        self.fc2 = tf.keras.layers.Dense(channels, activation='sigmoid')
        self.reshape = tf.keras.layers.Reshape((1, 1, channels))
        self.multiply = tf.keras.layers.Multiply()

        super(SEBlock, self).build(input_shape)

    def call(self, inputs):
        se = self.gap(inputs)
        se = self.fc1(se)
        se = self.fc2(se)
        se = self.reshape(se)
        return self.multiply([inputs, se])

    def get_config(self):
        config = super(SEBlock, self).get_config()
        config.update({'ratio': self.ratio})
        return config


class BicubicUpSampling2D(tf.keras.layers.Layer):
    def __init__(self, size, **kwargs):
        super(BicubicUpSampling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs):
        return tf.image.resize(inputs, [int(inputs.shape[1] * self.size[0]), int(inputs.shape[2] * self.size[1])],
                               method=tf.image.ResizeMethod.BILINEAR)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'size': self.size
        })
        return config


class SelfAttention2D(tf.keras.layers.Layer):
    """Self-attention for 2D feature maps"""

    def __init__(self, **kwargs):
        super(SelfAttention2D, self).__init__(**kwargs)

    def build(self, input_shape):
        channels = input_shape[-1]

        self.query_conv = tf.keras.layers.Conv2D(channels // 8, 1)
        self.key_conv = tf.keras.layers.Conv2D(channels // 8, 1)
        self.value_conv = tf.keras.layers.Conv2D(channels, 1)
        self.proj_conv = tf.keras.layers.Conv2D(channels, 1)

        self.gamma = self.add_weight(
            name='gamma',
            shape=(),
            initializer='zeros',
            trainable=True
        )

        self.channels = channels
        super(SelfAttention2D, self).build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        # Query, Key, Value projections
        query = self.query_conv(inputs)
        key = self.key_conv(inputs)
        value = self.value_conv(inputs)

        # Reshape: [B, H, W, C] -> [B, H*W, C]
        query = tf.reshape(query, [batch_size, height * width, self.channels // 8])
        key = tf.reshape(key, [batch_size, height * width, self.channels // 8])
        value = tf.reshape(value, [batch_size, height * width, self.channels])

        # Attention scores: [B, H*W, H*W]
        attention = tf.matmul(query, key, transpose_b=True)
        attention = tf.nn.softmax(attention / tf.sqrt(tf.cast(self.channels // 8, tf.float32)))

        # Apply attention to values: [B, H*W, C]
        out = tf.matmul(attention, value)

        # Reshape back: [B, H*W, C] -> [B, H, W, C]
        out = tf.reshape(out, [batch_size, height, width, self.channels])

        # Project back to original channels
        out = self.proj_conv(out)

        # Residual connection with learnable scale
        return inputs + self.gamma * out

    def get_config(self):
        return super(SelfAttention2D, self).get_config()


class CBAMBlock(tf.keras.layers.Layer):
    """CBAM: Convolutional Block Attention Module (channel + spatial attention)"""

    def __init__(self, ratio=8, **kwargs):
        super(CBAMBlock, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]

        # Channel attention components
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.max_pool = tf.keras.layers.GlobalMaxPooling2D()

        self.fc1_avg = tf.keras.layers.Dense(channels // self.ratio, activation='relu')
        self.fc2_avg = tf.keras.layers.Dense(channels)

        self.fc1_max = tf.keras.layers.Dense(channels // self.ratio, activation='relu')
        self.fc2_max = tf.keras.layers.Dense(channels)

        self.channel_add = tf.keras.layers.Add()
        self.channel_activation = tf.keras.layers.Activation('sigmoid')
        self.channel_reshape = tf.keras.layers.Reshape((1, 1, channels))
        self.channel_multiply = tf.keras.layers.Multiply()

        # Spatial attention components
        self.spatial_concat = tf.keras.layers.Concatenate(axis=-1)
        self.spatial_conv = tf.keras.layers.Conv2D(
            1, kernel_size=7, padding='same', activation='sigmoid'
        )
        self.spatial_multiply = tf.keras.layers.Multiply()

        super(CBAMBlock, self).build(input_shape)

    def call(self, inputs):
        # Channel attention
        avg_pool = self.avg_pool(inputs)
        max_pool = self.max_pool(inputs)

        fc1_avg_pool = self.fc1_avg(avg_pool)
        fc1_max_pool = self.fc1_max(max_pool)

        avg_pool = self.fc2_avg(fc1_avg_pool)
        max_pool = self.fc2_max(fc1_max_pool)

        channel_attention = self.channel_add([avg_pool, max_pool])
        channel_attention = self.channel_activation(channel_attention)
        channel_attention = self.channel_reshape(channel_attention)

        x = self.channel_multiply([inputs, channel_attention])

        # Spatial attention
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = self.spatial_concat([avg_pool, max_pool])

        spatial_attention = self.spatial_conv(concat)

        return self.spatial_multiply([x, spatial_attention])

    def get_config(self):
        config = super(CBAMBlock, self).get_config()
        config.update({'ratio': self.ratio})
        return config


class TimeFilmLayer(tf.keras.layers.Layer):
    """Feature-wise Linear Modulation (FiLM) conditioned on time of year

    Args:
        num_channels: Number of channels in spatial features to condition
        hidden_dim: Dimension of hidden time embedding (default: 128)
        use_sinusoidal: Whether to use sin/cos encoding for cyclical time (default: True)
        max_period: Maximum period for time normalization, e.g., 365 for days (default: 365)
    """

    def __init__(self, num_channels, hidden_dim=128, use_sinusoidal=True,
                 max_period=365, **kwargs):
        super(TimeFilmLayer, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.use_sinusoidal = use_sinusoidal
        self.max_period = max_period

    def build(self, input_shape):
        # input_shape is a list: [spatial_features_shape, time_shape]

        # Determine input dimension for time processing
        if self.use_sinusoidal:
            time_input_dim = 2  # sin and cos
        else:
            time_input_dim = 1  # raw time value

        # Time embedding network
        self.time_dense1 = tf.keras.layers.Dense(
            self.hidden_dim,
            activation='relu',
            name='time_dense1'
        )
        self.time_dense2 = tf.keras.layers.Dense(
            self.hidden_dim,
            activation='relu',
            name='time_dense2'
        )

        # FiLM parameters (scale and shift)
        self.gamma_dense = tf.keras.layers.Dense(
            self.num_channels,
            name='film_gamma',
            kernel_initializer='zeros',  # Start with identity transform
            bias_initializer='ones'
        )
        self.beta_dense = tf.keras.layers.Dense(
            self.num_channels,
            name='film_beta',
            kernel_initializer='zeros'  # Start with no shift
        )

        super(TimeFilmLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        """
        Args:
            inputs: List of [spatial_features, time_of_year]
                spatial_features: [batch, height, width, channels]
                time_of_year: [batch, 1] - time value (e.g., day of year)

        Returns:
            conditioned_features: [batch, height, width, channels]
        """
        spatial_features, time_of_year = inputs
        #time_of_year = tf.expand_dims(time_of_year, axis =-1)
        # Encode time with sin/cos for cyclical patterns
        if self.use_sinusoidal:
            time_normalized = (time_of_year /self.max_period) * 2 * np.pi
            time_sin = tf.sin(time_normalized)
            time_cos = tf.cos(time_normalized)
            time_encoded = tf.concat([time_sin, time_cos], axis=-1)
        else:
            time_encoded = time_of_year

        # Process time through embedding network
        time_emb = self.time_dense1(time_encoded)
        time_emb = self.time_dense2(time_emb)

        # Generate scale (gamma) and shift (beta) parameters
        gamma = self.gamma_dense(time_emb)  # [batch, num_channels]
        beta = self.beta_dense(time_emb)  # [batch, num_channels]
        # Reshape for broadcasting: [batch, 1, 1, num_channels]
        gamma = tf.reshape(gamma, [-1, 1, 1, self.num_channels])
        beta = tf.reshape(beta, [-1, 1, 1, self.num_channels])
        # Apply FiLM conditioning: γ * x + β
        conditioned_features = gamma * spatial_features + beta

        return conditioned_features

    def get_config(self):
        config = super(TimeFilmLayer, self).get_config()
        config.update({
            'num_channels': self.num_channels,
            'hidden_dim': self.hidden_dim,
            'use_sinusoidal': self.use_sinusoidal,
            'max_period': self.max_period
        })
        return config


def res_block_initial(x, num_filters, kernel_size, strides, name, attn_type= "None"):
    """Residual Unet block layer for first layer
    In the residual unet the first residual block does not contain an
    initial batch normalization and activation so we create this separate
    block for it.
    Args:
        x: tensor, image or image activation
        num_filters: list, contains the number of filters for each subblock
        kernel_size: int, size of the convolutional kernel
        strides: list, contains the stride for each subblock convolution
        name: name of the layer
    Returns:
        x1: tensor, output from residual connection of x and x1
    """

    if len(num_filters) == 1:
        num_filters = [num_filters[0], num_filters[0]]
        x1 = tf.keras.layers.Conv2D(filters=num_filters[0],
                                    kernel_size=kernel_size,
                                    strides=strides[0],
                                    padding='same', kernel_initializer ='he_normal')(x)

    x1 = tf.keras.layers.LeakyReLU(0.01)(x1)

    x1 = tf.keras.layers.Conv2D(filters=num_filters[1],
                                kernel_size=kernel_size,
                                strides=strides[1],
                                padding='same', kernel_initializer ='he_normal')(x1)

    x = tf.keras.layers.Conv2D(filters=num_filters[-1],
                               kernel_size=1,
                               strides=1,
                               padding='same', kernel_initializer ='he_normal')(x)
    x1 = tf.keras.layers.Add()([x, x1])
    if attn_type == "channel":
        x1 = SEBlock(ratio =8)(x1)#cbam_block(x1, name=name + 'spatial_channel_attn')se_block(x1, name=name + '_channel_attn')
    if attn_type == "cbam":
        x1 = CBAMBlock(ratio =8)(x1)#cbam_block(x1, name=name + 'spatial_channel_attn')
    if attn_type == "self":
        x1 = SelfAttention2D()(x1)#(x1, name=name + 'self_attn')
    x1 = tf.keras.layers.LeakyReLU(0.01)(x1)
    return x1


class GroupNormalization(tf.keras.layers.Layer):
    """MODIFIED FROM
        https://github.com/keras-team/keras/blob/v3.3.3/keras/src/layers/normalization/group_normalization.py#L10-L219
    """

    def __init__(
        self,
        groups=32,
        axis=-1,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError(
                f"Axis {self.axis} of input tensor should have a defined "
                "dimension but the layer received an input with shape "
                f"{input_shape}."
            )

        if self.groups == -1:
            self.groups = dim

        if dim < self.groups:
            raise ValueError(
                f"Number of groups ({self.groups}) cannot be more than the "
                f"number of channels ({dim})."
            )

        if dim % self.groups != 0:
            raise ValueError(
                f"Number of groups ({self.groups}) must be a multiple "
                f"of the number of channels ({dim})."
            )

        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes={self.axis: dim}
        )

        if self.scale:
            self.gamma = self.add_weight(
                shape=(dim,),
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                shape=(dim,),
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None

        super().build(input_shape)

    def call(self, inputs):
        reshaped_inputs = self._reshape_into_groups(inputs)
        normalized_inputs = self._apply_normalization(
            reshaped_inputs, inputs.shape
        )
        return tf.reshape(normalized_inputs, tf.shape(inputs))

    def _reshape_into_groups(self, inputs):
        input_shape = tf.shape(inputs)
        group_shape = list(inputs.shape)
        group_shape[0] = -1
        for i, e in enumerate(group_shape[1:]):
            if e is None:
                group_shape[i + 1] = input_shape[i + 1]

        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        reshaped_inputs = tf.reshape(inputs, group_shape)
        return reshaped_inputs

    def _apply_normalization(self, reshaped_inputs, input_shape):
        group_reduction_axes = list(range(1, len(reshaped_inputs.shape)))

        axis = -2 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        broadcast_shape = self._create_broadcast_shape(input_shape)
        mean, variance = tf.nn.moments(
            reshaped_inputs, axes=group_reduction_axes, keepdims=True
        )

        # Compute the batch normalization.
        inv = tf.math.rsqrt(variance + self.epsilon)
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)
            gamma = tf.cast(gamma, reshaped_inputs.dtype)
            inv = inv * gamma

        res = -mean * inv
        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
            beta = tf.cast(beta, reshaped_inputs.dtype)
            res = res + beta

        normalized_inputs = reshaped_inputs * inv + res
        return normalized_inputs

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(self.axis, self.groups)
        return broadcast_shape

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(self.gamma_initializer),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}

class SinusoidalTimeEmbedding(tf.keras.layers.Layer):
    """Transform an integer/float timestep t -> (B, embed_dim) sinusoidal vector"""
    def __init__(self, embed_dim=64, **kw):
        super().__init__(**kw)
        self.embed_dim = embed_dim

    def call(self, t):
        half = self.embed_dim // 2
        freq = tf.exp(
            tf.range(half, dtype=tf.float32) * -(tf.math.log(10000.0) / half)
        )
        args = tf.expand_dims(tf.cast(t, tf.float32), -1) * freq
        emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
        return emb
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim
        })
        return config

class SigmaEmbedding(tf.keras.layers.Layer):
    """
    Map a scalar σ -> (B, embed_dim) vector.
    Uses log σ followed by two dense GELU layers (EDM style).
    """
    def __init__(self, embed_dim=256, **kw):
        super().__init__(**kw)
        self.up = tf.keras.Sequential([
            tf.keras.layers.Dense(embed_dim, activation="gelu"),
            tf.keras.layers.Dense(embed_dim, activation="gelu"),
        ])

    def call(self, sigma):                          # σ shape: (B,)
        log_sigma = tf.math.log(sigma)              # EDM: cnoise(σ) = ¼ log σ works too
        #return self.up(tf.expand_dims(log_sigma, -1))
        return self.up(log_sigma)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim
        })
        return config

class FiLMResidual(tf.keras.layers.Layer):
    """
    Conv-GN-GELU-(FiLM)-Conv-GN with residual connection.
    `temb` must be a (B, d) vector; two Dense layers map it to scale/shift.
    """
    def __init__(self, n_filters, **kw):
        super().__init__(**kw)
        self.n = n_filters
        self.conv1 = tf.keras.layers.Conv2D(n_filters, 3, padding="same")
        self.gn1 = GroupNormalization(groups=8) # batch norm
        self.conv2 = tf.keras.layers.Conv2D(n_filters, 3, padding="same")
        self.gn2 = GroupNormalization(groups=8)
        self.act = tf.keras.layers.Activation("gelu")

        self.skip_proj = None

        self.scale = tf.keras.layers.Dense(n_filters)
        self.shift = tf.keras.layers.Dense(n_filters)

    def build(self, input_shape):
        in_ch = input_shape[-1]
        if in_ch != self.n:
            self.skip_proj = tf.keras.layers.Conv2D(self.n, 1, padding="same", name="conv2d_skip_proj")

    def call(self, x, temb):
        h = self.conv1(x)
        #h = self.gn1(h)
        h = self.act(h)

        # FiLM mod
        s = self.scale(temb)[:, None, None, :] # (B,1,1,C)
        b = self.shift(temb)[:, None, None, :] # (B,1,1,C)
        h = h * (1.0 + s) + b

        h = self.conv2(h)
        #h = self.gn2(h)
        h = self.act(h)

        if self.skip_proj is not None:
            x = self.skip_proj(x)

        return x + h
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_filters": self.n
        })
        return config

def downsample(n_filters):
    return tf.keras.layers.Conv2D(n_filters, 4, strides=2, padding="same")


def upsample(x, target_size):
    """"Upsampling function, upsamples the feature map
    Deep Residual Unet paper does not describe the upsampling function
    in detail. Original Unet uses a transpose convolution that downsamples
    the number of feature maps. In order to restrict the number of
    parameters here we use a bilinear resampling layer. This results in
    the concatentation layer concatenting feature maps with n and n/2
    features as opposed to n/2  and n/2 in the original unet.
    Args:
        x: tensor, feature map
        target_size: size to resize feature map to
    Returns:
        x_resized: tensor, upsampled feature map
    """

    x_resized = BicubicUpSampling2D((target_size, target_size))(x)  # tf.keras.layers.Lambda(lambda x: tf.image.resize(x, target_size))(x)
    return x_resized
# def upsample(n_filters, target_size = None):
#     return tf.keras.layers.Conv2DTranspose(n_filters, 4, strides=2, padding="same")

class BicubicUpSampling2D_dm(tf.keras.layers.Layer):
    def __init__(self, size, **kwargs):
        super(BicubicUpSampling2D_dm, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs):
        return tf.image.resize(inputs, [int(inputs.shape[1] * self.size[0]), int(inputs.shape[2] * self.size[1])],
                               method=tf.image.ResizeMethod.BILINEAR)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'size': self.size
        })
        return config

def upsample_v2(x, target_size):
    x_resized = BicubicUpSampling2D_dm((target_size, target_size))(x)
    return x_resized



def down_block(x, filters, kernel_size, i =1, use_pool=True, method ='unet', attn_type = "self"):

    x = res_block_initial(x, [filters], kernel_size, strides=[1, 1],
                          name='decoder_layer_v2' + str(i),
                              attn_type = attn_type)
    return tf.keras.layers.AveragePooling2D((2, 2))(x), x


def up_block(x, y, filters, kernel_size, i =1, method ='unet', concat = True, attn_type = "self"):
    x = upsample(x, 2)
    if concat:
        x = tf.keras.layers.Concatenate(axis=-1)([x, y])
    x = res_block_initial(x, [filters], kernel_size, strides=[1, 1],
                          name='encoder_layer_v2' + str(i),attn_type = attn_type)
    return x



def get_custom_dm_objects():
    return {
        'GroupNormalization': GroupNormalization,
        'SinusoidalTimeEmbedding': SinusoidalTimeEmbedding,
        'SigmaEmbedding': SigmaEmbedding,
        'FiLMResidual': FiLMResidual,
        'BicubicUpSampling2D_dm': BicubicUpSampling2D_dm
    }


def build_diffusion_unet(input_size, resize_output, num_filters, num_channels,
        time_embed_dim=256, orog_predictor = True
    ) -> tf.keras.Model:

    # input
    inp_mean_hr = tf.keras.Input(shape=[resize_output[0], resize_output[1], 1], name="y_mean_hr")
    if orog_predictor:
        inp_static_hr = tf.keras.Input(shape=[resize_output[0], resize_output[1], 1], name="static_hr")
    inp_lr = tf.keras.Input(shape=[input_size[0], input_size[1], num_channels], name="x_lr")
    inp_res_hr_noise  = tf.keras.Input(shape=[resize_output[0], resize_output[1], 1], name="y_res_hr_noise")
    inp_t = tf.keras.Input((), dtype=tf.int32, name="timestep")
    time_of_year = tf.keras.layers.Input(shape=[1], name='time_input')
    conditioned_large_scale_fields = TimeFilmLayer(
        num_channels=num_channels,
        hidden_dim=32,
        use_sinusoidal=True,
        max_period=365,
        name='time_film_a'
    )([inp_lr, time_of_year])

    conditioned_unet_fields = TimeFilmLayer(
        num_channels=num_channels,
        hidden_dim=32,
        use_sinusoidal=True,
        max_period=365,
        name='time_film_b'
    )([inp_mean_hr, time_of_year])
    if orog_predictor:
        x0 = tf.concat([inp_res_hr_noise, inp_static_hr, conditioned_unet_fields], axis=-1)
    else:
        x0 = tf.concat([inp_res_hr_noise, conditioned_unet_fields], axis=-1)
    # timestep embedding
    temb = SinusoidalTimeEmbedding(embed_dim=32)(inp_t)
    temb = tf.keras.layers.Dense(time_embed_dim, activation="gelu")(temb)
    # encoder
    x = tf.keras.layers.Conv2D(num_filters[0], 7, padding="same")(x0)
    x = FiLMResidual(32)(x, temb)
    x, temp1 = down_block(x, 32, kernel_size=5, i=4, use_pool=False,
                          attn_type ="channel") # self
    x = FiLMResidual(64)(x, temb)
    x, temp2 = down_block(x, 64, kernel_size=5, i=1, attn_type="None")
    x = FiLMResidual(128)(x, temb)
    x, temp3 = down_block(x, 128, kernel_size=3, i =2, attn_type ="None") # 16, 16

    inputs_abstract = conditioned_large_scale_fields
    x1 = res_block_initial(inputs_abstract, [32], 5, [1, 1], f"test123", attn_type ="channel") # cbam
    x1 = res_block_initial(x1, [64], 7, [1, 1], f"test11234", attn_type ="None") # cbam#down_block(x1, num_filters[2]*2, kernel_size=3, i=5, use_pool=False, attn_type ="cbam")
    x1 = res_block_initial(x1, [64], 5, [1, 1], f"test1", attn_type ="None")
    x1 = res_block_initial(x1, [128], 3, [1, 1], f"test2", attn_type ="None")

    # Allow Mixing of the Low and High Resolution fields
    concat_scales = tf.keras.layers.Concatenate(-1)([x1, x])
    x = res_block_initial(concat_scales, [128], 3, [1, 1], f"test3", attn_type ="self") # cbam
    # bottleneck
    x = FiLMResidual(128)(x, temb)
    # conditioned_latent = TimeFilmLayer(
    #     num_channels=128,
    #     hidden_dim=32,
    #     use_sinusoidal=True,
    #     max_period=365,
    #     name='time_film_c'
    # )([x, time_of_year])
    # x = conditioned_latent

    x = up_block(x, temp3, kernel_size=3, filters=128, i=0, concat=True, attn_type="None")
    x = FiLMResidual(128)(x, temb)# channel
    x = up_block(x, temp2, kernel_size=3, filters = 64, i =2, concat = True, attn_type ="None")
    x = FiLMResidual(64)(x, temb)# chan
    x = up_block(x, temp1, kernel_size=3, filters = 32, i =3, concat = True, attn_type ="channel") # self
    x = FiLMResidual(32)(x, temb)  # chan

    output = x
    output = res_block_initial(output, [32], 5, [1, 1], "output_convbbb12347", attn_type="None")
    out = tf.keras.layers.Conv2D(1, 5, padding="same", name="y_residual", activation ='linear')(output)
    if orog_predictor:
        input_layers = [inp_res_hr_noise, inp_t, inp_lr, inp_static_hr, inp_mean_hr, time_of_year]
    else:
        input_layers = [inp_res_hr_noise, inp_t, inp_lr, inp_mean_hr, time_of_year]

    return tf.keras.Model(
        inputs=input_layers,
        outputs=out,
        name="diffusion_residual_unet",
    )
def unet(input_size, resize_output, num_filters, num_channels, num_classes,
                          final_activation = 'linear', orog_predictor = True):

    """

    :param input_size:
    :param resize_output: high-resolution target fields, for CORDEX-ML-Bench this is 128x128.
    :param num_filters: a list consisting of number of filters in individual layrs [32, 64, 128] ...
    :param kernel_size: not used
    :param num_channels:
    :param num_classes:
    :param resize:
    :param final_activation:
    :return:
    """
    if orog_predictor:
        input_topography_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])

    large_scale_fields = tf.keras.layers.Input(
        shape=[input_size[0], input_size[1], num_channels],
        name='spatial_input'
    )
    # temperature conditioning on the outputs
    large_scale_slice = tf.keras.layers.GlobalAveragePooling2D(
        name="spatial_mean"
    )(large_scale_fields)
    time_of_year = tf.keras.layers.Input(shape=[1], name='time_input')
    conditioned_large_scale_fields = TimeFilmLayer(
        num_channels=num_channels,
        hidden_dim=128,
        use_sinusoidal=True,
        max_period=365,
        name='time_film_test'
    )([large_scale_fields, time_of_year])

    # High-resolution inputs
    if orog_predictor:
        x, temp1 = down_block(input_topography_fields, num_filters[0], kernel_size=3, i =0, attn_type ="None") # 64, 64
        x, temp2 = down_block(x, num_filters[1], kernel_size=3, i =1, attn_type ="None") # 32, 32
        x, temp3 = down_block(x, num_filters[2], kernel_size=3, i =2, attn_type ="None") # 16, 16

    # Film conditioned low resolution inputs (16 x 16 inputs)
    x1 = res_block_initial(conditioned_large_scale_fields, [num_filters[2]], 5, [1, 1], f"test2123434", attn_type ="channel")
    x1 = res_block_initial(x1, [num_filters[2]*2], 3, [1, 1], f"test21233434", attn_type ="channel") # Channel, and the above
    x1 = res_block_initial(x1, [64], 3, [1, 1], f"test1", attn_type ="None")
    x1 = res_block_initial(x1, [128], 5, [1, 1], f"test2", attn_type ="None")

    # Merge the inputs together
    if orog_predictor:
        concat_scales = tf.keras.layers.Concatenate(-1)([x1, x])
    else:
        concat_scales = x1

    x = res_block_initial(concat_scales, [256], 3, [1, 1], f"test3", attn_type ="self") # cbam
    # decode
    if orog_predictor:
        x = up_block(x, temp3, kernel_size=3, filters = num_filters[2], i =0, concat = True, attn_type ="cbam") # channel, channel, and self
        x = up_block(x, temp2, kernel_size=3, filters = num_filters[1], i =2, concat = True, attn_type ="channel")
        x = up_block(x, temp1, kernel_size=5, filters = num_filters[0], i =3, concat = True, attn_type ="None")
    else:
        x = up_block(x, [], kernel_size=3, filters = num_filters[2], i =0, concat = False, attn_type ="cbam") # channel, channel, self
        x = up_block(x, [], kernel_size=3, filters = num_filters[1], i =2, concat = False, attn_type ="channel")
        x = up_block(x, [], kernel_size=5, filters = num_filters[0], i =3, concat = False, attn_type ="None")
    # conditioned_large_scale_fields_v2 = conditioned_large_scale_fields[:,-5]
    # conditioned_large_scale_fields = TimeFilmLayer(
    #     num_channels=num_filters[0],
    #     hidden_dim=32,
    #     use_sinusoidal=True,
    #     max_period=365,
    #     name='time_film_test'
    # )([x, conditioned_large_scale_fields_v2])

    # Final output convolutions
    output = x#conditioned_large_scale_fields
    output = res_block_initial(output, [64], 5, [1, 1], "output_convbbb12347", attn_type ="channel")
    output = res_block_initial(output, [32], 5, [1, 1], "output_convbbb12347", attn_type="channel")
    output = tf.keras.layers.Conv2D(num_classes, 1, activation=final_activation, padding ='same')(output)
    if orog_predictor:
        input_layers = [large_scale_fields, input_topography_fields, time_of_year]
    else:
        input_layers = [large_scale_fields, time_of_year]
    model = tf.keras.models.Model(input_layers, output, name='unet')
    model.summary()
    return model






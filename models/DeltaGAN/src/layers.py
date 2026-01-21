
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow
import xarray as xr
from dask.diagnostics import ProgressBar
from tensorflow.keras.callbacks import Callback
import numpy as np
import pandas as pd
import tensorflow as tf


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

# def se_block(x, ratio=16, name='se'):
#     """Squeeze-and-Excitation block"""
#     channels = x.shape[-1]
#
#     # Squeeze: global spatial information
#     se = tf.keras.layers.GlobalAveragePooling2D(name=name + '_gap')(x)
#
#     # Excitation: learn channel interdependencies
#     se = tf.keras.layers.Dense(channels // ratio, activation='relu',
#                                name=name + '_fc1')(se)
#     se = tf.keras.layers.Dense(channels, activation='sigmoid',
#                                name=name + '_fc2')(se)
#
#     # Reshape and scale
#     se = tf.keras.layers.Reshape((1, 1, channels))(se)
#     se = tf.keras.layers.Multiply(name=name + '_scale')([x, se])
#     return se
#
#
# def self_attention_2d(x, name='attn'):
#     """Self-attention for 2D feature maps"""
#     channels = x.shape[-1]
#
#     # Query, Key, Value projections (reduce channels for efficiency)
#     query = tf.keras.layers.Conv2D(channels // 8, 1, name=name + '_query')(x)
#     key = tf.keras.layers.Conv2D(channels // 8, 1, name=name + '_key')(x)
#     value = tf.keras.layers.Conv2D(channels, 1, name=name + '_value')(x)
#
#     # Get spatial dimensions
#     batch_size = tf.shape(x)[0]
#     height = tf.shape(x)[1]
#     width = tf.shape(x)[2]
#
#     # Reshape: [B, H, W, C] -> [B, H*W, C]
#     query = tf.reshape(query, [batch_size, height * width, channels // 8])
#     key = tf.reshape(key, [batch_size, height * width, channels // 8])
#     value = tf.reshape(value, [batch_size, height * width, channels])
#
#     # Attention scores: [B, H*W, H*W]
#     attention = tf.matmul(query, key, transpose_b=True)
#     attention = tf.nn.softmax(attention / tf.sqrt(tf.cast(channels // 8, tf.float32)))
#
#     # Apply attention to values: [B, H*W, C]
#     out = tf.matmul(attention, value)
#
#     # Reshape back: [B, H*W, C] -> [B, H, W, C]
#     out = tf.reshape(out, [batch_size, height, width, channels])
#
#     # Project back to original channels
#     out = tf.keras.layers.Conv2D(channels, 1, name=name + '_proj')(out)
#
#     # Residual connection (learnable scale, starts at 0)
#     gamma = tf.Variable(0., trainable=True, name=name + '_gamma')
#
#     return x + gamma * out
#
#
# def cbam_block(x, ratio=8, name='cbam'):
#     """CBAM: channel + spatial attention"""
#     channels = x.shape[-1]
#
#     # Channel attention
#     avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
#     max_pool = tf.keras.layers.GlobalMaxPooling2D()(x)
#
#     avg_pool = tf.keras.layers.Dense(channels // ratio, activation='relu')(avg_pool)
#     avg_pool = tf.keras.layers.Dense(channels)(avg_pool)
#
#     max_pool = tf.keras.layers.Dense(channels // ratio, activation='relu')(max_pool)
#     max_pool = tf.keras.layers.Dense(channels)(max_pool)
#
#     channel_attention = tf.keras.layers.Add()([avg_pool, max_pool])
#     channel_attention = tf.keras.layers.Activation('sigmoid')(channel_attention)
#     channel_attention = tf.keras.layers.Reshape((1, 1, channels))(channel_attention)
#
#     x = tf.keras.layers.Multiply()([x, channel_attention])
#
#     # Spatial attention
#     avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
#     max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
#     concat = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
#
#     spatial_attention = tf.keras.layers.Conv2D(1, kernel_size=7,
#                                                padding='same',
#                                                activation='sigmoid',
#                                                name=name + '_spatial')(concat)
#
#     return tf.keras.layers.Multiply(name=name + '_out')([x, spatial_attention])

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



def res_block_initial(x, num_filters, kernel_size, strides, name, attn_type= "self"):
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


def conv_block(x, filters, activation, kernel_size=(7, 7), strides=(2, 2), padding="same",
               use_bias=True, use_bn=True, use_dropout=True, drop_value=0.5):

    x = layers.Conv2D(filters, kernel_size, strides=strides,
                      padding='same', use_bias=use_bias, kernel_initializer ='he_normal')(x)
    x = tf.keras.layers.LeakyReLU(0.01)(x)

    return x


def decoder_noise(x, num_filters, kernel_size):
    """Unet decoder
    Args:
        x: tensor, output from previous layer
        encoder_output: list, output from all previous encoder layers
        num_filters: list, number of filters for each decoder layer
        kernel_size: int, size of the convolutional kernel
    Returns:
        x: tensor, output from last layer of decoder
    """
    noise_inputs = []# at some intermediate layers
    for i in range(1, len(num_filters) + 1):
        layer2 = 'decoder_layer_v2' + str(i)
        x = upsample(x, 2)
        x = res_block_initial(x, [num_filters[-i]], kernel_size, strides=[1, 1], name='decoder_layer_v2' + str(i),
                              attn_type ="cbam")
    return x, noise_inputs


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



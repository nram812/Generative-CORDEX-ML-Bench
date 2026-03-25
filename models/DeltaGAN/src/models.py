import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow
import xarray as xr
from dask.diagnostics import ProgressBar
from tensorflow.keras.callbacks import Callback
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
from src.layers import res_block_initial, \
    BicubicUpSampling2D,upsample, conv_block,decoder_noise,down_block,up_block, TimeFilmLayer,SEBlock, SelfAttention2D, CBAMBlock

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import math



def critic(high_resolution_fields_size,
                            low_resolution_fields_size, use_bn=False,
                            use_dropout=False, use_bias=True,
                            high_resolution_feature_channels=(16, 32, 64, 128),
           conditioning = False):
    """
    Discriminator no longer uses the unet model to demonstrate realism
    **Purpose:**
      * To create a discriminator model that takes two streams of inputs, one from the low resolution predictor fields(X)
      and auxilary inputs (topography), it also takes in the high-resolution "regression prediction",
      which is used for residuals

    **Parameters:**
      * **high_resolution_fields_size (tuple):**  The size of the 2D high-resolution RCM fields, over the NZ region this (172, 179)
      * **low_resolution_fields_size (tuple):**  The size of the 2D low-resolution predictor fields (23, 26) over the New Zealand domain
      * **use_bn (bool, optional):** whether to use batchnormalization or not (default no bn)
      * **use_dropout (bool, optional):** whether to use dropout or not(default no dropout)
      * **use_bias (bool, optional):** whether to use bias or not (default bias =True)

    **Returns:**
        * a tf.keras.models.Model class

    **Example Usage:**
    ```python
    discriminator_model = get_discriminator_model((172, 179), (23, 26))
    ```
    """
    IMG_SHAPE = high_resolution_fields_size
    IMG_SHAPE2 = low_resolution_fields_size

    high_res_fields = layers.Input(shape=IMG_SHAPE) # real or fake predictions
    low_res_inputs = layers.Input(shape=IMG_SHAPE2)
    inputs_high_res = res_block_initial(high_res_fields, [high_resolution_feature_channels[0]], 5, [1, 1], f"test1234", attn_type ="None") #self
    inputs_high_res = res_block_initial(inputs_high_res, [high_resolution_feature_channels[0]], 5, [1, 1], "input_blocka", attn_type ="None") # cbam
    x = conv_block(inputs_high_res, high_resolution_feature_channels[1], kernel_size=(5, 5), strides=(2, 2),
                   use_bn=use_bn, use_bias=use_bias,
                   use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU()) # 64, 64
    x = res_block_initial(x, [high_resolution_feature_channels[0]], 7, [1, 1], "input_blockb", attn_type ="None") # cbam
    x = res_block_initial(x, [high_resolution_feature_channels[1]], 7, [1, 1], "input_blockc", attn_type ="None")
    x = conv_block(x, high_resolution_feature_channels[2], kernel_size=(5, 5), strides=(2, 2),
                   use_bn=use_bn, use_bias=use_bias,
                   use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU()) # 32, 32
    x = res_block_initial(x, [high_resolution_feature_channels[1]], 5, [1, 1], "input_blockd", attn_type ="None")
    x = res_block_initial(x, [high_resolution_feature_channels[2]], 5, [1, 1], "input_blocke", attn_type ="None")
    x_init_raw = conv_block(x, high_resolution_feature_channels[3], kernel_size=(5, 5), strides=(2, 2),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU()) # 16, 16

    # Weak conditioning on the large-scale fields
    # creating a low dimensional space for the large scale fields
    if conditioning:
        x_large_scale = res_block_initial(low_res_inputs, [2], 3, [1, 1], "input_blocka", attn_type ="None") # channel
        x_large_scale_flatten = tf.keras.layers.Flatten()(x_large_scale)
        dense_input = tf.keras.layers.Dense(1,activation ='linear')(x_large_scale_flatten)

        conditioned_inputs = TimeFilmLayer(
            num_channels=high_resolution_feature_channels[3],
            hidden_dim=128,
            use_sinusoidal=False,
            max_period=365,
            name='time_film'
        )([x_init_raw, dense_input])
    else:
        conditioned_inputs = x_init_raw
    concat_outputs = res_block_initial(conditioned_inputs, [high_resolution_feature_channels[3]], 5, [1, 1], "output_convbbb2", attn_type ="None") #self
    x_init_raw = conv_block(concat_outputs, high_resolution_feature_channels[3], kernel_size=(5, 5), strides=(2, 2),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0,
                            activation=tf.keras.layers.LeakyReLU(0.1))
    x_init_raw = conv_block(x_init_raw, 32, kernel_size=(5, 5), strides=(2, 2),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0,
                            activation=tf.keras.layers.LeakyReLU(0.1))
    flattened_output = tf.keras.layers.Flatten()(x_init_raw)
    dense2 = tf.keras.layers.Dense(256)(flattened_output)
    dense3 = tf.keras.layers.Dense(64)(dense2)

    x = layers.Dense(1)(dense3)

    d_model = keras.models.Model([high_res_fields, low_res_inputs], x,
                                 name="discriminator")
    return d_model

"""The below configuration works with tasmax"""


def res_gan(input_size, resize_output, num_filters, num_channels, num_classes,
                          final_activation = tf.keras.layers.LeakyReLU(1), orog_predictor = True, temp_conditioning = False):
    """new model has two layers of noise, and does not feed in the unet prediction"""

    unet_prediction_layer = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    if orog_predictor:
        input_topography_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    large_scale_fields = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])
    noise = tf.keras.layers.Input(shape=[input_size[0], input_size[1], num_channels])

    # Adding Embedding Layers at the start of the network.
    time_of_year = tf.keras.layers.Input(shape=[1], name='time_input')

    conditioned_large_scale_fields = TimeFilmLayer(
        num_channels=num_channels,
        hidden_dim=128,
        use_sinusoidal=True,
        max_period=365,
        name='time_film_a'
    )([large_scale_fields, time_of_year])

    conditioned_unet_fields = TimeFilmLayer(
        num_channels=1,
        hidden_dim=32,
        use_sinusoidal=True,
        max_period=365,
        name='time_film_b'
    )([unet_prediction_layer, time_of_year])


    # High-resolution fields
    if orog_predictor:
        concat_image_conditioned = tf.keras.layers.Concatenate(-1)([conditioned_unet_fields,  input_topography_fields])
    else:
        concat_image_conditioned = conditioned_unet_fields
    print(concat_image_conditioned, conditioned_large_scale_fields, unet_prediction_layer)
    x, temp1 = down_block(concat_image_conditioned, num_filters[2], kernel_size=5, i=4, use_pool=False,
                          attn_type ="channel") # self
    x, temp2 = down_block(x, num_filters[1], kernel_size=3, i =1, attn_type ="None") # 32, 32, channel
    x, temp3 = down_block(x, num_filters[2], kernel_size=3, i =2, attn_type ="None") # 16, 16

    # Low resolution fields
    inputs_abstract = tf.keras.layers.Concatenate(-1)([conditioned_large_scale_fields, noise])
    x1 = res_block_initial(inputs_abstract, [num_filters[2]*2], 3, [1, 1], f"test123", attn_type ="None") # cbam
    x1 = res_block_initial(x1, [ num_filters[2]*2], 3, [1, 1], f"test11234", attn_type ="None") # cbam#down_block(x1, num_filters[2]*2, kernel_size=3, i=5, use_pool=False, attn_type ="cbam")
    x1 = res_block_initial(x1, [64], 3, [1, 1], f"test1", attn_type ="None")
    x1 = res_block_initial(x1, [128], 5, [1, 1], f"test2", attn_type ="None")

    # Allow Mixing of the Low and High Resolution fields
    concat_scales = tf.keras.layers.Concatenate(-1)([x1, x])
    x = res_block_initial(concat_scales, [256], 5, [1, 1], f"test3", attn_type ="self") # cbam

    # decode
    x = up_block(x, temp3, kernel_size=3, filters = num_filters[2], i =0, concat = True, attn_type ="None") # channel
    noise2 = tf.keras.layers.Input(shape=[x.shape[1], x.shape[2], int(num_channels//2)])
    x = tf.keras.layers.Concatenate(-1)([noise2, x])
    x = up_block(x, temp2, kernel_size=3, filters = num_filters[1], i =2, concat = True, attn_type ="None")
    x = up_block(x, temp1, kernel_size=5, filters = num_filters[0], i =3, concat = True, attn_type ="None") # self

    # Final output convolutions

    output = x

    output = res_block_initial(output, [32], 5, [1, 1], "output_convbbb12347", attn_type ="None")
    output = tf.keras.layers.Conv2D(
        16,
        5,
        activation=final_activation,
        padding='same',
        kernel_initializer='glorot_uniform'
    )(output)
    output = tf.keras.layers.Conv2D(
        num_classes,
        1,
        activation=final_activation,
        padding='same',
        kernel_initializer='glorot_uniform'
    )(output)
    if orog_predictor:
        input_layers = [noise, noise2, unet_prediction_layer] + [large_scale_fields, input_topography_fields, time_of_year]
    else:
        input_layers = [noise, noise2, unet_prediction_layer] + [large_scale_fields, time_of_year]
    model = tf.keras.models.Model(input_layers, output, name='gan')
    model.summary()
    return model


def unet(input_size, resize_output, num_filters, num_channels, num_classes,
                          final_activation = 'linear', orog_predictor = True, temp_conditioning=False):

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
    time_of_year = tf.keras.layers.Input(shape=[1], name='time_input')
    # if temp_conditioning:
    #     conditioned_large_scale_fields = TimeFilmLayer(
    #         num_channels=num_channels,
    #         hidden_dim=128,
    #         use_sinusoidal=False,
    #         max_period=365,
    #         name='time_film_testas'
    #     )([large_scale_fields, time_of_year])
    #
    # else:

    conditioned_large_scale_fields = large_scale_fields
    #TimeFilmLayer(
    #         num_channels=num_channels,
    #         hidden_dim=128,
    #         use_sinusoidal=True,
    #         max_period=365,
    #         name='time_film_testas12'
    #     )([large_scale_fields, time_of_year])

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
        x = up_block(x, temp2, kernel_size=3, filters = num_filters[1], i =2, concat = True, attn_type ="None")
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
    output = res_block_initial(output, [64], 5, [1, 1], "output_convbbb12347", attn_type ="None")
    output = res_block_initial(output, [32], 5, [1, 1], "output_convbbb12347", attn_type="None")
    # if temp_conditioning:
    #     output = TimeFilmLayer(
    #         num_channels=32,
    #         hidden_dim=128,
    #         use_sinusoidal=False,
    #         max_period=365,
    #         name='time_film_testcdf'
    #     )([output, time_of_year])
    # else:
    #     output = TimeFilmLayer(
    #         num_channels=32,
    #         hidden_dim=128,
    #         use_sinusoidal=True,
    #         max_period=365,
    #         name='time_film_testcdf1'
    #     )([output, time_of_year])
    output = tf.keras.layers.Conv2D(32, 5, activation=final_activation, padding='same')(output)
    output = tf.keras.layers.Conv2D(16, 5, activation=final_activation, padding='same')(output)
    output = tf.keras.layers.Conv2D(num_classes, 1, activation=final_activation, padding ='same')(output)
    if orog_predictor:
        input_layers = [large_scale_fields, input_topography_fields, time_of_year]
    else:
        input_layers = [large_scale_fields, time_of_year]
    model = tf.keras.models.Model(input_layers, output, name='unet')
    model.summary()
    return model


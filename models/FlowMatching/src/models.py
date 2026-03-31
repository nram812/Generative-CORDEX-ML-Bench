# import tensorflow as tf
# import tensorflow.keras as keras
# import tensorflow.keras.layers as layers
# import tensorflow
# import xarray as xr
# from dask.diagnostics import ProgressBar
# from tensorflow.keras.callbacks import Callback
# import numpy as np
# import pandas as pd
# import sys
# import os
# sys.path.append(os.getcwd())
# from src.layers import res_block_initial, \
#     BicubicUpSampling2D,upsample, conv_block,down_block,up_block, TimeFilmLayer,SEBlock, SelfAttention2D, CBAMBlock
#
# import tensorflow as tf
# from tensorflow.keras.optimizers.schedules import LearningRateSchedule
# import math
#
#
# def build_flow_unet(
#         input_size,
#         resize_output,
#         num_filters,
#         num_channels,
#         num_classes=1,
#         time_embed_dim=256,
#         final_activation='linear',
#         orog_predictor=True,
#         temp_conditioning=False,
#         varname="tasmax"
# ):
#     """
#     Velocity-predicting UNet for single-stage flow matching.
#
#     Rewritten to be perfectly consistent with `res_gan` and `unet` arguments
#     and block definitions. It uses `TimeFilmLayer` for both seasonal conditioning
#     and flow-matching timestep embeddings (replacing separate FiLM/Sinusoidal blocks).
#     """
#
#     # ------------------------------------------------------------------ #
#     # Inputs
#     # ------------------------------------------------------------------ #
#     inp_x_t = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1], name="x_t")
#     large_scale_fields = tf.keras.layers.Input(shape=[input_size[0], input_size[1], num_channels], name='spatial_input')
#
#     # Timestep for flow matching (Continuous or Integer)
#     inp_t = tf.keras.layers.Input(shape=[1], name="timestep")
#     # Time of year for seasonal conditioning
#     time_of_year = tf.keras.layers.Input(shape=[1], name='time_input')
#
#     if orog_predictor:
#         input_topography_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1], name="static_hr")
#
#     # ------------------------------------------------------------------ #
#     # Seasonal FiLM conditioning on large-scale fields
#     # ------------------------------------------------------------------ #
#     conditioned_large_scale_fields = TimeFilmLayer(
#         num_channels=num_channels,
#         hidden_dim=128,
#         use_sinusoidal=True,
#         max_period=365,
#         name='time_film_lr',
#         varname=varname
#     )([large_scale_fields, time_of_year])
#
#     # ------------------------------------------------------------------ #
#     # High-Resolution Branch (Encoder)
#     # ------------------------------------------------------------------ #
#     if orog_predictor:
#         x0 = tf.keras.layers.Concatenate(axis=-1)([inp_x_t, input_topography_fields])
#     else:
#         x0 = inp_x_t
#
#     # Initial Step Conditioning
#     x = TimeFilmLayer(num_channels=x0.shape[-1], hidden_dim=time_embed_dim, use_sinusoidal=True, max_period=1000,
#                       name="time_film_t_in", varname=varname)([x0, inp_t])
#
#     x, temp1 = down_block(x, num_filters[0], kernel_size=5, i=4, use_pool=False, attn_type="channel", varname=varname)
#     x = TimeFilmLayer(num_channels=num_filters[0], hidden_dim=time_embed_dim, use_sinusoidal=True, max_period=1000,
#                       name="time_film_t_d1", varname=varname)([x, inp_t])
#
#     x, temp2 = down_block(x, num_filters[1], kernel_size=3, i=1, attn_type="cbam", varname=varname)
#     x = TimeFilmLayer(num_channels=num_filters[1], hidden_dim=time_embed_dim, use_sinusoidal=True, max_period=1000,
#                       name="time_film_t_d2", varname=varname)([x, inp_t])
#
#     x, temp3 = down_block(x, num_filters[2], kernel_size=3, i=2, attn_type="None", varname=varname)
#
#     # ------------------------------------------------------------------ #
#     # Low-Resolution Branch (Feature Extraction)
#     # ------------------------------------------------------------------ #
#     x1 = res_block_initial(conditioned_large_scale_fields, [num_filters[2]], 5, [1, 1], "ls_block1",
#                            attn_type="channel", varname=varname)
#     x1 = res_block_initial(x1, [num_filters[2] * 2], 3, [1, 1], "ls_block2", attn_type="channel", varname=varname)
#     x1 = res_block_initial(x1, [64], 3, [1, 1], "ls_block3", attn_type="None", varname=varname)
#     x1 = res_block_initial(x1, [128], 3, [1, 1], "ls_block4", attn_type="None", varname=varname)
#
#     # ------------------------------------------------------------------ #
#     # Merge HR and LR branches & Bottleneck
#     # ------------------------------------------------------------------ #
#     concat_scales = tf.keras.layers.Concatenate(axis=-1)([x1, x])
#     x = res_block_initial(concat_scales, [256], 3, [1, 1], "merge_block", attn_type="self", varname=varname)
#
#     # Bottleneck Flow Timestep Conditioning
#     x = TimeFilmLayer(
#         num_channels=256,
#         hidden_dim=time_embed_dim,
#         use_sinusoidal=True,
#         max_period=1000,
#         name="time_film_t_bottleneck",
#         varname=varname
#     )([x, inp_t])
#
#     # ------------------------------------------------------------------ #
#     # Decoder (Conditioned on Flow Timestep at every scale)
#     # ------------------------------------------------------------------ #
#     x = up_block(x, temp3, kernel_size=3, filters=num_filters[2], i=0, concat=True, attn_type="cbam", varname=varname)
#     x = TimeFilmLayer(num_channels=num_filters[2], hidden_dim=time_embed_dim, use_sinusoidal=True, max_period=1000,
#                       name="time_film_t_u1", varname=varname)([x, inp_t])
#
#     x = up_block(x, temp2, kernel_size=3, filters=num_filters[1], i=2, concat=True, attn_type="channel",
#                  varname=varname)
#     x = TimeFilmLayer(num_channels=num_filters[1], hidden_dim=time_embed_dim, use_sinusoidal=True, max_period=1000,
#                       name="time_film_t_u2", varname=varname)([x, inp_t])
#
#     x = up_block(x, temp1, kernel_size=5, filters=num_filters[0], i=3, concat=True, attn_type="None", varname=varname)
#     x = TimeFilmLayer(num_channels=num_filters[0], hidden_dim=time_embed_dim, use_sinusoidal=True, max_period=1000,
#                       name="time_film_t_u3", varname=varname)([x, inp_t])
#
#     # ------------------------------------------------------------------ #
#     # Output Head — Predicted Velocity Field
#     # ------------------------------------------------------------------ #
#     output = res_block_initial(x, [32], 5, [1, 1], "out_block", attn_type="channel", varname=varname)
#     output = tf.keras.layers.Conv2D(16, 5, activation=final_activation, padding="same")(output)
#     out = tf.keras.layers.Conv2D(num_classes, 1, activation=final_activation, padding="same", name="velocity")(output)
#
#     # ------------------------------------------------------------------ #
#     # Assemble Model
#     # ------------------------------------------------------------------ #
#     if orog_predictor:
#         input_layers = [inp_x_t, inp_t, large_scale_fields, input_topography_fields, time_of_year]
#     else:
#         input_layers = [inp_x_t, inp_t, large_scale_fields, time_of_year]
#
#     model = tf.keras.models.Model(
#         inputs=input_layers,
#         outputs=out,
#         name="single_stage_flow_unet",
#     )
#
#     model.summary()
#     return model
#
#
# def critic(high_resolution_fields_size,
#                             low_resolution_fields_size, use_bn=False,
#                             use_dropout=False, use_bias=True,
#                             high_resolution_feature_channels=(16, 32, 64, 128), conditioning = False):
#     """
#     Discriminator no longer uses the unet model to demonstrate realism
#     **Purpose:**
#       * To create a discriminator model that takes two streams of inputs, one from the low resolution predictor fields(X)
#       and auxilary inputs (topography), it also takes in the high-resolution "regression prediction",
#       which is used for residuals
#
#     **Parameters:**
#       * **high_resolution_fields_size (tuple):**  The size of the 2D high-resolution RCM fields, over the NZ region this (172, 179)
#       * **low_resolution_fields_size (tuple):**  The size of the 2D low-resolution predictor fields (23, 26) over the New Zealand domain
#       * **use_bn (bool, optional):** whether to use batchnormalization or not (default no bn)
#       * **use_dropout (bool, optional):** whether to use dropout or not(default no dropout)
#       * **use_bias (bool, optional):** whether to use bias or not (default bias =True)
#
#     **Returns:**
#         * a tf.keras.models.Model class
#
#     **Example Usage:**
#     ```python
#     discriminator_model = get_discriminator_model((172, 179), (23, 26))
#     ```
#     """
#     IMG_SHAPE = high_resolution_fields_size
#     IMG_SHAPE2 = low_resolution_fields_size
#
#     high_res_fields = layers.Input(shape=IMG_SHAPE) # real or fake predictions
#     low_res_inputs = layers.Input(shape=IMG_SHAPE2)
#     inputs_high_res = res_block_initial(high_res_fields, [high_resolution_feature_channels[0]], 5, [1, 1],
#                                         f"test1234", attn_type ="None", varname = "pr") #self
#     inputs_high_res = res_block_initial(inputs_high_res, [high_resolution_feature_channels[0]], 5, [1, 1],
#                                         "input_blocka", attn_type ="None", varname = "pr") # cbam
#     x = conv_block(inputs_high_res, high_resolution_feature_channels[1], kernel_size=(5, 5), strides=(2, 2),
#                    use_bn=use_bn, use_bias=use_bias,
#                    use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU(), varname = "pr") # 64, 64
#     x = res_block_initial(x, [high_resolution_feature_channels[0]], 7, [1, 1], "input_blockb", attn_type ="None", varname = "pr") # cbam
#     x = res_block_initial(x, [high_resolution_feature_channels[1]], 7, [1, 1], "input_blockc", attn_type ="None", varname = "pr")
#     x = conv_block(x, high_resolution_feature_channels[2], kernel_size=(5, 5), strides=(2, 2),
#                    use_bn=use_bn, use_bias=use_bias,
#                    use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU(), varname = "pr") # 32, 32
#     x = res_block_initial(x, [high_resolution_feature_channels[1]], 5, [1, 1], "input_blockd", attn_type ="None", varname = "pr")
#     x = res_block_initial(x, [high_resolution_feature_channels[2]], 5, [1, 1], "input_blocke", attn_type ="None", varname = "pr")
#     x_init_raw = conv_block(x, high_resolution_feature_channels[3], kernel_size=(5, 5), strides=(2, 2),
#                             use_bn=use_bn, use_bias=use_bias,
#                             use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU(), varname = "pr") # 16, 16
#
#     # Weak conditioning on the large-scale fields
#     # creating a low dimensional space for the large scale fields
#     if conditioning:
#         x_large_scale = res_block_initial(low_res_inputs, [2], 3, [1, 1], "input_blockaa", attn_type ="None", varname = "pr") # channel
#         x_large_scale_flatten = tf.keras.layers.Flatten()(x_large_scale)
#         dense_input = tf.keras.layers.Dense(1,activation ='linear')(x_large_scale_flatten)
#
#         conditioned_inputs = TimeFilmLayer(
#             num_channels=high_resolution_feature_channels[3],
#             hidden_dim=128,
#             use_sinusoidal=False,
#             max_period=365,
#             name='time_film'
#         )([x_init_raw, dense_input])
#     else:
#         conditioned_inputs = x_init_raw
#     concat_outputs = res_block_initial(conditioned_inputs, [high_resolution_feature_channels[3]], 5, [1, 1], "output_convbbb2", attn_type ="None", varname = "pr") #self
#     x_init_raw = conv_block(concat_outputs, high_resolution_feature_channels[3], kernel_size=(5, 5), strides=(2, 2),
#                             use_bn=use_bn, use_bias=use_bias,
#                             use_dropout=use_dropout, drop_value=0.0,
#                             activation=tf.keras.layers.LeakyReLU(0.1), varname = "pr")
#     x_init_raw = conv_block(x_init_raw, 32, kernel_size=(5, 5), strides=(2, 2),
#                             use_bn=use_bn, use_bias=use_bias,
#                             use_dropout=use_dropout, drop_value=0.0,
#                             activation=tf.keras.layers.LeakyReLU(0.1), varname = "pr")
#     flattened_output = tf.keras.layers.Flatten()(x_init_raw)
#     dense2 = tf.keras.layers.Dense(256)(flattened_output)
#     dense3 = tf.keras.layers.Dense(64)(dense2)
#
#     x = layers.Dense(1)(dense3)
#
#     d_model = keras.models.Model([high_res_fields, low_res_inputs], x,
#                                  name="discriminator")
#     return d_model
#
#
# def res_gan(input_size, resize_output, num_filters, num_channels, num_classes,
#                           final_activation = tf.keras.layers.LeakyReLU(1), orog_predictor = True, temp_conditioning = False, varname = "tasmax"):
#     """new model has two layers of noise, and does not feed in the unet prediction"""
#
#     unet_prediction_layer = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
#     if orog_predictor:
#         input_topography_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
#     large_scale_fields = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])
#     noise = tf.keras.layers.Input(shape=[input_size[0], input_size[1], num_channels])
#
#     # Adding Embedding Layers at the start of the network.
#     time_of_year = tf.keras.layers.Input(shape=[1], name='time_input')
#
#     conditioned_large_scale_fields = TimeFilmLayer(
#         num_channels=num_channels,
#         hidden_dim=128,
#         use_sinusoidal=True,
#         max_period=365,
#         name='time_film_a', varname = varname
#     )([large_scale_fields, time_of_year])
#
#     conditioned_unet_fields = TimeFilmLayer(
#         num_channels=1,
#         hidden_dim=32,
#         use_sinusoidal=True,
#         max_period=365,
#         name='time_film_b', varname = varname
#     )([unet_prediction_layer, time_of_year])
#
#
#     # High-resolution fields
#     if orog_predictor:
#         concat_image_conditioned = tf.keras.layers.Concatenate(-1)([conditioned_unet_fields,  input_topography_fields])
#     else:
#         concat_image_conditioned = conditioned_unet_fields
#     print(concat_image_conditioned, conditioned_large_scale_fields, unet_prediction_layer)
#     x, temp1 = down_block(concat_image_conditioned, num_filters[2], kernel_size=5, i=4, use_pool=False,
#                           attn_type ="channel", varname = varname) # self
#     x, temp2 = down_block(x, num_filters[1], kernel_size=3, i =1, attn_type ="cbam", varname = varname) # 32, 32, channel
#     x, temp3 = down_block(x, num_filters[2], kernel_size=3, i =2, attn_type ="None", varname = varname) # 16, 16
#
#     # Low resolution fields
#     inputs_abstract = tf.keras.layers.Concatenate(-1)([conditioned_large_scale_fields, noise])
#     x1 = res_block_initial(inputs_abstract, [num_filters[2]*2], 3, [1, 1], f"test123", attn_type ="None",
#                            varname = varname) # cbam
#     x1 = res_block_initial(x1, [ num_filters[2]*2], 3, [1, 1], f"test11234", attn_type ="None",varname = varname) # cbam#down_block(x1, num_filters[2]*2, kernel_size=3, i=5, use_pool=False, attn_type ="cbam")
#     x1 = res_block_initial(x1, [64], 3, [1, 1], f"test1", attn_type ="None",varname = varname)
#     x1 = res_block_initial(x1, [128], 5, [1, 1], f"test2", attn_type ="None",varname = varname)
#
#     # Allow Mixing of the Low and High Resolution fields
#     concat_scales = tf.keras.layers.Concatenate(-1)([x1, x])
#     x = res_block_initial(concat_scales, [256], 3, [1, 1], f"test3", attn_type ="self",varname = varname) # cbam
#
#     # decode
#     x = up_block(x, temp3, kernel_size=3, filters = num_filters[2], i =0, concat = True, attn_type ="None",varname = varname) # channel
#     noise2 = tf.keras.layers.Input(shape=[x.shape[1], x.shape[2], int(num_channels//2)])
#     x = tf.keras.layers.Concatenate(-1)([noise2, x])
#     x = up_block(x, temp2, kernel_size=3, filters = num_filters[1], i =2, concat = True, attn_type ="cbam", varname = varname)
#     x = up_block(x, temp1, kernel_size=5, filters = num_filters[0], i =3, concat = True, attn_type ="cbam",varname = varname) # self
#
#     # Final output convolutions
#
#     output = x
#     output = TimeFilmLayer(
#         num_channels=num_filters[0],
#         hidden_dim=128,
#         use_sinusoidal=True,
#         max_period=365,
#         name='time_film_testcdf1'
#     )([output, time_of_year])
#
#     output = res_block_initial(output, [32], 5, [1, 1], "output_convbbb12347", attn_type ="channel", varname = varname)
#     output = tf.keras.layers.Conv2D(
#         16,
#         5,
#         activation=final_activation,
#         padding='same'
#     )(output)
#     output = tf.keras.layers.Conv2D(
#         num_classes,
#         1,
#         activation=final_activation,
#         padding='same'
#     )(output)
#     if orog_predictor:
#         input_layers = [noise, noise2, unet_prediction_layer] + [large_scale_fields, input_topography_fields, time_of_year]
#     else:
#         input_layers = [noise, noise2, unet_prediction_layer] + [large_scale_fields, time_of_year]
#     model = tf.keras.models.Model(input_layers, output, name='gan')
#     model.summary()
#     return model
#
#
# def unet(input_size, resize_output, num_filters, num_channels, num_classes,
#                           final_activation = 'linear', orog_predictor = True, temp_conditioning=False, varname = "tasmax"):
#
#     """
#
#     :param input_size:
#     :param resize_output: high-resolution target fields, for CORDEX-ML-Bench this is 128x128.
#     :param num_filters: a list consisting of number of filters in individual layrs [32, 64, 128] ...
#     :param kernel_size: not used
#     :param num_channels:
#     :param num_classes:
#     :param resize:
#     :param final_activation:
#     :return:
#     """
#     if orog_predictor:
#         input_topography_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
#
#     large_scale_fields = tf.keras.layers.Input(
#         shape=[input_size[0], input_size[1], num_channels],
#         name='spatial_input'
#     )
#     # temperature conditioning on the outputs
#     time_of_year = tf.keras.layers.Input(shape=[1], name='time_input')
#
#     conditioned_large_scale_fields = large_scale_fields#TimeFilmLayer(num_channels=num_channels,
#     #         hidden_dim=128,
#     #         use_sinusoidal=True,
#     #         max_period=365,
#     #         name='time_film_testas12', varname = varname
#     #     )([large_scale_fields, time_of_year])
#
#
#     # High-resolution inputs
#     if orog_predictor:
#         x, temp1 = down_block(input_topography_fields, num_filters[0], kernel_size=3, i =0, attn_type ="None", varname =  varname) # 64, 64
#         x, temp2 = down_block(x, num_filters[1], kernel_size=3, i =1, attn_type ="None", varname =  varname) # 32, 32
#         x, temp3 = down_block(x, num_filters[2], kernel_size=3, i =2, attn_type ="None", varname =  varname) # 16, 16
#
#     # Film conditioned low resolution inputs (16 x 16 inputs)
#     x1 = res_block_initial(conditioned_large_scale_fields, [num_filters[2]], 5, [1, 1], f"test2123434", attn_type ="channel", varname = varname)
#     x1 = res_block_initial(x1, [num_filters[2]*2], 3, [1, 1], f"test21233434", attn_type ="channel", varname = varname) # Channel, and the above
#     x1 = res_block_initial(x1, [64], 3, [1, 1], f"test1", attn_type ="None", varname = varname)
#     x1 = res_block_initial(x1, [128], 3, [1, 1], f"test2", attn_type ="None", varname = varname)
#
#     # Merge the inputs together
#     if orog_predictor:
#         concat_scales = tf.keras.layers.Concatenate(-1)([x1, x])
#     else:
#         concat_scales = x1
#
#     x = res_block_initial(concat_scales, [256], 3, [1, 1], f"test3", attn_type ="self", varname = varname) # cbam
#     # decode
#     if orog_predictor:
#         x = up_block(x, temp3, kernel_size=3, filters = num_filters[2], i =0, concat = True, attn_type ="cbam", varname = varname) # channel, channel, and self
#         x = up_block(x, temp2, kernel_size=3, filters = num_filters[1], i =2, concat = True, attn_type ="channel", varname = varname)
#         x = up_block(x, temp1, kernel_size=3, filters = num_filters[0], i =3, concat = True, attn_type ="None", varname = varname)
#     else:
#         x = up_block(x, [], kernel_size=3, filters = num_filters[2], i =0, concat = False, attn_type ="cbam", varname = varname) # channel, channel, self
#         x = up_block(x, [], kernel_size=3, filters = num_filters[1], i =2, concat = False, attn_type ="channel", varname = varname)
#         x = up_block(x, [], kernel_size=3, filters = num_filters[0], i =3, concat = False, attn_type ="None", varname = varname)
#     # conditioned_large_scale_fields_v2 = conditioned_large_scale_fields[:,-5]
#     # conditioned_large_scale_fields = TimeFilmLayer(
#     #     num_channels=num_filters[0],
#     #     hidden_dim=32,
#     #     use_sinusoidal=True,
#     #     max_period=365,
#     #     name='time_film_test'
#     # )([x, conditioned_large_scale_fields_v2])
#
#     # Final output convolutions
#     output = x#conditioned_large_scale_fields
#     output = res_block_initial(output, [64], 3, [1, 1], "output_convbbb12347", attn_type ="channel", varname = varname)
#     output = res_block_initial(output, [32], 3, [1, 1], "output_convbbb12348", attn_type="channel", varname = varname)
#     # if temp_conditioning:
#     #     output = TimeFilmLayer(
#     #         num_channels=32,
#     #         hidden_dim=128,
#     #         use_sinusoidal=False,
#     #         max_period=365,
#     #         name='time_film_testcdf'
#     #     )([output, time_of_year])
#     # else:
#     # output = TimeFilmLayer(
#     #     num_channels=32,
#     #     hidden_dim=128,
#     #     use_sinusoidal=True,
#     #     max_period=365,
#     #     name='time_film_testcdf1'
#     # )([output, time_of_year])
#     output = tf.keras.layers.Conv2D(32, 5, activation=final_activation, padding='same')(output)
#     output = tf.keras.layers.Conv2D(16, 5, activation=final_activation, padding='same')(output)
#     output = tf.keras.layers.Conv2D(num_classes, 1, activation=final_activation, padding ='same')(output)
#     if orog_predictor:
#         input_layers = [large_scale_fields, input_topography_fields, time_of_year]
#     else:
#         input_layers = [large_scale_fields, time_of_year]
#     model = tf.keras.models.Model(input_layers, output, name='unet')
#     model.summary()
#     return model
#

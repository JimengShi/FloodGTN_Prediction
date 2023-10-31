#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : WaLeF
@ FileName: rnn.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 6/18/22 08:57
"""

from pandas import DataFrame
from pandas import concat
from pandas import concat, read_csv
from tensorflow import keras
from tensorflow.keras import Model, Input, layers
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, SimpleRNN, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.models import load_model


from tensorflow import keras
from tensorflow.keras import layers


# def multiheadattention(inputs, head_size, num_heads, ff_dim, dropout):
#     # Normalization and Attention
#     x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
#     x = layers.Dropout(dropout)(x)
#     x = layers.LayerNormalization(epsilon=1e-6)(x)
#     res = x + inputs

#     # Feed Forward Part
#     x = layers.Conv1D(filters=ff_dim, 
#                       kernel_size=2, 
#                       activation="relu", 
#                       padding='same')(res)
#     x = layers.Dropout(dropout)(x)
#     x = layers.Conv1D(filters=inputs.shape[-1], 
#                       kernel_size=2, 
#                       activation="relu", 
#                       padding='same')(x)
#     x = layers.LayerNormalization(epsilon=1e-6)(x)
#     return x + res


def multiheadattention(inputs, head_size, num_heads, ff_dim, dropout):
    # Normalization and Attention
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Dense(units=ff_dim, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(units=inputs.shape[-1], activation="relu")(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res



def transformer_encoder(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout, mlp_dropout, masked_value):
    
    inputs = keras.Input(shape=input_shape)
    masked = layers.Masking(mask_value=masked_value)(inputs)
    x = masked
    for _ in range(num_transformer_blocks):
        x = multiheadattention(x, head_size, num_heads, ff_dim, dropout)

    x = layers.Dense(mlp_units)(x)
    x = layers.Dropout(mlp_dropout)(x)
    x = Flatten()(x)
    outputs = layers.Dense(96)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model
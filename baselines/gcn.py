#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : WaLeF
@ FileName: gcn.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 6/20/23 15:31
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
from tensorflow.keras.layers import LSTM, Dense, Dropout, Concatenate, BatchNormalization, Flatten
from tensorflow.keras.models import load_model
from spektral.layers import GCNConv


def gcn(n_nodes, n_timesteps, gcn1, gcn2, lstm_unit, dropout, masked_value):
    inp_lap = Input((n_nodes, n_nodes))
    inp_seq = Input((n_nodes, n_timesteps))
    #inp_seq = layers.Masking(mask_value=masked_value)(inp_seq)
    
    # GCN
    x = GCNConv(gcn1, activation='relu')([inp_seq, inp_lap])
    x = GCNConv(gcn2, activation='relu')([x, inp_lap])

    # RNN
    xx = LSTM(lstm_unit, activation='relu', return_sequences=True)(inp_seq)


    x = Concatenate()([x, xx])
    x = Flatten()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(96)(x)

    model = Model(inputs=[inp_seq, inp_lap], outputs=outputs)

    return model, GCNConv

#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : WaLeF
@ FileName: helper.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 6/20/23 14:22
"""

from pandas import DataFrame
from pandas import concat
import typing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def stage_series_to_supervised(data, n_in, K, n_out, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in+K, K, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# class GraphInfo:
#     def __init__(self, edges: typing.Tuple[list, list], num_nodes: int):
#         self.edges = edges
#         self.num_nodes = num_nodes
        
# class GraphConv(layers.Layer):
#     def __init__(
#         self,
#         in_feat,
#         out_feat,
#         graph_info: GraphInfo,
#         aggregation_type="mean",
#         combination_type="concat",
#         activation: typing.Optional[str] = None,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.in_feat = in_feat
#         self.out_feat = out_feat
#         self.graph_info = graph_info
#         self.aggregation_type = aggregation_type
#         self.combination_type = combination_type
#         self.weight = tf.Variable(
#             initial_value=keras.initializers.glorot_uniform()(shape=(in_feat, out_feat), dtype="float32"),
#             trainable=True,
#         )
#         self.activation = layers.Activation(activation)

#     def aggregate(self, neighbour_representations: tf.Tensor):
#         aggregation_func = {
#             "sum": tf.math.unsorted_segment_sum,
#             "mean": tf.math.unsorted_segment_mean,
#             "max": tf.math.unsorted_segment_max,
#         }.get(self.aggregation_type)

#         if aggregation_func:
#             return aggregation_func(neighbour_representations, 
#                                     self.graph_info.edges[0],
#                                     num_segments=self.graph_info.num_nodes
#                                    )

#         raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")

#     def compute_nodes_representation(self, features: tf.Tensor):
#         """Computes each node's representation.

#         The nodes' representations are obtained by multiplying the features tensor with
#         `self.weight`. Note that
#         `self.weight` has shape `(in_feat, out_feat)`.

#         Args:
#             features: Tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`

#         Returns:
#             A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`
#         """
#         return tf.matmul(features, self.weight)

#     def compute_aggregated_messages(self, features: tf.Tensor):
#         neighbour_representations = tf.gather(features, self.graph_info.edges[1])
#         aggregated_messages = self.aggregate(neighbour_representations)
#         return tf.matmul(aggregated_messages, self.weight)

#     def update(self, nodes_representation: tf.Tensor, aggregated_messages: tf.Tensor):
#         if self.combination_type == "concat":
#             h = tf.concat([nodes_representation, aggregated_messages], axis=-1)
#         elif self.combination_type == "add":
#             h = nodes_representation + aggregated_messages
#         else:
#             raise ValueError(f"Invalid combination type: {self.combination_type}.")

#         return self.activation(h)

#     def call(self, features: tf.Tensor):
#         """Forward pass.
#         Args:
#             features: tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`
#         Returns:
#             A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`
#         """
#         nodes_representation = self.compute_nodes_representation(features)
#         aggregated_messages = self.compute_aggregated_messages(features)
#         return self.update(nodes_representation, aggregated_messages)
    
    
    
# class LSTMGC(layers.Layer):
#     """Layer comprising a convolution layer followed by LSTM and dense layers."""

#     def __init__(
#         self,
#         in_feat,
#         out_feat,
#         lstm_units: int,
#         input_seq_len: int,
#         output_seq_len: int,
#         l1_reg: float,
#         l2_reg: float,
#         graph_info: GraphInfo,
#         graph_conv_params: typing.Optional[dict] = None,
#         **kwargs,
        
#     ):
#         super().__init__(**kwargs)

#         # graph conv layer
#         if graph_conv_params is None:
#             graph_conv_params = {
#                 "aggregation_type": "mean",
#                 "combination_type": "concat",
#                 "activation": None,
#             }
#         self.graph_conv = GraphConv(in_feat, out_feat, graph_info, **graph_conv_params)
        
#         # lstm layer
#         self.lstm = layers.SimpleRNN(lstm_units, 
#                                      activation="relu", 
#                                      kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg)
#                                     )
#         self.dense = layers.Dense(output_seq_len, 
#                                   kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg)
#                                  )

#         self.input_seq_len = input_seq_len
#         self.output_seq_len = output_seq_len
#         self.l1_reg = l1_reg
#         self.l2_reg = l2_reg

        
#     def call(self, inputs):
#         """Forward pass.
#         Args:
#             inputs: tf.Tensor of shape `(batch_size, input_seq_len, num_nodes, in_feat)`
#         Returns:
#             A tensor of shape `(batch_size, output_seq_len, num_nodes)`.
#         """

#         # convert shape to  (num_nodes, batch_size, input_seq_len, in_feat)
#         inputs = tf.transpose(inputs, [2, 0, 1, 3])

#         gcn_out = self.graph_conv(inputs)         # gcn_out shape: (num_nodes, batch_size, input_seq_len, out_feat)
#         shape = tf.shape(gcn_out)
#         num_nodes, batch_size, input_seq_len, out_feat = (shape[0], shape[1], shape[2], shape[3])  # 5, 512, 96, 1

#         # LSTM takes only 3D tensors as input
# #         gcn_out = tf.reshape(gcn_out, (batch_size, input_seq_len, num_nodes*out_feat))
# #         lstm_out = self.lstm(gcn_out)             # lstm_out shape: (batch_size, lstm_units)

# #         dense_output = self.dense(lstm_out)       # dense_output shape: (batch_size, num_nodes * output_seq_len)
# #         output = tf.reshape(dense_output, (4, batch_size, self.output_seq_len))
# #         return tf.transpose(output, [1, 2, 0])    # returns Tensor of shape (batch_size, output_seq_len, num_nodes)
    
    
    
#         gcn_out = tf.reshape(gcn_out, (batch_size * num_nodes, input_seq_len, out_feat))
#         lstm_out = self.lstm(gcn_out)             # lstm_out shape: (batch_size * num_nodes, lstm_units)

#         dense_output = self.dense(lstm_out)       # dense_output shape: (batch_size * num_nodes, output_seq_len)
#         output = tf.reshape(dense_output, (num_nodes, batch_size, self.output_seq_len))
#         return tf.transpose(output, [1, 2, 0])    # returns Tensor of shape (batch_size, output_seq_len, num_nodes)

        
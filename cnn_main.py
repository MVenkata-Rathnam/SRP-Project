import numpy as np
import scipy as sio

from cnn_load_data import *
from cnn_parameters import *
from cnn_core_model import *
"""Loading the data"""
common_param = Parameters()
training_set,validation_set,test_set,test_truth_set = load_data(common_param)

"""Initializing the layer parameters"""
input_shape = training_set.shape
common_param.initialize_layer_parameters(input_shape)
layer_type = [{'conv_layer':ConvolutionLayer},{'pool_layer':PoolingLayer},{'fully_conn_layer':FullyConnectedLayer},{'output_layer':OutputLayer}]
layer_activation = [{'conv_layer':'tanh'},{'pool_layer':'maximum'},{'fully_conn_layer':'tanh'},{'output_layer':'softmax'}]

"""Initializing the model"""
net = Model(input_shape,layer_type,layer_activation,common_param)

"""Initializing the layers"""
net.initialize_layers()

"""Feed Forwarding in the CNN"""
for i in range(0,input_shape[0]):
    net.feed_forward(training_set[i])

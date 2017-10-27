import numpy as np
import scipy as sio
import math
import time
from matplotlib import pyplot as plt

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

"""Initializing the training parameters"""
error = math.inf
no_of_batches = input_shape[0] // common_param.batch_size
epoch = 0
start_time = time.time()
error_list = []
epoch_list = []

"""Checking feed forward
for i in range(0,input_shape[0]): 
    net.feed_forward(training_set[i])
"""

"""Training the network with the training set"""
while (error > common_param.minimum_error and epoch < 10):
    error = 0.0
    epoch += 1
    error_batch_list = []
    batch_list = []
    shuffle_in_unison(training_set,validation_set)
    print("epoch %d initiated:" % epoch)
    for batch in range(0,input_shape[0],common_param.batch_size):
        #Calling the gradient descent method
        print ("batch %d" % batch)
        error += (1/common_param.batch_size)*net.gradient_descent(training_set[batch:batch+(common_param.batch_size)],validation_set[batch:batch+(common_param.batch_size)])
        error_batch_list.append(error)
        batch_list.append(batch)

    #plt.plot(batch_list,error_batch_list)
    #plt.show()

    del error_batch_list
    del batch_list
    
    error = (error / no_of_batches)
    print (error)
    error_list.append(error)
    epoch_list.append(epoch)
    print ("Epoch %d over" % epoch)
    plt.plot(epoch_list,error_list)
    plt.show()
    
print (round(time.time() - start_time))

"""Testing the neural network"""
correct_predict_count = 0
index = count = 0
shuffle_in_unison(test_set,test_truth_set)
while count < 200:
    if(test_truth_set[index] != 0):
        net.feed_forward(test_set[index])
        print ("Actual Output : " , test_truth_set[index])
        if(net.layers[-1].predicted_output == test_truth_set[index]):
            correct_predict_count += 1
        count += 1
    index += 1

"""Calculating the accuracy"""
accuracy = (correct_predict_count / 200)*100
print ("Correct Predict count : " , correct_predict_count)
print ("Accuracy : ",accuracy)

import math
import random
import numpy as np
import scipy as sio

"""Class Model for Convolutional Neural Network""" 
class Model(object):
    def __init__(self,input_shape,layer_type,layer_activation,common_param):
        self.layers = []
        self.no_of_trains = input_shape[0];
        self.input_size = (input_shape[1],1)
        self.layer_type = layer_type
        self.layer_activation = layer_activation
        self.common_param = common_param

    """Instantiating the layers"""
    def initialize_layers(self):
        for element in self.layer_type:
            for key in element.keys():
                if(len(self.layers) == 0):
                    layer = element[key](self.input_size,self.common_param)
                else:
                    layer = element[key](self.layers[-1].input_size,self.common_param)
                self.layers.append(layer)
                
    def feed_forward(self):
        pass
    def back_propagation(self):
        pass
    def gradient_descent(self):
        pass
    def tanh(self):
        pass
    def maximum(self):
        pass
    def softmax(self):
        pass

"""Class Model for Neuron"""
class Neuron(object):
    def __init__(self):
        self.output_value = 0.0
        self.error = 0.0
        self.gradient=0.0

"""Class Model for Dendron"""
class Connection(object):
    def __init__(self,weight):
        self.weight = weight
        self.dweight = 0.0
        
"""Class Model for Convolution Layer"""
class ConvolutionLayer(object):
    def __init__(self,input_shape,common_param):
        """Initializing the layer parameters"""
        self.common_param = common_param
        self.input_size = input_shape
        self.output_size = self.common_param.convolutional_layer_size
        self.neurons = []
        self.dendrons = []
        self.connection_size = self.common_param.convolution_kernel_size*self.common_param.no_of_filters
    
        """Initializing the neurons in the layer"""
        for i in range(0,self.common_param.convolutional_layer_size):
            neuron = Neuron();
            self.neurons.append(neuron)

        """Initializing the dendrons of this layer"""
        for i in range(0,self.connection_size):
            self.dendrons.append(Connection(random.uniform(self.common_param.weight_minimum_limit,self.common_param.weight_maximum_limit)))

        print (len(self.dendrons))
    def convolve():
        pass
        
"""Class Model for Pooling Layer"""
class PoolingLayer(object):
    def __init__(self,input_shape,common_param):
        """Initializing the layer parameters"""
        self.common_param = common_param
        self.input_size = input_shape
        self.output_size = self.common_param.pooling_layer_size
        self.neurons = []

        """Initializing the neurons in the layer"""
        for i in range(0,self.common_param.pooling_layer_size):
            neuron = Neuron();
            self.neurons.append(neuron)
            
    def pool():
        pass

"""Class Model for Fully Connected Layer"""
class FullyConnectedLayer(object):
    def __init__(self,input_shape,common_param):
        """Initializing the layer parameters"""
        self.common_param = common_param
        self.input_size = input_shape
        self.output_size = self.common_param.fully_connected_layer_size
        self.connection_size = self.common_param.pooling_layer_size * self.output_size
        self.neurons = []
        self.dendrons = []

        """Initializing the neurons in the layer"""
        for i in range(0,self.common_param.fully_connected_layer_size):
            neuron = Neuron();
            self.neurons.append(neuron)

        """Initializing the dendrons of this layer"""
        for i in range(0,self.connection_size):
            self.dendrons.append(Connection(random.uniform(self.common_param.weight_minimum_limit,self.common_param.weight_maximum_limit)))

        print (len(self.dendrons))    
    def feed_forward():
        pass
    
"""Class Model for Output Layer"""
class OutputLayer(object):
    def __init__(self,input_shape,common_param):
        """Initializing the layer parameters"""
        self.common_param = common_param
        self.input_size = input_shape
        self.output_size = self.common_param.output_layer_size
        self.connection_size = self.common_param.fully_connected_layer_size * self.output_size
        self.neurons = []
        self.dendrons = []

        """Initializing the neurons in the layer"""
        for i in range(0,self.common_param.output_layer_size):
            neuron = Neuron();
            self.neurons.append(neuron)

        """Initializing the dendrons of this layer"""
        for i in range(0,self.connection_size):
            self.dendrons.append(Connection(random.uniform(self.common_param.weight_minimum_limit,self.common_param.weight_maximum_limit)))
        print (len(self.dendrons))

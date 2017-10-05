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
                    layer = element[key](self.layers[-1].output_size,self.common_param)
                self.layers.append(layer)
                
    def feed_forward(self,input_layer):
        for index,layer in enumerate(self.layers):
            if(isinstance(layer,ConvolutionLayer)):
                """Feedforward to convolution Layer"""
                layer.convolve(input_layer)   
            elif(isinstance(layer,PoolingLayer)):
                """Feedforward to pooling Layer"""
                layer.pool(self.layers[index - 1])
            elif(isinstance(layer,FullyConnectedLayer)):
                """Feedforward to Fully Connected Layer"""
                layer.feed_forward(self.layers[index - 1])
            elif(isinstance(layer,OutputLayer)):
                """Feedforward to Output Layer"""
                layer.feed_forward(self.layers[index - 1])
          
    def back_propagation(self):
        pass
    def gradient_descent(self):
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

    """Convolving the filters with the output of input layer"""
    def convolve(self,input_layer):
        """Initializing the required parameters"""
        neuron_index = 0
        filter_index = 0
        sum_of_multiple = 0.0

        """Performing the convolution operation for all the filters"""
        for i in range(0,self.common_param.no_of_filters):
            filter_index = i
            """Sliding the filter over the input with the decided stride"""
            for j in range(0,(self.input_size[0] - self.common_param.convolution_kernel_size + 1)):
                """Calculating the element wise multiplication and sum"""
                sum_of_multiple = 0.0
                for k in range(j,j+self.common_param.convolution_kernel_size):
                    element_wise_multiple = input_layer[j]*self.dendrons[k].weight
                    sum_of_multiple = sum_of_multiple + element_wise_multiple
                self.neurons[neuron_index].output_value = self.tanh(sum_of_multiple)
                neuron_index += 1
                
    def tanh(self, neuron_value):
        return math.tanh(neuron_value)

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
            
    def pool(self,input_layer):
        """Initializing the required parameters"""
        neuron_index = 0
        
        """Performing the downsapling"""
        for i in range(0,self.input_size - self.common_param.pooling_kernel_size + 1,self.common_param.pooling_kernel_size):
            self.neurons[neuron_index].output_value = self.maximum(input_layer,i,i+self.common_param.pooling_kernel_size)
                                                              
    def maximum(self,input_layer,start,end):
        max_value = -0.00001

        """Finding the maximum value within the filter size"""
        for i in range(start,end):
            if input_layer.neurons[i].output_value > max_value:
                max_value = input_layer.neurons[i].output_value

        return max_value

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
    def feed_forward(self,input_layer):
        """Initializing the required parameters"""
        neuron_index = 0
        dendron_index = 0
        net_output = 0.0
        for i in range(0,self.output_size):
            net_output = 0.0
            for j in range(0,self.input_size):
                net_output += input_layer.neurons[j].output_value * self.dendrons[dendron_index].weight
                dendron_index += 1
            self.neurons[neuron_index].output_value = self.tanh(net_output)
            neuron_index += 1
    def tanh(self,neuron_value):
        return math.tanh(neuron_value)
    
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

    def feed_forward(self,input_layer):
        """Initializing the required parameters"""
        neuron_index = 0
        dendron_index = 0
        net_output = 0.0
        for i in range(0,self.output_size):
            net_output = 0.0
            for j in range(0,self.input_size):
                net_output += input_layer.neurons[j].output_value * self.dendrons[dendron_index].weight
                dendron_index += 1
            self.neurons[neuron_index].output_value = net_output
            neuron_index += 1
        self.softmax()

    def softmax(self):
        """Initializing the required parameters"""
        max_value = -0.00001
        sum_value = 0.0
        for i in range(0,self.output_size):
            sum_value += self.neurons[i].output_value
            if self.neurons[i].output_value > max_value:
                max_value = self.neurons[i].output_value

        for i in range(0,self.output_size):
            e_x = math.exp(self.neurons[i].output_value - max_value)
            self.neurons[i].output_value = e_x/sum_value

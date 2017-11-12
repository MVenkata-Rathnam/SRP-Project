import math

class Parameters(object):

    def __init__(self):    
        """Initializing essential parameters"""
        self.x_axis = 0
        self.y_axis = 0
        self.no_of_classes = 9
        self.count_of_each_class = 200
        self.learning_rate = 0.01
        self.momentum_rate = 0.1
        self.epochs = 30
        self.stride = 1
        self.padding = 0
        self.batch_size = 10
        self.minimum_error = 0.005
        self.maximum_iteration = 500
        self.input_layer_size = 0
        self.convolution_kernel_size = 0
        self.no_of_filters = 20
        self.convolutional_layer_size = 0
        self.pooling_kernel_size = 3
        self.pooling_layer_size = 0
        self.fully_connected_layer_size = 100
        self.output_layer_size = self.no_of_classes
        self.weight_minimum_limit = -0.05
        self.weight_maximum_limit = 0.05
        self.final_result_set = []

    def initialize_layer_parameters(self,input_shape):
        """Initializing layer parameters"""    
        self.input_layer_size = input_shape[1]
        self.convolution_kernel_size = self.input_layer_size // self.no_of_classes
        self.convolutional_without_kernel = round(self.input_layer_size - self.convolution_kernel_size + 1)
        self.convolutional_layer_size = self.no_of_filters * self.convolutional_without_kernel
        self.pooling_without_kernel = round(self.convolutional_without_kernel/self.pooling_kernel_size)
        self.pooling_layer_size = self.no_of_filters * self.pooling_without_kernel

        """Testing the determined layer size
        print (self.input_layer_size)
        print (self.convolutional_without_kernel)
        print (self.convolutional_layer_size)
        print (self.pooling_layer_size)
        """

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
        self.total_error = 0.0
        self.accuracy = 0.0

    """Instantiating the layers"""
    def initialize_layers(self):
        for element in self.layer_type:
            for key in element.keys():
                if(len(self.layers) == 0):
                    layer = element[key](self.input_size,self.common_param)
                else:
                    layer = element[key](self.layers[-1].output_size,self.common_param)
                self.layers.append(layer)

    """feed forwarding through the layers"""            
    def feed_forward(self,input_layer):
        self.input_layer = input_layer
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

    """back propagating through the layers"""      
    def back_propagation(self,input_data,ground_truth_data):
        """back propagating from output to convolution"""
        for i in range(len(self.layers)-1,-1,-1):
            if(isinstance(self.layers[i],OutputLayer)):
                self.layers[i].back_propagate(input_data,ground_truth_data,self.layers[i-1])
            elif(isinstance(self.layers[i],ConvolutionLayer)):
                self.layers[i].back_propagate(input_data)
            else:
                self.layers[i].back_propagate(self.layers[i-1])

    def weight_updation(self):
        """Updating the weight after a batch is over"""
        for i in range(len(self.layers)-1,-1,-1):
            if(isinstance(self.layers[i],OutputLayer)):
                self.layers[i].weight_updation(self.layers[i-1])
            elif(isinstance(self.layers[i],PoolingLayer)):
                pass
            elif(isinstance(self.layers[i],ConvolutionLayer)):
                self.layers[i].weight_updation()
            else:
                self.layers[i].back_propagate(self.layers[i-1])

    """Updating the bias value of all the neurons"""
    def bias_updation(self):
        for i in range(len(self.layers)):
            if(isinstance(self.layers[i],PoolingLayer)):
               pass

    """Implementing the gradient descent"""
    def gradient_descent(self,input_data,ground_truth_data):
        no_of_samples = input_data.shape[0]
        self.initialize_gradient()
        error = 0.0
        for i in range(0,no_of_samples):
            self.feed_forward(input_data[i])
            self.back_propagation(input_data[i],ground_truth_data[i])
            for j in range(0,len(self.layers[-1].neurons)):
                if(j == int(ground_truth_data[i])):
                    #print ("Actual Output : %f" % self.layers[-1].neurons[i].output_value)
                    error = error + self.cost_function(self.layers[-1].neurons[j].output_value)
                    break
        self.total_error = self.total_error + error
        self.weight_updation()
        self.bias_updation()
        return error

    """Initializing the gradient of all the neurons for the batch"""
    def initialize_gradient(self):
        for i in range(0,len(self.layers)):
            for j in range(0,len(self.layers[i].neurons)):
                self.layers[i].neurons[j].gradient = 0.0
            if(isinstance(self.layers[i],ConvolutionLayer)):
                for k in range(0,len(self.layers[i].dendrons)):
                    self.layers[i].dendrons[k].dweight = 0.0
             
    """Defining the cost function"""
    def cost_function(self,output_value):
        return (-1/self.common_param.batch_size)*math.log(output_value)
        
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
        self.filter_bias = [1.0 for i in range(self.common_param.no_of_filters)]
        self.connection_size = self.common_param.convolution_kernel_size*self.common_param.no_of_filters
    
        """Initializing the neurons in the layer"""
        for i in range(0,self.common_param.convolutional_layer_size):
            neuron = Neuron();
            self.neurons.append(neuron)

        """Initializing the dendrons of this layer"""
        for i in range(0,self.connection_size):
            self.dendrons.append(Connection(random.uniform(self.common_param.weight_minimum_limit,self.common_param.weight_maximum_limit)))

        #print (len(self.dendrons))

    """Convolving the filters with the output of input layer"""
    def convolve(self,input_layer):
        """Initializing the required parameters"""
        neuron_index = 0
        filter_index = 0
        sum_of_multiple = 0.0

        """Performing the convolution operation for all the filters"""
        for i in range(0,self.common_param.no_of_filters):
            filter_index = i*self.common_param.convolution_kernel_size
            """Sliding the filter over the input with the decided stride"""
            for j in range(0,(self.input_size[0] - self.common_param.convolution_kernel_size + 1),self.common_param.stride):
                """Calculating the element wise multiplication and sum"""
                sum_of_multiple = 0.0
                filter_index = i*self.common_param.convolution_kernel_size
                for k in range(j,j+self.common_param.convolution_kernel_size):
                    element_wise_multiple = input_layer[j]*self.dendrons[filter_index].weight
                    #print (input_layer[j] , " ", self.dendrons[k].weight, " " , element_wise_multiple)
                    sum_of_multiple = sum_of_multiple + element_wise_multiple
                    filter_index += 1
                sum_of_multiple += self.filter_bias[i]
                #print ("Convolution output " , sum_of_multiple)
                self.neurons[neuron_index].output_value = self.tanh(sum_of_multiple)
                #print ("Convolution output " , self.neurons[neuron_index].output_value)
                neuron_index += 1

    """Defining the back propagation"""
    def back_propagate(self,input_layer):
        dendron_index = 0
        neuron_index = 0
        for i in range(0,self.common_param.no_of_filters):
            dendron_index = i*self.common_param.convolution_kernel_size
            for j in range(0,len(input_layer) - self.common_param.convolution_kernel_size + 1,self.common_param.stride):
                dendron_index = i*self.common_param.convolution_kernel_size
                for k in range(0,self.common_param.convolution_kernel_size):
                    #input_neuron_gradient = (self.neurons[neuron_index].gradient * self.dendrons[dendron_index].weight)
                    self.dendrons[dendron_index].dweight += input_layer[j] * self.neurons[neuron_index].gradient
                    #print ("dendron_index" , dendron_index , "dweight ", self.dendrons[dendron_index].dweight)
                    dendron_index += 1
                neuron_index += 1

    """Updating the weight"""
    def weight_updation(self):
        for i in range(0,len(self.dendrons)):
            self.dendrons[i].weight -= self.common_param.learning_rate * (self.dendrons[i].dweight/(self.common_param.batch_size*(self.input_size[0] - self.common_param.convolution_kernel_size)))
            #print ("Convolution Layer dweight and weight : ", self.dendrons[i].dweight, self.dendrons[i].weight)

    """Updating the bias"""
    def bias_updation(self):
        for i in range(0,len(self.neurons),self.input_size[0] - self.common_param.convolution_kernel_size):
            sum_value = 0.0
            for j in range(i,i+self.input_size[0] - self.common_param.convolution_kernel_size):
                sum_value += self.neurons[j].gradient
            self.neurons[i].bias -= self.common_param.learning_rate*(sum_value/(self.common_param.batch_size*(self.input_size[0] - self.common_param.convolution_kernel_size)))

    """Defining the activation function"""                
    def tanh(self, neuron_value):
        #return math.tanh(neuron_value)
        #print (neuron_value)
        return (math.exp(neuron_value) - math.exp(-neuron_value))/(math.exp(neuron_value) + math.exp(-neuron_value))

    """Defining the derivative of the activation function"""
    def tanh_deriv(self,neuron_value):
        #return 1.0 - math.tanh(neuron_value)**2
        return (1.0 - self.tanh(neuron_value))*(1.0 + self.tanh(neuron_value))

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
        
        """Performing the downsampling"""
        for i in range(0,self.input_size - self.common_param.pooling_kernel_size + 1,self.common_param.pooling_kernel_size):
            self.neurons[neuron_index].output_value = self.maximum(input_layer,i,i+self.common_param.pooling_kernel_size)
            neuron_index += 1
            #print (self.neurons[neuron_index].output_value)
            
    def maximum(self,input_layer,start,end):
        max_value = input_layer.neurons[start].output_value

        """Finding the maximum value within the filter size"""
        for i in range(start,end):
            if input_layer.neurons[i].output_value > max_value:
                max_value = input_layer.neurons[i].output_value

        return max_value

    def back_propagate(self,input_layer):
        dendron_index = 0
        for i in range(0,len(input_layer.neurons),self.common_param.pooling_kernel_size):
            for j in range(i,i+self.common_param.pooling_kernel_size):
                if(input_layer.neurons[j].output_value == self.neurons[dendron_index].output_value):
                    input_layer.neurons[j].gradient = self.neurons[dendron_index].gradient
                    #print ("Neuron Index " , j , "Convolution Neuron Gradient : " , input_layer.neurons[j].output_value, " ", self.neurons[dendron_index].output_value)
                else:
                    input_layer.neurons[j].gradient = 0.0
            dendron_index += 1
                
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
        self.bias_matrix = [1.0 for i in range(self.output_size)]

        """Initializing the neurons in the layer"""
        for i in range(0,self.common_param.fully_connected_layer_size):
            neuron = Neuron();
            self.neurons.append(neuron)

        """Initializing the dendrons of this layer"""
        for i in range(0,self.connection_size):
            self.dendrons.append(Connection(random.uniform(self.common_param.weight_minimum_limit,self.common_param.weight_maximum_limit)))

        #print (len(self.dendrons))    
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
            self.neurons[neuron_index].output_value += self.bias_matrix[i]
            self.neurons[neuron_index].output_value = self.tanh(net_output)
            neuron_index += 1

    """Defining the back propagation"""
    def back_propagate(self,input_layer):
        dendron_index = 0
        for i in range(0,len(self.neurons)):
            for j in range(0,len(input_layer.neurons)):
                input_layer.neurons[j].gradient += (self.neurons[i].gradient * self.dendrons[dendron_index].weight) * self.tanh_deriv(input_layer.neurons[j].output_value)
                dendron_index += 1

    """Updating the weight"""
    def weight_updation(self,input_layer):
        dendron_index = 0
        for i in range(0,len(self.neurons)):
            for j in range(0,len(input_layer.neurons)):
                self.dendrons[dendron_index].weight -= self.common_param.learning_rate*(input_layer.neurons[j].output_value*(self.neurons[i].gradient/self.common_param.batch_size))
                dendron_index += 1
                
    """Updating the bias"""
    def bias_updation(self):
        for i in range(0,len(self.neurons)):
            self.bias_matrix[i] -= self.common_param.learning_rate*(self.neurons[i].gradient/self.common_param.batch_size)
                                                                     
    """Defining the activation function"""
    def tanh(self,neuron_value):
        return (math.exp(neuron_value) - math.exp(-neuron_value))/(math.exp(neuron_value) + math.exp(-neuron_value))

    """Defining the derivative of the activation function"""
    def tanh_deriv(self,neuron_value):
        #return 1.0 - math.tanh(neuron_value)**2
        return (1.0 - self.tanh(neuron_value))*(1.0 + self.tanh(neuron_value))
    
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
        self.bias_matrix = [1.0 for i in range(self.output_size)]
        self.predicted_output = 0

        """Initializing the neurons in the layer"""
        for i in range(0,self.common_param.output_layer_size):
            neuron = Neuron();
            neuron.output_value = 0.0
            self.neurons.append(neuron)

        """Initializing the dendrons of this layer"""
        for i in range(0,self.connection_size):
            self.dendrons.append(Connection(random.uniform(self.common_param.weight_minimum_limit,self.common_param.weight_maximum_limit)))
        #print (len(self.dendrons))

    def feed_forward(self,input_layer):
        """Initializing the required parameters"""
        neuron_index = 0
        dendron_index = 0
        net_output = 0.0
        sum_of_exponential = 0.0
        for i in range(0,self.output_size):
            net_output = 0.0
            for j in range(0,self.input_size):
                net_output += input_layer.neurons[j].output_value * self.dendrons[dendron_index].weight
                dendron_index += 1
            self.neurons[neuron_index].output_value += self.bias_matrix[i]
            self.neurons[neuron_index].output_value = net_output
            sum_of_exponential += math.exp(self.neurons[neuron_index].output_value)
            neuron_index += 1
        #print ("Output Value")
        tempMax = -0.0000001
        """for i in range(0,self.output_size):
            self.neurons[i].output_value = self.sigmoid(self.neurons[i].output_value)
        """    
        for i in range(0,self.output_size):
            #print ("Actual Output : " , self.neurons[i].output_value)
            self.neurons[i].output_value = self.softmax(self.neurons[i].output_value,sum_of_exponential)
            print ("Probability : " , self.neurons[i].output_value)
            if( self.neurons[i].output_value > tempMax):
                pos = i
                tempMax = self.neurons[i].output_value
            #self.neurons[i].output_value = self.softmax(self.neurons[i].output_value)
        self.predicted_output = (pos + 1)
        print ("Predicted class : ", self.predicted_output)
        self.common_param.final_result_set.append(self.predicted_output)
        #print (self.neurons[i].output_value)
            
    """Defining the back propagation"""
    def back_propagate(self,input_data,ground_truth_data,input_layer):
        print ("Correct Class : " , ground_truth_data)
        for i in range(0,len(self.neurons)):
            if(i == (ground_truth_data-1)):
                self.neurons[i].gradient += - (1 - self.neurons[i].output_value)*self.dSigmoid(self.neurons[i].output_value)
            else:
                self.neurons[i].gradient += - (0 - self.neurons[i].output_value)*self.dSigmoid(self.neurons[i].output_value)
            
        dendron_index = 0
        for i in range(0,len(self.neurons)):
            for j in range(0,len(input_layer.neurons)):
                input_layer.neurons[j].gradient += (self.neurons[i].gradient * self.dendrons[dendron_index].weight) * self.tanh_deriv(input_layer.neurons[j].output_value)
                dendron_index += 1

    def weight_updation(self,input_layer):
        dendron_index = 0
        for i in range(0,len(self.neurons)):
            for j in range(0,len(input_layer.neurons)):
                self.dendrons[dendron_index].weight -= self.common_param.learning_rate * (input_layer.neurons[j].output_value*(self.neurons[i].gradient/self.common_param.batch_size))
                dendron_index += 1
                
    """Updating the bias"""
    def bias_updation(self):
        for i in range(0,len(self.neurons)):
            self.bias_matrix[i] -= self.common_param.learning_rate*(self.neurons[i].gradient/self.common_param.batch_size)
            
    """Defining the classifier function"""
    def softmax(self,neuron_value,sum_value):
        e_x = math.exp(neuron_value)
        return (e_x/sum_value)

    """Defining the derivative of the classifier function"""
    def softmax_deriv(self,neuron_value):
        return self.softmax(neuron_value)*(1 - self.softmax(neuron_value))

    def tanh(self, neuron_value):
        return (math.exp(neuron_value) - math.exp(-neuron_value))/(math.exp(neuron_value) + math.exp(-neuron_value))
    
    def tanh_deriv(self,neuron_value):
        #return 1.0 - math.tanh(neuron_value)**2
        return (1.0 - self.tanh(neuron_value))*(1.0 + self.tanh(neuron_value))

    def sigmoid(self,x):
        return 1 / (1 + math.exp(-x*1.0))

    def dSigmoid(self,x):
        return self.sigmoid(x)*(1.0-self.sigmoid(x))

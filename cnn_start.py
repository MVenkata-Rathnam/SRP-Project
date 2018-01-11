import numpy as np
import scipy as sio
import math
import time
import tkinter
from tkinter import *
from matplotlib import pyplot as plt

from cnn_load_data import *
from cnn_parameters import *
from cnn_core_model import *

class Main(object):
    def initialize_data(self):
        """Loading the data"""
        self.common_param = Parameters()
        self.training_set,self.validation_set,self.test_set,self.test_truth_set = load_data(self.common_param)

        """Initializing the layer parameters"""
        self.input_shape = self.training_set.shape
        self.common_param.initialize_layer_parameters(self.input_shape)
        self.layer_type = [{'conv_layer':ConvolutionLayer},{'pool_layer':PoolingLayer},{'fully_conn_layer':FullyConnectedLayer},{'output_layer':OutputLayer}]
        self.layer_activation = [{'conv_layer':'tanh'},{'pool_layer':'maximum'},{'fully_conn_layer':'tanh'},{'output_layer':'softmax'}]

        """Create a network with random parameters.
                Initializing the network"""
        self.net = Model(self.input_shape,self.layer_type,self.layer_activation,self.common_param)

        """Initializing the layers"""
        self.net.initialize_layers()

        """Field to check whether the network is trained or not"""
        self.trained_status = False

    def train_network(self):
        """Initializing the training parameters"""
        error = math.inf
        no_of_batches = self.input_shape[0] // self.common_param.batch_size
        epoch = 0
        self.start_time = time.time()
        error_list = []
        epoch_list = []

        """Checking feed forward
        for i in range(0,input_shape[0]): 
            net.feed_forward(training_set[i])
        """

        """Training the network with the training set"""
        self.training_set,self.validation_set = shuffle_in_order_of_class(self.training_set,self.validation_set, self.common_param)

        while (error > self.common_param.minimum_error and epoch < 10):
            error = 0.0
            epoch += 1
            error_batch_list = []
            batch_list = []
            #shuffle_in_unison(training_set,validation_set)
            #shuffle_in_order_of_class(training_set,validation_set)
            print("epoch %d initiated:" % epoch)
            for batch in range(0,self.input_shape[0],self.common_param.batch_size):
                #Calling the gradient descent method
                print ("batch %d" % batch)
                error += (1/self.common_param.batch_size)*self.net.gradient_descent(self.training_set[batch:batch+(self.common_param.batch_size)],self.validation_set[batch:batch+(self.common_param.batch_size)])
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
            #plt.plot(epoch_list,error_list)
            #plt.show()

    def test_network(self):
        """Testing the neural network"""
        correct_predict_count = 0
        index = count = 0
        start_time = time.time()
        self.common_param.final_result_set[:] = []
        #shuffle_in_unison(test_set,test_truth_set)
        #shuffle_in_order_of_class(test_set,test_truth_set)
        while (index < self.test_set.shape[0]):
            class_value = int(self.test_truth_set[index][0])
            if(class_value != 0):
                self.net.feed_forward(self.test_set[index])
                print ("Actual Output : " , self.test_truth_set[index])
                if(self.net.layers[-1].predicted_output == self.test_truth_set[index]):
                    correct_predict_count += 1
                count += 1
            else:
                self.common_param.final_result_set.append(0)
            index += 1

        self.end_time = time.time()
        """Calculating the accuracy"""
        accuracy = (correct_predict_count / count)*100
        self.net.accuracy = accuracy
        print ("Correct Predict count : " , correct_predict_count)
        print ("Accuracy : ",accuracy)
        print ("Time taken : ",round(self.end_time - self.start_time))

    def generate_gui(self):
        """Generating the GUI"""
        master=tkinter.Tk()
        label_colours=["gray","red","blue","green","yellow","purple","pink","cyan","brown4"]
        label_values=["Asphalt","Meadows","Gravel","Trees","Painted metal sheets","Bare soil","Bitumen","Self-blocking Bricks","Shadows"]
        legend=[]
        frame1=Frame(master)
        frame1.pack()
        frame2=Frame(master)
        frame2.pack()
        frame3=Frame(master)
        frame3.pack()
        frame4=Frame(master)
        frame4.pack()
        for i in range(len(label_colours)):
            legend.append(Label(frame1,text=label_values[i]+" - "+label_colours[i],bg=label_colours[i],fg="white"))

        for i in range(len(legend)):
           legend[i].pack(side=LEFT)

        text1=Label(frame2,text="Original Output",width=50)
        text1.pack(side=LEFT)
        text2=Label(frame2,text="CNN Output",width=50)
        text2.pack(side=LEFT)

        """Canvas for original output"""
        canvas1=Canvas(frame3,width=self.common_param.y_axis,height=self.common_param.x_axis)
        canvas1.pack(side=LEFT,padx=20)
        index = 0
        for i in range(self.common_param.x_axis):
            for j in range(self.common_param.y_axis):
                if(index < self.test_truth_set.shape[0]):
                    index_label=int(self.test_truth_set[index][0])
                    if(index_label != 0):
                        canvas1.create_rectangle(j,i,j+1,i+1,fill=label_colours[index_label - 1],width=0)
                index += 1

        """Canvas for tested output"""
        canvas2=Canvas(frame3,width=self.common_param.y_axis,height=self.common_param.x_axis)
        canvas2.pack(side=LEFT,padx=20)
        index = 0
        for i in range(self.common_param.x_axis):
            for j in range(self.common_param.y_axis):
                if(index < len(self.common_param.final_result_set)):
                    index_label = int(self.common_param.final_result_set[index])
                    if(index_label != 0):
                        canvas2.create_rectangle(j,i,j+1,i+1,fill =label_colours[index_label - 1], width=0)
                index += 1
        master.mainloop()

"""
main_object = Main()
main_object.initialize_data()

#Initializing the model
net = Model(main_object.input_shape,main_object.layer_type,main_object.layer_activation,main_object.common_param)

#Initializing the layers
net.initialize_layers()
main_object.train_network(net)
"""

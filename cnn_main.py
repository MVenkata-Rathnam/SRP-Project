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
#training_set,validation_set = shuffle_in_order_of_class(training_set,validation_set, common_param)

while (error > common_param.minimum_error and epoch < 25):
    error = 0.0
    epoch += 1
    error_batch_list = []
    batch_list = []
    #shuffle_in_unison(training_set,validation_set)
    #shuffle_in_order_of_class(training_set,validation_set)
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
index = 0
common_param.final_result_set[:] = []
#shuffle_in_unison(test_set,test_truth_set)
#shuffle_in_order_of_class(test_set,test_truth_set)
while (index < test_set.shape[0]):
    print (index)
    class_value = int(test_truth_set[index][0])
    if(class_value != 0):
        net.feed_forward(test_set[index])
        print ("Actual Output : " , test_truth_set[index])
        if(net.layers[-1].predicted_output == test_truth_set[index]):
            correct_predict_count += 1
    else:
        common_param.final_result_set.append(0)
    index += 1

"""Calculating the accuracy"""
accuracy = (correct_predict_count / index)*100
print ("Correct Predict count : " , correct_predict_count)
print ("Accuracy : ",accuracy)

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
canvas1=Canvas(frame3,width=common_param.y_axis,height=common_param.x_axis)
canvas1.pack(side=LEFT,padx=20)
index = 0
for i in range(common_param.x_axis):
    for j in range(common_param.y_axis):
        if(index < test_truth_set.shape[0]):
            index_label=int(test_truth_set[index][0])
            if(index_label != 0):
                canvas1.create_rectangle(j,i,j+1,i+1,fill=label_colours[index_label - 1],width=0)
        index += 1

"""Canvas for tested output"""
canvas2=Canvas(frame3,width=common_param.y_axis,height=common_param.x_axis)
canvas2.pack(side=LEFT,padx=20)
index = 0
for i in range(common_param.x_axis):
    for j in range(common_param.y_axis):
        if(index < len(common_param.final_result_set)):
            index_label = int(common_param.final_result_set[index])
            if(index_label != 0):
                canvas2.create_rectangle(j,i,j+1,i+1,fill =label_colours[index_label - 1], width=0)
        index += 1
master.mainloop()

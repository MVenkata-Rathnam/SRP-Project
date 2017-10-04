"""Importing the essential packages"""
import scipy.io as sio
import numpy as np

def load_data(common_param):
    """Defining the data set , ground truth file and header names"""
    data_set_file_name = "PaviaU"
    ground_truth_file_name = "PaviaU_gt"
    data_set_header_required = "paviaU"
    ground_truth_set_header_required = "paviaU_gt"

    """Loading the Dataset values"""
    data_set = sio.loadmat(data_set_file_name)

    """Loading the Ground Truth Values"""
    ground_truth_set = sio.loadmat(ground_truth_file_name)

    """Printing the values in the data_set
    for key,value in data_set.items():
            print (key)
            array = np.array(value)
            print (array.shape)
            #print ("Length : " + str(len(value)))
            print (value)
            print (type(value))
    """

    """ -> Deleting the items (eg. __header__, __globals__,__version__)
        -> Dictionary is changed to a numpy array"""
    data_set = data_set[data_set_header_required]
    ground_truth_set = ground_truth_set[ground_truth_set_header_required]
    data_set_shape_tuple = data_set.shape
    ground_truth_shape_tuple = ground_truth_set.shape

    """Initializing parameters"""
    total_sample_count = data_set_shape_tuple[0]*data_set_shape_tuple[1]
    training_sample_count = common_param.no_of_classes*common_param.count_of_each_class
    test_sample_count = total_sample_count - training_sample_count

    """Reshaping the data_set
    data_set_reshaped_set = data_set.reshape(data_set_shape_tuple[0]*data_set_shape_tuple[1],data_set_shape_tuple[2])

    Reshaping the ground_truth_set
    ground_truth_reshaped_set = ground_truth_set.reshape(ground_truth_shape_tuple[0]*ground_truth_shape_tuple[1])
    """
                                                         
    """Creating the training, validation and test sets"""
    training_set = np.ndarray(shape = (training_sample_count,data_set_shape_tuple[2]))
    validation_set = np.ndarray(shape = (training_sample_count,1))
    test_set = np.ndarray(shape = (test_sample_count,data_set_shape_tuple[2]))
    test_truth_set = np.ndarray(shape = (test_sample_count,1))
    count_set = np.zeros((1,common_param.no_of_classes+1))

    """Finding the class count
    for row in range(0,ground_truth_set.shape[0]):
        for column in range(0,ground_truth_set.shape[1]):
            count_set[0][ground_truth_set[row][column]] += 1

    print (count_set)
    """

    """Finding the class count --> Wrong answer
    for row in range(0,ground_truth_set.shape[0]):
        count_set[0][ground_truth_reshaped_set[row]] += 1
    print (count_set)
    """

    row_count_1 = 0
    row_count_2 = 0
    for row in range(0,ground_truth_set.shape[0]):
        for column in range(0,ground_truth_set.shape[1]):
            #print ((count_set[0][ground_truth_set[row][column]]))
            if(ground_truth_set[row][column] != 0 and count_set[0][ground_truth_set[row][column]] < common_param.count_of_each_class):
               count_set[0][ground_truth_set[row][column]] += 1
               training_set[row_count_1] = data_set[row][column]
               validation_set[row_count_1] = ground_truth_set[row][column]
               row_count_1 += 1
            else:
                test_set[row_count_2] = data_set[row][column]
                test_truth_set[row_count_2] = ground_truth_set[row][column]
                row_count_2 += 1
               
    """Checking the generated data 
    for row in range(0,common_param.no_of_classes*common_param.count_of_each_class):
        print (training_set[row][0] ," "),
        print (validation_set[row])
    """
    return training_set,validation_set,test_set,test_truth_set

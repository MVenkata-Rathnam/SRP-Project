"""Importing the essential packages"""
import scipy.io as sio
import numpy as np
import math

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
    common_param.x_axis = ground_truth_shape_tuple[0]
    common_param.y_axis = ground_truth_shape_tuple[1]
    total_sample_count = data_set_shape_tuple[0]*data_set_shape_tuple[1]
    training_sample_count = common_param.no_of_classes*common_param.count_of_each_class
    test_sample_count = total_sample_count

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

    """Scaling the reflectance values to -1.0 to +1.0"""
    
    """Finding the high and low values for scaling"""
    #max_reflectance_val = float(np.amax(data_set))
    #min_reflectance_val = float(np.amin(data_set))
    #print (max_reflectance_val)
    #print (min_reflectance_val)
    
    """Actual scaling
    data_set = np.float64(data_set)
    mean = np.mean(data_set)
    standard_deviation = np.std(data_set)
    for i in range(0,data_set_shape_tuple[0]):
        for j in range(0,data_set_shape_tuple[1]):
            for k in range(0,data_set_shape_tuple[2]):
                Scaling Formula 1
                Formula A = (A / max)*2 - 1
                data_set[i][j][k] = (data_set[i][j][k] / max_reflectance_val) * 2.0 - 1.0

                Scaling Formula 2
                Formula (b-a) + ((x - minx)/(maxx - minx)) + a
                data_set[i][j][k] = 2*((data_set[i][j][k] - min_reflectance_val)/(max_reflectance_val - min_reflectance_val)) - 1

                Normalization Formula 3 (Min - max normalization)
                Xnormalized = (Xcurrent - ((Xmax + Xmin)/2))/((Xmax - Xmin)/2)
                data_set[i][j][k] = ((data_set[i][j][k] - ((max_reflectance_val + min_reflectance_val)/2))/((max_reflectance_val - min_reflectance_val)/2))

                Guassian normalization
                Xnormalized = Xcurrent - mean / standard deviation
                data_set[i][j][k] = (data_set[i][j][k] - mean)/standard_deviation
                print (data_set[i][j][k])

    #print ("Mean : ", (sum_value / (data_set_shape_tuple[0]*data_set_shape_tuple[1]*data_set_shape_tuple[2])))
    print ("Mean : ", np.mean(data_set))
    print ("std : ",np.std(data_set))
    """

    """Normalizing the pixels locally within the band values"""
    data_set = np.float64(data_set)
    for i in range(0,data_set_shape_tuple[0]):
        for j in range(0,data_set_shape_tuple[1]):
            temp = data_set[i][j]
            min_val = float(np.amin(data_set[i][j]))
            max_val = float(np.amax(data_set[i][j]))
            for k in range(0,data_set_shape_tuple[2]):
                data_set[i][j][k] = ((data_set[i][j][k] - ((max_val + min_val)/2))/((max_val - min_val)/2))
                
    row_count_1 = 0
    row_count_2 = 0
    for row in range(0,ground_truth_set.shape[0]):
        for column in range(0,ground_truth_set.shape[1]):
            #print ((count_set[0][ground_truth_set[row][column]]))
            if(ground_truth_set[row][column] != 0.0):
                if(count_set[0][ground_truth_set[row][column]] < common_param.count_of_each_class):
                    count_set[0][ground_truth_set[row][column]] += 1
                    training_set[row_count_1] = data_set[row][column]
                    validation_set[row_count_1] = ground_truth_set[row][column]
                    row_count_1 += 1
            test_set[row_count_2] = data_set[row][column]
            test_truth_set[row_count_2] = ground_truth_set[row][column]
            row_count_2 += 1

    """Normalizing the training set
    mean = np.mean(training_set)
    standard_deviation = np.std(training_set)
    for row in range(0,training_set.shape[0]):
        for column in range(0,training_set.shape[1]):
            training_set[row][column] = (training_set[row][column] - mean)/standard_deviation
    """
    """Normalizing the test set
    mean = np.mean(test_set)
    standard_deviation = np.std(test_set)
    for row in range(0,test_set.shape[0]):
        for column in range(0,test_set.shape[1]):
            test_set[row][column] = (test_set[row][column] - mean)/standard_deviation
    """
    #print (row_count_2)   
    """Checking the generated data 
    for row in range(0,common_param.no_of_classes*common_param.count_of_each_class):
        print (test_set[row][0] ," "),
        print (test_truth_set[row])
    """
                   
    return training_set,validation_set,test_set,test_truth_set

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a, b

def shuffle_in_order_of_class(a, b, common_param):
    set_a = np.ndarray(shape = (a.shape[0],a.shape[1]))
    set_b = np.ndarray(shape = (b.shape[0],b.shape[1]))
    offset = 0
    for i in range(1, common_param.no_of_classes+1):
        individual_class_count = 1
        position = 0
        for j in range(0,a.shape[0]):
            if(individual_class_count < common_param.count_of_each_class):
                if(b[j] == i):
                    set_a[position*common_param.no_of_classes + offset] = a[j]
                    set_b[position*common_param.no_of_classes + offset] = b[j]
                    position += 1
                    individual_class_count += 1
        offset += 1
    return set_a,set_b

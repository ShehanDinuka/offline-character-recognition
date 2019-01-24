from mnist import MNIST
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from main import * # importing created module
import numpy as np
import math as m


# importing datasets to data_dir and extract them.
mndata = MNIST('./data')

"""datadir = "./data/"
mndata = MNIST(datadir)"""

# loading dataset
imgs, lbls = mndata.load_training()

examples = len(imgs)
pixel_size = m.sqrt(len(imgs[0]))
#print(examples, pixel_size)

# converting lists into arrays
images = np.array(imgs) # size - examples X pixel_size
labels = np.array(lbls) # size - examples X pixel_size
#print(images.shape)

#print(type(labels))

# spliting dataset into train, test, validation
X_train, images2, Y_train, labels2 = train_test_split(images, labels, test_size=0.4, random_state=0)
X_validate, X_test, Y_validate, Y_test = train_test_split(images2, labels2, test_size=0.5, random_state=0)

# printing the precentages for train, test, validation

# print(len(X_train)*100/len(images))
# print(len(X_test)*100/len(images))
# print(len(X_validate)*100/len(images))

#print(mndata.display(images[0]))
def preparing_labels_array(Y):
    """
    Argument:
    Y -- labels array(shape - no_of examples ) - simple array

    Returns:

    """

    no_of_examples = len(Y)
    prepared_Y = np.zeros((no_of_examples,10)).reshape(no_of_examples,10)
    example_no = 0
    for i in Y:
        prepared_Y[example_no,int(i)] = 1
        example_no += 1

    return prepared_Y

# Preparing L_layer_model arguments

layer_dims = [784, 60, 10] # creating Layer Dimension list
learning_rate = 0.01 # assigning learning rate
# Iterations
itr = 30
prepared_Y = preparing_labels_array(Y_train)
print(prepared_Y.shape)
parameters=L_layer_model(X_train.T, prepared_Y.T, layer_dims, learning_rate, itr , print_cost=False)
#parameters = L_layer_model( X_train, Y_train, layer_dims, learning_rate, itr)																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				print(parameters)
print(parameters)
from mnist import MNIST
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from main import * # importing created module
import numpy as np
import math as m


# importing datasets to data_dir and extract them.
mndata = MNIST('./data')

"""datadir = "./data/"
mndata = MNIST(datadir)"""

# loading dataset
imgs, lbls = mndata.load_training()

examples = len(imgs)
pixel_size = m.sqrt(len(imgs[0]))
#print(examples, pixel_size)

# converting lists into arrays
images = np.array(imgs) # size - examples X pixel_size
labels = np.array(lbls) # size - examples X pixel_size
#print(images.shape)

#print(type(labels))

# spliting dataset into train, test, validation
X_train, images2, Y_train, labels2 = train_test_split(images, labels, test_size=0.4, random_state=0)
X_validate, X_test, Y_validate, Y_test = train_test_split(images2, labels2, test_size=0.5, random_state=0)

# printing the precentages for train, test, validation

# print(len(X_train)*100/len(images))
# print(len(X_test)*100/len(images))
# print(len(X_validate)*100/len(images))

#print(mndata.display(images[0]))
def preparing_labels_array(Y):
    """
    Argument:
    Y -- labels array(shape - no_of examples ) - simple array

    Returns:

    """

    no_of_examples = len(Y)
    prepared_Y = np.zeros((no_of_examples,10)).reshape(no_of_examples,10)
    example_no = 0
    for i in Y:
        prepared_Y[example_no,int(i)] = 1
        example_no += 1

    return prepared_Y

# Preparing L_layer_model arguments

layer_dims = [784, 60, 10] # creating Layer Dimension list
learning_rate = 0.01 # assigning learning rate
# Iterations
itr = 30
prepared_Y = preparing_labels_array(Y_train)
print(prepared_Y.shape)
parameters=L_layer_model(X_train.T, prepared_Y.T, layer_dims, learning_rate, itr , print_cost=False)
#parameters = L_layer_model( X_train, Y_train, layer_dims, learning_rate, itr)																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				print(parameters)
print(parameters)

"""
    File containing functions for loading datasets 
    and saving them as theano shared variables
"""

import os
from os import listdir
from os.path import isfile, join
import gzip
import six.moves.cPickle as pickle
import numpy
import copy
import random
import theano
import theano.tensor as T
import scipy
from scipy.ndimage import rotate
from scipy.misc import face
from matplotlib import pyplot as plt
from PIL import Image


def load_mnist_data(dataset):
    ''' Loads the MNIST dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('Loading MNIST data...')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def unpickle(file):
    ''' 
        Unpickles 1 batch file (cifar data)
    '''

    # Open file and load it through pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict


def data_augment_rotate (x):
    '''
        Function for data augmentation,
        it rotates all the images and then returns
        the rasterized vector
    '''

    # Reshaping the input tensor of shape (no_of_examples, 3072) 
    # to (no_of_examples, 32, 32, 3) so that it is in proper image format
    shape_x = x.shape
    x = x.reshape(shape_x[0], 3, 32, 32)
    x = numpy.transpose(x, (0,2,3,1))

    # Loads one image, rotates it and saves it
    for i in range(shape_x[0]):

        img = Image.fromarray(x[i,:,:,:], 'RGB')

        # The angle by which the image has to be rotated is chosen 
        # randomly between -30 and 30 degrees
        angle = random.randint(1,60)-30
    
        rot = rotate(img, angle, reshape=False)
        x[i,:,:,:] = rot

    # Reshaping the tesnor of shape (no_of_examples, 32, 32, 3) 
    # to the rasterized format (no_of_examples, 3072) 
    x = numpy.transpose(x, (0,3,1,2))
    x = x.reshape(shape_x[0], 3072)
    return x


def data_augment_zoom_crop (x):
    '''
        Function for data augmentation,
        it zooms all the images and then crops the 
        image to take the center portion so that the
        size remains the same
    '''

    # Reshaping the input tensor of shape (no_of_examples, 3072) 
    # to (no_of_examples, 32, 32, 3) so that it is in proper image format
    shape_x = x.shape
    x = x.reshape(shape_x[0], 3, 32, 32)
    x = numpy.transpose(x, (0,2,3,1))    

    # Loads one image, zooms it, crops the center, and saves it
    for i in range(shape_x[0]):
        img = Image.fromarray(x[i,:,:,:], 'RGB')
        zoom = scipy.ndimage.zoom(img, 1.125, order=1)
        x[i,:,:,:] = zoom[2:-2, 2:-2, :]

    # Reshaping the tesnor of shape (no_of_examples, 32, 32, 3) 
    # to the rasterized format (no_of_examples, 3072) 
    x = numpy.transpose(x, (0,3,1,2))
    x = x.reshape(shape_x[0], 3072)
    return x


def load_cifar_data(dataset_folder):
    ''' Loads the CIFAR dataset

    :type dataset_folder: string
    :param dataset_folder: the path to the dataset folder (here CIFAR)
    '''

    #############
    # LOAD DATA #
    #############    

    # Join the realtive path to the dataset folder
    dataset_folder = os.path.join(
        os.path.split(__file__)[0],
        "..",
        "data",
        dataset_folder
    )

    print('Loading CIFAR data...')

    # Dividing the batch data into train and test
    train_filenames = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    test_filename = "test_batch"

    # Checking whether all the files are present in the folder or not
    filenames = copy.deepcopy(train_filenames)
    filenames.append(test_filename)
    for file in filenames:
        if ( not isfile( join(dataset_folder, file) ) ):
            print ('Some file missing in CIFAR folder specified. Exiting.')
            exit(1)

    # Initializing the dataset by adding one batch file
    # The data is being loaded to a dictionary "train_data_dict"
    initial_filename = train_filenames[0]
    train_filenames.remove(train_filenames[0])
    train_data_dict = {}
    batch_dict = unpickle( join(dataset_folder, initial_filename) )
    train_data_dict["data"] = batch_dict[b'data']
    train_data_dict["labels"] = batch_dict[b'labels']

    # Loading all the other training batch files to train_data_dict
    for file in train_filenames:
        batch_dict = unpickle( join(dataset_folder, file) )
        train_data_dict["data"] = numpy.vstack([train_data_dict["data"], 
                                                batch_dict[b'data']])
        train_data_dict["labels"].extend( batch_dict[b'labels'] )

    train_data_dict["labels"] = numpy.array(train_data_dict["labels"])
        
    # Creating a dictionary for loading test data and loading test batch in it
    test_data_dict = {}
    batch_dict = unpickle( join(dataset_folder, test_filename) )
    test_data_dict["data"] = batch_dict[b'data']
    test_data_dict["labels"] = numpy.array(batch_dict[b'labels'])


    ###################################################
    # Splitting training into training and validation #
    ###################################################

    train_data = numpy.column_stack( ( train_data_dict["data"], 
                                       train_data_dict["labels"] ) )   

    # Randomly shuffling the array to ensure no bias in datasets
    numpy.random.shuffle(train_data)
    first_val = True
    first_train = True

    # This vector signifies the number of examples
    # to be taken in validation data for each class
    # We take 1000 examples for each class 
    # (uniform distribution of classes)
    frequency = [1000]*10

    # The selected indices for validation datasets
    selected_index = []
    for i in range(train_data.shape[0]):
        if frequency[ train_data[i, 3072] ] > 0:
            frequency[ train_data[i, 3072] ] -= 1
            selected_index.append(i)    

    # The indices not in validation dataset would be used for training
    not_selected_index = [i for i in range(train_data.shape[0]) 
                                if i not in selected_index]

    # Partitioning the data into validation and training
    validation_data = numpy.array([ train_data[i] for i in selected_index ])
    training_data = numpy.array([ train_data[i] for i in not_selected_index ])


    ##################################################
    # Selecting only 5 classes out of the 10 classes #
    ##################################################

    # Selection in training data
    selected_indices = []
    for i in range( training_data.shape[0] ):
        if training_data[i, 3072] <=4 :
            selected_indices.append(i)
    training_data = numpy.array([ training_data[i] for i in selected_indices ])

    # Selecting only 5 classes out of the 10 classes
    # Selection in validation data
    selected_indices = []
    for i in range( validation_data.shape[0] ):
        if validation_data[i, 3072] <=4 :
            selected_indices.append(i)
    validation_data = numpy.array([ validation_data[i] for i in selected_indices ])

    # Selecting only 5 classes out of the 10 classes
    # Selection in test data
    selected_indices = []
    for i in range( len(test_data_dict["labels"]) ):
        if test_data_dict["labels"][i] <=4 :
            selected_indices.append(i)
    test_data_dict["data"] = numpy.array([ test_data_dict["data"][i] for i in selected_indices ])
    test_data_dict["labels"] = numpy.array([ test_data_dict["labels"][i] for i in selected_indices ])


    # Paritioning the image data from labels
    numpy.random.shuffle(training_data)
    numpy.random.shuffle(validation_data)
    basic_training_data_x = training_data[:, 0:3072]
    basic_training_data_y = training_data[:, 3072]
    validation_data_x = validation_data[:, 0:3072]
    validation_data_y = validation_data[:, 3072]
    testing_data_x = test_data_dict["data"]
    testing_data_y = test_data_dict["labels"]


    ########################
    ## Standardizing data ##
    ########################

    # Converting to the numpy arrays to type float
    basic_training_data_x = numpy.array(basic_training_data_x, dtype=numpy.float64)
    validation_data_x = numpy.array(validation_data_x, dtype=numpy.float64)
    testing_data_x = numpy.array(testing_data_x, dtype=numpy.float64)

    # Standardizing the data (Mean = 0, standard deviation = 0)
    # This is helpful in the convergance of gradient descent
    basic_training_data_x -= numpy.mean(basic_training_data_x , axis = 0)
    basic_training_data_x /= numpy.std(basic_training_data_x , axis = 0)
    validation_data_x -= numpy.mean(validation_data_x , axis = 0)
    validation_data_x /= numpy.std(validation_data_x , axis = 0)
    testing_data_x -= numpy.mean(testing_data_x , axis = 0)
    testing_data_x /= numpy.std(testing_data_x , axis = 0)

    #######################
    ## Data augmentation ##
    #######################

    # Creating a copy of the original data
    aug_data_x = copy.deepcopy(basic_training_data_x)
    aug_data_y = copy.deepcopy(basic_training_data_y)
    # Data rotation
    aug_data_x = data_augment_rotate (aug_data_x)
    # Stacking the augmented data
    training_data_x = numpy.vstack([basic_training_data_x, aug_data_x])
    training_data_y = numpy.concatenate([basic_training_data_y, aug_data_y])
    training_data = numpy.column_stack( ( training_data_x, training_data_y ) )   
    # Randomly shuffling data to get a good mix in batches
    numpy.random.shuffle(training_data)
    # Separating the image data and labels
    training_data_x = training_data[:, 0:3072]
    training_data_y = training_data[:, 3072]

    # Creating a copy of the original data
    aug_data_x = copy.deepcopy(basic_training_data_x)
    aug_data_y = copy.deepcopy(basic_training_data_y)
    # Data zoom crop
    aug_data_x = data_augment_zoom_crop (aug_data_x)
    # Stacking the augmented data
    training_data_x = numpy.vstack([training_data_x, aug_data_x])
    training_data_y = numpy.concatenate([training_data_y, aug_data_y])
    training_data = numpy.column_stack( ( training_data_x, training_data_y ) )   
    # Randomly shuffling data to get a good mix in batches
    numpy.random.shuffle(training_data)
    # Separating the image data and labels
    training_data_x = training_data[:, 0:3072]
    training_data_y = training_data[:, 3072]


    def shared_dataset(data_x, data_y, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(testing_data_x, testing_data_y)
    valid_set_x, valid_set_y = shared_dataset(validation_data_x, validation_data_y)
    train_set_x, train_set_y = shared_dataset(training_data_x, training_data_y)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

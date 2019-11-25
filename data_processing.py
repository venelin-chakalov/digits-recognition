import numpy as np
import pandas as pd
import pickle
import os
from sklearn import datasets

data, test_data_x, test_data_y = None, None, None
X, Y, W = [], [], []


# file operations section
def save_weights_to(file, weights):
    print('saving the generated values in %s...' % (file))
    with open(file, 'wb') as fp:
        pickle.dump(weights, fp)


def load_weights_from(file):
    weights = []
    try:
        with open(file, 'rb') as fp:
            weights = pickle.load(fp)
        print('found a training file...')
    except FileNotFoundError:
        print("no training file found...")
    return weights


def load_multiple_weights(base_filename, extension):
    multiple_weights = []
    for i in range(10):
        multiple_weights.append(load_weights_from(base_filename + str(i) + extension))
    return multiple_weights


def find_missing_weight(base_filename, extension):
    missing_weights = []
    for i in range(10):
        if not os.path.exists('./' + base_filename + str(i) + extension):
            missing_weights.append(i)


# end file operations section


# data processing section
def load_data(name):
    global data
    data = pd.read_csv(name)


def transform_data(transformation: dict, column):
    global data
    for key in transformation.keys():
        data.loc[data[column] == key, column] = transformation[key]


def remove_columns(columns):
    global data
    for column in columns:
        data = data.drop(column, axis=1)


def shuffle_data():
    global data
    data = data.sample(frac=1)


def to_np_array():
    global data
    data = np.array(data.values.tolist())


def split_data():
    test_data, train_data = data[:30], data[30:]
    return test_data, train_data


def add_dummy_feature():
    global X
    X = np.insert(X, 0, values=1, axis=1)


def load_weights():
    global W
    W = np.zeros((len(X[0]), 1))


# end data processing section


# this function serves as a pipeline for all data processing operations
def fit(name):
    global X, Y, W, test_data_x, test_data_y
    # first load the data
    print('loading the data...')
    load_data(name)
    print('start processing the data...')
    # remove the redundant columns, select the columns based on your dataset
    remove_columns(['id', 'Unnamed: 32'])
    # transform the results from Strings to Boolean representation, choose what to pick as a prediction goal
    transform_data({'B': 1, 'M': 0}, 'diagnosis')
    # shuffle the data
    shuffle_data()
    # transform it into np array
    to_np_array()

    # split data into training set and test set
    test_data, train_data = split_data()

    # separate the features and the results into two separate arrays
    X, Y = train_data[:, 1:], train_data[:, 0]

    # add x0 = 1 so we can use w0 as a free variable
    add_dummy_feature()

    # load initial values for W
    load_weights()

    test_data_x, test_data_y = test_data[:, 1:], test_data[:, 0]
    print('data processing is finished...')


# section for image processing
def normalize_number(number_goal):
    # since its binary classifier we need to know which value is the desired one and which is not
    # mark all the desired values as 1 and the others as 0
    for i in range(len(Y)):
        if Y[i] == number_goal:
            Y[i] = 1
        else:
            Y[i] = 0


# end section for image processing


# this function serves as a pipeline for all image processing operations
def fit_images(number_goal):
    global X, Y, test_data_x, test_data_y
    # here we download MNIST dataset with handwritten images
    # we use sklearn dataset module to access this data, alternatively we can download it manually
    print('loading image dataset...')
    digits = datasets.load_digits()

    # images are represented as 8*8 matrices, so we have to reshape it into a vector in order to ease the use of the h(X)
    # our new vector is in shape (64,1)
    images = digits.images.reshape((len(digits.images), -1))
    print('start preprocessing the data...')
    # split the data into features and labels
    X, Y = np.array(images), np.array(digits.target)
    # just for the test we select the first 10 records
    test_data_x, test_data_y, X, Y = X[:10], Y[:10], X[10:], Y[10:]

    # add x0 = 1 so we can use w0 as a free variable
    add_dummy_feature()

    # mark the number to predict as 1, the others as 0
    normalize_number(number_goal)

    # load initial values for W
    load_weights()
    print('data processing is finished')
# end image processing operations

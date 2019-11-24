import numpy as np
import pandas as pd
import pickle
from sklearn import datasets
from matplotlib import pyplot as plt

data, test_data_x, test_data_y = None, None, None
X, Y, W = [], [], []


def load_data(name):
    global data
    data = pd.read_csv(name)


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


def hypothesis(x, weights):
    z = np.dot(weights.T, x)

    y = 1 / (1 + np.exp(-z))
    y = np.minimum(y, 0.9999)  # Set upper bound
    y = np.maximum(y, 0.0001)  # Set lower bound
    return y


def cost_of(x, y, weights):
    # get the sum of difference between the predicted value and the real one
    sigma = hypothesis(x, weights) - y
    return sigma


def gradient(size, initial_weights):
    # we let gradient descent to iterate 300000 times in order to produce accurate results
    # i is counter variable which we use to access the current dataset
    i = 0
    weights = initial_weights
    # sample learning rate, choose one that fits nicely to your dataset
    learning_rate = 0.01
    for k in range(300000):
        if i == size - 1:
            i = 0
        # temp_W = weights[k] - (rate / 2*m) * derivative_of(-(ylog|h(x)| + (1-y)log|1-h(x)|)
        # temp_W = weights[k] - (rate / m) * ((h(X) - y) * X[k])
        # where k is the index of the current training example
        temp_weights = np.array(
            [weights[k] - (learning_rate * (cost_of(X[i], Y[i], weights) * (1 / size))) * X[i][k] for k in
             range(len(weights))])
        weights = temp_weights
        i += 1
    return weights


def predict(input_set):
    input_set = np.insert(input_set, 0, values=1, axis=0)

    # our threshold value is 0.5, if the probability is bigger than 0.5, we assume the result is correct
    return hypothesis(input_set, W) > 0.5


def normalize_number(number_goal):
    # since its binary classifier we need to know which value is the desired one and which is not
    # mark all the desired values as 1 and the others as 0
    for i in range(len(Y)):
        if Y[i] == number_goal:
            Y[i] = 1
        else:
            Y[i] = 0


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


def train_model(type_of_data):
    global W
    weights = []
    filename = ''
    if type_of_data == 'binary_classifier':
        desired_num = input('select a number that you want to compare to others: ')
        weights = load_weights_from("trained_images_" + desired_num + ".data")
        if len(weights) == 0:
            fit_images(int(desired_num))
            filename = "trained_images_" + desired_num + ".data"
    else:
        name_of_file = input('select the name of the file: ')
        weights = load_weights_from('trained_dataset.data')
        if len(weights) == 0:
            fit(name_of_file)
            filename = 'trained_dataset.data'

    if len(weights) == 0:
        print('searching for optimal weights...')
        weights = gradient(len(X), W)
        print('search done!')
        save_weights_to(filename, weights)
    else:
        weights = np.array(weights)

    W = weights


# sample function that checks whether the provided result is correct
def test_image_classifier():
    train_model('binary_classifier')
    num_3 = []
    print('loading the test data from a file...')
    try:
        with open('test_data_3.data', 'rb') as fp:
            num_3 = pickle.load(fp)
        print('data finished loading')
    except FileNotFoundError:
        print("no training file")

    print("is this the number 3? %s" % (predict(num_3)))
    plt.imshow(num_3.reshape(8, 8), interpolation='nearest')
    plt.show()


def test_tumor_classifier():
    train_model('normal')
    sample_x, sample_y = test_data_x[0], test_data_y[0]
    print('is benign? : predicted - %s, actual - %s' % (predict(sample_x), bool(sample_y)))


test_image_classifier()

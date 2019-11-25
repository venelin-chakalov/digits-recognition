import numpy as np
import pickle
from matplotlib import pyplot as plt

import data_processing as dp

data, test_data_x, test_data_y = None, None, None
X, Y, W = [], [], []


def init_values():
    global data, test_data_x, test_data_y, X, Y, W
    data, test_data_x, test_data_y = dp.data, dp.test_data_x, dp.test_data_y
    X, Y, W = dp.X, dp.Y, dp.W


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
        # temp_W = weights[k] - (rate / 2*m) * partial derivative_of(-(ylog|h(x)| + (1-y)log|1-h(x)|)
        # temp_W = weights[k] - (rate / m) * ((h(X) - y) * X[k])
        # where k is the index of the current training example
        temp_weights = np.array(
            [weights[k] - (learning_rate * (cost_of(X[i], Y[i], weights) * (1 / size))) * X[i][k] for k in
             range(len(weights))])
        weights = temp_weights
        i += 1
    return weights


def predict(input_set, weights):
    input_set = np.insert(input_set, 0, values=1, axis=0)

    # our threshold value is 0.5, if the probability is bigger than 0.5, we assume the result is correct
    return hypothesis(input_set, weights) > 0.5


def predict_multiclass(input_set):
    number = None
    for i in range(len(W)):
        if predict(input_set, W[i]):
            number = i
    return number


def fit_image_model(digit):
    dp.fit_images(digit)
    filename = "trained_images_" + digit + ".data"
    return filename


def fit_data_model(name_of_file):
    dp.fit(name_of_file)
    filename = 'trained_dataset.data'
    return filename


def generate_weights(filename):
    print('searching for optimal weights...')
    init_values()
    weights = gradient(len(X), W)
    print('search done!')
    dp.save_weights_to(filename, weights)
    return weights


def fit_missing_training_models(missing_weights, weights: list):
    for w in missing_weights:
        filename = fit_image_model(w)
        weights.insert(w - 1, generate_weights(filename))
    return weights


def train_model(type_of_data):
    global W, X, Y
    weights = []
    filename = ''
    if type_of_data == 'binary_classifier':
        desired_num = input('select a number that you want to compare to others: ')
        weights = dp.load_weights_from("trained_images_" + desired_num + ".data")
        if len(weights) == 0:
            filename = fit_image_model(int(desired_num))

    elif type_of_data == 'multiclass_classifier':
        weights = dp.load_multiple_weights("trained_images_", ".data")
        if len(weights) != 10:
            missing_weights = dp.find_missing_weight("trained_images_", ".data")
            weights = fit_missing_training_models(missing_weights, weights)

    else:
        name_of_file = input('select the name of the file: ')
        weights = dp.load_weights_from('trained_dataset.data')
        if len(weights) == 0:
            filename = fit_data_model(name_of_file)

    if len(weights) == 0:
        weights = generate_weights(filename)
    else:
        weights = np.array(weights)

    W = weights


# sample function that checks whether the provided result is correct
def test_image_classifier():
    train_model('multiclass_classifier')
    num_3 = []
    print('loading the test data from a file...')
    try:
        with open('test_data_3.data', 'rb') as fp:
            num_3 = pickle.load(fp)
        print('data finished loading')
    except FileNotFoundError:
        print("no training file")
    print('what is this number? %d' % (predict_multiclass(num_3)))
    # print("is this the number 3? %s" % (predict(num_3, W)))
    plt.imshow(num_3.reshape(8, 8), interpolation='nearest')
    plt.show()


def test_tumor_classifier():
    train_model('normal')
    sample_x, sample_y = test_data_x[0], test_data_y[0]
    print('is benign? : predicted - %s, actual - %s' % (predict(sample_x, W), bool(sample_y)))


test_image_classifier()

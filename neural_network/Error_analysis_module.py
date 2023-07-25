import numpy as np

# mean squared error loss function
def mse_loss(y_expected, y_activation):
    return np.mean(np.power(y_expected - y_activation, 2)) / 2

# binary cross entropy loss function
def logistic_loss(y_expected, y_activation):
    Loss = -y_expected * np.log(y_activation) - (1 - y_expected)*np.log(1-y_activation)
    Loss = np.sum(Loss) / np.size(y_expected) # This must be a float not matrix or vector
    return Loss

# mean squared error loss gradient: for back propagation
# this gradient function can also be used for the binary cross entropy loss because the math works out automatically for (specifically binary cross entropy) the only difference is the way of calculating the y_activation (which is specific to the activation function being used). 
def logistic_loss_prime(y_expected, y_activation):
    return (y_activation-y_expected) / ((y_activation)*(1-y_activation))

# I made a custom error calculation function for sigmoid function as the output layer for the network
def binary_error(y_expected, y_activation):
    i=0
    for y,yi in zip(y_activation, y_expected):
        if int(y) != int(yi):
            i+=1
    return i / np.size(y_expected)


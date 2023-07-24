import numpy as np

# mean squared error loss function
def mse_loss(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

# mean squared error loss gradient: for back propagation
# this gradient function can also be used for the binary cross entropy loss because the math works out automatically for (specifically binary cross entropy) the only difference is the way of calculating the y_pred. 
def cost_prime(y_true, y_pred):
    return 2 * (y_pred-y_true) / np.size(y_true)

def logistic_cost(y_true, y_pred):
    Loss = -y_true * np.log(y_pred) - (1 - y_true)*np.log(1-y_pred)
    Cost = np.sum(Loss) / np.size(y_true)
    return Cost

# I made a custom error calculation function for sigmoid function as the output layer for the network
def binary_error(y_true, y_pred):
    i=0
    for y,yi in zip(y_pred, y_true):
        if int(y) != int(yi):
            i+=1
    return i / np.size(y_true)


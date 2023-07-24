import Activation_module as AM
import Error_analysis_module as EM
import Layers_module as LM
import Data_handling_train as Data
import Data_handling_test as Data_t
import numpy as np
import math

# This the function applying forward propagation on all the layers and predicting the final output.
def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

# Training function
def train(network, err, cost_func, cost_prime, x_train, y_train, iters = 100, learning_rate = 0.01, verbose = True):
    y_predicted = np.zeros_like(y_train)

    # No of times the training must run
    for e in range(iters):
        error = 0
        cost = 0
        i=0
        # iterating over all the examples
        for x, y in zip(x_train, y_train):
            # forward propagation on all layers
            output = predict(network, x)

            # Since we are training a classification model
            classfied_output = np.zeros_like(output)
            # Here the assumption I took is that if the output generated via any binary classification activation function is greater than .5 consider this as 1 else 0.
            classfied_output += int(output >= .5)
            y_predicted[i] = classfied_output
            i+=1

            # error calculation on the final output 
            error += err(y, classfied_output)
            cost += cost_func(y, output)

            # backward propagation on all layers
            grad = cost_prime(y, output) # Applying Back propagation on the final output 
            # Backward propagation as it sounds runs in backward direction therefore the network is reversed
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        # printing the cost and iterations
        error /= len(x_train)
        if verbose and e % math.ceil(iters/10) == 0 or e == (iters-1):
            print(f"Iterations: {e}, cost: {cost:2.2f}")

    print(f"prediction complete on training set, accuracy: {(1-error)*100:2.2f}")
        

def test(network, err, x_test, y_test):
    y_predicted = np.zeros_like(y_test)
    error = 0
    i=0
    # iterating over all the examples
    for x, y in zip(x_test, y_test):
        # predicting the values for each layer
        output = predict(network, x)

        # Classified output(1 or 0)
        classfied_output = np.zeros_like(output)
        # Here the assumption I took is that if the output generated via any binary classification activation function is greater than .5 consider this as 1 else 0.
        classfied_output += int(output >= .5)
        y_predicted[i] = classfied_output
        i+=1

        # error calculation on the final output 
        error += err(y, classfied_output)

    # printing the error/accuracy
    error /= len(x_test)
    print(f"prediction complete on test set, accuracy: {(1-error)*100:2.2f}")


if __name__ == "__main__":
    '''This is the method of creating the Neural Network.'''
    # Usage: Dense(no of neurons in previous layer, no neurons to be generated in this particular layer, regularisation parameter) default value of the parameter is zero.
    Layer1_dense = LM.Dense(2, 3, 0.001)
    Layer1_Activation = AM.ReLU()
    Layer2_dense = LM.Dense(3, 1, 0.001)
    Layer2_Activation = AM.Sigmoid()

    Neural_Network = [
        # 1st Neuron Layer: Input Layer (Hidden Layer)
        Layer1_dense, 
        Layer1_Activation,

        # 2nd Neuron Layer: Output Layer
        Layer2_dense, 
        Layer2_Activation,
    ]

    train(Neural_Network, EM.binary_error, EM.logistic_cost, EM.cost_prime, Data.X_train, Data.Y_train, iters=100, learning_rate=.001)
    test(Neural_Network, EM.binary_error, Data_t.X_test, Data_t.Y_test)

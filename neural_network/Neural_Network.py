# IMPORTANT!!!!!: This is to note that I have implemented the Neural Network with only one hidden layer. Adding hidden layers will require to manually change the neural network instances through out the file but is possible.

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
def train(network, err, loss_func, loss_prime, x_train, y_train, iters = 100, learning_rate = 0.01, verbose = True):
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
            cost += loss_func(y, output)

            # backward propagation on all layers
            grad = loss_prime(y, output) # Applying Back propagation on the final output 
            # Backward propagation as it sounds runs in backward direction therefore the network is reversed
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        # printing the cost and iterations
        error /= len(x_train)
        if verbose and (e % math.ceil(iters/10) == 0 or e == (iters-1)):
            print(f"Iterations: {e}, cost: {cost:2.2f}")

    return (1-error)
        

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
    return 1-error


def Training_with_GridSearch(Lambda_arr, alpha_arr, X_train, Y_train, X_test,
                             Y_test):
    Model_data = []
    for Lambda_ in Lambda_arr:
        for alpha_ in alpha_arr:
            Layer1_dense = LM.Dense(2, 3, Lambda_)
            Layer1_Activation = AM.ReLU()
            Layer2_dense = LM.Dense(3, 1, Lambda_)
            Layer2_Activation = AM.Sigmoid()

            Neural_Network = [
                # 1st Neuron Layer: Input Layer (Hidden Layer)
                Layer1_dense, 
                Layer1_Activation,

                # 2nd Neuron Layer: Output Layer
                Layer2_dense, 
                Layer2_Activation,

            ]
            # Prediction on train set
            train_test_score = train(Neural_Network, EM.binary_error, EM.logistic_loss, EM.logistic_loss_prime, Data.X_train, Data.Y_train, iters=10, learning_rate=alpha_, verbose=False)
            # prediction on test set
            test_test_score = test(Neural_Network, EM.binary_error, Data_t.X_test, Data_t.Y_test)
            # Calculating mean_test_score for comparing the models
            mean_test_score = (train_test_score + test_test_score) / 2
            # tracking the trainer data
            data = {'Model': Neural_Network, 'mean_test_score': mean_test_score,
                    'train_test_score': train_test_score, 'test_test_score': test_test_score, 'alpha':alpha_, 'lambda':Lambda_}
            Model_data.append(data)
            
            

    # Finding the best model
    Best_model = Model_data[0]
    for data in Model_data:
        if Best_model['mean_test_score'] < data['mean_test_score']:
            Best_model = data

    # Reporting
    print("Best model Lambda parameter: ", Best_model['lambda'])
    print("Best model alpha parameter: ", Best_model['alpha'])
    print("mean_test_score of the Best model: ", Best_model['mean_test_score'])
    print("Score of the Best model on training set",
          Best_model['train_test_score'])
    print("Score of the Best model on test set", Best_model['test_test_score'])


if __name__ == "__main__":
    '''This is the method of implementing the Neural Network without GridSearch HYPERTUNING.'''
    '''
    Regularisation_param = .001
    # Usage: Dense(no of neurons in previous layer, no neurons to be generated in this particular layer, regularisation parameter) default value of the parameter is zero.
    Layer1_dense = LM.Dense(2, 3, Regularisation_param)
    Layer1_Activation = AM.ReLU()
    Layer2_dense = LM.Dense(3, 1, Regularisation_param)
    Layer2_Activation = AM.Sigmoid()

    Neural_Network = [
        # 1st Neuron Layer: Input Layer (Hidden Layer)
        Layer1_dense, 
        Layer1_Activation,

        # 2nd Neuron Layer: Output Layer
        Layer2_dense, 
        Layer2_Activation,
    ]

    train(Neural_Network, EM.binary_error, EM.logistic_loss, EM.logistic_loss_prime, Data.X_train, Data.Y_train, iters=100, learning_rate=.001)
    test(Neural_Network, EM.binary_error, Data_t.X_test, Data_t.Y_test)
    '''

    # Training with GridSearch hypertuning using 2 hyperparameters
    lambda_arr = [.1, .01, .001]
    alpha_arr = [.0001, .002, .001]

    Training_with_GridSearch(Lambda_arr=lambda_arr, alpha_arr=alpha_arr, X_train= Data.X_train, Y_train=Data.Y_train, X_test=Data_t.X_test,
                             Y_test=Data_t.Y_test)





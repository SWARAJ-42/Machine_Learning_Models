import numpy as np
import Data_handling_train as Data
import Data_handling_test as Data_t
import random
import math


class Logistic_regressor:
    def __init__(self, X_train, Y_train, alpha, num_iters, X_test=None, Y_test=None, lambda_val=0):
        # Training data
        self.X_train = X_train
        self.Y_train = Y_train

        # Testing data
        self.X_test = X_test
        self.Y_test = Y_test

        # declaring weights, bias and hyperparameters
        self.weights = None
        self.bias = None
        self.alpha = alpha
        self.num_iters = num_iters
        self.lv = lambda_val

    # Here we initialize the weights and bias with random values
    def generate_random_w_b(self):
        m, f = np.shape(self.X_train)
        weights = np.random.randn(f)
        bias = np.random.random()
        self.weights = weights
        self.bias = bias

    # computing sigmoid for all the examples in the dataset at the same time
    def compute_sigmoid(self, test=False):
        X_data = self.X_train
        Y_data = self.Y_train
        if test == True:
            X_data = self.X_test
            Y_data = self.Y_test

        # array of shape same as the output for storing the sigmoid output for each example
        sigmoid_output = np.zeros_like(Y_data)
        # Iterating over each example
        for x, s in zip(X_data, sigmoid_output):
            z = np.dot(self.weights, x) + self.bias
            s[0] = 1 / (1 + np.exp(-z))  # note: s is an single element array

        return sigmoid_output

    # computing the cost function: binary cross entropy
    def compute_cost(self):
        # No of examples
        m, f = np.shape(self.X_train)

        # computing and generating an array of sigmoid values of X_train
        sigmoid_output = self.compute_sigmoid()

        total_loss_without_regularisation = 0
        # Iterating over each example and calculating the loss(without regularisation) due to it
        for y, s in zip(self.Y_train, sigmoid_output):
            Loss = -y[0] * np.log(s[0]) - (1 - y[0]) * np.log(1-s[0])
            total_loss_without_regularisation += Loss

        cost = total_loss_without_regularisation / m

        # calculating the regularisation cost
        reg_cost = 0
        for j in range(f):
            reg_cost += self.weights[j] ** 2
        reg_cost *= self.lv / (2*m)

        cost += reg_cost

        return cost

    # computing the gradient of the weights and bias
    def compute_gradient(self):
        # No of examples
        m, n = np.shape(self.X_train)

        sigmoid_output = self.compute_sigmoid()

        dj_dw = np.zeros(self.weights.shape)
        dj_db = 0.

        for x, y, s in zip(self.X_train, self.Y_train, sigmoid_output):
            i = 0
            diff = s[0]-y[0]
            for f in x:
                dj_dw[i] += diff * f
                i += 1
            dj_db += diff

        # Applying regularisation
        for j in range(n):
            dj_dw[j] += (self.lv / m) * self.weights[j]

        return dj_dw, dj_db

    # updating the weights and biases
    def gradient_descent(self, verbose=True):
        # number of training examples
        m = len(self.X_train)

        J_history = []

        for i in range(self.num_iters):

            # Calculate the gradient and update the parameters
            dj_dw, dj_db = self.compute_gradient()

            # Update Parameters using w, b, alpha and gradient
            self.weights -= self.alpha * dj_dw
            self.bias -= self.alpha * dj_db

            # Save cost J at each iteration
            if i < 100000:      # prevent resource exhaustion
                cost = self.compute_cost()
                J_history.append(cost)

            # Print cost every at intervals 10 times or as many iterations if < 10
            if verbose and (i % math.ceil(self.num_iters/10) == 0 or i == (self.num_iters-1)):
                print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

        # return w and J,w history for graphing
        return self.weights, self.bias, J_history

    # training with the training set
    def Train(self, verbose=True):
        self.generate_random_w_b()
        self.compute_gradient()
        self.gradient_descent(verbose)

    # prediction with the test dataset
    def test(self, test=True):
        # number of training examples
        X_data = self.X_test
        Y_data = self.Y_test
        if test == False:
            X_data = self.X_train
            Y_data = self.Y_train

        m, n = X_data.shape
        p = np.zeros(m)

        # Calculate the prediction for this example
        f_wb = self.compute_sigmoid(test)

        # Apply the threshold
        # Loop over each example to check the error
        error = 0
        for p_v, y, f_wb_i in zip(p, Y_data, f_wb):
            p_v = (int)(f_wb_i > 0.5)
            if p_v != y[0]:
                error += 1
        if test:
            print(
                f"Prediction complete on test set, accuracy: {((1 - error / m) * 100):2.2f}")
        else:
            print(
                f"Prediction complete on training set, accuracy: {((1 - error / m) * 100):2.2f}")
        return p


if __name__ == "__main__":
    # Reshaping the data for making it fit for training on the LogisticRegression model
    X_train = np.reshape(Data.X_train, (Data.m, 2))
    Y_train = np.reshape(Data.Y_train, (Data.m, 1))

    # Reshaping the data for making it fit for testing on the LogisticRegression model
    X_test = np.reshape(Data_t.X_test, (Data_t.m, 2))
    Y_test = np.reshape(Data_t.Y_test, (Data_t.m, 1))

    Trainer = Logistic_regressor(X_train=X_train, Y_train=Y_train, X_test=X_test,
                                 Y_test=Y_test, alpha=.01, num_iters=1000, lambda_val=0)
    Trainer.Train(verbose=False)

    # Prediction on train set
    Trainer.test(test=False)
    # prediction on test set
    Trainer.test()

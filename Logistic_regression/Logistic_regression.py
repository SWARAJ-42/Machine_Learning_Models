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
        # No of examples and features
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
        return ((1 - error / m))


# GridSearch hypertuning implementation only for learning_rate and Regularisation parameter.
# Here this function is made independent to the Logistic regressor class and in this function I have manually called the logistic regressor class and did not pass it as an argument. Check line 201 on how to manually implement the model and this function will make a lot more sense. 
def Training_with_GridSearch(Lambda_arr, alpha_arr, X_train, Y_train, X_test,
                             Y_test):
    Model_data = []
    for Lambda_ in Lambda_arr:
        for alpha_ in alpha_arr:
            Trainer = Logistic_regressor(X_train=X_train, Y_train=Y_train, X_test=X_test,
                                         Y_test=Y_test, alpha=alpha_, num_iters=100, lambda_val=Lambda_)
            Trainer.Train(verbose=False)
            # Prediction on train set
            train_test_score = Trainer.test(test=False)
            # prediction on test set
            test_test_score = Trainer.test()
            # Calculating mean_test_score for comparing the models
            mean_test_score = (train_test_score + test_test_score) / 2
            # tracking the trainer data
            data = {'Model': Trainer, 'mean_test_score': mean_test_score,
                    'train_test_score': train_test_score, 'test_test_score': test_test_score}
            Model_data.append(data)

    # Finding the best model
    Best_model = Model_data[0]
    for data in Model_data:
        if Best_model['mean_test_score'] < data['mean_test_score']:
            Best_model = data

    # Reporting
    print("Best model parameters: ", "alpha=",
          Best_model['Model'].alpha, ", Lambda=", Best_model['Model'].lv)
    print("mean_test_score of the Best model: ", Best_model['mean_test_score'])
    print("Score of the Best model on training set",
          Best_model['train_test_score'])
    print("Score of the Best model on test set", Best_model['test_test_score'])


if __name__ == "__main__":
    # Reshaping the data for making it fit for training on the LogisticRegression model
    X_train = np.reshape(Data.X_train, (Data.m, 2))
    Y_train = np.reshape(Data.Y_train, (Data.m, 1))

    # Reshaping the data for making it fit for testing on the LogisticRegression model
    X_test = np.reshape(Data_t.X_test, (Data_t.m, 2))
    Y_test = np.reshape(Data_t.Y_test, (Data_t.m, 1))

    # Training without GridSearch hypertuning
    '''
    Trainer = Logistic_regressor(X_train=X_train, Y_train=Y_train, X_test=X_test,
                                 Y_test=Y_test, alpha=.01, num_iters=100, lambda_val=0)
    Trainer.Train(verbose=False)

    # Prediction on train set
    Trainer.test(test=False)
    # prediction on test set
    Trainer.test()
    '''
    # Training with GridSearch hypertuning using 2 hyperparameters
    Lambda_arr = [.01, .1, 1]
    alpha_arr = [.0001, .001, .01]
    Training_with_GridSearch(Lambda_arr=Lambda_arr, alpha_arr=alpha_arr, X_train=X_train, Y_train=Y_train, X_test=X_test,
                             Y_test=Y_test)

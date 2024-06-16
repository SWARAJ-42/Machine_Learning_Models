import numpy as np

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

# binary cross entropy loss function
def logistic_loss(y_expected, y_activation):
    Loss = -y_expected * np.log(y_activation) - (1 - y_expected)*np.log(1-y_activation)
    Loss = np.sum(Loss) / np.size(y_expected) # This must be a float not matrix or vector
    return Loss

# logistic loss gradient: for back propagation 
def logistic_loss_prime(y_expected, y_activation):
    return (y_activation-y_expected) / ((y_activation)*(1-y_activation))
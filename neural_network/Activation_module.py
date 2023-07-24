import numpy as np
import Layers_module as LM

# Here are a list of some activation functions we can use

# Sigmoid activation functions are used in binary classfications i.e. that is when the output only has two possibilities
# These are generally used in output layer of the neural network
class Sigmoid(LM.Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

# ReLU activation functions are the most common functions used in hidden layers
# This function is used instead of sigmoid or any other functions because these increase the efficiency of the neural network
class ReLU(LM.Activation):
    def __init__(self):
        def reLU(x):
            return np.maximum(0, x)

        def reLU_prime(x):
            return x >= 0
        super().__init__(reLU, reLU_prime)
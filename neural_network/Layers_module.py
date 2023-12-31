import numpy as np

# This is the basic template of the layers we will be making
# All the custom layers are going to inherit this basic template
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # The forward propagation step
    # The input matrix is dervived from applying forward propagation on the previous activation layer
    def forward(self, input):
        pass
    # The backward propagation step
    # The output_gradient matrix is dervived from applying back propagation on the upcomming activation layer
    def backward(self, output_gradient, learning_rate):
        pass


'''
    This the Dense layer where will be creating a layer of output
    Here the output is generated by applying linear activation function to the previous outputs
    Note: linear activation is not technically an activation function though but applying will take all the neuron values into account for forward propagation. The final output for this neuron layer will be processed on the next activation layer.
'''
class Dense(Layer):
    def __init__(self, input_size, output_size, lambda_value = 0):
        self.lambda_value = lambda_value # regularisation parameter for applying regularisation
        self.output_size = output_size
        self.input_size = input_size

        # generating random weights and biases for the particular layer
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
    
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias # returning the linear activation output for the corresponding activation layer
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        # Applying gradient descent on the weights and biases along with regularisation on weights only
        # To be specific this is the actual learning part on every layer because the weights and biases are being updated
        self.weights -= (learning_rate * weights_gradient + learning_rate * self.lambda_value / np.size(self.output_size) * self.weights)
        self.bias -= learning_rate * output_gradient 
        return np.dot(self.weights.T, output_gradient) # The output is for the previous layer


'''
    This is the Activation layer: This is an exclusive layer I made to apply the desired custom activations for simplicity purposes. 
                                Consider this as a part of the corresponding previous dense layer. In other words this layer and 
                                correspoding previous layer combine to form a single neuron layer on the network.
    Here we will apply the desired activation functions like Sigmoid, ReLU, Tanh for the derived linear output generated from the previous dense layer
    Note:- This is the base template of the Activation layer. The custom activation layers are in the Activation_module.py which the inherit this
        class and apply their forward prop and back prop function
'''
class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
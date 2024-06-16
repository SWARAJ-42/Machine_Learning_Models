import numpy
from scipy import signal

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

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth # This also the depth of each kernel
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1) # Here for this implemention we have taken stride to be 1 i.e. the kernel shift after each calculation 
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size) # input_depth = depth of each kernel; depth = number of kernels, this also shows the depth of the next input layer which will be the output layer of this present CONV layer

        # Parameters: filling random values to the fields of the kernels and bias
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input_parameters):
        self.input = input_parameters
        self.output = np.copy(self.biases) # We have already joined the biases before the cross-correlation computation
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
            return self.output
        
    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], 'valid')
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], 'full')
            
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient

        return input_gradient




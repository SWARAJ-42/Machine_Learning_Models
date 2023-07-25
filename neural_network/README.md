# User Guide
There are two main files on this project.
* <a href="./Neural_Network.py">Neural_Network.py</a> : My implementation of Neural Networks from scratch.
* <a href="./sklearn_neural_network.py">sklearn_neural_network.py</a> : My implementation of Neural Networks using MLPClassifier from the scikit-learn Library.

Simply run the python files to get the accuracy on both training set and test set by both implementations.

# Neural Networks

Artificial neural networks (ANNs) are comprised of a node layers, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network.

Working mechanism of a Neural Network is quite complicated to understand. Lets look into the bigger picture of how do we make a 
ANN:

- Step 1: Feed the input data into the neural network model.
- Step 2: The data flows from layer to layer until we have the output. (This step is called Forward propagation)
Once we have the output, we can calculate the error which is a scalar.
- step 3: Finally we can adjust a given parameter (weight or bias) by subtracting the derivative of the error with respect to the parameter itself. (This is the learning step and is called back propagation)
- We iterate through the process again. 

After a certain amount of iteration the model reaches a convergence with elegent parameters on every layer.

## Layers

Every layer in has certain common properties on its level in the <a href="./Neural_Network.py">Neural_Network.py</a>.
* It must be able to forward propagate the data it is fed with.
* It must be able to back propagate from the gradient of the data it generated for the next layer.

Here is the layout of a layer used in every custom layer:

```python
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
```

Every Neuron Layer consists of two sub layers but for simplicity we will consider them as separate Neuron Layers:
* Dense Layer

    Work of the Dense layer is only to handle linear regression on fed neurons. And generate new neurons. These are then sent to the next layer(Activation layer) to perform non linear operations.

    ```python
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
        ```

* Activation Layer template: 

    Activation Layer are used to perform non linear operations to the Dense Layer. As linear operations donot make the model to learn in classification. Recall the Logistic regression in previous task where we calculated the linear regression output then we calculated the sigmoid of the output. The sigmoid step is done by this layer on the neurons at the same time.

    ```python
        class Activation(Layer):
        def __init__(self, activation, activation_prime):
            self.activation = activation
            self.activation_prime = activation_prime

        def forward(self, input):
            self.input = input
            return self.activation(self.input)
        
        def backward(self, output_gradient, learning_rate):
            return np.multiply(output_gradient, self.activation_prime(self.input))
    ```
    * Here the `activation`, `activation_prime` are custom activation and gradient functions respectively. Written in <a href="./Activation_module.py">Activation_module.py</a>


## Forward Propagation

Lets understand how forward propagation works on every custom Layer.

* Dense Layer

    lets assume,

    * there are j neurons(values) $X = [x_1,x_2,x_3....x_j]$ on the previous layer of the selected Dense Layer.

    * there are i neurons $Y = [y_1, y_2...y_i]$ on the selected Dense Layer.

    forward propagation in this layer is given by,

    $$Y = W.X + B$$ 
    where $W = [w_1,w_2,w_3....w_j]$ and $B = [b]$ are the local parameters for every `Dense(Layer)` are generated randomly by the layer class initially and then are updated to train the model.

    Here is the code implemented:

    ```python
        class Dense(Layer):
            ...
                self.weights = np.random.randn(output_size, input_size)
                self.bias = np.random.randn(output_size, 1)
            ...
            ...
            def forward(self, input):
                self.input = input
                return np.dot(self.weights, self.input) + self.bias 
            ...
    ```

* Activation Layer:

    In the previous step we generated $Y$, now $Y$ will be the input for this corresponding Activation layer. Activation layer has same neurons as that of it previous Dense Layer because it only does the work of mapping. i.e.

    $$Y_a := Activation(Y)$$
    this implies,
    $$y_{a1} := Activation( y_1 )$$
    $$y_{a2} := Activation( y_2 )$$
    $$...$$
    $$...$$
    $$y_{aj} := Activation( y_j )$$

    Here is the code implemented:

    ```python
        class Activation(Layer):
        ...
            def forward(self, input):
                self.input = input
                return self.activation(self.input)
        ...
    ```
    Below is an example of a specific (sigmoid) activation function to be used:
    ```python
        class Sigmoid(LM.Activation):
            def __init__(self):
                def sigmoid(x):
                    return 1 / (1 + np.exp(-x))

                ...

                super().__init__(sigmoid, sigmoid_prime)
    ```

    After calculating the neuron values for the activation layer. These Values $Y_a$ is going to be the new $X$ for the next Dense layer. This goes on and on until the output layer is reached.

## Cost Function

After reaching the output layer on forward propagation we have to find the cost function on the output layer for the parameters $W$ and $B$ on every layers. We can recall the importance of cost function from the <a href="../Logistic_regression/README.md">Logistic Regression</a> task.

Since we are working on a classification model. We will be using sigmoid as the activation function of the output layer.

This is the cost function (binary cross entropy function) implementation for sigmoid from <a href="./Error_analysis_module.py">Error_analysis_module.py</a>:

```python
    # binary cross entropy loss function for single example.
    def logistic_cost(y_expected, y_activation):
        Loss = -y_expected * np.log(y_activation) - (1 - y_expected)*np.log(1-y_activation)
        Cost = np.sum(Loss) / np.size(y_expected)
        return Cost
```

## Back propagation











    


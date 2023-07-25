**_NOTE:_**  Before going through this module. Please take a look into <a href="https://github.com/SWARAJ-42/Machine_Learning_Models/blob/main/Logistic_regression/README.md">Logistic Regression</a> first because the concepts used there are also implemented here.

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

Here is the layout of a layer used in every custom layer in <a href="./Layers_module.py">Layers_module.py</a>:

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
* Dense Layer in <a href="./Layers_module.py">Layers_module.py</a>:

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

* Activation Layer template in <a href="./Layers_module.py">Layers_module.py</a>: 

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
    where $W = [W_1,W_2,W_3....W_i][w_1,w_2,w_3....w_j]$ and $B = [b_1, b_2...b_i]$ are the local parameters for every `Dense(Layer)` are generated randomly by the layer class initially and then are updated to train the model.

    Here is the code implemented in <a href="./Layers_module.py">Layers_module.py</a>:

    ```python
        class Dense(Layer):
        ...
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

    Here is the code implemented in <a href="./Layers_module.py">Layers_module.py</a>:

    ```python
        class Activation(Layer):
        ...
            def forward(self, input):
                self.input = input
                return self.activation(self.input)
        ...
    ```
    Below is an example of a specific (sigmoid) activation function to be used in <a href="./Activation_module.py">Activation_module.py</a>:
    ```python
        class Sigmoid(LM.Activation):
            def __init__(self):
                def sigmoid(x):
                    return 1 / (1 + np.exp(-x))

                ...

                super().__init__(sigmoid, sigmoid_prime)
    ```

    After calculating the neuron values for the activation layer. These Values $Y_a$ is going to be the new $X$ for the next Dense layer. This goes on and on until the output layer is reached.

## Loss Function

After reaching the output layer on forward propagation we have to find the loss function on the output layer for the parameters $W$ and $B$ on every layers. We can recall the importance of loss function from the <a href="https://github.com/SWARAJ-42/Machine_Learning_Models/blob/main/Logistic_regression/README.md">Logistic Regression</a> task.

Since we are working on a classification model. We will be using sigmoid as the activation function of the output layer.

**_NOTE:_** This is loss function we are talking about. Not cost function. Cost function is the summation of loss functions of all the examples.

This is the loss function (binary cross entropy function) implementation for sigmoid from <a href="./Error_analysis_module.py">Error_analysis_module.py</a>:

```python
    # binary cross entropy loss function for each example
    def logistic_loss(y_expected, y_activation):
        loss = -y_expected * np.log(y_activation) - (1 - y_expected)*np.log(1-y_activation)
        loss = np.sum(loss) / np.size(y_activation) # Since this must be a float not matrix or vector
        return loss
```

## Backward propagation

After finding the cost function we get the error. Now its time for training the parameters using backpropagation. We have to handle the backpropagation in 3 different types of places(layers).

* The output layer (this is the activation layer only but the backpropagation algorithm will be different here than any other activation).
* Other activation layers
* Dense layers

The order of backpropagation will opposite to that of <a href="#forward-propagation">forward propagation</a>.

Each Layer has to go through the following procedure:

$$...Layer \leftarrow \frac{\partial E}{\partial X} \leftarrow Layer(\text{W and B are updated}) \leftarrow \frac{\partial E}{\partial Y} \leftarrow Layer...$$

$Layer \text{ is represented as}:$
$$\frac{\partial E}{\partial X}\leftarrow Dense\leftarrow \frac{\partial E}{\partial Y}\leftrightarrow \frac{\partial E}{\partial X_a}\leftarrow Activation\leftarrow \frac{\partial E}{\partial Y_a}\leftrightarrow \frac{\partial E}{\partial Y}$$

here each $layer$ is the combination of a Dense Layer and its corresponding activation layer.

Backpropagation has 3 things we have to take care in each layer:
* Updating the $W$ i.e. $W := (1-\alpha\frac{\lambda}{i})W-\alpha\frac{\partial E}{\partial W}$.
* Updating the $B$ i.e. $B := B-\alpha\frac{\partial E}{\partial B}$.
* Calculating the local gradient for it to be input for previous layer $\frac{\partial E}{\partial X}$. 

Here $E$ is the cost function generated in final output layer and $X$ is the input values in the present layer or the output values of previous layer. $W$ and $B$ are matrix of parameters for each of next layer neuron specific to the present layer itself.

$\lambda$ is the regularisation hyperparameter set by us. $i$ is the number neurons in the next Dense Layer or we can say size of $Y$.

we can't directly calculate(program) $\frac{\partial E}{\partial W}$, $\frac{\partial E}{\partial B}$, $\frac{\partial E}{\partial X}$ but we can use chain rule i.e. we can calculate these using $\frac{\partial E}{\partial Y}$.

* Underlying equation of activation layer
$$\frac{\partial E}{\partial X_a} = \frac{\partial E}{\partial Y_a} *f^1(X_a) \tag{1}$$

Here is the code implementation in <a href="./Layers_module.py">Layers_module.py</a>:
```python
    class Activation(Layer):
    ...
    ...
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
```

* Underlying equation of corresponding dense layer
$$\frac{\partial E}{\partial W} = X^{transpose}\frac{\partial E}{\partial Y} \tag{1}$$
$$\frac{\partial E}{\partial B} = \frac{\partial E}{\partial Y} \tag{2}$$
$$\frac{\partial E}{\partial W} = \frac{\partial E}{\partial Y}W^{transpose} \tag{3}$$

Here is the code implementation in <a href="./Layers_module.py">Layers_module.py</a>:
```python
    class Dense(Layer):
    ...
    ...
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        # Applying gradient descent on the weights and biases along with regularisation on weights only
        # To be specific this is the actual learning part on every layer because the weights and biases are being updated
        self.weights -= (learning_rate * weights_gradient + learning_rate * self.lambda_value / np.size(self.output_size) * self.weights)
        self.bias -= learning_rate * output_gradient 
        return np.dot(self.weights.T, output_gradient) # The output is for the previous layer
```

We shall not go into the underlying mathematics of above equations. Keep in mind that this works out. To go deeper into the mathematics of neural networks refer <a href="https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65">here</a>.

We have handled backpropagation for every layer except for the output layer because for every layer $\frac{\partial E}{\partial Y}$ comes from next layer but for output layer there is no next layer. But recall, $E$ is calculated in this step only therefore we donot require $\frac{\partial E}{\partial Y}$. rather we can directly calculate $\frac{\partial E}{\partial X}$ for this layer.

Since we will be using sigmoid function for the final output layer. We will use binary cross entropy for $\frac{\partial E}{\partial X}$.

$$
\frac{\partial E}{\partial X}  = \frac{(f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - \mathbf{y}^{(i)})}{(f_{\mathbf{w},b}(\mathbf{x}^{(i)}))(1-f_{\mathbf{w},b}(\mathbf{x}^{(i)}))}f_{\mathbf{w},b}^1(\mathbf{x}^{(i)})
$$

Here is the code implementation of gradient of binary cross entropy loss function in <a href="./Error_analysis_module.py">Error_analysis_module.py</a> and <a href="./Activation_module.py">Activation_module.py</a> respectively:


for calculating $\frac{(f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - \mathbf{y}^{(i)})}{(f_{\mathbf{w},b}(\mathbf{x}^{(i)}))(1-f_{\mathbf{w},b}(\mathbf{x}^{(i)}))}$
```python
    def logistic_cost_prime(y_expected, y_activation):
    return (y_activation-y_expected) / ((y_activation)*(1-y_activation)) 
```
for calculating $f_{\mathbf{w},b}^1(\mathbf{x}^{(i)})$
```python
    class Sigmoid(LM.Activation):
        ...
        ...

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)
```

## Iteration

All these calculations program functions are defined for each example. Now we have to iterate the whole process for every example i.e. if we have $i$ examples and $j$ features $X = [X_1, X_2..., X_i]$ where $X_m=[x_1, x_2...x_j]$ then the above process is being done for only $X_m$.

After we are done with this, we will complete 1 training iteration. For reaching the convergence we have to repeat the above processes multiple times.

Here is the code implementation in <a href="./Neural_Network.py">Neural_Networks.py</a>:

```python
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
        if verbose and e % math.ceil(iters/10) == 0 or e == (iters-1):
            print(f"Iterations: {e}, cost: {cost:2.2f}")

    print(f"prediction complete on training set, accuracy: {(1-error)*100:2.2f}")
```

## Congratulations
We are done with our cool neural network.














    


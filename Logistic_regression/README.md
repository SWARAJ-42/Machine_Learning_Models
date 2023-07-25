# User Guide
There are two main files on this project.
* <a href="./Logistic_regression.py">Logistic_regression.py</a> : My implementation of Logistic regression from scratch.
* <a href="./sklearn_Logistic_regression.py">sklearn_Logistic_regression.py</a> : My implementation of Logistic regression using the scikit-learn Library.

Simply run the python files to get the accuracy on both training set and test set by both implementations.

# Logistic Regression
The principal use of the logistic regression is in classification assignments where the objective is to estimate the likelihood that a given instance belongs to a particular class. Regression is used because it uses a sigmoid function to estimate the probability for the given class using the output of a linear regression function as input. 

Logistic regression differs from linear regression in that it predicts the likelihood that an instance will belong to a specific class or not, whereas the output of the former is a continuous value that can be anything.

Here the task we are given is based on classification as the possibility of the output is either 0 or 1. So logistic regression can be our way to go with.

Lets take a look into the step by step method of applying it.

## Sigmoid function

The logistic regression model is represented as

$$f_{\mathbf{w},b}(x) = g(\mathbf{w}\cdot \mathbf{x} + b)$$
where function $g$ is the sigmoid function. The sigmoid function is defined as:

$$g(z) = \frac{1}{1+e^{-z}}$$

where,
$z = \mathbf{w}\cdot \mathbf{x} + b$ and $\mathbf{w}$ and $b$ are learnable parameters

* Code implementation in 
<a href="./Logistic_regression.py">Logistic_regression.py</a>:

```python
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
```

after calculating $g(z)$ we divide the value into 2 classes by a given threshold (we call it a decision boundary). Like for example :-

if $g(z) >= .5$ the output is $y = 1$ else $y = 0$

Here our main goal is to update $\mathbf{w}$ and $b$ till the decision boundary successfully classifies the data with minimum error.

## Genetrating Random parameters

In the first step we generate random initial values for $\mathbf{w}$ and $b$. 

* Code implementation in 
<a href="./Logistic_regression.py">Logistic_regression.py</a>:

```python
    def generate_random_w_b(self):
        m, f = np.shape(self.X_train)
        weights = np.random.randn(f)
        bias = np.random.random()
        self.weights = weights
        self.bias = bias
```

## Cost Function for the logistics regression

After this we calculate the cost function for the values of $\mathbf{w}$ and $b$.

A cost function is an important parameter that determines how well a machine learning model performs for a given dataset. It calculates the difference between the expected values and predicted values and represents it as a single real number.

Cost function $J(\mathbf{w},b)$ is given by,


$$ J(\mathbf{w},b) = \frac{1}{m}\sum_{i=0}^{m-1} \left[ loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) \right]$$

Here,

* $m$ is the number of examples for training.
* $loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)})$ is the cost for a single data point, which is specific to the prediction function i.e. in our case we are dealing with `sigmoid function`.

    * Therefore, 

$$loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = (-y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)$$

* Code implementation in 
<a href="./Logistic_regression.py">Logistic_regression.py</a>:
    * Note:- <a href="#regularisation">Regularisation</a> is used in this.
```python
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
```

## Gradient descent for logistic regression

This step is where the actual learning happens. Here by judging the cost function we optimize the parameters. Our goal in Machine Learning is to minimize $J(\mathbf{w},b)$. The lesser its value the more accurate the model gets.

After calculating $J(\mathbf{w},b)$ we update the parameters by using $(1)$ and $(2)$. Then we will repeat the process again by calculating the $J(\mathbf{w},b)$ until it minimizes.

$\text{repeat until minimization achieved} \lbrace$
$$w_j := w_j - \alpha\frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{1}$$
$$\text{for j := 0...n-1}$$
$$b := b - \alpha\frac{\partial J(\mathbf{w},b)}{\partial b} \tag{2}$$

$\rbrace$

where,

$$
\frac{\partial J(\mathbf{w},b)}{\partial b}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - \mathbf{y}^{(i)}) \tag{3}
$$

$$
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - \mathbf{y}^{(i)})x_{j}^{(i)} \tag{4}
$$

Here, 
* $\alpha$ is the learning rate (a hypertuning parameter). This value is generally defined by us.
* parameters $b$, $w_j$ (this is the parameter for each feature, remember $\mathbf{w}$ is a matrix of $w_j$ as elements) are all updated simultaniously.

Code implementation for $(3)$ and $(4)$ in 
<a href="./Logistic_regression.py">Logistic_regression.py</a>:
* Note:- <a href="#regularisation">Regularisation</a> is used in this.

```python

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
```

Code implementation for $(1)$ and $(2)$ in 
<a href="./Logistic_regression.py">Logistic_regression.py</a>:
```python
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

        # return w, b
        return self.weights, self.bias
```


## Regularisation

Every machine learning can face the problem of overfitting. Overfitting occurs when the model tries to overly fit the training data and performs very badly on testing data. 

Therefore we have what's called regularisation. This process reduces the influence of the parameters and helps in producing a more generalised model.

Regularisation is very helpful when there more than required features. Here the task we are given consists only 2 input features which makes regularisation quite ineffective. But still we will apply it on our model. We will apply regularisation to $\mathbf{w}$

The only change in math are following:

* The cost function changes to:

    $$J(\mathbf{w},b) = \frac{1}{m}\sum_{i=0}^{m-1} \left[ loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) \right] + \frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$$    

    * $\lambda$ is the regularised parameter(hyperparameter). which is also a value we decide. The more the $\lambda$ the more is the regularisation and vice versa.
    * $n$ is the number of features

* The gradient descent algorithm too changes:

    $\text{repeat until minimization achieved} \lbrace$
    $$w_j := (1 - \alpha \frac{\lambda}{m})w_j - \alpha\frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{3}$$
    $$\text{for j := 0...n-1}$$
    $$b := b - \alpha\frac{\partial J(\mathbf{w},b)}{\partial b} \tag{4}$$

    $\rbrace$
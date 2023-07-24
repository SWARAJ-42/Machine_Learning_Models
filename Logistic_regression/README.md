# Logistic Regression
The principal use of the logistic regression is in classification assignments where the objective is to estimate the likelihood that a given instance belongs to a particular class. Regression is used because it uses a sigmoid function to estimate the probability for the given class using the output of a linear regression function as input. 

Logistic regression differs from linear regression in that it predicts the likelihood that an instance will belong to a specific class or not, whereas the output of the former is a continuous value that can be anything.

Here the task we are given is based on classification as the possibility of the output is either 0 or 1. So logistic regression can be our way to go with.

Lets take a look into the step by step method of applying it.

## Sigmoid function

The logistic regression model is represented as

$$ f_{\mathbf{w},b}(x) = g(\mathbf{w}\cdot \mathbf{x} + b)$$
where function $g$ is the sigmoid function. The sigmoid function is defined as:

$$g(z) = \frac{1}{1+e^{-z}}$$

where,
$z = \mathbf{w}\cdot \mathbf{x} + b$ and $\mathbf{w}$ and $b$ are learnable parameters

after calculating $g(z)$ we divide the value into 2 classes by a given threshold (we call it a decision boundary). Like for example :-

if $g(z) >= .5$ the output is $y = 1$ else $y = 0$

Here our main goal is to update $\mathbf{w}$ and $b$ till the decision boundary successfully classifies the data with minimum error.

## Genetrating Random parameters

In the first step we generate random initial values for $\mathbf{w}$ and $b$.

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


## Gradient descent for logistic regression

This step is where the actual learning happens. Here by judging the cost function we optimize the parameters. Our goal in Machine Learning is to minimize $J(\mathbf{w},b)$. The lesser its value the more accurate the model gets.

After calculating $J(\mathbf{w},b)$ we update the parameters by using $(1)$ and $(2)$. Then we will repeat the process again by calculating the $J(\mathbf{w},b)$ until it minimizes.

$\text{repeat until minimization achieved} \; \lbrace$
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


## Regularisation

Every machine learning can face the problem of overfitting. Overfitting occurs when the model tries to overly fit the training data and performs very badly on testing data. 

Therefore we have what's called regularisation. This process reduces the influence of the parameters and helps in producing a more generalised model.

Regularisation is very helpful when there more than required features. Here the task we are given consists only 2 input features which makes regularisation quite ineffective. But still we will apply it on our model. We will apply regularisation to $\mathbf{w}$

The only change in math are following:

* The cost function changes to:

    $$ J(\mathbf{w},b) = \frac{1}{m}\sum_{i=0}^{m-1} \left[ loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) \right] + \frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$$

    * $\lambda$ is the regularised parameter(hyperparameter). which is also a value we decide. The more the $\lambda$ the more is the regularisation and vice versa.
    * $n$ is the number of features

* The gradient descent algorithm too changes:

    $\text{repeat until minimization achieved} \; \lbrace$
    $$w_j := (1 - \alpha \frac{\lambda}{m})w_j - \alpha\frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{3}$$
    $$\text{for j := 0...n-1}$$
    $$b := b - \alpha\frac{\partial J(\mathbf{w},b)}{\partial b} \tag{4}$$

    $\rbrace$










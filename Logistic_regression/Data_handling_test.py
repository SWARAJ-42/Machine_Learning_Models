import pandas as pd
import numpy as np

# extracted the desired data for training
data = pd.read_csv('./Data/ds1_test.csv')

# Converting it to numpy array for ease of applying desired math on the data
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

# Generating desired variable arrays for input features and outputs
X_test = np.zeros((m, 2))
Y_test = np.zeros((m, 1))

# Populating X_test and Y_test
i = 0
for datum in data:
    X_test[i] = datum[0:2]
    Y_test[i] = datum[2]
    i+=1

# Applying z-score normalization for evenly spreading of the data
Mean_X = np.mean(X_test, axis = 0)
SD_X = np.std(X_test, axis = 0)
X_test -= Mean_X
X_test /= SD_X

# Reshaping the data for making it fit for training on the neural network
X_test = np.reshape(X_test, (m, 2, 1))
Y_test = np.reshape(Y_test, (m, 1, 1))

# Test the shape of the data
if __name__ == "__main__":
    # print(X_test[0:10])
    # print(Y_test[0:10])
    m, f, v = np.shape(X_test)
    weights = np.random.randn( f, 1)
    print(X_test)

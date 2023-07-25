import pandas as pd
import numpy as np

# extracted the desired data for training
data = pd.read_csv('./Data/ds1_train.csv')

# Converting it to numpy array for ease of applying desired math on the data
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

# Generating desired variable arrays for input features and outputs
X_train = np.zeros((m, 2))
Y_train = np.zeros((m, 1))

# Populating X_train and Y_train
i = 0
for datum in data:
    X_train[i] = datum[0:2]
    Y_train[i] = datum[2]
    i+=1

# Applying z-score normalization for evenly spreading of the data
Mean_X = np.mean(X_train, axis = 0)
SD_X = np.std(X_train, axis = 0)
X_train -= Mean_X
X_train /= SD_X

# Reshaping the data for making it fit for training on Logistic regression
X_train = np.reshape(X_train, (m, 2))
Y_train = np.reshape(Y_train, (m, 1))


# Test the shape of the data
if __name__ == "__main__":
    print(X_train[0:10])
    print(Y_train[0:10])
   

from sklearn.neural_network import MLPClassifier
import numpy as np
import Data_handling_train as Data
import Data_handling_test as Data_t


def scikit_NN_model_execute(X_train, Y_train, X_test, Y_test):
    # Defining the neural network
    clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(3), random_state=1,
                solver='lbfgs', max_iter=1000)
    
    # Training the model
    clf = clf.fit(X_train, Y_train)

    # printing the shape of the neural network (I made it similar to my custom neural network)
    print("Shape of the neural network: ",[coef.shape for coef in clf.coefs_])

    # Reporting
    print("Accuracy on training set:", clf.score(X_train, Y_train)*100)
    print("Accuracy on test set:", clf.score(X_test, Y_test)*100)

if __name__ == "__main__":
    # Reshaping the data for making it fit for training on the sklearn_MLPClassifier model
    X_train = np.reshape(Data.X_train, (Data.m, 2))
    Y_train = np.reshape(Data.Y_train, (Data.m))

    # Reshaping the data for making it fit for testing on the sklearn_MLPClassifier model
    X_test = np.reshape(Data_t.X_test, (Data_t.m, 2))
    Y_test = np.reshape(Data_t.Y_test, (Data_t.m))
    scikit_NN_model_execute(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd
import Data_handling_train as Data
import Data_handling_test as Data_t


def scikit_NN_model_execute(X_train, Y_train, X_test, Y_test):
    # Defining the neural network
    NN_model = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(3), random_state=1,
                solver='lbfgs', max_iter=1000)
    
    # Applying hypertuning using RandomizedSearch to avoid over computation
    NN_model = RandomizedSearchCV(NN_model, {
        'alpha': [.1, .01, .0001], # learning rate parameter
        'solver': ['adam', 'lbfgs'], # Learning optimizers
        'hidden_layer_sizes':[(2),(3),(4)] # Neural network structure
    }, cv=3, n_iter=5, return_train_score=False)
    
    # training the model with the train set
    NN_model.fit(X_train, Y_train)
    df = pd.DataFrame(NN_model.cv_results_)

    # Reporting
    print(df[['param_alpha', 'param_solver', 'param_hidden_layer_sizes','mean_test_score']])
    print(f"Best model: ",NN_model.best_estimator_)
    print("mean_test_score of the Best model: ", NN_model.best_score_)
    print("Score of the Best model on training set", NN_model.best_estimator_.score(X_train, Y_train))
    print("Score of the Best model on test set", NN_model.best_estimator_.score(X_test, Y_test))

if __name__ == "__main__":
    # Reshaping the data for making it fit for training on the sklearn_MLPClassifier model
    X_train = np.reshape(Data.X_train, (Data.m, 2))
    Y_train = np.reshape(Data.Y_train, (Data.m))

    # Reshaping the data for making it fit for testing on the sklearn_MLPClassifier model
    X_test = np.reshape(Data_t.X_test, (Data_t.m, 2))
    Y_test = np.reshape(Data_t.Y_test, (Data_t.m))
    scikit_NN_model_execute(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
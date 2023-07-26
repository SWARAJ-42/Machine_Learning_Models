from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import Data_handling_train as Data
import Data_handling_test as Data_t
from sklearn.model_selection import GridSearchCV

def sklearn_logistic_reg_execute(X_train, Y_train, X_test, Y_test):
    # Initializing the model with default args
    '''
        LOGISTIC REGRESSION MODEL WITH DEFAULT ARGUMENTS
        ************************************************

        LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                    warm_start=False)
    '''
    lr_model = LogisticRegression()
    # Applying hypertuning
    lr_model = GridSearchCV(lr_model, {
    'C': [1, 10, 20], # regularisation parameter
    'tol': [.1, .01, .0001] # tolerance parameter
    }, cv=3, return_train_score=False)

    # training the model with the train set
    lr_model.fit(X_train, Y_train)
    df = pd.DataFrame(lr_model.cv_results_)

    # Reporting
    print(df[['param_C','param_tol','mean_test_score']])
    print(f"Best model: ",lr_model.best_estimator_)
    print("mean_test_score of the Best model: ", lr_model.best_score_)
    print("Score of the Best model on training set", lr_model.best_estimator_.score(X_train, Y_train))
    print("Score of the Best model on test set", lr_model.best_estimator_.score(X_test, Y_test))

if __name__ == "__main__":
    # Reshaping the data for making it fit for training on the LogisticRegression model
    X_train = np.reshape(Data.X_train, (Data.m, 2))
    Y_train = np.reshape(Data.Y_train, (Data.m))

    # Reshaping the data for making it fit for testing on the LogisticRegression model
    X_test = np.reshape(Data_t.X_test, (Data_t.m, 2))
    Y_test = np.reshape(Data_t.Y_test, (Data_t.m))

    sklearn_logistic_reg_execute(X_train, Y_train, X_test, Y_test)
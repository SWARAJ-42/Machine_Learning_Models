from sklearn.linear_model import LogisticRegression
import numpy as np
import Data_handling_train as Data
import Data_handling_test as Data_t

def sklearn_logistic_reg_execute():
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

    # training the model with the train set
    lr_model.fit(X_train, Y_train)

    # predict the train set with the test set
    y_pred_train = lr_model.predict(X_train)
    # predict the test_set with the test set
    y_pred_test = lr_model.predict(X_test)

    # Reporting
    print("Accuracy on training set:", lr_model.score(X_train, Y_train)*100)
    print("Accuracy on test set:", lr_model.score(X_test, Y_test)*100)

if __name__ == "__main__":
    # Reshaping the data for making it fit for training on the LogisticRegression model
    X_train = np.reshape(Data.X_train, (Data.m, 2))
    Y_train = np.reshape(Data.Y_train, (Data.m))

    # Reshaping the data for making it fit for testing on the LogisticRegression model
    X_test = np.reshape(Data_t.X_test, (Data_t.m, 2))
    Y_test = np.reshape(Data_t.Y_test, (Data_t.m))

    sklearn_logistic_reg_execute()
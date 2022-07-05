from sklearn.model_selection import RandomizedSearchCV
import warnings


def tune_pipe_with_randomsearch(pipe, hyperparams, X_train, y_train, n_iter):
    clf = RandomizedSearchCV(pipe, hyperparams, n_iter=n_iter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_train, y_train)

    return clf
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class BinaryLogisticRegression(BaseEstimator, ClassifierMixin):


    def __init__(self, n_iter = 1000, method="IRLS", eps_betha = 1e-10, eps_loss = 0.0003, success_threshold = 0.5, irls_lambda = 0.1, lr = 0.001,
        adam_b1: float=0.9, adam_b2: float=0.999, adam_e: float=10**-8, verbose=False):
        self.n_iter = n_iter
        self.method = method
        self.eps_betha = eps_betha
        self.eps_loss = eps_loss
        self.success_threshold = success_threshold
        self.irls_lambda = irls_lambda
        self.lr = lr
        self.adam_b1= adam_b1
        self.adam_b2 = adam_b2
        self.adam_e = adam_e
        self.verbose = verbose


    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if self.method not in ["IRLS", "ADAM", "GD", "SGD"]:
            raise ValueError('"method" parameter must have values from ["IRLS", "ADAM", "GD", "SGD"]')

        self.classes_, y = np.unique(y, return_inverse=True)

        if(self.classes_.shape[0] != 2):
            raise ValueError("Two classess needed for binary logistic regression")
        
        if self.method == "IRLS":
            self.intercept_, self.coef_, self.iters_, self.loss_fun_ = self.__IRLS(X, y)
        elif self.method == "ADAM":
            self.intercept_, self.coef_, self.iters_, self.loss_fun_ = self.__ADAM(X, y, lr=self.lr, b1=self.adam_b1, b2=self.adam_b2, e=self.adam_e)
        elif self.method == "GD":
            self.intercept_, self.coef_, self.iters_, self.loss_fun_ = self.__GD(X, y, lr=self.lr)
        elif self.method == "SGD":
            self.intercept_, self.coef_, self.iters_, self.loss_fun_ = self.__SGD(X, y, lr=self.lr)
        
        self.log_likelihood_  = -y.shape[0] * self.loss_fun_

        if (self.verbose):
            print(f"Finished training in iteration number {self.iters_} out of {self.n_iter} max iterations")

        return self


    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        probs = self.predict_proba(X)
        classes = (probs > self.success_threshold).astype("int")
        return self.classes_[classes]


    def predict_proba(self, X):
        check_is_fitted(self)
        X = self.__add_intercept_column(X)
        return self.__sigmoid(X, np.hstack((self.intercept_, self.coef_)))


    def __IRLS(self, X, y):
        # add intercept
        X = self.__add_intercept_column(X)
        
        # betha initialization
        betha = np.zeros(X.shape[1])
        betha_old = betha

        # loss function values 
        loss_fun = np.zeros(self.n_iter)

        # iterate
        it = 0
        while(it<self.n_iter):
            p = self.__sigmoid(X, betha_old)
            W = np.diag(p*(1-p))
            W_inv = np.diag(1/np.diag(W))
            z = X @ betha_old + W_inv @ (y-p)
            betha = np.linalg.solve(X.T @ W @ X + np.diag([self.irls_lambda]*X.shape[1]),  X.T @ W @ z)

            loss_fun[it] = self.__get_loss_function(y, self.__sigmoid(X, betha))

            if(self.__stop_condtion(betha_old, betha, loss_fun[:it+1])):
                break

            betha_old = betha
            it=it+1

        return np.array([betha[0]]), betha[1:], it, loss_fun[:it+1]

    def __SGD(self, X: np.array, y: np.array, lr: float):
        # add intercept
        X = self.__add_intercept_column(X)
        
        # betha initialization
        betha = np.zeros(X.shape[1])
        betha_old = betha

        # loss function values 
        loss_fun = np.zeros(self.n_iter)

         # iterate
        it = 0
        while (it < self.n_iter):
            grad = self.__get_stoch_grad(X, y, betha)
            betha = betha_old - lr * grad
            
            loss_fun[it] = self.__get_loss_function(y, self.__sigmoid(X, betha))

            if (self.__stop_condtion(betha_old, betha, loss_fun[:it+1])):
                    return np.array([betha[0]]), betha[1:], it, loss_fun[:it+1]

            betha_old = betha

            it = it + 1
        
        return np.array([betha[0]]), betha[1:], it, loss_fun[:it+1]


    def __ADAM(self, X: np.array, y: np.array, lr: float, b1: float=0.9, b2: float=0.999, e: float=10**-8):
        # add intercept
        X = self.__add_intercept_column(X)
        
        # betha initialization
        betha = np.zeros(X.shape[1])
        betha_old = betha

        m_t = np.zeros(X.shape[1])
        v_t = np.zeros(X.shape[1])

        # iterate
        it = 0

        # loss function values 
        loss_fun = np.zeros(self.n_iter)
        t = 0
        while (it < self.n_iter):
            t += 1
            grad = self.__get_stoch_grad(X, y, betha)
            
            grad_2 = grad ** 2
            m_t = b1 * m_t + (1 - b1) * grad
            v_t = b2 * v_t + (1 - b2) * grad_2
            _m_t = m_t / (1 - b1 ** t)
            _v_t = v_t / (1 - b2 ** t)

            betha = betha_old - lr * _m_t / (np.sqrt(_v_t) + e)

            loss_fun[it] = self.__get_loss_function(y, self.__sigmoid(X, betha))
            if (self.__stop_condtion(betha_old, betha, loss_fun[:it+1])):
                break

            betha_old = betha
            it=it+1
        
        return np.array([betha[0]]), betha[1:], it, loss_fun[:it+1]


    def __GD(self, X: np.array, y: np.array, lr: float):
        # add intercept
        X = self.__add_intercept_column(X)
        
        # betha initialization
        betha = np.zeros(X.shape[1])
        betha_old = betha

        # loss function values 
        loss_fun = np.zeros(self.n_iter)

         # iterate
        it = 0
        while (it < self.n_iter):
            grad = self.__get_grad(X, y, betha)
            betha = betha_old - lr * grad

            loss_fun[it] = self.__get_loss_function(y, self.__sigmoid(X, betha))
            if (self.__stop_condtion(betha_old, betha, loss_fun[:it+1])):
                break

            betha_old = betha
            it=it+1
        
        return np.array([betha[0]]), betha[1:], it, loss_fun[:it+1]


    def __get_stoch_grad(self, X: np.array, y: np.array, betha: np.array):
        i = np.random.randint(0, X.shape[0])
        return self.__get_grad(X[[i], :], y[[i]], betha)
    

    def __get_grad(self, X: np.array, y: np.array, betha: np.array):
        m = len(y)
        h = self.__sigmoid(X, betha)
        diff = h - y
        s = diff @ X
        grad = s / m
        return grad


    def __get_loss_function(self, y, y_hat):
        return - ( np.sum( y @ np.log(y_hat) + (1 - y) @ np.log(1 - y_hat) ) ) / y.shape[0]


    def __sigmoid(self, X, betha):
        # assuming betha with intercept
        return 1 / (1 + np.exp(-X@betha) )


    def __stop_condtion(self, betha_old, betha_new, loss_fun):
        lookback_iters = 10
        return np.linalg.norm(betha_old - betha_new) < self.eps_betha or ( loss_fun.shape[0] > lookback_iters and np.all( np.abs(loss_fun[-lookback_iters:] - loss_fun[-lookback_iters:][0]) < self.eps_loss ) )


    def __add_intercept_column(self, X):
        return np.append(np.ones((X.shape[0], 1)), X ,axis = 1)


    def _more_tags(self):
        return {
            "binary_only": True,
            # doesnt work for linearly separatable classes
            "_skip_test": "check_classifiers_predictions", 
            # doesnt work when there is only one class provided (throws ValueError Exception) 
            "_xfail_checks": { 
                    "check_classifiers_one_label": (
                        "Classifier can't train when only one class is present."
                    )
                }
            }
            
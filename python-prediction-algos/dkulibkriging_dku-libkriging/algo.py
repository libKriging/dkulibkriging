# This file is the actual code for the kriging algorithm
from dataiku.doctor.plugins.custom_prediction_algorithm import BaseCustomPredictionAlgorithm

from sklearn.base import BaseEstimator
import pylibkriging as lk
import numpy as np
import warnings

class KrigingEstimator(BaseEstimator):
    
    def __init__(self, kernel="matern3_2" ,regmodel = "constant" ,normalize = False ,optim = "BFGS" ,objective = "LL" ,noise = None ,parameters = None):
        self.kernel = kernel
        self.regmodel = regmodel
        self.normalize = normalize
        self.optim = optim
        self.objective = objective
        self.noise = noise
        self.parameters = parameters
        if self.parameters is None:
            self.parameters = {}
        if not self.noise is None:
            warnings.warn("noise type is:", str(type(self.noise)))
            if type(self.noise) is int:
                self.noise = float(self.noise)
            if (type(self.noise) is float) & (self.noise == 0.0):
                self.noise = None
        if self.noise is None:
            self.kriging = lk.Kriging(self.kernel)
        elif type(self.noise) is float: # homoskedastic user-defined "noise"
            self.kriging = lk.NoiseKriging(self.kernel)
        else:
            raise Exception("noise type not supported:", type(self.noise))
        warnings.warn(self.kriging.summary())
        
    def fit(self, X, y):
        y = y.values
        ## for debug:
        #np.savetxt("y.csv", y, delimiter=",")
        #np.savetxt("X.csv", X, delimiter=",")
        # for (later) pickling:
        self.X_train = X
        self.y_train = y
        if self.noise is None:
            self.kriging.fit(y, X, self.regmodel, self.normalize, self.optim, self.objective, self.parameters)      
        elif type(self.noise) is float: # homoskedastic user-defined "noise"
            self.kriging.fit(y, np.repeat(self.noise, y.size), X, self.regmodel, self.normalize, self.optim, self.objective, self.parameters)
        else:
            raise Exception("noise type not supported:", type(self.noise))
        warnings.warn(self.kriging.summary())
    
    def predict(self, X): # just return mean predicted :(
        return self.kriging.predict(X, False, False, False)[0][:,0]
    
    # Unused for now:
    #def sample_y(self, X, n_samples = 1, random_state = 0):
    #    return self.kriging.simulate(nsim = n_samples, seed = random_state, x = X)
    #
    #def log_marginal_likelihood(self, theta=None, eval_gradient=False):
    #    if theta is None:
    #        return self.kriging.logLikeliHood()
    #    else:
    #        return self.kriging.logLikeliHoodFun(theta, eval_gradient)
    
    # hack to support/circum. pickling
    def __getstate__(self):
        state = {'X_train': self.X_train, 'y_train':self.y_train}
        return state
    def __setstate__(self, newstate):
        self.kriging.fit(newstate['y_train'], newstate['X_train'], self.regmodel, self.normalize, self.optim, self.objective, self.parameters)
        

class CustomPredictionAlgorithm(BaseCustomPredictionAlgorithm):    
    """
        Class defining the behaviour of `kriging` algorithm:
        - how it handles parameters passed to it
        - how the estimator works

        Example here defines an Adaboost Regressor from Scikit Learn that would work for regression
        (see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)

        You need to at least define a `get_clf` method that must return a scikit-learn compatible model

        Args:
            prediction_type (str): type of prediction for which the algorithm is used. Is relevant when 
                                   algorithm works for more than one type of prediction.
                                   Possible values are: "BINARY_CLASSIFICATION", "MULTICLASS", "REGRESSION"
            params (dict): dictionary of params set by the user in the UI.
    """
    
    def __init__(self, prediction_type=None, params=None):        
        self.clf = KrigingEstimator(kernel =     params.get("kernel","matern3_2"),
                                    regmodel =   params.get("regmodel","constant"),
                                    normalize =  params.get("normalize",False),
                                    optim =      params.get("optim","BFGS"),
                                    objective =  params.get("objective","LL"),
                                    noise =      params.get("noise",None),
                                    parameters = None)
        super(CustomPredictionAlgorithm, self).__init__(prediction_type, params)
    
    def get_clf(self):
        """
        This method must return a scikit-learn compatible model, ie:
        - have a fit(X,y) and predict(X) methods. If sample weights
          are enabled for this algorithm (in algo.json), the fit method
          must have instead the signature fit(X, y, sample_weight=None)
        - have a get_params() and set_params(**params) methods
        """
        return self.clf
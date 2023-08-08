from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Product, WhiteKernel, RationalQuadratic, Exponentiation
import numpy as np
import warnings
import re

class GaussianProcessShapeRegressor:
    def __init__(self, alpha=0.001, n_restarts_optimizer=20):
        self.kernel = RBF(length_scale=(1e-3, 1e-3), length_scale_bounds=(1e-2, 0.8)) + \
                      WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-7, 1))
        self.model = GaussianProcessRegressor(kernel=self.kernel, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer)
        
    def add_data_point(self, x_new, y_new):
        try:
            X = self.model.X_train_
            y = self.model.y_train_
        except:
            X = None
            y = None
                    
        if X is None:
            X = x_new.reshape(1, -1)
            y = np.array([y_new])
        else:
            X = np.vstack((X, x_new))
            y = np.append(y, y_new)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y)
        
    def predict_shape(self, x_new):
        y_pred, sigma = self.model.predict(x_new, return_std=True)
        return y_pred, sigma
    
    def compute_score(self):
        return self.model.score(self.model.X_train_, self.model.y_train_)
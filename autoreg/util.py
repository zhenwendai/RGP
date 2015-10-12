
import numpy as np
from numpy.lib.stride_tricks import as_strided
from GPy.core.parameterization.transformations import Transformation, _lim_val, epsilon

def get_conv_1D(arr, win):
    assert win>0
    arr = arr.copy()
    if win==1:
        return arr
    else:
        return as_strided(arr, shape=(arr.shape[0]-win+1,win)+arr.shape[1:], strides=(arr.strides[0],)+arr.strides)
    
class LogexpInv(Transformation):
    def f(self, x):
        return np.where(x>_lim_val, x, np.log(np.exp(x+1e-20) - 1.))
    def finv(self, f):
        return np.where(f>_lim_val, f, np.log1p(np.exp(np.clip(f, -_lim_val, _lim_val)))) + epsilon        
    def gradfactor(self, f, df):
        return np.einsum('i,i->i', df, np.where(f>_lim_val, 1., 1. - np.exp(-f)))
    def initialize(self, f):
        if np.any(f < 0.):
            print("Warning: changing parameters to satisfy constraints")
        return np.abs(f)
    def log_jacobian(self, model_param):
        return np.where(model_param>_lim_val, model_param, np.log(np.exp(model_param+1e-20) - 1.)) - model_param
    def log_jacobian_grad(self, model_param):
        return 1./(np.exp(model_param)-1.)
    def __str__(self):
        return '+ve'



import numpy as np

class NormalEntropy(object):
    constant = 1.+np.log(2.*np.pi)
    
    def comp_value(self, variational_posterior):
        var = variational_posterior.variance
        return -(NormalEntropy.constant+np.log(var)).sum()/2.

    def update_gradients(self, variational_posterior):
        variational_posterior.variance.gradient +=  1. / (variational_posterior.variance*2.)
        
class NormalPrior(object):
    
    def comp_value(self, variational_posterior):
        var_mean = np.square(variational_posterior.mean).sum()
        var_S = (variational_posterior.variance - np.log(variational_posterior.variance)).sum()
        return 0.5 * (var_mean + var_S) - 0.5 * variational_posterior.input_dim * variational_posterior.num_data

    def update_gradients(self, variational_posterior):
        # dL:
        variational_posterior.mean.gradient -= variational_posterior.mean
        variational_posterior.variance.gradient -= (1. - (1. / (variational_posterior.variance))) * 0.5
        
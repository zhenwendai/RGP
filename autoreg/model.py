
import sys
import numpy as np
from scipy.linalg import LinAlgError
from GPy import Model,likelihoods
from GPy.core.parameterization.variational import VariationalPosterior,\
    NormalPosterior
from .layers import Layer

class DeepAutoreg(Model):
    
    def __init__(self, wins, Y, U=None, U_win=1, X_variance=0.01, num_inducing=10, likelihood = None, name='autoreg', kernels=None ):
        super(DeepAutoreg, self).__init__(name=name)
        
        self.nLayers = len(wins)
        self.wins = wins
        self.input_dim = 1
        self.output_dim = 1
        
        self._log_marginal_likelihood = np.nan
        
        self.Y = Y
        self.U = U
        self.U_win = U_win
        if self.U is not None: assert Y.shape[0]==U.shape[0] - U_win+1
        
        self.Xs = self._init_X(wins, Y, U, X_variance)
        
        # Parameters which exist differently per layer but specified as single componenents are here expanded to each layer
        if not isinstance(num_inducing, list or tuple): num_inducing = [num_inducing]*self.nLayers

        # Initialize Layers
        self.layers = []
        for i in range(self.nLayers-1,-1,-1):
            if i==self.nLayers-1:
                self.layers.append(Layer(None, self.Xs[i-1], X_win=wins[i], U=None, U_win=U_win, num_inducing=num_inducing[i],  kernel=kernels[i] if kernels is not None else None, noise_var=0.01, name='layer_'+str(i)))
            elif i==0:
                self.layers.append(Layer(self.layers[-1], Y, X_win=wins[i], U=self.Xs[i], U_win=wins[i+1], num_inducing=num_inducing[i],  kernel=kernels[i] if kernels is not None else None, likelihood=likelihood, noise_var=1., name='layer_'+str(i)))
            else:
                self.layers.append(Layer(self.layers[-1], self.Xs[i-1], X_win=wins[i], U=self.Xs[i], U_win=wins[i+1], num_inducing=num_inducing[i],  kernel=kernels[i] if kernels is not None else None, noise_var=0.01, name='layer_'+str(i)))
        self.layers[0].set_as_toplayer()
        self.link_parameters(*self.layers)
            
    def _init_X(self, wins, Y, U, X_variance, init='equal'):
        Xs = []
        if init=='equal':
            for i in range(len(wins)-1):
                mean = np.zeros((wins[i+1]-1+Y.shape[0],Y.shape[1]))
                mean[wins[i+1]-1:] = Y
                var = np.zeros((wins[i+1]-1+Y.shape[0],Y.shape[1]))+X_variance
                Xs.append(NormalPosterior(mean,var))
        return Xs
        
    def log_likelihood(self):
        return self._log_marginal_likelihood
        
    def parameters_changed(self):
        self._log_marginal_likelihood = np.sum([l._log_marginal_likelihood for l in self.layers])
        [l.update_latent_gradients() for l in self.layers[::-1]]

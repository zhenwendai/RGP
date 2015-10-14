
import sys
import numpy as np
from scipy.linalg import LinAlgError
from GPy import Model,likelihoods
from GPy.core.parameterization.variational import VariationalPosterior,\
    NormalPosterior
from .layers import Layer

class DeepAutoreg(Model):
    """
    :param U_pre_step: If true, the current signal is assumed to be controlled by the control signal of the previous time step.
    :type U_pre_step: Boolean
    """
    
    def __init__(self, wins, Y, U=None, U_win=1, X_variance=0.01, num_inducing=10, likelihood = None, name='autoreg', kernels=None, U_pre_step=True, init='Y', back_cstr=False, MLP_dims=None):
        super(DeepAutoreg, self).__init__(name=name)
        
        self.nLayers = len(wins)
        self.back_cstr = back_cstr
        self.wins = wins = [i+1 for i in wins]
        self.input_dim = 1
        self.output_dim = 1
        self._log_marginal_likelihood = np.nan
        self.U_pre_step = U_pre_step
        
        
        if U is not None:
            assert Y.shape[0]==U.shape[0], "the signal and control should be aligned."
            if U_pre_step:
                U = U[:-1].copy()
                Y = Y[U_win:].copy()
            else:
                Y = Y[U_win-1:].copy()
            self.U = NormalPosterior(U.copy(),np.ones(U.shape))
            self.U.variance[:] = 1e-10
        else:
            self.U = U
        self.Y = Y
        self.U_win = U_win

        self.Xs = self._init_X(wins, Y, U, X_variance, init=init)
        
        # Parameters which exist differently per layer but specified as single componenents are here expanded to each layer
        if not isinstance(num_inducing, list or tuple): num_inducing = [num_inducing]*self.nLayers

        # Initialize Layers
        self.layers = []
        for i in range(self.nLayers-1,-1,-1):
            if i==self.nLayers-1:
                self.layers.append(Layer(None, self.Xs[i-1], X_win=wins[i], U=self.U, U_win=U_win, num_inducing=num_inducing[i],  kernel=kernels[i] if kernels is not None else None, noise_var=0.01, name='layer_'+str(i),  back_cstr=back_cstr, MLP_dims=MLP_dims))
            elif i==0:
                self.layers.append(Layer(self.layers[-1], Y, X_win=wins[i], U=self.Xs[i], U_win=wins[i+1]-1, num_inducing=num_inducing[i],  kernel=kernels[i] if kernels is not None else None, likelihood=likelihood, noise_var=1., name='layer_'+str(i)))
            else:
                self.layers.append(Layer(self.layers[-1], self.Xs[i-1], X_win=wins[i], U=self.Xs[i], U_win=wins[i+1]-1, num_inducing=num_inducing[i],  kernel=kernels[i] if kernels is not None else None, noise_var=0.01, name='layer_'+str(i), back_cstr=back_cstr, MLP_dims=MLP_dims))
        self.link_parameters(*self.layers)
            
    def _init_X(self, wins, Y, U, X_variance, init='Y'):
        Xs = []
        if init=='Y':
            for i in range(len(wins)-1):
                mean = np.zeros((wins[i+1]-1+Y.shape[0],Y.shape[1]))
                mean[wins[i+1]-1:] = Y
                var = np.zeros((wins[i+1]-1+Y.shape[0],Y.shape[1]))+X_variance
                Xs.append(NormalPosterior(mean,var))
        elif init=='rand' :
            for i in range(len(wins)-1):
                mean = np.zeros((wins[i+1]-1+Y.shape[0],Y.shape[1]))
                mean[:] = np.random.randn(*mean.shape)
        elif init=='zero':
            for i in range(len(wins)-1):
                mean = np.zeros((wins[i+1]-1+Y.shape[0],Y.shape[1]))
                mean[:] = np.random.randn(*mean.shape)*0.01
        var = np.zeros((wins[i+1]-1+Y.shape[0],Y.shape[1]))+X_variance
        Xs.append(NormalPosterior(mean,var))
        return Xs
        
    def log_likelihood(self):
        return self._log_marginal_likelihood
        
    def parameters_changed(self):
        self._log_marginal_likelihood = np.sum([l._log_marginal_likelihood for l in self.layers])
        [l.update_latent_gradients() for l in self.layers[::-1]]
        
    def freerun(self, init_Xs=None, step=None, U=None, m_match=True):
        assert self.U_pre_step, "The other case is not implemented yet!"
        if U is None and self.layers[0].withControl: raise "The model needs control signals!"
        if U is not None and step is None: step=U.shape[0] - self.layers[0].U_win
        elif step is None: step=100
        
        con = U
        con_win = self.layers[0].U_win - 1
        for i in range(self.nLayers):
            con = con[con_win-self.layers[i].U_win+1:]
            X = self.layers[i].freerun(init_Xs=None if init_Xs is None else init_Xs[-i-1], step=step,U=con,m_match=m_match)
            con = X
            con_win = self.layers[i].X_win-1
        return X
            
        
#         Xs = []
#         con = U
#         con_win = self.layers[0].U_win-1
#         for i in range(self.nLayers):
#             layer = self.layers[i]
#             X_win,U_win = layer.X_win, layer.U_win
#             con = con[con_win-U_win+1:]
#             if X_win>1:
#                 X = np.empty((init_Xs[i].shape[0]+step,layer.X_flat.shape[1]))
#                 X[:init_Xs[i].shape[0]] = init_Xs[i]
#                 X_in = np.empty((1,layer.X.mean.shape[1]))
#                 for n in range(step):
#                     X_in[0,:X_win-1] = X[n:n+X_win-1].flat
#                     if layer.withControl: X_in[0,X_win-1:] = con[n:n+U_win].flat
#                     X[layer.X_win-1+n] = layer._raw_predict(X_in)[0]
#             else:
#                 X = np.empty((step,layer.X_flat.shape[1]))
#                 X_in = np.empty((1,layer.X.mean.shape[1]))
#                 for n in range(step):
#                     if layer.withControl: X_in[0,X_win-1:] = con[n:n+U_win].flat
#                     X[layer.X_win-1+n] = layer._raw_predict(X_in)[0]
#             Xs.append(X)
#             con = X
#             con_win = X_win-1
#         return con

        
        

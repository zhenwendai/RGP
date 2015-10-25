
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
    
    def __init__(self, wins, Y, U=None, U_win=1, nDims=None, X_variance=0.01, num_inducing=10, likelihood = None, name='autoreg', kernels=None, U_pre_step=True, init='Y', back_cstr=False, MLP_dims=None):
        super(DeepAutoreg, self).__init__(name=name)
        Ys, Us = Y,U
        if isinstance(Ys, np.ndarray): Ys = [Ys]
        if Us is not None and isinstance(Us, np.ndarray): Us = [Us]
        
        self.nLayers = len(wins)
        self.back_cstr = back_cstr
        self.wins = wins
        self.input_dim = 1
        self.output_dim = 1
        self._log_marginal_likelihood = np.nan
        self.U_pre_step = U_pre_step
        self.nDims = nDims if nDims is not None else [Ys[0].shape[1]]+[1]*(self.nLayers-1)
        
        if Us is not None:
            assert len(Ys)==len(Us)
            self.Us = []
            self.Ys = []
            for i in range(len(Ys)):
                Y, U = Ys[i], Us[i]
                assert Y.shape[0]==U.shape[0], "the signal and control should be aligned."
                if U_pre_step:
                    U = U[:-1].copy()
                    Y = Y[U_win:].copy()
                else:
                    Y = Y[U_win-1:].copy()
                self.Us.append(NormalPosterior(U.copy(),np.ones(U.shape)*1e-10))
                self.Ys.append(Y)
        else:
            self.Us = Us
            self.Ys = Ys

        self.Xs = self._init_X(wins, self.Ys, self.Us, X_variance, init=init, nDims=self.nDims)
        
        # Parameters which exist differently per layer but specified as single componenents are here expanded to each layer
        if not isinstance(num_inducing, list or tuple): num_inducing = [num_inducing]*self.nLayers

        # Initialize Layers
        self.layers = []
        for i in range(self.nLayers-1,-1,-1):
            if i==self.nLayers-1:
                self.layers.append(Layer(None, self.Xs[i-1], X_win=wins[i], Us=self.Us, U_win=U_win, num_inducing=num_inducing[i],  kernel=kernels[i] if kernels is not None else None, noise_var=0.01, name='layer_'+str(i),  back_cstr=back_cstr, MLP_dims=MLP_dims))
            elif i==0:
                self.layers.append(Layer(self.layers[-1], self.Ys, X_win=wins[i], Us=self.Xs[i], U_win=wins[i+1], num_inducing=num_inducing[i],  kernel=kernels[i] if kernels is not None else None, likelihood=likelihood, noise_var=1., name='layer_'+str(i)))
            else:
                self.layers.append(Layer(self.layers[-1], self.Xs[i-1], X_win=wins[i], Us=self.Xs[i], U_win=wins[i+1], num_inducing=num_inducing[i],  kernel=kernels[i] if kernels is not None else None, noise_var=0.01, name='layer_'+str(i), back_cstr=back_cstr, MLP_dims=MLP_dims))
        self.link_parameters(*self.layers)
            
    def _init_X(self, wins, Ys, Us, X_variance, nDims, init='Y'):
        Xs = []
        for i_layer in range(1,self.nLayers):
            X = []
            for i_seq in range(len(Ys)):
                Y = Ys[i_seq]
                U_win, U_dim = wins[i_layer], nDims[i_layer]
                if init=='Y':
                    mean = np.zeros((U_win+Y.shape[0], U_dim))
                    mean[U_win:] = Y[:,:U_dim]
                    var = np.zeros((U_win+Y.shape[0],U_dim))+X_variance
                    X.append(NormalPosterior(mean,var,name='qX_'+str(i_seq)))
                elif init=='rand' :
                    mean = np.zeros((U_win+Y.shape[0], U_dim))
                    mean[:] = np.random.randn(*mean.shape)
                    var = np.zeros((U_win+Y.shape[0],U_dim))+X_variance
                    X.append(NormalPosterior(mean,var,name='qX_'+str(i_seq)))
                elif init=='zero':
                    mean = np.zeros((U_win+Y.shape[0], U_dim))
                    mean[:] = np.random.randn(*mean.shape)*0.01
                    var = np.zeros((U_win+Y.shape[0],U_dim))+X_variance
                    X.append(NormalPosterior(mean,var,name='qX_'+str(i_seq)))
            Xs.append(X)
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
            con_win = self.layers[i].X_win
        return X

        
        

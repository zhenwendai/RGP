#
#

import numpy as np

from GPy.core import SparseGP
from GPy import likelihoods
from GPy import kern
from GPy.core.parameterization.variational import NormalPosterior, NormalPrior, VariationalPosterior
from .variational import NormalEntropy,NormalPrior

from .util import get_conv_1D

class Layer(SparseGP):
    
    def __init__(self, layer_upper, X, X_win=1, U=None, U_win=1, Z=None, num_inducing=10,  kernel=None, inference_method=None, likelihood=None, noise_var=1., name='layer'):

        self.layer_upper = layer_upper
        self.X_win = X_win # if X_win==1, it is not autoregressive.
        self.U_win = U_win
        self.Q_dim = X.shape[1]
        
        self.X_flat = X
        if U is None:
            self.withControl = False
            self.U_flat = None
        else:
            self.withControl = True
            self.U_flat = U
        self.X_observed = False if isinstance(X, VariationalPosterior) else True 
            
        if self.X_observed:
            self.Y = X[X_win-1:]
        else:
            self.Y = NormalPosterior(X.mean.values[X_win-1:].copy(),X.variance.values[X_win-1:].copy())
        N = self.Y.shape[0]
        self._update_conv()
        if not self.withControl:
            self.X = NormalPosterior(self.X_mean_conv.copy(),self.X_var_conv.copy())
        elif X_win==1:
            self.X = NormalPosterior(self.U_mean_conv.copy(),self.U_var_conv.copy())
        else:
            self.X = NormalPosterior(np.hstack([self.X_mean_conv.copy(),self.U_mean_conv.copy()]),
                                     np.hstack([self.X_var_conv.copy().copy(),self.U_var_conv.copy()]))
        
        if Z is None:
            Z = np.random.permutation(self.X.mean.copy())[:num_inducing]
        assert Z.shape[1] == self.X.shape[1]
        
        if kernel is None: kernel = kern.RBF(self.X.shape[1], ARD = True)
        
        if inference_method is None: 
            from .inference import VarDTC
            inference_method = VarDTC()
        if likelihood is None: likelihood = likelihoods.Gaussian(variance=noise_var)
        self._toplayer_ = False
        self.variationalterm = NormalEntropy()
        super(Layer, self).__init__(self.X, self.Y, Z, kernel, likelihood, inference_method=inference_method, name=name)
        if not self.X_observed: self.link_parameter(self.X_flat)
    
    def plot_latent(self, labels=None, which_indices=None,
                resolution=50, ax=None, marker='o', s=40,
                fignum=None, plot_inducing=True, legend=True,
                plot_limits=None,
                aspect='auto', updates=False, predict_kwargs={}, imshow_kwargs={}):
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from GPy.plotting.matplot_dep import dim_reduction_plots

        return dim_reduction_plots.plot_latent(self, labels, which_indices,
                resolution, ax, marker, s,
                fignum, plot_inducing, legend,
                plot_limits, aspect, updates, predict_kwargs, imshow_kwargs)

    def _update_conv(self):
        N = self.Y.shape[0]
        if not self.withControl:
            self.X_mean_conv = get_conv_1D(self.X_flat.mean.values[:-1], self.X_win-1).reshape(N,-1)
            self.X_var_conv = get_conv_1D(self.X_flat.variance.values[:-1], self.X_win-1).reshape(N,-1)
        elif self.X_win==1:
            self.U_mean_conv = get_conv_1D(self.U_flat.mean.values[-N-self.U_win+1:], self.U_win).reshape(N,-1)
            self.U_var_conv = get_conv_1D(self.U_flat.variance.values[-N-self.U_win+1:], self.U_win).reshape(N,-1)
        else:
            self.X_mean_conv = get_conv_1D(self.X_flat.mean.values[:-1], self.X_win-1).reshape(N,-1)
            self.X_var_conv = get_conv_1D(self.X_flat.variance.values[:-1], self.X_win-1).reshape(N,-1)
            self.U_mean_conv = get_conv_1D(self.U_flat.mean.values[-N-self.U_win+1:], self.U_win).reshape(N,-1)
            self.U_var_conv = get_conv_1D(self.U_flat.variance.values[-N-self.U_win+1:], self.U_win).reshape(N,-1)
    
    def _update_X(self):
        self._update_conv()
        if not self.X_observed:
            self.Y.mean[:] = self.X_flat.mean[self.X_win-1:]
            self.Y.variance[:] = self.X_flat.variance[self.X_win-1:]
        if not self.withControl:
            self.X.mean[:] = self.X_mean_conv
            self.X.variance[:] = self.X_var_conv
        elif self.X_win==1:
            self.X.mean[:] = self.U_mean_conv
            self.X.variance[:] = self.U_var_conv
        else:
            self.X.mean[:,:self.X_mean_conv.shape[1]] = self.X_mean_conv
            self.X.variance[:,:self.X_mean_conv.shape[1]] = self.X_var_conv
            self.X.mean[:,self.X_mean_conv.shape[1]:] = self.U_mean_conv
            self.X.variance[:,self.X_mean_conv.shape[1]:] = self.U_var_conv
            
    def update_latent_gradients(self):
        N = self.Y.shape[0]
        for n in xrange(self.X.shape[0]):
            if not self.withControl:
                self.X_flat.mean.gradient[n:n+self.X_win-1] += self.X.mean.gradient[n].reshape(-1,self.Q_dim)
                self.X_flat.variance.gradient[n:n+self.X_win-1] += self.X.variance.gradient[n].reshape(-1,self.Q_dim)
            elif self.X_win==1:
                offset = -N-self.U_win+1+self.U_flat.shape[0]
                self.U_flat.mean.gradient[offset+n:offset+n+self.U_win] += self.X.mean.gradient[n].reshape(-1,self.Q_dim)
                self.U_flat.variance.gradient[offset+n:offset+n+self.U_win] += self.X.variance.gradient[n].reshape(-1,self.Q_dim)
            else:
                offset = -N-self.U_win+1+self.U_flat.shape[0]
                self.X_flat.mean.gradient[n:n+self.X_win-1] += self.X.mean.gradient[n,:self.X_mean_conv.shape[1]].reshape(-1,self.Q_dim)
                self.X_flat.variance.gradient[n:n+self.X_win-1] += self.X.variance.gradient[n,:self.X_mean_conv.shape[1]].reshape(-1,self.Q_dim)
                self.U_flat.mean.gradient[offset+n:offset+n+self.U_win] += self.X.mean.gradient[n,self.X_mean_conv.shape[1]:].reshape(-1,self.Q_dim)
                self.U_flat.variance.gradient[offset+n:offset+n+self.U_win] += self.X.variance.gradient[n,self.X_mean_conv.shape[1]:].reshape(-1,self.Q_dim)
    
    def _update_qX_gradients(self):
        self.X.mean.gradient, self.X.variance.gradient = self.kern.gradients_qX_expectations(
                                            variational_posterior=self.X,
                                            Z=self.Z,
                                            dL_dpsi0=self.grad_dict['dL_dpsi0'],
                                            dL_dpsi1=self.grad_dict['dL_dpsi1'],
                                            dL_dpsi2=self.grad_dict['dL_dpsi2'])
        delta = -self.variationalterm.comp_value(self.X)
        self._log_marginal_likelihood += delta
        self.variationalterm.update_gradients(self.X)
        
    def parameters_changed(self):
        self._update_X()
        super(Layer,self).parameters_changed()
        self._update_qX_gradients()
        if not self.X_observed: 
            self.X_flat.mean.gradient[:] = 0.
            self.X_flat.variance.gradient[:] = 0.
        if not self.X_observed:
            self.X_flat.mean.gradient[self.X_win-1:] += self.grad_dict['dL_dYmean']
            self.X_flat.variance.gradient[self.X_win-1:] += self.grad_dict['dL_dYvar'][:,None]

    def set_as_toplayer(self, flag=True):
        if flag:
            self.variationalterm = NormalPrior()
        else:
            self.variationalterm = NormalEntropy()
        self._toplayer_ = flag


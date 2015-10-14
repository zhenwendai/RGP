#
#

import numpy as np

from GPy.core import SparseGP, Param
from GPy import likelihoods
from GPy import kern
from GPy.core.parameterization.variational import NormalPosterior, NormalPrior, VariationalPosterior
from .variational import NormalEntropy,NormalPrior

from .util import get_conv_1D

class Layer(SparseGP):
    
    def __init__(self, layer_upper, X, X_win=1, U=None, U_win=1, Z=None, num_inducing=10,  kernel=None, inference_method=None, likelihood=None, noise_var=1., back_cstr=False, MLP_dims=None,name='layer'):

        self.signal_dim = X.shape[1]
        self.layer_upper = layer_upper
        self.X_win = X_win # if X_win==1, it is not autoregressive.
        self.U_win = U_win
        self.Q_dim = X.shape[1]
        self.back_cstr = back_cstr
        
        self.X_flat = X
        if U is None:
            self.withControl = False
            self.U_flat = None
            self.U_win = 0
        else:
            self.withControl = True
            self.U_flat = U
        self.X_observed = False if isinstance(X, VariationalPosterior) else True 
            
        if self.X_observed:
            self.Y = X[X_win-1:]
        else:
            self.Y = NormalPosterior(X.mean.values[X_win-1:].copy(),X.variance.values[X_win-1:].copy())
        N = self.Y.shape[0]
        if not self.X_observed and back_cstr:
            from .mlp import MLP
            from copy import deepcopy
            from GPy.core.parameterization.transformations import Logexp
            assert self.X_win>1
            Q = (self.X_win-1+self.U_win)*self.signal_dim
            self.init_X = NormalPosterior(self.X_flat.mean.values[:self.X_win-1],self.X_flat.variance.values[:self.X_win-1])
            self.encoder = MLP([Q, Q*2, Q+self.Y.shape[1]/2, self.Y.shape[1]] if MLP_dims is None else [Q]+deepcopy(MLP_dims)+[self.Y.shape[1]])
            self.var_trans = Logexp()
            self.X_var = Param('X_var',self.X_flat.variance.values[self.X_win-1:].copy(), Logexp())
        self._update_conv()
        if not self.withControl:
            self.X = NormalPosterior(self.X_mean_conv.copy(),self.X_var_conv.copy())
        elif X_win==1:
            self.X = NormalPosterior(self.U_mean_conv.copy(),self.U_var_conv.copy())
        else:
            self.X = NormalPosterior(np.hstack([self.X_mean_conv.copy(),self.U_mean_conv.copy()]),
                                     np.hstack([self.X_var_conv.copy().copy(),self.U_var_conv.copy()]))
        
        if Z is None:
            Z = np.random.permutation(self.X.mean.values.copy())[:num_inducing]
#             from sklearn.cluster import KMeans
#             m = KMeans(n_clusters=num_inducing,n_init=1000,max_iter=100)
#             m.fit(self.X.mean.values.copy())
#             Z = m.cluster_centers_.copy()
        assert Z.shape[1] == self.X.shape[1]
        
        if kernel is None: kernel = kern.RBF(self.X.shape[1], ARD = True)
        
        if inference_method is None: 
            from .inference import VarDTC
            inference_method = VarDTC()
        if likelihood is None: likelihood = likelihoods.Gaussian(variance=noise_var)
        self.normalPrior = NormalPrior()
        self.normalEntropy = NormalEntropy()
        super(Layer, self).__init__(self.X, self.Y, Z, kernel, likelihood, inference_method=inference_method, name=name)
        if not self.X_observed: 
            if back_cstr:
                from .mlp import MLP
                from copy import deepcopy
                from GPy.core.parameterization.transformations import Logexp
                assert self.X_win>1
                self.link_parameters(self.init_X, self.encoder, self.X_var)
            else:
                self.link_parameter(self.X_flat)
    
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
        if self.back_cstr: self._encoder_freerun()
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
        if self.back_cstr: self._encoder_update_gradient()
    
    def _update_qX_gradients(self):
        self.X.mean.gradient, self.X.variance.gradient = self.kern.gradients_qX_expectations(
                                            variational_posterior=self.X,
                                            Z=self.Z,
                                            dL_dpsi0=self.grad_dict['dL_dpsi0'],
                                            dL_dpsi1=self.grad_dict['dL_dpsi1'],
                                            dL_dpsi2=self.grad_dict['dL_dpsi2'])
        
    def parameters_changed(self):
        self._update_X()
        super(Layer,self).parameters_changed()
        self._update_qX_gradients()
        if self.withControl and self.layer_upper is None:
            self.U_flat.mean.gradient[:] = 0
            self.U_flat.variance.gradient[:] = 0
        if not self.X_observed: 
            self.X_flat.mean.gradient[:] = 0.
            self.X_flat.variance.gradient[:] = 0.
        if not self.X_observed:
            self.X_flat.mean.gradient[self.X_win-1:] += self.grad_dict['dL_dYmean']
            self.X_flat.variance.gradient[self.X_win-1:] += self.grad_dict['dL_dYvar'][:,None]
            
            delta = 0
            if self.X_win>1:
                delta += -self.normalPrior.comp_value(self.X_flat[:self.X_win-1])
                self.normalPrior.update_gradients(self.X_flat[:self.X_win-1])
            delta += -self.normalEntropy.comp_value(self.X_flat[self.X_win-1:])
            self.normalEntropy.update_gradients(self.X_flat[self.X_win-1:])
            self._log_marginal_likelihood += delta

    def _encoder_freerun(self):
        Q, N = self.signal_dim, self.X_flat.shape[0]-self.X_win+1
        X_win, U_win = self.X_win, self.U_win
        X_dim = (X_win-1+U_win)*Q
        Y_dim = Q

        self.X_flat.variance[:X_win-1] = self.init_X.variance.values
        self.X_flat.variance[X_win-1:] = self.X_var.values
        self.X_flat.mean[:X_win-1] = self.init_X.mean.values
        X_in = np.zeros((X_dim,))
        U_offset = self.U_flat.shape[0]-N-U_win+1
        for n in range(N):
            X_in[:(X_win-1)*Q] = self.X_flat.mean[n:n+X_win-1].flat
            if self.withControl: 
                X_in[(X_win-1)*Q:X_dim] = self.U_flat.mean[U_offset+n:U_offset+n+U_win].flat
            X_out = self.encoder.predict(X_in[None,:])
            self.X_flat.mean[X_win-1+n] = X_out[0,:Y_dim].reshape(-1,Q)
    
    def _encoder_update_gradient(self):
        self.encoder.prepare_grad()
        Q, N = self.signal_dim, self.X_flat.shape[0]-self.X_win+1
        X_win, U_win = self.X_win, self.U_win
        X_dim = self.X.shape[1]
        Y_dim = self.Y.shape[1]

        X_in = np.zeros((X_dim,))
        dL = np.zeros((Y_dim,))
        U_offset = self.U_flat.shape[0]-N-U_win+1
        for n in range(self.Y.shape[0]-1,-1,-1):
            X_in[:X_dim] = self.X.mean[n]
            dL[:Y_dim] = self.X_flat.mean.gradient[X_win-1+n].flat
            dX = self.encoder.update_gradient(X_in[None,:], dL[None,:])
            self.X_flat.mean.gradient[n:n+X_win-1] += dX[0,:(X_win-1)*Q].reshape(-1,Q)
            if self.withControl:
                self.U_flat.mean.gradient[U_offset+n:U_offset+n+U_win] += dX[0,(X_win-1)*Q:X_dim].reshape(-1,Q)
        self.init_X.mean.gradient[:] = self.X_flat.mean.gradient[:X_win-1]
        self.X_var.gradient[:] = self.X_flat.variance.gradient[X_win-1:]
        self.init_X.variance.gradient[:] = self.X_flat.variance.gradient[:X_win-1]
        
    def freerun(self, init_Xs=None, step=None, U=None, m_match=True):
        if U is None and self.withControl: raise "The model needs control signals!"
        if U is not None and step is None: step=U.shape[0] - self.U_win
        elif step is None: step=100
        if init_Xs is None and self.X_win>1:
            if m_match:
                init_Xs = NormalPosterior(np.zeros((self.X_win-1,self.X_flat.shape[1])),np.ones((self.X_win-1,self.X_flat.shape[1]))*1)
            else:
                init_Xs = np.zeros((self.X_win-1,self.X_flat.shape[1])) 
        Q = self.signal_dim
        X_win, U_win = self.X_win, self.U_win
        
        if m_match:
            X = NormalPosterior(np.empty((X_win-1+step, self.X_flat.shape[1])),np.ones((X_win-1+step, self.X_flat.shape[1])))
            if X_win>1:
                X.mean[:X_win-1] = init_Xs.mean[-X_win+1:]
                X.variance[:X_win-1] = init_Xs.variance[-X_win+1:]
            X_in =NormalPosterior(np.empty((1, self.X.mean.shape[1])),np.ones((1, self.X.mean.shape[1])))
            X_in.variance[:] = 1e-10
            for n in range(step):
                if X_win>1:
                    X_in.mean[0,:(X_win-1)*Q] = X.mean[n:n+X_win-1].flat
                    X_in.variance[0,:(X_win-1)*Q] = X.variance[n:n+X_win-1].flat
                if self.withControl: 
                    if isinstance(U, NormalPosterior):
                        X_in.mean[0,(X_win-1)*Q:] = U.mean[n:n+U_win].flat
                        X_in.variance[0,(X_win-1)*Q:] = U.variance[n:n+U_win].flat
                    else:
                        X_in.mean[0,(X_win-1)*Q:] = U[n:n+U_win].flat
                X_out = self._raw_predict(X_in)
                X.mean[X_win-1+n] = X_out[0]
                if np.any(X_out[1]<=0.): print X_out[1]
                X.variance[X_win-1+n] = X_out[1]
        else:
            X = np.empty((X_win-1+step, self.X_flat.shape[1]))
            X_in = np.empty((1,self.X.mean.shape[1]))
            if X_win>1: # depends on history                
                X[:X_win-1] = init_Xs[-X_win+1:]
            for n in range(step):
                if X_win>1: X_in[0,:(X_win-1)*Q] = X[n:n+X_win-1].flat
                if self.withControl: X_in[0,(X_win-1)*Q:] = U[n:n+U_win].flat
                X[X_win-1+n] = self._raw_predict(X_in)[0]
        return X


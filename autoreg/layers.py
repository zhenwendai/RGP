#
#

import numpy as np

from GPy.core import SparseGP, Param
from GPy import likelihoods
from GPy import kern
from GPy.core.parameterization.variational import NormalPosterior, NormalPrior, VariationalPosterior
from .variational import NormalEntropy,NormalPrior
from .inference import VarDTC

from .util import get_conv_1D

class Layer(SparseGP):
    
    def __init__(self, layer_upper, Xs, X_win=0, Us=None, U_win=1, Z=None, num_inducing=10,  kernel=None, inference_method=None, likelihood=None, noise_var=1., back_cstr=False, MLP_dims=None,name='layer'):

        self.layer_upper = layer_upper
        self.nSeq = len(Xs)
        self.back_cstr = back_cstr

        self.X_win = X_win # if X_win==0, it is not autoregressive.        
        self.X_dim = Xs[0].shape[1]
        self.Xs_flat = Xs
        self.X_observed = False if isinstance(Xs[0], VariationalPosterior) else True 
        
        self.withControl = Us is not None
        self.U_win = U_win
        self.U_dim = Us[0].shape[1] if self.withControl else None
        self.Us_flat = Us
        if self.withControl: assert len(Xs)==len(Us), "The number of signals should be equal to the number controls!"
        
        if not self.X_observed and back_cstr: self._init_encoder(MLP_dims)
        self._init_XY()
        
        if Z is None:
            if back_cstr:
                Z = np.random.randn(num_inducing,self.X.shape[1])
            else:
                from sklearn.cluster import KMeans
                m = KMeans(n_clusters=num_inducing,n_init=1000,max_iter=100)
                m.fit(self.X.mean.values.copy())
                Z = m.cluster_centers_.copy()
        assert Z.shape[1] == self.X.shape[1]
        
        if kernel is None: kernel = kern.RBF(self.X.shape[1], ARD = True)
        
        if inference_method is None: inference_method = VarDTC()
        if likelihood is None: likelihood = likelihoods.Gaussian(variance=noise_var)
        self.normalPrior, self.normalEntropy = NormalPrior(), NormalEntropy()
        super(Layer, self).__init__(self.X, self.Y, Z, kernel, likelihood, inference_method=inference_method, name=name)
        if not self.X_observed: 
            if back_cstr:
                assert self.X_win>0
                self.link_parameters(*(self.init_Xs + self.Xs_var+[self.encoder]))
            else:
                self.link_parameters(*self.Xs_flat)
                
    def _init_encoder(self, MLP_dims):
        from .mlp import MLP
        from copy import deepcopy
        from GPy.core.parameterization.transformations import Logexp
        X_win, X_dim, U_win, U_dim = self.X_win, self.X_dim, self.U_win, self.U_dim
        assert X_win>0, "Neural Network constraints only applies autoregressive structure!"
        Q = X_win*X_dim+U_win*U_dim if self.withControl else X_win*X_dim
        self.init_Xs = [NormalPosterior(self.Xs_flat[i].mean.values[:X_win],self.Xs_flat[i].variance.values[:X_win], name='init_Xs_'+str(i)) for i in range(self.nSeq)]
        for init_X in self.init_Xs: init_X.mean[:] = np.random.randn(*init_X.shape)*1e-2
        self.encoder = MLP([Q, Q*2, Q+X_dim/2, X_dim] if MLP_dims is None else [Q]+deepcopy(MLP_dims)+[X_dim])
        self.Xs_var = [Param('X_var_'+str(i),self.Xs_flat[i].variance.values[X_win:].copy(), Logexp()) for i in range(self.nSeq)]

    def _init_XY(self):
        X_win, X_dim, U_win, U_dim = self.X_win, self.X_dim, self.U_win, self.U_dim
        self._update_conv()
        if X_win>0: X_mean_conv, X_var_conv = np.vstack(self.X_mean_conv), np.vstack(self.X_var_conv)
        if self.withControl: U_mean_conv, U_var_conv = np.vstack(self.U_mean_conv), np.vstack(self.U_var_conv)
        
        if not self.withControl:
            self.X = NormalPosterior(X_mean_conv, X_var_conv)
        elif X_win==0:
            self.X = NormalPosterior(U_mean_conv, U_var_conv)
        else:
            self.X = NormalPosterior(np.hstack([X_mean_conv, U_mean_conv]), np.hstack([X_var_conv, U_var_conv]))

        if self.X_observed:
            self.Y = np.vstack([x[X_win:] for x in self.Xs_flat])
        else:
            self.Y = NormalPosterior(np.vstack([x.mean.values[X_win:] for x in self.Xs_flat]), np.vstack([x.variance.values[X_win:] for x in self.Xs_flat]))
    
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
        X_win, X_dim, U_win, U_dim = self.X_win, self.X_dim, self.U_win, self.U_dim
        if self.back_cstr: self._encoder_freerun()
        self.X_mean_conv, self.X_var_conv, self.U_mean_conv, self.U_var_conv = [], [], [], []
        for i_seq in range(self.nSeq):
            N = self.Xs_flat[i_seq].shape[0]-X_win
            if X_win>0:
                self.X_mean_conv.append(get_conv_1D(self.Xs_flat[i_seq].mean.values[:-1], X_win).reshape(N,-1))
                self.X_var_conv.append(get_conv_1D(self.Xs_flat[i_seq].variance.values[:-1], X_win).reshape(N,-1))
            if self.withControl:
                self.U_mean_conv.append(get_conv_1D(self.Us_flat[i_seq].mean.values[-N-U_win+1:], U_win).reshape(N,-1))
                self.U_var_conv.append(get_conv_1D(self.Us_flat[i_seq].variance.values[-N-U_win+1:], U_win).reshape(N,-1))
    
    def _update_X(self):
        self._update_conv()
        X_offset, Y_offset = 0, 0
        for i_seq in range(self.nSeq):
            if self.X_win>0:
                N, Q = self.X_mean_conv[i_seq].shape
                self.X.mean[X_offset:X_offset+N, :Q] = self.X_mean_conv[i_seq]
                self.X.variance[X_offset:X_offset+N, :Q] = self.X_var_conv[i_seq]
            else: Q=0
            if self.withControl:
                N = self.U_mean_conv[i_seq].shape[0]
                self.X.mean[X_offset:X_offset+N, Q:] = self.U_mean_conv[i_seq]
                self.X.variance[X_offset:X_offset+N, Q:] = self.U_var_conv[i_seq]
            X_offset += N
            
            if not self.X_observed:
                N = self.Xs_flat[i_seq].shape[0]-self.X_win
                self.Y.mean[Y_offset:Y_offset+N] = self.Xs_flat[i_seq].mean[self.X_win:]
                self.Y.variance[Y_offset:Y_offset+N] = self.Xs_flat[i_seq].variance[self.X_win:]
                Y_offset += N
            
    def update_latent_gradients(self):
        X_offset = 0
        X_win, X_dim, U_win, U_dim = self.X_win, self.X_dim, self.U_win, self.U_dim
        for i_seq in range(self.nSeq):
            N = self.Xs_flat[i_seq].shape[0] -  X_win
            if self.withControl: U_offset = -N-U_win+1+self.Us_flat[i_seq].shape[0]
            for n in range(N):
                if X_win>0:
                    Q = self.X_mean_conv[i_seq].shape[1]
                    self.Xs_flat[i_seq].mean.gradient[n:n+X_win] += self.X.mean.gradient[X_offset+n,:Q].reshape(-1, X_dim)
                    self.Xs_flat[i_seq].variance.gradient[n:n+X_win] += self.X.variance.gradient[X_offset+n,:Q].reshape(-1, X_dim)
                else: Q=0
                if self.withControl:
                    self.Us_flat[i_seq].mean.gradient[U_offset+n:U_offset+n+U_win] += self.X.mean.gradient[X_offset+n,Q:].reshape(-1, U_dim)
                    self.Us_flat[i_seq].variance.gradient[U_offset+n:U_offset+n+U_win] += self.X.variance.gradient[X_offset+n,Q:].reshape(-1, U_dim)
            X_offset += N
        if self.back_cstr: self._encoder_update_gradient()
    
    def _update_qX_gradients(self):
        self.X.mean.gradient, self.X.variance.gradient = self.kern.gradients_qX_expectations(
                                            variational_posterior=self.X,
                                            Z=self.Z,
                                            dL_dpsi0=self.grad_dict['dL_dpsi0'],
                                            dL_dpsi1=self.grad_dict['dL_dpsi1'],
                                            dL_dpsi2=self.grad_dict['dL_dpsi2'])
    
    def _prepare_gradients(self):
        X_win = self.X_win
        if self.withControl and self.layer_upper is None:
            for U in self.Us_flat:
                U.mean.gradient[:] = 0
                U.variance.gradient[:] = 0
        if not self.X_observed: 
            Y_offset = 0
            delta = 0
            for X in self.Xs_flat:
                N = X.shape[0] - X_win
                X.mean.gradient[:] = 0
                X.variance.gradient[:] = 0
                X.mean.gradient[X_win:] += self.grad_dict['dL_dYmean'][Y_offset:Y_offset+N]
                X.variance.gradient[X_win:] += self.grad_dict['dL_dYvar'][Y_offset:Y_offset+N,None]
                if X_win>0:
                    delta += -self.normalPrior.comp_value(X[:X_win])
                    self.normalPrior.update_gradients(X[:X_win])
                delta += -self.normalEntropy.comp_value(X[X_win:])
                self.normalEntropy.update_gradients(X[X_win:])
                Y_offset += N
            self._log_marginal_likelihood += delta
                
    def parameters_changed(self):
        self._update_X()
        super(Layer,self).parameters_changed()
        self._update_qX_gradients()
        self._prepare_gradients()

    def _encoder_freerun(self):
        X_win, X_dim, U_win, U_dim = self.X_win, self.X_dim, self.U_win, self.U_dim        
        Q = X_win*X_dim+U_win*U_dim if self.withControl else X_win*X_dim
        
        X_in = np.zeros((Q,))
        for i_seq in range(self.nSeq):
            X_flat, init_X, X_var = self.Xs_flat[i_seq], self.init_Xs[i_seq], self.Xs_var[i_seq]
            if self.withControl: U_flat = self.Us_flat[i_seq]
            X_flat.mean[:X_win] = init_X.mean.values
            X_flat.variance[:X_win] = init_X.variance.values
            X_flat.variance[X_win:] = X_var.values
            
            N = X_flat.shape[0] - X_win
            if self.withControl: U_offset = U_flat.shape[0]-N-U_win+1
            for n in range(N):
                X_in[:X_win*X_dim] = X_flat.mean[n:n+X_win].flat
                if self.withControl: 
                    X_in[X_win*X_dim:] = U_flat.mean[U_offset+n:U_offset+n+U_win].flat
                X_out = self.encoder.predict(X_in[None,:])
                X_flat.mean[X_win+n] = X_out[0]
    
    def _encoder_update_gradient(self):
        self.encoder.prepare_grad()        
        X_win, X_dim, U_win, U_dim = self.X_win, self.X_dim, self.U_win, self.U_dim        
        Q = X_win*X_dim+U_win*U_dim if self.withControl else X_win*X_dim
        
        X_in = np.zeros((Q,))
        dL =np.zeros((X_dim,))
        for i_seq in range(self.nSeq):
            X_flat, init_X, X_var = self.Xs_flat[i_seq], self.init_Xs[i_seq], self.Xs_var[i_seq]
            if self.withControl: U_flat = self.Us_flat[i_seq]
            N = X_flat.shape[0] - X_win
            if self.withControl: U_offset = U_flat.shape[0]-N-U_win+1
            
            for n in range(N-1,-1,-1):
                X_in[:X_win*X_dim] = X_flat.mean[n:n+X_win].flat
                if self.withControl: 
                    X_in[X_win*X_dim:] = U_flat.mean[U_offset+n:U_offset+n+U_win].flat
                dL[:] = X_flat.mean.gradient[X_win+n].flat
                dX = self.encoder.update_gradient(X_in[None,:], dL[None,:])
                X_flat.mean.gradient[n:n+X_win] += dX[0,:X_win*X_dim].reshape(-1,X_dim)
                if self.withControl:
                    U_flat.mean.gradient[U_offset+n:U_offset+n+U_win] += dX[0, X_win*X_dim:].reshape(-1,U_dim)
            init_X.mean.gradient[:] = X_flat.mean.gradient[:X_win]
            init_X.variance.gradient[:] = X_flat.variance.gradient[:X_win]
            X_var.gradient[:] = X_flat.variance.gradient[X_win:]
                
    def freerun(self, init_Xs=None, step=None, U=None, m_match=True):
        X_win, X_dim, U_win, U_dim = self.X_win, self.X_dim, self.U_win, self.U_dim
        Q = X_win*X_dim+U_win*U_dim if self.withControl else X_win*X_dim
        if U is None and self.withControl: raise "The model needs control signals!"
        if U is not None and step is None: step=U.shape[0] - U_win
        elif step is None: step=100
        if init_Xs is None and X_win>0:
            if m_match:
                init_Xs = NormalPosterior(np.zeros((X_win,X_dim)),np.ones((X_win,X_dim)))
            else:
                init_Xs = np.zeros((X_win,X_dim))
        if U is not None: assert U.shape[1]==U_dim, "The dimensionality of control signal has to be "+str(U_dim)+"!"
        
        if m_match: # free run with moment matching
            X = NormalPosterior(np.empty((X_win+step, X_dim)),np.ones((X_win+step, X_dim)))
            if X_win>0:
                X.mean[:X_win] = init_Xs.mean[-X_win:]
                X.variance[:X_win] = init_Xs.variance[-X_win:]
            X_in =NormalPosterior(np.empty((1, Q)),np.ones((1, Q)))
            X_in.variance[:] = 1e-10
            for n in range(step):
                if X_win>0:
                    X_in.mean[0,:X_win*X_dim] = X.mean[n:n+X_win].flat
                    X_in.variance[0,:X_win*X_dim] = X.variance[n:n+X_win].flat
                if self.withControl: 
                    if isinstance(U, NormalPosterior):
                        X_in.mean[0,X_win*X_dim:] = U.mean[n:n+U_win].flat
                        X_in.variance[0,X_win*X_dim:] = U.variance[n:n+U_win].flat
                    else:
                        X_in.mean[0,X_win*X_dim:] = U[n:n+U_win].flat
                X_out = self._raw_predict(X_in)
                X.mean[X_win+n] = X_out[0]
                if np.any(X_out[1]<=0.): X_out[1][X_out[1]<=0.] = 1e-10
                X.variance[X_win+n] = X_out[1]
        else:
            X = np.empty((X_win+step, X_dim))
            X_in = np.empty((1,Q))
            if X_win>0: X[:X_win] = init_Xs[-X_win+1:]
            for n in range(step):
                if X_win>0: X_in[0,:X_win*X_dim] = X[n:n+X_win].flat
                if self.withControl: X_in[0,X_win*X_dim:] = U[n:n+U_win].flat
                X[X_win+n] = self._raw_predict(X_in)[0]
        return X


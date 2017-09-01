#
#

import numpy as np

from GPy.core import SparseGP, Param
from GPy import likelihoods
from GPy import kern
from GPy.core.parameterization.variational import NormalPosterior, NormalPrior, VariationalPosterior
from GPy.core.parameterization.transformations import Logexp
from .variational import NormalEntropy,NormalPrior
from .inference import VarDTC


from .util import get_conv_1D

# TODO: maybe later use sparse_gp_mpi from GPy
class SparseGP_MPI(SparseGP):
    
    def __init__(self, X, Y, Z, kernel, likelihood, mean_function=None, inference_method=None,
                 name='sparse gp', Y_metadata=None, normalizer=False, mpi_comm=None, mpi_root=0, auto_update=True):
        self.mpi_comm = mpi_comm
        self.mpi_root = mpi_root
        self.psicov = False
        self.svi = False
        self.qU_ratio = 1.
        self.auto_update = auto_update

        if inference_method is None:
            from .inference import VarDTC
            if mpi_comm is None:
                inference_method = VarDTC()
            else:
                # Alex: comment out this deepgp code ->
                #inference_method = VarDTC_parallel(mpi_comm, mpi_root)
                raise NotImplemented()
                # Alex: comment out this deepgp code <-
                
        elif inference_method=='inferentia' and mpi_comm is None:
            from ..inference import VarDTC_Inferentia
            inference_method = VarDTC_Inferentia()
            self.psicov = True
        elif inference_method=='svi':
            from .inference import SVI_VarDTC
            inference_method = SVI_VarDTC()
            self.svi = True
        
        super(SparseGP_MPI, self).__init__(X, Y, Z, kernel, likelihood, mean_function=mean_function, inference_method=inference_method,
                 name=name, Y_metadata=Y_metadata, normalizer=normalizer)
        
        if self.svi:
            from .util import comp_mapping
            W = comp_mapping(self.X, self.Y)
            qu_mean = self.Z.dot(W)
            self.qU_mean = Param('qU_m', qu_mean)
            self.qU_W = Param('qU_W', np.random.randn(Z.shape[0], Z.shape[0])*0.01) 
            self.qU_a = Param('qU_a', 1e-3, Logexp())
            self.link_parameters(self.qU_mean, self.qU_W, self.qU_a)

    def parameters_changed(self):
        if self.auto_update: self.update_layer()
        
    def update_layer(self):
        self._inference_vardtc()

    def _inference_vardtc(self):
        #import pdb; pdb.set_trace()
        
        if self.svi:
            from GPy.util.linalg import tdot
            self.qU_var = tdot(self.qU_W)+np.eye(self.Z.shape[0])*self.qU_a
            self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.Z, self.likelihood, self.Y, self.qU_mean , self.qU_var, Kuu_sigma=self.Kuu_sigma if hasattr(self, 'Kuu_sigma') else None)
            
            if self.mpi_comm is None or (self.mpi_comm is not None and self.mpi_comm.rank==self.mpi_root):
                KL, dKL_dqU_mean, dKL_dqU_var, dKL_dKuu = self.inference_method.comp_KL_qU(self.qU_mean ,self.qU_var)
                self._log_marginal_likelihood += -KL*self.qU_ratio        
                self.grad_dict['dL_dqU_mean'] += -dKL_dqU_mean*self.qU_ratio
                self.grad_dict['dL_dqU_var'] += -dKL_dqU_var*self.qU_ratio
                self.grad_dict['dL_dKmm'] += -dKL_dKuu*self.qU_ratio
        else:
            self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.Z, self.likelihood, self.Y, self.Y_metadata, Kuu_sigma=self.Kuu_sigma if hasattr(self, 'Kuu_sigma') else None)

        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
        dL_dKmm = self.grad_dict['dL_dKmm']
        if (self.mpi_comm is None or (self.mpi_comm is not None and self.mpi_comm.rank==self.mpi_root)) and (hasattr(self, 'Kuu_sigma') and self.Kuu_sigma is not None):
            self.Kuu_sigma.gradient = np.diag(dL_dKmm)

        if isinstance(self.X, VariationalPosterior):
            #gradients wrt kernel
            
            if self.psicov:
                self.kern.update_gradients_expectations_psicov(variational_posterior=self.X,
                                                        Z=self.Z,
                                                        dL_dpsi0=self.grad_dict['dL_dpsi0'],
                                                        dL_dpsi1=self.grad_dict['dL_dpsi1'],
                                                        dL_dpsicov=self.grad_dict['dL_dpsicov'])
            else:
                self.kern.update_gradients_expectations(variational_posterior=self.X,
                                                        Z=self.Z,
                                                        dL_dpsi0=self.grad_dict['dL_dpsi0'],
                                                        dL_dpsi1=self.grad_dict['dL_dpsi1'],
                                                        dL_dpsi2=self.grad_dict['dL_dpsi2'])
            kerngrad = self.kern.gradient.copy()
            if self.mpi_comm is None:
                self.kern.update_gradients_full(dL_dKmm, self.Z, None)
                kerngrad += self.kern.gradient.copy()
                self.kern.gradient = kerngrad
            else:
                # Alex: comment out this deepgp code ->
#                kerngrad = reduceArrays([kerngrad], self.mpi_comm, self.mpi_root)[0]
#                if self.mpi_comm.rank==self.mpi_root:
#                    self.kern.update_gradients_full(dL_dKmm, self.Z, None)
#                    kerngrad += self.kern.gradient.copy()
#                    self.kern.gradient = kerngrad
                raise NotImplemented()
                # Alex: comment out this deepgp code <-
                
            #gradients wrt Z
            if self.psicov:
                self.Z.gradient = self.kern.gradients_Z_expectations_psicov(
                                   self.grad_dict['dL_dpsi0'],
                                   self.grad_dict['dL_dpsi1'],
                                   self.grad_dict['dL_dpsicov'],
                                   Z=self.Z,
                                   variational_posterior=self.X)
            else:
                self.Z.gradient = self.kern.gradients_Z_expectations(
                                   self.grad_dict['dL_dpsi0'],
                                   self.grad_dict['dL_dpsi1'],
                                   self.grad_dict['dL_dpsi2'],
                                   Z=self.Z,
                                   variational_posterior=self.X)
            if self.mpi_comm is None:
                self.Z.gradient += self.kern.gradients_X(dL_dKmm, self.Z)
            else:
                # Alex: comment out this deepgp code ->
#                self.Z.gradient =  reduceArrays([self.Z.gradient], self.mpi_comm, self.mpi_root)[0]
#                if self.mpi_comm.rank == self.mpi_root:
#                    self.Z.gradient += self.kern.gradients_X(dL_dKmm, self.Z)
                raise NotImplemented()
                # Alex: comment out this deepgp code <-
        else:
            #gradients wrt kernel
            self.kern.update_gradients_diag(self.grad_dict['dL_dKdiag'], self.X)
            kerngrad = self.kern.gradient.copy()
            self.kern.update_gradients_full(self.grad_dict['dL_dKnm'], self.X, self.Z)
            kerngrad += self.kern.gradient
            if self.mpi_comm is None:
                self.kern.update_gradients_full(dL_dKmm, self.Z, None)
                self.kern.gradient += kerngrad
            else:
                # Alex: comment out this deepgp code ->
#                kerngrad = reduceArrays([kerngrad], self.mpi_comm, self.mpi_root)[0]
#                if self.mpi_comm.rank==self.mpi_root:
#                    self.kern.update_gradients_full(dL_dKmm, self.Z, None)
#                    kerngrad += self.kern.gradient.copy()
#                    self.kern.gradient = kerngrad
                raise NotImplemented()
                # Alex: comment out this deepgp code <-
                
            #gradients wrt Z
            self.Z.gradient = self.kern.gradients_X(self.grad_dict['dL_dKnm'].T, self.Z, self.X)
            if self.mpi_comm is None:
                self.Z.gradient += self.kern.gradients_X(dL_dKmm, self.Z)
            else:
                # Alex: comment out this deepgp code ->
#                self.Z.gradient =  reduceArrays([self.Z.gradient], self.mpi_comm, self.mpi_root)[0]
#                if self.mpi_comm.rank == self.mpi_root:
#                    self.Z.gradient += self.kern.gradients_X(dL_dKmm, self.Z)
                raise NotImplemented()
                # Alex: comment out this deepgp code <

        if self.svi:
            self.qU_mean.gradient = self.grad_dict['dL_dqU_mean']
            self.qU_W.gradient = (self.grad_dict['dL_dqU_var']+self.grad_dict['dL_dqU_var'].T).dot(self.qU_W)
            self.qU_a.gradient = np.diag(self.grad_dict['dL_dqU_var']).sum()
            
class Layer_new(SparseGP_MPI):
    
    def __init__(self, layer_upper, Xs, X_win=0, Us=None, U_win=1, Z=None, num_inducing=10,  kernel=None, inference_method=None, 
                 likelihood=None, noise_var=1., inducing_init='kmeans',
                 back_cstr=False, MLP_dims=None, auto_update=True, minibatch_inference = False, 
                 mb_inf_tot_data_size=None, mb_inf_init_xs_means='all', mb_inf_init_xs_vars='all', 
                 mb_inf_sample_idxes=None,  name='layer'):
        """
        minibatch_inference: bool
            Minibatch inference is used
        
        mb_inf_tot_data_size: int
            Total dataset size when minibatch inference is used. It is needed to
            initialize 'self.init_Xs' when  'mb_inf_init_xs_means="all"' e.g.
            one initial value for each datapoint. If 'mb_inf_init_xs_means != "all"'
            it can be zero.
        
        mb_inf_init_xs_means: string one of ['one', 'all', 'mlp']
            How to handle initial values for for Xs. 'single' - only one parameter. 
            Does not make much sence. 'all' one parameter ofr each data point,
            'mlp' - encoded by mlp
            
        mb_inf_init_xs_vars: string one of ['all', 'one']
            'all' - single variance for each data point (not separated between init and main)
            'one' - one balue for all data points
        
        """
        self.layer_upper = layer_upper
        self.nSeq = len(Xs)
        
        self.X_win = X_win # if X_win==0, it is not autoregressive.        
        self.X_dim = Xs[0].shape[1]
        self.Xs_flat = Xs
        self.X_observed = False if isinstance(Xs[0], VariationalPosterior) else True # in deep GP code there is a separate parameter for it: uncertain_inputs
        
        self.withControl = Us is not None
        self.U_win = U_win
        self.U_dim = Us[0].shape[1] if self.withControl else None
        self.Us_flat = Us
        if self.withControl: assert len(Xs)==len(Us), "The number of signals should be equal to the number controls!"
        
        self.auto_update = auto_update
        
        assert ((minibatch_inference == back_cstr) if (minibatch_inference == True) else True), "Minibatch inference only works with back constrains"
        self.minibatch_inference = minibatch_inference
        
        #import pdb; pdb.set_trace()
        if minibatch_inference:
            assert ((mb_inf_init_xs_means != 'all') and (mb_inf_init_xs_vars != 'all')) or (mb_inf_tot_data_size is not None), "'mb_inf_tot_data_size' must be provided."
            assert ((mb_inf_init_xs_means != 'all') and (mb_inf_init_xs_vars != 'all')) or (mb_inf_sample_idxes is not None), "Data indices must be provided."
            
            # set qU_ratio ->
            assert  mb_inf_sample_idxes is not None, "Need to provide ititial indixes"
            qU_ratio = float( len(mb_inf_sample_idxes) ) / mb_inf_tot_data_size
            self.qU_ratio = qU_ratio
            # set qU_ratio <-
        
            #assert (mb_inf_init_xs_means != 'all') or (mb_inf_tot_data_size == self.nSeq), "Wrong total data size"
            
            self.mb_inf_tot_data_size = mb_inf_tot_data_size
            self.mb_inf_sample_idxes = mb_inf_sample_idxes
            
            self.mb_inf_init_xs_means = mb_inf_init_xs_means
            self.mb_inf_init_xs_vars = mb_inf_init_xs_vars
        else:
            self.qU_ratio = 1
            
        if not self.X_observed and back_cstr: self._init_encoder(MLP_dims); self.back_cstr = True
        else: self.back_cstr = False
        
        
        
        self._init_XY() 
        
        if Z is None:
            if not back_cstr and inducing_init=='kmeans':
                from sklearn.cluster import KMeans
                m = KMeans(n_clusters=num_inducing,n_init=1000,max_iter=100)
                m.fit(self.X.mean.values.copy())
                Z = m.cluster_centers_.copy()
            else:
                Z = np.random.randn(num_inducing,self.X.shape[1])
        assert Z.shape[1] == self.X.shape[1]
        
        if kernel is None: kernel = kern.RBF(self.X.shape[1], ARD = True)
        
        if inference_method is None: inference_method = VarDTC()
            
        if likelihood is None: likelihood = likelihoods.Gaussian(variance=noise_var)
        self.normalPrior, self.normalEntropy = NormalPrior(), NormalEntropy()
        super(Layer_new, self).__init__(self.X, self.Y, Z, kernel, likelihood, inference_method=inference_method, auto_update=auto_update, name=name)
        #super(Layer, self).__init__(X, Y, Z, kernel, likelihood, inference_method=inference_method, mpi_comm=mpi_comm, mpi_root=mpi_root, auto_update=auto_update, name=name)
        
        #import pdb; pdb.set_trace()
        if not self.X_observed: 
            if back_cstr:
                if self.minibatch_inference:
                    self.link_parameters(*([self.encoder,]) )
                    
                    # means ->
                    if self.mb_inf_init_xs_means == 'one':
                        self.link_parameters(*(self.init_Xs))
                    
                    elif self.mb_inf_init_xs_means == 'all':
                        self.link_parameters(*(self.init_Xs))
                        
                    elif self.mb_inf_init_xs_means == 'mlp':     
                        self.link_parameters(*(self.init_encoder ))
                
                    # vars ->
                    if self.mb_inf_init_xs_vars == 'one':
                        self.link_parameters(*(self.Xs_var))
                        
                    elif self.mb_inf_init_xs_means == 'all':
                        self.link_parameters(*(self.Xs_var))
                
                
                else:
                    self.link_parameters(*(self.init_Xs + self.Xs_var+[self.encoder]))
                    
            else:
                self.link_parameters(*self.Xs_flat)
                
    def set_inputs_and_outputs(self, batch_size, Xs=None, Us=None, samples_idxes=None):
        """
        This function is called from the model during minibatch inference.
        It updates the observed inputs and outputs for layers which have those. For
        hidden layers it only sets self.nSeq.
        
        Inputs:
        -----------------------------
        
        samples_idxes: sequence
            absolute indices of this minibatch samples, indices start from 0
        """
        
        #import pdb; pdb.set_trace()
        
        assert self.minibatch_inference, "This is developed and tested only for minibatch inference"
        assert batch_size == len(samples_idxes), "Length must be correct"
        
        self.nSeq = batch_size # new batch size

        if self.withControl == (Us is not None): 
            self.Us_flat = Us
        else:
            pass
            #raise AssertionError("Us type must be preserved")
        
        # set qU_ratio ->
        qU_ratio = float( self.nSeq ) / self.mb_inf_tot_data_size
        self.qU_ratio = qU_ratio
        # set qU_ratio <-
        
        self.Xs_flat = Xs # Need to change the sample sizes for each layer
        if self.X_observed:
            assert not isinstance(Xs[0], VariationalPosterior), "self.X_observed status must not change"
        
            #self.Xs_flat = Xs
        else:
            # init_xs means ->
            if self.mb_inf_init_xs_means == 'one':
                pass
            
            elif self.mb_inf_init_xs_means == 'all':
                # save current parameters
                for loc_ind, glob_ind in enumerate(self.mb_inf_sample_idxes):
                    self.init_Xs_all[glob_ind].mean = self.init_Xs[loc_ind].mean
                    self.init_Xs_all[glob_ind].variance = self.init_Xs[loc_ind].variance
                    
                if (self.nSeq == len(self.mb_inf_sample_idxes)): # minibatch size is the same
                    for loc_ind, glob_ind in enumerate(samples_idxes):
                        self.init_Xs[loc_ind].mean[:] = self.init_Xs_all[glob_ind].mean
                        self.init_Xs[loc_ind].variance[:] = self.init_Xs_all[glob_ind].variance
                else: # minibatch size is different
                    
                    for init_Xs in self.init_Xs:
                        self.unlink_parameter(init_Xs)
                    
                    self.init_Xs = [NormalPosterior(self.Xs_flat[i].mean.values[:self.X_win].copy(),self.Xs_flat[i].variance.values[:self.X_win].copy(), name='init_Xs_' + str(i)) for i in range(self.nSeq) ] # only initialize
                    for loc_ind, glob_ind in enumerate(samples_idxes):
                        self.init_Xs[loc_ind].mean[:] = self.init_Xs_all[glob_ind].mean
                        self.init_Xs[loc_ind].variance[:] = self.init_Xs_all[glob_ind].variance
                    
                    self.link_parameters(*self.init_Xs)
                    
                if not (self.mb_inf_init_xs_vars == 'all'):
                    self.mb_inf_sample_idxes = samples_idxes
                    
            elif self.mb_inf_init_xs_means == 'mlp':
                # init encoder has been update externaly already
                pass
        
            # init_xs vars ->
            if self.mb_inf_init_xs_vars == 'one':
                pass
                
            elif self.mb_inf_init_xs_vars == 'all':
                # save current parameters
                for loc_ind, glob_ind in enumerate(self.mb_inf_sample_idxes):
                    self.Xs_var_all[glob_ind].values[:] = self.Xs_var[loc_ind].values
                
                if (self.nSeq == len(self.mb_inf_sample_idxes)): # minibatch size is the same
                    for loc_ind, glob_ind in enumerate(samples_idxes):
                        self.Xs_var[loc_ind].values[:] = self.Xs_var_all[glob_ind].values
                else: # minibatch size is different
                    for Xs_var in self.Xs_var:
                        self.unlink_parameter(Xs_var)
                    
                    self.Xs_var = [Param('X_var_'+str(i),self.Xs_flat[i].variance.values[0].copy(), Logexp()) for i in range(self.nSeq) ] # only initialize
                    for loc_ind, glob_ind in enumerate(samples_idxes):
                        self.Xs_var[loc_ind].values[:] = self.Xs_var_all[glob_ind].values
                        
                    self.link_parameters(*(self.Xs_var))
                
                self.mb_inf_sample_idxes = samples_idxes
            
            
        self._init_XY() # without it gradient chack fails
        
    def _init_encoder(self, MLP_dims):
        from .mlp import MLP
        from copy import deepcopy
        
        X_win, X_dim, U_win, U_dim = self.X_win, self.X_dim, self.U_win, self.U_dim
        assert X_win>0, "Neural Network constraints only applies autoregressive structure!"
        Q = X_win*X_dim+U_win*U_dim if self.withControl else X_win*X_dim
        
        #import pdb; pdb.set_trace()
        
        if self.minibatch_inference:
            
            # Means ->
            if self.mb_inf_init_xs_means == 'one':
                self.init_Xs = [NormalPosterior(self.Xs_flat[0].mean.values[:X_win].copy(),self.Xs_flat[0].variance.values[:X_win].copy(), name='init_Xs_only'), ]
                
            elif self.mb_inf_init_xs_means == 'all':
                self.init_Xs_all = [NormalPosterior(self.Xs_flat[0].mean.values[:self.X_win].copy(), self.Xs_flat[0].variance.values[:self.X_win].copy(), name='init_Xs_all_'+str(i)) for i in range(self.mb_inf_tot_data_size)]
                self.init_Xs = [NormalPosterior(self.Xs_flat[i].mean.values[:X_win].copy(),self.Xs_flat[i].variance.values[:X_win].copy(), name='init_Xs_'+str(i)) for i in range(self.nSeq) ]
                
                # Why do we need second initialization here?
                # for init_X in self.init_Xs: init_X.mean[:] = np.random.randn(*init_X.shape)*1e-2
                
            elif self.mb_inf_init_xs_means == 'mlp':     
                self.init_Xs = [NormalPosterior(self.Xs_flat[i].mean.values[:X_win].copy(),self.Xs_flat[i].variance.values[:X_win].copy(), name='init_Xs_'+str(i)) for i in range(self.nSeq)]
                
                # Why do we need second initialization here?
                # for init_X in self.init_Xs: init_X.mean[:] = np.random.randn(*init_X.shape)*1e-2
            
            # Variances ->
            if self.mb_inf_init_xs_vars == 'one':
                self.Xs_var = [Param('X_var_only_'+str(0),self.Xs_flat[0].variance.values[0].copy(), Logexp()), ] # only one variance per episode
                
            elif self.mb_inf_init_xs_vars == 'all':
                self.Xs_var_all = [Param('X_var_all_'+str(i), self.Xs_flat[0].variance.values[0].copy(), Logexp()) for i in range(self.mb_inf_tot_data_size)] # only one variance per episode
                self.Xs_var = [Param('X_var_'+str(i),self.Xs_flat[i].variance.values[0].copy(), Logexp()) for i in range(self.nSeq) ]

        else:
            self.init_Xs = [NormalPosterior(self.Xs_flat[i].mean.values[:X_win],self.Xs_flat[i].variance.values[:X_win], name='init_Xs_'+str(i)) for i in range(self.nSeq)]
            for init_X in self.init_Xs: init_X.mean[:] = np.random.randn(*init_X.shape)*1e-2
        
            self.Xs_var = [Param('X_var_'+str(i),self.Xs_flat[i].variance.values[X_win:].copy(), Logexp()) for i in range(self.nSeq)]

        self.encoder = MLP([Q, Q*2, Q+X_dim/2, X_dim] if MLP_dims is None else [Q]+deepcopy(MLP_dims)+[X_dim])
    
    def _init_initialization_encoder(self, lower_layer_win, lower_layer_dim, mlp_dims=None):
        """
        This function is called from the model to initalize layer encoder which
        computes initial values for Xs.
        """
        from .mlp import MLP
        from copy import deepcopy
    
        Q = lower_layer_win * lower_layer_dim
        D = self.X_dim * self.X_win
        
        self.init_encoder = MLP([Q, Q*2, Q+D/2, D] if mlp_dims is None else [Q,]+deepcopy(mlp_dims)+[D,])
        self.link_parameters( *([self.init_encoder,]) )
    
    def _update_initial_encoder(self, lower_layer_init_data):
        """
        This function is called from the model to update the values of self.init_Xs
        """
        #import pdb; pdb.set_trace()
        
        assert (self.minibatch_inference and self.mb_inf_init_xs_means == 'mlp'), "Not right inference is used"
        
        assert self.nSeq == len(lower_layer_init_data), "Sequence number must coincide"
        
        for i,sample in enumerate(lower_layer_init_data): #iterate over sequences
            if isinstance(sample,NormalPosterior): # lower layer is not observable
                X_out = self.init_encoder.predict(sample.mean.flatten())
            else:
                X_out = self.init_encoder.predict(sample.flatten())
            self.init_Xs[i].mean[:] = X_out[0].reshape( self.init_Xs[i].mean.shape )
            #var - nothing
            
    def _init_XY(self):
        """
        Initialize variables self.X and self.Y from self.Xs_flat
        """
        
        self._update_conv()
        if self.X_win>0: X_mean_conv, X_var_conv = np.vstack(self.X_mean_conv), np.vstack(self.X_var_conv)
        if self.withControl: U_mean_conv, U_var_conv = np.vstack(self.U_mean_conv), np.vstack(self.U_var_conv)
         
        if not self.withControl:
            self.X = NormalPosterior(X_mean_conv, X_var_conv)
        elif self.X_win==0:
            self.X = NormalPosterior(U_mean_conv, U_var_conv)
        else:
            self.X = NormalPosterior(np.hstack([X_mean_conv, U_mean_conv]), np.hstack([X_var_conv, U_var_conv]))

        if self.X_observed:
            self.Y = np.vstack([x[self.X_win:] for x in self.Xs_flat])
        else:
            self.Y = NormalPosterior(np.vstack([x.mean.values[self.X_win:] for x in self.Xs_flat]), np.vstack([x.variance.values[self.X_win:] for x in self.Xs_flat]))
    
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
        """
        Intermidiate function used in self._init_XY and self._update_X.
        Also if self.back_cstr run the decoder to obtain self.Xs_flat
        """
        
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
        """
        Update variables self.X and self.Y from self.Xs_flat
        """
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
        """
        Compute updates of gradients "self.Xs_flat" and "self.Us_flat" from gradients of "self.X" 
        """
        
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
        """
        Compute parts of of gradients "self.Xs_flat" from gradients of self.Y
        """
        #import pdb; pdb.set_trace()
        
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
                
                
                if not self.svi:
                    X.variance.gradient[X_win:] += self.grad_dict['dL_dYvar'][Y_offset:Y_offset+N,None]
                else:
                    X.variance.gradient[X_win:] += self.grad_dict['dL_dYvar'][Y_offset:Y_offset+N]
                
                if X_win>0: # What about encoder? are these affect the encoder?
                    delta += -self.normalPrior.comp_value(X[:X_win])
                    self.normalPrior.update_gradients(X[:X_win])
                    
                delta += -self.normalEntropy.comp_value(X[X_win:])
                self.normalEntropy.update_gradients(X[X_win:])
                Y_offset += N
            self._log_marginal_likelihood += delta
    
    def update_layer(self):
        self._update_X() # Update variables self.X and self.Y from self.Xs_flat
        super(Layer_new,self).update_layer() 
        self._update_qX_gradients() # computes gradients wrt self.X
        self._prepare_gradients() # computes parts of gradients "self.Xs_flat" from gradients of self.Y
            
    def _encoder_freerun(self):
        """
        This function updates X_flat after parameters have changed
        """
        
        X_win, X_dim, U_win, U_dim = self.X_win, self.X_dim, self.U_win, self.U_dim        
        Q = X_win*X_dim+U_win*U_dim if self.withControl else X_win*X_dim
        
        #import pdb; pdb.set_trace()
        
        X_in = np.zeros((Q,))
        for i_seq in range(self.nSeq):
            if not self.minibatch_inference:
                X_flat, init_X, X_var = self.Xs_flat[i_seq], self.init_Xs[i_seq], self.Xs_var[i_seq]
            else:
                # init means ->
                if self.mb_inf_init_xs_means == 'one':
                    X_flat, init_X = self.Xs_flat[i_seq], self.init_Xs[0]
                    
                elif self.mb_inf_init_xs_means == 'all':
                    X_flat, init_X = self.Xs_flat[i_seq], self.init_Xs[i_seq]
                    
                # init vas
                if self.mb_inf_init_xs_vars == 'one':
                    X_var = self.Xs_var[0]
                    
                elif self.mb_inf_init_xs_vars == 'all':
                    X_var = self.Xs_var[i_seq]
            
            #import pdb; pdb.set_trace()
            
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
        """
        Updates the gradient of L wrt q_X_init (and q_X), taking into account that each value of q_X
        was used to generate future values.
        """
        self.encoder.prepare_grad() # zero all theano gradients       
        X_win, X_dim, U_win, U_dim = self.X_win, self.X_dim, self.U_win, self.U_dim
        Q = X_win*X_dim+U_win*U_dim if self.withControl else X_win*X_dim
        
        X_in = np.zeros((Q,))
        dL =np.zeros((X_dim,))
                
        #import pdb; pdb.set_trace()
            
        for i_seq in range(self.nSeq):
            if not self.minibatch_inference:
                X_flat, init_X, X_var = self.Xs_flat[i_seq], self.init_Xs[i_seq], self.Xs_var[i_seq]
            else:
                # init means ->
                if self.mb_inf_init_xs_means == 'one':
                    X_flat, init_X = self.Xs_flat[i_seq], self.init_Xs[0]
                    
                elif self.mb_inf_init_xs_means == 'all':
                    X_flat, init_X = self.Xs_flat[i_seq], self.init_Xs[i_seq]
                    
                # init vars
                if self.mb_inf_init_xs_vars == 'one':
                    X_var = self.Xs_var[0]
                    
                elif self.mb_inf_init_xs_vars == 'all':
                    X_var = self.Xs_var[i_seq]
                
            if self.withControl: U_flat = self.Us_flat[i_seq]
            N = X_flat.shape[0] - X_win
            if self.withControl: U_offset = U_flat.shape[0]-N-U_win+1
            
            for n in range(N-1,-1,-1):
                X_in[:X_win*X_dim] = X_flat.mean[n:n+X_win].flat
                if self.withControl: 
                    X_in[X_win*X_dim:] = U_flat.mean[U_offset+n:U_offset+n+U_win].flat
                dL[:] = X_flat.mean.gradient[X_win+n].flat
                dX = self.encoder.update_gradient(X_in[None,:], dL[None,:]) # Set gradient from theano to Python parameter
                X_flat.mean.gradient[n:n+X_win] += dX[0,:X_win*X_dim].reshape(-1,X_dim)
                if self.withControl:
                    U_flat.mean.gradient[U_offset+n:U_offset+n+U_win] += dX[0, X_win*X_dim:].reshape(-1,U_dim)
            
            #import pdb; pdb.set_trace()
            
            if not self.minibatch_inference: # not minibatch inference
                # Gradients wrt initial values and variational variances are handeled regularly
                init_X.mean.gradient[:] = X_flat.mean.gradient[:X_win]
                init_X.variance.gradient[:] = X_flat.variance.gradient[:X_win]
            
                X_var.gradient[:] = X_flat.variance.gradient[X_win:]
            else:
                
                # Means ->
                if self.mb_inf_init_xs_means == 'one':
                    if i_seq == 0: 
                        init_X.mean.gradient[:] = X_flat.mean.gradient[:X_win]
                        init_X.variance.gradient[:] = X_flat.variance.gradient[:X_win]
                        
                    else:
                        init_X.mean.gradient[:] += X_flat.mean.gradient[:X_win]
                        init_X.variance.gradient[:] += X_flat.variance.gradient[:X_win]
                    
                elif self.mb_inf_init_xs_means == 'all':
                    init_X.mean.gradient[:] = X_flat.mean.gradient[:X_win]
                    init_X.variance.gradient[:] = X_flat.variance.gradient[:X_win]
                    
                elif self.mb_inf_init_xs_means == 'mlp':     
                    raise NotImplemented("sdfbdsf")
                
                # Variances ->
                if self.mb_inf_init_xs_vars == 'one':
                    if i_seq == 0: 
                        X_var.gradient = np.sum( X_flat.variance.gradient[X_win:], axis=0 )
                    else:
                        X_var.gradient += np.sum( X_flat.variance.gradient[X_win:], axis=0 )
                    
                elif self.mb_inf_init_xs_vars == 'all':
                    #if i_seq == 0: 
                    X_var.gradient = np.sum( X_flat.variance.gradient[X_win:], axis=0 )
                    #else:
                    #    X_var.gradient += np.sum( X_flat.variance.gradient[X_win:], axis=0 )
                    
            # update gradients of initial parameters MLPs ->
            
            # update gradients of initial parameters MLPs <-
    
    def _encoder_update_initial_gradient(self, low_input, top_grad=None):
        """
        This function computes the gradient wrt parameters of MLPs which
        encode initial values of Xs.
        
        Input:
        --------------------
        top_grad: array
            Gradient of ELBO wrt inputs of the previous layer (these are outputs of this layer).
        
        low_input: array
            Initial values of the lower layer or observations if lower layer is an observed one.
        """
        import pdb; pdb.set_trace()
        
        self.init_encoder.prepare_grad() # zero all theano gradients
        
        init_In = np.zeros( self.init_Xs[0].mean[:self.X_win].shape )
        init_In_grad = np.zeros( self.init_Xs[0].gradient[:self.X_win].shape )
        
        dX_2 = []
        for i_seq in range(self.nSeq):
            
            init_In[:] = self.init_Xs[i_seq].mean[:self.X_win]
            init_In_grad[:] = self.init_Xs[i_seq].gradient[:self.X_win]
            
            dX_1 = self.init_ecoder.update_gradient(low_input[i_seq], init_In_grad)
            
            if top_grad is not None:
                dX_2.append( self.init_ecoder.update_gradient(init_In, top_grad[i_seq] ) )
            
        return dX_2
            
      
    def freerun(self, init_Xs=None, step=None, U=None, m_match=True, encoder=False):
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
        encoder = encoder and self.back_cstr
        
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
        elif encoder:
            X = np.empty((X_win+step, X_dim))
            X_in = np.empty((1,Q))
            if X_win>0: X[:X_win] = init_Xs[-X_win:]
            for n in range(step):
                if X_win>0: X_in[0,:X_win*X_dim] = X[n:n+X_win].flat
                if self.withControl: X_in[0,X_win*X_dim:] = U[n:n+U_win].flat
                X[X_win+n] = self.encoder.predict(X_in)[0]
        else:
            X = np.empty((X_win+step, X_dim))
            X_in = np.empty((1,Q))
            if X_win>0: X[:X_win] = init_Xs[-X_win:]
            for n in range(step):
                if X_win>0: X_in[0,:X_win*X_dim] = X[n:n+X_win].flat
                if self.withControl: X_in[0,X_win*X_dim:] = U[n:n+U_win].flat
                X[X_win+n] = self._raw_predict(X_in)[0]
        return X

class Layer_rnn(SparseGP_MPI):
    
    def __init__(self, layer_upper, Xs, X_win=0, Us=None, U_win=1, Z=None, num_inducing=10,  kernel=None, inference_method=None, 
                 likelihood=None, noise_var=1., inducing_init='kmeans',
                 back_cstr=False, auto_update=True, minibatch_inference = False, 
                 mb_inf_tot_data_size=None, mb_inf_sample_idxes=None,  name='layer'):
        """
        minibatch_inference: bool
            Minibatch inference is used
        
        mb_inf_tot_data_size: int
            Total dataset size when minibatch inference is used. It is needed to
            initialize 'self.init_Xs' when  'mb_inf_init_xs_means="all"' e.g.
            one initial value for each datapoint. If 'mb_inf_init_xs_means != "all"'
            it can be zero.
        
        mb_inf_init_xs_means: string one of ['one', 'all', 'mlp']
            How to handle initial values for for Xs. 'single' - only one parameter. 
            Does not make much sence. 'all' one parameter ofr each data point,
            'mlp' - encoded by mlp
            
        mb_inf_init_xs_vars: string one of ['all', 'one']
            'all' - single variance for each data point (not separated between init and main)
            'one' - one balue for all data points
        
        """
        self.layer_upper = layer_upper
        self.nSeq = len(Xs)
        
        self.X_win = X_win # if X_win==0, it is not autoregressive.        
        self.X_dim = Xs[0].shape[1]
        self.Xs_flat = Xs
        self.X_observed = False if isinstance(Xs[0], VariationalPosterior) else True # in deep GP code there is a separate parameter for it: uncertain_inputs
        
        self.withControl = Us is not None
        self.U_win = U_win
        self.U_dim = Us[0].shape[1] if self.withControl else None
        self.Us_flat = Us
        if self.withControl: assert len(Xs)==len(Us), "The number of signals should be equal to the number controls!"
        
        self.auto_update = auto_update
        
        assert ((minibatch_inference == back_cstr) if (minibatch_inference == True) else True), "Minibatch inference only works with back constrains"
        self.minibatch_inference = minibatch_inference
        
        assert back_cstr==True, "Must be true for this model"
        #import pdb; pdb.set_trace()
        if minibatch_inference:
            # set qU_ratio ->
            assert  mb_inf_sample_idxes is not None, "Need to provide ititial indixes"
            qU_ratio = float( len(mb_inf_sample_idxes) ) / mb_inf_tot_data_size
            self.qU_ratio = qU_ratio
            # set qU_ratio <-
        
            #assert (mb_inf_init_xs_means != 'all') or (mb_inf_tot_data_size == self.nSeq), "Wrong total data size"
            
            self.mb_inf_tot_data_size = mb_inf_tot_data_size
            self.mb_inf_sample_idxes = mb_inf_sample_idxes
        else:
            self.qU_ratio = 1
            
        if not self.X_observed and back_cstr: self.back_cstr = True
        else: self.back_cstr = False # TODO: Check this
        
        self._init_XY() 
        
        if Z is None:
            if not back_cstr and inducing_init=='kmeans':
                from sklearn.cluster import KMeans
                m = KMeans(n_clusters=num_inducing,n_init=1000,max_iter=100)
                m.fit(self.X.mean.values.copy())
                Z = m.cluster_centers_.copy()
            else:
                Z = np.random.randn(num_inducing,self.X.shape[1])
        assert Z.shape[1] == self.X.shape[1]
        
        if kernel is None: kernel = kern.RBF(self.X.shape[1], ARD = True)
        
        if inference_method is None: inference_method = VarDTC()
            
        if likelihood is None: likelihood = likelihoods.Gaussian(variance=noise_var)
        self.normalPrior, self.normalEntropy = NormalPrior(), NormalEntropy()
        super(Layer_rnn, self).__init__(self.X, self.Y, Z, kernel, likelihood, inference_method=inference_method, auto_update=auto_update, name=name)
        #super(Layer, self).__init__(X, Y, Z, kernel, likelihood, inference_method=inference_method, mpi_comm=mpi_comm, mpi_root=mpi_root, auto_update=auto_update, name=name)
        
        #import pdb; pdb.set_trace()
        if not self.X_observed: 
            if back_cstr:
                if self.minibatch_inference:
                    pass
                else:
                    self.link_parameters(*(self.init_Xs + self.Xs_var+[self.encoder]))
                    
            else:
                self.link_parameters(*self.Xs_flat)
                
    def set_inputs_and_outputs(self, batch_size, Xs=None, Us=None, samples_idxes=None):
        """
        This function is called from the model during minibatch inference.
        It updates the observed inputs and outputs for layers which have those. For
        hidden layers it only sets self.nSeq.
        
        Inputs:
        -----------------------------
        
        samples_idxes: sequence
            absolute indices of this minibatch samples, indices start from 0
        """
        
        #import pdb; pdb.set_trace()
        
        assert self.minibatch_inference, "This is developed and tested only for minibatch inference"
        assert batch_size == len(samples_idxes), "Length must be correct"
        
        self.nSeq = batch_size # new batch size

        if self.withControl == (Us is not None): 
            self.Us_flat = Us
        else:
            pass
            #raise AssertionError("Us type must be preserved")
        
        # set qU_ratio ->
        qU_ratio = float( self.nSeq ) / self.mb_inf_tot_data_size
        self.qU_ratio = qU_ratio
        # set qU_ratio <-
        
        self.Xs_flat = Xs # From model encoder
        if self.X_observed:
            assert not isinstance(Xs[0], VariationalPosterior), "self.X_observed status must not change"
        self._init_XY() # without it gradient chack fails
        
#    def _init_encoder(self, MLP_dims):
#        from .mlp import MLP
#        from copy import deepcopy
#        
#        X_win, X_dim, U_win, U_dim = self.X_win, self.X_dim, self.U_win, self.U_dim
#        assert X_win>0, "Neural Network constraints only applies autoregressive structure!"
#        Q = X_win*X_dim+U_win*U_dim if self.withControl else X_win*X_dim
#        
#        #import pdb; pdb.set_trace()
#        
#        if self.minibatch_inference:
#            
#            # Means ->
#            if self.mb_inf_init_xs_means == 'one':
#                self.init_Xs = [NormalPosterior(self.Xs_flat[0].mean.values[:X_win].copy(),self.Xs_flat[0].variance.values[:X_win].copy(), name='init_Xs_only'), ]
#                
#            elif self.mb_inf_init_xs_means == 'all':
#                self.init_Xs_all = [NormalPosterior(self.Xs_flat[0].mean.values[:self.X_win].copy(), self.Xs_flat[0].variance.values[:self.X_win].copy(), name='init_Xs_all_'+str(i)) for i in range(self.mb_inf_tot_data_size)]
#                self.init_Xs = [NormalPosterior(self.Xs_flat[i].mean.values[:X_win].copy(),self.Xs_flat[i].variance.values[:X_win].copy(), name='init_Xs_'+str(i)) for i in range(self.nSeq) ]
#                
#                # Why do we need second initialization here?
#                # for init_X in self.init_Xs: init_X.mean[:] = np.random.randn(*init_X.shape)*1e-2
#                
#            elif self.mb_inf_init_xs_means == 'mlp':     
#                self.init_Xs = [NormalPosterior(self.Xs_flat[i].mean.values[:X_win].copy(),self.Xs_flat[i].variance.values[:X_win].copy(), name='init_Xs_'+str(i)) for i in range(self.nSeq)]
#                
#                # Why do we need second initialization here?
#                # for init_X in self.init_Xs: init_X.mean[:] = np.random.randn(*init_X.shape)*1e-2
#            
#            # Variances ->
#            if self.mb_inf_init_xs_vars == 'one':
#                self.Xs_var = [Param('X_var_only_'+str(0),self.Xs_flat[0].variance.values[0].copy(), Logexp()), ] # only one variance per episode
#                
#            elif self.mb_inf_init_xs_vars == 'all':
#                self.Xs_var_all = [Param('X_var_all_'+str(i), self.Xs_flat[0].variance.values[0].copy(), Logexp()) for i in range(self.mb_inf_tot_data_size)] # only one variance per episode
#                self.Xs_var = [Param('X_var_'+str(i),self.Xs_flat[i].variance.values[0].copy(), Logexp()) for i in range(self.nSeq) ]
#
#        else:
#            self.init_Xs = [NormalPosterior(self.Xs_flat[i].mean.values[:X_win],self.Xs_flat[i].variance.values[:X_win], name='init_Xs_'+str(i)) for i in range(self.nSeq)]
#            for init_X in self.init_Xs: init_X.mean[:] = np.random.randn(*init_X.shape)*1e-2
#        
#            self.Xs_var = [Param('X_var_'+str(i),self.Xs_flat[i].variance.values[X_win:].copy(), Logexp()) for i in range(self.nSeq)]
#
#        self.encoder = MLP([Q, Q*2, Q+X_dim/2, X_dim] if MLP_dims is None else [Q]+deepcopy(MLP_dims)+[X_dim])
            
    def _init_XY(self):
        """
        Initialize variables self.X and self.Y from self.Xs_flat
        """
        
        self._update_conv()
        if self.X_win>0: X_mean_conv, X_var_conv = np.vstack(self.X_mean_conv), np.vstack(self.X_var_conv)
        if self.withControl: U_mean_conv, U_var_conv = np.vstack(self.U_mean_conv), np.vstack(self.U_var_conv)
         
        if not self.withControl:
            self.X = NormalPosterior(X_mean_conv, X_var_conv)
        elif self.X_win==0:
            self.X = NormalPosterior(U_mean_conv, U_var_conv)
        else:
            self.X = NormalPosterior(np.hstack([X_mean_conv, U_mean_conv]), np.hstack([X_var_conv, U_var_conv]))

        if self.X_observed:
            self.Y = np.vstack([x[self.X_win:] for x in self.Xs_flat])
        else:
            self.Y = NormalPosterior(np.vstack([x.mean.values[self.X_win:] for x in self.Xs_flat]), np.vstack([x.variance.values[self.X_win:] for x in self.Xs_flat]))
    
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
        """
        Intermidiate function used in self._init_XY and self._update_X.
        Also if self.back_cstr run the decoder to obtain self.Xs_flat
        """
        #import pdb; pdb.set_trace()
        
        X_win, X_dim, U_win, U_dim = self.X_win, self.X_dim, self.U_win, self.U_dim
        #if self.back_cstr: self._encoder_freerun()
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
        """
        Update variables self.X and self.Y from self.Xs_flat
        """
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
        """
        Compute updates of gradients "self.Xs_flat" and "self.Us_flat" from gradients of "self.X" 
        """
        
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
        #if self.back_cstr: self._encoder_update_gradient()
        
    def _update_qX_gradients(self):
        self.X.mean.gradient, self.X.variance.gradient = self.kern.gradients_qX_expectations(
                                            variational_posterior=self.X,
                                            Z=self.Z,
                                            dL_dpsi0=self.grad_dict['dL_dpsi0'],
                                            dL_dpsi1=self.grad_dict['dL_dpsi1'],
                                            dL_dpsi2=self.grad_dict['dL_dpsi2'])
    
    def _prepare_gradients(self):
        """
        Compute parts of of gradients "self.Xs_flat" from gradients of self.Y
        """
        #import pdb; pdb.set_trace()
        
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
                
                
                if not self.svi:
                    X.variance.gradient[X_win:] += self.grad_dict['dL_dYvar'][Y_offset:Y_offset+N,None]
                else:
                    X.variance.gradient[X_win:] += self.grad_dict['dL_dYvar'][Y_offset:Y_offset+N]
                
                if X_win>0: # What about encoder? are these affect the encoder?
                    delta += -self.normalPrior.comp_value(X[:X_win])
                    self.normalPrior.update_gradients(X[:X_win])
                    
                delta += -self.normalEntropy.comp_value(X[X_win:])
                self.normalEntropy.update_gradients(X[X_win:])
                Y_offset += N
            self._log_marginal_likelihood += delta
    
    def update_layer(self):
        self._update_X() # Update variables self.X and self.Y from self.Xs_flat
        super(Layer_rnn,self).update_layer() 
        self._update_qX_gradients() # computes gradients wrt self.X
        self._prepare_gradients() # computes parts of gradients "self.Xs_flat" from gradients of self.Y
            
#    def _encoder_freerun(self):
#        """
#        This function updates X_flat after parameters have changed
#        """
#        
#        X_win, X_dim, U_win, U_dim = self.X_win, self.X_dim, self.U_win, self.U_dim        
#        Q = X_win*X_dim+U_win*U_dim if self.withControl else X_win*X_dim
#        
#        #import pdb; pdb.set_trace()
#        
#        X_in = np.zeros((Q,))
#        for i_seq in range(self.nSeq):
#            if not self.minibatch_inference:
#                X_flat, init_X, X_var = self.Xs_flat[i_seq], self.init_Xs[i_seq], self.Xs_var[i_seq]
#            else:
#                # init means ->
#                if self.mb_inf_init_xs_means == 'one':
#                    X_flat, init_X = self.Xs_flat[i_seq], self.init_Xs[0]
#                    
#                elif self.mb_inf_init_xs_means == 'all':
#                    X_flat, init_X = self.Xs_flat[i_seq], self.init_Xs[i_seq]
#                    
#                # init vas
#                if self.mb_inf_init_xs_vars == 'one':
#                    X_var = self.Xs_var[0]
#                    
#                elif self.mb_inf_init_xs_vars == 'all':
#                    X_var = self.Xs_var[i_seq]
#            
#            #import pdb; pdb.set_trace()
#            
#            if self.withControl: U_flat = self.Us_flat[i_seq]
#            X_flat.mean[:X_win] = init_X.mean.values
#            X_flat.variance[:X_win] = init_X.variance.values
#            X_flat.variance[X_win:] = X_var.values
#            
#            N = X_flat.shape[0] - X_win
#            if self.withControl: U_offset = U_flat.shape[0]-N-U_win+1
#            for n in range(N):
#                X_in[:X_win*X_dim] = X_flat.mean[n:n+X_win].flat
#                if self.withControl: 
#                    X_in[X_win*X_dim:] = U_flat.mean[U_offset+n:U_offset+n+U_win].flat
#                X_out = self.encoder.predict(X_in[None,:])
#                X_flat.mean[X_win+n] = X_out[0]
    
#    def _encoder_update_gradient(self):
#        """
#        Updates the gradient of L wrt q_X_init (and q_X), taking into account that each value of q_X
#        was used to generate future values.
#        """
#        self.encoder.prepare_grad() # zero all theano gradients       
#        X_win, X_dim, U_win, U_dim = self.X_win, self.X_dim, self.U_win, self.U_dim
#        Q = X_win*X_dim+U_win*U_dim if self.withControl else X_win*X_dim
#        
#        X_in = np.zeros((Q,))
#        dL =np.zeros((X_dim,))
#                
#        #import pdb; pdb.set_trace()
#            
#        for i_seq in range(self.nSeq):
#            X_flat = self.Xs_flat[i_seq]
#                
#            if self.withControl: U_flat = self.Us_flat[i_seq]
#            N = X_flat.shape[0] - X_win
#            if self.withControl: U_offset = U_flat.shape[0]-N-U_win+1
#            
#            for n in range(N-1,-1,-1):
#                X_in[:X_win*X_dim] = X_flat.mean[n:n+X_win].flat
#                if self.withControl: 
#                    X_in[X_win*X_dim:] = U_flat.mean[U_offset+n:U_offset+n+U_win].flat
#                dL[:] = X_flat.mean.gradient[X_win+n].flat
#                dX = self.encoder.update_gradient(X_in[None,:], dL[None,:]) # Set gradient from theano to Python parameter
#                X_flat.mean.gradient[n:n+X_win] += dX[0,:X_win*X_dim].reshape(-1,X_dim)
#                if self.withControl:
#                    U_flat.mean.gradient[U_offset+n:U_offset+n+U_win] += dX[0, X_win*X_dim:].reshape(-1,U_dim)
            
            #import pdb; pdb.set_trace()
            
    def freerun(self, init_Xs=None, step=None, U=None, m_match=True, encoder=False):
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
        encoder = encoder and self.back_cstr
        
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
        elif encoder:
            X = np.empty((X_win+step, X_dim))
            X_in = np.empty((1,Q))
            if X_win>0: X[:X_win] = init_Xs[-X_win:]
            for n in range(step):
                if X_win>0: X_in[0,:X_win*X_dim] = X[n:n+X_win].flat
                if self.withControl: X_in[0,X_win*X_dim:] = U[n:n+U_win].flat
                X[X_win+n] = self.encoder.predict(X_in)[0]
        else:
            X = np.empty((X_win+step, X_dim))
            X_in = np.empty((1,Q))
            if X_win>0: X[:X_win] = init_Xs[-X_win:]
            for n in range(step):
                if X_win>0: X_in[0,:X_win*X_dim] = X[n:n+X_win].flat
                if self.withControl: X_in[0,X_win*X_dim:] = U[n:n+U_win].flat
                X[X_win+n] = self._raw_predict(X_in)[0]
        return X
    
class Layer(SparseGP):
    
    def __init__(self, layer_upper, Xs, X_win=0, Us=None, U_win=1, Z=None, num_inducing=10,  kernel=None, inference_method=None, 
                 likelihood=None, noise_var=1., inducing_init='kmeans',
                 back_cstr=False, MLP_dims=None,name='layer'):

        #import pdb; pdb.set_trace() # Alex
        
        self.layer_upper = layer_upper
        self.nSeq = len(Xs)

        self.X_win = X_win # if X_win==0, it is not autoregressive.        
        self.X_dim = Xs[0].shape[1]
        self.Xs_flat = Xs
        self.X_observed = False if isinstance(Xs[0], VariationalPosterior) else True 
        
        self.withControl = Us is not None
        self.U_win = U_win
        self.U_dim = Us[0].shape[1] if self.withControl else None
        self.Us_flat = Us
        if self.withControl: assert len(Xs)==len(Us), "The number of signals should be equal to the number controls!"
        
        if not self.X_observed and back_cstr: self._init_encoder(MLP_dims); self.back_cstr = True
        else: self.back_cstr = False
        self._init_XY() 
        
        if Z is None:
            if not back_cstr and inducing_init=='kmeans':
                from sklearn.cluster import KMeans
                m = KMeans(n_clusters=num_inducing,n_init=1000,max_iter=100)
                m.fit(self.X.mean.values.copy())
                Z = m.cluster_centers_.copy()
            else:
                Z = np.random.randn(num_inducing,self.X.shape[1])
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
        #import pdb; pdb.set_trace() # Alex
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
        #import pdb; pdb.set_trace() # Alex
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
                X_in[:X_win*X_dim] = X_flsat.mean[n:n+X_win].flat
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
                
    def freerun(self, init_Xs=None, step=None, U=None, m_match=True, encoder=False):
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
        encoder = encoder and self.back_cstr
        
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
        elif encoder:
            X = np.empty((X_win+step, X_dim))
            X_in = np.empty((1,Q))
            if X_win>0: X[:X_win] = init_Xs[-X_win:]
            for n in range(step):
                if X_win>0: X_in[0,:X_win*X_dim] = X[n:n+X_win].flat
                if self.withControl: X_in[0,X_win*X_dim:] = U[n:n+U_win].flat
                X[X_win+n] = self.encoder.predict(X_in)[0]
        else:
            X = np.empty((X_win+step, X_dim))
            X_in = np.empty((1,Q))
            if X_win>0: X[:X_win] = init_Xs[-X_win:]
            for n in range(step):
                if X_win>0: X_in[0,:X_win*X_dim] = X[n:n+X_win].flat
                if self.withControl: X_in[0,X_win*X_dim:] = U[n:n+U_win].flat
                X[X_win+n] = self._raw_predict(X_in)[0]
        return X
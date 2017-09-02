
import sys
import numpy as np
from scipy.linalg import LinAlgError
from GPy import Model,likelihoods
from GPy.core.parameterization.variational import VariationalPosterior,\
    NormalPosterior
from .layers import Layer
from .layers import Layer_new, Layer_rnn
try:
    from .rnn_encoder import seq_encoder
except ImportError:
    pass

class DeepAutoreg_new(Model):
    """
    :param wins: windows sizes of layers
    :type wins: list[num of layers]. 0-th element correspond to output layer.
    
    :param U_pre_step: If true, the current signal is assumed to be controlled by the control signal of the previous time step.
    :type U_pre_step: Boolean
    
    :param nDims: output dimensions of layers
    :type nDims: list[num of layers]. 0-th element is output dimensionality.
    
    """
    
    def __init__(self, wins, Y, U=None, U_win=1, nDims=None, X_variance=0.01, num_inducing=10, 
                 likelihood = None, name='autoreg', kernels=None, U_pre_step=True, init='Y', 
                 inducing_init='kmeans', back_cstr=False, MLP_dims=None, inference_method=None,
                 minibatch_inference=False, mb_inf_tot_data_size = None,
                 mb_inf_init_xs_means='all',mb_inf_init_xs_vars='all',
                 mb_inf_sample_idxes=None):
        super(DeepAutoreg_new, self).__init__(name=name)
        
        Ys, Us = Y,U
        if isinstance(Ys, np.ndarray): Ys = [Ys]
        if Us is not None and isinstance(Us, np.ndarray): Us = [Us]
        
        # If data_streamer is planned to be used, here the first batch must be provided as input data
        self.data_streamer = None
        
        self.nLayers = len(wins)
        self.back_cstr = back_cstr
        self.wins = wins
        self.U_win = U_win
        #self.input_dim = 1
        #self.output_dim = 1
        self._log_marginal_likelihood = np.nan
        self.U_pre_step = U_pre_step
        self.nDims = nDims if nDims is not None else [Ys[0].shape[1]]+[1]*(self.nLayers-1)
        
        if Us is not None:
            assert len(Ys)==len(Us)
            self.Us = []
            self.Ys = []
            for i in range(len(Ys)):
                Y, U = Ys[i], Us[i]
#                 assert Y.shape[0]==U.shape[0], "the signal and control should be aligned."
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

        #import pdb; pdb.set_trace()
        
        Xs = self._init_X(wins, self.Ys, self.Us, X_variance, init=init, nDims=self.nDims)
        
        
        self.auto_update = False if inference_method=='svi' else True
        auto_update = self.auto_update
        
        # Parameters which exist differently per layer but specified as single componenents are here expanded to each layer
        if not isinstance(num_inducing, list or tuple): num_inducing = [num_inducing]*self.nLayers

        
        if minibatch_inference:
            assert (mb_inf_init_xs_means != 'all' and mb_inf_init_xs_vars != 'all') or ( mb_inf_tot_data_size is not None and mb_inf_sample_idxes is not None), "Total data size and initial indices must be provided"
            self.mb_inf_init_xs_means = mb_inf_init_xs_means
            self.mb_inf_init_xs_vars = mb_inf_init_xs_vars
            self.minibatch_inference = minibatch_inference
            self.mb_inf_tot_data_size = mb_inf_tot_data_size
        else:
            self.minibatch_inference = False
            
        # Initialize Layers. The top layer goes first in the list
        self.layers = []
        for i in range(self.nLayers-1,-1,-1):
            if i==self.nLayers-1: # Top layer
                self.layers.append(Layer_new(None, Xs[i-1], X_win=wins[i], Us=self.Us, U_win=U_win, num_inducing=num_inducing[i],  kernel=kernels[i] if kernels is not None else None, 
                                         noise_var=0.01, name='layer_'+str(i),  back_cstr=back_cstr, MLP_dims=MLP_dims, inducing_init=inducing_init, auto_update=auto_update, inference_method=inference_method,
                                         minibatch_inference=minibatch_inference, mb_inf_tot_data_size=mb_inf_tot_data_size, mb_inf_init_xs_means=mb_inf_init_xs_means, mb_inf_init_xs_vars=mb_inf_init_xs_vars,
                                         mb_inf_sample_idxes = mb_inf_sample_idxes))
            elif i==0: # Observed layer
                self.layers.append(Layer_new(self.layers[-1], self.Ys, X_win=wins[i], Us=Xs[i], U_win=wins[i+1], num_inducing=num_inducing[i],  kernel=kernels[i] if kernels is not None else None, 
                                         likelihood=likelihood, noise_var=1., back_cstr=back_cstr, name='layer_'+str(i), inducing_init=inducing_init, auto_update=auto_update, inference_method=inference_method,
                                         minibatch_inference=minibatch_inference, mb_inf_tot_data_size=mb_inf_tot_data_size, mb_inf_init_xs_means=mb_inf_init_xs_means, mb_inf_init_xs_vars=mb_inf_init_xs_vars,
                                         mb_inf_sample_idxes = mb_inf_sample_idxes))
            else: # Observed layer Other layers
                self.layers.append(Layer_new(self.layers[-1], Xs[i-1], X_win=wins[i], Us=Xs[i], U_win=wins[i+1], num_inducing=num_inducing[i],  kernel=kernels[i] if kernels is not None else None, 
                                         noise_var=0.01, name='layer_'+str(i), back_cstr=back_cstr, MLP_dims=MLP_dims, inducing_init=inducing_init, auto_update=auto_update, inference_method=inference_method,
                                         minibatch_inference=minibatch_inference, mb_inf_tot_data_size=mb_inf_tot_data_size, mb_inf_init_xs_means=mb_inf_init_xs_means, mb_inf_init_xs_vars=mb_inf_init_xs_vars,
                                         mb_inf_sample_idxes = mb_inf_sample_idxes))
        

            
        #  Initialize MLPs for back_cstr initial vlaues ->
        if (minibatch_inference and  mb_inf_init_xs_means=='mlp'):
            import pdb; pdb.set_trace()
            
            layer_0 = self.layers[-1] # observable layer
            previous_layer_dim = layer_0.X_dim 
            previous_layer_window = self.layers[-2].X_win # assume that window on observation layer equals the window on first hidden layer. Need this for a code to work
            
            for i in range(self.nLayers-2,-1,-1): # ignore observable layer, start from lowest after observation
                self.layers[i]._init_initialization_encoder(previous_layer_window,previous_layer_dim, [ (previous_layer_window*previous_layer_dim)**2, ])
                previous_layer_window = self.layers[i].X_win
                previous_layer_dim = self.layers[i].X_dim
        #  Initialize MLPs for back_cstr initial vlaues <-
            
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
                elif init=='nan':
                    mean = np.empty((U_win+Y.shape[0], U_dim))
                    var = np.empty((U_win+Y.shape[0],U_dim))
                    X.append(NormalPosterior(mean,var,name='qX_'+str(i_seq)))
            Xs.append(X)
        return Xs
        
    def log_likelihood(self):
        return self._log_marginal_likelihood
        
    def parameters_changed(self):
        #import pdb; pdb.set_trace()
        
        # generate initial values for layer back_cstr recurssion ->
        if (self.minibatch_inference and self.mb_inf_init_xs_means=='mlp'):
            
            previous_layer_data = [ Y[ 0:self.layers[-2].X_win ] for Y in self.Ys ]# observable layer
            
            for i in range(self.nLayers-2,-1,-1): # ignore observable layer, start from second after observable
                self.layers[i]._update_initial_encoder(previous_layer_data)
                previous_layer_data = self.layers[i].init_Xs
            
        # generate initial values for layer back_cstr recurssion <-        
                
        if not self.auto_update: [l.update_layer() for l in self.layers] # starting from top layer (with inputs)
        self._log_marginal_likelihood = np.sum([l._log_marginal_likelihood for l in self.layers])
        [l.update_latent_gradients() for l in self.layers[::-1]] # start from lowest layer.
        
        # update gradients of initial values recursion parameters ->
        if (self.minibatch_inference and  self.mb_inf_init_xs_means=='mlp'):
            top_grad = None
            for i in range(0,self.nLayers-1): # ignore the observed layer, start from top layer
                l = self.layers[i]
                next_layer = self.layers[i+1]
                top_grad = l._encoder_update_initial_gradient( [ e.mean for e in next_layer.init_Xs ] , top_grad=top_grad)
                
        # update gradients of initial values recursion parameters <-
        
    def freerun(self, init_Xs=None, step=None, U=None, m_match=True, encoder=False):
        assert self.U_pre_step, "The other case is not implemented yet!"
        if U is None and self.layers[0].withControl: raise "The model needs control signals!"
        if U is not None and step is None: step=U.shape[0] - self.layers[0].U_win
        elif step is None: step=100
        
        # layer 0 is the top layer
        
        layers_output = []
        con = U
        con_win = self.layers[0].U_win - 1 if self.layers[0].withControl else 0
        for i in range(self.nLayers):
            con = con[con_win-self.layers[i].U_win+1:] if self.layers[i].withControl else None
            X = self.layers[i].freerun(init_Xs=None if init_Xs is None else init_Xs[-i-1], step=step,U=con,m_match=m_match, encoder=encoder)
            con = X
            con_win = self.layers[i].X_win
            
            layers_output.append(X)
        return layers_output
    
    @Model.optimizer_array.setter
    def optimizer_array(self, p):
        if self.data_streamer is not None:
            self._next_minibatch()
        Model.optimizer_array.fset(self, p)

    def _next_minibatch(self):
        
        batch_idx, prev_sample_ix = self.data_streamer.get_cur_index()
        batch_idx, samples_idxes, Ys_n, Us_n = self.data_streamer.next_minibatch()
        # make U_pre_step shift ->
        Ys = []; Us = [];
        for i in range(len(Ys_n)):
            Y, U = Ys_n[i], Us_n[i]
            if self.U_pre_step:
                U = U[:-1].copy()
                Y = Y[self.U_win:].copy()
            else:
                Y = Y[self.U_win-1:].copy()
            Us.append(NormalPosterior(U,np.ones(U.shape)*1e-10))
            Ys.append(Y)
        # make U_pre_step shift <-
        
        #import pdb; pdb.set_trace()
        
        Xs = self._init_X(self.wins, Ys, None, None, nDims=self.nDims, init='nan')
        self.Us = Us
        self.Ys = Ys
        self.mb_inf_sample_idxes = samples_idxes[:]
        
        previous_layer = None
        for i in range(self.nLayers-1,-1,-1): # start from top layer
            layer = self.layers[self.nLayers - i - 1]
            layer_Us = None
            layer_Xs = None
            
#            if i==self.nLayers-1: # Top layer
#                layer_Us = Us if layer.withControl else None
#            elif i==0: # Observed layer
#                layer_Xs = Ys
#            #else: # Other layers
#            #    layer_Us = Xs[i]
#            #    layer_Xs = Xs[i-1]
#
#            layer.set_inputs_and_outputs(len(Ys), Xs=layer_Xs, Us=layer_Us, samples_idxes=samples_idxes)
            
            if i==self.nLayers-1: # Top layer
                layer_Us = Us if layer.withControl else None
                layer_Xs = Xs[i-1] 
            elif i==0: # Observed layer
                layer_Us = previous_layer.Xs_flat if (previous_layer is not None) else None 
                layer_Xs = Ys
            else: # Other layers
                layer_Us = previous_layer.Xs_flat if (previous_layer is not None) else None
                layer_Xs = Xs[i-1]
            
            #import pdb; pdb.set_trace()
            layer.set_inputs_and_outputs(len(Ys), Xs=layer_Xs, Us=layer_Us, samples_idxes=samples_idxes)
            previous_layer = layer
        
    def set_DataStreamer(self, data_streamer, back_cstr_initial="mlp"):
        from math import ceil
        self.data_streamer = data_streamer # this is an indicator of minibatch inference
        dataset_size = data_streamer.total_size
        
        if back_cstr_initial=="single": # single initial value for all subsequences (episodes). This is strange setting
            pass
        elif back_cstr_initial=="all": # for every subsequence (episode) there is its own initial value
            pass
        elif back_cstr_initial=="mlp": # initial values are the result of MLP
            pass
        else:
            raise ValueError('Wrong value for "back_cstr_initial".')


class DeepAutoreg_rnn(Model):
    """
    :param wins: windows sizes of layers
    :type wins: list[num of layers]. 0-th element correspond to output layer.
    
    :param U_pre_step: If true, the current signal is assumed to be controlled by the control signal of the previous time step.
    :type U_pre_step: Boolean
    
    :param nDims: output dimensions of layers
    :type nDims: list[num of layers]. 0-th element is output dimensionality.
    
    """
    
    def __init__(self, wins, Y, U=None, U_win=1, nDims=None, num_inducing=10, 
                 likelihood = None, name='autoreg', kernels=None, U_pre_step=True, 
                 inducing_init='kmeans', back_cstr=False, rnn_type='rnn',rnn_hidden_dims=None, 
                 rnn_bidirectional=False, rnn_h0_init='zero', inference_method=None,
                 minibatch_inference=True, mb_inf_tot_data_size = None,
                 mb_inf_sample_idxes=None):
        
        super(DeepAutoreg_rnn, self).__init__(name=name)
        
        Ys, Us = Y,U
        if isinstance(Ys, np.ndarray): Ys = [Ys]
        if Us is not None and isinstance(Us, np.ndarray): Us = [Us]
        
        # If data_streamer is planned to be used, here the first batch must be provided as input data
        self.data_streamer = None
        
        self.nLayers = len(wins)
        self.back_cstr = back_cstr
        self.wins = wins
        self.U_win = U_win
        #self.input_dim = 1
        #self.output_dim = 1
        self._log_marginal_likelihood = np.nan
        self.U_pre_step = U_pre_step
        self.nDims = nDims if nDims is not None else [Ys[0].shape[1]]+[1]*(self.nLayers-1)
        
        if Us is not None:
            assert len(Ys)==len(Us)
            self.Us = []
            self.Ys = []
            self.Ys_all = []
            for i in range(len(Ys)):
                Y, U = Ys[i], Us[i]
#                 assert Y.shape[0]==U.shape[0], "the signal and control should be aligned."
                if self.U_pre_step:
                    U = U[:-1].copy()
                    Y_t = Y[U_win:].copy()
                else:
                    Y_t = Y[U_win-1:].copy()
                self.Us.append(NormalPosterior(U.copy(),np.ones(U.shape)*1e-10))
                self.Ys.append(Y_t)
                self.Ys_all.append(Y) # needed for encoder input
        else:
            self.Us = Us
            self.Ys = Ys
            self.Ys_all = Ys
        #import pdb; pdb.set_trace()
        
        #Xs = self._init_X(wins, self.Ys, self.Us, X_variance, init=init, nDims=self.nDims)
        
        assert back_cstr==True, "This is the typical situation for this model."
        
        self.auto_update = False if inference_method=='svi' else True
        auto_update = self.auto_update
        
        # Parameters which exist differently per layer but specified as single componenents are here expanded to each layer
        if not isinstance(num_inducing, list or tuple): num_inducing = [num_inducing]*self.nLayers

        if minibatch_inference:
            assert len(mb_inf_sample_idxes) == len(self.Ys), "Sizes must be equal"
            
            self.mb_inf_sample_idxes = mb_inf_sample_idxes
            self.minibatch_inference = minibatch_inference
            self.mb_inf_tot_data_size = mb_inf_tot_data_size
            
            self.minibatch_size = len(self.mb_inf_sample_idxes) 
        else:
            self.minibatch_inference = False
            self.minibatch_size = len(self.Ys)
            
        self.rnn_hidden_dims = rnn_hidden_dims
        self.encoder_rnn_type = rnn_type
        self.encoder_h0_type = 'zero'
        self.encoder_bidirectional = rnn_bidirectional
        #import pdb; pdb.set_trace()
        
        self._init_encoder()
        Xs = self._run_encoder_forward()
        # Initialize Layers. The top layer goes first in the list
        self.layers = []
        for i in range(self.nLayers-1,-1,-1):
            if i==self.nLayers-1: # Top layer
                self.layers.append(Layer_rnn(None, Xs[i-1], X_win=wins[i], Us=self.Us, U_win=U_win, num_inducing=num_inducing[i],  kernel=kernels[i] if kernels is not None else None, 
                                         noise_var=0.01, name='layer_'+str(i),  back_cstr=back_cstr, inducing_init=inducing_init, auto_update=auto_update, inference_method=inference_method,
                                         minibatch_inference=minibatch_inference, mb_inf_tot_data_size=mb_inf_tot_data_size,
                                         mb_inf_sample_idxes = mb_inf_sample_idxes))
            elif i==0: # Observed layer
                self.layers.append(Layer_rnn(self.layers[-1], self.Ys, X_win=wins[i], Us=Xs[i], U_win=wins[i+1], num_inducing=num_inducing[i],  kernel=kernels[i] if kernels is not None else None, 
                                         likelihood=likelihood, noise_var=1., back_cstr=back_cstr, name='layer_'+str(i), inducing_init=inducing_init, auto_update=auto_update, inference_method=inference_method,
                                         minibatch_inference=minibatch_inference, mb_inf_tot_data_size=mb_inf_tot_data_size, 
                                         mb_inf_sample_idxes = mb_inf_sample_idxes))
            else: # Observed layer Other layers
                self.layers.append(Layer_rnn(self.layers[-1], Xs[i-1], X_win=wins[i], Us=Xs[i], U_win=wins[i+1], num_inducing=num_inducing[i],  kernel=kernels[i] if kernels is not None else None, 
                                         noise_var=0.01, name='layer_'+str(i), back_cstr=back_cstr, inducing_init=inducing_init, auto_update=auto_update, inference_method=inference_method,
                                         minibatch_inference=minibatch_inference, mb_inf_tot_data_size=mb_inf_tot_data_size,
                                         mb_inf_sample_idxes = mb_inf_sample_idxes))
            
        self.link_parameters(*self.layers)
        self.link_parameter(self.encoder)
        
    def _init_encoder(self,):
        """
        Init encoder
        """
        self.encoder_input_dims = self.nDims[:-1]
        self.encoder_output_dims = self.nDims[1:]
        self.encoder_hidden_dim = self.rnn_hidden_dims[0] if self.rnn_hidden_dims is not None else 10
        
        #import pdb; pdb.set_trace()
        self.encoder = seq_encoder(self.nLayers-1, self.encoder_input_dims, self.encoder_output_dims, self.encoder_hidden_dim, 
                                   h_0_type=self.encoder_h0_type, rnn_type=self.encoder_rnn_type, bidirectional=self.encoder_bidirectional)
        
    def _run_encoder_forward(self,):
        """
        Runs the encoder and generate layer's q_X.
        """
        
        # stack batches over axis 1. This is required by pytorch.
        out_means, out_vars = self.encoder.forward_computation(  np.stack(self.Ys_all, axis=1)  )
        
        Xs = []
        import pdb; pdb.set_trace()
        # iteration over layers (start from lower, observed is 0), inside each layer over i_seq
        for layer_idx in range(len(out_means)): # iteration over layers
            Xs_layer = []
            for sample_idx in range(len(out_means[layer_idx])): # iteration over layers
                Xs_layer.append( NormalPosterior( out_means[layer_idx][sample_idx], out_vars[layer_idx][sample_idx], name='qX_'+str( layer_idx ) )  ) 
                
            Xs.append( Xs_layer )
        return Xs   
    
    
#        for i in len(out_means):       
#            Xs.append( layers[i] )        
#            
#        for i_seq in range(self.minibatch_size):
#            out_means, out_vars = self.encoder.forward_computation(self.Ys_all[i_seq])
#            if (i_seq == 0):
#                for i in len(out_means):
#                    layers = {}
#                    layers[i] = []
#                    
#            for i in len(out_means):
#                layers[i].append(NormalPosterior(out_means[i], out_vars[i],name='qX_'+str(i_seq)) )
#                
#        for i in len(out_means):       
#            Xs.append( layers[i] ) 
        
         
        
#    def _init_X(self, wins, Ys, Us, X_variance, nDims, init='Y'):
#        self._init_encoder()
#        
#        Xs = []
#        for i_layer in range(1,self.nLayers):
#            X = []
#            for i_seq in range(len(Ys)):
#                Y = Ys[i_seq]
#                U_win, U_dim = wins[i_layer], nDims[i_layer]
#                if init=='Y':
#                    mean = np.zeros((U_win+Y.shape[0], U_dim))
#                    mean[U_win:] = Y[:,:U_dim]
#                    var = np.zeros((U_win+Y.shape[0],U_dim))+X_variance
#                    X.append(NormalPosterior(mean,var,name='qX_'+str(i_seq)))
#                elif init=='rand' :
#                    mean = np.zeros((U_win+Y.shape[0], U_dim))
#                    mean[:] = np.random.randn(*mean.shape)
#                    var = np.zeros((U_win+Y.shape[0],U_dim))+X_variance
#                    X.append(NormalPosterior(mean,var,name='qX_'+str(i_seq)))
#                elif init=='zero':
#                    mean = np.zeros((U_win+Y.shape[0], U_dim))
#                    mean[:] = np.random.randn(*mean.shape)*0.01
#                    var = np.zeros((U_win+Y.shape[0],U_dim))+X_variance
#                    X.append(NormalPosterior(mean,var,name='qX_'+str(i_seq)))
#                elif init=='nan':
#                    mean = np.empty((U_win+Y.shape[0], U_dim))
#                    var = np.empty((U_win+Y.shape[0],U_dim))
#                    X.append(NormalPosterior(mean,var,name='qX_'+str(i_seq)))
#            Xs.append(X)
#        return Xs
        
    def log_likelihood(self):
        return self._log_marginal_likelihood
    
    def _set_layer_Xs(self,):
        # encoder generate hidden states ->
        
        #import pdb; pdb.set_trace()
        
        Xs = self._run_encoder_forward()
        previous_layer = None
        for i in range(self.nLayers-1,-1,-1): # start from top layer
            layer = self.layers[self.nLayers - i - 1]
            layer_Us = None
            layer_Xs = None
            
            if i==self.nLayers-1: # Top layer
                layer_Us = self.Us if layer.withControl else None
                layer_Xs = Xs[i-1] 
            elif i==0: # Observed layer
                layer_Us = previous_layer.Xs_flat if (previous_layer is not None) else None 
                layer_Xs = self.Ys
            else: # Other layers
                layer_Us = previous_layer.Xs_flat if (previous_layer is not None) else None
                layer_Xs = Xs[i-1]
            
            #import pdb; pdb.set_trace()
            if self.minibatch_inference:
                layer.set_inputs_and_outputs(len(self.Ys), Xs=layer_Xs, Us=layer_Us, samples_idxes=self.mb_inf_sample_idxes)
            else:
                layer.set_inputs_and_outputs(len(self.Ys), Xs=layer_Xs, Us=layer_Us, samples_idxes=None)
            previous_layer = layer
        # encoder generate hidden states ->
        
    def parameters_changed(self):
        
        self._set_layer_Xs()
        
        if not self.auto_update: [l.update_layer() for l in self.layers] # starting from top layer (with inputs)
        self._log_marginal_likelihood = np.sum([l._log_marginal_likelihood for l in self.layers])
        [l.update_latent_gradients() for l in self.layers[::-1]] # start from lowest layer.
        
        
        #import pdb; pdb.set_trace()
        # Update encoder parameters ->
        layer_list_means = [ [] for ll in range(self.nLayers-1) ]
        layer_list_vars = [ [] for ll in range(self.nLayers-1) ]
        for i_seq in range(self.minibatch_size):
            for idx,l in enumerate( self.layers[-2::-1] ): # exclude the lowest layer because it is the observed one
                layer_list_means[idx].append( l.Xs_flat[i_seq].mean.gradient )
                layer_list_vars[idx].append( l.Xs_flat[i_seq].variance.gradient )
                
                
        pmeans = [ np.stack(ll, axis=1) for ll in layer_list_means ] 
        pvars = [ np.stack(ll, axis=1) for ll in layer_list_vars ]
        pmeans.extend(pvars)
        
        #import pdb; pdb.set_trace()
        
        self.encoder.backward_computation( pmeans ) # sets gradients of numpy parameters
        # Update encoder parameters <-
        
        #import pdb; pdb.set_trace()
        
    def freerun(self, init_Xs=None, step=None, U=None, m_match=True, encoder=False):
        assert self.U_pre_step, "The other case is not implemented yet!"
        if U is None and self.layers[0].withControl: raise "The model needs control signals!"
        if U is not None and step is None: step=U.shape[0] - self.layers[0].U_win
        elif step is None: step=100
        
        # layer 0 is the top layers
        
        layers_output = []
        con = U
        con_win = self.layers[0].U_win - 1 if self.layers[0].withControl else 0
        for i in range(self.nLayers):
            con = con[con_win-self.layers[i].U_win+1:] if self.layers[i].withControl else None
            X = self.layers[i].freerun(init_Xs=None if init_Xs is None else init_Xs[-i-1], step=step,U=con,m_match=m_match, encoder=encoder)
            con = X
            con_win = self.layers[i].X_win
            
            layers_output.append(X)
        return layers_output
    
    @Model.optimizer_array.setter
    def optimizer_array(self, p):
        if (self.data_streamer is not None) and self.minibatch_inference:
            self._next_minibatch()
        Model.optimizer_array.fset(self, p)

    def _next_minibatch(self):
        
        #import pdb; pdb.set_trace()
        
        batch_idx, prev_sample_ix = self.data_streamer.get_cur_index()
        batch_idx, samples_idxes, Ys_n, Us_n = self.data_streamer.next_minibatch()
        # make U_pre_step shift ->
        Ys = []; Us = []; Ys_all = [];
        for i in range(len(Ys_n)):
            Y, U = Ys_n[i], Us_n[i]
            if self.U_pre_step:
                U = U[:-1].copy()
                Y_t = Y[self.U_win:].copy()
            else:
                Y_t = Y[self.U_win-1:].copy()
                
            Ys_all.append(Y)
            Us.append(NormalPosterior(U,np.ones(U.shape)*1e-10))
            Ys.append(Y_t)
        
        # make U_pre_step shift <-
        #import pdb; pdb.set_trace()
        self.Us = Us
        self.Ys = Ys
        self.Ys_all = Ys_all
        
        
        self.Us = Us
        self.Ys = Ys
        self.Ys_all = Ys_all
        self.mb_inf_sample_idxes = samples_idxes[:]
        self.minibatch_size = len(self.mb_inf_sample_idxes)
        
    def set_DataStreamer(self, data_streamer, back_cstr_initial="mlp"):
        from math import ceil
        assert self.minibatch_inference, "Datastreamer is supported only for minibatch inference."
        
        self.data_streamer = data_streamer # this is an indicator of minibatch inference
        dataset_size = data_streamer.total_size
        
        if back_cstr_initial=="single": # single initial value for all subsequences (episodes). This is strange setting
            pass
        elif back_cstr_initial=="all": # for every subsequence (episode) there is its own initial value
            pass
        elif back_cstr_initial=="mlp": # initial values are the result of MLP
            pass
        else:
            raise ValueError('Wrong value for "back_cstr_initial".')
                
        
class DeepAutoreg(Model):
    """
    :param U_pre_step: If true, the current signal is assumed to be controlled by the control signal of the previous time step.
    :type U_pre_step: Boolean
    """
    
    def __init__(self, wins, Y, U=None, U_win=1, nDims=None, X_variance=0.01, num_inducing=10, 
                 likelihood = None, name='autoreg', kernels=None, U_pre_step=True, init='Y', 
                 inducing_init='kmeans', back_cstr=False, MLP_dims=None):
        super(DeepAutoreg, self).__init__(name=name)
        
        #import pdb; pdb.set_trace() # Alex
        Ys, Us = Y,U
        if isinstance(Ys, np.ndarray): Ys = [Ys]
        if Us is not None and isinstance(Us, np.ndarray): Us = [Us]
        
        self.nLayers = len(wins)
        self.back_cstr = back_cstr
        self.wins = wins
        #self.input_dim = 1
        #self.output_dim = 1
        self._log_marginal_likelihood = np.nan
        self.U_pre_step = U_pre_step
        self.nDims = nDims if nDims is not None else [Ys[0].shape[1]]+[1]*(self.nLayers-1)
        
        if Us is not None:
            assert len(Ys)==len(Us)
            self.Us = []
            self.Ys = []
            for i in range(len(Ys)):
                Y, U = Ys[i], Us[i]
#                 assert Y.shape[0]==U.shape[0], "the signal and control should be aligned."
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
                self.layers.append(Layer(None, self.Xs[i-1], X_win=wins[i], Us=self.Us, U_win=U_win, num_inducing=num_inducing[i],  kernel=kernels[i] if kernels is not None else None, noise_var=0.01, name='layer_'+str(i),  back_cstr=back_cstr, MLP_dims=MLP_dims, inducing_init=inducing_init))
            elif i==0:
                self.layers.append(Layer(self.layers[-1], self.Ys, X_win=wins[i], Us=self.Xs[i], U_win=wins[i+1], num_inducing=num_inducing[i],  kernel=kernels[i] if kernels is not None else None, likelihood=likelihood, noise_var=1., back_cstr=back_cstr, name='layer_'+str(i), inducing_init=inducing_init))
            else:
                self.layers.append(Layer(self.layers[-1], self.Xs[i-1], X_win=wins[i], Us=self.Xs[i], U_win=wins[i+1], num_inducing=num_inducing[i],  kernel=kernels[i] if kernels is not None else None, noise_var=0.01, name='layer_'+str(i), back_cstr=back_cstr, MLP_dims=MLP_dims, inducing_init=inducing_init))
        self.link_parameters(*self.layers)
            
    def _init_X(self, wins, Ys, Us, X_variance, nDims, init='Y'):
        #import pdb; pdb.set_trace()
        
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
        
    def freerun(self, init_Xs=None, step=None, U=None, m_match=True, encoder=False):
        assert self.U_pre_step, "The other case is not implemented yet!"
        if U is None and self.layers[0].withControl: raise "The model needs control signals!"
        if U is not None and step is None: step=U.shape[0] - self.layers[0].U_win
        elif step is None: step=100
        
        # layer 0 is the top layer
        
        con = U
        con_win = self.layers[0].U_win - 1 if self.layers[0].withControl else 0
        for i in range(self.nLayers):
            con = con[con_win-self.layers[i].U_win+1:] if self.layers[i].withControl else None
            X = self.layers[i].freerun(init_Xs=None if init_Xs is None else init_Xs[-i-1], step=step,U=con,m_match=m_match, encoder=encoder)
            con = X
            con_win = self.layers[i].X_win
        return X
        
        

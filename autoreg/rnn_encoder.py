#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 18:51:22 2017

@author: grigoral
"""
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn

from GPy.core import Model, Parameterized, Param
import numpy as np
import sys
import matplotlib.pyplot as plt

#_ = torch.manual_seed(1)


class Mean_var_rnn(nn.Module):
    
    def __init__(self, p_input_dim, p_output_dim, p_hidden_dim, rnn_type='rnn', with_input_variance = True, bidirectional=False):
        super(Mean_var_rnn, self).__init__()

        if rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=p_input_dim, hidden_size=p_hidden_dim, num_layers=1, bidirectional=bidirectional)
            # input: ( seq_len, batch, input_size)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=p_input_dim, hidden_size=p_hidden_dim, num_layers=1, bidirectional=bidirectional)
            # input: (seq_len, batch, input_size)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=p_input_dim, hidden_size=p_hidden_dim, num_layers=1, bidirectional=bidirectional)
        else:
            raise ValueError("Unknow rnn type")
        
        self.rnn_type = rnn_type
        self.bidirectional=bidirectional
        self.with_input_variance = with_input_variance
        
        # in_features, out_features
        self.dir_number = 1 if bidirectional==False else 2
        self.linear_mean = nn.Linear(in_features=p_hidden_dim*self.dir_number,  out_features=p_output_dim)
        
        
        
        self.linear_var = nn.Linear(in_features=p_hidden_dim*self.dir_number,  out_features=p_output_dim)
        
        self.soft_plus = nn.Softplus()
        
    def forward(self, mean_input, h_0, var_input=None, c_0=None):
        """
        
        Input:
        ------------
        c_0 - lstm init cell state
        """
        
        #import pdb; pdb.set_trace()
        
        if self.with_input_variance:
            #import pdb; pdb.set_trace()
            comb_input = torch.cat( (mean_input, var_input), dim=2 )
        else:
            comb_input = mean_input
            #comb_input = torch.cat( (mean_input, torch.zeros(mean_input.size()).double() ), dim=2 )
        #import pdb; pdb.set_trace()
        if self.rnn_type=='lstm':
            rnn_outputs,_ = self.rnn( comb_input, (h_0, c_0) ) # (seq_len, batch, hidden_size * num_directions)
        else:    
            rnn_outputs,_ = self.rnn( comb_input, h_0) # (seq_len, batch, hidden_size * num_directions)
        # Linear input: (N,∗,in_features)
        self.mean_out = self.linear_mean(rnn_outputs) #(N,∗,out_features)
        
        #if self.with_input_variance:
        self.var_out = self.soft_plus( self.linear_var(rnn_outputs) ) 
        #else:
        #    self.var_out = None
        
        # maybe transform shapes
        return self.mean_out, self.var_out   
        
        
class Mean_var_multilayer(nn.Module):
    def __init__(self, p_layer_num, p_input_dims, p_output_dims, p_hidden_dim, rnn_type='rnn', h_0_type='zero',
                 bidirectional=False):
        """
        
        """
        super(Mean_var_multilayer, self).__init__()
        
        #import pdb; pdb.set_trace()
        
        assert p_layer_num == len(p_input_dims), "Layer num must be correct"
        assert len(p_input_dims) == len(p_output_dims), " Dim lengths must match"
        
        self.layer_num = p_layer_num
        self.input_dims = [ (ss if i==0 else ss*2) for (i, ss) in enumerate(p_input_dims)  ] # lower layers first 
        self.output_dims = p_output_dims #[ ss*2 for ss in p_output_dims] # lower layers first 
        self.hidden_dim = p_hidden_dim # asssume that hidden dim of all layers is equal
        
        self.rnn_type = rnn_type
        self.bidirectional=bidirectional
        
        if h_0_type=='zero':
            #self.h_0 = Variable( 0.1*torch.eye(self.hidden_dim).double() )
            self.h_0 = np.zeros((p_hidden_dim,) )
        else:
            raise NotImplemented("Other initialization is not currently implemented")
        
        if (rnn_type=='lstm'):
            c_0_type = h_0_type
            self.c_0 = np.zeros((p_hidden_dim,) )
        
        self.layers = []
        for l in range(self.layer_num): # layer 0 is observed,  layer 1 is the next after observed etc.
            # layers are created separately in order to make python references to the hidden layers outputs
            with_input_variance = False if (l==0) else True # input variance
            layer = Mean_var_rnn(self.input_dims[l], self.output_dims[l], 
                                 self.hidden_dim, rnn_type=rnn_type, with_input_variance = with_input_variance, bidirectional=bidirectional)
            setattr(self, 'layer_' + str(l), layer)
            self.layers.append(layer)
        
    def forward(self, inp_l0):
        
        #import pdb; pdb.set_trace()
        # prepare h_0 ->
        h_0 = Variable( torch.from_numpy( np.broadcast_to(self.h_0, ( 2 if self.bidirectional else 1,inp_l0.size()[1],self.h_0.shape[0]) ) ).double() )
        
        if self.rnn_type =='lstm':
            c_0 = Variable( torch.from_numpy( np.broadcast_to(self.c_0, ( 2 if self.bidirectional else 1,inp_l0.size()[1],self.h_0.shape[0]) ) ).double() )
        else:
            c_0 = None
            
        #import pdb; pdb.set_trace()
        
        # prepare h_0 <-
        self.out_means = []
        self.out_vars = []
        out_mean, out_var = inp_l0, None
        for l in range(self.layer_num):
            layer = self.layers[l]
            #import pdb; pdb.set_trace()
            out_mean, out_var = layer( out_mean, h_0,  var_input=out_var, c_0=c_0)
            
            # Store outputs
            self.out_means.append(out_mean )
            self.out_vars.append( out_var)

        return self.out_means, self.out_vars
        
class seq_encoder(Parameterized):

    def __init__(self, p_layer_num, p_input_dims, p_output_dims, p_hidden_dim, h_0_type='zero', rnn_type='rnn', bidirectional=False,
                 name='seq_encoder'):
        """
        
        """
        super(seq_encoder, self).__init__(name=name)
        #import pdb; pdb.set_trace()
        
        self.encoder = Mean_var_multilayer(p_layer_num, p_input_dims, p_output_dims, p_hidden_dim, h_0_type=h_0_type, rnn_type=rnn_type,
                                           bidirectional=bidirectional).double()
        #self.encoder.double() # convert all the parameters to float64
        
        self.params_dict= {}
        self.encoder_param_names_dics = {} # inverse transform from pytorch to gpy
        for ee in self.encoder.named_parameters():
            param_name = ee[0].replace('.','_') # transform paparm name from pytorch to gpy
            self.encoder_param_names_dics[param_name] = ee[0]
            
            tt = ee[1].data.numpy().copy()
            param = Param( param_name, tt )
            setattr(self, param_name, param )
            self.params_dict[param_name] = getattr(self, param_name)
            
            self.link_parameters(param)
        pass
    
    def _zero_grads(self,):
        self.encoder.zero_grad()
        
    def _params_from_gpy(self,):
        """
        Copy parameters from GPy to pytorch
        """
        
        for p_name, p_val in self.params_dict.iteritems():
            gpy_param = getattr(self, p_name).values.copy()
#            if p_name == 'layer_0_linear_var_bias':
#                gpy_param[:] = 32
#                import pdb; pdb.set_trace()
                
            self.encoder.state_dict()[ self.encoder_param_names_dics[p_name] ].copy_( torch.from_numpy(gpy_param) ) # this seems to work
            
            #setattr( self.encoder, self.encoder_param_names_dics[p_name],  Variable( torch.from_numpy(gpy_param) ) )
            
        #import pdb; pdb.set_trace()
            
    def gradients_to_gpy(self,):
        """
        Sets the gradients of encoder parameters to the computed values.
        This function must be called after all smaples in minibatch are
        processed.
        """
        
        #import pdb; pdb.set_trace()
        params_dict = {ii[0]:ii[1] for ii in self.encoder.named_parameters()}
        
        for p_name, p_val in self.params_dict.iteritems():
            pytorch_param = params_dict[ self.encoder_param_names_dics[p_name] ] 
            pytorch_param_grad = pytorch_param.grad.data.numpy()
            gpy_param = getattr(self, p_name)
            assert gpy_param.gradient.shape == pytorch_param_grad.shape, "Shapes must be equal"
            gpy_param.gradient = pytorch_param_grad.copy()
        
        
    def forward_computation(self, l0_input):
        """
        Given the parameters of the neural networks computes outputs of each layer
        
        Input:
        ------------------
        l0_input: list
        list of size batch size, in each element the ndarray of shape (seq_len, input_dim)
        """
        #import pdb; pdb.set_trace()
        
        batch_size = l0_input.shape[1]
        
        self._zero_grads()
        self._params_from_gpy()
        
        l0_input = Variable( torch.from_numpy(l0_input) ) # right shape: (seq_len, batch, hidden_size * num_directions) 
        
        
        # comp. from botton to top. Lists of computed means and vars from layers.
        self.forward_means_list, self.forward_vars_list = self.encoder.forward( l0_input )
        
        # Transformation to the required output form: list of lists of (sample size, dimensions). First list is 
        # over layers (starting from the one after the output), second list is over batch
        out_means_list = [ [ ll.squeeze() for ll in np.split( pp.data.numpy().copy(), batch_size, axis=1) ] for pp in self.forward_means_list  ]
        out_vars_list = [ [ ll.squeeze() for ll in np.split( pp.data.numpy().copy(), batch_size, axis=1) ] for pp in self.forward_vars_list  ]
        
        
        # return values are a list of lares outputs starting from the lower, the lowest one (output is excluded since it is only the ipnut layer)
        return out_means_list, out_vars_list
    
    def backward_computation(self, input_gradient_list ):
        """
        Computes the gradient of parameters given the gradients of outputs of each layer.
        
        Input:
        ---------------
        input_gradient_list: list
            Contains gradients of X means and variances. First gradients of means, then gradients of variances,
                in order from lower layer to the top. (lowest is the one after the output layer).
        
        """
        #import pdb; pdb.set_trace()
        
        input_gradient_list = [ torch.from_numpy(gg) for gg in input_gradient_list]
        torch.autograd.backward( variables=self.forward_means_list + self.forward_vars_list, 
                                 grad_variables = input_gradient_list, retain_graph=False  )
        
        
        self.gradients_to_gpy()
        
        # Resent the computaitonal graph
        self.forward_means_list = None
        self.forward_vars_list = None
        
def test_graph():
    
    
    y = Variable(  torch.from_numpy( np.array((1,2.0)) ) )
    
    w1 = Variable( torch.from_numpy( np.array( (2.0,) ) ), requires_grad=True )
    w2 = Variable( torch.from_numpy( np.array( (3.0,) ) ), requires_grad=True )
    
    
    x1 = y*w1
    x2 = x1 * w2
    
    torch.autograd.backward( variables =(x1,x2), grad_variables = (torch.from_numpy( np.array((1,1.0)) ), torch.from_numpy( np.array((1,1.0)) ) ), retain_graph=True  )
    raise ValueError("sdfb")
    #torch.autograd.backward( variables =(x2,), grad_variables = (torch.from_numpy( np.array((1,1.0)) ), ) ) 
    
    
    globals().update(locals());
    
    

if __name__ == '__main__':
    #rnn = nn.RNN(input_size=5, hidden_size=10, num_layers=1, batch_first=True, bidirectional=False)
    test_graph()
    #tt = Mean_var_multilayer(2, [2,3], [3,4], 5, h_0_type='unit', rnn_type='rnn')
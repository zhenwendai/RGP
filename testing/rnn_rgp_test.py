#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 10:16:16 2017

@author: grigoral
"""

import unittest
import numpy as np
import GPy
import os
import copy

import autoreg
from autoreg.data_streamers import TrivialDataStreamer, RandomPermutationDataStreamer, StdMemoryDataStreamer

def generate_data( seq_num, seq_length, u_dim = 1, y_dim = 1):
    """
    Generates data
    """
    
    #np.random.seed()
    
    U = []
    Y = []
    
    for i in range(seq_num):
        uu = np.random.randn( seq_length, u_dim ) * 10
        yy = np.random.randn( seq_length, y_dim ) * 100
        
        U.append(uu)
        Y.append(yy)
        
    return U, Y

class Rnn_RGP_Test(unittest.TestCase):
    """
    Test the Deepautoreg_rnn model. Test rnn as a recognition model.
    
    The test classes [ Rnn_RGP_Test, Lstm_RGP_Test, Gru_RGP_Test, Gru_bidirect_RGP_Test ],
    do exactly the same testing except the back constrain neural network is different for each of them.
    """
    
    def setUp(self):
        
        u_dim = 2
        y_dim = 3
        ts_length = 20
        sequences_no = 3
        #U, Y = generate_data( sequences_no, ts_length, u_dim = u_dim, y_dim = y_dim)
        U_2, Y_2 = generate_data( sequences_no*2, ts_length, u_dim = u_dim, y_dim = y_dim)
        
        Q = 3 # 200 # Inducing points num. Take small number ofr speed
        
        back_cstr = True
        inference_method = 'svi'
        minibatch_inference = True
        
#        # 1 layer:
#        win_out = 3
#        win_in = 2
#        wins = [0, win_out] # 0-th is output layer
#        nDims = [y_dim,2]
        
        # 2 layers:
        win_out = 3
        win_in = 2
        wins = [0, win_out, win_out]
        nDims = [y_dim, 2,3] # 
        
        rnn_hidden_dims = [9,] # rnn hidden dimension
        rnn_type='rnn'
        rnn_bidirectional=False
        rnn_h0_init='zero'
        
        #print("Input window:  ", win_in)
        #print("Output window:  ", win_out)
        
        data_streamer = RandomPermutationDataStreamer(Y_2, U_2)
        minibatch_index, minibatch_indices, Y_mb, X_mb = data_streamer.next_minibatch()
        
        m_1 = autoreg.DeepAutoreg_rnn(wins, Y_mb, U=X_mb, U_win=win_in,
                                num_inducing=Q, back_cstr=back_cstr, nDims=nDims,
                                
                                rnn_hidden_dims=rnn_hidden_dims,
                                rnn_type=rnn_type,
                                rnn_bidirectional=rnn_bidirectional,
                                rnn_h0_init=rnn_h0_init,
                                
                                inference_method=inference_method, # Inference method
                                minibatch_inference = minibatch_inference,
                                mb_inf_tot_data_size = sequences_no*2,
                                mb_inf_sample_idxes = minibatch_indices,
#                                # 1 layer:
#                                kernels=[GPy.kern.RBF(win_out*nDims[1],ARD=True,inv_l=True),
#                                         GPy.kern.RBF( (win_in + win_out) * nDims[1], ARD=True,inv_l=True)] )
        
                                 # 2 layers:
                                kernels=[GPy.kern.RBF(win_out*nDims[1],ARD=True,inv_l=True),
                                         GPy.kern.RBF(win_out*nDims[1] + win_out*nDims[2],ARD=True,inv_l=True),
                                         GPy.kern.RBF(win_out*nDims[2] + win_in*u_dim,ARD=True,inv_l=True)])
        
        self.model_1 = m_1
        self.model_1._trigger_params_changed()
        
        
        self.mll_1_1 = float(self.model_1._log_marginal_likelihood)
        #self.g_mll_1_1 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        self.g_mll_1_1 = self.model_1._log_likelihood_gradients().copy()
        
        self.model_1.checkgrad(verbose=False)
        
#        self.model_2 = copy.deepcopy(m_1)
        
        self.model_1.set_DataStreamer(data_streamer)
        self.model_1._trigger_params_changed()
        
        self.model_1._next_minibatch()
        self.model_1._trigger_params_changed()
        
        self.mll_1_2 = float(self.model_1._log_marginal_likelihood)
        #self.g_mll_1_2 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        self.g_mll_1_2 = self.model_1._log_likelihood_gradients().copy()
        
        data_streamer_1 = StdMemoryDataStreamer(Y_2, U_2, sequences_no)
        
        self.model_1.set_DataStreamer(data_streamer_1)
        self.model_1._next_minibatch()
        self.model_1._trigger_params_changed()
        
        self.mll_2_1 = float(self.model_1._log_marginal_likelihood)
        
        
        # exclude 'init_Xs' and 'X_var' from gradients
        #self.g_mll_2_1 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        self.g_mll_2_1 = self.model_1._log_likelihood_gradients().copy()
        
        #import pdb; pdb.set_trace()
        
        self.model_1._next_minibatch()
        self.model_1._trigger_params_changed()
        
        self.mll_2_2 = float(self.model_1._log_marginal_likelihood)
        
        # exclude 'init_Xs' and 'X_var' from gradients
        #self.g_mll_2_2 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        self.g_mll_2_2 = self.model_1._log_likelihood_gradients().copy()
    
    def test_perm_ds_two_minibatches(self,):
        
        #import pdb; pdb.set_trace()
        
        #np.testing.assert_almost_equal( self.mll_1_2, self.mll_1_1, decimal=9, err_msg="Likelihoods must be equal" )
        np.testing.assert_equal( np.isclose(self.mll_1_2, self.mll_1_1, atol = 0, rtol = 1e-14), True, err_msg="Likelihoods must be equal" )
        
        #np.testing.assert_array_equal( self.g_mll_1_2, self.g_mll_1_1, err_msg="Likelihood gradients must be equal" )
        np.testing.assert_equal( np.all( np.isclose(self.g_mll_1_2, self.g_mll_1_1, atol = 0, rtol = 1e-11)), True, err_msg="Likelihood gradients must be equal" )
    
    def test_perm_ds_sum_minibatches(self,):
        
        
        #import pdb; pdb.set_trace()
        #np.testing.assert_equal( self.mll_2_1 + self.mll_2_2, self.mll_1_1, err_msg="Likelihoods must be equal" ) #decimal=9
        np.testing.assert_equal( np.isclose(float(self.mll_2_1) + float(self.mll_2_2), self.mll_1_1, atol = 0, rtol = 1e-14), True, err_msg="Likelihoods must be equal" )
        
        #np.testing.assert_array_equal( self.g_mll_2_1 + self.g_mll_2_2, self.g_mll_1_1, err_msg="Likelihood gradients must be equal" )
        np.testing.assert_equal( np.all( np.isclose(self.g_mll_2_1 + self.g_mll_2_2, self.g_mll_1_1, atol = 0, rtol = 1e-11)), True, err_msg="Likelihood gradients must be equal" )

class Lstm_RGP_Test(unittest.TestCase):
    """
    Test the Deepautoreg_rnn model. Test rnn as a recognition model.
    
    The test classes [ Rnn_RGP_Test, Lstm_RGP_Test, Gru_RGP_Test, Gru_bidirect_RGP_Test ],
    do exactly the same testing except the back constrain neural network is different for each of them.
    """
    
    def setUp(self):
        
        u_dim = 2
        y_dim = 3
        ts_length = 20
        sequences_no = 3
        #U, Y = generate_data( sequences_no, ts_length, u_dim = u_dim, y_dim = y_dim)
        U_2, Y_2 = generate_data( sequences_no*2, ts_length, u_dim = u_dim, y_dim = y_dim)
        
        Q = 3 # 200 # Inducing points num. Take small number ofr speed
        
        back_cstr = True
        inference_method = 'svi'
        minibatch_inference = True
        
#        # 1 layer:
#        win_out = 3
#        win_in = 2
#        wins = [0, win_out] # 0-th is output layer
#        nDims = [y_dim,2]
        
        # 2 layers:
        win_out = 3
        win_in = 2
        wins = [0, win_out, win_out]
        nDims = [y_dim, 2,3] # 
        
        rnn_hidden_dims = [9,] # rnn hidden dimension
        rnn_type='lstm'
        rnn_bidirectional=False
        rnn_h0_init='zero'
        
        #print("Input window:  ", win_in)
        #print("Output window:  ", win_out)
        
        data_streamer = RandomPermutationDataStreamer(Y_2, U_2)
        minibatch_index, minibatch_indices, Y_mb, X_mb = data_streamer.next_minibatch()
        
        m_1 = autoreg.DeepAutoreg_rnn(wins, Y_mb, U=X_mb, U_win=win_in,
                                num_inducing=Q, back_cstr=back_cstr, nDims=nDims,
                                
                                rnn_hidden_dims=rnn_hidden_dims,
                                rnn_type=rnn_type,
                                rnn_bidirectional=rnn_bidirectional,
                                rnn_h0_init=rnn_h0_init,
                                
                                inference_method=inference_method, # Inference method
                                minibatch_inference = minibatch_inference,
                                mb_inf_tot_data_size = sequences_no*2,
                                mb_inf_sample_idxes = minibatch_indices,
#                                # 1 layer:
#                                kernels=[GPy.kern.RBF(win_out*nDims[1],ARD=True,inv_l=True),
#                                         GPy.kern.RBF( (win_in + win_out) * nDims[1], ARD=True,inv_l=True)] )
        
                                 # 2 layers:
                                kernels=[GPy.kern.RBF(win_out*nDims[1],ARD=True,inv_l=True),
                                         GPy.kern.RBF(win_out*nDims[1] + win_out*nDims[2],ARD=True,inv_l=True),
                                         GPy.kern.RBF(win_out*nDims[2] + win_in*u_dim,ARD=True,inv_l=True)])
        
        self.model_1 = m_1
        self.model_1._trigger_params_changed()
        
        
        self.mll_1_1 = float(self.model_1._log_marginal_likelihood)
        #self.g_mll_1_1 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        self.g_mll_1_1 = self.model_1._log_likelihood_gradients().copy()
        
        self.model_1.checkgrad(verbose=False)
        
#        self.model_2 = copy.deepcopy(m_1)
        
        self.model_1.set_DataStreamer(data_streamer)
        self.model_1._trigger_params_changed()
        
        self.model_1._next_minibatch()
        self.model_1._trigger_params_changed()
        
        self.mll_1_2 = float(self.model_1._log_marginal_likelihood)
        #self.g_mll_1_2 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        self.g_mll_1_2 = self.model_1._log_likelihood_gradients().copy()
        
        data_streamer_1 = StdMemoryDataStreamer(Y_2, U_2, sequences_no)
        
        self.model_1.set_DataStreamer(data_streamer_1)
        self.model_1._next_minibatch()
        self.model_1._trigger_params_changed()
        
        self.mll_2_1 = float(self.model_1._log_marginal_likelihood)
        
        
        # exclude 'init_Xs' and 'X_var' from gradients
        #self.g_mll_2_1 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        self.g_mll_2_1 = self.model_1._log_likelihood_gradients().copy()
        
        #import pdb; pdb.set_trace()
        
        self.model_1._next_minibatch()
        self.model_1._trigger_params_changed()
        
        self.mll_2_2 = float(self.model_1._log_marginal_likelihood)
        
        # exclude 'init_Xs' and 'X_var' from gradients
        #self.g_mll_2_2 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        self.g_mll_2_2 = self.model_1._log_likelihood_gradients().copy()
    
    def test_perm_ds_two_minibatches(self,):
        
        #import pdb; pdb.set_trace()
        
        #np.testing.assert_almost_equal( self.mll_1_2, self.mll_1_1, decimal=9, err_msg="Likelihoods must be equal" )
        np.testing.assert_equal( np.isclose(self.mll_1_2, self.mll_1_1, atol = 0, rtol = 1e-14), True, err_msg="Likelihoods must be equal" )
        
        #np.testing.assert_array_equal( self.g_mll_1_2, self.g_mll_1_1, err_msg="Likelihood gradients must be equal" )
        np.testing.assert_equal( np.all( np.isclose(self.g_mll_1_2, self.g_mll_1_1, atol = 0, rtol = 1e-11)), True, err_msg="Likelihood gradients must be equal" )
    
    def test_perm_ds_sum_minibatches(self,):
        
        
        #import pdb; pdb.set_trace()
        #np.testing.assert_equal( self.mll_2_1 + self.mll_2_2, self.mll_1_1, err_msg="Likelihoods must be equal" ) #decimal=9
        np.testing.assert_equal( np.isclose(float(self.mll_2_1) + float(self.mll_2_2), self.mll_1_1, atol = 0, rtol = 1e-14), True, err_msg="Likelihoods must be equal" )
        
        #np.testing.assert_array_equal( self.g_mll_2_1 + self.g_mll_2_2, self.g_mll_1_1, err_msg="Likelihood gradients must be equal" )
        np.testing.assert_equal( np.all( np.isclose(self.g_mll_2_1 + self.g_mll_2_2, self.g_mll_1_1, atol = 0, rtol = 1e-11)), True, err_msg="Likelihood gradients must be equal" )

class Gru_RGP_Test(unittest.TestCase):
    """
    Test the Deepautoreg_rnn model. Test rnn as a recognition model.
    
    The test classes [ Rnn_RGP_Test, Lstm_RGP_Test, Gru_RGP_Test, Gru_bidirect_RGP_Test ],
    do exactly the same testing except the back constrain neural network is different for each of them.
    """
    
    def setUp(self):
        
        u_dim = 2
        y_dim = 3
        ts_length = 20
        sequences_no = 3
        #U, Y = generate_data( sequences_no, ts_length, u_dim = u_dim, y_dim = y_dim)
        U_2, Y_2 = generate_data( sequences_no*2, ts_length, u_dim = u_dim, y_dim = y_dim)
        
        Q = 3 # 200 # Inducing points num. Take small number ofr speed
        
        back_cstr = True
        inference_method = 'svi'
        minibatch_inference = True
        
#        # 1 layer:
#        win_out = 3
#        win_in = 2
#        wins = [0, win_out] # 0-th is output layer
#        nDims = [y_dim,2]
        
        # 2 layers:
        win_out = 3
        win_in = 2
        wins = [0, win_out, win_out]
        nDims = [y_dim, 2,3] # 
        
        rnn_hidden_dims = [9,] # rnn hidden dimension
        rnn_type='gru'
        rnn_bidirectional=False
        rnn_h0_init='zero'
        
        #print("Input window:  ", win_in)
        #print("Output window:  ", win_out)
        
        data_streamer = RandomPermutationDataStreamer(Y_2, U_2)
        minibatch_index, minibatch_indices, Y_mb, X_mb = data_streamer.next_minibatch()
        
        m_1 = autoreg.DeepAutoreg_rnn(wins, Y_mb, U=X_mb, U_win=win_in,
                                num_inducing=Q, back_cstr=back_cstr, nDims=nDims,
                                
                                rnn_hidden_dims=rnn_hidden_dims,
                                rnn_type=rnn_type,
                                rnn_bidirectional=rnn_bidirectional,
                                rnn_h0_init=rnn_h0_init,
                                
                                inference_method=inference_method, # Inference method
                                minibatch_inference = minibatch_inference,
                                mb_inf_tot_data_size = sequences_no*2,
                                mb_inf_sample_idxes = minibatch_indices,
#                                # 1 layer:
#                                kernels=[GPy.kern.RBF(win_out*nDims[1],ARD=True,inv_l=True),
#                                         GPy.kern.RBF( (win_in + win_out) * nDims[1], ARD=True,inv_l=True)] )
        
                                 # 2 layers:
                                kernels=[GPy.kern.RBF(win_out*nDims[1],ARD=True,inv_l=True),
                                         GPy.kern.RBF(win_out*nDims[1] + win_out*nDims[2],ARD=True,inv_l=True),
                                         GPy.kern.RBF(win_out*nDims[2] + win_in*u_dim,ARD=True,inv_l=True)])
        
        self.model_1 = m_1
        self.model_1._trigger_params_changed()
        
        
        self.mll_1_1 = float(self.model_1._log_marginal_likelihood)
        #self.g_mll_1_1 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        self.g_mll_1_1 = self.model_1._log_likelihood_gradients().copy()
        
        self.model_1.checkgrad(verbose=False)
        
#        self.model_2 = copy.deepcopy(m_1)
        
        self.model_1.set_DataStreamer(data_streamer)
        self.model_1._trigger_params_changed()
        
        self.model_1._next_minibatch()
        self.model_1._trigger_params_changed()
        
        self.mll_1_2 = float(self.model_1._log_marginal_likelihood)
        #self.g_mll_1_2 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        self.g_mll_1_2 = self.model_1._log_likelihood_gradients().copy()
        
        data_streamer_1 = StdMemoryDataStreamer(Y_2, U_2, sequences_no)
        
        self.model_1.set_DataStreamer(data_streamer_1)
        self.model_1._next_minibatch()
        self.model_1._trigger_params_changed()
        
        self.mll_2_1 = float(self.model_1._log_marginal_likelihood)
        
        
        # exclude 'init_Xs' and 'X_var' from gradients
        #self.g_mll_2_1 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        self.g_mll_2_1 = self.model_1._log_likelihood_gradients().copy()
        
        #import pdb; pdb.set_trace()
        
        self.model_1._next_minibatch()
        self.model_1._trigger_params_changed()
        
        self.mll_2_2 = float(self.model_1._log_marginal_likelihood)
        
        # exclude 'init_Xs' and 'X_var' from gradients
        #self.g_mll_2_2 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        self.g_mll_2_2 = self.model_1._log_likelihood_gradients().copy()
    
    def test_perm_ds_two_minibatches(self,):
        
        #import pdb; pdb.set_trace()
        
        #np.testing.assert_almost_equal( self.mll_1_2, self.mll_1_1, decimal=9, err_msg="Likelihoods must be equal" )
        np.testing.assert_equal( np.isclose(self.mll_1_2, self.mll_1_1, atol = 0, rtol = 1e-14), True, err_msg="Likelihoods must be equal" )
        
        #np.testing.assert_array_equal( self.g_mll_1_2, self.g_mll_1_1, err_msg="Likelihood gradients must be equal" )
        np.testing.assert_equal( np.all( np.isclose(self.g_mll_1_2, self.g_mll_1_1, atol = 0, rtol = 1e-11)), True, err_msg="Likelihood gradients must be equal" )
    
    def test_perm_ds_sum_minibatches(self,):
        
        
        #import pdb; pdb.set_trace()
        #np.testing.assert_equal( self.mll_2_1 + self.mll_2_2, self.mll_1_1, err_msg="Likelihoods must be equal" ) #decimal=9
        np.testing.assert_equal( np.isclose(float(self.mll_2_1) + float(self.mll_2_2), self.mll_1_1, atol = 0, rtol = 1e-14), True, err_msg="Likelihoods must be equal" )
        
        #np.testing.assert_array_equal( self.g_mll_2_1 + self.g_mll_2_2, self.g_mll_1_1, err_msg="Likelihood gradients must be equal" )
        np.testing.assert_equal( np.all( np.isclose(self.g_mll_2_1 + self.g_mll_2_2, self.g_mll_1_1, atol = 0, rtol = 1e-11)), True, err_msg="Likelihood gradients must be equal" )

class Gru_bidirect_RGP_Test(unittest.TestCase):
    """
    Test the Deepautoreg_rnn model. Test rnn as a recognition model.
    
    The test classes [ Rnn_RGP_Test, Lstm_RGP_Test, Gru_RGP_Test, Gru_bidirect_RGP_Test ],
    do exactly the same testing except the back constrain neural network is different for each of them.
    """
    
    def setUp(self):
        
        u_dim = 2
        y_dim = 3
        ts_length = 20
        sequences_no = 3
        #U, Y = generate_data( sequences_no, ts_length, u_dim = u_dim, y_dim = y_dim)
        U_2, Y_2 = generate_data( sequences_no*2, ts_length, u_dim = u_dim, y_dim = y_dim)
        
        Q = 3 # 200 # Inducing points num. Take small number ofr speed
        
        back_cstr = True
        inference_method = 'svi'
        minibatch_inference = True
        
#        # 1 layer:
#        win_out = 3
#        win_in = 2
#        wins = [0, win_out] # 0-th is output layer
#        nDims = [y_dim,2]
        
        # 2 layers:
        win_out = 3
        win_in = 2
        wins = [0, win_out, win_out]
        nDims = [y_dim, 2,3] # 
        
        rnn_hidden_dims = [9,] # rnn hidden dimension
        rnn_type='gru'
        rnn_bidirectional=True
        rnn_h0_init='zero'
        
        #print("Input window:  ", win_in)
        #print("Output window:  ", win_out)
        
        data_streamer = RandomPermutationDataStreamer(Y_2, U_2)
        minibatch_index, minibatch_indices, Y_mb, X_mb = data_streamer.next_minibatch()
        
        m_1 = autoreg.DeepAutoreg_rnn(wins, Y_mb, U=X_mb, U_win=win_in,
                                num_inducing=Q, back_cstr=back_cstr, nDims=nDims,
                                
                                rnn_hidden_dims=rnn_hidden_dims,
                                rnn_type=rnn_type,
                                rnn_bidirectional=rnn_bidirectional,
                                rnn_h0_init=rnn_h0_init,
                                
                                inference_method=inference_method, # Inference method
                                minibatch_inference = minibatch_inference,
                                mb_inf_tot_data_size = sequences_no*2,
                                mb_inf_sample_idxes = minibatch_indices,
#                                # 1 layer:
#                                kernels=[GPy.kern.RBF(win_out*nDims[1],ARD=True,inv_l=True),
#                                         GPy.kern.RBF( (win_in + win_out) * nDims[1], ARD=True,inv_l=True)] )
        
                                 # 2 layers:
                                kernels=[GPy.kern.RBF(win_out*nDims[1],ARD=True,inv_l=True),
                                         GPy.kern.RBF(win_out*nDims[1] + win_out*nDims[2],ARD=True,inv_l=True),
                                         GPy.kern.RBF(win_out*nDims[2] + win_in*u_dim,ARD=True,inv_l=True)])
        
        
        self.model_1 = m_1
        self.model_1._trigger_params_changed()
        
        
        self.mll_1_1 = float(self.model_1._log_marginal_likelihood)
        #self.g_mll_1_1 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        self.g_mll_1_1 = self.model_1._log_likelihood_gradients().copy()
        
        self.model_1.checkgrad(verbose=False)
        
#        self.model_2 = copy.deepcopy(m_1)
        
        self.model_1.set_DataStreamer(data_streamer)
        self.model_1._trigger_params_changed()
        
        self.model_1._next_minibatch()
        self.model_1._trigger_params_changed()
        
        self.mll_1_2 = float(self.model_1._log_marginal_likelihood)
        #self.g_mll_1_2 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        self.g_mll_1_2 = self.model_1._log_likelihood_gradients().copy()
        
        data_streamer_1 = StdMemoryDataStreamer(Y_2, U_2, sequences_no)
        
        self.model_1.set_DataStreamer(data_streamer_1)
        self.model_1._next_minibatch()
        self.model_1._trigger_params_changed()
        
        self.mll_2_1 = float(self.model_1._log_marginal_likelihood)
        
        
        # exclude 'init_Xs' and 'X_var' from gradients
        #self.g_mll_2_1 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        self.g_mll_2_1 = self.model_1._log_likelihood_gradients().copy()
        
        #import pdb; pdb.set_trace()
        
        self.model_1._next_minibatch()
        self.model_1._trigger_params_changed()
        
        self.mll_2_2 = float(self.model_1._log_marginal_likelihood)
        
        # exclude 'init_Xs' and 'X_var' from gradients
        #self.g_mll_2_2 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        self.g_mll_2_2 = self.model_1._log_likelihood_gradients().copy()
    
    def test_perm_ds_two_minibatches(self,):
        
        #import pdb; pdb.set_trace()
        
        #np.testing.assert_almost_equal( self.mll_1_2, self.mll_1_1, decimal=9, err_msg="Likelihoods must be equal" )
        np.testing.assert_equal( np.isclose(self.mll_1_2, self.mll_1_1, atol = 0, rtol = 1e-14), True, err_msg="Likelihoods must be equal" )
        
        #np.testing.assert_array_equal( self.g_mll_1_2, self.g_mll_1_1, err_msg="Likelihood gradients must be equal" )
        np.testing.assert_equal( np.all( np.isclose(self.g_mll_1_2, self.g_mll_1_1, atol = 0, rtol = 1e-11)), True, err_msg="Likelihood gradients must be equal" )
    
    def test_perm_ds_sum_minibatches(self,):
        
        
        #import pdb; pdb.set_trace()
        #np.testing.assert_equal( self.mll_2_1 + self.mll_2_2, self.mll_1_1, err_msg="Likelihoods must be equal" ) #decimal=9
        np.testing.assert_equal( np.isclose(float(self.mll_2_1) + float(self.mll_2_2), self.mll_1_1, atol = 0, rtol = 1e-14), True, err_msg="Likelihoods must be equal" )
        
        #np.testing.assert_array_equal( self.g_mll_2_1 + self.g_mll_2_2, self.g_mll_1_1, err_msg="Likelihood gradients must be equal" )
        np.testing.assert_equal( np.all( np.isclose(self.g_mll_2_1 + self.g_mll_2_2, self.g_mll_1_1, atol = 0, rtol = 1e-11)), True, err_msg="Likelihood gradients must be equal" )


if __name__ == '__main__':
    pass

#    tt1 = Rnn_RGP_Test('test_perm_ds_two_minibatches')
#    tt1.setUp()
#    tt1.test_perm_ds_two_minibatches()
#    #tt.test_gradients()
#    
    tt2 = Rnn_RGP_Test('test_perm_ds_sum_minibatches')
    tt2.setUp()
    tt2.test_perm_ds_two_minibatches()
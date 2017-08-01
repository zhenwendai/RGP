#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import GPy
import os
import copy

import autoreg
from autoreg.data_streamers import TrivialDataStreamer, RandomPermutationDataStreamer

# base_path = os.path.dirname(__file__)


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
        
class TrivialStreamer_Test(unittest.TestCase):
    """
    This class tests that the model without minibatch turned on and with it
    give the same likelihood. Gradients are not compared but tested separately
    """
    
    def setUp(self):
        
        print("ho-ho")
        
        u_dim = 2
        y_dim = 3
        U, Y = generate_data( 3, 20, u_dim = 2, y_dim = 3)
        
        Q = 3 # 200 # Inducing points num. Take small number ofr speed
        
        back_cstr = True
        inference_method = 'svi'
        minibatch_inference = True
        
    #        # 1 layer:
    #        wins = [0, win_out] # 0-th is output layer
    #        nDims = [out_train.shape[1],1]
        
        # 2 layers:
        win_out = 3
        win_in = 2
        wins = [0, win_out, win_out]
        nDims = [y_dim, 2,3] # 
        
        MLP_dims = [3,2] # !!! 300, 200 For speed.
        #print("Input window:  ", win_in)
        #print("Output window:  ", win_out)
        
        m = autoreg.DeepAutoreg_new(wins, Y, U=U, U_win=win_in,
                                num_inducing=Q, back_cstr=back_cstr, MLP_dims=MLP_dims, nDims=nDims,
                                init='Y', # how to initialize hidden states means
                                X_variance=0.05, # how to initialize hidden states variances
                                inference_method=inference_method, # Inference method
                                minibatch_inference = minibatch_inference,
                                # 1 layer:
                                # kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True),
                                 #        GPy.kern.RBF(win_in + win_out,ARD=True,inv_l=True)] )
        
                                # 2 layers:
                                kernels=[GPy.kern.RBF(win_out*nDims[1],ARD=True,inv_l=True),
                                         GPy.kern.RBF(win_out*nDims[1] + win_out*nDims[2],ARD=True,inv_l=True),
                                         GPy.kern.RBF(win_out*nDims[2] + win_in*u_dim,ARD=True,inv_l=True)])
        
        self.model_1 = m
        self.model_1._trigger_params_changed()
        
        self.model_2 = copy.deepcopy(m)
        
        data_streamer = TrivialDataStreamer(Y, U)
        self.model_2.set_DataStreamer(data_streamer)
        self.model_2._trigger_params_changed()
        print("ho-ho")
        
    def test_gradient(self):
        self.assertTrue(self.model_1.checkgrad())
        self.assertTrue(self.model_2.checkgrad())
        
    def test_likelihoodEquivalence(self):
        
        self.model_2._next_minibatch()
        self.model_2._trigger_params_changed()
        
        self.assertEqual( self.model_1._log_marginal_likelihood, self.model_2._log_marginal_likelihood, msg="Likelihoods must be equal" )
        
        
        
class TrivialStreamer_Test(unittest.TestCase):
    """
    This class tests that the model without minibatch turned on and with it
    give the same likelihood. Gradients are not compared but tested separately
    """
    
    def setUp(self):
        
        u_dim = 2
        y_dim = 3
        U, Y = generate_data( 3, 20, u_dim = 2, y_dim = 3)
        
        Q = 3 # 200 # Inducing points num. Take small number ofr speed
        
        back_cstr = True
        inference_method = 'svi'
        minibatch_inference = True
        
    #        # 1 layer:
    #        wins = [0, win_out] # 0-th is output layer
    #        nDims = [out_train.shape[1],1]
        
        # 2 layers:
        win_out = 3
        win_in = 2
        wins = [0, win_out, win_out]
        nDims = [y_dim, 2,3] # 
        
        MLP_dims = [3,2] # !!! 300, 200 For speed.
        #print("Input window:  ", win_in)
        #print("Output window:  ", win_out)
        
        m = autoreg.DeepAutoreg_new(wins, Y, U=U, U_win=win_in,
                                num_inducing=Q, back_cstr=back_cstr, MLP_dims=MLP_dims, nDims=nDims,
                                init='Y', # how to initialize hidden states means
                                X_variance=0.05, # how to initialize hidden states variances
                                inference_method=inference_method, # Inference method
                                minibatch_inference = minibatch_inference,
                                # 1 layer:
                                # kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True),
                                 #        GPy.kern.RBF(win_in + win_out,ARD=True,inv_l=True)] )
        
                                # 2 layers:
                                kernels=[GPy.kern.RBF(win_out*nDims[1],ARD=True,inv_l=True),
                                         GPy.kern.RBF(win_out*nDims[1] + win_out*nDims[2],ARD=True,inv_l=True),
                                         GPy.kern.RBF(win_out*nDims[2] + win_in*u_dim,ARD=True,inv_l=True)])
        
        self.model_1 = m
        self.model_1._trigger_params_changed()
        
        self.model_2 = copy.deepcopy(m)
        
        data_streamer = TrivialDataStreamer(Y, U)
        self.model_2.set_DataStreamer(data_streamer)
        self.model_2._trigger_params_changed()
        
    def test_gradient(self):
        self.assertTrue(self.model_1.checkgrad())
        self.assertTrue(self.model_2.checkgrad())
        
    def test_likelihoodEquivalence(self):
        
        self.model_2._next_minibatch()
        self.model_2._trigger_params_changed()
        
        self.assertEqual( self.model_1._log_marginal_likelihood, self.model_2._log_marginal_likelihood, msg="Likelihoods must be equal" )
        

class RandomPermutationStreamer_Test(unittest.TestCase):
    """
    This class tests that randomly permuted minibatches return the same
    likelihood. Gradients are not compared but tested separately
    """
    
    def setUp(self):
        
        u_dim = 2
        y_dim = 3
        U, Y = generate_data( 3, 20, u_dim = 2, y_dim = 3)
        
        Q = 3 # 200 # Inducing points num. Take small number ofr speed
        
        back_cstr = True
        inference_method = 'svi'
        minibatch_inference = True
        
    #        # 1 layer:
    #        wins = [0, win_out] # 0-th is output layer
    #        nDims = [out_train.shape[1],1]
        
        # 2 layers:
        win_out = 3
        win_in = 2
        wins = [0, win_out, win_out]
        nDims = [y_dim, 2,3] # 
        
        MLP_dims = [3,2] # !!! 300, 200 For speed.
        #print("Input window:  ", win_in)
        #print("Output window:  ", win_out)
        
        m = autoreg.DeepAutoreg_new(wins, Y, U=U, U_win=win_in,
                                num_inducing=Q, back_cstr=back_cstr, MLP_dims=MLP_dims, nDims=nDims,
                                init='Y', # how to initialize hidden states means
                                X_variance=0.05, # how to initialize hidden states variances
                                inference_method=inference_method, # Inference method
                                minibatch_inference = minibatch_inference,
                                # 1 layer:
                                # kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True),
                                 #        GPy.kern.RBF(win_in + win_out,ARD=True,inv_l=True)] )
        
                                # 2 layers:
                                kernels=[GPy.kern.RBF(win_out*nDims[1],ARD=True,inv_l=True),
                                         GPy.kern.RBF(win_out*nDims[1] + win_out*nDims[2],ARD=True,inv_l=True),
                                         GPy.kern.RBF(win_out*nDims[2] + win_in*u_dim,ARD=True,inv_l=True)])
        
        self.model_1 = m
        self.model_1._trigger_params_changed()
        
        self.mll_1 = self.model_1._log_marginal_likelihood
        self.g_mll_1 = self.model_1._log_likelihood_gradients
        
        
        data_streamer = RandomPermutationDataStreamer(Y, U)
        self.model_1.set_DataStreamer(data_streamer)
        self.model_1._trigger_params_changed()
        
    def test_two_minibatches(self):
        self.assertTrue(self.model_1.checkgrad())
        
        self.model_1._next_minibatch()
        self.model_1._trigger_params_changed()
        
        self.assertEqual( self.model_1._log_marginal_likelihood, self.mll_1, msg="Likelihoods must be equal" )
        self.assertEqual( self.model_1._log_likelihood_gradients, self.g_mll_1, msg="Likelihood gradients must be equal" )
        
        self.model_1._next_minibatch()
        self.model_1._trigger_params_changed()
        
        self.assertEqual( self.model_1._log_marginal_likelihood, self.mll_1, msg="Likelihoods must be equal" )
        self.assertEqual( self.model_1._log_likelihood_gradients, self.g_mll_1, msg="Likelihood gradients must be equal" )
        
#    def test_likelihoodEquivalence(self):
#        
#        self.model_2._next_minibatch()
#        self.model_2._trigger_params_changed()
#        
#        self.assertEqual( self.model_1._log_marginal_likelihood, self.model_2._log_marginal_likelihood, msg="Likelihoods must be equal" )
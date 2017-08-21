#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import GPy
import os
import copy

import autoreg
from autoreg.data_streamers import TrivialDataStreamer, RandomPermutationDataStreamer, StdMemoryDataStreamer

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
                                mb_inf_tot_data_size = len(Y),
                                mb_inf_init_xs_means='one',
                                mb_inf_init_xs_vars='one',
                                mb_inf_sample_idxes = range( len(Y)),
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
        
        
        
class SviRGPparams_One_TrivialDS_Test(unittest.TestCase):
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
                                mb_inf_tot_data_size = len(Y),
                                mb_inf_init_xs_means='one',
                                mb_inf_init_xs_vars='one',
                                mb_inf_sample_idxes = range( len(Y)), 
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
        

class SviRGPparams_One_PermStdDS_Test(unittest.TestCase):
    """
    This class tests that randomly permuted minibatches and standard minibetches return the same
    likelihood. Gradients are not compared but tested separately
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
        
        data_streamer = RandomPermutationDataStreamer(Y_2, U_2)
        minibatch_index, minibatch_indices, Y_mb, X_mb = data_streamer.next_minibatch()
        
        m_1 = autoreg.DeepAutoreg_new(wins, Y_mb, U=X_mb, U_win=win_in,
                                num_inducing=Q, back_cstr=back_cstr, MLP_dims=MLP_dims, nDims=nDims,
                                init='Y', # how to initialize hidden states means
                                X_variance=0.05, # how to initialize hidden states variances
                                inference_method=inference_method, # Inference method
                                minibatch_inference = minibatch_inference,
                                mb_inf_tot_data_size = sequences_no*2,
                                mb_inf_init_xs_means='one',
                                mb_inf_init_xs_vars='one',
                                mb_inf_sample_idxes = minibatch_indices,
                                # 1 layer:
                                # kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True),
                                 #        GPy.kern.RBF(win_in + win_out,ARD=True,inv_l=True)] )
        
                                # 2 layers:
                                kernels=[GPy.kern.RBF(win_out*nDims[1],ARD=True,inv_l=True),
                                         GPy.kern.RBF(win_out*nDims[1] + win_out*nDims[2],ARD=True,inv_l=True),
                                         GPy.kern.RBF(win_out*nDims[2] + win_in*u_dim,ARD=True,inv_l=True)])
        
        self.model_1 = m_1
        self.model_1._trigger_params_changed()
        
        self.mll_1_1 = float(self.model_1._log_marginal_likelihood)
        self.g_mll_1_1 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        #self.g_mll_1_1 = self.model_1._log_likelihood_gradients().copy()
        
        self.model_2 = copy.deepcopy(m_1)
        
        self.model_1.set_DataStreamer(data_streamer)
        self.model_1._trigger_params_changed()
        
        
        self.model_1._next_minibatch()
        self.model_1._trigger_params_changed()
        
        self.mll_1_2 = float(self.model_1._log_marginal_likelihood)
        self.g_mll_1_2 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        #self.g_mll_1_2 = self.model_1._log_likelihood_gradients().copy()
        
        data_streamer_1 = StdMemoryDataStreamer(Y_2, U_2, sequences_no)
        
        self.model_1.set_DataStreamer(data_streamer_1)
        self.model_1._next_minibatch()
        self.model_1._trigger_params_changed()
        
        self.mll_2_1 = float(self.model_1._log_marginal_likelihood)
        self.g_mll_2_1 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        
        #import pdb; pdb.set_trace()
        
        self.model_1._next_minibatch()
        self.model_1._trigger_params_changed()
        
        self.mll_2_2 = float(self.model_1._log_marginal_likelihood)
        self.g_mll_2_2 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        
    def test_gradients(self):
        self.assertTrue(self.model_2.checkgrad())
    
            
    def test_perm_ds_two_minibatches(self,):
        #np.testing.assert_almost_equal( self.mll_1_2, self.mll_1_1, decimal=9, err_msg="Likelihoods must be equal" )
        np.testing.assert_equal( np.isclose(self.mll_1_2, self.mll_1_1, atol = 0, rtol = 1e-14), True, err_msg="Likelihoods must be equal" )
        
        #np.testing.assert_array_equal( self.g_mll_1_2, self.g_mll_1_1, err_msg="Likelihood gradients must be equal" )
        np.testing.assert_equal( np.all( np.isclose(self.g_mll_1_2, self.g_mll_1_1, atol = 0, rtol = 1e-11)), True, err_msg="Likelihood gradients must be equal" )
    
    def test_perm_ds_sum_minibatches(self,):
        
        #np.testing.assert_equal( self.mll_2_1 + self.mll_2_2, self.mll_1_1, err_msg="Likelihoods must be equal" ) #decimal=9
        np.testing.assert_equal( np.isclose(float(self.mll_2_1) + float(self.mll_2_2), self.mll_1_1, atol = 0, rtol = 1e-14), True, err_msg="Likelihoods must be equal" )
        
        #import pdb; pdb.set_trace()
        
        #np.testing.assert_array_equal( self.g_mll_2_1 + self.g_mll_2_2, self.g_mll_1_1, err_msg="Likelihood gradients must be equal" )
        np.testing.assert_equal( np.all( np.isclose(self.g_mll_2_1 + self.g_mll_2_2, self.g_mll_1_1, atol = 0, rtol = 1e-11)), True, err_msg="Likelihood gradients must be equal" )
        
class SviRGPparams_All_PermStdDS_Test(unittest.TestCase):
    """
    This class tests that randomly permuted minibatches (and standard minibatch) return the same
    likelihood. 
    
    Gradients are not compared but tested separately
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
        
        data_streamer = RandomPermutationDataStreamer(Y_2, U_2)
        minibatch_index, minibatch_indices, Y_mb, X_mb = data_streamer.next_minibatch()
        
        m_1 = autoreg.DeepAutoreg_new(wins, Y_mb, U=X_mb, U_win=win_in,
                                num_inducing=Q, back_cstr=back_cstr, MLP_dims=MLP_dims, nDims=nDims,
                                init='Y', # how to initialize hidden states means
                                X_variance=0.05, # how to initialize hidden states variances
                                inference_method=inference_method, # Inference method
                                minibatch_inference = minibatch_inference,
                                mb_inf_tot_data_size = sequences_no*2,
                                mb_inf_init_xs_means='all',
                                mb_inf_init_xs_vars='all',
                                mb_inf_sample_idxes = minibatch_indices,
                                # 1 layer:
                                # kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True),
                                 #        GPy.kern.RBF(win_in + win_out,ARD=True,inv_l=True)] )
        
                                # 2 layers:
                                kernels=[GPy.kern.RBF(win_out*nDims[1],ARD=True,inv_l=True),
                                         GPy.kern.RBF(win_out*nDims[1] + win_out*nDims[2],ARD=True,inv_l=True),
                                         GPy.kern.RBF(win_out*nDims[2] + win_in*u_dim,ARD=True,inv_l=True)])
        
        self.model_1 = m_1
        self.model_1._trigger_params_changed()
        
        
        self.mll_1_1 = float(self.model_1._log_marginal_likelihood)
        self.g_mll_1_1 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        #self.g_mll_1_1 = self.model_1._log_likelihood_gradients().copy()
        
        self.model_2 = copy.deepcopy(m_1)
        
        self.model_1.set_DataStreamer(data_streamer)
        self.model_1._trigger_params_changed()
        
        self.model_1._next_minibatch()
        self.model_1._trigger_params_changed()
        
        self.mll_1_2 = float(self.model_1._log_marginal_likelihood)
        self.g_mll_1_2 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        #self.g_mll_1_2 = self.model_1._log_likelihood_gradients().copy()
        
        data_streamer_1 = StdMemoryDataStreamer(Y_2, U_2, sequences_no)
        
        self.model_1.set_DataStreamer(data_streamer_1)
        self.model_1._next_minibatch()
        self.model_1._trigger_params_changed()
        
        self.mll_2_1 = float(self.model_1._log_marginal_likelihood)
        
        
        # exclude 'init_Xs' and 'X_var' from gradients
        self.g_mll_2_1 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        
        #import pdb; pdb.set_trace()
        
        self.model_1._next_minibatch()
        self.model_1._trigger_params_changed()
        
        self.mll_2_2 = float(self.model_1._log_marginal_likelihood)
        
        # exclude 'init_Xs' and 'X_var' from gradients
        self.g_mll_2_2 = np.hstack( self.model_1[pp.replace(' ', '_')].gradient.flatten() for pp in self.model_1.parameter_names() if ('init_Xs' not in pp) and ('X_var' not in pp) ).copy()
        
    def test_gradients(self):
        self.assertTrue(self.model_2.checkgrad())
    
    def test_perm_ds_two_minibatches(self,):
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
#    tt = SviRGPparams_All_PermutDS_Test('test_perm_ds_two_minibatches')
#    tt.setUp()
#    tt.test_perm_ds_two_minibatches()
    
#    tt = SviRGPparams_All_PermStdDS_Test('test_perm_ds_sum_minibatches')
#    tt.setUp()
#    tt.test_perm_ds_sum_minibatches()
    
#    tt = SviRGPparams_One_PermutDS_Test('test_two_minibatches')
#    tt.setUp()
#    tt.test_two_minibatches()

#    tt = SviRGPparams_One_TrivialDS_Test('test_likelihoodEquivalence')
#    tt.setUp()
#    tt.test_likelihoodEquivalence()
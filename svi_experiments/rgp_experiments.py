#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 11:51:21 2017

@author: grigoral
"""
from __future__ import print_function

import autoreg
import GPy
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as io
from autoreg.benchmark import tasks

# Define class for normalization
class Normalize(object):
    
    def __init__(self, data, name, norm_name):
        
        self.data_mean = data.mean(axis=0)
        self.data_std = data.std(axis=0)
        self.normalization_computed = True
        
        setattr(self, name, data)                         
        setattr(self, norm_name, (data-self.data_mean) / self.data_std )
        
    def normalize(self, data, name, norm_name):
            if hasattr(self,norm_name):
                raise ValueError("This normalization name already exist, choose another one")
            
            setattr(self, name, data )
            setattr(self, norm_name, (data-self.data_mean) / self.data_std )
            
            
                                     
    def denormalize(self, data):
                                   
        return data*self.data_std + self.data_mean 

def prepare_data(task_name, normalize=False):
    task = getattr( tasks, task_name)
    task = task()
    task.load_data()
    
    print("Data OUT train shape:  ", task.data_out_train.shape)
    print("Data IN train shape:  ", task.data_in_train.shape)
    print("Data OUT test shape:  ", task.data_out_test.shape)
    print("Data IN test shape:  ", task.data_in_test.shape)

    normalize = True
    in_data = Normalize(task.data_in_train,'in_train','in_train_norm' )
    out_data = Normalize(task.data_out_train,'out_train','out_train_norm' )
    
    in_data.normalize(task.data_in_test, 'in_test','in_test_norm')
    out_data.normalize(task.data_out_test, 'out_test','out_test_norm')
    
    if normalize:
        out_train = out_data.out_train_norm #out_data.out_train 
        in_train = in_data.in_train_norm # in_data.in_train
        out_test = out_data.out_test_norm #out_data.out_test
        in_test = in_data.in_test_norm #in_data.in_test
    else:
        out_train = out_data.out_train  #out_data.out_train 
        in_train = in_data.in_train # in_data.in_train
        out_test = out_data.out_test #out_data.out_test
        in_test = in_data.in_test #in_data.in_test
    
    return out_train, in_train, task

class IE5_experiment_1( object ):
    """
    Tested parameters are: initial number of optimization runs, number of hidden dims, number of inducing points.
    
    After the first experiment the conclusion is that 1 hidden dim is the best, but also
    the optimization is not very explorative.
    Probably there was an error in the experi,ent setup because I did not change the number of hidden layers
    only the number of hidden dimensions in 1 layer.
    
    Best values: ini_runs = 160.0,     hidden dim=1., Q=50. (237.44060068)
    Iteration 21
    """
    
    def __init__( self, initial_counter, iter_nums):
        self.counter = initial_counter
        self.iter_nums = iter_nums
        
    def __call__( self, *args, **kwargs ):
        #import pdb; pdb.set_trace()
        new_args = (self.counter,self.iter_nums,) + tuple( [ int(args[0][0,i]) for i in range(args[0].shape[1]) ] )
        ret = self.IE5_experiment_1_code( *new_args, **kwargs)
        self.counter += 1
        return ret
    
    @staticmethod
    def IE5_experiment_1_code( bo_iter_no, p_iter_num, p_init_runs, p_hidden_dims, p_Q):
        """
        Hyper parameter search for IE5 data, varying small number of parameters.
        One hidden layer.
        """
        
        # p_iter_num # How many iteration are needed to evaluate one set of hyper parameterss
        
        # task names:
        # Actuator, Ballbeam, Drive, Gas_furnace, Flutter, Dryer, Tank,
        # IdentificationExample1..5
        #import pdb; pdb.set_trace()
        out_train, in_train, task = prepare_data('IdentificationExample5', normalize=True)
        
        p_task_name = 'IE5_1'
        #p_iteration = 
        train_U = in_train.copy()
        train_Y = out_train.copy()
        p_init_runs = p_init_runs
        p_max_runs = 10000
        p_num_layers = 1
        p_hidden_dims = [p_hidden_dims,]
        p_inference_method = None
        p_back_cstr = False
        p_MLP_Dims = None
        p_Q = p_Q
        p_win_in = task.win_in
        p_win_out = task.win_out
        p_init = 'Y'
        p_x_init_var = 0.05
        
        result = list()
        for i_no in range(0, p_iter_num): # iterations take into account model randomness e.g. initialization of inducing points
            res = rgp_experiment_raw(p_task_name, bo_iter_no*10+i_no, train_U, train_Y, p_init_runs, p_max_runs, p_num_layers, p_hidden_dims, 
                               p_inference_method, p_back_cstr, p_MLP_Dims, p_Q, p_win_in, p_win_out, p_init, p_x_init_var)
            result.append(res[0])
        #import pdb; pdb.set_trace()
        
        return np.array(((np.min(result),),))
    
class IE5_experiment_2( object ):
    """
    Tested parameters are: initial number of optimization runs, number of inducing points.
    
    Conclusions after the experiment: The output file contains only variables as var_1, var_2 etc.
    but Xavier said that the order is presearved.
    
    The optimal values are: init_runs = 110, Q (ind. num) = 200. (run 3), 240.44817869
    Bu the results are still the same from run to run.
    Total running time was 40hours on GPU machine.
    Maybe we can reduce the number of intrinsic iterations per evaluation.
    
    Now idea is to use manually designad initial values to run more proper experiment. (Experiment 3)
    
    """
    def __init__( self, initial_counter, iter_nums):
        self.counter = initial_counter
        self.iter_nums = iter_nums
        
    def __call__( self, *args, **kwargs ):
        #import pdb; pdb.set_trace()
        new_args = (self.counter,self.iter_nums,) + tuple( [ int(args[0][0,i]) for i in range(args[0].shape[1]) ] )
        ret = self.IE5_experiment_2_code( *new_args, **kwargs)
        self.counter += 1
        return ret
    
    @staticmethod
    def IE5_experiment_2_code( bo_iter_no, p_iter_num, p_init_runs, p_Q):
        """
        Hyper parameter search for IE5 data, varying small number of parameters.
        One hidden layer.
        """
        
        # p_iter_num # How many iteration are needed to evaluate one set of hyper parameterss
        
        # task names:
        # Actuator, Ballbeam, Drive, Gas_furnace, Flutter, Dryer, Tank,
        # IdentificationExample1..5
        #import pdb; pdb.set_trace()
        out_train, in_train, task = prepare_data('IdentificationExample5', normalize=True)
        
        p_task_name = 'IE5_2'
        #p_iteration = 
        train_U = in_train.copy()
        train_Y = out_train.copy()
        p_init_runs = p_init_runs
        p_max_runs = 10000
        p_num_layers = 1
        p_hidden_dims = [1,]
        p_inference_method = None
        p_back_cstr = False
        p_MLP_Dims = None
        p_Q = p_Q
        p_win_in = task.win_in
        p_win_out = task.win_out
        p_init = 'Y'
        p_x_init_var = 0.05
        
        result = list()
        for i_no in range(0, p_iter_num): # iterations take into account model randomness e.g. initialization of inducing points
            res = rgp_experiment_raw(p_task_name, bo_iter_no*10+i_no, train_U, train_Y, p_init_runs, p_max_runs, p_num_layers, p_hidden_dims, 
                               p_inference_method, p_back_cstr, p_MLP_Dims, p_Q, p_win_in, p_win_out, p_init, p_x_init_var)
            result.append(res[0])
        #import pdb; pdb.set_trace()
        
        return np.array(((np.min(result),),))

class IE5_experiment_3( object ):
    """
    Tested parameters are: initial number of number of layers, number of inducing points.
    Also, I do initial evaluations manually.
    
    Best values: 1 layer, 40 inducing points, (run 7) 242.67636311
    This value is laregr than in the other experiments. Manybe becuase there were only 2 internal runs
    in every function evaluation.
    
    """
    def __init__( self, initial_counter, iter_nums):
        self.counter = initial_counter
        self.iter_nums = iter_nums
        
    def __call__( self, *args, **kwargs ):
        #import pdb; pdb.set_trace()
        new_args = (self.counter,self.iter_nums,) + tuple( [ int(args[0][0,i]) for i in range(args[0].shape[1]) ] )
        ret = self.IE5_experiment_3_code( *new_args, **kwargs)
        self.counter += 1
        return ret
    
    @staticmethod
    def IE5_experiment_3_code( bo_iter_no, p_iter_num, p_layer_num, p_Q):
        """
        Hyper parameter search for IE5 data, varying small number of parameters.
        One hidden layer.
        """
        
        # p_iter_num # How many iteration are needed to evaluate one set of hyper parameterss
        
        # task names:
        # Actuator, Ballbeam, Drive, Gas_furnace, Flutter, Dryer, Tank,
        # IdentificationExample1..5
        #import pdb; pdb.set_trace()
        out_train, in_train, task = prepare_data('IdentificationExample5', normalize=True)
        
        p_task_name = 'IE5_2'
        #p_iteration = 
        train_U = in_train.copy()
        train_Y = out_train.copy()
        p_init_runs = 130
        p_max_runs = 10000
        p_num_layers = p_layer_num
        p_hidden_dims = [1,1,]
        p_inference_method = None
        p_back_cstr = False
        p_MLP_Dims = None
        p_Q = p_Q
        p_win_in = task.win_in
        p_win_out = task.win_out
        p_init = 'Y'
        p_x_init_var = 0.05
        
        result = list()
        for i_no in range(0, p_iter_num): # iterations take into account model randomness e.g. initialization of inducing points
            res = rgp_experiment_raw(p_task_name, bo_iter_no*10+i_no, train_U, train_Y, p_init_runs, p_max_runs, p_num_layers, p_hidden_dims, 
                               p_inference_method, p_back_cstr, p_MLP_Dims, p_Q, p_win_in, p_win_out, p_init, p_x_init_var)
            result.append(res[0])
        #import pdb; pdb.set_trace()
        
        return np.array(((np.min(result),),))

class IE5_experiment_4( object ):
    """
    SVI inference
    Tested parameters are: number of initial runs, number of inducing points.
    
    First SVI experiment.
    """
    def __init__( self, initial_counter, iter_nums):
        self.counter = initial_counter
        self.iter_nums = iter_nums
        
    def __call__( self, *args, **kwargs ):
        #import pdb; pdb.set_trace()
        new_args = (self.counter,self.iter_nums,) + tuple( [ int(args[0][0,i]) for i in range(args[0].shape[1]) ] )
        ret = self.IE5_experiment_4_code( *new_args, **kwargs)
        self.counter += 1
        return ret
    
    @staticmethod
    def IE5_experiment_4_code( bo_iter_no, p_iter_num, p_init_runs, p_Q):
        """
        Hyper parameter search for IE5 data, varying small number of parameters.
        One hidden layer.
        """
        
        # p_iter_num # How many iteration are needed to evaluate one set of hyper parameterss
        
        # task names:
        # Actuator, Ballbeam, Drive, Gas_furnace, Flutter, Dryer, Tank,
        # IdentificationExample1..5
        #import pdb; pdb.set_trace()
        out_train, in_train, task = prepare_data('IdentificationExample5', normalize=True)
        
        p_task_name = 'IE5_4'
        #p_iteration = 
        train_U = in_train.copy()
        train_Y = out_train.copy()
        p_init_runs = p_init_runs
        p_max_runs = 12000
        p_num_layers = 1
        p_hidden_dims = [1,1,]
        p_inference_method = 'svi'
        p_back_cstr = False
        p_MLP_Dims = None
        p_Q = p_Q
        p_win_in = task.win_in
        p_win_out = task.win_out
        p_init = 'Y'
        p_x_init_var = 0.05
        
        result = list()
        for i_no in range(0, p_iter_num): # iterations take into account model randomness e.g. initialization of inducing points
            res = rgp_experiment_raw(p_task_name, bo_iter_no*10+i_no, train_U, train_Y, p_init_runs, p_max_runs, p_num_layers, p_hidden_dims, 
                               p_inference_method, p_back_cstr, p_MLP_Dims, p_Q, p_win_in, p_win_out, p_init, p_x_init_var)
            result.append(res[0])
        #import pdb; pdb.set_trace()
        
        return np.array(((np.min(result),),))

class IE5_experiment_5( object ):
    """
    Back constrains + SVI inference
    Tested parameters are: number of initial runs, number of inducing points.
    
    """
    def __init__( self, initial_counter, iter_nums):
        self.counter = initial_counter
        self.iter_nums = iter_nums
        
    def __call__( self, *args, **kwargs ):
        #import pdb; pdb.set_trace()
        new_args = (self.counter,self.iter_nums,) + tuple( [ int(args[0][0,i]) for i in range(args[0].shape[1]) ] )
        ret = self.IE5_experiment_5_code( *new_args, **kwargs)
        self.counter += 1
        return ret
    
    @staticmethod
    def IE5_experiment_5_code( bo_iter_no, p_iter_num, p_init_runs, p_Q):
        """
        Hyper parameter search for IE5 data, varying small number of parameters.
        One hidden layer.
        """
        
        # p_iter_num # How many iteration are needed to evaluate one set of hyper parameterss
        
        # task names:
        # Actuator, Ballbeam, Drive, Gas_furnace, Flutter, Dryer, Tank,
        # IdentificationExample1..5
        #import pdb; pdb.set_trace()
        out_train, in_train, task = prepare_data('IdentificationExample5', normalize=True)
        
        p_task_name = 'IE5_5'
        #p_iteration = 
        train_U = in_train.copy()
        train_Y = out_train.copy()
        p_init_runs = p_init_runs
        p_max_runs = 12000
        p_num_layers = 1
        p_hidden_dims = [1,1,]
        p_inference_method = 'svi'
        p_back_cstr = True
        
        p_rnn_type='gru'
        p_rnn_hidden_dims=[20,]
        
        p_Q = p_Q
        p_win_in = task.win_in
        p_win_out = task.win_out
        
        result = list()
        for i_no in range(0, p_iter_num): # iterations take into account model randomness e.g. initialization of inducing points
            res = rgp_experiment_bcstr_raw(p_task_name, bo_iter_no*10+i_no, train_U, train_Y, p_init_runs, p_max_runs, p_num_layers, p_hidden_dims, 
                               p_inference_method, p_back_cstr, p_rnn_type, p_rnn_hidden_dims, p_Q, p_win_in, p_win_out)
            result.append(res[0])
        
        #import pdb; pdb.set_trace()
        
        return np.array(((np.min(result),),))

class IE5_experiment_6( object ):
    """
    Same as experiment 5, only the model has changes now it includes the RGP inputs at
    encoder inputs.
    
    Back constrains + SVI inference
    Tested parameters are: number of initial runs, number of inducing points.
    
    """
    def __init__( self, initial_counter, iter_nums):
        self.counter = initial_counter
        self.iter_nums = iter_nums
        
    def __call__( self, *args, **kwargs ):
        #import pdb; pdb.set_trace()
        new_args = (self.counter,self.iter_nums,) + tuple( [ int(args[0][0,i]) for i in range(args[0].shape[1]) ] )
        ret = self.IE5_experiment_6_code( *new_args, **kwargs)
        self.counter += 1
        return ret
    
    @staticmethod
    def IE5_experiment_6_code( bo_iter_no, p_iter_num, p_init_runs, p_Q):
        """
        Hyper parameter search for IE5 data, varying small number of parameters.
        One hidden layer.
        """
        
        # p_iter_num # How many iteration are needed to evaluate one set of hyper parameterss
        
        # task names:
        # Actuator, Ballbeam, Drive, Gas_furnace, Flutter, Dryer, Tank,
        # IdentificationExample1..5
        #import pdb; pdb.set_trace()
        out_train, in_train, task = prepare_data('IdentificationExample5', normalize=True)
        
        p_task_name = 'IE5_6'
        #p_iteration = 
        train_U = in_train.copy()
        train_Y = out_train.copy()
        p_init_runs = p_init_runs
        p_max_runs = 15000
        p_num_layers = 1
        p_hidden_dims = [1,1,]
        p_inference_method = 'svi'
        p_back_cstr = True
        
        p_rnn_type='gru'
        p_rnn_hidden_dims=[20,]
        
        p_Q = p_Q
        p_win_in = task.win_in
        p_win_out = task.win_out
        
        result = list()
        for i_no in range(0, p_iter_num): # iterations take into account model randomness e.g. initialization of inducing points
            res = rgp_experiment_bcstr_raw(p_task_name, bo_iter_no*10+i_no, train_U, train_Y, p_init_runs, p_max_runs, p_num_layers, p_hidden_dims, 
                               p_inference_method, p_back_cstr, p_rnn_type, p_rnn_hidden_dims, p_Q, p_win_in, p_win_out)
            result.append(res[0])
        
        #import pdb; pdb.set_trace()
        
        return np.array(((np.min(result),),))
    
def rgp_experiment_bcstr_raw(p_task_name, p_iteration, train_U, train_Y, p_init_runs, p_max_runs, p_num_layers, p_hidden_dims, 
                       p_inference_method, p_back_cstr, p_rnn_type, p_rnn_hidden_dims, p_Q, p_win_in, p_win_out):
    """
    Experiment file for NON MINIBATCH inference.
    So, DeepAutoreg is run here.
    
    Inputs:
    -------------------------------
        p_task_name: string
            Experiment name, used only in file name
        p_iteration: int or string
            Iteration of the experiment, used only in file name
    
        p_init_runs: int:
             Number of initial runs when likelihood variances and covariance magnitudes are fixed
        p_max_runs: int
            Maximum runs of general optimization
        p_num_layers: int [1,2]
            Number of RGP layers
        p_hidden_dims: list[ length is the number of hidden layers]
            Dimensions of hidden layers
        p_inference_method: string
            If 'svi' then SVI inference is used.
        p_back_cstr: bool
            Use back constrains or not.
        p_rnn_hidden_dims: int
            Hidden dimension of neural network.
        p_Q: int
            Number of inducing points
        p_win_in, p_win_out: int
            Inpput window and hidden layer window.
        
    """
    win_in = p_win_in # 20
    win_out = p_win_out # 20
    
    inference_method = p_inference_method if p_inference_method == 'svi' else None
    #import pdb; pdb.set_trace()
    
    if p_num_layers == 1:
        # 1 layer:
        wins = [0, win_out] # 0-th is output layer
        nDims = [train_Y.shape[1], p_hidden_dims[0]]
        
        kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True),
                                GPy.kern.RBF(win_in + win_out,ARD=True,inv_l=True)]
    elif p_num_layers == 2:
        # 2 layers:
        wins = [0, win_out, win_out]
        nDims = [train_Y.shape[1], p_hidden_dims[0], p_hidden_dims[1]]
        
        kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True),
                 GPy.kern.RBF(win_out+win_out,ARD=True,inv_l=True),
                 GPy.kern.RBF(win_out+win_in,ARD=True,inv_l=True)]
    else:
        raise NotImplemented()
    
    print("Input window:  ", win_in)
    print("Output window:  ", win_out)

    p_Q = 120 #!!!!! TODO:
    m = autoreg.DeepAutoreg_rnn(wins, train_Y, U=train_U, U_win=win_in,
                        num_inducing=p_Q, back_cstr=p_back_cstr, nDims=nDims,
                        rnn_type=p_rnn_type,
                        rnn_hidden_dims=p_rnn_hidden_dims,
                        minibatch_inference=False,
                        inference_method=inference_method, # Inference method
                        kernels=kernels)
    
    # pattern for model name: #task_name, inf_meth=?, wins=layers, Q = ?, backcstr=?,MLP_dims=?, nDims=
    model_file_name = '%s_%s--inf_meth=%s--backcstr=%s--wins=%s_%s--Q=%i--nDims=%s' % (p_task_name, str(p_iteration),
        'reg' if inference_method is None else inference_method, 
        str(p_back_cstr) if p_back_cstr==False else str(p_back_cstr) + '_' + p_rnn_type + str(p_rnn_hidden_dims[0]), 
        str(win_in), str(wins), p_Q, str(nDims))
    
    print('Model file name:  ',  model_file_name)
    print(m)
    
    import pdb; pdb.set_trace()
    #Initialization
    # Here layer numbers are different than in initialization. 0-th layer is the top one
    for i in range(m.nLayers):
        m.layers[i].kern.inv_l[:]  = np.mean( 1./((m.layers[i].X.mean.values.max(0)-m.layers[i].X.mean.values.min(0))/np.sqrt(2.)) )
        m.layers[i].likelihood.variance[:] = 0.01*train_Y.var()
        m.layers[i].kern.variance.fix(warning=False)
        m.layers[i].likelihood.fix(warning=False)
    print(m)
    
    #init_runs = 50 if out_train.shape[0]<1000 else 100
    print("Init runs:  ", p_init_runs)
    m.optimize('bfgs',messages=1,max_iters=p_init_runs)
    for i in range(m.nLayers):
        m.layers[i].kern.variance.constrain_positive(warning=False)
        m.layers[i].likelihood.constrain_positive(warning=False)
    m.optimize('bfgs',messages=1,max_iters=p_max_runs)
    
    io.savemat(model_file_name, {'params': m.param_array[:]} )
    print(m)
    
    return -float(m._log_marginal_likelihood), m


def rgp_experiment_raw(p_task_name, p_iteration, train_U, train_Y, p_init_runs, p_max_runs, p_num_layers, p_hidden_dims, 
                       p_inference_method, p_back_cstr, p_MLP_Dims, p_Q, p_win_in, p_win_out, p_init, p_x_init_var):
    """
    Experiment file for NON MINIBATCH inference.
    So, DeepAutoreg is run here.
    
    Inputs:
    -------------------------------
        p_task_name: string
            Experiment name, used only in file name
        p_iteration: int or string
            Iteration of the experiment, used only in file name
    
        p_init_runs: int:
             Number of initial runs when likelihood variances and covariance magnitudes are fixed
        p_max_runs: int
            Maximum runs of general optimization
        p_num_layers: int [1,2]
            Number of RGP layers
        p_hidden_dims: list[ length is the number of hidden layers]
            Dimensions of hidden layers
        p_inference_method: string
            If 'svi' then SVI inference is used.
        p_back_cstr: bool
            Use back constrains or not.
        p_MLP_Dims: list[length is the number of MLP hidden layers, ignoring input and output layers]
            Values are the number of neurons at each layer.
        p_Q: int
            Number of inducing points
        p_win_in, p_win_out: int
            Inpput window and hidden layer window.
        p_init: string 'Y', 'rand', 'zero'
            Initialization of RGP hidden layers
        p_x_init_var: float
            Initial variance for X, usually 0.05 for data close to normalized data.
    """
    win_in = p_win_in # 20
    win_out = p_win_out # 20
    
    inference_method = p_inference_method if p_inference_method == 'svi' else None
    #import pdb; pdb.set_trace()
    
    if p_num_layers == 1:
        # 1 layer:
        wins = [0, win_out] # 0-th is output layer
        nDims = [train_Y.shape[1], p_hidden_dims[0]]
        
        kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True),
                                GPy.kern.RBF(win_in + win_out,ARD=True,inv_l=True)]
    elif p_num_layers == 2:
        # 2 layers:
        wins = [0, win_out, win_out]
        nDims = [train_Y.shape[1], p_hidden_dims[0], p_hidden_dims[1]]
        
        kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True),
                 GPy.kern.RBF(win_out+win_out,ARD=True,inv_l=True),
                 GPy.kern.RBF(win_out+win_in,ARD=True,inv_l=True)]
    else:
        raise NotImplemented()
    
    print("Input window:  ", win_in)
    print("Output window:  ", win_out)

    m = autoreg.DeepAutoreg_new(wins, train_Y, U=train_U, U_win=win_in,
                        num_inducing=p_Q, back_cstr=p_back_cstr, MLP_dims=p_MLP_Dims, nDims=nDims,
                        init=p_init, # how to initialize hidden states means
                        X_variance=p_x_init_var, #0.05, # how to initialize hidden states variances
                        inference_method=inference_method, # Inference method
                        kernels=kernels)

    # pattern for model name: #task_name, inf_meth=?, wins=layers, Q = ?, backcstr=?,MLP_dims=?, nDims=
    model_file_name = '%s_%s--inf_meth=%s--backcstr=%s--wins=%s_%s--Q=%i--nDims=%s--init=%s--x_init=%s' % (p_task_name, str(p_iteration),
        'reg' if inference_method is None else inference_method, 
        str(p_back_cstr) if p_back_cstr==False else str(p_back_cstr) + '_' + str(p_MLP_Dims), 
        str(win_in), str(wins), p_Q, str(nDims), p_init, str(p_x_init_var))
    
    print('Model file name:  ',  model_file_name)
    print(m)
    
    #import pdb; pdb.set_trace()
    #Initialization
    # Here layer numbers are different than in initialization. 0-th layer is the top one
    for i in range(m.nLayers):
        m.layers[i].kern.inv_l[:]  = np.mean( 1./((m.layers[i].X.mean.values.max(0)-m.layers[i].X.mean.values.min(0))/np.sqrt(2.)) )
        m.layers[i].likelihood.variance[:] = 0.01*train_Y.var()
        m.layers[i].kern.variance.fix(warning=False)
        m.layers[i].likelihood.fix(warning=False)
    print(m)
    
    #init_runs = 50 if out_train.shape[0]<1000 else 100
    print("Init runs:  ", p_init_runs)
    m.optimize('bfgs',messages=1,max_iters=p_init_runs)
    for i in range(m.nLayers):
        m.layers[i].kern.variance.constrain_positive(warning=False)
        m.layers[i].likelihood.constrain_positive(warning=False)
    m.optimize('bfgs',messages=1,max_iters=p_max_runs)
    
    io.savemat(model_file_name, {'params': m.param_array[:]} )
    print(m)
    
    return -float(m._log_marginal_likelihood), m


def bo_run_1():
    """
    Run the bayesian optimization experiemnt 1.
    Tested parameters are: initial number of optimization runs, number of hidden dims, number of inducing points.
    
    After the first experiment the conclusion is that 1 hidden dim is the best, but also
    the optimization is not very explorative.
    Probably there was an error in the experi,ent setup because I did not change the number of hidden layers
    only the number of hidden dimensions in 1 layer.
    
    Best values: ini_runs = 160.0,     hidden dim=1., Q=50. (237.44060068)
    Iteration 21
    """
    
    import GPyOpt
    import pickle
    
    exper = IE5_experiment_1(0,4)
    
    domain =[{'name': 'init_runs', 'type': 'discrete', 'domain': np.arange(10,201,10) },
               {'name': 'hidden_dims', 'type': 'discrete', 'domain': (1,2,3,4)},
               {'name': 'Q', 'type': 'discrete', 'domain': np.arange(10,201,10) } ]
    
    
    Bopt = GPyOpt.methods.BayesianOptimization(f=exper,                   # function to optimize       
                                             domain=domain,        # box-constrains of the problem
                                             model_type = 'GP_MCMC', 
                                             initial_design_numdata = 2,# number data initial design
                                             acquisition_type='EI_MCMC',      # Expected Improvement
                                             exact_feval = False)
    #import pdb; pdb.set_trace()
    # --- Stop conditions
    max_time  = None
    max_iter  = 20
    tolerance = 1e-4     # distance between two consecutive observations  
    
    # Run the optimization             
    report_file = 'report_IE5_experiment_1(0,1)'
    evaluations_file = 'eval_IE5_experiment_1(0,1)'
    models_file = 'model_IE5_experiment_1(0,1)'
    Bopt.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance, verbosity=True,    
                          report_file = report_file, evaluations_file= evaluations_file, models_file=models_file) 

    #acquisition_type ='LCB',       # LCB acquisition
    #acquisition_weight = 0.1)
    
    ff = open('IE5_experiment_1(0,4)','w')
    pickle.dump(Bopt,ff)
    ff.close()

def bo_run_2():
    """
    Run the bayesian optimization experiemnt 2.
    
    Tested parameters are: initial number of optimization runs, number of inducing points.
    
    Conclusions after the experiment: The output file contains only variables as var_1, var_2 etc.
    but Xavier said that the order is presearved.
    
    The optimal values are: init_runs = 110, Q (ind. num) = 200. (run 3), 240.44817869
    Bu the results are still the same from run to run.
    Total running time was 40hours on GPU machine.
    Maybe we can reduce the number of intrinsic iterations per evaluation.
    
    Now idea is to use manually designad initial values to run more proper experiment. (Experiment 3)
    """
    
    import GPyOpt
    import pickle
    
    exper = IE5_experiment_2(0,3)
    
    domain =[{'name': 'init_runs', 'type': 'discrete', 'domain': np.arange(50,201,10) },
               #{'name': 'hidden_dims', 'type': 'discrete', 'domain': (1,2,3,4)},
               {'name': 'Q', 'type': 'discrete', 'domain': np.arange(40,201,10) } ]
    
    
    Bopt = GPyOpt.methods.BayesianOptimization(f=exper,                   # function to optimize       
                                             domain=domain,        # box-constrains of the problem
                                             model_type = 'GP_MCMC',
                                             initial_design_numdata = 3,# number data initial design
                                             acquisition_type='EI_MCMC',      # Expected Improvement
                                             exact_feval = False)
    #import pdb; pdb.set_trace()
    # --- Stop conditions
    max_time  = None
    max_iter  = 7
    tolerance = 1e-4     # distance between two consecutive observations  
    
    # Run the optimization             
    report_file = 'report_IE5_experiment_2(0,3)'
    evaluations_file = 'eval_IE5_experiment_2(0,3)'
    models_file = 'model_IE5_experiment_2(0,3)'
    Bopt.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance, verbosity=True,    
                          report_file = report_file, evaluations_file= evaluations_file, models_file=models_file, acquisition_par=3) # acquisition_par is
                          # used to make it more explorative. It seems it did not help.

    #acquisition_type ='LCB',       # LCB acquisition
    #acquisition_weight = 0.1)
    
    ff = open('IE5_experiment_2(0,3)','w')
    pickle.dump(Bopt,ff)
    ff.close()

def bo_run_3():
    """
    Run the bayesian optimization experiemnt 2.
    
    Tested parameters are: initial number of number of layers, number of inducing points.
    Also, I do initial evaluations manually.
    
    Best values: 1 layer, 40 inducing points, (run 7) 242.67636311
    This value is laregr than in the other experiments. Manybe becuase there were only 2 internal runs
    in every function evaluation.
    """
    
    import GPyOpt
    import pickle
    
    exper = IE5_experiment_3(0,2)
    
    domain =[  #{ 'name': 'init_runs', 'type': 'discrete', 'domain': np.arange(50,201,10) },
               #{'name': 'hidden_dims', 'type': 'discrete', 'domain': (1,2,3,4)},
               { 'name': 'layer_num', 'type': 'discrete', 'domain': (1,2) },
               {'name': 'Q', 'type': 'discrete', 'domain': np.arange(40,201,10) } ]
    
    #out = exper( np.array( (( 2.0,100.0),) ) ) # input_shape: (array([[   2.,  120.]]),) ### outputshape:  array([[ 413.67619157]])
    
    input1 = np.array( (( 1.0,50.0),) ); out1 = exper( input1 )
    input2 = np.array( (( 2.0,50.0),) ); out2 = exper( input2 )
    input3 = np.array( (( 1.0,100.0),) ); out3 = exper( input3 )
    input4 = np.array( (( 2.0,100.0),) ); out4 = exper( input4 )
    input5 = np.array( (( 1.0,200.0),) ); out5 = exper( input5 )
    input6 = np.array( (( 2.0,200.0),) ); out6 = exper( input6 )
    
#    init_input = np.vstack( (input1,input2,) )
#    init_out = np.vstack( (out1,out2,) )
    
    init_input = np.vstack( (input1,input2,input3,input4,input5,input6) )
    init_out = np.vstack( (out1,out2,out3,out4,out5,out6) )
    
    #import pdb; pdb.set_trace(); #return
    #exper()
    #import pdb; pdb.set_trace()
    Bopt = GPyOpt.methods.BayesianOptimization(f=exper,                   # function to optimize       
                                             domain=domain,        # box-constrains of the problem
                                             model_type = 'GP_MCMC',
                                             X=init_input,
                                             Y=init_out,
                                             #initial_design_numdata = 3,# number data initial design
                                             acquisition_type='EI_MCMC',      # Expected Improvement
                                             exact_feval = False)
    
    
    
    #import pdb; pdb.set_trace(); #return
    # --- Stop conditions
    max_time  = None
    max_iter  = 10
    tolerance = 1e-4     # distance between two consecutive observations  
    
    # Run the optimization             
    report_file = 'report_IE5_experiment_3(0,2)'
    evaluations_file = 'eval_IE5_experiment_3(0,2)'
    models_file = 'model_IE5_experiment_3(0,2)'
    Bopt.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance, verbosity=True,    
                          report_file = report_file, evaluations_file= evaluations_file, models_file=models_file, acquisition_par=3) # acquisition_par is
                          # used to make it more explorative. It seems it did not help.

    #acquisition_type ='LCB',       # LCB acquisition
    #acquisition_weight = 0.1)
    
    ff = open('IE5_experiment_3(0,2)','w')
    pickle.dump(Bopt,ff)
    ff.close()

def bo_run_4():
    """
    Run the bayesian optimization experiemnt 2.
    
    SVI inference
    Tested parameters are: number of initial runs, number of inducing points.
    
    First SVI experiment.
    """
    
    import GPyOpt
    import pickle
    
    exper = IE5_experiment_4(0,3)
    
    domain =[{'name': 'init_runs', 'type': 'discrete', 'domain': np.arange(50,501,50) },
               #{'name': 'hidden_dims', 'type': 'discrete', 'domain': (1,2,3,4)},
               {'name': 'Q', 'type': 'discrete', 'domain': np.arange(40,201,10) } ]
    
    
    Bopt = GPyOpt.methods.BayesianOptimization(f=exper,                   # function to optimize       
                                             domain=domain,        # box-constrains of the problem
                                             model_type = 'GP_MCMC', 
                                             initial_design_numdata = 5,# number data initial design
                                             acquisition_type='EI_MCMC',      # Expected Improvement
                                             exact_feval = False)
    #import pdb; pdb.set_trace()
    # --- Stop conditions
    max_time  = None
    max_iter  = 2
    tolerance = 1e-4     # distance between two consecutive observations  
    
    # Run the optimization             
    report_file = 'report_IE5_experiment_4(0,3)'
    evaluations_file = 'eval_IE5_experiment_4(0,3)'
    models_file = 'model_IE5_experiment_4(0,3)'
    Bopt.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance, verbosity=True,    
                          report_file = report_file, evaluations_file= evaluations_file, models_file=models_file, acquisition_par=3) # acquisition_par is
                          # used to make it more explorative. It seems it did not help.

    #acquisition_type ='LCB',       # LCB acquisition
    #acquisition_weight = 0.1)
    
    ff = open('IE5_experiment_4(0,3)','w')
    pickle.dump(Bopt,ff)
    ff.close()

def bo_run_5():
    """
    Run the bayesian optimization experiemnt 5.
    
    Back constrains + SVI inference
    Tested parameters are: number of initial runs, number of inducing points.
    
    The optimal value: 340.816199019	350.0(init_runs)	120.0(ind points), iteration 9 (8 in file names)
    
    
    """
    
    import GPyOpt
    import pickle
    
    exper = IE5_experiment_5(0,3)
    
    domain =[{'name': 'init_runs', 'type': 'discrete', 'domain': np.arange(50,501,50) },
               #{'name': 'hidden_dims', 'type': 'discrete', 'domain': (1,2,3,4)},
               {'name': 'Q', 'type': 'discrete', 'domain': np.arange(40,201,10) } ]
    
    
    Bopt = GPyOpt.methods.BayesianOptimization(f=exper,                   # function to optimize       
                                             domain=domain,        # box-constrains of the problem
                                             model_type = 'GP_MCMC', 
                                             initial_design_numdata = 5,# number data initial design
                                             acquisition_type='EI_MCMC',      # Expected Improvement
                                             exact_feval = False)
    #import pdb; pdb.set_trace()
    # --- Stop conditions
    max_time  = None
    max_iter  = 5
    tolerance = 1e-4     # distance between two consecutive observations  
    
    # Run the optimization             
    report_file = 'report_IE5_experiment_5(0,3)'
    evaluations_file = 'eval_IE5_experiment_5(0,3)'
    models_file = 'model_IE5_experiment_5(0,3)'
    Bopt.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance, verbosity=True,    
                          report_file = report_file, evaluations_file= evaluations_file, models_file=models_file, acquisition_par=3) # acquisition_par is
                          # used to make it more explorative. It seems it did not help.

    #acquisition_type ='LCB',       # LCB acquisition
    #acquisition_weight = 0.1)
    
    ff = open('IE5_experiment_5(0,3)','w')
    pickle.dump(Bopt,ff)
    ff.close()

def bo_run_6():
    """
    Run the bayesian optimization experiemnt 6.
    
    Same as experiment 5, only the model has changes now it includes the RGP inputs at
    encoder inputs.
    
    Back constrains + SVI inference
    Tested parameters are: number of initial runs, number of inducing points.
    
    The optimal values: 361.667338238	300.0(init_runs)	80.0(ind points), inter. 4 (3 in file name)
    """
    
    import GPyOpt
    import pickle
    
    exper = IE5_experiment_6(0,2)
    
    domain =[{'name': 'init_runs', 'type': 'discrete', 'domain': np.arange(50,501,50) },
               #{'name': 'hidden_dims', 'type': 'discrete', 'domain': (1,2,3,4)},
               {'name': 'Q', 'type': 'discrete', 'domain': np.arange(40,201,10) } ]
    
    
    Bopt = GPyOpt.methods.BayesianOptimization(f=exper,                   # function to optimize       
                                             domain=domain,        # box-constrains of the problem
                                             model_type = 'GP_MCMC', 
                                             initial_design_numdata = 5,# number data initial design
                                             acquisition_type='EI_MCMC',      # Expected Improvement
                                             exact_feval = False)
    #import pdb; pdb.set_trace()
    # --- Stop conditions
    max_time  = None
    max_iter  = 2
    tolerance = 1e-4     # distance between two consecutive observations  
    
    # Run the optimization             
    report_file = 'report_IE5_experiment_6(0,2)'
    evaluations_file = 'eval_IE5_experiment_6(0,2)'
    models_file = 'model_IE5_experiment_6(0,2)'
    Bopt.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance, verbosity=True,    
                          report_file = report_file, evaluations_file= evaluations_file, models_file=models_file, acquisition_par=3) # acquisition_par is
                          # used to make it more explorative. It seems it did not help.

    #acquisition_type ='LCB',       # LCB acquisition
    #acquisition_weight = 0.1)
    
    ff = open('IE5_experiment_6(0,2)','w')
    pickle.dump(Bopt,ff)
    ff.close()

def bo_run_7():
    """
    Run the bayesian optimization experiemnt 7.
    
    
    Same as experiment 6, but with ARD kernel, different tolerance
    and different max_iter.
    
    """
    
    import GPyOpt
    import pickle
    
    exper = IE5_experiment_6(0,2)
    
    domain =[{'name': 'init_runs', 'type': 'discrete', 'domain': np.arange(200,501,30) },
               #{'name': 'hidden_dims', 'type': 'discrete', 'domain': (1,2,3,4)},
               {'name': 'Q', 'type': 'discrete', 'domain': np.arange(60,201,10) } ]
    
    
    kernel = GPy.kern.RBF(len(domain),ARD=True)
    
    Bopt = GPyOpt.methods.BayesianOptimization(f=exper,                   # function to optimize       
                                             domain=domain,        # box-constrains of the problem
                                             model_type = 'GP_MCMC',
                                             kernel=kernel,
                                             initial_design_numdata = 3,# number data initial design
                                             acquisition_type='EI_MCMC',      # Expected Improvement
                                             exact_feval = False)
    #import pdb; pdb.set_trace()
    # --- Stop conditions
    max_time  = None
    max_iter  = 7
    tolerance = 1e-2     # distance between two consecutive observations  
    
    # Run the optimization             
    report_file = 'report_IE5_experiment_7(0,2)'
    evaluations_file = 'eval_IE5_experiment_7(0,2)'
    models_file = 'model_IE5_experiment_7(0,2)'
    Bopt.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance, verbosity=True,    
                          report_file = report_file, evaluations_file= evaluations_file, models_file=models_file, acquisition_par=3) # acquisition_par is
                          # used to make it more explorative. It seems it did not help.

    #acquisition_type ='LCB',       # LCB acquisition
    #acquisition_weight = 0.1)
    
    ff = open('IE5_experiment_7(0,2)','w')
    pickle.dump(Bopt,ff)
    ff.close()
    
if __name__ == '__main__':
    bo_run_5()
    
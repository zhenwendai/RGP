
# coding: utf-8

# In[1]:

from __future__ import print_function
import autoreg
import GPy
import numpy as np
from autoreg.data_streamers import TrivialDataStreamer, RandomPermutationDataStreamer

from matplotlib import pyplot as plt

#get_ipython().magic(u'matplotlib inline')

from autoreg.benchmark import tasks


# In[2]:


# Function to compute root mean square error:

def comp_RMSE(a,b):
    return np.sqrt(np.square(a-b).mean())


# In[3]:


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


# In[4]:

def get_data(task_name = 'IdentificationExample5'):

    # task_name = 'IdentificationExample5'
    # task names:
    # Actuator, Ballbeam, Drive, Gas_furnace, Flutter, Dryer, Tank,
    # IdentificationExample1..5
    
    
    # In[5]:
    
    task = getattr( tasks, task_name)
    task = task()
    task.load_data()
    print("Data OUT train shape:  ", task.data_out_train.shape)
    print("Data IN train shape:  ", task.data_in_train.shape)
    print("Data OUT test shape:  ", task.data_out_test.shape)
    print("Data IN test shape:  ", task.data_in_test.shape)
    
    # ### Normalize training and test data:
    
    # In[6]:
    
    normalize = False
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
    
    print("Training OUT mean:  ", out_train.mean(0)); 
    print("Training OUT std:  ", out_train.std(0))
    print("")
    print("Test OUT mean:  ", out_test.mean(0)); 
    print("Test OUT std:  ", out_test.std(0))
    print("")
    print("Training IN mean:  ", in_train.mean(0)); 
    print("Training IN std:  ", in_train.std(0))
    print("")
    print("Test IN mean:  ", in_test.mean(0)); 
    print("Test IN std:  ", in_test.std(0))
    
    
    # ### Plot training and test data:
    
    # In[7]:
    
    
    # Plot training:
    fig1 = plt.figure(1,figsize=(20,8))
    fig1.suptitle('Training data')
    ax1 = plt.subplot(1,2,1)
    ax1.plot(out_train)
    ax1.set_title('Data OUT training')
    
    ax2 = plt.subplot(1,2,2)
    ax2.plot(in_train)
    ax2.set_title('Data IN training')
    
    fig2 = plt.figure(2,figsize=(20,8))
    fig2.suptitle('Test data')
    ax3 = plt.subplot(1,2,1)
    ax3.plot(out_test)
    ax3.set_title('Data OUT test')
    
    ax4 = plt.subplot(1,2,2)
    ax4.plot(in_test)
    ax4.set_title('Data IN test')
    
    del ax1, ax2, ax3, ax4

    globals().update(locals());
    return

# ### Model definition:

# In[49]:

def svi_test_2():
    """
    The goal of this function is to compare the minibatch SVI with
    not minibatch SVI.
    """
    
    trainned_models_folder_name = "/Users/grigoral/work/code/RGP/examples/identif_trainded"
    
    Q = 3 # 200 # Inducing points num
    win_in = task.win_in # 20
    win_out = task.win_out # 20
    use_controls = True
    back_cstr = True
    inference_method = 'svi'
    minibatch_inference = True
    # 1 layer:
    wins = [0, win_out] # 0-th is output layer
    nDims = [out_train.shape[1],1]
    
    # 2 layers:
    # wins = [0, win_out, win_out]
    # nDims = [out_train.shape[1],1,1]
    
    MLP_dims = [3,2] # !!! 300, 200
    print("Input window:  ", win_in)
    print("Output window:  ", win_out)
    
    
    m = autoreg.DeepAutoreg_new(wins, out_train, U=in_train, U_win=win_in,
                            num_inducing=Q, back_cstr=back_cstr, MLP_dims=MLP_dims, nDims=nDims,
                            init='Y', # how to initialize hidden states means
                            X_variance=0.05, # how to initialize hidden states variances
                            inference_method=inference_method, # Inference method
                            minibatch_inference = minibatch_inference,
                            # 1 layer:
                            kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True),
                                     GPy.kern.RBF(win_in + win_out,ARD=True,inv_l=True)] )
    
                            # 2 layers:
                            #kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True),
                            #         GPy.kern.RBF(win_out+win_out,ARD=True,inv_l=True),
                            #         GPy.kern.RBF(win_out+win_in,ARD=True,inv_l=True)])
    
    data_streamer = TrivialDataStreamer( out_train, in_train)
    m.set_DataStreamer(data_streamer)
    
    m._trigger_params_changed()
    print(m)
    m._next_minibatch()
    m._trigger_params_changed()
    #m = autoreg.DeepAutoreg([0,win_out],out_train, U=in_train, U_win=win_in,X_variance=0.01,
    #                        num_inducing=50)
    
    # pattern for model name: #task_name, inf_meth=?, wins=layers, Q = ?, backcstr=?,MLP_dims=?, nDims=
    model_file_name = '%s--inf_meth=%s--wins=%s--Q=%i--backcstr=%i--nDims=%s' % (task.name, 
        'reg' if inference_method is None else inference_method, str(wins), Q, back_cstr, str(nDims))
    if back_cstr == True:
        model_file_name += '--MLP_dims=%s' % (MLP_dims,)
        
    print('Model file name:  ',  model_file_name)
    print(m)
    m.checkgrad(verbose=True)
    return
    
    # ### Model initialization:
    
    # In[36]:
    
    
    # Here layer numbers are different than in initialization. 0-th layer is the top one
    for i in range(m.nLayers):
        m.layers[i].kern.inv_l[:]  = np.mean( 1./((m.layers[i].X.mean.values.max(0)-m.layers[i].X.mean.values.min(0))/np.sqrt(2.)) )
        m.layers[i].likelihood.variance[:] = 0.01*out_train.var()
        m.layers[i].kern.variance.fix(warning=False)
        m.layers[i].likelihood.fix(warning=False)
    print(m)
    
    
    # In[37]:
    
    
    print(m.layer_1.kern.inv_l)
    print(m.layer_0.kern.inv_l)
    print( np.mean(1./((m.layer_1.X.mean.values.max(0)-m.layer_1.X.mean.values.min(0))/np.sqrt(2.))) )
    
    
    # In[38]:
    
    
    # Plot initialization of hidden layer:
    def plot_hidden_states(fig_no, layer, layer_start_point=None, layer_end_point=None,
                                  data_start_point=None, data_end_point=None):
        if layer_start_point is None: layer_start_point=0;
        if layer_end_point is None: layer_end_point = len(layer.mean)
        
        if data_start_point is None: data_start_point=0;
        if data_end_point is None: layer_end_point = len(out_train)
            
        data = out_train[data_start_point:data_end_point]
        layer_means = layer.mean[layer_start_point:layer_end_point]
        layer_vars = layer.variance[layer_start_point:layer_end_point]
    
        fig4 = plt.figure(fig_no,figsize=(10,8))
        ax1 = plt.subplot(1,1,1)
        fig4.suptitle('Hidden layer plotting')
        ax1.plot(out_train[data_start_point:data_end_point], label="Orig data Train_out", color = 'b')
        ax1.plot( layer_means, label = 'pred mean', color = 'r' )
        ax1.plot( layer_means +                     2*np.sqrt( layer_vars ), label = 'pred var', color='r', linestyle='--' )
        ax1.plot( layer_means -                     2*np.sqrt( layer_vars ), label = 'pred var', color='r', linestyle='--' )
        ax1.legend(loc=4)        
        ax1.set_title('Hidden layer vs Training data')
    
        del ax1
    
    plot_hidden_states(5,m.layer_1.qX_0)
    #plot_hidden_states(6,m.layer_2.qX_0)
    
    
    # ### Model training:
    
    # In[39]:
    
    
    #init_runs = 50 if out_train.shape[0]<1000 else 100
    init_runs = 100
    print("Init runs:  ", init_runs)
    m.optimize('bfgs',messages=1,max_iters=init_runs)
    for i in range(m.nLayers):
        m.layers[i].kern.variance.constrain_positive(warning=False)
        m.layers[i].likelihood.constrain_positive(warning=False)
    m.optimize('bfgs',messages=1,max_iters=10000)
    
    print(m)
    
    
    # ### Look at trained parameters
    
    # In[40]:
    
    
    if hasattr(m, 'layer_1'):
        print("Layer 1:  ")
        print("States means (min and max), shapes:  ", m.layer_1.qX_0.mean.min(), 
              m.layer_1.qX_0.mean.max(), m.layer_1.qX_0.mean.shape)
        print("States variances (min and max), shapes:  ", m.layer_1.qX_0.variance.min(), 
              m.layer_1.qX_0.variance.max(), m.layer_1.qX_0.mean.shape)
        print("Inverse langthscales (min and max), shapes:  ", m.layer_1.rbf.inv_lengthscale.min(),
              m.layer_1.rbf.inv_lengthscale.max(), m.layer_1.rbf.inv_lengthscale.shape )
        
    if hasattr(m, 'layer_0'):
        print("")
        print("Layer 0 (output):  ")
        print("Inverse langthscales (min and max), shapes:  ", m.layer_0.rbf.inv_lengthscale.min(),
              m.layer_0.rbf.inv_lengthscale.max(), m.layer_0.rbf.inv_lengthscale.shape )
    
    
    # In[41]:
    
    
    print(m.layer_0.rbf.inv_lengthscale)
    
    
    # In[42]:
    
    
    print(m.layer_1.rbf.inv_lengthscale)
    
    
    # ### Analyze and plot model on test data:
    
    # In[43]:
    
    
    # Free-run on the train data
    
    # initialize to last part of trained latent states
    #init_Xs = [None, m.layer_1.qX_0[0:win_out]] # init_Xs for train prediction
    
    # initialize to zeros
    init_Xs = None
    predictions_train = m.freerun(init_Xs = init_Xs, U=in_train, m_match=True)
    
    # initialize to last part of trainig latent states
    #init_Xs = [None, m.layer_1.qX_0[-win_out:] ] # init_Xs for test prediction
    #U_test = np.vstack( (in_train[-win_in:], in_test) )
    
    # initialize to zeros
    init_Xs = None
    U_test = in_test
    
    # Free-run on the test data
    predictions_test = m.freerun(init_Xs = init_Xs, U=U_test, m_match=True)
    del init_Xs, U_test
    
    
    # In[44]:
    
    
    # Plot predictions
    def plot_predictions(fig_no,posterior_train, posterior_test=None, layer_no = None):
        """
        Plots the output data along with posterior of the layer.
        Used for plotting the hidden states or
        
        layer_no: int or Normal posterior
            plot states of this layer (0-th is output). There is also some logic about compting
            the MSE, and aligning with actual data.
        """
        
        if layer_no is None: #default
            layer_no = 1
    
        if posterior_test is None:
            no_test_data = True
        else:
            no_test_data = False
    
        if isinstance(posterior_train, list): 
            layer_in_list = len(predictions_train)-1-layer_no # standard layer no (like in printing the model)
            predictions_train_layer = predictions_train[layer_in_list]
        else:
            predictions_train_layer = posterior_train
    
        if not no_test_data:
            if isinstance(posterior_test, list): 
                predictions_test_layer = predictions_test[layer_in_list]
            else:
                predictions_test_layer = posterior_test
    
        # Aligning the data ->
        # training of test data can be longer than leyer data because of the initial window.
        if out_train.shape[0] > predictions_train_layer.mean.shape[0]:
            out_train_tmp = out_train[win_out:]
        else:
            out_train_tmp = out_train
            
        if out_test.shape[0] > predictions_test_layer.mean.shape[0]:
            out_test_tmp = out_test[win_out:]
        else:
            out_test_tmp = out_test
        # Aligning the data <-
            
        if layer_no == 0:
            # Not anymore! Compute RMSE ignoring first output values of length "win_out"
            train_rmse = [comp_RMSE(predictions_train_layer.mean,
                                    out_train_tmp)]
            print("Train overall RMSE: ", str(train_rmse))
            
            if not no_test_data:
                # Compute RMSE ignoring first output values of length "win_out"
                test_rmse = [comp_RMSE(predictions_test_layer.mean,
                                       out_test_tmp)]
                print("Test overall RMSE: ", str(test_rmse))
        
        # Plot predictions:
        if not no_test_data:
            fig5 = plt.figure(10,figsize=(20,8))
        else:
            fig5 = plt.figure(10,figsize=(10,8))
            
        fig5.suptitle('Predictions on Training and Test data')
        if not no_test_data:
            ax1 = plt.subplot(1,2,1)
        else:
            ax1 = plt.subplot(1,1,1)
        ax1.plot(out_train_tmp, label="Train_out", color = 'b')
        ax1.plot( predictions_train_layer.mean, label = 'pred mean', color = 'r' )
        ax1.plot( predictions_train_layer.mean +                     2*np.sqrt( predictions_train_layer.variance ), label = 'pred var', color='r', linestyle='--' )
        ax1.plot( predictions_train_layer.mean -                     2*np.sqrt( predictions_train_layer.variance ), label = 'pred var', color='r', linestyle='--' )
        ax1.legend(loc=4)        
        ax1.set_title('Predictions on Train')
    
        if not no_test_data:
            ax2 = plt.subplot(1,2,2)
            ax2.plot(out_test_tmp, label="Test_out", color = 'b')
    
            ax2.plot( predictions_test_layer.mean, label = 'pred mean', color = 'r' )
            #ax2.plot( predictions_test_layer.mean +\
            #                 2*np.sqrt( predictions_test_layer.variance ), label = 'pred var', color='r', linestyle='--' )
            #ax2.plot( predictions_test_layer.mean -\
            #                 2*np.sqrt( predictions_test_layer.variance ), label = 'pred var', color='r', linestyle='--' )
            ax2.legend(loc=4)        
            ax2.set_title('Predictions on Test')
    
            del ax2
        del ax1 
    plot_predictions(7,predictions_train, predictions_test , layer_no = 0)
    
    # In[47]:
    
    
    #plot_hidden_states(8,m.layer_1.qX_0)
    #plot_hidden_states(9,m.layer_2.qX_0)
    
    
    # In[ ]:
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

import copy
def svi_test_3():
    """
    This class tests that the model with minibatch turned on and with 
    one initial values for all samples and one latent space variance for all the samples.
    
    Gradients are not compared but tested separately
    """
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
                            mb_inf_init_xs_means='one',
                            mb_inf_init_xs_vars='one',
                            # 1 layer:
                            # kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True),
                             #        GPy.kern.RBF(win_in + win_out,ARD=True,inv_l=True)] )
    
                            # 2 layers:
                            kernels=[GPy.kern.RBF(win_out*nDims[1],ARD=True,inv_l=True),
                                     GPy.kern.RBF(win_out*nDims[1] + win_out*nDims[2],ARD=True,inv_l=True),
                                     GPy.kern.RBF(win_out*nDims[2] + win_in*u_dim,ARD=True,inv_l=True)])
                            
    model_1 = m
    model_1._trigger_params_changed()
    
    mll_1 = model_1._log_marginal_likelihood
    g_mll_1 = model_1._log_likelihood_gradients
    
    model_1.checkgrad(verbose=True)
    #self.assertTrue(self.model_1.checkgrad())
    
    data_streamer = RandomPermutationDataStreamer(Y, U)
    model_1.set_DataStreamer(data_streamer)
    model_1._trigger_params_changed()
    
    model_1._next_minibatch()
    model_1._trigger_params_changed()
    
    np.testing.assert_equal( model_1._log_marginal_likelihood, mll_1, err_msg="Likelihoods must be equal" )
    np.testing.assert_array_equal( model_1._log_likelihood_gradients, g_mll_1, err_msg="Likelihood gradients must be equal" )
    
    model_1._next_minibatch()
    model_1._trigger_params_changed()
    
    np.testing.assert_equal( model_1._log_marginal_likelihood, mll_1, err_msg="Likelihoods must be equal" )
    np.testing.assert_array_equal( model_1._log_likelihood_gradients, g_mll_1, err_msg="Likelihood gradients must be equal" )


def svi_test_4():
    """
    This class tests that the model with minibatch turned on and with 
    a separate initial value for every samples and one latent space variance for every sample.
    
    Gradients are not compared but tested separately
    """
    u_dim = 2
    y_dim = 3
    ts_length = 20
    sequences_no = 3
    U, Y = generate_data( sequences_no, ts_length, u_dim = u_dim, y_dim = y_dim)
    
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
    
    data_streamer = TrivialDataStreamer(Y, U)
    minibatch_index, minibatch_indices, Y_mb, X_mb = data_streamer.next_minibatch()
    
    m = autoreg.DeepAutoreg_new(wins, Y_mb, U=X_mb, U_win=win_in,
                            num_inducing=Q, back_cstr=back_cstr, MLP_dims=MLP_dims, nDims=nDims,
                            init='Y', # how to initialize hidden states means
                            X_variance=0.05, # how to initialize hidden states variances
                            inference_method=inference_method, # Inference method
                            minibatch_inference = minibatch_inference,
                            mb_inf_tot_data_size = sequences_no,
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
                            
    model_1 = m
    model_1._trigger_params_changed()
    
    mll_1 = model_1._log_marginal_likelihood
    g_mll_1 = model_1._log_likelihood_gradients
    
    #self.assertTrue(self.model_1.checkgrad())
    #model_1.checkgrad(verbose=True)
    #return
    data_streamer = RandomPermutationDataStreamer(Y, U)
    #data_streamer = TrivialDataStreamer(Y, U)
    
    model_1.set_DataStreamer(data_streamer)
    model_1._trigger_params_changed()
    
    #model_1.checkgrad(verbose=True)
    
    model_1._next_minibatch()
    model_1._trigger_params_changed()
    
    model_1.checkgrad(verbose=True)

    np.testing.assert_equal( model_1._log_marginal_likelihood, mll_1, err_msg="Likelihoods must be equal" )
    np.testing.assert_array_equal( model_1._log_likelihood_gradients, g_mll_1, err_msg="Likelihood gradients must be equal" )
    
    model_1._next_minibatch()
    model_1._trigger_params_changed()
    
    np.testing.assert_equal( model_1._log_marginal_likelihood, mll_1, err_msg="Likelihoods must be equal" )
    np.testing.assert_array_equal( model_1._log_likelihood_gradients, g_mll_1, err_msg="Likelihood gradients must be equal" )




def svi_test_5():
    """
    This class tests the initial mlp implemetration
    """
    
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
                            mb_inf_init_xs_vals='mlp',
                            # 1 layer:
                            # kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True),
                             #        GPy.kern.RBF(win_in + win_out,ARD=True,inv_l=True)] )
    
                            # 2 layers:
                            kernels=[GPy.kern.RBF(win_out*nDims[1],ARD=True,inv_l=True),
                                     GPy.kern.RBF(win_out*nDims[1] + win_out*nDims[2],ARD=True,inv_l=True),
                                     GPy.kern.RBF(win_out*nDims[2] + win_in*u_dim,ARD=True,inv_l=True)])
    
    model_1 = m
    model_1._trigger_params_changed()
    
    mll_1 = model_1._log_marginal_likelihood
    g_mll_1 = model_1._log_likelihood_gradientsss
    
    return
    
    data_streamer = RandomPermutationDataStreamer(Y, U)
    model_1.set_DataStreamer(data_streamer)
    model_1._trigger_params_changed()
    
    model_1._next_minibatch()
    model_1._trigger_params_changed()
    
    np.testing.assert_equal( model_1._log_marginal_likelihood, mll_1, err_msg="Likelihoods must be equal" )
    np.testing.assert_array_equal( model_1._log_likelihood_gradients, g_mll_1, err_msg="Likelihood gradients must be equal" )
    
    model_1._next_minibatch()
    model_1._trigger_params_changed()
    
    np.testing.assert_equal( model_1._log_marginal_likelihood, mll_1, err_msg="Likelihoods must be equal" )
    np.testing.assert_array_equal( model_1._log_likelihood_gradients, g_mll_1, err_msg="Likelihood gradients must be equal" )


if __name__ == '__main__':
    
    task_name = 'IdentificationExample5'
    # task names:
    # Actuator, Ballbeam, Drive, Gas_furnace, Flutter, Dryer, Tank,
    # IdentificationExample1..5


    #get_data(task_name = task_name)
    #svi_test_2()
    
    #svi_test_3()
    svi_test_4()
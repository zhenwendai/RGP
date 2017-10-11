
# coding: utf-8

# # Regression Example on Mocap data

# In[1]:

from __future__ import print_function

import autoreg
import GPy
import numpy as np
#from pylab import *
import tables
#from IPython.display import HTML
#get_ipython().magic(u'matplotlib inline')
from matplotlib import pyplot as plt
import os
#import pdb;pdb.set_trace()

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
# In[2]:


def comp_RMSE(a,b):
    return np.sqrt(np.square(a-b).mean())


# In[3]:

def gen_frames(data, data_mean, data_std, skel, imgpath):
    import os
    import GPy
    a = np.zeros((62,))
    fig = plt.figure(figsize=(8,10))
    ax = fig.add_subplot(111, projection='3d',aspect='equal')
    ax.view_init(elev=20., azim=65)
    fig.tight_layout()
    a[3:] = (data[0])*data_std+data_mean
    p = GPy.plotting.matplot_dep.visualize.skeleton_show(a, skel ,axes=ax)
    for i in xrange(data.shape[0]):
        a[3:] = (data[i])*data_std+data_mean
        p.modify(a)
        fig.savefig(os.path.join(imgpath,'%05d'%i+'.png'))


# In[4]:

def cmu_mocap_xyz(subject, train_motions, test_motions=[], sample_every=4, data_set='cmu_mocap'):
    """Load a given subject's training and test motions from the CMU motion capture data."""
    import GPy.util.datasets as dts
    
    
    # Load in subject skeleton.
    subject_dir = os.path.join(dts.data_path, data_set)

    # Make sure the data is downloaded.
    all_motions = train_motions + test_motions
    resource = dts.cmu_urls_files(([subject], [all_motions]))
    dts.data_resources[data_set] = dts.data_resources['cmu_mocap_full'].copy()
    dts.data_resources[data_set]['files'] = resource['files']
    dts.data_resources[data_set]['urls'] = resource['urls']
    if resource['urls']:
        dts.download_data(data_set)

    skel = GPy.util.mocap.acclaim_skeleton(os.path.join(subject_dir, subject + '.asf'))
    
    
    for i in range(len(skel.vertices)):
        print(i, skel.vertices[i].name)     

    
    # 0 root
    # 1 lhipjoint
    # 2 lfemur
    # 3 ltibia
    # 4 lfoot
    # 5 ltoes
    # 6 rhipjoint
    # 7 rfemur
    # 8 rtibia
    # 9 rfoot
    # 10 rtoes
    # 11 lowerback
    # 12 upperback
    # 13 thorax
    # 14 lowerneck
    # 15 upperneck
    # 16 head
    # 17 lclavicle
    # 18 lhumerus
    # 19 lradius
    # 20 lwrist
    # 21 lhand
    # 22 lfingers
    # 23 lthumb
    # 24 rclavicle
    # 25 rhumerus
    # 26 rradius
    # 27 rwrist
    # 28 rhand
    # 29 rfingers
    # 30 rthumb

    
    
    
    
    # Set up labels for each sequence
    exlbls = np.eye(len(train_motions))

    # Load sequences
    tot_length = 0
    temp_Y = []
    temp_Yxyz = []
    temp_lbls = []
    #u_inds = [15,16,17]
    #root_inds = [0,1,2]
    u_inds=[17]
    root_inds = [2]
    
    for i in range(len(train_motions)):
        temp_chan = skel.load_channels(os.path.join(subject_dir, subject + '_' + train_motions[i] + '.amc'))
        #temp_xyz_chan = skel.to_xyz(temp_chan.flatten()) ## A
        # Apparently the above is equiv. to giving temp_chan[0,:]. It's returning a 31 x 3 matrix.
        # I need to do this for every temp_chan[j,:], and serialize the result. The toe should be the
        # very last dimension (I think).
        #temp_xyz_chan = np.zeros((temp_chan.shape[0],93))
        #A -------
        temp_xyz_chan = np.zeros((temp_chan.shape[0],len(u_inds)))                        
        for j in range(temp_xyz_chan.shape[0]):                                 
            foo = skel.to_xyz(temp_chan[j,:]).flatten()
            temp_xyz_chan[j,:] = foo[u_inds] - foo[root_inds]
        #----A
        temp_Y.append(temp_chan[::sample_every, :])
        temp_Yxyz.append(temp_xyz_chan[::sample_every, :]) ## A
        temp_lbls.append(np.tile(exlbls[i, :], (temp_Y[i].shape[0], 1)))
        tot_length += temp_Y[i].shape[0]

    Y = np.zeros((tot_length, temp_Y[0].shape[1]))
    Yxyz = np.zeros((tot_length, temp_Yxyz[0].shape[1])) #A
    lbls = np.zeros((tot_length, temp_lbls[0].shape[1]))

    #pb.plot(temp_Yxyz[-1][:,15:18]-temp_Yxyz[-1][:,0:3],'x-')

    end_ind = 0
    for i in range(len(temp_Y)):
        start_ind = end_ind
        end_ind += temp_Y[i].shape[0]
        Y[start_ind:end_ind, :] = temp_Y[i]
        Yxyz[start_ind:end_ind, :] = temp_Yxyz[i] #A
        lbls[start_ind:end_ind, :] = temp_lbls[i]
    if len(test_motions) > 0:
        temp_Ytest = []
        temp_lblstest = []
        temp_Yxyz_test = []

        testexlbls = np.eye(len(test_motions))
        tot_test_length = 0
        for i in range(len(test_motions)):
            temp_chan = skel.load_channels(os.path.join(subject_dir, subject + '_' + test_motions[i] + '.amc'))
            #A -------
            temp_xyz_chan = np.zeros((temp_chan.shape[0],len(u_inds)))                        
            for j in range(temp_xyz_chan.shape[0]):                                 
                foo = skel.to_xyz(temp_chan[j,:]).flatten()
                temp_xyz_chan[j,:] = foo[u_inds] - foo[root_inds]
            #----A
            temp_Ytest.append(temp_chan[::sample_every, :])
            temp_Yxyz_test.append(temp_xyz_chan[::sample_every, :]) ## A
            temp_lblstest.append(np.tile(testexlbls[i, :], (temp_Ytest[i].shape[0], 1)))
            tot_test_length += temp_Ytest[i].shape[0]

        # Load test data
        Ytest = np.zeros((tot_test_length, temp_Ytest[0].shape[1]))
        Yxyz_test = np.zeros((tot_test_length, temp_Yxyz_test[0].shape[1])) #A
        lblstest = np.zeros((tot_test_length, temp_lblstest[0].shape[1]))

        end_ind = 0
        for i in range(len(temp_Ytest)):
            start_ind = end_ind
            end_ind += temp_Ytest[i].shape[0]
            Ytest[start_ind:end_ind, :] = temp_Ytest[i]
            Yxyz_test[start_ind:end_ind, :] = temp_Yxyz_test[i] #A
            lblstest[start_ind:end_ind, :] = temp_lblstest[i]
    else:
        Ytest = None
        lblstest = None

    info = 'Subject: ' + subject + '. Training motions: '
    for motion in train_motions:
        info += motion + ', '
    info = info[:-2]
    if len(test_motions) > 0:
        info += '. Test motions: '
        for motion in test_motions:
            info += motion + ', '
        info = info[:-2] + '.'
    else:
        info += '.'
    if sample_every != 1:
        info += ' Data is sub-sampled to every ' + str(sample_every) + ' frames.'
    return dts.data_details_return({'Y': Y, 'lbls' : lbls, 'Ytest': Ytest, 'lblstest' : lblstest, 'info': info, 'skel': skel,'Yxyz':Yxyz,'Yxyz_test':Yxyz_test,'u_inds':u_inds,'root_inds':root_inds,'Yxyz_list':temp_Yxyz,'Yxyz_list_test':temp_Yxyz_test}, data_set)




# ## Load the dataset

# In[5]:


def load_data():
    from GPy.util.datasets import cmu_mocap
    train_motions = ['01', '02', '03', '04', # walking
                '17', '18', '19', '20'] # running
    test_motions = ['05','06','07','08','21','22','23','24']
    data = cmu_mocap('35', train_motions, test_motions, sample_every=4, data_set='cmu_mocap')
    return data

def load_data_xyz():
    #train_motions = ['01', '02', '03', '04', # walking
    #            '17', '18', '19', '20'] # running
    #test_motions = ['05','06','07','08','09','10','11','12','13','14','15','16','21','22','23','24','25','26']
    train_motions = ['01', '02', '03', '04'] # walking
    test_motions = ['05','06','07','08']
    data = cmu_mocap_xyz('35', train_motions, test_motions, sample_every=4, data_set='cmu_mocap')
    return data


# In[6]:

def experiment1(debug = False, train_model=False, model = 1, input_scaling_factor=1):
    """
    Here the first experiment with training RGP on mocap data is performed.
    There is a corresponding ipynb file.
    """
    experiment_path = '/Users/grigoral/work/code/RGP/examples'
    
    #data = load_data()
    data = load_data_xyz()
    
    
    # In[7]:
    
    y = data['Y']
    u = data['Yxyz_list']
    u_flat = np.vstack(u)
    
    lbls = data['lbls']
    data_out_train = y
    
    # Ask: why first 3 dimensions are removed? # 44 and 56 -output variable is 0.
    data_out_train = y[:,3:]
    data_out_mean  = data_out_train.mean(axis=0)
    data_out_std   = data_out_train.std(axis=0)
    data_out_train = (y[:,3:]-data_out_mean)/data_out_std
    #data_out_train_list = [data_out_train[np.where(lbls[:,i]==1)[0]][1:] for i in range(lbls.shape[1])]
    data_out_train_list = [data_out_train[np.where(lbls[:,i]==1)[0]] for i in range(lbls.shape[1])]
    
                        
    # Create controls
    #data_in_train_list = [y[np.where(lbls[:,i]==1)[0]][:,2][1:] - y[np.where(lbls[:,i]==1)[0]][:,2][:-1] for i in range(lbls.shape[1])]
    #from scipy.ndimage.filters import gaussian_filter1d
    #data_in_train_list = [np.ones(d.shape+(1,))*d.mean() for d in data_in_train_list]
    
    ##data_in_train_list = [gaussian_filter1d(d,8.)[:,None] for d in data_in_train_list]
    ##data_in_train_list = [np.vstack([d[:10],d]) for d in data_in_train_list]
    
    data_in_train_list = u
    u_flat_mean = u_flat.mean(axis=0)
    u_flat_std = u_flat.std(axis=0)
    data_in_train = (u_flat-u_flat_mean)/u_flat_std
        
    #data_in_train_list = u
    data_in_train_list = [(d-u_flat_mean)/u_flat_std for d in data_in_train_list]
    
    # In[8]:
    
    
#    print data_in_train_list[0].shape
#    print data_out_train_list[0].shape
#    
#    for i in range(len(data_in_train_list)):
#        plt.figure()
#        plt.plot(data_in_train_list[i], 'x-')
#        plt.title(i)
#        print data_in_train_list[i].shape[0]
    
    
    # In[9]:
    
    
    print(y.shape)
    print(data_out_train.shape)
    print(u_flat.shape)
    print(data_in_train.shape)
    
    
    # In[10]:
    
    if debug: import pdb; pdb.set_trace()
    ytest = data['Ytest']
    lblstest = data['lblstest']
    u = data['Yxyz_list_test']
    
    #data_out_test = ytest
    data_out_test= ytest[:,3:]
    
    data_out_test = (ytest[:,3:]-data_out_mean)/data_out_std
    
    #data_out_test_list = [data_out_test[np.where(lblstest[:,i]==1)[0]][1:] for i in range(lblstest.shape[1])]
    data_out_test_list = [data_out_test[np.where(lblstest[:,i]==1)[0]] for i in range(lblstest.shape[1])]
    
    # Create controls
    #data_in_test_list = [ytest[np.where(lblstest[:,i]==1)[0]][:,2][1:] - ytest[np.where(lblstest[:,i]==1)[0]][:,2][:-1] for i in range(lblstest.shape[1])]
    #data_in_test_list = [np.ones(d.shape+(1,))*d.mean() for d in data_in_test_list]
    
    #data_in_test_list = u
    
    data_in_test_list = u
    #data_in_test = (u_flat-u_flat_mean)/u_flat_std
    data_in_test_list = [(d-u_flat_mean)/u_flat_std for d in u]
    
    # ## Fit a model without NN-constraint
    
    # In[11]:
    
    
    # Down-scaling the input signals
    #data_in_train_list = [d*0.1 for d in data_in_train_list]
    #data_in_test_list = [d*0.1 for d in data_in_test_list]
    #data_in_train = data_in_train*0.1
    
    # In[13]:
    
    if debug: import pdb; pdb.set_trace()
    #=============================
    # Initialize a model
    #=============================
    
    Q = 100 # 200
    win_in = 20 # 20
    win_out = 20 # 20
    use_controls = True
    back_cstr = False
    
    if input_scaling_factor is None:
        input_scaling_factor = 1
        
    if model == 1:
    # create the model
        if use_controls:
            #m = autoreg.DeepAutoreg([0, win_out], data_out_train, U=data_in_train, U_win=win_in, X_variance=0.05,
            #                    num_inducing=Q, back_cstr=back_cstr, MLP_dims=[300,200], nDims=[data_out_train.shape[1],1],
            #                     kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True, useGPU=True),
            #                     GPy.kern.RBF(win_out+win_in,ARD=True,inv_l=True, useGPU=True)])
            
            # Model without lists
    #        m = autoreg.DeepAutoreg([0, win_out, win_out], data_out_train, U=data_in_train, U_win=win_in, X_variance=0.05,
    #                            num_inducing=Q, back_cstr=back_cstr, MLP_dims=[300,200], nDims=[data_out_train.shape[1],1,1],
    #                             kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True, useGPU=False),
    #                             GPy.kern.RBF(win_out+win_out,ARD=True,inv_l=True, useGPU=False),
    #                             GPy.kern.RBF(win_out+win_in,ARD=True,inv_l=True, useGPU=False)])
            
            # Model with lists
            m = autoreg.DeepAutoreg([0, win_out, win_out], data_out_train_list, U=[d*input_scaling_factor for d in data_in_train_list], U_win=win_in, X_variance=0.05,
                                num_inducing=Q, back_cstr=back_cstr, MLP_dims=[300,200], nDims=[data_out_train.shape[1],1,1],
                                 kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True, useGPU=False),
                                 GPy.kern.RBF(win_out+win_out,ARD=True,inv_l=True, useGPU=False),
                                 GPy.kern.RBF(win_out+win_in,ARD=True,inv_l=True, useGPU=False)])
            
            
            # used with back_cstr=True in the end of the notebook
    #        m = autoreg.DeepAutoreg([0, win_out], data_out_train_list, U=[d*0.1 for d in data_in_train_list], U_win=win_in, X_variance=0.05,
    #                        num_inducing=Q, back_cstr=back_cstr, MLP_dims=[500,200], nDims=[data_out_train.shape[1],1],
    #                         kernels=[GPy.kern.MLP(win_out,bias_variance=10.),
    #                         GPy.kern.MLP(win_out+win_in,bias_variance=10.)])
        else:
            m = autoreg.DeepAutoreg([0, win_out], data_in_train, U=None, U_win=win_in, X_variance=0.05,
                                num_inducing=Q, back_cstr=back_cstr, MLP_dims=[200,100], nDims=[data_out_train.shape[1],1],
                                 kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True, useGPU=False),
                                 GPy.kern.RBF(win_out,ARD=True,inv_l=True, useGPU=False)])
    
    elif model == 2:
        # Ask: no b tern in NLP regularization.
        #=============================
        # Model with NN-constraint
        #=============================
        Q = 500
        win_in = 20
        win_out = 20
        
        use_controls = True
        back_cstr = True
        
        m = autoreg.DeepAutoreg([0, win_out], data_out_train_list, U=[d*input_scaling_factor for d in data_in_train_list], U_win=win_in, X_variance=0.05,
                            num_inducing=Q, back_cstr=back_cstr, MLP_dims=[500,200], nDims=[data_out_train.shape[1],1],
                             kernels=[GPy.kern.MLP(win_out,bias_variance=10.),
                             GPy.kern.MLP(win_out+win_in,bias_variance=10.)])
        #                      kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True, useGPU=True),
        #                      GPy.kern.RBF(win_out+win_in,ARD=True,inv_l=True, useGPU=True)])
    # In[14]:
    
    
    #=============================
    # Load a trained model
    #=============================
    
#    import tables
#    with tables.open_file('./walk_run_2.h5','r') as f:
#        ps = f.root.param_array[:]
#        f.close()
#        m.param_array[:] = ps
#        m._trigger_params_changed()
    
    
    # In[68]:
    
    # Initialize latent variables and inducing points
    
    #globals().update(locals()); return # Alex
    if debug: import pdb; pdb.set_trace()
    if not back_cstr:
        pp = GPy.util.pca.PCA(data_out_train)
        pca_projection = pp.project(data_out_train, 1)
        pca_projection = (pca_projection - pca_projection.mean()) / pca_projection.std()
        # Alex ->
        # m.layer_1.Xs_flat[0].mean[:] = pca_projection 
        # m.layer_2.Xs_flat[0].mean[:] = pca_projection
        
        for i_seq in range(lbls.shape[1]):
            m.layer_1.Xs_flat[i_seq].mean[:] = pca_projection[ np.where(lbls[:,i_seq]==1)[0],:]
            m.layer_2.Xs_flat[i_seq].mean[:] = pca_projection[ np.where(lbls[:,i_seq]==1)[0],:]
            
        # Alex <-
        m._trigger_params_changed()
        
        # Random permutation for Z
    #     perm = np.random.permutation(range(m.layer_1.X.mean.shape[0]))
    #     m.layer_1.Z[:] = m.layer_1.X.mean[perm[0:Q],:].values.copy()
        
        # K-means initialization
# Alex -> This was done already in layer initialization?       
#        from sklearn.cluster import KMeans
#        km = KMeans(n_clusters=m.layer_1.Z.shape[0],n_init=1000,max_iter=100)
#        km.fit(m.layer_1.X.mean.values.copy())
#        m.layer_1.Z[:] = km.cluster_centers_.copy()
#        
#        km = KMeans(n_clusters=m.layer_0.Z.shape[0],n_init=1000,max_iter=100)
#        km.fit(m.layer_0.X.mean.values.copy())
#        m.layer_0.Z[:] = km.cluster_centers_.copy()
#        
#        km = KMeans(n_clusters=m.layer_2.Z.shape[0],n_init=1000,max_iter=100)
#        km.fit(m.layer_2.X.mean.values.copy())
#        m.layer_2.Z[:] = km.cluster_centers_.copy()
#    
#        m._trigger_params_changed()
# Alex <-      
    
    # In[69]:
    
    
    #m_init = m.copy()
    
    
    # In[22]:
    
    
    #m = m_init.copy()
    
    
    # In[70]:
    # Initialize kernel parameters
    
    if debug: import pdb; pdb.set_trace()
    # initialization
    for i in range(m.nLayers):
        if not back_cstr:
            m.layers[i].kern.inv_l[:]  = 1./((m.layers[i].X.mean.values.max(0)-m.layers[i].X.mean.values.min(0))/np.sqrt(2.))**2
        else:
            # Ask: MLP kernel?
            # m.layers[i].kern.inv_l[:]  = 1./9.#((m.layers[i].X.mean.values.max(0)-m.layers[i].X.mean.values.min(0))/np.sqrt(2.))
            pass
        m.layers[i].likelihood.variance[:] = 0.01*data_out_train.var()
        m.layers[i].kern.variance.fix(warning=False)
        m.layers[i].likelihood.fix(warning=False)
    
    # Alex ->     
    # m.layer_1.kern.variance[:] = m.layer_1.Xs_flat[0].mean.var()
    # m.layer_2.kern.variance[:] = m.layer_2.Xs_flat[0].mean.var()
    # Ask: why such initialization?
    # m.layer_1.kern.variance[:] = np.vstack([ xs.mean.values for xs in m.layer_1.Xs_flat]).var()
    # m.layer_2.kern.variance[:] = np.vstack([ xs.mean.values for xs in m.layer_2.Xs_flat]).var()
    # Alex <-
    
    if back_cstr:
        # Alex ->     
        # m.layer_1.kern.variance[:] = m.layer_1.Xs_flat[0].mean.var()
        # m.layer_2.kern.variance[:] = m.layer_2.Xs_flat[0].mean.var()
        # Ask: why such initialization?
        m.layer_1.kern.variance[:] = np.vstack([ xs.mean.values for xs in m.layer_1.Xs_flat]).var()
        #m.layer_2.kern.variance[:] = np.vstack([ xs.mean.values for xs in m.layer_2.Xs_flat]).var()
        # Alex <-
    
        m.layer_1.likelihood.variance[:] = 0.01
        m.layer_1.Z.fix()
    else:
        # Alex ->     
        # m.layer_1.kern.variance[:] = m.layer_1.Xs_flat[0].mean.var()
        # m.layer_2.kern.variance[:] = m.layer_2.Xs_flat[0].mean.var()
        # Ask: why such initialization?
        m.layer_1.kern.variance[:] = np.vstack([ xs.mean.values for xs in m.layer_1.Xs_flat]).var()
        m.layer_2.kern.variance[:] = np.vstack([ xs.mean.values for xs in m.layer_2.Xs_flat]).var()
        # Alex <-
    
        m.layer_1.likelihood.variance[:] = 0.01 * m.layer_1.Xs_flat[0].mean.var()
        m.layer_2.likelihood.variance[:] = 0.01 * m.layer_2.Xs_flat[0].mean.var()
    
    m._trigger_params_changed()
    m_init = m.copy()
    m = m_init.copy()
    print('Model after initialization:')
    print(m)
    
    
    # In[ ]:
    file_name = os.path.join(experiment_path, 'alex_walk_run_m%i_sf%1.1f.h5' % (model, input_scaling_factor) )
    
    if debug: import pdb; pdb.set_trace()
    if train_model:
        #import pdb; pdb.set_trace()
        # optimization
        m.optimize('bfgs',messages=1,max_iters=100) # 100
        #m.layer_1.Z.unfix()
        for i in range(m.nLayers):
            m.layers[i].kern.variance.constrain_positive(warning=False)
        #m.optimize('bfgs',messages=1,max_iters=200) # 0
        for i in range(m.nLayers):
            m.layers[i].likelihood.constrain_positive(warning=False)
        m.optimize('bfgs',messages=1,max_iters=10000)
        
        m.save(file_name )
    else:
        
        #=============================
        # Load a trained model
        #=============================
        import tables
        with tables.open_file(file_name,'r') as f:
            ps = f.root.param_array[:]
            f.close()
            m.param_array[:] = ps
            m._trigger_params_changed()
    # In[ ]:
    
    
    #m.optimize('bfgs',messages=1,max_iters=10000)
    
    
    # In[41]:
    
    print('Trained or loaded model:')
    print(m)
    
    
    # In[67]:
    
    
    # Save the model parameters
    # m.save('walk_run_2_new.h5')
    
    
    # ## Evaluate the model
    
    # In[22]:
    
    # Ask: why the mse computation starts later? Because it is not precidse in the beginning?
    # Ask: what about Qx start values? During free run, they are zero, but why they are variational parameters?
    # Ask: when doing predictions, what to do with output noise variance? Trean like summation of two Gaussians?
    if debug: import pdb; pdb.set_trace()
    # Free-run on the training data
    pds_train = [m.freerun(U=data_in, m_match=True) for data_in in data_in_train_list]
    
    rmse = [comp_RMSE(pd.mean[win_out:],gt[win_out+win_in:]) for pd, gt in zip(pds_train, data_out_train_list)]
    rmse_all = comp_RMSE(np.vstack([pd.mean[win_out:] for pd in pds_train]),np.vstack([gt[win_out+win_in:] for gt in data_out_train_list]))
    print(rmse)
    print("Train overall RMSE: "+str(rmse_all))
    
    
    # In[43]:
    
    
    # Free-run on the test data
    pds_test = [m.freerun(U=data_in, m_match=True) for data_in in data_in_test_list]
    
    # Unnormalize data and predictions
    #pds = [ p*data_out_std + data_out_mean for p in pds]
    #data_out_test_list_original = [ p*data_out_std + data_out_mean for p in data_in_test_list]
    #rmse = [comp_RMSE(pd[win_out:],gt[win_out+win_in:]) for pd, gt in zip(pds, data_out_test_list_original)]
    #rmse_all = comp_RMSE(np.vstack([pd[win_out:] for pd in pds]),np.vstack([gt[win_out+win_in:] for gt in data_out_test_list_original]))
    
    rmse = [comp_RMSE(pd.mean[win_out:],gt[win_out+win_in:]) for pd, gt in zip(pds_test, data_out_test_list)]
    rmse_all = comp_RMSE(np.vstack([pd.mean[win_out:] for pd in pds_test]),np.vstack([gt[win_out+win_in:] for gt in data_out_test_list]))
    
    print(rmse)
    print("Test overall RMSE: "+str(rmse_all))
    # 0.53, total 0.55, 1 layer
    # 0.50, total 0.54, 2 layers
    # 0.48, total 0.50, 2 layers 200 inducing points
    # 0.48, total 0.49, 2 layers 200 inducing points
    
    
    # In[44]:
    import pdb; pdb.set_trace()
    # Plot predictions
    numTest = 0
    numOutput = 3
    plt.figure(10)
    plt.plot(data_out_test_list[numTest][win_out:,numOutput])
    plt.plot(pds_test[numTest].mean[:,numOutput])
    plt.show()
    
    # In[ ]:
    
    
    #data_out_test_list_original
    
    
    # In[126]:
    
    
    #data_in_test_list[0].mean()
    
    
    # ## Others
    
    # In[146]:
    
    
    #m.layer_1.Y.mean.var()
    
    
    # In[145]:
    
    
    #m.layer_1.kern.lengthscale
    
    
    # In[147]:
    
    
    #m.layer_1.X.mean.values.std(0)
    
    
    # In[148]:
    
    
    #m.layer_1.X_var_0
    
    
    # In[ ]:
    
    
    #print m.layer_0.Y.var()
    
    #print m
    
    
    # In[31]:
    
    
    #m.save('walk_run_cesar_2layers.h5')
    #m.save('walk_run_cesar_2layers_200ind_2.h5')
    
    
    # In[ ]:
    
    
    # for i in range(m.nLayers):
    #     m.layers[i].likelihood.constrain_positive(warning=False)
    #m.optimize('bfgs',messages=1,max_iters=100000)
    #print m
    
    
    # In[ ]:
    
    
    
    
    
    # In[122]:
    
    # Ask: Scaling the inputs?
    
    #b = data_in_train.copy()
    #b[:] = data_in_train.mean()
    #pd = m.freerun(U=b, m_match=False)
    
    # Test on training data
    #ts_inp = data_in_train_list[0].copy()   # data_in_test_list[0]
    #ts_out = data_out_train_list[0].copy()  # data_out_test_list[0]
    
    #pd = m.freerun(U=ts_inp*0.1, m_match=False)
    # pd = m.freerun(init_Xs=[None,m.layer_1.init_Xs_0.copy()],U=ts_inp*0.1, m_match=False)
    
    #pd = pd.mean.values*data_out_std+data_out_mean
    #mean_pred = m.layer_0.Y.mean(0)*data_out_std+data_out_mean
    
    
    # In[64]:
    
    
    #plot(data_out_train_list[0][win_out:,1])
    #plot(pd[:,1])
    
    
    # In[23]:
    
    
    #pd.shape, data_out_train_list[0].shape
    
    
    # In[124]:
    
    
    #len(data_in_test_list)
    
    
    # In[152]:
    
    
    #ts_inp = data_in_test_list[0]
    #ts_out = data_out_test_list[0]
    
    #pd = m.freerun(U=ts_inp*0.1, m_match=False)
    
    #pd = pd*data_out_std+data_out_mean
    
    
    # In[153]:
    
    
    #plot(data_out_test_list[0][20:,38])
    #plot(pd[:,38])
    
    
    # In[113]:
    
    
    #data_in_test_list[0].shape
    
    
    # In[65]:
    
    
    #mean_pred
    
    
    # In[38]:
    
    
    #print pd[0,:]
    #print pd[0,:]*data_out_std+data_out_mean
    #print ts_out[0,:]
    #mean_pred.shape
    
    
    
    # In[ ]:
    
    # denormalize:
    #for i in range(pd.shape[0]):
    #    pd[i,:] = pd[i,:]*data_out_std+data_out_mean
    
    
    # In[39]:
    
    
    #for i in range(1):
    #    plt.figure()
    #    plt.plot(ts_out[:,i])
    #    plt.figure()
    #    plt.plot(pd[:,i])
    
    #print pd[0:50:3,28]
    #print mean_pred[28]
    
    
    # In[40]:
    
    
    #_=plt.plot(pd[:,0])
    #_=plt.plot(pd[:,1])
    #_=plt.plot(pd[:,4])
    #_=plot(data_out_train[win_out:100,0],'r')
    #_=plot(data_out_train[win_out:100,1],'y')
    
    
    # In[181]:
    if debug: import pdb; pdb.set_trace()
    #get_ipython().system(u'rm imgs/*.png')
    os.system( u'rm %s/*.png ' % (os.path.join( experiment_path, 'imgs'), ) )    
    gen_frames(pds_test[0].mean[win_out:],data_out_mean, data_out_std, data['skel'], os.path.join( experiment_path, 'imgs' ) )
    
    
    # In[182]:
    video_file_name = os.path.join(experiment_path, 'alex_pred_walk_run_m%i_sf%1.1f.mp4' % (model, input_scaling_factor) )
    video_pattern = os.path.join(experiment_path, 'imgs/%05d.png' )
    os.system(u'avconv -y -r 10 -i %s -qscale 2 %s' % ( video_pattern, video_file_name) )
    #get_ipython().system(u' avconv -y -r 10 -i ./imgs/%05d.png -qscale 2  pred_walk_run_cesar5.mp4')
    
    
    # In[183]:
    
    
#    HTML("""
#    <video width="480" height="480" controls>
#      <source src="pred_walk_run_cesar5.mp4" type="video/mp4">
#    </video>
#    """)
    
    
    # In[ ]:
    
    
    #m.layer_1.X.mean.values
    
    
    # In[ ]:
    
    
    #m.layer_1.Us_flat[0].variance
    
    
    # In[ ]:
    
    
    #m.layer_1.kern.lengthscale
    
    
    # In[ ]:
    
    
    #m.layer_0.kern.lengthscale
    
    
    # In[ ]:
    
    
    #m.layer_1.X.mean.std(0)
    
    
    # In[ ]:
    
    
    #plt.plot(data_in_train_list[0])
    #plt.plot(data_in_train_list[6])
    
    
    # In[ ]:
    
    
    #pd = m.freerun(U=np.vstack([data_in_train_list[0],data_in_train_list[5],data_in_train_list[6],data_in_train_list[7]]),m_match=False)
    
    
    # In[ ]:
    
    
#    #=============================
#    # Model with NN-constraint
#    #=============================
#    Q = 500
#    win_in = 20
#    win_out = 20
#    
#    use_controls = True
#    back_cstr = True
#    
#    m = autoreg.DeepAutoreg([0, win_out], data_out_train_list, U=[d*0.1 for d in data_in_train_list], U_win=win_in, X_variance=0.05,
#                        num_inducing=Q, back_cstr=back_cstr, MLP_dims=[500,200], nDims=[data_out_train.shape[1],1],
#                         kernels=[GPy.kern.MLP(win_out,bias_variance=10.),
#                         GPy.kern.MLP(win_out+win_in,bias_variance=10.)])
#    #                      kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True, useGPU=True),
#    #                      GPy.kern.RBF(win_out+win_in,ARD=True,inv_l=True, useGPU=True)])
        
    globals().update(locals()); return # Alex

def svi_test_1(debug = False, train_model=False, model = 1, second_model_svi= False, input_scaling_factor=1):
    """
    After new svi classes are implemented the first test is chaking that non-svi
    inference is not broken.
    
    Basically, two similar models are created using corresponding classes and
    the results are tested. This case is tested when second_model_svi= False.
    
    We can also test 
    """
    experiment_path = '/Users/grigoral/work/code/RGP/examples'
    
    #data = load_data()
    data = load_data_xyz()
    
    
    # In[7]:
    
    y = data['Y']
    u = data['Yxyz_list']
    u_flat = np.vstack(u)
    
    lbls = data['lbls']
    data_out_train = y
    
    # Ask: why first 3 dimensions are removed? # 44 and 56 -output variable is 0.
    data_out_train = y[:,3:]
    data_out_mean  = data_out_train.mean(axis=0)
    data_out_std   = data_out_train.std(axis=0)
    data_out_train = (y[:,3:]-data_out_mean)/data_out_std
    #data_out_train_list = [data_out_train[np.where(lbls[:,i]==1)[0]][1:] for i in range(lbls.shape[1])]
    data_out_train_list = [data_out_train[np.where(lbls[:,i]==1)[0]] for i in range(lbls.shape[1])]
    
                        
    # Create controls
    #data_in_train_list = [y[np.where(lbls[:,i]==1)[0]][:,2][1:] - y[np.where(lbls[:,i]==1)[0]][:,2][:-1] for i in range(lbls.shape[1])]
    #from scipy.ndimage.filters import gaussian_filter1d
    #data_in_train_list = [np.ones(d.shape+(1,))*d.mean() for d in data_in_train_list]
    
    ##data_in_train_list = [gaussian_filter1d(d,8.)[:,None] for d in data_in_train_list]
    ##data_in_train_list = [np.vstack([d[:10],d]) for d in data_in_train_list]
    
    data_in_train_list = u
    u_flat_mean = u_flat.mean(axis=0)
    u_flat_std = u_flat.std(axis=0)
    data_in_train = (u_flat-u_flat_mean)/u_flat_std
        
    #data_in_train_list = u
    data_in_train_list = [(d-u_flat_mean)/u_flat_std for d in data_in_train_list]
    
    # In[8]:
    
    
#    print data_in_train_list[0].shape
#    print data_out_train_list[0].shape
#    
#    for i in range(len(data_in_train_list)):
#        plt.figure()
#        plt.plot(data_in_train_list[i], 'x-')
#        plt.title(i)
#        print data_in_train_list[i].shape[0]
    
    
    # In[9]:
    
    
    print(y.shape)
    print(data_out_train.shape)
    print(u_flat.shape)
    print(data_in_train.shape)
    
    
    # In[10]:
    
    if debug: import pdb; pdb.set_trace()
    ytest = data['Ytest']
    lblstest = data['lblstest']
    u = data['Yxyz_list_test']
    
    #data_out_test = ytest
    data_out_test= ytest[:,3:]
    
    data_out_test = (ytest[:,3:]-data_out_mean)/data_out_std
    
    #data_out_test_list = [data_out_test[np.where(lblstest[:,i]==1)[0]][1:] for i in range(lblstest.shape[1])]
    data_out_test_list = [data_out_test[np.where(lblstest[:,i]==1)[0]] for i in range(lblstest.shape[1])]
    
    # Create controls
    #data_in_test_list = [ytest[np.where(lblstest[:,i]==1)[0]][:,2][1:] - ytest[np.where(lblstest[:,i]==1)[0]][:,2][:-1] for i in range(lblstest.shape[1])]
    #data_in_test_list = [np.ones(d.shape+(1,))*d.mean() for d in data_in_test_list]
    
    #data_in_test_list = u
    
    data_in_test_list = u
    #data_in_test = (u_flat-u_flat_mean)/u_flat_std
    data_in_test_list = [(d-u_flat_mean)/u_flat_std for d in u]
    
    # ## Fit a model without NN-constraint
    
    # In[11]:
    
    
    # Down-scaling the input signals
    #data_in_train_list = [d*0.1 for d in data_in_train_list]
    #data_in_test_list = [d*0.1 for d in data_in_test_list]
    #data_in_train = data_in_train*0.1
    
    # In[13]:
    
    if debug: import pdb; pdb.set_trace()
    #=============================
    # Initialize a model
    #=============================
    
    Q = 100 # 200
    win_in = 20 # 20
    win_out = 20 # 20
    use_controls = True
    back_cstr = False
    
    if input_scaling_factor is None:
        input_scaling_factor = 1
        
    if model == 1:
    # create the model
        if use_controls:
            #m = autoreg.DeepAutoreg([0, win_out], data_out_train, U=data_in_train, U_win=win_in, X_variance=0.05,
            #                    num_inducing=Q, back_cstr=back_cstr, MLP_dims=[300,200], nDims=[data_out_train.shape[1],1],
            #                     kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True, useGPU=True),
            #                     GPy.kern.RBF(win_out+win_in,ARD=True,inv_l=True, useGPU=True)])
            
            # Model without lists
    #        m = autoreg.DeepAutoreg([0, win_out, win_out], data_out_train, U=data_in_train, U_win=win_in, X_variance=0.05,
    #                            num_inducing=Q, back_cstr=back_cstr, MLP_dims=[300,200], nDims=[data_out_train.shape[1],1,1],
    #                             kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True, useGPU=False),
    #                             GPy.kern.RBF(win_out+win_out,ARD=True,inv_l=True, useGPU=False),
    #                             GPy.kern.RBF(win_out+win_in,ARD=True,inv_l=True, useGPU=False)])
            
            # Model with lists
            m = autoreg.DeepAutoreg([0, win_out, win_out], data_out_train_list, U=[d*input_scaling_factor for d in data_in_train_list], U_win=win_in, X_variance=0.05,
                                num_inducing=Q, back_cstr=back_cstr, MLP_dims=[300,200], nDims=[data_out_train.shape[1],1,1],
                                 kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True, useGPU=False),
                                 GPy.kern.RBF(win_out+win_out,ARD=True,inv_l=True, useGPU=False),
                                 GPy.kern.RBF(win_out+win_in,ARD=True,inv_l=True, useGPU=False)])
        
            if not second_model_svi:
                m_svi = autoreg.DeepAutoreg_new([0, win_out, win_out], data_out_train_list, U=[d*input_scaling_factor for d in data_in_train_list], U_win=win_in, X_variance=0.05,
                                    num_inducing=Q, back_cstr=back_cstr, MLP_dims=[300,200], nDims=[data_out_train.shape[1],1,1],
                                     kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True, useGPU=False),
                                     GPy.kern.RBF(win_out+win_out,ARD=True,inv_l=True, useGPU=False),
                                     GPy.kern.RBF(win_out+win_in,ARD=True,inv_l=True, useGPU=False)])
        
                m_svi.param_array[:] = m.param_array
                m_svi._trigger_params_changed()
                
            else:
                m_svi = autoreg.DeepAutoreg_new([0, win_out, win_out], data_out_train_list, U=[d*input_scaling_factor for d in data_in_train_list], U_win=win_in, X_variance=0.05,
                                    num_inducing=Q, back_cstr=back_cstr, MLP_dims=[300,200], nDims=[data_out_train.shape[1],1,1],
                                     kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True, useGPU=False),
                                     GPy.kern.RBF(win_out+win_out,ARD=True,inv_l=True, useGPU=False),
                                     GPy.kern.RBF(win_out+win_in,ARD=True,inv_l=True, useGPU=False)], inference_method='svi')
            
            
            # used with back_cstr=True in the end of the notebook
    #        m = autoreg.DeepAutoreg([0, win_out], data_out_train_list, U=[d*0.1 for d in data_in_train_list], U_win=win_in, X_variance=0.05,
    #                        num_inducing=Q, back_cstr=back_cstr, MLP_dims=[500,200], nDims=[data_out_train.shape[1],1],
    #                         kernels=[GPy.kern.MLP(win_out,bias_variance=10.),
    #                         GPy.kern.MLP(win_out+win_in,bias_variance=10.)])
        else:
            m = autoreg.DeepAutoreg([0, win_out], data_in_train, U=None, U_win=win_in, X_variance=0.05,
                                num_inducing=Q, back_cstr=back_cstr, MLP_dims=[200,100], nDims=[data_out_train.shape[1],1],
                                 kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True, useGPU=False),
                                 GPy.kern.RBF(win_out,ARD=True,inv_l=True, useGPU=False)])
        
            if not second_model_svi:
                m_svi = autoreg.DeepAutoreg_new([0, win_out, win_out], data_out_train_list, U=[d*input_scaling_factor for d in data_in_train_list], U_win=win_in, X_variance=0.05,
                                    num_inducing=Q, back_cstr=back_cstr, MLP_dims=[300,200], nDims=[data_out_train.shape[1],1,1],
                                     kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True, useGPU=False),
                                     GPy.kern.RBF(win_out+win_out,ARD=True,inv_l=True, useGPU=False),
                                     GPy.kern.RBF(win_out+win_in,ARD=True,inv_l=True, useGPU=False)])
        
                m_svi.param_array[:] = m.param_array
                m_svi._trigger_params_changed()
                
            else:
                m_svi = autoreg.DeepAutoreg_new([0, win_out, win_out], data_out_train_list, U=[d*input_scaling_factor for d in data_in_train_list], U_win=win_in, X_variance=0.05,
                                    num_inducing=Q, back_cstr=back_cstr, MLP_dims=[300,200], nDims=[data_out_train.shape[1],1,1],
                                     kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True, useGPU=False),
                                     GPy.kern.RBF(win_out+win_out,ARD=True,inv_l=True, useGPU=False),
                                     GPy.kern.RBF(win_out+win_in,ARD=True,inv_l=True, useGPU=False)], inference_method='svi')
            
    elif model == 2:
        # Ask: no b tern in NLP regularization.
        #=============================
        # Model with NN-constraint
        #=============================
        Q = 500
        win_in = 20
        win_out = 20
        
        use_controls = True
        back_cstr = True
        
        m = autoreg.DeepAutoreg([0, win_out], data_out_train_list, U=[d*input_scaling_factor for d in data_in_train_list], U_win=win_in, X_variance=0.05,
                            num_inducing=Q, back_cstr=back_cstr, MLP_dims=[500,200], nDims=[data_out_train.shape[1],1],
                             kernels=[GPy.kern.MLP(win_out,bias_variance=10.),
                             GPy.kern.MLP(win_out+win_in,bias_variance=10.)])
        #                      kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True, useGPU=True),
        #                      GPy.kern.RBF(win_out+win_in,ARD=True,inv_l=True, useGPU=True)])
        if not second_model_svi:
            m_svi = autoreg.DeepAutoreg_new([0, win_out], data_out_train_list, U=[d*input_scaling_factor for d in data_in_train_list], U_win=win_in, X_variance=0.05,
                                num_inducing=Q, back_cstr=back_cstr, MLP_dims=[500,200], nDims=[data_out_train.shape[1],1],
                                 kernels=[GPy.kern.MLP(win_out,bias_variance=10.),
                                 GPy.kern.MLP(win_out+win_in,bias_variance=10.)])
        #                      kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True, useGPU=True),
        #                      GPy.kern.RBF(win_out+win_in,ARD=True,inv_l=True, useGPU=True)])
        
            m_svi.param_array[:] = m.param_array
            m_svi._trigger_params_changed()
            
        else:
            m_svi = autoreg.DeepAutoreg_new([0, win_out], data_out_train_list, U=[d*input_scaling_factor for d in data_in_train_list], U_win=win_in, X_variance=0.05,
                                num_inducing=Q, back_cstr=back_cstr, MLP_dims=[500,200], nDims=[data_out_train.shape[1],1],
                                 kernels=[GPy.kern.MLP(win_out,bias_variance=10.),
                                 GPy.kern.MLP(win_out+win_in,bias_variance=10.)], inference_method='svi')
        
        #                      kernels=[GPy.kern.RBF(win_out,ARD=True,inv_l=True, useGPU=True),
        #                      GPy.kern.RBF(win_out+win_in,ARD=True,inv_l=True, useGPU=True)])
        
    print("Old model:")
    print(m)
    print("New model:")
    print(m_svi)
    
    if not second_model_svi:
        print("Maximum ll difference:  ", np.max( np.abs(m._log_marginal_likelihood - m_svi._log_marginal_likelihood ) ) )
        print("Maximum ll_grad difference:  ", np.max( np.abs(m._log_likelihood_gradients() - m_svi._log_likelihood_gradients()) ) )
        
    globals().update(locals()); return # Alex
    
        
        
if __name__ == '__main__':
    #experiment1(debug=False, train_model=False, model = 1, input_scaling_factor=1)
    svi_test_1(debug = False, train_model=False, model = 2, second_model_svi= True, input_scaling_factor=1)
    
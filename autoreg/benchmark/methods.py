# Copyright (c) 2015, Zhenwen Dai
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import abc
import numpy as np
import GPy
import autoreg

class AutoregMethod(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        self.preprocess = True
        
    def _preprocess(self, data,  train):
        """Zero-mean, unit-variance normalization by default"""
        if train:
            inputs, labels = data
            self.data_mean = inputs.mean(axis=0)
            self.data_std = inputs.std(axis=0)
            self.labels_mean = labels.mean(axis=0)
            self.labels_std = labels.std(axis=0)
            return ((inputs-self.data_mean)/self.data_std, (labels-self.labels_mean)/self.labels_std)
        else:
            return (data-self.data_mean)/self.data_std
    
    def _reverse_trans_labels(self, labels):
        return labels*self.labels_std+self.labels_mean
        
    def fit(self, train_data):
        if self.preprocess:
            train_data = self._preprocess(train_data, True)
        return self._fit(train_data)
    
    def predict(self, test_data):
        if self.preprocess:
            test_data = self._preprocess(test_data, False)
        labels = self._predict(test_data)
        if self.preprocess:
            labels = self._reverse_trans_labels(labels)
        return labels
    
    @abc.abstractmethod
    def _fit(self, train_data):
        """Fit the model. Return True if successful"""
        return True
    
    @abc.abstractmethod
    def _predict(self, test_data):
        """Predict on test data"""
        return None
    
class Autoreg_onelayer(AutoregMethod):
    name = 'onelayer'
    def __init__(self, win_in, win_out):
        super(Autoreg_onelayer, self).__init__()
        self.win_in = win_in
        self.win_out = win_out
    
    def _fit(self, train_data):
        data_in_train, data_out_train= train_data
        Q0, Q1 = 50,50
        init_runs = 50 if data_in_train.shape[0]<1000 else 100
        

        # create the model
        self.model = autoreg.DeepAutoreg([0,self.win_out],data_out_train, U=data_in_train, U_win=self.win_in,X_variance=0.05,
                            num_inducing=[Q0, Q1],
                             kernels=[GPy.kern.RBF(self.win_out,ARD=True,inv_l=True),GPy.kern.RBF(self.win_out+self.win_in,ARD=True,inv_l=True)])
        m = self.model
        # initialization
        for i in range(m.nLayers):
            m.layers[i].kern.inv_l[:]  = 1./((m.layers[i].X.mean.values.max(0)-m.layers[i].X.mean.values.min(0))/np.sqrt(2.))**2
            m.layers[i].likelihood.variance[:] = 0.01*data_out_train.var()
            m.layers[i].kern.variance.fix(warning=False)
            m.layers[i].likelihood.fix(warning=False)

        # optimization
        m.optimize('bfgs',messages=0,max_iters=init_runs)
        for i in range(m.nLayers):
            m.layers[i].kern.variance.constrain_positive(warning=False)
            m.layers[i].likelihood.constrain_positive(warning=False)
        m.optimize('bfgs',messages=0,max_iters=5000)
        
        return True
    
    def _predict(self, test_data):
        # evaluate on test data
        return self.model.freerun(U=test_data,m_match=False)[self.win_out:]
    
class Autoreg_onelayer_bfgs(AutoregMethod):
    name = 'onelayer-bfgs'
    def __init__(self, win_in, win_out):
        super(Autoreg_onelayer_bfgs, self).__init__()
        self.win_in = win_in
        self.win_out = win_out
    
    def _fit(self, train_data):
        data_in_train, data_out_train= train_data
        Q0, Q1 = 50,50
        init_runs = 50 if data_in_train.shape[0]<1000 else 100
        

        # create the model
        self.model = autoreg.DeepAutoreg([0,self.win_out],data_out_train, U=data_in_train, U_win=self.win_in,X_variance=0.05,
                            num_inducing=[Q0, Q1],
                             kernels=[GPy.kern.RBF(self.win_out,ARD=True,inv_l=True),GPy.kern.RBF(self.win_out+self.win_in,ARD=True,inv_l=True)])
        m = self.model
        # initialization
        for i in range(m.nLayers):
            m.layers[i].kern.inv_l[:]  = 1./((m.layers[i].X.mean.values.max(0)-m.layers[i].X.mean.values.min(0))/np.sqrt(2.))**2
            m.layers[i].likelihood.variance[:] = 0.01*data_out_train.var()
            m.layers[i].kern.variance.fix(warning=False)
            m.layers[i].likelihood.fix(warning=False)

        # optimization
        m.optimize('org-bfgs',messages=0,max_iters=init_runs)
        for i in range(m.nLayers):
            m.layers[i].kern.variance.constrain_positive(warning=False)
            m.layers[i].likelihood.constrain_positive(warning=False)
        m.optimize('org-bfgs',messages=0,max_iters=5000)
        
        return True
    
    def _predict(self, test_data):
        # evaluate on test data
        return self.model.freerun(U=test_data,m_match=False)[self.win_out:]

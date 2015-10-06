# Copyright (c) 2015, Zhenwen Dai
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import abc
import numpy as np
import GPy
import autoreg.benchmark
from autoreg.benchmark.methods import AutoregMethod
    
class Autoreg_twolayer(AutoregMethod):
    name = 'twolayer'
    def __init__(self, win_in, win_out):
        super(Autoreg_twolayer, self).__init__()
        self.win_in = win_in
        self.win_out = win_out
    
    def _fit(self, train_data):
        data_in_train, data_out_train= train_data
        Q0, Q1, Q2 = 50,50, 50
        init_runs = 50 if data_in_train.shape[0]<1000 else 100
        

        # create the model
        self.model = autoreg.DeepAutoreg([0, self.win_out, self.win_out],data_out_train, U=data_in_train, U_win=self.win_in,X_variance=0.05,
                            num_inducing=[Q0, Q1, Q2],
                             kernels=[GPy.kern.RBF(self.win_out,ARD=True,inv_l=True),GPy.kern.RBF(self.win_out+self.win_out,ARD=True,inv_l=True),
                             GPy.kern.RBF(self.win_out+self.win_in,ARD=True,inv_l=True)])
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
    

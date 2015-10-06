# Copyright (c) 2015, Zhenwen Dai
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import abc
import os
import numpy as np
import scipy.io

class AutoregTask(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, datapath=os.path.join(os.path.dirname(__file__),'../../datasets/system_identification')):
        self.datapath = datapath
        
    def _enforce_2d(self):
        if self.data_in_train.ndim==1: self.data_in_train = self.data_in_train[:,None]
        if self.data_out_train.ndim==1: self.data_out_train = self.data_out_train[:,None]
        if self.data_in_test.ndim==1: self.data_in_test = self.data_in_test[:,None]
        if self.data_out_test.ndim==1: self.data_out_test = self.data_out_test[:,None]
    
    @abc.abstractmethod
    def load_data(self):
        """Download the dataset if not exist. Return True if successful"""
        return True
    
#     @abc.abstractmethod
#     def get_training_data(self):
#         """Return the training data: training data and labels"""
#         return None
#     
#     @abc.abstractmethod
#     def get_test_data(self):
#         """Return the test data: training data and labels"""
#         return None

    def get_training_data(self):
        return self.data_in_train, self.data_out_train
    
    def get_test_data(self):
        return self.data_in_test, self.data_out_test
    
class IdentificationExample(AutoregTask):
    
    name='IdentificationExample'
    filename = 'identificationExample.mat'
    
    def load_data(self):
        data = scipy.io.loadmat(os.path.join(self.datapath, self.filename))
        self.data_in = data['u']
        self.data_out = data['y']
        self.win_in = 1
        self.win_out = 1
        self.split_point = 150        
        self.data_in_train = self.data_in[:self.split_point]
        self.data_in_test = self.data_in[self.split_point:]
        self.data_out_train = self.data_out[:self.split_point]
        self.data_out_test = self.data_out[self.split_point:]
        self._enforce_2d()
        return True
    
class IdentificationExample1(AutoregTask):
    
    name='IdentificationExample1'
    filename = 'identificationExample1.mat'
    
    def load_data(self):
        data = scipy.io.loadmat(os.path.join(self.datapath, self.filename))
        self.data_in_train = data['u']
        self.data_out_train = data['y']
        self.data_in_test = data['uNew']
        self.data_out_test = data['yNew']
        self.win_in = 1
        self.win_out = 1
        self._enforce_2d()
        return True
    
class IdentificationExample2(AutoregTask):
    
    name='IdentificationExample2'
    filename = 'identificationExample2.mat'
    
    def load_data(self):
        data = scipy.io.loadmat(os.path.join(self.datapath, self.filename))
        self.data_in_train = data['u']
        self.data_out_train = data['y']
        self.data_in_test = data['uNew']
        self.data_out_test = data['yNew']
        self.win_in = 1
        self.win_out = 2
        self._enforce_2d()        
        return True

class IdentificationExample3(AutoregTask):
    
    name='IdentificationExample3'
    filename = 'identificationExample3.mat'
    
    def load_data(self):
        data = scipy.io.loadmat(os.path.join(self.datapath, self.filename))
        self.data_in_train = data['u']
        self.data_out_train = data['y']
        self.data_in_test = data['uNew']
        self.data_out_test = data['yNew']
        self.win_in = 1
        self.win_out = 1
        self._enforce_2d()        
        return True

class IdentificationExample4(AutoregTask):
    
    name='IdentificationExample4'
    filename = 'identificationExample4.mat'
    
    def load_data(self):
        data = scipy.io.loadmat(os.path.join(self.datapath, self.filename))
        self.data_in_train = data['u']
        self.data_out_train = data['y']
        self.data_in_test = data['uNew']
        self.data_out_test = data['yNew']
        self.win_in = 1
        self.win_out = 2
        self._enforce_2d()        
        return True
    
class IdentificationExample5(AutoregTask):
    
    name='IdentificationExample5'
    filename = 'identificationExample5.mat'
    
    def load_data(self):
        data = scipy.io.loadmat(os.path.join(self.datapath, self.filename))
        self.data_in_train = data['u']
        self.data_out_train = data['y']
        self.data_in_test = data['uNew']
        self.data_out_test = data['yNew']
        self.win_in = 5
        self.win_out = 5
        self._enforce_2d()        
        return True
    
class Actuator(AutoregTask):
    
    name='actuator'
    filename = 'actuator.mat'
    
    def load_data(self):
        data = scipy.io.loadmat(os.path.join(self.datapath, self.filename))
        self.data_in = data['u']
        self.data_out = data['p']
        self.win_in = 10
        self.win_out = 10
        self.split_point = 512        
        self.data_in_train = self.data_in[:self.split_point]
        self.data_in_test = self.data_in[self.split_point:]
        self.data_out_train = self.data_out[:self.split_point]
        self.data_out_test = self.data_out[self.split_point:]
        self._enforce_2d()
        return True
    
class Ballbeam(AutoregTask):
    
    name='ballbeam'
    filename = 'ballbeam.dat'
    
    def load_data(self):
        data = np.loadtxt(os.path.join(self.datapath, self.filename))
        self.data_in = data[:,0]
        self.data_out = data[:,1]
        self.win_in = 10
        self.win_out = 10
        self.split_point = 500
        self.data_in_train = self.data_in[:self.split_point]
        self.data_in_test = self.data_in[self.split_point:]
        self.data_out_train = self.data_out[:self.split_point]
        self.data_out_test = self.data_out[self.split_point:]
        self._enforce_2d()
        return True
    
class Drive(AutoregTask):
    
    name='drive'
    filename = 'drive.mat'
    
    def load_data(self):
        data = scipy.io.loadmat(os.path.join(self.datapath, self.filename))
        self.data_in = data['u1']
        self.data_out = data['z1']
        self.win_in = 10
        self.win_out = 10
        self.split_point = 250    
        self.data_in_train = self.data_in[:self.split_point]
        self.data_in_test = self.data_in[self.split_point:]
        self.data_out_train = self.data_out[:self.split_point]
        self.data_out_test = self.data_out[self.split_point:]
        self._enforce_2d()
        return True
    
class Gas_furnace(AutoregTask):
    
    name='gas_furnace'
    filename = 'gas_furnace.csv'
    
    def load_data(self):
        data = np.loadtxt(os.path.join(self.datapath, self.filename),skiprows=1,delimiter=',')
        self.data_in = data[:,0]
        self.data_out = data[:,1]
        self.win_in = 3
        self.win_out = 3
        self.split_point = 148
        self.data_in_train = self.data_in[:self.split_point]
        self.data_in_test = self.data_in[self.split_point:]
        self.data_out_train = self.data_out[:self.split_point]
        self.data_out_test = self.data_out[self.split_point:]
        self._enforce_2d()
        return True

class Flutter(AutoregTask):
    
    name='flutter'
    filename = 'flutter.dat'
    
    def load_data(self):
        data = np.loadtxt(os.path.join(self.datapath, self.filename))
        self.data_in = data[:,0]
        self.data_out = data[:,1]
        self.win_in = 5
        self.win_out = 5
        self.split_point = 512
        self.data_in_train = self.data_in[:self.split_point]
        self.data_in_test = self.data_in[self.split_point:]
        self.data_out_train = self.data_out[:self.split_point]
        self.data_out_test = self.data_out[self.split_point:]
        self._enforce_2d()
        return True

class Dryer(AutoregTask):
    
    name='dryer'
    filename = 'dryer.dat'
    
    def load_data(self):
        data = np.loadtxt(os.path.join(self.datapath, self.filename))
        self.data_in = data[:,0]
        self.data_out = data[:,1]
        self.win_in = 2
        self.win_out = 2
        self.split_point = 500
        self.data_in_train = self.data_in[:self.split_point]
        self.data_in_test = self.data_in[self.split_point:]
        self.data_out_train = self.data_out[:self.split_point]
        self.data_out_test = self.data_out[self.split_point:]
        self._enforce_2d()
        return True
    
class Tank (AutoregTask):
    
    name='tank'
    filename = 'tank.mat'
    
    def load_data(self):
        data = scipy.io.loadmat(os.path.join(self.datapath, self.filename))
        self.data_in = data['u'].T
        self.data_out = data['y'].T
        self.win_in = 1
        self.win_out = 3
        self.split_point = 1250        
        self.data_in_train = self.data_in[:self.split_point]
        self.data_in_test = self.data_in[self.split_point:]
        self.data_out_train = self.data_out[:self.split_point]
        self.data_out_test = self.data_out[self.split_point:]
        self._enforce_2d()
        return True
    
all_tasks = [IdentificationExample, IdentificationExample1, IdentificationExample2, IdentificationExample3, IdentificationExample4, IdentificationExample5,
             Actuator, Ballbeam, Drive, Gas_furnace, Flutter, Dryer]


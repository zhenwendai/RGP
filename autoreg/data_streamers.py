#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import abc
import numpy as np
import warnings

class DataStreamerTemplate(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def next_minibatch(self, ):
        """
        """
        return None
    
    @abc.abstractproperty
    def get_cur_index(self, ):
        """
        """
    
    @abc.abstractproperty
    def minibatch_size(self, ):
        """
        """
        return None
    
    @abc.abstractproperty
    def total_size(self, ):
        """
        """
        return None
    
    
class TrivialDataStreamer(DataStreamerTemplate):
    """
    This trivial data_streamer returns all the data each iteration.
    
    """
    def __init__(self, Y, X ):
        """
        
        """
        
        if Y is not None and isinstance(Y, np.ndarray): Y = [Y,]
        if X is not None and isinstance(X, np.ndarray): X = [X,]
        
        assert len(Y) == len(X), "Input and output size must match"
        
        self.iterations_started = False
        self.minibatch_size=len(Y)
        self.next_minibatch_start_idx = 0
        self.minibatch_index = 0
        self.last_in_epoch = False
        
        self.Y = Y
        self.X = X
        
        self.total_size = len(Y)
    
    def next_minibatch(self,):
        """
        """
        
        self.iterations_started = True
        self.minibatch_index += 1
        
        st_idx = self.next_minibatch_start_idx
        
        if (self.next_minibatch_start_idx + self.minibatch_size) < self.total_size:
            end_idx = self.next_minibatch_start_idx + self.minibatch_size
            self.last_in_epoch = False
        else:
            end_idx = np.min( (self.next_minibatch_start_idx + self.minibatch_size, self.total_size) ) 
            self.last_in_epoch = True
            
        Y_out = self.Y[st_idx:end_idx]
        X_out = self.X[st_idx:end_idx]
        minibatch_index_out = self.minibatch_index
        
        if self.last_in_epoch:
            self.next_minibatch_start_idx = 0
            self.minibatch_index = 0
        
        return minibatch_index_out, range(len(self.Y)), Y_out, X_out

    def get_cur_index(self, ):
        """
        """
        
        return self.minibatch_index, range(self.minibatch_size)
    
    def minibatch_size(self, ):
        """
        """
        return self.minibatch_size
    

    def total_size(self, ):
        """
        """
        return self.total_size
    
    def minibatch_last_in_epoch(self,):
        """
        """
        
        return None
    
class RandomPermutationDataStreamer(DataStreamerTemplate):
    """
    This trivial data_streamer returns random permutation at each iteration.
    
    """
    def __init__(self, Y, X ):
        """
        
        """
        
        if Y is not None and isinstance(Y, np.ndarray): 
            Y = [Y,]
            warnings.warn("Input has only one sequence. No permutation functionality will be used.", RuntimeWarning)
            
        if X is not None and isinstance(X, np.ndarray): 
            X = [X,]
        
        assert len(Y) == len(X), "Input and output size must match"
        
        self.iterations_started = False
        self.minibatch_size=len(Y)
        self.next_minibatch_start_idx = 0
        self.minibatch_index = 0
        self.previous_indexes_out = None
        self.last_in_epoch = False
        
        self.Y = Y
        self.X = X
        
        self.total_size = len(Y)
        
        
    
    def next_minibatch(self,):
        """
        """
        import random
        
        self.iterations_started = True
        self.minibatch_index += 1
        
        st_idx = self.next_minibatch_start_idx
        
        if (self.next_minibatch_start_idx + self.minibatch_size) < self.total_size:
            end_idx = self.next_minibatch_start_idx + self.minibatch_size
            self.last_in_epoch = False
        else:
            end_idx = np.min( (self.next_minibatch_start_idx + self.minibatch_size, self.total_size) ) 
            self.last_in_epoch = True
            
        Y_out = self.Y[st_idx:end_idx]
        X_out = self.X[st_idx:end_idx]
        
        
        rand_inds = random.sample(range(len(Y_out)),len(Y_out))
        self.previous_indexes_out = rand_inds[:] # copying
        
        Y_out = [ Y_out[i] for i in rand_inds ]
        X_out = [ X_out[i] for i in rand_inds ]
        
        minibatch_index_out = self.minibatch_index
        
        if self.last_in_epoch:
            self.next_minibatch_start_idx = 0
            self.minibatch_index = 0
        
        return minibatch_index_out, rand_inds, Y_out, X_out

    def get_cur_index(self, ):
        """
        """
        
        return self.minibatch_index, self.previous_indexes_out
    
    def minibatch_size(self, ):
        """
        """
        return self.minibatch_size
    

    def total_size(self, ):
        """
        """
        return self.total_size
    
    def minibatch_last_in_epoch(self,):
        """
        """
        
        return None
    
    
    
class StdMemoryDataStreamer(DataStreamerTemplate):
    """
    This is a standard data_streamer for the data which fits into memorys.
    Data is assumed to be in lists.
    """
    def __init__(self, Y, X, minibatch_size ):
        """
        
        """
        
        if Y is not None and isinstance(Y, np.ndarray): 
            Y = [Y,]
            warnings.warn("Input has only one sequence. No permutation functionality will be used.", RuntimeWarning)
            
        if X is not None and isinstance(X, np.ndarray): 
            X = [X,]
        
        assert len(Y) == len(X), "Input and output size must match"
        assert minibatch_size <= len(Y), "Minibatch size must be less than the data size."
        
        self.iterations_started = False
        self.minibatch_size=minibatch_size
        self.next_minibatch_start_idx = 0
        self.minibatch_index = 0
        self.last_in_epoch = False
        self.previous_indexes_out = None
        
        self.Y = Y
        self.X = X
        
        self.total_size = len(Y)
        
        
    
    def next_minibatch(self,):
        """
        """
        #import pdb; pdb.set_trace()
        
        self.iterations_started = True
        self.minibatch_index += 1
        
        st_idx = self.next_minibatch_start_idx
        
        if (self.next_minibatch_start_idx + self.minibatch_size) < self.total_size:
            end_idx = self.next_minibatch_start_idx + self.minibatch_size
            self.last_in_epoch = False
        else:
            end_idx = np.min( (self.next_minibatch_start_idx + self.minibatch_size, self.total_size) ) 
            self.last_in_epoch = True
            
        Y_out = self.Y[st_idx:end_idx]
        X_out = self.X[st_idx:end_idx]
        indexes_out = range(st_idx,end_idx)
        self.previous_indexes_out = indexes_out[:] # copying
        
        minibatch_index_out = self.minibatch_index
        
        if self.last_in_epoch:
            self.next_minibatch_start_idx = 0
            self.minibatch_index = 0
        else:
            self.next_minibatch_start_idx += self.minibatch_size
        
        return minibatch_index_out, indexes_out, Y_out, X_out

    def get_cur_index(self, ):
        """
        """
        
        return self.minibatch_index, self.previous_indexes_out
    
    def minibatch_size(self, ):
        """
        """
        return self.minibatch_size
    

    def total_size(self, ):
        """
        """
        return self.total_size
    
    def minibatch_last_in_epoch(self,):
        """
        """
        
        return None
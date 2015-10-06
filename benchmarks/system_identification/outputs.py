# Copyright (c) 2015, Zhenwen Dai
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import print_function
import abc
import os
import numpy as np

class Output(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def output(self, config, task_list, method_list, results):
        """Return the test data: training data and labels"""
        return None

# class ScreenOutput(Output):
#             
#     def output(self, config, results):
#         print('='*10+'Report'+'='*10)
#         print('\t'.join([' ']+[m.name+'('+e+')' for m in config['methods'] for e in [a.name for a in config['evaluations']]+['time']]))
#         for task_i in range(len(config['tasks'])):
#             print(config['tasks'][task_i].name+'\t', end='')
# 
#             outputs = []
#             for method_i in range(len(config['methods'])):
#                 for ei in range(len(config['evaluations'])+1):
#                     m,s = results[task_i, method_i, ei].mean(), results[task_i, method_i, ei].std()
#                     outputs.append('%e(%e)'%(m,s))
#             print('\t'.join(outputs))

class CSV_Summary(Output):
    
    def __init__(self, outpath, prjname):
        self.fname = os.path.join(outpath, prjname+'.csv')
        
    def output(self, config, task_list, method_list, results):
        import pandas
        
        nEvals = results.columns[4:].shape[0]
        s = pandas.DataFrame(index=task_list['task'], columns=[m+'-'+e+'-'+x for m in method_list['method'] for e in results.columns[4:] for x in ['mean', 'std']])
        
        for task_i in range(task_list.shape[0]):
            for method_i in range(method_list.shape[0]):
                for e_i in range(nEvals):
                    s.iloc[task_i,method_i*nEvals*2+e_i*2] = results[(results.task_id==task_i) & (results.method_id==method_i)][results.columns[e_i+4]].mean()
                    s.iloc[task_i,method_i*nEvals*2+e_i*2+1] = results[(results.task_id==task_i) & (results.method_id==method_i)][results.columns[e_i+4]].std()  
        s.to_csv(self.fname)
        
            
class HDF5Output(Output):
    
    def __init__(self, outpath, prjname):
        self.fname = os.path.join(outpath, prjname+'.h5')
        
    def output(self, config, task_list, method_list, results):
            try:
                from pandas import HDFStore
                store = HDFStore(self.fname)
                store['task_list'] = task_list
                store['method_list'] = method_list
                store['results'] = results
                store.close()
            except:
                raise 'Fails to write the parameters into a HDF5 file!'

class PickleOutput(Output):
    
    def __init__(self, outpath, prjname):
        self.fname = os.path.join(outpath, prjname)
        
    def output(self, config, task_list, method_list, results):
            try:
                task_list.to_pickle(self.fname+'_tasklist.pkl')
                method_list.to_pickle(self.fname+'_methodlist.pkl')
                results.to_pickle(self.fname+'_results.pkl')
            except:
                raise 'Fails to pickle the results!'


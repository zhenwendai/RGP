# Copyright (c) 2015, Zhenwen Dai
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import print_function
from evaluation import RMSE
from methods import Autoreg_onelayer
from tasks import all_tasks
from outputs import PickleOutput, CSV_Summary
import numpy as np
import time
import pandas

outpath = '.'
prjname = 'autoreg'
config = {
          'use_jug': True,
          'evaluations':[RMSE,'time'],
          'methods':[Autoreg_onelayer],
          'tasks': all_tasks,
          'repeats': 5,
          'outputs': [PickleOutput(outpath, prjname), CSV_Summary(outpath, prjname)]
          }


if __name__=='__main__':
    
    nTask = len(config['tasks'])
    nMethod = len(config['methods'])
    nRepeats = int(config['repeats'])
    task_list = pandas.DataFrame({'task_id':range(nTask), 'task':[t.name for t in config['tasks']]})
    method_list = pandas.DataFrame({'method_id':range(nMethod), 'method':[m.name for m in config['methods']]})
    
    results = pandas.DataFrame(index=range(nTask*nMethod*nRepeats),columns=['task_id','method_id','repeat_id','model']+[e.name if not isinstance(e, str) else e for e in config['evaluations']])
    
    for task_i in range(nTask):
        dataset = config['tasks'][task_i]()
        print('Benchmarking on '+dataset.name)
        res = dataset.load_data()
        win_in, win_out = dataset.win_in, dataset.win_out
        if not res: print('Fail to load '+config['tasks'][task_i].name); continue
        train = dataset.get_training_data()
        test = dataset.get_test_data()
        
        for method_i in range(nMethod):
            method = config['methods'][method_i]
            print('With the method '+method.name, end='')
            for ri in range(nRepeats):
                res_idx = ri+method_i*nRepeats+task_i*nRepeats*nMethod
                
                t_st = time.time()
                m = method(win_in, win_out)
                m.fit(train)
                pred = m.predict(test[0])
                t_pd = time.time() - t_st
                results.iloc[res_idx,0], results.iloc[res_idx,1], results.iloc[res_idx,2] = task_i, method_i, ri
                results.iloc[res_idx,3] = m.model.param_array.copy()
                for ei in range(len(config['evaluations'])):
                    if config['evaluations'][ei] !='time':    
                        evalu = config['evaluations'][ei]()
                        results.iloc[res_idx,ei+4] = evalu.evaluate(test[1][win_out+win_in:], pred)
                    else:
                        results.iloc[res_idx,ei+4] = t_pd
                print('.',end='')
            print()
    
        [out.output(config, task_list, method_list, results) for out in config['outputs']]


            
            


import sys
import pickle
from outputs import CSV_Summary

if __name__=='__main__':
    if len(sys.argv)==1: print 'Need the project name!'; exit()
    prjname = sys.argv[1]
    
    with open(prjname+'_tasklist.pkl','r') as f:
        task_list = pickle.load(f)
        f.close()
    with open(prjname+'_methodlist.pkl','r') as f:
        method_list = pickle.load(f)
        f.close()
    with open(prjname+'_results.pkl','r') as f:
        results = pickle.load(f)
        f.close()
        
    s = CSV_Summary('.', prjname)
    s.output(None, task_list, method_list, results)
    
    
import autoreg.benchmark
from autoreg.benchmark.run import run
from autoreg.benchmark.methods import Autoreg_onelayer
from autoreg.benchmark.tasks import all_tasks
from autoreg.benchmark.evaluation import RMSE
from autoreg.benchmark.outputs import PickleOutput, CSV_Summary

outpath = '.'
prjname = 'autoreg'
config = {
          'evaluations':[RMSE,'time'],
          'methods':[Autoreg_onelayer],
          'tasks': all_tasks,
          'repeats': 5,
          'outputs': [PickleOutput(outpath, prjname), CSV_Summary(outpath, prjname)]
          }

if __name__=='__main__':
	run(config)

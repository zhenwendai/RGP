import autoreg.benchmark
from autoreg.benchmark.run import run
from methods import Autoreg_twolayer
from autoreg.benchmark.tasks import all_tasks
from autoreg.benchmark.evaluation import RMSE
from autoreg.benchmark.outputs import PickleOutput, CSV_Summary

outpath = '.'
prjname = 'autoreg-twolayer'
config = {
          'evaluations':[RMSE,'time'],
          'methods':[Autoreg_twolayer],
          'tasks': all_tasks,
          'repeats': 5,
          'outputs': [PickleOutput(outpath, prjname), CSV_Summary(outpath, prjname)]
          }

if __name__=='__main__':
	run(config)

from __future__ import division

import datetime
import glob
import itertools
import os
import re
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from bix.evaluation.study import Study


class GridSearch(Study):
    """ GridSearch

    A grid search class for parameter grid search.
    Evaluates given classifier over streams with initialized parameters.
    Results are saved in csv per stream
    Best runs by means of accuracy and selected parameters in "Best_runs_<stream>.csv" file

    Parameters
    ----------
    streams: list(Stream)
        List of streams which will be evaluated. If no streams given, standard streams are initialized.

    path: String (Default: "search")
        Path to directory for save of gird search results. Default directory is "search".
        Folder will be created if non existent.

    clf: BaseEstimator
        Any scikit based classifier based on scikit estimator. Can't be none.

    max_samples: int (Default: 50000)
        Amount of samples generated by given streams.
    Notes
    -----

    Examples
    -----
    >>> import numpy as np
    >>> from bix.evaluation.gridsearch import GridSearch
    >>> from bix.classifiers.rrslvq import RRSLVQ
    >>> grid = {"sigma": np.arange(2,3,2), "prototypes_per_class": np.arange(2,3,2)}
    >>> clf = RRSLVQ()
    >>> gs = GridSearch(clf=clf,grid=grid,max_samples=1000)
    >>> gs.search()
    >>> gs.save_summary()

    """

    def __init__(self,clf,grid,streams=None,path="search",max_samples = 50000):
        super().__init__(streams=streams,path=path)
        if clf == None:
            raise ValueError("Classifier Instance must be set!")
        if grid == None or type(grid) != dict:
            raise ValueError("Parameter Grid must be set and be dictionary!")

        self.clf = clf
        for elem in grid.values():
            if elem.size == 0:
                raise ValueError("Parameter Grid must be must contain arrays with size greater than zero!")
        self.grid = grid
        self.best_runs = []
        self.path = path
        self.max_samples = max_samples

    def reset(self):
        self.__init__(streams=self.streams,path=self.path)


    def search(self):
        start = time.time()
        self.best_runs.extend(Parallel(n_jobs=1,max_nbytes=None)(delayed(self.self_job)(stream,self.clf,self.grid,self.metrics,self.max_samples) for stream in self.streams))
        end = time.time() - start
        print("\n--------------------------\n")
        print("Duration of grid search "+str(end)+" seconds")

    def self_job(self,stream,clf,grid,metrics,max_samples):
        results = []
        matrix = list(itertools.product(*[list(v) for v in grid.values()]))
        for param_tuple in matrix:
            try: clf.reset()
            except NotImplementedError: clf.__init__()
            for i,param in enumerate(param_tuple):
                clf.__dict__[list(grid.keys())[i]] = int(param) if param.dtype == 'int32' else param
            stream.prepare_for_use()
            evaluator = EvaluatePrequential(show_plot=False,max_samples=self.max_samples, restart_stream=True,batch_size=10,metrics=metrics)
            evaluator.evaluate(stream=stream, model=clf)
            results.append(list(param_tuple)+np.array([[m for m in evaluator._data_buffer.data[n]["mean"]] for n in evaluator._data_buffer.data]).T.flatten().tolist())
        s_name = stream.basename if stream.name==None else stream.name
        dfr = pd.DataFrame(results,columns=list(self.grid.keys())+np.array([[*evaluator._data_buffer.data]]).flatten().tolist())
        dfr = dfr.round(3)

        self.chwd_root()
        os.chdir(os.path.join(os.getcwd(),self.path))

        dfr.to_csv(path_or_buf="Result_"+"_"+self.date+"_"+s_name+"_"+self.clf.__class__.__name__+".csv")
        print("\n ------------------ \n")
        print("Best run on "+s_name+" with "+" "+self.clf.__class__.__name__+" "+str(dfr.values[dfr["accuracy"].values.argmax()]))
        return [s_name]+[self.clf.__class__.__name__]+dfr.values[dfr["accuracy"].values.argmax()].tolist()

    def merge_summary(self):
        self.chwd_root()
        os.chdir(os.path.join(os.getcwd(),self.path))

        for file in glob.glob("*.csv"):
            dfr = pd.read_csv(file,index_col=0,header=0)

            s_name = "NA"
            for s in self.streams:
                if s.name in file:
                    s_name = s.name

            self.best_runs.append([s_name]+[self.clf.__class__.__name__] + dfr.values[dfr["accuracy"].values.argmax()].tolist())
        self.save_summary()

    def save_summary(self):
        if len(self.best_runs) == 0:
            raise ValueError("No results to save! Run test prior!")
        self.chwd_root()
        os.chdir(os.path.join(os.getcwd(),self.path))
        df = pd.DataFrame(self.best_runs,columns=["Stream","Classifier"]+list(self.grid.keys())+self.metrics)
        df.to_csv(path_or_buf="Best_runs_"+"_"+self.date+"_"+self.clf.__class__.__name__+"_.csv", index=False)
        self.chwd_root()

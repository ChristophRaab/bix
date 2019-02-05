from __future__ import division

import datetime
import glob
import os
import re
import time
from datetime import datetime
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import copy
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from bix.evaluation.study import Study


class CrossValidation(Study):
    """CrossValidation
    A class for creating cross validation studies. Executes a number
    of tests on given streams evaluated by classifiers.  
    Saves summarized performance of classifiers and stream into "<RandomNumber>_CV_{*}.csv" file, one per metric.
    Saves single performance of one classifier on one stream into csv over all metrics.
    Main class for BIX-studies!

    Parameters
    ----------
    clfs: list(BaseEstimators)

    streams: list(Stream)
        List of streams which will be evaluated. If no streams given, standard streams are initialized.

    test_size: int (Default: 1)
        Number of test runs on given setups. For multiple test runs, performance metrics 
        are summarized as mean over single stream evaluated by classifier.

    path: String (Default: "study")
        Path to directory for save of cross validation study results. Default directory is "study".
        Folder will be created if non existent. 

    param_path: String (Default: "search")
        Path where best_runs parameters created by GridSearch are searched. Searching for Best runs and name of estimator files.
        Loads parameters by given path and uses them for this study.  

    max_samples: int (Default: 1000000)
        Amount of samples generated by given streams. 

    Notes
    -----


    Examples
    -----
    >>> # Imports
    >>> import numpy as np
    >>> from bix.evaluation.crossvalidation import CrossValidation
    >>> from bix.classifiers.rrslvq import RRSLVQ
    >>> from skmultiflow.bayes.naive_bayes import NaiveBayes
    >>> # Init estimator objects in list.  
    >>> cv = CrossValidation(clfs=[RRSLVQ(),NaiveBayes()],max_samples=500,test_size=1)
    >>> # Init some non standard streams
    >>> cv.streams = cv.init_reoccuring_streams()
    >>> # Test and save summary
    >>> cv.test()
    >>> cv.save_summary()
    """

    def __init__(self, clfs, streams=None, test_size=1, path="study", param_path="search", max_samples=1000000):
        super().__init__(streams=streams, path=path)

        if type(clfs) == 'list':
            raise ValueError("Must be classifier list")

        self.non_multiflow_metrics = [
            "time", "sliding_mean", "mean_std", "window_std"]
        self.clfs = clfs
        self.param_path = param_path
        self.test_size = test_size
        self.result = []
        self.max_samples = max_samples
        self.time_result = []
        self.grid_result = []

    def reset(self):
        self.__init__(self, self.clfs)

    def create_grid(self, clfs, streams):
        grid = []
        for clf in clfs:
            for stream in streams:
                grid.append([copy.deepcopy(clf), stream])
        return grid

    def test(self):
        start = time.time()
        grid = self.create_grid(self.clfs, self.streams)
        self.result.extend(Parallel(n_jobs=1,max_nbytes=None )
                           (delayed(self.grid_job)(elem[0], elem[1]) for elem in grid))
        # for elem in grid:
        #     self.result.append(self.grid_job(elem[0],elem[1]))
        self.result = self.process_results(self.clfs, self.result)
        end = time.time() - start
        print("\n--------------------------\n")
        print("Duration of grid study validation "+str(end)+" seconds")

    def grid_job(self, clf, stream):
        clf_result = []
        time_result = []
        params = self.search_best_parameters(clf)
        self.chwd_root()
        os.chdir(os.path.join(os.getcwd(), self.path))
        print(clf.__class__.__name__)
        clf = self.set_clf_params(clf, params, stream.name)
        local_result = []
        for i in range(self.test_size):
            stream.prepare_for_use()
            stream.name = stream.basename if stream.name == None else stream.name
            path_to_save = clf.__class__.__name__ + \
                "_performance_on_"+stream.name+"_"+self.date+".csv"
            evaluator = EvaluatePrequential(
                show_plot=False, max_samples=self.max_samples, restart_stream=True, batch_size=10, metrics=self.metrics, output_file=path_to_save)
            evaluator.evaluate(stream=stream, model=clf)
            saved_metric = pd.read_csv(
                path_to_save, comment='#', header=0).astype(np.float32)
            saved_values = saved_metric.values[:, 1:3]
            saved_values.setflags(write=1)
            stds = np.std(saved_values, axis=0).tolist()
            sliding_mean = [np.mean(saved_metric.values[:, 2], axis=0)]
            output = np.array([[m for m in evaluator._data_buffer.data[n]["mean"]] for n in evaluator._data_buffer.data]+[
                [evaluator.running_time_measurements[0]._total_time]]).T.flatten().tolist()+sliding_mean+stds
            print(path_to_save+" "+str(output))
            local_result.append(output)

        clf_result = np.mean(local_result, axis=0).tolist()

        return [clf.__class__.__name__]+clf_result

    def process_results(self, clfs, result):
        new_result = []
        for clf in self.clfs:
            name = clf.__class__.__name__
            r = [k[1:] for k in result if name in k]
            new_result.append([name]+r)
        return new_result

    def set_clf_params(self, clf, df, name):
        if isinstance(df, pd.DataFrame):
            row = df[df['Stream'] == name]
            if len(row) == 1:
                for k, v in zip(list(row.keys()), row.values[0]):
                    if k in clf.__dict__.keys():
                        clf.__dict__[k] = int(v) if type(v) == float else v
        return clf

    def search_best_parameters(self, clf):
        self.chwd_root()
        os.chdir(os.path.join(os.getcwd(), self.param_path))
        try:
            files = glob.glob("Best_runs*"+clf.__class__.__name__+"*.csv")

            file = self.determine_newest_file(files)
            return pd.read_csv(files[0]) if len(file) > 0 else []
        except FileNotFoundError:
            return None

    def save_summary(self):
        if len(self.result) == 0:
            raise ValueError("No results to save! Run test prior!")
        self.chwd_root()
        os.chdir(os.path.join(os.getcwd(), self.path))
        for i, metric in enumerate(self.metrics+self.non_multiflow_metrics):
            values = np.array([elem[1:]
                               for elem in self.result])[:, :, i].tolist()
            names = [[elem[0]] for elem in self.result]
            df = pd.DataFrame([n+elem for n, elem in zip(names, values)],
                              columns=["Classifier"]+[s.name for s in self.streams])
            df = df.round(3)
            df.to_csv(path_or_buf=str(np.random.randint(10))+"_CV_Study"+"_"+metric+"_"+self.date +
                      "_N_Classifier_"+str(len(self.clfs))+".csv", index=False)

    def determine_newest_file(self, files):
        dates = [datetime.strptime(re.search(
            r'\d{4}-\d{2}-\d{2} \d{2}-\d{2}', file).group(), self.date_format) for file in files]
        return files[dates.index(max(dates))] if len(dates) > 0 else []

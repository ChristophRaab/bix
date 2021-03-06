#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 3 08:58:46 2019

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""
import os
import numpy as np
from bix.evaluation.gridsearch import GridSearch
from bix.evaluation.crossvalidation import CrossValidation
from bix.classifiers.arslvq import ARSLVQ
from skmultiflow.trees import HAT
from bix.classifiers.rslvq import RSLVQ
from skmultiflow.lazy import SAMKNN, KNN
from skmultiflow.meta import OzaBagging, OzaBaggingAdwin, AdaptiveRandomForest
from skmultiflow.data import LEDGeneratorDrift
from bix.data.reoccuringdriftstream import ReoccuringDriftStream
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.data.file_stream import FileStream
def test_parameter_grid_search_arslvq():
    grid = {"sigma": np.append(1, np.arange(
        2, 11, 2)), "prototypes_per_class": np.append(1, np.arange(2, 11, 2)),
        "gamma": np.array([0.7, 0.9, 0.999]),
        "confidence": np.array([0.01, 0.001]),
        "window_size": np.array([100, 200, 300, 800])}
    clf = ARSLVQ()
    gs = GridSearch(clf=clf, grid=grid, max_samples=50000)
    gs.streams = gs.init_real_world() + gs.init_standard_streams()  + gs.init_reoccuring_standard_streams()
    gs.search()
    gs.save_summary()

def test_missing_streams():
    grid = {"sigma": np.append(1, np.arange(
        2, 11, 2)), "prototypes_per_class": np.append(1, np.arange(2, 11, 2)),
            "gamma": np.array([0.7, 0.9, 0.999]),
            "confidence": np.array([0.01, 0.001]),
            "window_size": np.array([800])}
    clf = ARSLVQ()
    led_a = ConceptDriftStream(stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                               drift_stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0,
                                                              n_drift_features=7),
                               random_state=None,
                               alpha=90.0,  # angle of change grade 0 - 90
                               position=250000,
                               width=1)

    led_a.name = "led_a"
    led_g = ConceptDriftStream(stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                               drift_stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0,
                                                              n_drift_features=7),
                               random_state=None,
                               position=250000,
                               width=50000)
    led_g.name = "led_g"
    led_fa = ReoccuringDriftStream(stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                                  drift_stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0,
                                                                 n_drift_features=7),
                                  random_state=None,
                                  alpha=90.0,  # angle of change grade 0 - 90
                                  position=2000,
                                  width=1)

    led_fg = ReoccuringDriftStream(stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                                  drift_stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0,
                                                                 n_drift_features=7),
                                  random_state=None,
                                  position=2000,
                                  width=1000)
    covertype = FileStream(os.path.realpath('covtype.csv'))  # Label failure
    covertype.name = "covertype"
    poker = FileStream(os.path.realpath('poker.csv'))  # label failure
    poker.name = "poker"
    airlines = FileStream(os.path.realpath('airlines.csv'))
    airlines.name = "airport"
    gs = GridSearch(clf=clf, grid=grid, max_samples=50000)

    gs.streams = [led_a,led_g,led_fa,led_fg,covertype,poker,airlines]
    gs.search()
    gs.save_summary()

def test():
    grid = {"sigma": np.append(1, np.arange(
        2, 11, 2)), "prototypes_per_class": np.append(1, np.arange(2, 11, 2)),
        "gamma": np.array([0.7, 0.9, 0.999]),
        "confidence": np.array([0.01, 0.001]),
        "window_size": np.array([100, 200, 300, 800])}
    clf = ARSLVQ()
    gs = GridSearch(clf=clf, grid=grid, max_samples=50000)
    gs.streams = gs.init_real_world() + gs.init_standard_streams()  + gs.init_reoccuring_streams()+gs.init_reoccuring_standard_streams()
    gs.merge_summary()

def test_rslvq():
    from bix.classifiers.rrslvq import RRSLVQ
    clf =[ RRSLVQ(prototypes_per_class=4,sigma=6,confidence=0.001)]
    gs = CrossValidation(clfs=clf)
    gs.streams = gs.init_real_world() + gs.init_standard_streams()  +gs.init_reoccuring_standard_streams()
    gs.test()
    gs.save_summary()


def test_grid():
    clfs = [
            AdaptiveRandomForest(), 
            SAMKNN(),
            HAT()
    ]
    cv = CrossValidation(clfs=clfs, max_samples=1000000, test_size=1)
    cv.streams = cv.init_real_world() + cv.init_standard_streams()  + cv.init_reoccuring_standard_streams()
    cv.test()
    cv.save_summary()
def test_led():
    led_a = ConceptDriftStream(stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                               drift_stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0,
                                                              n_drift_features=7),
                               random_state=None,
                               alpha=90.0,  # angle of change grade 0 - 90
                               position=250000,
                               width=1)

    led_a.name = "led_a"
    led_g = ConceptDriftStream(stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                               drift_stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0,
                                                              n_drift_features=7),
                               random_state=None,
                               position=250000,
                               width=50000)
    led_g.name = "led_g"
    led_fa = ReoccuringDriftStream(stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                                  drift_stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0,
                                                                 n_drift_features=7),
                                  random_state=None,
                                  alpha=90.0,  # angle of change grade 0 - 90
                                  position=2000,
                                  width=1)

    led_fg = ReoccuringDriftStream(stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                                  drift_stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0,
                                                                 n_drift_features=7),
                                  random_state=None,
                                  position=2000,
                                  width=1000)

    np = 2
    sigma = 3
    clfs = [ARSLVQ(prototypes_per_class=np, sigma=sigma, confidence=0.0001, window_size=1500), OzaBaggingAdwin(),
            AdaptiveRandomForest(), HAT(), RSLVQ(prototypes_per_class=np,sigma=sigma), SAMKNN()]

    cv = CrossValidation(clfs=clfs,parallel=1)
    cv.streams = [led_a,led_g,led_fa,led_fg]
    cv.search()
    cv.save_summary()


if __name__ == "__main__":
   # test_parameter_grid_search_arslvq()
   #test_missing_streams()
    test_led()
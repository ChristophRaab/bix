#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 09:53:09 2019

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

import unittest
from bix.classifiers.glvq import GLVQ
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
import numpy as np
from skmultiflow.data.concept_drift_stream import ConceptDriftStream

class TestGLVQ(unittest.TestCase):
       
    def test_correct_init(self):
        GLVQ(prototypes_per_class=2, beta=2, C=None)
    
    def test_accuracy_stream(self):
        stream = ConceptDriftStream(stream=SEAGenerator(random_state=112, noise_percentage=0.1), 
                            drift_stream=SEAGenerator(random_state=112, 
                                                          classification_function=1, noise_percentage=0.1),
                            random_state=None,
                            position=20000,
                            width=50000)
        stream.prepare_for_use()

        clf = GLVQ(prototypes_per_class=6, beta=2, C=None,decay_rate=0.9,gradient_descent="SGD")

        evaluator = EvaluatePrequential(pretrain_size=1, show_plot=False, max_samples=20000,
                                        batch_size=1)

        evaluator.evaluate(stream, clf, model_names=['GLVQ'])

        measurements = np.asarray(evaluator.get_measurements()[0])[0]

        self.assertTrue(measurements.get_accuracy() >= 0.93,
                        msg='Accuracy was {} but has to be greater than 0.93'.
                        format(measurements.get_accuracy()))
        self.assertTrue(measurements.get_kappa() >= 0.84,
                        msg='Kappa was {} but has to be greater than 0.84'.
                        format(measurements.get_kappa()))
        
if __name__ == '__main__':
    unittest.main()
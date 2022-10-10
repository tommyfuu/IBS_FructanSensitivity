# -*- coding: utf-8 -*-

"""
Author      : Tom Fu
Date        : 2020 June 29
FileName    : svmModel.py (for Coarfa Lab)
Description : Vectorised implementation
"""

from sklearn.svm import SVC

import evaluatePipeline
from sklearn.model_selection import ParameterGrid



def svmEvaluate(addressX, addressY):
    # preparing for optimising
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gammas = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    kernels = ['rbf', 'sigmoid']
    param_grid = {'C': Cs, 'gamma' : gammas, 'kernel': kernels}
    # param_grid = ParameterGrid(param_grid)
    return evaluatePipeline.evaluate(addressX, addressY, SVC(), "svm", parameter_grid = param_grid)
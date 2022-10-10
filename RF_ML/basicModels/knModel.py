# -*- coding: utf-8 -*-

"""
Author      : Tom Fu
Date        : 2020 June 29
FileName    : knModel.py (for Coarfa Lab)
Description : Vectorised implementation
"""

from sklearn.neighbors import KNeighborsClassifier
import evaluatePipeline

def knEvaluate(addressX, addressY):
    # preparing for optimising
    k_range = list(range(1,31))
    weight_options = ["uniform", "distance"]
    param_grid = dict(n_neighbors = k_range, weights = weight_options)
    return evaluatePipeline.evaluate(addressX, addressY, KNeighborsClassifier(), "kn", parameter_grid = param_grid)
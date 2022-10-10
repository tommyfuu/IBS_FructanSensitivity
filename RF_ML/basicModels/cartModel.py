# -*- coding: utf-8 -*-

"""
Author      : Tom Fu
Date        : 2020 June 29
FileName    : svmModel.py (for Coarfa Lab)
Description : Vectorised implementation
"""



import evaluatePipeline

from sklearn.tree import DecisionTreeClassifier

def cartEvaluate(addressX, addressY):
    criterions = ['gini', 'entropy']
    max_depth = [1, 5, 10, 20, 30]
    param_grid = {'criterion': criterions,
                'max_depth': max_depth,}
    return evaluatePipeline.evaluate(addressX, addressY, DecisionTreeClassifier(), "cart", parameter_grid = param_grid)
# -*- coding: utf-8 -*-

"""
Author      : Tom Fu
Date        : 2020 June 29
FileName    : LogisticModel.py (for Coarfa Lab)
Description : Vectorised implementation
"""

from sklearn.linear_model import LogisticRegression

import evaluatePipeline



def LogisticEvaluate(addressX, addressY):
    # preparing for optimising
    return evaluatePipeline.evaluate(addressX, addressY, LogisticRegression(), "Logistic")
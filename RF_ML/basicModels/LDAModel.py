# -*- coding: utf-8 -*-

"""
Author      : Tom Fu
Date        : 2020 June 29
FileName    : LDAModel.py (for Coarfa Lab)
Description : Vectorised implementation
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import evaluatePipeline



def LDAEvaluate(addressX, addressY):
    # preparing for optimising
    return evaluatePipeline.evaluate(addressX, addressY, LinearDiscriminantAnalysis(), "LDA")
"""
Author      : Tom Fu
Date        : 2020 June 2
FileName    : normaliseBINU.py (for Coarfa Lab)
Description : Transform and normalise input txt files such as EGS_pa
"""

import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
# from scipy.stats import zscore


def formXList(address):
    """
    Intro: Turn txt into a list form dataframe and invert it
    Para: address, a string address of a import file
    Output: a data frame with each row being a list of data for each
    sample/individual
    """
    df = pd.read_csv(address, delimiter='\t')
    Xdf = df.T
    X = Xdf.values.tolist()
    return X


def rowNormalise(row):
    """
    Intro: normalise a row of a dafaframe
    Para: row, a list
    Output: normalisedRow, a normalised list
    """
    normalisedRow = row
    featureName = normalisedRow.pop(0)
    maxRow = max(row)  # the max number of the row
    minRow = min(row)
    rangeRow = maxRow - minRow

    # list comprehension to normalise each element
    normalisedRow = [(num-minRow)/rangeRow for num in normalisedRow]
    normalisedRow.insert(0, featureName)
    return normalisedRow


def BINUlabelDictsWGS(addressY):
    """
    Intro: turn a y address into three dicts where you input the name of the 
            variable to get the index, with which you can search for its 
            optimised ML parameters in either its binary or numeric dict. It
            also returns the 3 datasets matched with relevent datatypes
    Para: addressY, a string address of a import file for y
    Output: binaryLabelDict, a dict for binary variables
            numericLabelDict, a dict for numeric variables
            categoricLabelDict, a dict for categoric variables
            dfBinaryY, a pd dataframe containing all binary variable data for y
            dfCategoricY, a pd dataframe containing all categoric variable data for y
            dfNumericY, a pd dataframe containing all numeric variable data for y
    """
    # initialise
    binary_cols_Y = []
    binaryLabelDict = {}

    # recognise and pick out binary/categorical/numeric dataset
    binary_cols_Y = ['FRUCTANSENSITIVE']

    # get y data
    dfY = pd.read_excel(addressY)
    dfBinaryY = dfY[binary_cols_Y]

    # form labelDicts
    for binaryIndex in range(0, len(binary_cols_Y)):
        binaryLabelDict.update({binary_cols_Y[binaryIndex]: binaryIndex})

    return binaryLabelDict, dfBinaryY


def sampleDropper(X, currentY):
    """
    Intro: for a specific variable, drop samples in that y that is nan and make a respective copyY that doesn't have that sample
    Para: X, normalised X sample
          currentY, all y data belonging to the current variable
    Output: realX, realCurrentY, which are X and currentY with nan values dropped
    """
    # drop for y
    nanIndices = [i for i, v in enumerate(currentY) if np.isnan(v)]
    realCurrentY = np.delete(currentY, nanIndices, axis=0)
    realCurrentX = np.delete(X, nanIndices, axis=0)

    return realCurrentX, realCurrentY


def XGetter(addressX, returnSampleNames=False):
    # get X
    df = pd.read_csv(addressX, delimiter='\t')
    df1 = df.iloc[:, list(range(1, len(df.columns)))]
    featureList = list(df.iloc[:, 0])
    columnNames = list(df1.columns)
    df1Array = df1.T.values
    scaler = StandardScaler()
    scaler.fit(df1Array)
    X = scaler.transform(df1Array)
    if returnSampleNames == False:
        return X, featureList
    else:
        return X, featureList, columnNames


def sampleDropperAll(X, currentYType):
    nanIndices2D = np.argwhere(np.isnan(currentYType))
    nanIndices = [sample[0] for sample in nanIndices2D]
    nanIndices = list(dict.fromkeys(nanIndices))
    realCurrentTypeY = np.delete(currentYType, nanIndices, axis=0)
    realCurrentX = np.delete(X, nanIndices, axis=0)
    return realCurrentX, realCurrentTypeY


################## overarching function ##################

def BINUnormalise(addressX, addressY):
    """
    Intro: normalise a dafaframe, a 2D list
    Para: addressX and addressY, two address corresponding to X and Y
    Output: normalised X, formatted y, and the respect label dicts
    """
    # get X
    X, allFeatureList = XGetter(addressX)
    # get Y and label dicts
    binaryLabelDict, dfBinaryY = BINUlabelDictsWGS(
        addressY)

    # for binary, change yes to 1 no to 0
    # mapping = {"Yes": 1, "No": 0}
    dfBinaryY = dfBinaryY.applymap(
        lambda v: 0 if v == "No" else (1 if v == "Yes" else v))

    # binaryY = dfBinaryY
    binaryY = np.array(dfBinaryY)
    # fill out missing values (TODO: MAY NEED TO DELETE THIS LATER)
    # binaryY = np.where(pd.isnull(binaryY), 0, binaryY)

    return X, binaryY, binaryLabelDict, allFeatureList

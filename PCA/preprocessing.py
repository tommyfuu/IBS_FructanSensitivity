"""
Author      : Tom Fu
Date        : 2020 June 2
FileName    : normaliseBINU.py (for Coarfa Lab)
Description : Transform and normalise input txt files such as EGS_pa
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


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




def XGetter(addressX, returnSampleNames=True):
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



################## overarching function ##################

def BINUnormalise(addressX, addressY):
    """
    Intro: normalise a dafaframe, a 2D list
    Para: addressX and addressY, two address corresponding to X and Y
    Output: normalised X, formatted y, and the respect label dicts
    """
    # get X
    X, allFeatureList, sampleNames = XGetter(addressX, returnSampleNames=True)
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

# -*- coding: utf-8 -*-

"""
Author      : Tom Fu
Date        : 2020 June 29
FileName    : rfModel.py (for Coarfa Lab)
Description : Vectorised implementation
"""

import pickle
import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
import plotly.graph_objects as go
import os

import time
import preprocessing

################################## global variables ##################################

VARIABLENAMES = ['FRUCTANSENSITIVE']

################################## check model ##################################


def evaluatePipelineRepeated(addressX, addressY, omicName, outputRoot):
    print("DULU")

    if not os.path.exists('./outputsRF/' + outputRoot):
        os.makedirs('./outputsRF/' + outputRoot)

    # initialise
    numOfRepetitions = 100

    # normalise and arrayise
    X, binaryY, binaryLabelDict, allFeatureList = preprocessing.BINUnormalise(
        addressX, addressY)

    # Create the random grid to prep for gridSearchCV
    criterions = ['gini', 'entropy']
    n_estimators = [10, 50, 100, 500, 1000]
    max_depth = [1, 5, 10, 20, 30]

    parameter_grid = {'criterion': criterions,
                      'n_estimators': n_estimators,
                      'max_depth': max_depth,
                      }
    # paramGrid = ParameterGrid(parameter_grid)
    # prepare to optimise w gridSearchCV to get the best model
    grid_search = GridSearchCV(estimator=RandomForestClassifier(
    ), param_grid=parameter_grid, cv=3, n_jobs=-1, verbose=2)
    currentOmicResults = {'fpr': [], 'tpr': [], 'auc': []}

    # split, optimise to find the best parameters
    numOfYVars = len(binaryY[0])  # will be equal to the length of RFParaList
    for index in range(0, numOfYVars):
        print("4omics")
        # get the current Y
        currentY = [eachY[index] for eachY in binaryY]
        # variableName
        currentVarName = VARIABLENAMES[index]
        print(currentVarName)
        # drop out missing values
        currentRealX, currentRealY = preprocessing.sampleDropper(X, currentY)
        print(len(currentRealX))
        print(len(currentRealX[0]))
        print(currentRealY)

        # initialise things
        counter = 0
        F1List = []
        AccuracyScoreList = []
        AUCList = []

        finalFeatureDict = {}

        aovOutputWriter = open('./outputsRF/'+ outputRoot+'/'+'RepeatedNOAUCrfPerformance'+outputRoot+'-'+str(numOfRepetitions)+'.csv', 'wt')
        print("AAAA, AAAA")
        aovOutputWriter.write(',F1 scores,Accuracy Scores,AUC Scores'+'\n')
        print('BBBB')
        # repeat 100 times to test robustness
        for repeat in range(0, numOfRepetitions):
            # count
            counter += 1
            print(counter)

            # also optimise by feature selection - boruta - repeated Boruta
            feat_selector = BorutaPy(
                estimator=RandomForestClassifier(), max_iter=50, verbose=2)
            feat_selector.fit(currentRealX, currentRealY)
            X_filtered = feat_selector.transform(currentRealX)

            X_filteredOrderedL = X_filtered.T.tolist()
            currentRealXOrderedL = currentRealX.T.tolist()
            for featureIndex in range(len(X_filteredOrderedL)):
                if X_filteredOrderedL[featureIndex] in currentRealXOrderedL:
                    featureOGIndex = currentRealXOrderedL.index(
                        X_filteredOrderedL[featureIndex])
                    featureName = allFeatureList[featureOGIndex]
                    if featureName not in finalFeatureDict:
                        finalFeatureDict[featureName] = 1
                    else:
                        finalFeatureDict[featureName] = finalFeatureDict[featureName]+1
            print(finalFeatureDict)

            if len(X_filtered[0]) == 0:
                print("All features are not relevant to the current label")
                X_filtered = X
            # split
            trainX, testX, trainYs, testYs = train_test_split(
                X_filtered, currentRealY, test_size=0.3)
            # fit
            grid_search.fit(trainX, trainYs)
            currentModel = grid_search.best_estimator_
            currentPara = grid_search.best_params_
            # predict
            currentPredictions = currentModel.predict(testX)
            print(testYs)
            print(currentPredictions)

            # save model
            currentRFFileName = './outputsRF/'+ outputRoot+'/'+'RFModel' + currentVarName + omicName + str(repeat)+'.pkl'
            if repeat % 3 == 0:
                with open(currentRFFileName, 'wb') as f:
                    pickle.dump(currentModel, f)

            # metrics
            fpr, tpr, thresholds = roc_curve(testYs, currentPredictions)
            currentOmicResults['fpr'].append(fpr)
            currentOmicResults['tpr'].append(tpr)

            ## compare and evaluate
            # AUC scores
            try:
                currentAUCscore = roc_auc_score(testYs, currentPredictions)
                AUCList.append(currentAUCscore)
                print(currentAUCscore)
            except ValueError:
                currentAUCscore = 0
                print("currentAUCscore1 = 0")
                pass
            currentOmicResults['auc'].append(currentAUCscore)
            # F1 scores
            currentF1Score = f1_score(
                testYs, currentPredictions, average="weighted")
            F1List.append(currentF1Score)
            print(currentF1Score)
            # accuracy scores
            currentAccuracyScore = accuracy_score(testYs, currentPredictions)
            AccuracyScoreList.append(currentAccuracyScore)
            print(currentAccuracyScore)
            usefulLineContent = [str(repeat), str(currentF1Score), str(currentAccuracyScore), str(currentAUCscore)]
            lineWritten = ','.join(usefulLineContent)
            aovOutputWriter.write(lineWritten+'\n')

            print('current median AUC', np.median(AUCList))
        # plot histograms
        # stacking plots
        fig, axs = plt.subplots(3)

        #fig.suptitle('Performance Consistency - binary '+str(index), fontsize=16)
        # F1 scores
        axs[0].hist(F1List)
        axs[0].set_title('F1 scores')

        # accuracy scores
        axs[1].hist(AccuracyScoreList)
        axs[1].set_title('Accuracy scores')

        # AUC scores
        axs[2].hist(AUCList)
        axs[2].set_title('AUC scores')

        # show
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        fig.text(.5, 0.06, 'Performance Consistency - binary' +
                 str(index+1), fontsize=18, ha='center')
        
        plt.savefig('./outputsRF/'+ outputRoot+'/'+'RepeatedNOAUChistogramsRF' +
                    outputRoot + ".png")
        # plt.show(block=False)
        # plt.pause(3)
        plt.close()
        
        # form pds and csvs
        with open("./outputsRF/"+ outputRoot+'/'+outputRoot+'.rocDict_small.pkl', 'wb') as f:
            pickle.dump(currentOmicResults,f, pickle.HIGHEST_PROTOCOL)
        dictData = {'F1 scores': pd.Series(F1List),
                    'Accuracy Scores': pd.Series(AccuracyScoreList), 'AUC Scores': pd.Series(AUCList)}
        performanceDF1 = pd.DataFrame(dictData)
        performanceDF1.to_csv(
            './outputsRF/'+ outputRoot+'/'+'RepeatedNOAUCrfPerformance'+outputRoot+'-'+str(numOfRepetitions)+'.csv')
        finalFeaturesPD = pd.DataFrame(finalFeatureDict.items())
        finalFeaturesPD.to_csv(
            './outputsRF/'+ outputRoot+'/'+'RepeatedrfFeatures'+outputRoot+'-'+str(numOfRepetitions)+'.csv')
    else:
        # heterogeneous enough, we are good
        print("heterogeneous enough, we are good")

    aovOutputWriter.close()
    return



################################## form roc plot ##################################

c_fill_L = ['rgba(255, 51, 255, 0.1)',
            'rgba(51, 153, 255, 0.1)', 'rgba(102, 204, 0, 0.1)', 'rgba(153, 153, 0, 0.15)', 'rgba(255, 0, 0, 0.1)', 'rgba(100, 100, 100, 0.1)', 'rgba(0, 100, 200, 0.1)']
c_line_L = ['rgba(255, 51, 255, 0.25)',
            'rgba(51, 153, 255, 0.25)', 'rgba(102, 204, 0, 0.25)', 'rgba(153, 153, 0, 0.25)', 'rgba(255, 0, 0, 0.25)', 'rgba(100, 100, 100, 0.25)', 'rgba(0, 100, 200, 0.25)']
c_mainLine_L = ['rgba(255, 51, 255, 1)',
                'rgba(51, 153, 255, 1)', 'rgba(102, 204, 0, 1)', 'rgba(153, 153, 0, 1)', 'rgba(255, 0, 0, 1)', 'rgba(100, 100, 100, 1)', 'rgba(0, 100, 200, 1)']


def evaluatePipelineROC(dictAddressL, omicNameL, outputRoot):
    # plotting
    goList = []
    altGoList = []
    c_grid = 'rgba(189, 195, 199, 0.5)'
    # c_annot = 'rgba(149, 165, 166, 0.5)'
    # c_highlight = 'rgba(192, 57, 43, 1.0)'
    fpr_mean = np.linspace(0, 1, 10)
    interp_tprs = []

    for omic_index in range(len(dictAddressL)):
        dictFile = open(dictAddressL[omic_index], "rb")
        loaded_dictionary = pickle.load(dictFile)
    
        # for omic_index in range(len(addressXL)):
        c_fill = c_fill_L[omic_index]
        c_line = c_line_L[omic_index]
        c_line_main = c_mainLine_L[omic_index]
        omicName = omicNameL[omic_index]
        for i in range(40):
            fpr = loaded_dictionary['fpr'][i]
            tpr = loaded_dictionary['tpr'][i]
            interp_tpr = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
        tpr_mean = np.median(interp_tprs, axis=0)
        print("current", omicName, tpr_mean)
        tpr_mean[-1] = 1.0
        tpr_std = 2*np.std(interp_tprs, axis=0)
        tpr_upper = np.clip(tpr_mean+tpr_std, 0, 1)
        tpr_lower = tpr_mean-tpr_std
        auc = np.median(loaded_dictionary['auc'])
        goList.extend([
            go.Scatter(
                    x=fpr_mean,
                    y=tpr_upper,
                    line=dict(color=c_line, width=1),
                    hoverinfo="skip",
                    showlegend=False, 
                    name='upper'),
                go.Scatter(
                    x=fpr_mean,
                    y=tpr_lower,
                    fill='tonexty',
                    fillcolor=c_fill,
                    line=dict(color=c_line, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name='lower'),
            go.Scatter(
                x=fpr_mean,
                y=tpr_mean,
                line=dict(color=c_line_main, width=2),
                hoverinfo="skip",
                showlegend=True,
                name=f'{omicNameL[omic_index]} AUC: {auc:.3f}'
                # name=f'{omicNameL[omic_index]} AUC: {auc:.3f}'
                )
        ])
        
    fig = go.Figure(goList)
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        xaxis_title="1 - Specificity",
        yaxis_title="Sensitivity",
        width=800,
        height=800,
        legend=dict(
            yanchor="bottom",
            xanchor="right",
            x=0.95,
            y=0.01,
        )
    )
    fig.update_yaxes(
        range=[0, 1],
        gridcolor=c_grid,
        scaleanchor="x",
        scaleratio=1,
        linecolor='black')
    fig.update_xaxes(
        range=[0, 1],
        gridcolor=c_grid,
        constrain='domain',
        linecolor='black')
    
    fig.write_image(outputRoot+"1.png")
    return





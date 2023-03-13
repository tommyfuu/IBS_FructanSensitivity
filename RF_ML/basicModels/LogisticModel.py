# -*- coding: utf-8 -*-

"""
Author      : Tom Fu
Date        : 2020 June 29
FileName    : LogisticModel.py (for Coarfa Lab)
Description : Vectorised implementation
"""

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import time

import evaluatePipeline
from evaluatePipeline import *



# def LogisticEvaluate(addressX, addressY):
#     # preparing for optimising
#     return evaluatePipeline.evaluate(addressX, addressY, LogisticRegression(), "Logistic")
def evaluate(addressX, addressY, model, modelName, parameter_grid = []):
    # initialise
    numOfRepetitions = 100
    gridBool = False
    # normalise and arrayise
    X, binaryY, binaryLabelDict, allFeatureList = preprocessing.BINUnormalise(addressX, addressY)
    
    # print(len(allFeatureList))
    #### prepare to optimise w gridSearchCV to get the best model
    if parameter_grid != []:
        grid_search = GridSearchCV(estimator = model, param_grid = parameter_grid, cv = 3, n_jobs = -1, verbose = 2)
        gridBool = True
    else: grid_search = model
    
    # split, optimise to find the best parameters 
    numOfYVars = len(binaryY[0]) # will be equal to the length of RFParaList
    for index in range(0, numOfYVars):
        ## get the current Y
        currentY = [eachY[index] for eachY in binaryY]
        ## variableName
        currentVarName = VARIABLENAMES[index]
        ## drop out missing values
        currentRealX, currentRealY = preprocessing.sampleDropper(X, currentY)
        ## check homogeneity
        currentRealYList = currentRealY.tolist()
        zeroRatio = (currentRealYList.count(0))/len(currentRealYList)
        
        counter = 0
        F1List = []
        AccuracyScoreList = []
        AUCList = []

        all_feature_abs_coef_l = []
        ### feature selector
        # currentRealX = SelectPercentile(chi2, percentile=25).fit_transform(currentRealX, currentRealY)
        #### repeat 100 times to test robustness
        for repeat in range(0, numOfRepetitions):
            ## count
            counter +=1 
            print(counter)
            ## split
            trainX, testX, trainYs, testYs = train_test_split(currentRealX, currentRealY, test_size = 0.3)
            ## fit
            grid_search.fit(trainX, trainYs)
            if gridBool == True:
                currentModel = grid_search.best_estimator_
                currentLegend = grid_search.best_params_
            else: 
                currentModel = grid_search
                currentLegend = "No necessary parameters needed"
            ## get important features
            feature_coefs = currentModel.coef_[0]
            print("AA", len(feature_coefs))
            abs_feature_coefs = list(np.abs(feature_coefs))
            all_feature_abs_coef_l.append(abs_feature_coefs)
            # # largest 10 features
            # ind = np.argpartition(abs_feature_coefs, -10)[-10:]
            # print("largest features", np.array(allFeatureList)[ind])
            # print("largest feature values", abs_feature_coefs[ind])
            # time.sleep(10)

            ## predict

            currentPredictions = currentModel.predict(testX)
            print(testYs)
            print(currentPredictions)
            ## compare and evaluate
            ### AUC scores
            try:
                currentAUCscore = roc_auc_score(testYs, currentPredictions)
                #### make sure there is no value smaller than 0.5
                # if currentAUCscore < 0.5:
                #     for prediction in currentPredictions:
                #         if prediction == 0: prediction = 1
                #         else: prediction = 0
                currentAUCscore = roc_auc_score(testYs, currentPredictions)
                print(currentAUCscore)
            except ValueError:
                currentAUCscore = 0
                print("currentAUCscore1 = 0")
                pass
            AUCList.append(currentAUCscore)
            ### F1 scores
            currentF1Score = f1_score(testYs, currentPredictions, average="weighted")
            F1List.append(currentF1Score)
            print(currentF1Score)
            ### accuracy scores
            currentAccuracyScore = accuracy_score(testYs, currentPredictions)
            AccuracyScoreList.append(currentAccuracyScore)
            print(currentAccuracyScore)


        # plot histograms
        # stacking plots
        fig, axs = plt.subplots(3)
        
        #fig.suptitle('Performance Consistency - binary '+str(index), fontsize=16)
        ## F1 scores
        axs[0].hist(F1List)
        axs[0].set_title('F1 scores')

        ## accuracy scores
        axs[1].hist(AccuracyScoreList)
        axs[1].set_title('Accuracy scores')

        # AUC scores
        axs[2].hist(AUCList)
        axs[2].set_title('AUC scores')

        ## show
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        fig.text(.5,0.06,'Performance Consistency - binary'+str(index+1), fontsize=18, ha='center')
        fig.text(.5,0.02,'Parameters: '+str(currentLegend),fontsize=10,ha='center')

        plt.savefig('./outputs' + modelName + '/testNoAUChistogramsSel' + str(index+1) + ".png")
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        # form pds and csvs
        dictData = {'F1 scores': F1List, 'Accuracy Scores': AccuracyScoreList, 'AUC Scores': AUCList}
        performanceDF1 = pd.DataFrame(dictData)
        performanceDF1.to_csv('./outputs' + modelName + '/testNOAUCSelPerformance'+str(index+1)+'-'+str(numOfRepetitions)+'.csv')
        
        all_feature_abs_coef_l = np.array(all_feature_abs_coef_l)
        print("AAA", all_feature_abs_coef_l.shape)
        median_feature_abs_coef_l = np.median(all_feature_abs_coef_l, axis=0)
        print("AAAAA", median_feature_abs_coef_l.shape)

        # largest 10 features
        ind = np.argpartition(median_feature_abs_coef_l, -100)[-100:]
        ind = ind[np.argsort(median_feature_abs_coef_l[ind])]
        ind = np.flip(ind)
        print("largest features", np.array(allFeatureList)[ind])
        print("largest feature values", median_feature_abs_coef_l[ind])
        top_100_features_df = pd.DataFrame({"top 100 features": np.array(allFeatureList)[ind], 
                    "top 100 feature values": median_feature_abs_coef_l[ind]})
        top_100_features_df.to_csv('./outputs' + modelName + '/topfeatureImportance'+str(index+1)+'-'+str(numOfRepetitions)+'.csv')
    return

def LogisticEvaluate(addressX, addressY, outputRoot):
    # preparing for optimising
    return evaluate(addressX, addressY, LogisticRegression(), "Logistic_"+outputRoot)


# 4omicsall
addressX = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/4omicsAllX.txt'
addressY = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/AllDietsY.xlsx'
LogisticEvaluate(addressX, addressY, "4omicsAll")

# 4omicA
addressX = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/dietSubsets/1009_A.txt'
addressY = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/dietSubsets/1009_A_y.xlsx'
LogisticEvaluate(addressX, addressY, "4omicsA")

# 4omicsB
addressX = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/dietSubsets/1009_B.txt'
addressY = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/dietSubsets/1009_B_y.xlsx'
LogisticEvaluate(addressX, addressY, "4omicsB")

# 4omicsBL
addressX = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/dietSubsets/1009_BL.txt'
addressY = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/dietSubsets/1009_BL_y.xlsx'
LogisticEvaluate(addressX, addressY, "4omicsBL")

# 4omicsAAndB
addressX = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/dietSubsets/1009_A_B.txt'
addressY = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/dietSubsets/1009_A_B_y.xlsx'
LogisticEvaluate(addressX, addressY, "4omicsAAndB")

# 4omicsAAndBL
addressX = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/dietSubsets/1009_A_BL.txt'
addressY = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/dietSubsets/1009_A_BL_y.xlsx'
LogisticEvaluate(addressX, addressY, "4omicsAAndBL")

# 4omicsBAndBL
addressX = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/dietSubsets/1009_B_BL.txt'
addressY = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/dietSubsets/1009_B_BL_y.xlsx'
LogisticEvaluate(addressX, addressY, "4omicsBAndBL")

# Humann3
addressX = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/1011humann3X.txt'
addressY = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/AllDietsY.xlsx'
LogisticEvaluate(addressX, addressY, "humann3")

# metabolites
addressX = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/1011MetabolitesX.txt'
addressY = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/AllDietsY.xlsx'
LogisticEvaluate(addressX, addressY, "metabolites")

# metaphlan
addressX = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/1011MetaphlanX.txt'
addressY = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/AllDietsY.xlsx'
LogisticEvaluate(addressX, addressY, "metaphlan")

# lipidomics
addressX = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/1011LipidomicsX.txt'
addressY = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/AllDietsY.xlsx'
LogisticEvaluate(addressX, addressY, "lipidomics")


logistic_4omicsAll = pd.read_csv("/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/RF_ML/basicModels/outputsLogistic_4omicsAll/testNOAUCSelPerformance1-100.csv")
logistic_4omicsA = pd.read_csv("/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/RF_ML/basicModels/outputsLogistic_4omicsA/testNOAUCSelPerformance1-100.csv")
logistic_4omicsB = pd.read_csv("/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/RF_ML/basicModels/outputsLogistic_4omicsB/testNOAUCSelPerformance1-100.csv")
logistic_4omicsBL = pd.read_csv("/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/RF_ML/basicModels/outputsLogistic_4omicsBL/testNOAUCSelPerformance1-100.csv")
logistic_4omicsAAndB = pd.read_csv("/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/RF_ML/basicModels/outputsLogistic_4omicsAAndB/testNOAUCSelPerformance1-100.csv")
logistic_4omicsAAndBL = pd.read_csv("/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/RF_ML/basicModels/outputsLogistic_4omicsAAndBL/testNOAUCSelPerformance1-100.csv")
logistic_4omicsBAndBL = pd.read_csv("/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/RF_ML/basicModels/outputsLogistic_4omicsBAndBL/testNOAUCSelPerformance1-100.csv")
logistic_humann3 = pd.read_csv("/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/RF_ML/basicModels/outputsLogistic_humann3/testNOAUCSelPerformance1-100.csv")
logistic_lipidomics = pd.read_csv("/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/RF_ML/basicModels/outputsLogistic_lipidomics/testNOAUCSelPerformance1-100.csv")
logistic_metabolites = pd.read_csv("/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/RF_ML/basicModels/outputsLogistic_metabolites/testNOAUCSelPerformance1-100.csv")
logistic_metaphlan = pd.read_csv("/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/RF_ML/basicModels/outputsLogistic_metaphlan/testNOAUCSelPerformance1-100.csv")

print("logistic_4omicsAll")
print(np.median(logistic_4omicsAll['AUC Scores']))
print("logistic_4omicsA")
print(np.median(logistic_4omicsA['AUC Scores']))
print("logistic_4omicsB")
print(np.median(logistic_4omicsB['AUC Scores']))
print("logistic_4omicsBL")
print(np.median(logistic_4omicsBL['AUC Scores']))
print("logistic_4omicsAAndB")
print(np.median(logistic_4omicsAAndB['AUC Scores']))
print("logistic_4omicsAAndBL")
print(np.median(logistic_4omicsAAndBL['AUC Scores']))
print("logistic_4omicsBAndBL")
print(np.median(logistic_4omicsBAndBL['AUC Scores']))
print("logistic_metaphlan")
print(np.median(logistic_metaphlan['AUC Scores']))
print("logistic_humann3")
print(np.median(logistic_humann3['AUC Scores']))
print("logistic_metabolites")
print(np.median(logistic_metabolites['AUC Scores']))
print("logistic_lipidomics")
print(np.median(logistic_lipidomics['AUC Scores']))
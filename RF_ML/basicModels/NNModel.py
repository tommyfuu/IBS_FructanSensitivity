# -*- coding: utf-8 -*-

"""
Author      : Tom Fu
Date        : 2020 June 29
FileName    : NNModel.py (for Coarfa Lab)
Description : Vectorised implementation
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

import preprocessing

VARIABLENAMES = ['Depression_Remission', 'Anxiety_Remission', 'DepOutMod', 'AnxOutMod', 'SSRI',
                    'SNRI', 'ANTIDEPRESSANTOTHER', 'MOODSTABILIZER_OR_ANTICONVULSANT', 'ATYPICALANTIPSYCHOTIC', 
                    'BENZODIAZEPINE', 'ALTERNATIVE_OR_COMPLEMENTARY']

# baseline model
def create_baseline():
	# create model
	model = Sequential([
        Dense(100, input_dim=10403, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(50, activation='relu'),
        Dense(1, activation='sigmoid')])
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def evaluateNewNN(addressX, addressY):
    counter = 0
    numOfRepetitions = 100
    # preprocessing and arrayise
    X, binaryY, categoricalY, numericY, binaryLabelDict, categoricLabelDict, numericLabelDict, allFeatureList = preprocessing.BINUnormalise(addressX, addressY)
    print(X)
    # split, optimise to find the best parameters 
    for index in [2, 3, 5, 8]:
        ## get the current Y
        currentY = [eachY[index] for eachY in binaryY]
        ## drop out missing values
        currentRealX, currentRealY = preprocessing.sampleDropper(X, currentY)
        
        ## get the current model
        currentModel = KerasClassifier(build_fn=create_baseline, epochs=50, batch_size=15, verbose=2)
        F1List = []
        AccuracyScoreList = []
        AUCList = []
        
        # loop through to produce histograms and csvs
        for repeat in range(0, numOfRepetitions):
            counter +=1
            print(counter)
            ## split
            trainX, testX, trainYs, testYs = train_test_split(currentRealX, currentRealY, test_size = 0.3)
            print(testYs)
            ## fit model
            currentModel.fit(trainX, trainYs)
            # get predictions
            test_predictions1 = currentModel.predict(testX)
            test_predictions1 = [prediction[0] for prediction in test_predictions1]
            print(test_predictions1)
            print(type(test_predictions1))
            ## calculate performance measures
            ### F1 scores
            currentF1Score1 = f1_score(testYs, test_predictions1, average="weighted")
            print(currentF1Score1)
            ### accuracy scores
            currentAccuracyScore1 = accuracy_score(test_predictions1, testYs)
            print(currentAccuracyScore1)
            ### AUC scores
            currentAUCscore1 = roc_auc_score(test_predictions1, testYs)
            print(currentAUCscore1)
            # store data
            F1List.append(currentF1Score1)
            AccuracyScoreList.append(currentAccuracyScore1)
            AUCList.append(currentAUCscore1)
    
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
        fig.text(.5,0.02,'keras classifer' ,fontsize=10,ha='center')
        # plt.title('Parameters: '+str(currentPara), fontsize=16)
        plt.savefig('./outputsNN/histogramsNNBinary' + str(index+1) + ".png")
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        # form pds and csvs
        dictData = {'F1 scores': F1List, 'Accuracy Scores': AccuracyScoreList, 'AUC Scores': AUCList}
        performanceDF1 = pd.DataFrame(dictData)
        performanceDF1.to_csv('./outputsNN/NNPerformanceModel-'+str(index+1)+'-'+str(numOfRepetitions)+'.csv')

    return
    
            
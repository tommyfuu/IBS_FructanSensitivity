## AUTHOR: CHENLIAN FU
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import preprocessing
import matplotlib.pyplot as plt


def plotPCA(ZMatrix, binaryY, outputRoot, targets=['A', 'B', 'BS'], colors=['r', 'g', 'b']):
    '''
    main function generating PCA plots
    '''
    # run PCA on ZMatrix
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(ZMatrix)
    principalComponents = principalComponents.tolist()

    print(pca.explained_variance_ratio_)
    finalPrincipalComponents = []
    finalBinaryYL = []

    for i in range(len(principalComponents)):
        principalComponents[i].append(binaryY[i])
    finalDf = pd.DataFrame(data=principalComponents, columns=[
        'principal component 1', 'principal component 2', 'target'])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title(outputRoot+'2 component PCA', fontsize=20)

    for target, color in zip(targets, colors):
        # print("Trying")
        # print(target, color)
        indicesToKeep = finalDf['target'] == target
        # print(indicesToKeep)
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'], c=color, s=50)
    ax.legend(targets)
    if outputRoot=='4omicsPatientIDPCA':
        ax.legend(targets, loc=(1.04,0))
        fig.set_size_inches(15, 10.5)
    ax.grid()

    fig.savefig(outputRoot+".png")
    fig.savefig(outputRoot+".pdf")
    return


def datasetPrep(Xfile, Yfile, columnValue='Diet'):
    X, binaryY, _, _ = preprocessing.BINUnormalise(Xfile)

    # get y data
    dfY = pd.read_excel(Yfile)
    dfBinaryY = dfY[columnValue]
    binaryY = np.array(dfBinaryY)
    return X, binaryY


def completePCA(Xfile, Yfile, outputRoot, columnValue='Diet', targets=['A', 'B', 'BS'], colors=['r', 'g', 'b']):
    X, binaryY = datasetPrep(Xfile, Yfile, columnValue)
    print(outputRoot)
    plotPCA(X, binaryY, outputRoot, targets, colors)


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

# to reproduce our results
patientIDs = [3003,
              3004,
              3005,
              3007,
              3008,
              3009,
              3012,
              3013,
              3014,
              3016,
              3020,
              3021,
              3022,
              3023,
              3025,
              3026,
              3027,
              3028,
              3029,
              3030,
              3031,
              3032,
              3033,
              3036,
              3037,
              3039,
              3042,
              3043,
              3044,
              3046,
              3047,
              3048,
              3049]

patientID_colorL = ["black",
                    "dimgray", "bisque", "darkorange", "tan", "forestgreen", "limegreen", "lime", "darkslategray", "aqua", "darkturquoise", "lightslategray",
                    "lightsteelblue", "cornflowerblue", "midnightblue", "indigo", "darkorchid",  "thistle", "plum", "magenta", "hotpink", "crimson",
                    "lightpink", "slategray", "dodgerblue", "aliceblue", "lightgreen", "honeydew", "darkolivegreen", "yellow", "olive", "beige", "darkkhaki"]


allOmicsX = './source/4omicsAllX.txt'
humann3Y = './source/4omicsYFinal.xlsx'
humann3YW = './source/fructanYWDiet0525.xlsx'

completePCA(allOmicsX, humann3YW, outputRoot='4omicsDietsPCA', columnValue='Diet',
            targets=['A', 'B', 'BS'], colors=['r', 'g', 'b'])
completePCA(allOmicsX, humann3YW, outputRoot='4omicsFructanPCA', columnValue='FRUCTANSENSITIVE',
            targets=[0, 1], colors=['r', 'g'])
completePCA(allOmicsX, humann3YW, outputRoot='4omicsPatientIDPCA', columnValue='Patient_ID',
            targets=patientIDs, colors=patientID_colorL)
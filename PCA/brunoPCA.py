## AUTHOR: CHENLIAN FU
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway, ttest_ind

def plotPCA(ZMatrix, binaryY, outputRoot, targets=['A', 'B', 'BS'], colors=['r', 'g', 'b'], figsize_input = (8,8), limit=False):
    '''
    main function generating PCA plots
    '''
    # run PCA on ZMatrix
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(ZMatrix)
    principalComponents = principalComponents.tolist()
    
    print(pca.explained_variance_ratio_)

    for i in range(len(principalComponents)):
        principalComponents[i].append(binaryY[i])
    finalDf = pd.DataFrame(data=principalComponents, columns=[
        'principal component 1', 'principal component 2', 'target'])

    fig = plt.figure(figsize=figsize_input)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title(outputRoot+'2 component PCA', fontsize=20)
    if limit:
        # ax.set_xlim(left=-50, right=50)
        # ax.set_ylim(bottom=-50, top=50)
        ax.set_xlim(left=-10, right=10)
        ax.set_ylim(bottom=-10, top=10)

    for target, color in zip(targets, colors):
        # print("Trying")
        # print(target, color)
        indicesToKeep = finalDf['target'] == target
        # print(indicesToKeep)
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'], c=color, s=50)
    print("finalDf")
    # print(finalDf)
    ax.legend(targets, loc="right")

    file = open(outputRoot+"_statistical_tests.txt", "w")
    for colname in ["principal component 1", "principal component 2"]:
        values = list(finalDf[colname])
        targets = list(finalDf["target"])

        all_values = []

        for unique_target in np.unique(list(finalDf['target'])):
            current_vals = []
            for idx, target in enumerate(targets):
                current_vals.append(values[idx])
            all_values.append(current_vals)
        # determine the test used
        if len(np.unique(list(finalDf['target']))) == 3:
            # use anova
            print("ANOVAAA")
            print(f_oneway(*all_values))
            file.write("test type is ANOVA")
            file.write(str(f_oneway(*all_values)) + '\n')
        elif len(np.unique(list(finalDf['target']))) == 2:
            # use student t test
            print("ttestAAA")
            print(ttest_ind(*all_values))
            file.write("test type is t test")
            file.write(str(ttest_ind(*all_values)) + '\n')
        
    if '4omicsPatientIDPCA' in outputRoot:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        ax.legend(targets, loc='center right', bbox_to_anchor=(1.15, 0.5))
        fig.set_size_inches(18, 10.5)
    ax.grid()

    fig.savefig(outputRoot+".png")
    fig.savefig(outputRoot+".pdf")
    return


def datasetPrep(Xfile, Yfile, columnValue='Diet'):
    X, binaryY, _, _ = preprocessing.BINUnormalise(Xfile, Yfile)

    # get y data
    dfY = pd.read_excel(Yfile)
    dfBinaryY = dfY[columnValue]
    binaryY = np.array(dfBinaryY)
    return X, binaryY

def ecdf_values(x):
    """
    Generate values for empirical cumulative distribution function
    
    Params
    --------
        x (array or list of numeric values): distribution for ECDF
    
    Returns
    --------
        x (array): x values
        y (array): percentile values
    """
    
    # Sort values and find length
    x = np.sort(x)
    n = len(x)
    # Create percentiles
    y = np.arange(1, n + 1, 1) / n
    return x, y

def ecdf_plot(x, name = 'Value', plot_normal = False, log_scale=False, save=False, save_name='Default'):
    """
    ECDF plot of x
    Params
    --------
        x (array or list of numerics): distribution for ECDF
        name (str): name of the distribution, used for labeling
        plot_normal (bool): plot the normal distribution (from mean and std of data)
        log_scale (bool): transform the scale to logarithmic
        save (bool) : save/export plot
        save_name (str) : filename to save the plot
    
    Returns
    --------
        none, displays plot
    
    """
    xs, ys = ecdf_values(x)
    fig = plt.figure(figsize = (10, 6))
    ax = plt.subplot(1, 1, 1)
    plt.step(xs, ys, linewidth = 2.5, c= 'b');
    
    plot_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    fig_sizex = fig.get_size_inches()[0]
    data_inch = plot_range / fig_sizex
    right = 0.6 * data_inch + max(xs)
    gap = right - max(xs)
    left = min(xs) - gap
    
    if log_scale:
        ax.set_xscale('log')
        
    if plot_normal:
        gxs, gys = ecdf_values(np.random.normal(loc = xs.mean(), 
                                                scale = xs.std(), 
                                                size = 100000))
        plt.plot(gxs, gys, 'g');

    plt.vlines(x=min(xs), 
               ymin=0, 
               ymax=min(ys), 
               color = 'b', 
               linewidth = 2.5)
    
    # Add ticks
    plt.xticks(size = 16)
    plt.yticks(size = 16)
    # Add Labels
    plt.xlabel(f'{name}', size = 18)
    plt.ylabel('Percentile', size = 18)

    plt.vlines(x=min(xs), 
               ymin = min(ys), 
               ymax=0.065, 
               color = 'r', 
               linestyle = '-', 
               alpha = 0.8, 
               linewidth = 1.7)
    
    plt.vlines(x=max(xs), 
               ymin=0.935, 
               ymax=max(ys), 
               color = 'r', 
               linestyle = '-', 
               alpha = 0.8, 
               linewidth = 1.7)

    # Add Annotations
    
    ps = [0.25, 0.5, 0.75]

    for p in ps:

        ax.set_xlim(left = left, right = right)
        ax.set_ylim(bottom = 0)

        value = xs[np.where(ys > p)[0][0] - 1]
        pvalue = ys[np.where(ys > p)[0][0] - 1]

        plt.hlines(y=p, xmin=left, xmax = value,
                    linestyles = ':', colors = 'r', linewidth = 1.4);

        plt.vlines(x=value, ymin=0, ymax = pvalue, 
                   linestyles = ':', colors = 'r', linewidth = 1.4)
        
        plt.text(x = p / 3, y = p - 0.01, 
                 transform = ax.transAxes,
                 s = f'{int(100*p)}%', size = 15,
                 color = 'r', alpha = 0.7)

        plt.text(x = value, y = 0.01, size = 15,
                 horizontalalignment = 'left',
                 s = f'{value:.2f}', color = 'r', alpha = 0.8);

    # fit the labels into the figure
    plt.title(f'ECDF of {name}', size = 20)
    plt.tight_layout()
    

    if save:
        plt.savefig(save_name + '.png')
        plt.savefig(save_name + '.pdf')

def feature_wise_z_score_norm(OTU_df):
    """feature-wise (taxa-wise) z-score (standard scaler) normalization
    Args:
        OTU_df (_pandas df_): output of load_data or check_metadata
    """
    scaler = StandardScaler()
    data = OTU_df.T.to_numpy()
    scaler.fit(data)
    ss_scaled = scaler.transform(data).T
    ss_OTU_df = pd.DataFrame(ss_scaled, columns = OTU_df.columns, index=OTU_df.index)
    return ss_OTU_df

def completePCA(Xfile, Yfile, outputRoot, columnValue='Diet', targets=['A', 'B', 'BS'], colors=['r', 'g', 'b'], figsizeInput = (8,8), limit=False):
    # ecdf plot
    X_df = pd.read_csv(Xfile, delimiter='\t', index_col='# Gene Family')
    X_df = feature_wise_z_score_norm(X_df)
    # ecdf_plot(X_df.std().sort_values(ascending = False), name = 'standard deviation of genes before PCA', save=True, save_name=outputRoot+"_ecdf")
    X, binaryY = datasetPrep(Xfile, Yfile, columnValue)
    print(outputRoot)
    plotPCA(X, binaryY, outputRoot, targets, colors, figsize_input=figsizeInput, limit=limit)


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


allOmicsX = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/PCA/source/4omicsAllX.txt'
humann3YW = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/4omicsAllY_wmetadata.xlsx'

# completePCA(allOmicsX, humann3YW, outputRoot='./output/4omicsDietsPCA', columnValue='Diet',
#             targets=['A', 'B', 'BL'], colors=['r', 'g', 'b'])
# completePCA(allOmicsX, humann3YW, outputRoot='./output/4omicsFructanPCA', columnValue='FRUCTANSENSITIVE',
#             targets=[0, 1], colors=['r', 'g'])
# completePCA(allOmicsX, humann3YW, outputRoot='./output/4omicsPatientIDPCA', columnValue='Patient_ID',
#             targets=patientIDs, colors=patientID_colorL, figsizeInput=(20, 8))


# completePCA(allOmicsX, humann3YW, outputRoot='./output/4omicsDietsPCA_limited', columnValue='Diet',
#             targets=['A', 'B', 'BL'], colors=['r', 'g', 'b'], limit=True)
# completePCA(allOmicsX, humann3YW, outputRoot='./output/4omicsFructanPCA_limited', columnValue='FRUCTANSENSITIVE',
#             targets=[0, 1], colors=['r', 'g'], limit=True)
# completePCA(allOmicsX, humann3YW, outputRoot='./output/4omicsPatientIDPCA_limited', columnValue='Patient_ID',
#             targets=patientIDs, colors=patientID_colorL, figsizeInput=(20, 8), limit=True)

# humann3X = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/PCA/humann3PCA031323.txt'
# completePCA(humann3X, humann3YW, outputRoot='./output/humann3DietsPCA', columnValue='Diet',
#             targets=['A', 'B', 'BL'], colors=['r', 'g', 'b'])
# completePCA(humann3X, humann3YW, outputRoot='./output/humann3FructanPCA', columnValue='FRUCTANSENSITIVE',
#             targets=[0, 1], colors=['r', 'g'])
# completePCA(humann3X, humann3YW, outputRoot='./output/humann3PatientIDPCA', columnValue='Patient_ID',
#             targets=patientIDs, colors=patientID_colorL, figsizeInput=(20, 8))

# lipidsX = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/PCA/lipidsPCA031323.txt'
# completePCA(lipidsX, humann3YW, outputRoot='./output/lipidsDietsPCA', columnValue='Diet',
#             targets=['A', 'B', 'BL'], colors=['r', 'g', 'b'])
# completePCA(lipidsX, humann3YW, outputRoot='./output/lipidsFructanPCA', columnValue='FRUCTANSENSITIVE',
#             targets=[0, 1], colors=['r', 'g'])
# completePCA(lipidsX, humann3YW, outputRoot='./output/lipidsPatientIDPCA', columnValue='Patient_ID',
#             targets=patientIDs, colors=patientID_colorL, figsizeInput=(20, 8))

# metabolitesX = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/PCA/metabolitesPCA031323.txt'
# completePCA(metabolitesX, humann3YW, outputRoot='./output/metabolitesDietsPCA', columnValue='Diet',
#             targets=['A', 'B', 'BL'], colors=['r', 'g', 'b'])
# completePCA(metabolitesX, humann3YW, outputRoot='./output/metabolitesFructanPCA', columnValue='FRUCTANSENSITIVE',
#             targets=[0, 1], colors=['r', 'g'])
# completePCA(metabolitesX, humann3YW, outputRoot='./output/metabolitesPatientIDPCA', columnValue='Patient_ID',
#             targets=patientIDs, colors=patientID_colorL, figsizeInput=(20, 8))

metaphlanX = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/PCA/metaphlanPCA031323.txt'
# completePCA(metaphlanX, humann3YW, outputRoot='./output/metaphlanDietsPCA', columnValue='Diet',
#             targets=['A', 'B', 'BL'], colors=['r', 'g', 'b'])
# completePCA(metaphlanX, humann3YW, outputRoot='./output/metaphlanFructanPCA', columnValue='FRUCTANSENSITIVE',
#             targets=[0, 1], colors=['r', 'g'])
# completePCA(metaphlanX, humann3YW, outputRoot='./output/metaphlanPatientIDPCA', columnValue='Patient_ID',
#             targets=patientIDs, colors=patientID_colorL, figsizeInput=(20, 8))

completePCA(metaphlanX, humann3YW, outputRoot='./output/metaphlanDietsPCA_limited', columnValue='Diet',
            targets=['A', 'B', 'BL'], colors=['r', 'g', 'b'], limit=True)
completePCA(metaphlanX, humann3YW, outputRoot='./output/metaphlanFructanPCA_limited', columnValue='FRUCTANSENSITIVE',
            targets=[0, 1], colors=['r', 'g'], limit=True)
completePCA(metaphlanX, humann3YW, outputRoot='./output/metaphlanPatientIDPCA_limited', columnValue='Patient_ID',
            targets=patientIDs, colors=patientID_colorL, figsizeInput=(20, 8), limit=True)



# humann3X = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/PCA/humann3PCA031323.txt'
# completePCA(humann3X, humann3YW, outputRoot='./output/humann3DietsPCA_limited', columnValue='Diet',
#             targets=['A', 'B', 'BL'], colors=['r', 'g', 'b'], limit=True)
# completePCA(humann3X, humann3YW, outputRoot='./output/humann3FructanPCA_limited', columnValue='FRUCTANSENSITIVE',
#             targets=[0, 1], colors=['r', 'g'], limit=True)
# completePCA(humann3X, humann3YW, outputRoot='./output/humann3PatientIDPCA_limited', columnValue='Patient_ID',
#             targets=patientIDs, colors=patientID_colorL, figsizeInput=(20, 8), limit=True)

# lipidsX = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/PCA/lipidsPCA031323.txt'
# completePCA(lipidsX, humann3YW, outputRoot='./output/lipidsDietsPCA_limited', columnValue='Diet',
#             targets=['A', 'B', 'BL'], colors=['r', 'g', 'b'], limit=True)
# completePCA(lipidsX, humann3YW, outputRoot='./output/lipidsFructanPCA_limited', columnValue='FRUCTANSENSITIVE',
#             targets=[0, 1], colors=['r', 'g'], limit=True)
# completePCA(lipidsX, humann3YW, outputRoot='./output/lipidsPatientIDPCA_limited', columnValue='Patient_ID',
#             targets=patientIDs, colors=patientID_colorL, figsizeInput=(20, 8), limit=True)

# metabolitesX = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/PCA/metabolitesPCA031323.txt'
# completePCA(metabolitesX, humann3YW, outputRoot='./output/metabolitesDietsPCA_limited', columnValue='Diet',
#             targets=['A', 'B', 'BL'], colors=['r', 'g', 'b'], limit=True)
# completePCA(metabolitesX, humann3YW, outputRoot='./output/metabolitesFructanPCA_limited', columnValue='FRUCTANSENSITIVE',
#             targets=[0, 1], colors=['r', 'g'], limit=True)
# completePCA(metabolitesX, humann3YW, outputRoot='./output/metabolitesPatientIDPCA_limited', columnValue='Patient_ID',
#             targets=patientIDs, colors=patientID_colorL, figsizeInput=(20, 8), limit=True)

# metaphlanX = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/PCA/metaphlanPCA031323.txt'
# completePCA(metaphlanX, humann3YW, outputRoot='./output/metaphlanDietsPCA_limited', columnValue='Diet',
#             targets=['A', 'B', 'BL'], colors=['r', 'g', 'b'], limit=True)
# completePCA(metaphlanX, humann3YW, outputRoot='./output/metaphlanFructanPCA_limited', columnValue='FRUCTANSENSITIVE',
#             targets=[0, 1], colors=['r', 'g'], limit=True)
# completePCA(metaphlanX, humann3YW, outputRoot='./output/metaphlanPatientIDPCA_limited', columnValue='Patient_ID',
#             targets=patientIDs, colors=patientID_colorL, figsizeInput=(20, 8), limit=True)


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import time
import numpy as np
from matplotlib.patches import Patch
#for this post we will use MinMaxScaler
scaler=StandardScaler()

def all_heatmap(csv_path, csv_colNames, resultFile, out):

    # data cleaning
    data = pd.read_csv(csv_path, index_col=0)
    colNameDf = pd.read_csv(csv_colNames)
    # generate heatmap column names
    colNameDict_og = colNameDf.transpose().to_dict()
    colNameDict = {v: k for k, v in colNameDict_og[0].items()}

    # generate color labelling
    sampleNames = list(data.columns)
    color_dict = {}
    palette = sns.color_palette()
    for col in sampleNames:
        if "sensitive" in col:
            color_dict[col] = palette[0]
        else:
            color_dict[col] = palette[1]
    color_rows = pd.Series(color_dict)
    # get final data
    dataRows = list(data.index)
    data=pd.DataFrame(scaler.fit_transform(data.T).T,columns=data.columns)
    data.index = dataRows
    data.index.names = ['Name']
    data.columns = [colNameDict[sampleName] for sampleName in data.columns]


    # get ups and downs info
    # print("dataRows", dataRows)
    numAll = len(dataRows)
    print("total", len(dataRows))
    current_result = pd.read_csv(resultFile, delimiter = "\t")
    current_result_dict = current_result.set_index("TaxoName").T.to_dict('list')
    
    foldChanges = [current_result_dict[featureName][4] for featureName in dataRows]
    numTol = (np.array(foldChanges)>1).sum()
    numSen = len(foldChanges)-(np.array(foldChanges)>1).sum()
    print("ups", numTol)
    print("downs", numSen)
    print(data)

    # set title
    title = out.split(".")[0].split("_")
    title = ' '.join(title) + ' clustermap'

    # 0 generate cluster map (dendoOrdered version)
    # g = sns.clustermap(data, cmap="YlGnBu", col_cluster=False, cbar_kws={"orientation": "horizontal"}, col_colors = [color_rows])
    data_copy = data.copy()
    data_copy['names'] = data_copy.index
    colortick_l = ["red" if fc<1 else "blue" for fc in foldChanges]
    colortick_dict = dict(zip(list(data_copy['names']), colortick_l))
    data_copy.index = foldChanges
    data_copy = data_copy.sort_index(ascending=True)
    data_copy.index = data_copy['names']
    del data_copy['names']
    g = sns.clustermap(data_copy, cmap="vlag", row_cluster=False, col_cluster=False, cbar_kws={"orientation": "horizontal"}, col_colors = [color_rows], vmin=-4, vmax=4)
    
    g.ax_row_dendrogram.set_visible(True)
    x0, _y0, _w, _h = g.cbar_pos
    g.ax_cbar.set_position([x0, 0.9, g.ax_row_dendrogram.get_position().width, 0.05])
    g.ax_cbar.set_title('z score')
    g.fig.suptitle(title)

    for tick_label in g.ax_heatmap.axes.get_yticklabels():
        tick_text = tick_label.get_text()
        # species_name = data_copy.loc[int(tick_text)]
        tick_label.set_color(colortick_dict[tick_text])
    plt.legend([Patch(facecolor=palette[0]), Patch(facecolor=palette[1])], ["sensitive", "tolerant"], title='Fructan Sensitivity',
           bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
    plt.savefig("1013combo/dendoOrdered/dendoOrdered"+out, dpi=300)
    plt.close()

    # time.sleep(20)
    # 1 generate cluster map (dendo version)
    # g = sns.clustermap(data, cmap="YlGnBu", col_cluster=False, cbar_kws={"orientation": "horizontal"}, col_colors = [color_rows])
    g = sns.clustermap(data, cmap="vlag", col_cluster=False, cbar_kws={"orientation": "horizontal"}, col_colors = [color_rows], vmin=-4, vmax=4)
    
    g.ax_row_dendrogram.set_visible(True)
    x0, _y0, _w, _h = g.cbar_pos
    g.ax_cbar.set_position([x0, 0.9, g.ax_row_dendrogram.get_position().width, 0.05])
    g.ax_cbar.set_title('z score')
    g.fig.suptitle(title)
    plt.legend([Patch(facecolor=palette[0]), Patch(facecolor=palette[1])], ["sensitive", "tolerant"], title='Fructan Sensitivity',
           bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
    plt.savefig("1013combo/dendo/dendo"+out, dpi=300)
    plt.close()
    # time.sleep(15)


    # 2 generate cluster map (no dendo version)
    g = sns.clustermap(data, cmap="vlag", col_cluster=False, cbar_kws={"orientation": "vertical"}, col_colors = [color_rows], vmin=-4, vmax=4)
    g.ax_cbar.set_position([x0, 0.9, g.ax_row_dendrogram.get_position().width, 0.05])
    g.ax_cbar.set_title('z score')
    g.ax_row_dendrogram.set_visible(False)
    dendro_box = g.ax_row_dendrogram.get_position()
    dendro_box.x0 = (dendro_box.x0 + 2 * dendro_box.x1) / 3
    g.cax.set_position(dendro_box)
    g.cax.yaxis.set_ticks_position("left")
    g.fig.suptitle(title)
    plt.legend([Patch(facecolor=palette[0]), Patch(facecolor=palette[1])], ["sensitive", "tolerant"], title='Fructan Sensitivity',
           bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
    plt.savefig("1013combo/nodendo/nodendo"+out, dpi=300)
    plt.close()


    # 3 generate cluster map (dendo version) (all clustered)
    g = sns.clustermap(data, cmap="vlag", cbar_kws={"orientation": "horizontal"}, col_colors = [color_rows], vmin=-4, vmax=4)
    g.ax_row_dendrogram.set_visible(True)
    x0, _y0, _w, _h = g.cbar_pos
    g.ax_cbar.set_position([x0, 0.9, g.ax_row_dendrogram.get_position().width, 0.05])
    g.ax_cbar.set_title('z score')
    g.fig.suptitle(title)
    plt.legend([Patch(facecolor=palette[0]), Patch(facecolor=palette[1])], ["sensitive", "tolerant"], title='Fructan Sensitivity',
           bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
    plt.savefig("1013combo/dendoCol/dendoCol"+out, dpi=300)
    plt.close()

    # 4 generate cluster map (no dendo version) (all clustered)
    g = sns.clustermap(data, cmap="vlag", cbar_kws={"orientation": "vertical"}, col_colors = [color_rows], vmin=-4, vmax=4)
    g.ax_cbar.set_position([x0, 0.9, g.ax_row_dendrogram.get_position().width, 0.05])
    g.ax_cbar.set_title('z score')
    g.ax_row_dendrogram.set_visible(False)
    dendro_box = g.ax_row_dendrogram.get_position()
    dendro_box.x0 = (dendro_box.x0 + 2 * dendro_box.x1) / 3
    g.cax.set_position(dendro_box)
    g.cax.yaxis.set_ticks_position("left")
    g.fig.suptitle(title)
    plt.legend([Patch(facecolor=palette[0]), Patch(facecolor=palette[1])], ["sensitive", "tolerant"], title='Fructan Sensitivity',
           bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
    plt.savefig("1013combo/nodendoCol/nodendoCol"+out, dpi=300)
    plt.close()
    df = pd.DataFrame(list(zip([out], [numAll], [numTol], [numSen])),
               columns =['omics', 'number of biomarkers in total', 'number enriched in tolerant kids', 'number enriched in sensitive kids'])
    df.to_csv("1013combo/"+out+'heatmap_up_down_summary.csv')

def per_diet_heatmap(csv_path_l, result_path_l, omicNames, csv_colNames, out):
    row_names = []
    finalData = pd.DataFrame()
    numAll = []
    numTol = []
    numSen = []
    for index, csv_path in enumerate(csv_path_l):
        # data cleaning
        data = pd.read_csv(csv_path, index_col=0)
        # print(data.columns)
        sampleNames = list(data.columns)
        colNameDf = pd.read_csv(csv_colNames[index])
        row_names.extend([omicNames[index] for _ in range(len(data))])
        
        # generate heatmap column names
        colNameDict_og = colNameDf.transpose().to_dict()
        colNameDict = {v: k for k, v in colNameDict_og[0].items()}

        # get current final data
        dataRows = list(data.index)
        data=pd.DataFrame(scaler.fit_transform(data.T).T,columns=data.columns)
        data.index = dataRows
        data.index.names = ['Name']
        data.columns = [colNameDict[sampleName] for sampleName in data.columns]

        # get ups and downs info
        # print("dataRows", dataRows)
        numAll.append(len(dataRows))
        current_result = pd.read_csv(result_path_l[index], delimiter = "\t")
        current_result_dict = current_result.set_index("TaxoName").T.to_dict('list')
        
        foldChanges = [current_result_dict[featureName][4] for featureName in dataRows]
        numTol.append((np.array(foldChanges)>1).sum())
        numSen.append(len(foldChanges)-(np.array(foldChanges)>1).sum())

        # get final data
        if finalData.empty:
            finalData = data
        else:
            finalData = finalData.append(data)

    print(sampleNames)
    print("AAAAAAA")
    # time.sleep(15)
    color_dict = {}
    palette = sns.color_palette()
    for col in sampleNames:
        if "sensitive" in col:
            color_dict[col] = palette[0]
        else:
            color_dict[col] = palette[1]
    color_rows = pd.Series(color_dict)

    # set title
    title = out.split(".")[0].split("_")
    title = ' '.join(title) + ' clustermap'

    # 1 generate cluster map (dendo version)
    print(finalData)
    g = sns.clustermap(finalData, cmap="vlag", col_cluster=False, cbar_kws={"orientation": "horizontal"}, col_colors = [color_rows], 
    vmin=-4, vmax=4)
    
    g.ax_row_dendrogram.set_visible(True)
    x0, _y0, _w, _h = g.cbar_pos
    g.ax_cbar.set_position([x0, 0.9, g.ax_row_dendrogram.get_position().width, 0.05])
    g.ax_cbar.set_title('z score')
    g.fig.suptitle(title)
    plt.legend([Patch(facecolor=palette[0]), Patch(facecolor=palette[1])], ["sensitive", "tolerant"], title='Fructan Sensitivity',
           bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
    plt.savefig("1013combo_4omics/dendo/dendo"+out, dpi=300)
    plt.close()

    # 2 generate cluster map (no dendo version)
    g = sns.clustermap(finalData, cmap="vlag", col_cluster=False, cbar_kws={"orientation": "vertical"}, col_colors = [color_rows], vmin=-4, vmax=4)
    g.ax_cbar.set_position([x0, 0.9, g.ax_row_dendrogram.get_position().width, 0.05])
    g.ax_cbar.set_title('z score')
    g.ax_row_dendrogram.set_visible(False)
    dendro_box = g.ax_row_dendrogram.get_position()
    dendro_box.x0 = (dendro_box.x0 + 2 * dendro_box.x1) / 3
    g.cax.set_position(dendro_box)
    g.cax.yaxis.set_ticks_position("left")
    g.fig.suptitle(title)
    plt.legend([Patch(facecolor=palette[0]), Patch(facecolor=palette[1])], ["sensitive", "tolerant"], title='Fructan Sensitivity',
           bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
    plt.savefig("1013combo_4omics/nodendo/nodendo"+out, dpi=300)
    plt.close()


    # 3 generate cluster map (dendo version) (all clustered)
    g = sns.clustermap(finalData, cmap="vlag", cbar_kws={"orientation": "horizontal"}, col_colors = [color_rows], vmin=-4, vmax=4)
    g.ax_row_dendrogram.set_visible(True)
    x0, _y0, _w, _h = g.cbar_pos
    g.ax_cbar.set_position([x0, 0.9, g.ax_row_dendrogram.get_position().width, 0.05])
    g.ax_cbar.set_title('z score')
    g.fig.suptitle(title)
    plt.legend([Patch(facecolor=palette[0]), Patch(facecolor=palette[1])], ["sensitive", "tolerant"], title='Fructan Sensitivity',
           bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
    plt.savefig("1013combo_4omics/dendoCol/dendoCol"+out, dpi=300)
    plt.close()

    # 4 generate cluster map (no dendo version) (all clustered)
    g = sns.clustermap(finalData, cmap="vlag", cbar_kws={"orientation": "vertical"}, col_colors = [color_rows], vmin=-4, vmax=4)
    g.ax_cbar.set_position([x0, 0.9, g.ax_row_dendrogram.get_position().width, 0.05])
    g.ax_cbar.set_title('z score')
    g.ax_row_dendrogram.set_visible(False)
    dendro_box = g.ax_row_dendrogram.get_position()
    dendro_box.x0 = (dendro_box.x0 + 2 * dendro_box.x1) / 3
    g.cax.set_position(dendro_box)
    g.cax.yaxis.set_ticks_position("left")
    g.fig.suptitle(title)
    plt.legend([Patch(facecolor=palette[0]), Patch(facecolor=palette[1])], ["sensitive", "tolerant"], title='Fructan Sensitivity',
           bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
    plt.savefig("1013combo/nodendoCol/nodendoCol"+out, dpi=300)
    plt.close()

    df = pd.DataFrame(list(zip(omicNames, numAll, numTol, numSen)),
               columns =['omics', 'number of biomarkers in total', 'number enriched in tolerant kids', 'number enriched in sensitive kids'])

    df.to_csv("1013combo_4omics/"+out+'heatmap_up_down_summary.csv')
    




# all_heatmap("./filteredOutputs/humann3KruskalFinalFeatures_all_heatmapOutput.csv", "./kruskalOutputs/kruskal.Humann3.all.101322heatmapSampleMatch.csv","./kruskalOutputs/kruskal.Humann3.all.101322aovResults.xls", "humann3_all.pdf")

# all_heatmap("./filteredOutputs/lipidomicsKruskalFinalFeatures_A_heatmapOutput.csv", "./kruskalOutputs/kruskal.Lipids.A.101322heatmapSampleMatch.csv", "./kruskalOutputs/kruskal.Lipids.A.101322aovResults.xls", "lipidomics_A.pdf")
all_heatmap("./filteredOutputs/lipidomicsKruskalFinalFeatures_AAndB_heatmapOutput.csv", "./kruskalOutputs/kruskal.Lipids.AAndB.101322heatmapSampleMatch.csv", "./kruskalOutputs/kruskal.Lipids.AAndB.101322aovResults.xls","lipidomics_AAndB.pdf")
# all_heatmap("./filteredOutputs/lipidomicsKruskalFinalFeatures_AAndBS_heatmapOutput.csv", "./kruskalOutputs/kruskal.Lipids.AAndBS.101322heatmapSampleMatch.csv","./kruskalOutputs/kruskal.Lipids.AAndBS.101322aovResults.xls", "lipidomics_AAndBS.pdf")
# all_heatmap("./filteredOutputs/lipidomicsKruskalFinalFeatures_all_heatmapOutput.csv", "./kruskalOutputs/kruskal.Lipids.all.101322heatmapSampleMatch.csv", "./kruskalOutputs/kruskal.Lipids.all.101322aovResults.xls", "lipidomics_all.pdf")
# all_heatmap("./filteredOutputs/lipidomicsKruskalFinalFeatures_BAndBS_heatmapOutput.csv", "./kruskalOutputs/kruskal.Lipids.BAndBS.101322heatmapSampleMatch.csv", "./kruskalOutputs/kruskal.Lipids.BAndBS.101322aovResults.xls","lipidomics_BAndBS.pdf")
# all_heatmap("./filteredOutputs/lipidomicsKruskalFinalFeatures_BS_heatmapOutput.csv", "./kruskalOutputs/kruskal.Lipids.BS.101322heatmapSampleMatch.csv","./kruskalOutputs/kruskal.Lipids.BS.101322aovResults.xls", "lipidomics_BS.pdf")

# all_heatmap("./filteredOutputs/metabolitesKruskalFinalFeatures_AAndB_heatmapOutput.csv", "./kruskalOutputs/kruskal.Lipids.AAndB.101322heatmapSampleMatch.csv", "./kruskalOutputs/kruskal.Metabolites.AAndB.101322aovResults.xls","metabolites_AAndB.pdf")
# all_heatmap("./filteredOutputs/metabolitesKruskalFinalFeatures_AAndBS_heatmapOutput.csv", "./kruskalOutputs/kruskal.Lipids.AAndBS.101322heatmapSampleMatch.csv",  "./kruskalOutputs/kruskal.Metabolites.AAndBS.101322aovResults.xls", "metabolites_AAndBS.pdf")
# all_heatmap("./filteredOutputs/metabolitesKruskalFinalFeatures_all_heatmapOutput.csv", "./kruskalOutputs/kruskal.Metabolites.all.101322heatmapSampleMatch.csv",  "./kruskalOutputs/kruskal.Metabolites.all.101322aovResults.xls","metabolites_all.pdf")
# all_heatmap("./filteredOutputs/metabolitesKruskalFinalFeatures_BAndBS_heatmapOutput.csv", "./kruskalOutputs/kruskal.Metabolites.BAndBS.101322heatmapSampleMatch.csv",  "./kruskalOutputs/kruskal.Metabolites.BAndBS.101322aovResults.xls","metabolites_BAndBS.pdf")

# all_heatmap("./filteredOutputs/metaphlanKruskalFinalFeatures_AAndBS_heatmapOutput.csv", "./kruskalOutputs/kruskal.Metaphlan.AAndBS.101322heatmapSampleMatch.csv",  "./kruskalOutputs/kruskal.Metaphlan.AAndB.101322aovResults.xls", "metaphlan_AAndBS.pdf")
# all_heatmap("./filteredOutputs/metaphlanKruskalFinalFeatures_all_heatmapOutput.csv", "./kruskalOutputs/kruskal.Metaphlan.all.101322heatmapSampleMatch.csv",  "./kruskalOutputs/kruskal.Metaphlan.all.101322aovResults.xls", "metaphlan_all.pdf")



# # all
# all_csv_path_l = ["./filteredOutputs/humann3KruskalFinalFeatures_all_heatmapOutput.csv",
# "./filteredOutputs/lipidomicsKruskalFinalFeatures_all_heatmapOutput.csv",
# "./filteredOutputs/metabolitesKruskalFinalFeatures_all_heatmapOutput.csv",
# "./filteredOutputs/metaphlanKruskalFinalFeatures_all_heatmapOutput.csv"]

# omicNames_all = ['humann3', 'lipidomics', 'metabolites', 'metaphlan']

# result_file_l = ['/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Humann3.all.101322aovResults.xls',
# '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Lipids.all.101322aovResults.xls',
# '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metabolites.all.101322aovResults.xls',
# '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metaphlan.all.101322aovResults.xls']

# all_csv_colNames = ["./kruskalOutputs/kruskal.Humann3.all.101322heatmapSampleMatch.csv", 
# "./kruskalOutputs/kruskal.Lipids.all.101322heatmapSampleMatch.csv",
# "./kruskalOutputs/kruskal.Metabolites.all.101322heatmapSampleMatch.csv",
# "./kruskalOutputs/kruskal.Metaphlan.all.101322heatmapSampleMatch.csv"
# ]

# per_diet_heatmap(all_csv_path_l, result_file_l, omicNames_all, all_csv_colNames, "allDiet.pdf")

# AAndB

all_csv_path_l = ["./filteredOutputs/lipidomicsKruskalFinalFeatures_AAndB_heatmapOutput.csv",
        "./filteredOutputs/metabolitesKruskalFinalFeatures_AAndB_heatmapOutput.csv",]

omicNames_all = ['lipidomics', 'metabolites']

result_file_l = ['/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Lipids.AAndB.101322aovResults.xls',
'/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metabolites.AAndB.101322aovResults.xls']

all_csv_colNames = ["./kruskalOutputs/kruskal.Lipids.AAndB.101322heatmapSampleMatch.csv",
            "./kruskalOutputs/kruskal.Metabolites.AAndB.101322heatmapSampleMatch.csv"]

per_diet_heatmap(all_csv_path_l, result_file_l, omicNames_all, all_csv_colNames, "AAndBDietAAA.pdf")


# # AAndBS

# all_csv_path_l = ["./filteredOutputs/lipidomicsKruskalFinalFeatures_AAndBS_heatmapOutput.csv",
# "./filteredOutputs/metabolitesKruskalFinalFeatures_AAndBS_heatmapOutput.csv",
# "./filteredOutputs/metaphlanKruskalFinalFeatures_AAndBS_heatmapOutput.csv"]

# omicNames_all = ['lipidomics', 'metabolites', 'metaphlan']

# result_file_l = ['/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Lipids.AAndBS.101322aovResults.xls',
# '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metabolites.AAndBS.101322aovResults.xls',
# '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metaphlan.AAndBS.101322aovResults.xls']

# all_csv_colNames = ["./kruskalOutputs/kruskal.Lipids.AAndBS.101322heatmapSampleMatch.csv",
# "./kruskalOutputs/kruskal.Metabolites.AAndBS.101322heatmapSampleMatch.csv",
# "./kruskalOutputs/kruskal.Metaphlan.AAndBS.101322heatmapSampleMatch.csv"
# ]

# per_diet_heatmap(all_csv_path_l, result_file_l, omicNames_all, all_csv_colNames, "AAndBSDiet.pdf")

# # BAndBS
# all_csv_path_l = ["./filteredOutputs/lipidomicsKruskalFinalFeatures_BAndBS_heatmapOutput.csv",
# "./filteredOutputs/metabolitesKruskalFinalFeatures_BAndBS_heatmapOutput.csv"]

# omicNames_all = ['lipidomics', 'metabolites']

# result_file_l = ['/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Lipids.BAndBS.101322aovResults.xls',
# '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metabolites.BAndBS.101322aovResults.xls']

# all_csv_colNames = ["./kruskalOutputs/kruskal.Lipids.BAndBS.101322heatmapSampleMatch.csv",
# "./kruskalOutputs/kruskal.Metabolites.BAndBS.101322heatmapSampleMatch.csv"
# ]

# per_diet_heatmap(all_csv_path_l, result_file_l, omicNames_all, all_csv_colNames, "BAndBSDiet.pdf")

# # BS

# all_csv_path_l = ["./filteredOutputs/lipidomicsKruskalFinalFeatures_BS_heatmapOutput.csv"]

# omicNames_all = ['lipidomics']

# result_file_l = ['/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Lipids.BS.101322aovResults.xls']

# all_csv_colNames = ["./kruskalOutputs/kruskal.Lipids.BS.101322heatmapSampleMatch.csv"]

# per_diet_heatmap(all_csv_path_l, result_file_l, omicNames_all, all_csv_colNames, "BSDiet.pdf")


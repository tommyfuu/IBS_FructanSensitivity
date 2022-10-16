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
    plt.savefig("1013combo/dendo/dendo"+out, dpi=300)
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
    plt.savefig("1013combo/nodendo/nodendo"+out, dpi=300)
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
    plt.savefig("1013combo/dendoCol/dendoCol"+out, dpi=300)
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

    df.to_csv("1013combo/"+out+'heatmap_up_down_summary.csv')
    
# def mean_heatmap(csv_path, out):
#     data = pd.read_csv(csv_path, index_col=0)
#     data=pd.DataFrame(scaler.fit_transform(data.T).T,columns=data.columns)
#     print(data)
#     data.index.names = ['Name']
#     g = sns.heatmap(data, cmap="YlGnBu")
#     g.set_yticklabels(g.get_yticklabels(), rotation=0)
#     g.set_title('Heatmap')
#     plt.tight_layout()
#     plt.savefig(out)
#     plt.show()



# all_heatmap("./finalStuff/humann3KruskalFinalFeatures_all_heatmapOutput.csv", "./kruskalOutputs/kruskal.Humann3.all.052522heatmapSampleMatch.csv", "humann3_all.pdf")
# all_heatmap("./finalStuff/humann3KruskalFinalFeatures_BAndBS_heatmapOutput.csv", "./kruskalOutputs/kruskal.Humann3.BAndBS.052522heatmapSampleMatch.csv", "humann3_BAndBS.pdf")
# all_heatmap("./finalStuff/lipidomicsKruskalFinalFeatures_AAndBS_heatmapOutput.csv", "./kruskalOutputs/kruskal.Lipids.AAndBS.052522heatmapSampleMatch.csv", "lipidomics_AAndBS.pdf")
# all_heatmap("./finalStuff/lipidomicsKruskalFinalFeatures_all_heatmapOutput.csv", "./kruskalOutputs/kruskal.Lipids.all.052522heatmapSampleMatch.csv", "lipidomics_all.pdf")
# all_heatmap("./finalStuff/lipidomicsKruskalFinalFeatures_BAndBS_heatmapOutput.csv", "./kruskalOutputs/kruskal.Lipids.BAndBS.052522heatmapSampleMatch.csv", "lipidomics_BAndBS.pdf")
# all_heatmap("./finalStuff/lipidomicsKruskalFinalFeatures_BS_heatmapOutput.csv", "./kruskalOutputs/kruskal.Lipids.BS.052522heatmapSampleMatch.csv", "lipidomics_BS.pdf")
# all_heatmap("./finalStuff/metabolitesKruskalFinalFeatures_AAndB_heatmapOutput.csv", "./kruskalOutputs/kruskal.Lipids.AAndB.052522heatmapSampleMatch.csv", "metabolites_AAndB.pdf")
# all_heatmap("./finalStuff/metabolitesKruskalFinalFeatures_AAndBS_heatmapOutput.csv", "./kruskalOutputs/kruskal.Lipids.AAndBS.052522heatmapSampleMatch.csv", "metabolites_AAndBS.pdf")
# all_heatmap("./finalStuff/metabolitesKruskalFinalFeatures_all_heatmapOutput.csv", "./kruskalOutputs/kruskal.Metabolites.all.052522heatmapSampleMatch.csv", "metabolites_all.pdf")
# all_heatmap("./finalStuff/metabolitesKruskalFinalFeatures_BAndBS_heatmapOutput.csv", "./kruskalOutputs/kruskal.Metabolites.BAndBS.052522heatmapSampleMatch.csv", "metabolites_BAndBS.pdf")
# all_heatmap("./finalStuff/metaphlanKruskalFinalFeatures_AAndBS_heatmapOutput.csv", "./kruskalOutputs/kruskal.Metabolites.AAndBS.052522heatmapSampleMatch.csv", "metaphlan_AAndBS.pdf")
# all_heatmap("./finalStuff/metaphlanKruskalFinalFeatures_all_heatmapOutput.csv", "./kruskalOutputs/kruskal.Metaphlan.all.052522heatmapSampleMatch.csv", "metaphlan_all.pdf")
# all_heatmap("./finalStuff/metaphlanKruskalFinalFeatures_AAndBS_heatmapOutput.csv", "./kruskalOutputs/kruskal.Metaphlan.AAndBS.052522heatmapSampleMatch.csv", "metaphlan_AAndBS.pdf")

# all_heatmap("./finalStuff/humann3KruskalFinalFeatures_all_heatmapOutput.csv", "./kruskalOutputs/kruskal.Humann3.all.052522heatmapSampleMatch.csv", '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Humann3.all.052522aovResults.xls', "humann3_all.pdf")
# all_heatmap("./finalStuff/lipidomicsKruskalFinalFeatures_AAndBS_heatmapOutput.csv", "./kruskalOutputs/kruskal.Lipids.AAndBS.052522heatmapSampleMatch.csv", '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Lipids.AAndBS.052522aovResults.xls',"lipidomics_AAndBS.pdf", )
# all_heatmap("./finalStuff/lipidomicsKruskalFinalFeatures_all_heatmapOutput.csv", "./kruskalOutputs/kruskal.Lipids.all.052522heatmapSampleMatch.csv", '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Lipids.all.052522aovResults.xls',"lipidomics_all.pdf")
# all_heatmap("./finalStuff/lipidomicsKruskalFinalFeatures_BAndBS_heatmapOutput.csv", "./kruskalOutputs/kruskal.Lipids.BAndBS.052522heatmapSampleMatch.csv", '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Lipids.BAndBS.052522aovResults.xls',"lipidomics_BAndBS.pdf")
# all_heatmap("./finalStuff/lipidomicsKruskalFinalFeatures_BS_heatmapOutput.csv", "./kruskalOutputs/kruskal.Lipids.BS.052522heatmapSampleMatch.csv", '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Lipids.BS.052522aovResults.xls', "lipidomics_BS.pdf")
# # NEED TO NOTICE THIS, did not do in by diet
# all_heatmap("./finalStuff/metabolitesKruskalFinalFeatures_AAndB_heatmapOutput.csv", "./kruskalOutputs/kruskal.Lipids.AAndB.052522heatmapSampleMatch.csv", '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metabolites.AAndB.052522aovResults.xls', "metabolites_AAndB.pdf")

# all_heatmap("./finalStuff/metabolitesKruskalFinalFeatures_AAndBS_heatmapOutput.csv", "./kruskalOutputs/kruskal.Lipids.AAndBS.052522heatmapSampleMatch.csv", '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metabolites.AAndBS.052522aovResults.xls',"metabolites_AAndBS.pdf")
# all_heatmap("./finalStuff/metabolitesKruskalFinalFeatures_all_heatmapOutput.csv", "./kruskalOutputs/kruskal.Metabolites.all.052522heatmapSampleMatch.csv", '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metabolites.all.052522aovResults.xls',"metabolites_all.pdf")
# all_heatmap("./finalStuff/metabolitesKruskalFinalFeatures_BAndBS_heatmapOutput.csv", "./kruskalOutputs/kruskal.Metabolites.BAndBS.052522heatmapSampleMatch.csv", '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metabolites.BAndBS.052522aovResults.xls',"metabolites_BAndBS.pdf")
# all_heatmap("./finalStuff/metaphlanKruskalFinalFeatures_BAndBS_heatmapOutput.csv", "./kruskalOutputs/kruskal.Metaphlan.BAndBS.052522heatmapSampleMatch.csv", '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metaphlan.BAndBS.052522aovResults.xls',"metaphlan_BAndBS.pdf")
# all_heatmap("./finalStuff/metaphlanKruskalFinalFeatures_AAndBS_heatmapOutput.csv", "./kruskalOutputs/kruskal.Metaphlan.AAndBS.052522heatmapSampleMatch.csv", '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metaphlan.AAndBS.052522aovResults.xls',"metaphlan_AAndBS.pdf")
# all_heatmap("./finalStuff/metaphlanKruskalFinalFeatures_all_heatmapOutput.csv", "./kruskalOutputs/kruskal.Metaphlan.all.052522heatmapSampleMatch.csv", '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metaphlan.all.052522aovResults.xls', "metaphlan_all.pdf")


all_csv_path_l = ["./filteredOutputs/humann3KruskalFinalFeatures_all_heatmapOutput.csv",
"./filteredOutputs/lipidomicsKruskalFinalFeatures_all_heatmapOutput.csv",
"./filteredOutputs/metabolitesKruskalFinalFeatures_all_heatmapOutput.csv",
"./filteredOutputs/metaphlanKruskalFinalFeatures_all_heatmapOutput.csv"]

omicNames_all = ['humann3', 'lipidomics', 'metabolites', 'metaphlan']

result_file_l = ['/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Humann3.all.101322aovResults.xls',
'/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Lipids.all.101322aovResults.xls',
'/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metabolites.all.101322aovResults.xls',
'/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metaphlan.all.101322aovResults.xls']

all_csv_colNames = ["./kruskalOutputs/kruskal.Humann3.all.101322heatmapSampleMatch.csv", 
"./kruskalOutputs/kruskal.Lipids.all.101322heatmapSampleMatch.csv",
"./kruskalOutputs/kruskal.Metabolites.all.101322heatmapSampleMatch.csv",
"./kruskalOutputs/kruskal.Metaphlan.all.101322heatmapSampleMatch.csv"
]

per_diet_heatmap(all_csv_path_l, result_file_l, omicNames_all, all_csv_colNames, "allDiet.pdf")

all_csv_path_l = ["./filteredOutputs/lipidomicsKruskalFinalFeatures_AAndBS_heatmapOutput.csv",
"./filteredOutputs/metabolitesKruskalFinalFeatures_AAndBS_heatmapOutput.csv",
"./filteredOutputs/metaphlanKruskalFinalFeatures_AAndBS_heatmapOutput.csv"]

omicNames_all = ['lipidomics', 'metabolites', 'metaphlan']

result_file_l = ['/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Lipids.AAndBS.101322aovResults.xls',
'/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metabolites.AAndBS.101322aovResults.xls',
'/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metaphlan.AAndBS.101322aovResults.xls']

all_csv_colNames = ["./kruskalOutputs/kruskal.Lipids.AAndBS.101322heatmapSampleMatch.csv",
"./kruskalOutputs/kruskal.Metabolites.AAndBS.101322heatmapSampleMatch.csv",
"./kruskalOutputs/kruskal.Metaphlan.AAndBS.101322heatmapSampleMatch.csv"
]

per_diet_heatmap(all_csv_path_l, result_file_l, omicNames_all, all_csv_colNames, "AAndBSDiet.pdf")

all_csv_path_l = ["./filteredOutputs/lipidomicsKruskalFinalFeatures_BAndBS_heatmapOutput.csv",
"./filteredOutputs/metabolitesKruskalFinalFeatures_BAndBS_heatmapOutput.csv"]

omicNames_all = ['lipidomics', 'metabolites']

result_file_l = ['/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Lipids.BAndBS.101322aovResults.xls',
'/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metabolites.BAndBS.101322aovResults.xls']

all_csv_colNames = ["./kruskalOutputs/kruskal.Lipids.BAndBS.101322heatmapSampleMatch.csv",
"./kruskalOutputs/kruskal.Metabolites.BAndBS.101322heatmapSampleMatch.csv"
]

per_diet_heatmap(all_csv_path_l, result_file_l, omicNames_all, all_csv_colNames, "BAndBSDiet.pdf")


all_csv_path_l = ["./filteredOutputs/lipidomicsKruskalFinalFeatures_BS_heatmapOutput.csv"]

omicNames_all = ['lipidomics']

result_file_l = ['/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Lipids.BS.101322aovResults.xls']

all_csv_colNames = ["./kruskalOutputs/kruskal.Lipids.BS.101322heatmapSampleMatch.csv"]

per_diet_heatmap(all_csv_path_l, result_file_l, omicNames_all, all_csv_colNames, "BSDiet.pdf")


all_csv_path_l = ["./filteredOutputs/metabolitesKruskalFinalFeatures_AAndB_heatmapOutput.csv"]

omicNames_all = ['metabolites']

result_file_l = ['/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metabolites.AAndB.101322aovResults.xls',]

all_csv_colNames = ["./kruskalOutputs/kruskal.Lipids.AAndB.101322heatmapSampleMatch.csv"]

per_diet_heatmap(all_csv_path_l, result_file_l, omicNames_all, all_csv_colNames, "AAndBDiet.pdf")


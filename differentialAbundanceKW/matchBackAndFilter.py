## AUTHOR: CHENLIAN FU
import pandas as pd
import os

def matchBackAndFilter(sourceFile, resultFileL, dietsList, out, taxaName):
    """
    get the kruskal wallis outputs from each experiment, filter to find significant biomarkers, and print out a summary
    """
    # read source and initialize
    source = pd.read_csv(sourceFile, sep="\t", header=None)
    source.columns = source.iloc[0] 
    print("matching back and filtering...")
    unique_biomarkers = []
    unique_biomarkers_omics = {} # collect in which omics/diets this biomarker is considered significant
    unique_biomarkers_df = pd.DataFrame(columns='AscensionNum\tTaxoName\tpValue\tfdrCorrectedpVal\tCohenEffectSize\tfoldChanges\twhetherSignificant\tmean0_tolerant\tmean1_sensitive'.split('\t'))
    unique_biomarkers_data = pd.DataFrame(columns='taxaName\t3003A\t3003B\t3003	3004A	3004B	3004	3005A	3005B	3005	3007A	3007B	3007	3008A	3008B	3008	3009A	3009B	3009	3012A	3012B	3012	3013A	3013B	3013	3014A	3014B	3014	3016A	3016B	3016	3020A	3020B	3020	3021A	3021B	3021	3022A	3022B	3022	3023A	3023B	3023	3025A	3025B	3025	3026A	3026B	3026	3027A	3027B	3027	3028A	3028B	3028	3029A	3029B	3029	3030A	3030B	3030	3031A	3031B	3031	3032A	3032B	3032	3033A	3033B	3033	3036A	3036B	3036	3037A	3037B	3037	3039A	3039B	3039	3042A	3042B	3042	3043A	3043B	3043	3044A	3044B	3044	3046A	3046B	3046	3047B	3047	3048A	3048B	3048	3049A	3049B	3049'.split('\t'))
    
    # loop thru all files
    for index, resultFile in enumerate(resultFileL):
        current_result = pd.read_csv(resultFile, delimiter = "\t")
        current_result_dict = current_result.set_index("TaxoName").T.to_dict('list')

        # # 060622 addition, just to print mean0 and mean1

        # loop thru all taxa
        for taxa in current_result_dict:

            if current_result_dict[taxa][1] <= 0.05 and current_result_dict[taxa][2] <= 0.25 and current_result_dict[taxa][4] != float('inf'):
                if taxa not in unique_biomarkers_omics.keys():
                    unique_biomarkers_omics[taxa] = [dietsList[index]]
                    unique_biomarkers.append(taxa)
                    current_biomarker_row = current_result.loc[current_result["TaxoName"] == taxa]
                    unique_biomarkers_df = unique_biomarkers_df.append(current_biomarker_row, ignore_index=True)
                    unique_biomarkers_data = unique_biomarkers_data.append(source.loc[source[taxaName] == taxa], ignore_index=True)
                else:
                    unique_biomarkers_omics[taxa].append(dietsList[index])
    


    # assign each significant biomarker with the diets in which it's considered significant, might be more than 1
    # since python dictionaries are insertion-ordered, the dict can be assigned to the dataframe directly
    unique_biomarkers_df['sig_in_omics'] = list(unique_biomarkers_omics.values())
    # get whether a biomarker is enriched in fructan tolerant kids or not; otherwise they are enriched in sensitive ones
    unique_biomarkers_df['enriched_in_tolerant'] = list(unique_biomarkers_df['foldChanges'].astype(float)>1)
    del unique_biomarkers_df["whetherSignificant"]
    del unique_biomarkers_df['AscensionNum']
    del unique_biomarkers_data[taxaName]
    del unique_biomarkers_data["taxaName"]
    ndf = pd.concat([unique_biomarkers_df, unique_biomarkers_data], axis=1)

    # print summary
    print(out)
    print("number of significant features: ", len(unique_biomarkers))
    print("number enriched in tolerant kids", sum(list(unique_biomarkers_df['foldChanges'].astype(float)>1)))
    print("number enriched in sensitive kids", len(unique_biomarkers)-sum(list(unique_biomarkers_df['foldChanges']>1)))
    ndf.to_csv(out+'.csv')
    return ndf

# # EXAMPLE: lipidomics
# sourceFile = './exampleSource/kruskalLipidomicsSource0525.txt'
# resultFileL = ['./exampleKruskalOutputs/kruskal.Lipids.A.052522aovResults.xls',
#             './exampleKruskalOutputs/kruskal.Lipids.AAndB.052522aovResults.xls',
#             './exampleKruskalOutputs/kruskal.Lipids.AAndBS.052522aovResults.xls',
#             './exampleKruskalOutputs/kruskal.Lipids.all.052522aovResults.xls',
#             './exampleKruskalOutputs/kruskal.Lipids.B.052522aovResults.xls',
#             './exampleKruskalOutputs/kruskal.Lipids.BAndBS.052522aovResults.xls',
#             './exampleKruskalOutputs/kruskal.Lipids.BS.052522aovResults.xls']
# dietsList = ['A', 'AAndB', 'AAndBS', 'all', 'B', 'BAndBS', 'BS']
# out = './exampleSummaryOutput/lipidomicsKruskalFinalFeatures_new'
# a = matchBackAndFilter(sourceFile, resultFileL, dietsList, out, 'BIOCHEMICAL')

def matchAllBackAndFilter(resultFileL, dietsList, out, taxaName):

    unique_biomarkers = []
    num_of_enriched_in_tolerant = 0
    num_of_enriched_in_sensitive = 0
    unique_biomarkers_omics = {} # collect in which omics/diets this biomarker is considered significant    
    resultFileWriter = open(out+"succintOutput.csv", 'wt')
    WRITEFIRSTLINE = False
    # loop thru all files
    for index, resultFile in enumerate(resultFileL):

        heatmapFileWriter = open(out+"_"+dietsList[index]+"_heatmapOutput.csv", 'wt')
        heatmapSmallFileWriter = open(out+"_"+dietsList[index]+"_heatmapSmallOutput.csv", 'wt')
        with open(resultFile) as topo_file:
            for line in topo_file:
                currentLine = line.split('\t') 
                resultFileLine = currentLine[1:8]
                heatmapFileLine = currentLine[8:]
                heatmapFileLine.insert(0, currentLine[1])
                if currentLine[0] == 'AscensionNum':
                    if WRITEFIRSTLINE == False:
                        resultFileWriter.write(','.join(resultFileLine)+'\n')
                        WRITEFIRSTLINE = True
                    heatmapFileWriter.write(','.join(heatmapFileLine))
                    heatmapSmallFileWriter.write('TaxoName, mean0_tolerant, mean1_sensitive\n')
                else:
                    taxa = currentLine[1]
                    pVal = float(currentLine[2])
                    FDRqVal = float(currentLine[3])
                    foldChange = float(currentLine[5])
                    if pVal <= 0.05 and FDRqVal <= 0.25 and foldChange != float('inf'):
                        heatmapFileWriter.write(','.join(heatmapFileLine))
                        heatmapSmallFileWriter.write(','.join([taxa, currentLine[6], currentLine[7]])+'\n')
                        if taxa not in unique_biomarkers_omics.keys():
                            unique_biomarkers_omics[taxa] = [dietsList[index]]
                            unique_biomarkers.append(taxa)
                            resultFileWriter.write(','.join(resultFileLine)+'\n')
                            if foldChange>1:
                                num_of_enriched_in_tolerant +=1
                            else:
                                num_of_enriched_in_sensitive +=1
                        else:
                            unique_biomarkers_omics[taxa].append(dietsList[index])
        heatmapFileWriter.close()
        num_lines = sum(1 for line in open(out+"_"+dietsList[index]+"_heatmapOutput.csv"))
        if num_lines <= 2:
            os.remove(out+"_"+dietsList[index]+"_heatmapOutput.csv") 
            os.remove(out+"_"+dietsList[index]+"_heatmapSmallOutput.csv") 
    dietsDF = pd.DataFrame(unique_biomarkers_omics.items(), columns=['TaxoName', 'Diets'])
    resultFileWriter.close()
    dietsDF.to_csv(out+"_biomarkersOmics.csv")

    # print summary
    print(out)
    print("number of significant features: ", len(unique_biomarkers))
    print("number enriched in tolerant kids", num_of_enriched_in_tolerant)
    print("number enriched in sensitive kids", num_of_enriched_in_sensitive)

    return len(unique_biomarkers), num_of_enriched_in_tolerant, num_of_enriched_in_sensitive

# # humann3
# sourceFile = '/Users/chenlianfu/Documents/GitHub/BrunoProject/finalKruskal/source/humann3KruskalSource.txt'
# resultFileL = ['/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Humann3.A.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Humann3.AAndB.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Humann3.AAndBS.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Humann3.all.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Humann3.B.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Humann3.BAndBS.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Humann3.BS.052522aovResults.xls']
# dietsList = ['A', 'AAndB', 'AAndBS', 'all', 'B', 'BAndBS', 'BS']
# out = '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/finalStuff/humann3KruskalFinalFeatures'
# a = matchBackAndFilter(sourceFile, resultFileL, dietsList, out, '# Gene Family')

# # metabolites
# sourceFile = '/Users/chenlianfu/Documents/GitHub/BrunoProject/finalKruskal/source/kruskalMetabolitesSource0525.txt'
# resultFileL = ['/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metabolites.A.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metabolites.AAndB.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metabolites.AAndBS.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metabolites.all.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metabolites.B.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metabolites.BAndBS.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metabolites.BS.052522aovResults.xls']
# dietsList = ['A', 'AAndB', 'AAndBS', 'all', 'B', 'BAndBS', 'BS']
# out = '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/finalStuff/metabolitesKruskalFinalFeatures'
# a = matchBackAndFilter(sourceFile, resultFileL, dietsList, out, 'Unnamed: 0')


# # lipidomics
# sourceFile = '/Users/chenlianfu/Documents/GitHub/BrunoProject/finalKruskal/source/kruskalLipidomicsSource0525.txt'
# resultFileL = ['/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Lipids.A.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Lipids.AAndB.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Lipids.AAndBS.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Lipids.all.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Lipids.B.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Lipids.BAndBS.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Lipids.BS.052522aovResults.xls']
# dietsList = ['A', 'AAndB', 'AAndBS', 'all', 'B', 'BAndBS', 'BS']
# out = '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/finalStuff/lipidomicsKruskalFinalFeatures'
# a = matchBackAndFilter(sourceFile, resultFileL, dietsList, out, 'BIOCHEMICAL')


# # metaphlan
# sourceFile = '/Users/chenlianfu/Documents/GitHub/BrunoProject/finalKruskal/source/kruskalMetaphlanRelAbSource0525.txt'
# resultFileL = ['/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metaphlan.A.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metaphlan.AAndB.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metaphlan.AAndBS.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metaphlan.all.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metaphlan.B.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metaphlan.BAndBS.052522aovResults.xls',
#             '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metaphlan.BS.052522aovResults.xls']
# dietsList = ['A', 'AAndB', 'AAndBS', 'all', 'B', 'BAndBS', 'BS']
# out = '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/finalStuff/metaphlanKruskalFinalFeatures'
# a = matchBackAndFilter(sourceFile, resultFileL, dietsList, out, 'clade_name')


# humann3
resultFileL = ['/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Humann3.A.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Humann3.AAndB.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Humann3.AAndBS.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Humann3.all.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Humann3.B.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Humann3.BAndBS.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Humann3.BS.052522aovResults.xls']
dietsList = ['A', 'AAndB', 'AAndBS', 'all', 'B', 'BAndBS', 'BS']
out = '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/finalStuff/humann3KruskalFinalFeatures'
ha = matchAllBackAndFilter(resultFileL, dietsList, out, '# Gene Family')

# metabolites
resultFileL = ['/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metabolites.A.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metabolites.AAndB.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metabolites.AAndBS.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metabolites.all.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metabolites.B.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metabolites.BAndBS.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metabolites.BS.052522aovResults.xls']
dietsList = ['A', 'AAndB', 'AAndBS', 'all', 'B', 'BAndBS', 'BS']
out = '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/finalStuff/metabolitesKruskalFinalFeatures'
ma = matchAllBackAndFilter(resultFileL, dietsList, out, 'Unnamed: 0')


# lipidomics
resultFileL = ['/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Lipids.A.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Lipids.AAndB.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Lipids.AAndBS.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Lipids.all.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Lipids.B.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Lipids.BAndBS.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Lipids.BS.052522aovResults.xls']
dietsList = ['A', 'AAndB', 'AAndBS', 'all', 'B', 'BAndBS', 'BS']
out = '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/finalStuff/lipidomicsKruskalFinalFeatures'
la = matchAllBackAndFilter(resultFileL, dietsList, out, 'BIOCHEMICAL')

# # metaphlan
resultFileL = ['/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metaphlan.A.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metaphlan.AAndB.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metaphlan.AAndBS.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metaphlan.all.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metaphlan.B.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metaphlan.BAndBS.052522aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/kruskalOutputs/kruskal.Metaphlan.BS.052522aovResults.xls']
dietsList = ['A', 'AAndB', 'AAndBS', 'all', 'B', 'BAndBS', 'BS']
out = '/Users/chenlianfu/Documents/GitHub/IBSFructanCodes/WGS_KruskalWallis/finalStuff/metaphlanKruskalFinalFeatures'
ta = matchAllBackAndFilter(resultFileL, dietsList, out, 'clade_name')

omics = ['humann3', 'lipidomics', 'metabolites', 'metaphlan']
numAll = [ha[0], la[0], ma[0], ta[0]]
numTol = [ha[1], la[1], ma[1], ta[1]]
numSen = [ha[2], la[2], ma[2], ta[2]]

df = pd.DataFrame(list(zip(omics, numAll, numTol, numSen)),
               columns =['omics', 'number of biomarkers in total', 'number enriched in tolerant kids', 'number enriched in sensitive kids'])

df.to_csv('kruskal_wallis_up_down_summary.csv')
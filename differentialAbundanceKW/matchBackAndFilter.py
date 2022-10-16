## AUTHOR: CHENLIAN FU
import pandas as pd
import os


def matchAllBackAndFilter(resultFileL, dietsList, out, taxaName):

    unique_biomarkers = []
    num_of_enriched_in_tolerant = 0
    num_of_enriched_in_sensitive = 0
    unique_biomarkers_omics = {} # collect in which omics/diets this biomarker is considered significant    
    resultFileWriter = open(out+"succintOutput.csv", 'wt')
    WRITEFIRSTLINE = False
    # loop thru all files
    for index, resultFile in enumerate(resultFileL):
        print("current file", resultFile.split("/")[-1])
        current_biomarkers = []
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
                    if FDRqVal <= 0.25 and foldChange != float('inf') and foldChange != 0:
                        
                    # if pVal <= 0.05 and FDRqVal <= 0.25 and foldChange != float('inf') and foldChange != 0:
                        heatmapFileWriter.write(','.join(heatmapFileLine))
                        heatmapSmallFileWriter.write(','.join([taxa, currentLine[6], currentLine[7]])+'\n')
                        if taxa not in unique_biomarkers_omics.keys():
                            unique_biomarkers_omics[taxa] = [dietsList[index]]
                            unique_biomarkers.append(taxa)
                            current_biomarkers.append(taxa)
                            resultFileWriter.write(','.join(resultFileLine)+'\n')
                            if foldChange>1:
                                num_of_enriched_in_tolerant +=1
                            else:
                                num_of_enriched_in_sensitive +=1
                        else:
                            unique_biomarkers_omics[taxa].append(dietsList[index])
            print("len of current_biomarkers", len(current_biomarkers),num_of_enriched_in_tolerant, num_of_enriched_in_sensitive)
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

    return len(unique_biomarkers), num_of_enriched_in_tolerant, num_of_enriched_in_sensitive, unique_biomarkers


# humann3
resultFileL = ['/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Humann3.all.101322aovResults.xls', 
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Humann3.A.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Humann3.AAndB.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Humann3.AAndBS.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Humann3.B.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Humann3.BAndBS.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Humann3.BS.101322aovResults.xls']
dietsList = ['all','A', 'AAndB', 'AAndBS', 'B', 'BAndBS', 'BS']
out = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/filteredOutputs/humann3KruskalFinalFeatures'
ha = matchAllBackAndFilter(resultFileL, dietsList, out, '# Gene Family')

# metabolites
resultFileL = ['/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metabolites.all.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metabolites.A.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metabolites.AAndB.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metabolites.AAndBS.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metabolites.B.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metabolites.BAndBS.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metabolites.BS.101322aovResults.xls']
dietsList = ['all', 'A', 'AAndB', 'AAndBS', 'B', 'BAndBS', 'BS']
out = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/filteredOutputs/metabolitesKruskalFinalFeatures'
ma = matchAllBackAndFilter(resultFileL, dietsList, out, 'Unnamed: 0')


# lipidomics
resultFileL = ['/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Lipids.all.101322aovResults.xls',
                '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Lipids.A.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Lipids.AAndB.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Lipids.AAndBS.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Lipids.B.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Lipids.BAndBS.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Lipids.BS.101322aovResults.xls']
dietsList = ['all', 'A', 'AAndB', 'AAndBS',  'B', 'BAndBS', 'BS']
out = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/filteredOutputs/lipidomicsKruskalFinalFeatures'
la = matchAllBackAndFilter(resultFileL, dietsList, out, 'BIOCHEMICAL')

# # metaphlan
resultFileL = ['/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metaphlan.all.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metaphlan.A.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metaphlan.AAndB.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metaphlan.AAndBS.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metaphlan.B.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metaphlan.BAndBS.101322aovResults.xls',
            '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/kruskalOutputs/kruskal.Metaphlan.BS.101322aovResults.xls']
dietsList = ['all', 'A', 'AAndB', 'AAndBS',  'B', 'BAndBS', 'BS']
out = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/filteredOutputs/metaphlanKruskalFinalFeatures'
ta = matchAllBackAndFilter(resultFileL, dietsList, out, 'clade_name')

omics = ['humann3', 'lipidomics', 'metabolites', 'metaphlan']
numAll = [ha[0], la[0], ma[0], ta[0]]
numTol = [ha[1], la[1], ma[1], ta[1]]
numSen = [ha[2], la[2], ma[2], ta[2]]

df = pd.DataFrame(list(zip(omics, numAll, numTol, numSen)),
               columns =['omics', 'number of biomarkers in total', 'number enriched in tolerant kids', 'number enriched in sensitive kids'])

df.to_csv('kruskal_wallis_up_down_summary.csv')
## AUTHOR: CHENLIAN FU

from copy import deepcopy
import numpy as np
import pandas as pd
import statsmodels.stats.multitest as tests
import scipy.stats as stats
import math
import time
import os

def WGSTTestwFDR(source, differentiator, outRoot, taxoVar, diet=None):
    """Automatically generate one-way aov (kruskal wallis) outputs for WGS-formatted files
    Args:
        source ([file address]): file address of the input file
        varName ([str]): e.g.'FRUCTANSENSITIVE'
        outRoot ([str]): output file prefix
        taxoVar ([str]): name of the taxonomy
        diet ([str or list]): dietary subsets for data we are considering

    Returns:
        generates a xls file summarizing your one-way anova result
    """
    # read source files
    file = open(source, "r")
    # initiate variables
    index = 0
    fructansensitiveDF = pd.DataFrame()
    # dynamically writing temporary outputs to avoid big memory usage
    aovOutputWriter = open('./kruskalOutputs/' +
                           outRoot + 'tempaovResults.xls', 'wt')
    heatmapSampleMatcher = open('./kruskalOutputs/' +
                           outRoot + 'heatmapSampleMatch.csv', 'wt')
    WRITEFIRSTLINE = False
    rawPL = []
    var1IndexL = []
    foldChangesL = []
    # calculate raw p
    while(True):
        line = file.readline()
        lineContent = line.split('\t')
        index += 1
        try:
            if lineContent[0] == taxoVar:
                currentRawDietIndices = lineContent[1:]
                heatmapSampleL = lineContent[1:]
                heatmapSampleMatcher.write(','.join(heatmapSampleL))
                # enable statistical tests for all dietary subsets
                if diet != None:
                    if type(diet) is not list:
                        if diet in ['A', 'B']:
                            currentDietIndices = [i for i, value in enumerate(
                                currentRawDietIndices) if diet in value]
                        if diet == 'BS':
                            currentDietIndices = [i for i, value in enumerate(
                                currentRawDietIndices) if len(value) == 4]
                    else:
                        currentDietIndices = []
                        if 'A' in diet:
                            currentDietIndices += [i for i, value in enumerate(
                                currentRawDietIndices) if 'A' in value]
                        if 'B' in diet:
                            currentDietIndices += [i for i, value in enumerate(
                                currentRawDietIndices) if 'B' in value]
                        if 'BS' in diet:
                            currentDietIndices += [i for i, value in enumerate(
                                currentRawDietIndices) if len(value) == 4]
                else:
                    # the case where all diets are covered
                    currentDietIndices = currentRawDietIndices
            
            # get the fructan sensitivity y variable from the second line
            elif lineContent[0] == differentiator:
                fructansensitiveDF = pd.DataFrame(np.array(lineContent).T)
            
            # conduct raw statistical tests
            elif lineContent[0] not in [taxoVar, differentiator]:
                
                # 0608 edit
                lineContent = lineContent
                
                # print("lineContent", lineContent)
               
                # try:
                currentVarName = lineContent[0]
                lineContent[0] = 'ASN' + str(index)
                currentLineDF = pd.DataFrame(np.array(lineContent).T)
                testDF = pd.concat(
                    [fructansensitiveDF, currentLineDF], axis=1)
                testDF.reset_index(drop=True)

                testDF.columns = [differentiator, 'ASN' + str(index)]


                testDF = testDF[1:]

                testDF.to_csv(outRoot + str(index) + 'try.csv', index=False)
                newdf = pd.read_csv(outRoot + str(index) + 'try.csv')

                # stats starts here
                if var1IndexL == []:
                    var1IndexL = newdf.index[newdf[differentiator] == 1].tolist(
                    )
                    var0IndexL = newdf.index[newdf[differentiator] == 0].tolist(
                    )
                currentASNCol = newdf[lineContent[0]].dropna()


                print("currentASNCol", len(currentASNCol))
                print("first 3 elements", currentASNCol[0])
               
                if (max(var1IndexL) > len(currentASNCol)-2) or (max(var1IndexL) > len(currentASNCol)-2):
                    while max(var1IndexL) > len(currentASNCol)-2:
                        var1IndexL.remove(max(var1IndexL))
                    while max(var0IndexL) > len(currentASNCol)-2:
                        var0IndexL.remove(max(var0IndexL))
                # filter to make sure the 1/0 indices are in diet index list
                if diet != None:
                    var1IndexL = [
                        index for index in var1IndexL if index in currentDietIndices]
                    var0IndexL = [
                        index for index in var0IndexL if index in currentDietIndices]
                group1 = list(currentASNCol[var1IndexL].dropna())
                group0 = list(currentASNCol[var0IndexL].dropna())
                if WRITEFIRSTLINE == False:
                    group1_names = ["sensitiveSample_"+str(i+1) for i in range(len(group1))]
                    group0_names = ["tolerantSample_"+str(j+1) for j in range(len(group0))]
                    firstLineText = 'AscensionNum\tTaxoName\tpValue\tCohenEffectSize\tfoldChanges\tMean0_tolerant\tmean1_sensitive\t'+ '\t'.join(group1_names)+'\t'+'\t'.join(group0_names)
                    aovOutputWriter.write(firstLineText+'\n')

                    heatMapNameL = []
                    sensitiveIndex = 1
                    tolerantIndex = 1
                    for indexHeat in range(len(currentASNCol)):
                        if indexHeat in var1IndexL:
                            heatMapNameL.append("sensitiveSample_"+str(sensitiveIndex))
                            sensitiveIndex += 1
                        else:
                            heatMapNameL.append("tolerantSample_"+str(tolerantIndex))
                            tolerantIndex += 1
                    heatmapSampleMatcher.write(','.join(heatMapNameL))
                    WRITEFIRSTLINE = True
                    heatmapSampleMatcher.close()
                print(group1)
                print(group0)
                print('totalSize', len(group1)+len(group0))
                print("____________")
                try:
                    if ";" in list(group1)[-1]:
                        group1 = list(group1)[:-1]
                except TypeError:
                    print("Go on")
                try:
                    if ";" in list(group0)[-1]:
                        group1 = list(group0)[:-1]
                except TypeError:
                    print("Go on")
                # statistical test raw pval
                group1 = np.array(group1).astype(np.float)
                group0 = np.array(group0).astype(np.float)
                try:
                    pVal = stats.kruskal(group1, group0)[1]
                    print('tried')
                except:
                    pVal = -float('inf')
                    print(lineContent[0])
                    print(group0, group1)
                    print("Two groups all the same")
                os.remove(outRoot + str(index) + 'try.csv')
                combined = np.append(group1, group0)
                if pVal > 0:
                    # cohan effect size
                    # reference: https://books.google.com/books?id=HFphCwAAQBAJ&pg=PA247&dq=anova+%22cohen%27s+d%22&hl=en&sa=X&ved=0ahUKEwjGrrvHob7aAhXDqFkKHe27C9MQuwUILTAA#v=onepage&q=anova%20%22cohen's%20d%22&f=false
                    mean1 = stats.tmean(group1)
                    mean0 = stats.tmean(group0)
                    std = stats.tstd(combined)
                    CohanEffectSize = abs(mean1-mean0)/(2*std)
                    foldChanges = abs(mean0)/abs(mean1)
                    rawPL.append(pVal)
                    foldChangesL.append(foldChanges)
                    print(lineContent[0], currentVarName, " pval: ", pVal,
                          " Effect size: ", CohanEffectSize, foldChanges)
                    # placeHolder = ["%s"]*(7+len(group1)+len(group0))
                    currentLineL = [lineContent[0], currentVarName, pVal, CohanEffectSize, foldChanges, mean0, mean1]+list(group1)+list(group0)

                    currentLineL = [str(el) for el in currentLineL]
                    currentLine = "\t".join(currentLineL)
                    print(currentLine)
                    aovOutputWriter.write(currentLine+'\n')
        except:
            print("Temporarily stopped")
            break
    aovOutputWriter.close()

    # get fdr corrected p list
    fdrRawPL = list(tests.fdrcorrection(rawPL)[1])
    # calculate fdr corrected p values and judge whether significant or not
    aovOutputWriter = open('./kruskalOutputs/' +
                           outRoot + 'aovResults.xls', 'wt')
    time.sleep(5)
    tempaovOutputReader = open(
        './kruskalOutputs/'+outRoot + 'tempaovResults.xls', "r")
    fdrCounter = 0

    # produce actual outputs
    for line in tempaovOutputReader:
        lineContent = line.split('\t')
        if lineContent[0] == 'AscensionNum':
            newLineContent = lineContent
            newLineContent.insert(3, 'fdrCorrectedpVal')
            aovOutputWriter.write("\t".join(newLineContent))
        elif lineContent[0][:3] == 'ASN':
            rawP = lineContent[2]
            currentFDRP = fdrRawPL[fdrCounter]
            fdrCounter += 1
            newLineContent = lineContent
            newLineContent.insert(3, currentFDRP)
            newLineContent = [str(el) for el in newLineContent]
            print(currentFDRP)
            aovOutputWriter.write("\t".join(newLineContent))

    aovOutputWriter.close()
    os.system('say "Your program has finished"')
    return


LipidomicsSource = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/source/1011LipidomicsX.txt'
WGSTTestwFDR(LipidomicsSource, 'FRUCTANSENSITIVE', 'kruskal.Lipids.A.101322', '# Gene Family', diet='A')
WGSTTestwFDR(LipidomicsSource, 'FRUCTANSENSITIVE', 'kruskal.Lipids.B.101322', '# Gene Family', diet='B')
WGSTTestwFDR(LipidomicsSource, 'FRUCTANSENSITIVE', 'kruskal.Lipids.BS.101322', '# Gene Family', diet='BS')
WGSTTestwFDR(LipidomicsSource, 'FRUCTANSENSITIVE', 'kruskal.Lipids.AAndB.101322', '# Gene Family', diet=['A', 'B'])
WGSTTestwFDR(LipidomicsSource, 'FRUCTANSENSITIVE', 'kruskal.Lipids.AAndBS.101322', '# Gene Family', diet=['A', 'BS'])
WGSTTestwFDR(LipidomicsSource, 'FRUCTANSENSITIVE', 'kruskal.Lipids.BAndBS.101322', '# Gene Family', diet=['B', 'BS'])


Humann3Source = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/source/1011humann3X.txt'
WGSTTestwFDR(Humann3Source, 'FRUCTANSENSITIVE', 'kruskal.Humann3.all.101322', '# Gene Family')
WGSTTestwFDR(Humann3Source, 'FRUCTANSENSITIVE', 'kruskal.Humann3.A.101322', '# Gene Family', diet='A')
WGSTTestwFDR(Humann3Source, 'FRUCTANSENSITIVE', 'kruskal.Humann3.B.101322', '# Gene Family', diet='B')
WGSTTestwFDR(Humann3Source, 'FRUCTANSENSITIVE', 'kruskal.Humann3.BS.101322', '# Gene Family', diet='BS')
WGSTTestwFDR(Humann3Source, 'FRUCTANSENSITIVE', 'kruskal.Humann3.AAndB.101322', '# Gene Family', diet=['A', 'B'])
WGSTTestwFDR(Humann3Source, 'FRUCTANSENSITIVE', 'kruskal.Humann3.AAndBS.101322', '# Gene Family', diet=['A', 'BS'])
WGSTTestwFDR(Humann3Source, 'FRUCTANSENSITIVE', 'kruskal.Humann3.BAndBS.101322', '# Gene Family', diet=['B', 'BS'])


MetabolitesSource = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/source/1011MetabolitesX.txt'
print("WTF")
WGSTTestwFDR(MetabolitesSource, 'FRUCTANSENSITIVE', 'kruskal.Metabolites.all.101322', '# Gene Family')
WGSTTestwFDR(MetabolitesSource, 'FRUCTANSENSITIVE', 'kruskal.Metabolites.A.101322', '# Gene Family', diet='A')
WGSTTestwFDR(MetabolitesSource, 'FRUCTANSENSITIVE', 'kruskal.Metabolites.B.101322', '# Gene Family', diet='B')
WGSTTestwFDR(MetabolitesSource, 'FRUCTANSENSITIVE', 'kruskal.Metabolites.BS.101322', '# Gene Family', diet='BS')
WGSTTestwFDR(MetabolitesSource, 'FRUCTANSENSITIVE', 'kruskal.Metabolites.AAndB.101322', '# Gene Family', diet=['A', 'B'])
WGSTTestwFDR(MetabolitesSource, 'FRUCTANSENSITIVE', 'kruskal.Metabolites.AAndBS.101322', '# Gene Family', diet=['A', 'BS'])
WGSTTestwFDR(MetabolitesSource, 'FRUCTANSENSITIVE', 'kruskal.Metabolites.BAndBS.101322', '# Gene Family', diet=['B', 'BS'])


MetaphlanSource = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/differentialAbundanceKW/source/1011MetaphlanX.txt'
print("WTF")
WGSTTestwFDR(MetaphlanSource, 'FRUCTANSENSITIVE', 'kruskal.Metaphlan.all.101322', '# Gene Family')
WGSTTestwFDR(MetaphlanSource, 'FRUCTANSENSITIVE', 'kruskal.Metaphlan.A.101322', '# Gene Family', diet='A')
WGSTTestwFDR(MetaphlanSource, 'FRUCTANSENSITIVE', 'kruskal.Metaphlan.B.101322', '# Gene Family', diet='B')
WGSTTestwFDR(MetaphlanSource, 'FRUCTANSENSITIVE', 'kruskal.Metaphlan.BS.101322', '# Gene Family', diet='BS')
WGSTTestwFDR(MetaphlanSource, 'FRUCTANSENSITIVE', 'kruskal.Metaphlan.AAndB.101322', '# Gene Family', diet=['A', 'B'])
WGSTTestwFDR(MetaphlanSource, 'FRUCTANSENSITIVE', 'kruskal.Metaphlan.AAndBS.101322', '# Gene Family', diet=['A', 'BS'])
WGSTTestwFDR(MetaphlanSource, 'FRUCTANSENSITIVE', 'kruskal.Metaphlan.BAndBS.101322', '# Gene Family', diet=['B', 'BS'])



# # TRYING SOMETHING
# sourceAAA = '/Users/chenlianfu/Documents/GitHub/BrunoProject/sandy3-overlapViolinSource.txt'
# WGSTTestwFDR(sourceAAA, 'FRUCTANSENSITIVE', 'ADCDSCSXXS', '# Gene Family') # DONE
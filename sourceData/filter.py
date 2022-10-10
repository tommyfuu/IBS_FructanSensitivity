import time
import pandas as pd

source = '/Users/chenlianfu/Documents/Github/BrunoAdditional/combinedWGSBruno.txt'


def WGSFilder(source, outRoot):
    """
    """
    file = open(source, "r")
    index = 0
    emptyList = [['0']*97][0]
    aovOutputWriter = open(outRoot + 'filtered.txt', 'wt')
    # calculate raw p
    while(True):
        print(index)
        index += 1
        # for index in range(0, 3):
        line = file.readline()
        lineContent = line.split('\t')
        lineContent = '	'.join(lineContent).split()
        if len(lineContent) == 98 and lineContent[1:] != emptyList:
            newlineContent = '	'.join(lineContent) + '	\n'
            aovOutputWriter.write(newlineContent)
            # time.sleep(3)
        else:
            print(lineContent)
    aovOutputWriter.close()
    return


source = 'DiffCoExClean/source/here.txt'


def getThing(source, combination):
    file = open(source, "r")
    aovOutputWriter = open('diet.txt', 'wt')
    # while(True):
    for index in range(0, 100):
        line = file.readline()
        lineContent = line.split('\n')[0]
        if "A" in lineContent:
            aovOutputWriter.write('1\n')
        elif "B" in lineContent:
            aovOutputWriter.write('0\n')
        print(lineContent)
    aovOutputWriter.close()
    return



def filterByDiet(sourceX, outputRoot, biomarkerType, sourceY, diets):
    file = open(sourceX, "r")
    aovOutputWriter = open(outputRoot+'.txt', 'wt')
    outputSamples = []
    # outputSamplesBinary = []  # order: [A, B] = [A, BS] = [B, BS] =[1, 0]
    usefulIndices = []
    # dietComboBarcode = {('A', 'B'): {'A': 1, 'B': 0}, ('A', 'BS'): {'A': 1, 'BS': 0}, (
    #     'B', 'BS'): {'B': 1, 'BS': 0}}
    # currentBarcode = dietComboBarcode[dietCombo]
    for i in range(10000):
        line = file.readline()
        if len(line.split('\t')) == 1:
            break
        else:
            lineContent = line.split('\t')
            # print(lineContent[0])
            if biomarkerType == lineContent[0]:
                # print(biomarkerType)
                samples = lineContent[1:]
                # print("samples", samples)
                usefulIndices.append(0)
                index = 0
                # print("WHAT THE")
                for sample in samples:
                    index += 1
                    if 'A' in sample and 'A' in diets:
                        outputSamples.append(sample)
                        # outputSamplesBinary.append(currentBarcode['A'])
                        usefulIndices.append(index)
                    if 'B' in sample and 'B' in diets:
                        outputSamples.append(sample)
                        # outputSamplesBinary.append(currentBarcode['B'])
                        usefulIndices.append(index)
                    if 'A' not in sample and 'B' not in sample and 'BS' in diets:
                        if '\n' not in sample:
                            outputSamples.append(sample)
                        else: 
                            outputSamples.append(sample.split('\n')[0])
                        # outputSamplesBinary.append(currentBarcode['BS'])
                        usefulIndices.append(index)
                print(outputSamples, usefulIndices)
                usefulLineContent = [biomarkerType]
                usefulLineContent.extend(outputSamples)
                # print("usefulLineContent", usefulLineContent)
                lineWritten = '\t'.join(usefulLineContent)
                # print(lineWritten)
                aovOutputWriter.write(lineWritten+'\n')
            else:
                usefulLineContent = [lineContent[index] for index in usefulIndices]
                print(usefulLineContent)
                usefulLineContent[-1] = usefulLineContent[-1].split('\n')[0]
                lineWritten = '\t'.join(usefulLineContent)
                # print(lineWritten)
                aovOutputWriter.write(lineWritten+'\n')
    aovOutputWriter.close()
    
    df = pd.read_excel(sourceY)
    index_list = usefulIndices[1:]
    index_list = [i-1 for i in index_list]
    # print(index_list)
    df = df[df.index.isin(index_list)]
    df.to_excel(outputRoot+'_y.xlsx', index=False)
    return



# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/metaphlanRelAb_0331.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData524/4omicsSubs/metaphlanByDietAAndB'
# biomarkerType = 'clade_name'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# filterByDiet(source, outputRoot, biomarkerType, sourceY)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/4omicsAll0331X.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData524/4omicsSubs/humann3A0331'
# biomarkerType = '# Gene Family'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['A']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/4omicsAll0331X.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/humann3B0331'
# biomarkerType = '# Gene Family'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['B']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/4omicsAll0331X.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/humann3BS0331'
# biomarkerType = '# Gene Family'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['BS']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/4omicsAll0331X.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/4omicsSubs/4omicsAandB0524'
# biomarkerType = '# Gene Family'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['A', 'B']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/4omicsAll0331X.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/4omicsSubs/4omicsAandBS0524'
# biomarkerType = '# Gene Family'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['A', 'BS']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/4omicsAll0331X.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/4omicsSubs/4omicsBandBS0524'
# biomarkerType = '# Gene Family'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['B', 'BS']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)





# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/humann3ECX_0331.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/humann3A0620'
# biomarkerType = '# Gene Family'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['A']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/humann3ECX_0331.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/humann3B0620'
# biomarkerType = '# Gene Family'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['B']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/humann3ECX_0331.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/humann3BS0620'
# biomarkerType = '# Gene Family'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['BS']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/humann3ECX_0331.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/humann3AandB0620'
# biomarkerType = '# Gene Family'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['A', 'B']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/humann3ECX_0331.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/humann3AandBS0620'
# biomarkerType = '# Gene Family'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['A', 'BS']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/humann3ECX_0331.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/humann3BandBS0620'
# biomarkerType = '# Gene Family'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['B', 'BS']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)


# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/metabolitesX.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/metabolitesA0620'
# biomarkerType = 'Unnamed: 0'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['A']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/metabolitesX.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/metabolitesB0620'
# biomarkerType = 'Unnamed: 0'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['B']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/metabolitesX.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/metabolitesBS0620'
# biomarkerType = 'Unnamed: 0'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['BS']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/metabolitesX.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/metabolitesAandB0620'
# biomarkerType = 'Unnamed: 0'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['A', 'B']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/metabolitesX.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/metabolitesAandBS0620'
# biomarkerType = 'Unnamed: 0'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['A', 'BS']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/metabolitesX.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/metabolitesBandBS0620'
# biomarkerType = 'Unnamed: 0'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['B', 'BS']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/lipidomics0331X.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/lipidomicsA0620'
# biomarkerType = 'BIOCHEMICAL'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['A']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/lipidomics0331X.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/lipidomicsB0620'
# biomarkerType = 'BIOCHEMICAL'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['B']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/lipidomics0331X.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/lipidomicsBS0620'
# biomarkerType = 'BIOCHEMICAL'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['BS']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/lipidomics0331X.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/lipidomicsAandB0620'
# biomarkerType = 'BIOCHEMICAL'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['A', 'B']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/lipidomics0331X.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/lipidomicsAandBS0620'
# biomarkerType = 'BIOCHEMICAL'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['A', 'BS']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/lipidomics0331X.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/lipidomicsBandBS0620'
# biomarkerType = 'BIOCHEMICAL'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['B', 'BS']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/metaphlanRelAb_0331.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/metaphlanA0620'
# biomarkerType = 'clade_name'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['A']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/metaphlanRelAb_0331.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/metaphlanB0620'
# biomarkerType = 'clade_name'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['B']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/metaphlanRelAb_0331.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/metaphlanBS0620'
# biomarkerType = 'clade_name'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['BS']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/metaphlanRelAb_0331.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/metaphlanAandB0620'
# biomarkerType = 'clade_name'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['A', 'B']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/metaphlanRelAb_0331.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/metaphlanAandBS0620'
# biomarkerType = 'clade_name'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['A', 'BS']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoMLData0331/metaphlanRelAb_0331.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0620finalDiffCoExSource/metaphlanBandBS0620'
# biomarkerType = 'clade_name'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['B', 'BS']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/2overlapFeaturesRFSource.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0805RFSource/0805RFSourceA0620'
# biomarkerType = '# Gene Family'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['A']
# print("AAAAAAAAAA")
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/2overlapFeaturesRFSource.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0805RFSource/0805RFSourceB0620'
# biomarkerType = '# Gene Family'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['B']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/2overlapFeaturesRFSource.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0805RFSource/0805RFSourceBS0620'
# biomarkerType = '# Gene Family'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['BS']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/2overlapFeaturesRFSource.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0805RFSource/0805RFSourceAandB0620'
# biomarkerType = '# Gene Family'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['A', 'B']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/2overlapFeaturesRFSource.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0805RFSource/0805RFSourceAandBS0620'
# biomarkerType = '# Gene Family'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['A', 'BS']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

# source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/2overlapFeaturesRFSource.txt'
# outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0805RFSource/0805RFSourceBandBS0620'
# biomarkerType = '# Gene Family'
# sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
# diets = ['B', 'BS']
# filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)


### 0911
source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0911_2OverlapX.txt'
outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0911RFSource/0911RFSourceA'
biomarkerType = '# Gene Family'
sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
diets = ['A']
print("AAAAAAAAAA")
filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0911_2OverlapX.txt'
outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0911RFSource/0911RFSourceB'
biomarkerType = '# Gene Family'
sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
diets = ['B']
filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0911_2OverlapX.txt'
outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0911RFSource/0911RFSourceBS'
biomarkerType = '# Gene Family'
sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
diets = ['BS']
filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0911_2OverlapX.txt'
outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0911RFSource/0911RFSourceAandB'
biomarkerType = '# Gene Family'
sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
diets = ['A', 'B']
filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0911_2OverlapX.txt'
outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0911RFSource/0911RFSourceAandBS'
biomarkerType = '# Gene Family'
sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
diets = ['A', 'BS']
filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

source = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0911_2OverlapX.txt'
outputRoot = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/0911RFSource/0911RFSourceBandBS'
biomarkerType = '# Gene Family'
sourceY = '/Users/chenlianfu/Documents/GitHub/BrunoProject/brunoML0524Final/BRUWHYDEADALLY.xlsx'
diets = ['B', 'BS']
filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)
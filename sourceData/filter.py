import time
import pandas as pd

source = '/Users/chenlianfu/Documents/Github/BrunoAdditional/combinedWGSBruno.txt'



def filterByDiet(sourceX, outputRoot, biomarkerType, sourceY, diets):
    file = open(sourceX, "r")
    aovOutputWriter = open(outputRoot+'.txt', 'wt')
    outputSamples = []
    usefulIndices = []
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
                    if 'A' not in sample and 'B' not in sample and 'BL' in diets:
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
                
                if not 'B' in diets:
                    print(lineContent[-2:])
                    aovOutputWriter.write(lineWritten+'\n')
                else:
                    aovOutputWriter.write(lineWritten)
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


### 0911
source = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/4omicsAllX.txt'
outputRoot = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/dietSubsets/1009_A'
biomarkerType = '# Gene Family'
sourceY = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/AllDietsY.xlsx'
diets = ['A']
filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

source = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/4omicsAllX.txt'
outputRoot = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/dietSubsets/1009_B'
biomarkerType = '# Gene Family'
sourceY = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/AllDietsY.xlsx'
diets = ['B']
filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

source = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/4omicsAllX.txt'
outputRoot = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/dietSubsets/1009_BL'
biomarkerType = '# Gene Family'
sourceY = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/AllDietsY.xlsx'
diets = ['BL']
filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

source = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/4omicsAllX.txt'
outputRoot = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/dietSubsets/1009_A_B'
biomarkerType = '# Gene Family'
sourceY = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/AllDietsY.xlsx'
diets = ['A', 'B']
filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

source = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/4omicsAllX.txt'
outputRoot = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/dietSubsets/1009_A_BL'
biomarkerType = '# Gene Family'
sourceY = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/AllDietsY.xlsx'
diets = ['A', 'BL']
filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)

source = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/4omicsAllX.txt'
outputRoot = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/dietSubsets/1009_B_BL'
biomarkerType = '# Gene Family'
sourceY = '/Users/chenlianfu/Documents/GitHub/IBS_FructanSensitivity/sourceData/AllDietsY.xlsx'
diets = ['B', 'BL']
filterByDiet(source, outputRoot, biomarkerType, sourceY, diets)
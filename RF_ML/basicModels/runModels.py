from cartModel import *
from knModel import *
from LDAModel import *
from LogisticModel import *
from svmModel import *

addressX = '/home/fuc/brunoML/source/4omics/combined4omicsX.txt'
addressY = '/home/fuc/brunoML/source/fructanYhumann3.xlsx'

cartEvaluate(addressX, addressY)
knEvaluate(addressX, addressY)
LDAEvaluate(addressX, addressY)
LogisticEvaluate(addressX, addressY)
svmEvaluate(addressX, addressY)

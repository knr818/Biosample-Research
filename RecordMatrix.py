# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 14:02:56 2019

@author: Connor
"""

import numpy as np
    
def toInt(string):
    try:
        returnVal = int(string)
    except ValueError:
        returnVal = string
    return returnVal

def isInt(value):
    try:
        int(value)
        return 1
    except ValueError:
        return 0

class RecordMatrix:
    def __init__(self, infile, classificationValue, classificationIndex):
        Data = []
        SumVertex = np.zeros((5), dtype=int)
        RecordsPerSample = np.zeros((5), dtype=int)
        for line in infile:
            curLine = line.rstrip().split(',')
            if int(curLine[classificationIndex])==classificationValue:
                #Add to Data set
                row = [toInt(curLine[val]) for val in range(5)]
                Data.append(row)
                #Add to mean vertex
                validSamples = np.asarray([isInt(curLine[val]) for val in range(5)])
                RecordsPerSample = np.add(RecordsPerSample, validSamples)
                ValuesQuestionMarkRemoved = (np.where(validSamples==1, np.asarray(row, dtype=object), 0))
                SumVertex = np.add(SumVertex, ValuesQuestionMarkRemoved)
        #Take full sum and calculate by number of each biosample to get mean
        MeanVertex = np.divide(SumVertex, RecordsPerSample)

        #Create Data Array
        self.DataArray = np.asarray(Data, dtype=object)
        #Replace all '?'s with the appropriate value from the Mean Vertex
        for index in np.asarray(np.where(self.DataArray == '?')).T:
            self.DataArray[index[0], index[1]] = MeanVertex[index[1]]
        #Now it really is all numbers! Cast it properly
        self.DataArray = self.DataArray.astype(float)
        
    def getMeanVertex(self):
        return np.mean(self.DataArray, axis=0, dtype=float)
        
    def getDeviationVertex(self):
        return np.std(self.DataArray, axis=0)

    def getVariationVertex(self):
        return np.var(self.DataArray, axis=0)

    def getCovarianceMatrix(self):
        return np.cov(self.DataArray, rowvar=False)

'''
infile = open("mammo.txt", 'r')
pTest = RecordMatrix(infile, 0, 5)
print(pTest.getDeviationVertex())
print(pTest.getVariationVertex())
print(pTest.getCovarianceMatrix())
'''
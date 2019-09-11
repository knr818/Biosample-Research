# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 20:00:22 2019

@author: Connor
"""

#import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm
#import scipy
#import sympy
#from sympy import symbols, diff
#from scipy import  diff
#from scipy.stats import t
#from scipy.stats import chi2
#from numpy import vstack
#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import RecordMatrix
import sys
#import itertools
#from matplotlib import pyplot
#import matplotlib.patches as mpatches



#***********************************************************************

def BiomarkerCombination(HealthyData, DiseasedData, t):
    diseasedVarianceMatrix = t*DiseasedData.getCovarianceMatrix()
    healthyVarianceMatrix = (1-t)*HealthyData.getCovarianceMatrix()
    
    sumDeviation = np.add(diseasedVarianceMatrix, healthyVarianceMatrix)
    deviationTerm = np.linalg.inv(sumDeviation)
    
    meanTerm = np.subtract(DiseasedData.getMeanVertex(),
                           HealthyData.getMeanVertex())

    b = np.dot(deviationTerm, meanTerm)
    return b
    

def OptimalCutpoint(HealthyData, DiseasedData, b):
    
    #Calculate C's numerator: b'*meanD*sqrt(b'*covarH*b) + b'*meanH*sqrt(b'*covarD*b)
    numeratorTerm1 = np.dot(b, DiseasedData.getMeanVertex()) * math.sqrt(np.dot(np.dot(b, HealthyData.getCovarianceMatrix()), b)  )

    numeratorTerm2 = np.dot(b, HealthyData.getMeanVertex()) * math.sqrt(np.dot(np.dot(b, DiseasedData.getCovarianceMatrix()), b)  )
    
    #Calculate C's denominator: (b'*covarH*b) + sqrt(b'*covarD*b)
    denominatorTerm = math.sqrt(np.dot(np.dot(b, DiseasedData.getCovarianceMatrix()), b)) + math.sqrt(np.dot(np.dot(b, HealthyData.getCovarianceMatrix()), b)  )

    if(denominatorTerm != 0):
        return (numeratorTerm1 + numeratorTerm2)/denominatorTerm
    else:
        raise ZeroDivisionError("The denominator for the optimal cutpoint is zero!")

def FalsePositiveRate(HealthyData, b, optimalCutpoint):
    numeratorTerm = optimalCutpoint - np.dot(b, HealthyData.getMeanVertex())
    denominatorTerm = math.sqrt(np.dot(np.dot(b, HealthyData.getCovarianceMatrix()), b))
    return (1-norm.cdf(numeratorTerm/denominatorTerm))

def TruePositiveRate(DiseasedData, b, optimalCutpoint):
    numeratorTerm = np.dot(b, DiseasedData.getMeanVertex()) - optimalCutpoint
    denominatorTerm = math.sqrt(np.dot(np.dot(b, DiseasedData.getCovarianceMatrix()), b))
    return (norm.cdf(numeratorTerm/denominatorTerm))

#******************************************************************************

def main():
    #Import data
    infile = open("mammo.txt", 'r')
    HealthyData = RecordMatrix.RecordMatrix(infile, 0, 5)
    
    infile = open("mammo.txt", 'r')
    DiseasedData = RecordMatrix.RecordMatrix(infile, 1, 5)
    
    '''
    b = 0
    c = 0
    highT = 0
    for t in range(10001):
        if(t % 100 == 0):
            print(t)
            print(highT)
            print(c)
        tempB = BiomarkerCombination(HealthyData, DiseasedData, t/10000)
        tempC = OptimalCutpoint(HealthyData, DiseasedData, tempB)
        if tempC > c:
            c = tempC
            b = tempB
            highT = t
    '''
    
    b = BiomarkerCombination(HealthyData, DiseasedData, 0.5)
    c = OptimalCutpoint(HealthyData, DiseasedData, b)
    
    FPR = FalsePositiveRate(HealthyData, b, c)
    TPR = TruePositiveRate(DiseasedData, b, c)
    
    print("B=" + str(b))
    print("C=" + str(c))
    print(FPR)
    print(TPR)
    
    JoudenIndex = TPR - FPR
    print(JoudenIndex)
    '''
    print("Optimal cutpoint: ", c)
    print(TPR)
    print(FPR)
    print(JoudenIndex)
    '''
    '''
    print(DiseasedData.DataArray)
    for variableNum in range(5):
        plt.figure(variableNum)
        np.set_printoptions(threshold=sys.maxsize)
        #Printing Healthy Data
        x = np.linspace(HealthyData.getMeanVertex()[variableNum] - 3*HealthyData.getDeviationVertex()[variableNum], HealthyData.getMeanVertex()[variableNum] + 3*HealthyData.getDeviationVertex()[variableNum], 100)
        plt.plot(x, norm.pdf(x, HealthyData.getMeanVertex()[variableNum], HealthyData.getDeviationVertex()[variableNum]))
        y = np.linspace(DiseasedData.getMeanVertex()[variableNum] - 3*DiseasedData.getDeviationVertex()[variableNum], DiseasedData.getMeanVertex()[variableNum] + 3*DiseasedData.getDeviationVertex()[variableNum], 100)
        print(y)
        plt.plot(y, norm.pdf(y, DiseasedData.getMeanVertex()[variableNum], DiseasedData.getDeviationVertex()[variableNum]))
        #plt.show()
    '''

main()
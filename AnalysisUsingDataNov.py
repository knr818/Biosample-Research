#import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
#import sympy
from scipy.stats import norm
#from sympy import symbols, diff
#from scipy import  diff
#from scipy.stats import t
#from scipy.stats import chi2
#from numpy import vstack
#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
#import itertools
#from matplotlib import pyplot
#import matplotlib.patches as mpatches
# I checked the Youden index for following data
alpha=0.1
#HealthyInfo=[6.5,0.09]
#DiseasedInfo=[6.753,0.25]
#SampleMeanH = HealthyInfo[0]
#SamplevarianceH = HealthyInfo[1]
#SampleMeanD = DiseasedInfo[0]
#SamplevarianceD = DiseasedInfo[1]
K=10
NumSimulation = 10 # number of simulations
J0=0.2
Jarray= [None]*K
Carray= [None]*K
Deltaarray = [None]*NumSimulation
Nitaarray = [None]*NumSimulation
lengthCI = [None]*NumSimulation


# 'GetSampleMeanVariance' function gives the sample means and the variances for given diseased and healthy people
def GetSampleMeanVariance(HealthyData,DiseasedData,nD,nH):
    SampleMeanD=sum(DiseasedData) / nD #sample mean for diseased
    sumVarD = 0
    for i in range(nD):
        sumVarD = sumVarD + (DiseasedData[i] - SampleMeanD) ** 2
        SamplevarianceD = sumVarD / (nD-1) #sample variance  for diseased

    SampleMeanH = sum(HealthyData) / nH  # sample mean for healthy
    sumVarH = 0
    for i in range(nH):
        sumVarH = sumVarH + (HealthyData[i] - SampleMeanH) ** 2
        SamplevarianceH = sumVarH / (nH-1) #sample variance  for healthy
    return [SampleMeanD,SamplevarianceD,SampleMeanH,SamplevarianceH];
#***********************************************************************
#****************************************************************************************
def ClassicalCIforJ(SampleMeanD, SamplevarianceD, SampleMeanH, SamplevarianceH,nD,nH,alpha):
    a= SampleMeanD-SampleMeanH
    STSamplevarianceD = math.sqrt(SamplevarianceD)
    STSamplevarianceH = math.sqrt(SamplevarianceH)
    b=STSamplevarianceD/STSamplevarianceH
    term=b*b-1
    rad=a*a+(b*b-1)*SamplevarianceH*math.log(b * b,math.e)
    C = (SampleMeanH * (b * b - 1) - a + b * math.sqrt(
        a * a + (b * b - 1) * (SamplevarianceH) * math.log(b * b, math.e))) / (
                    b * b - 1)
    DabbaCMuD=(-1+a*b*(rad)**(-0.5))/term
    DabbaCMuH=(b*b+a*b*(rad)**(-0.5)*(-1))/term
    DabbaCsigmaD = (2 * a * b )/ (term ** 2 * STSamplevarianceH) + (
                ((-b * b - 1) * (rad ** 0.5)) / (term ** 2 * STSamplevarianceH)) + (
                               (STSamplevarianceD * b * rad ** (-0.5)) / term) *(math.log(b * b,math.e)+ 1 - b ** (-2))
    DabbaCsigmaH = (-2 * a * b*b) / (term ** 2 * STSamplevarianceH) + (
                (b*(b * b + 1) * (rad ** 0.5)) / (term ** 2 * STSamplevarianceH)) - (
                               (STSamplevarianceH * b * rad ** (-0.5)) / term) *(math.log(b * b,math.e)+ b**2 - 1)

    zD = (SampleMeanD-C)/STSamplevarianceD
    zH = (C - SampleMeanH) / STSamplevarianceH
    #Create distributions (cdf = cumulative distribution function)
    cdfzD = scipy.stats.norm(0, 1).cdf(zD)
    cdfzH = scipy.stats.norm(0, 1).cdf(zH)
    hatJ =cdfzD+cdfzH-1 # this is the corresponding Youden index
    
    #Calculate Variance
    DabbaJmuD=(cdfzD/STSamplevarianceD)+DabbaCMuD*((cdfzH/STSamplevarianceH)-(cdfzD/STSamplevarianceD))
    DabbaJmuH = (-cdfzH / STSamplevarianceH) + DabbaCMuH * ((cdfzH / STSamplevarianceH) - (cdfzD / STSamplevarianceD))
    DabbaJvarianceD = (-zD/STSamplevarianceD)*cdfzD+DabbaCsigmaD*((cdfzH/STSamplevarianceH)-(cdfzD/STSamplevarianceD))
    DabbaJvarianceH = (-zH / STSamplevarianceH) * cdfzH + DabbaCsigmaH * (
                cdfzH / STSamplevarianceH - cdfzD / STSamplevarianceD)
    VarmuD = SamplevarianceD / nD
    VarmuH = SamplevarianceH / nH
    VarSigmaD = SamplevarianceD / 2*(nD-1)
    VarSigmaH = SamplevarianceH / 2 * (nH - 1)
    VarChat = DabbaCMuD ** 2 * VarmuD + DabbaCsigmaD ** 2 * VarSigmaD + DabbaCMuH ** 2 * VarmuH + VarSigmaD ** 2 * VarSigmaH
    ChatL = C - scipy.stats.norm.ppf(1 - alpha / 2) * math.sqrt(VarChat)
    ChatU = C + scipy.stats.norm.ppf(1 - alpha / 2) * math.sqrt(VarChat)
    VarJhat=DabbaJmuD**2*VarmuD + DabbaJvarianceD**2*VarSigmaD + DabbaJmuH**2*VarmuH + DabbaJvarianceH**2*VarSigmaH #This is the variance
    
    Jhat=scipy.stats.norm(0, 1).cdf((SamplevarianceD-C)/float(STSamplevarianceD))+scipy.stats.norm(0, 1).cdf((C-SamplevarianceH)/float(STSamplevarianceH))-1
    JhatL = Jhat - scipy.stats.norm.ppf(1 - alpha / 2) * math.sqrt(VarJhat)
    JhatU = Jhat + scipy.stats.norm.ppf(1 - alpha / 2) * math.sqrt(VarJhat)
    return [C,ChatL,ChatU, hatJ, VarJhat, JhatL,JhatU];
#****************************************************************************************


def Measures(NumSimulation):

    for i in range(NumSimulation):
        [Rc, Rj, GCIforC, GCIforJ] = GenaralizedCIforJandC(alpha)
        if (GCIforJ[0]>=Rj and Rj<=GCIforJ[1]):
            Deltaarray[i] = 1
        else:
            Deltaarray[i] = 0
        if (Rj >= J0):
            Nitaarray[i] = 1
        else:
            Nitaarray[i] = 0
        lengthCI[i]= GCIforJ[1]-GCIforJ[0]
    coverageProb=format(sum(Deltaarray)/float(NumSimulation),'.4f') # coverage Probability
    power= format(sum(Nitaarray) / float(NumSimulation), '.4f')  # power
    ExpectedLength = format(sum(lengthCI) / float(NumSimulation), '.4f')  # expected length
    return [coverageProb,power,ExpectedLength]


#Import data
infile = open("mammo.txt", 'r')
list1 = [line.rstrip() for line in infile]
HealthyData = []
DiseasedData = []
nH=0
nD=0

#For each line, check if diseased or healthy and append age to appropriate list
for entry in range(len(list1)):
    list1[entry] = list1[entry].split(',')
    try:
        if(list1[entry][5] == '0'):
            HealthyData.append(int(list1[entry][1]))
            nH = nH + 1
        if(list1[entry][5] == '1'):
            DiseasedData.append(int(list1[entry][1]))
            nD = nD + 1
    except ValueError:
        #Got a ?
        print()


#Prepare plot
#Healthy
t = np.array(range(100))
s = np.zeros(100)

for val in HealthyData:
    s[val] += 1
    
plt.plot(t, s)

#Diseased
t = np.array(range(100))
s = np.zeros(100)

for val in DiseasedData:
    s[val] += 1
    
plt.plot(t, s)

plt.xlabel('age')
plt.ylabel('frequency')
plt.title('Healthy and Diseased')
plt.grid(True)
plt.show()



#Get Mean and Variance
[SampleMeanD, SamplevarianceD, SampleMeanH, SamplevarianceH]=GetSampleMeanVariance(HealthyData,DiseasedData,nD,nH)

#Get CI
#[C,ChatL,ChatU, hatJ, VarJhat, JhatL,JhatU] = ClassicalCIforJ(SampleMeanD, SamplevarianceD, SampleMeanH, SamplevarianceH,nD,nH,alpha)
[C,ChatL,ChatU, hatJ, VarJhat, JhatL,JhatU] = ClassicalCIforJ(SampleMeanD, SamplevarianceD, SampleMeanH, SamplevarianceH,nD,nH,alpha)
#[coverageProb,power,ExpectedLength]=Measures(nH+nD)

print ('Mean of Diseased')
print (SampleMeanD)
print ('Variance of Diseased')
print (SamplevarianceD)
print ('Mean of Healthy')
print (SampleMeanH)
print ('Variance of Healthy')
print (SamplevarianceH)


print ('C')
print (C)
print ('ChatL')
print (ChatL)
print ('ChatU')
print (ChatU)
print ('Jouden Index')
print (hatJ)
print ('Jouden Variance')
print (VarJhat)
print ('JhatL')
print (JhatL)
print ('JhatU')
print (JhatU)

print ('')

#print ('This is the coverage probability')
#print (coverageProb)
#print ('This is the power of the test')
#print (power)
#print ('This is the expected length')
#print (ExpectedLength)

import numpy as np
import math
import scipy
import sympy
from scipy.stats import norm

import matplotlib.pyplot as plt

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


# Import data
infile = open("mammo.txt", 'r')
list1 = [line.rstrip() for line in infile]
HealthyData = []
DiseasedData = []
nH = 0
nD = 0

# For each line, check if diseased or healthy and append age to appropriate list
for entry in range(len(list1)):
    list1[entry] = list1[entry].split(',')
    try:
        if (list1[entry][5] == '0'):
            HealthyData.append(int(list1[entry][1]))
            nH = nH + 1
        if (list1[entry][5] == '1'):
            DiseasedData.append(int(list1[entry][1]))
            nD = nD + 1
    except ValueError:
        # Got a ?
        print()

# Prepare plot
# Healthy
t = np.array(range(100))
s = np.zeros(100)

for val in HealthyData:
    s[val] += 1

plt.plot(t, s)

# Diseased
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
print ('Mean of Diseased')
print (SampleMeanD)
print ('Variance of Diseased')
print (SamplevarianceD)
print ('Mean of Healthy')
print (SampleMeanH)
print ('Variance of Healthy')
print (SamplevarianceH)
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from math import sqrt
import pandas as pd


data = np.loadtxt('crash.txt')
training = data[:92:2]
test = data[1::2]
# Start of Question 1 Part 1
x = data[:,1]
y = data[:,0]

plt.scatter(x,y, label = "Data Relationship")
# End of Question 1 Part 1
# Start of Question 1 Part 2

xtraining = training[:,1]
ytraining = training[:,0]
Ntraining = len(xtraining)
x2training = xtraining**2
slopeTraining = (Ntraining*np.sum(xtraining*ytraining) - xtraining.sum()*ytraining.sum()) / (Ntraining*x2training.sum() - xtraining.sum()**2)
interceptTraining = (ytraining.sum() - slopeTraining*xtraining.sum()) / Ntraining

actualValue = test[:,0]
actualX = test[:,1]
predicValue = slopeTraining*actualX+interceptTraining 
SSE = (actualValue - predicValue)**2
RMSE = sqrt(SSE.sum()/Ntraining) #(abs(predicValue - actualValue) / abs(actualValue)).mean()* 100 #

print("RMSE Accuracy: ")
print(RMSE)

xMean = xtraining.mean()
yMean = ytraining.mean()
xSubMean = xtraining-xMean
ySubMean = ytraining-yMean
correlation = (np.sum(xSubMean*ySubMean)) / (sqrt(np.sum(xSubMean**2)*np.sum(ySubMean**2)))
print("Training Correlation: ")
print(correlation)

# End of Question 1 Part 2
# Start of Question 1 Part 3

#x2 = x**2
#N = len(x)
#slope = (N*np.sum(x*y) - x.sum()*y.sum()) / (N*x2.sum() - x.sum()**2)
#intercept = (y.sum() - slope*x.sum()) / N
#plt.plot(x, slope*x+intercept, linestyle='solid', color = 'k', label = "All Data Fit")

Ntest = len(actualX)
x2test = actualX**2
slopeTest = (Ntest*np.sum(actualX*actualValue) - actualX.sum()*actualValue.sum()) / (Ntest*x2test.sum() - actualX.sum()**2)
interceptTest = (actualValue.sum() - slopeTest*actualX.sum()) / Ntest

plt.plot(xtraining, slopeTraining*xtraining + interceptTraining, linestyle='solid', color = 'r', label = "Prediction from Training data")
plt.plot(actualX, slopeTest*actualX + interceptTest, linestyle='solid', color = 'g', label = "Test Data Fit")
plt.legend(loc='upper left')
plt.grid()
plt.show()

# End of Question 1 Part 3
# Start of Question 2

diabetes = pd.read_csv("diabetes.csv", sep=",")
#print(diabetes)
diabetesTest = diabetes[:54:][::]
diabetesTraining = diabetes[54::][::]
testOut = diabetesTest.Outcome
testIn = diabetesTest.loc(:, diabetesTest!='Outcome')
print(testIn)
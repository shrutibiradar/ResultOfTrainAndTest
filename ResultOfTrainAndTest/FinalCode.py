# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 18:04:12 2018

@author: Basavaraj Chalva
"""
#importing required libraries

import pandas as pd

#importing the required data set
trainSet = pd.read_csv('D:\Train and Test\data_train.csv')

#Matrix for dependent and independent variables
X = trainSet.iloc[:,:-1].values
y = trainSet.iloc[:,57].values

#dealing with missing values
from sklearn.preprocessing import Imputer

mValueMean = Imputer(missing_values="NaN",
                 strategy= "mean",
                 axis=0)
mValueMedian = Imputer(missing_values="NaN",
                 strategy= "median",
                 axis=0)
mValueMode = Imputer(missing_values="NaN",
                 strategy= "most_frequent",
                 axis=0)

mValueMedian = mValueMedian.fit(X[:,[1,2,12,16,17,24,25,26,56]])
X[:,[1,2,12,16,17,24,25,26,56]] = mValueMedian.transform(X[:,[1,2,12,16,17,24,25,26,56]])

mValueMode = mValueMode.fit(X[:,3:12])
X[:,3:12] = mValueMode.transform(X[:,3:12])


mValueMode = mValueMode.fit(X[:,[13,14,15,19]])
X[:,[13,14,15,19]] = mValueMode.transform(X[:,[13,14,15,19]])

mValueMean = mValueMean.fit(X[:,[18,20,21,22,23]])
X[:,[18,20,21,22,23]] = mValueMean.transform(X[:,[18,20,21,22,23]])


mValueMode = mValueMode.fit(X[:,27:56])
X[:,27:56] = mValueMode.transform(X[:,27:56])


#fit our model on Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
DTRegressor = DecisionTreeRegressor(random_state = 0)
DTRegressor.fit(X,y)

testSet = pd.read_csv('D:\Train and Test\data_test.csv')

X_test = testSet.iloc[:,:].values

#dealing with missing vlaues
mValueMedian = mValueMedian.fit(X_test[:,[1,2,12,16,17,24,25,26,56]])
X_test[:,[1,2,12,16,17,24,25,26,56]] = mValueMedian.transform(X_test[:,[1,2,12,16,17,24,25,26,56]])

mValueMode = mValueMode.fit(X_test[:,3:12])
X_test[:,3:12] = mValueMode.transform(X_test[:,3:12])

mValueMode = mValueMode.fit(X_test[:,[13,14,15,19]])
X_test[:,[13,14,15,19]] = mValueMode.transform(X_test[:,[13,14,15,19]])

mValueMean = mValueMean.fit(X_test[:,[18,20,21,22,23]])
X_test[:,[18,20,21,22,23]] = mValueMean.transform(X_test[:,[18,20,21,22,23]])

mValueMode = mValueMode.fit(X_test[:,27:56])
X_test[:,27:56] = mValueMode.transform(X_test[:,27:56])

#Now see hoe accurately decision tree regressor predict
prediction = DTRegressor.predict(X_test)

#exporting data into a CSV file

df = pd.DataFrame(prediction,X_test[:,0])
df.to_csv("D:\\Train and Test\\result.csv")


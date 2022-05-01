# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 01:12:01 2022

@author: ashakya
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_csv("heart_2020_cleaned.csv")

def normalize(numbers):
    minvalue = min(numbers)
    maxvalue = max(numbers)
    newValues = (numbers-minvalue)/(maxvalue-minvalue)
    return newValues

def binaryYesNo(value):
    if value == 'Yes': return 1
    else: return 0

def ageClassify(value):
    if value == "18-24": return 0
    elif value == "25-29": return 1
    elif value == "30-34": return 2
    elif value == "35-39": return 3
    elif value == "40-44": return 4
    elif value == "45-49": return 5
    elif value == "50-54": return 6
    elif value == "55-59": return 7 
    elif value == "60-64": return 8
    elif value == "65-69": return 9
    elif value == "70-74": return 10
    elif value == "75-79": return 11   
    elif value == "80 or older": return 12

#-----------------Normalizing and Categorizing our data----------------
# Normalize BMI
df['nBMI'] = normalize(df['BMI'])
# Smoking categorical to numerical
df['nSmoke'] = df['Smoking'].apply(lambda x:0 if x == 'No' else 1)
# AlcoholDrinking categorical to numerical
df['nAlcohol'] = df['AlcoholDrinking'].apply(lambda x:0 if x == 'No' else 1)
# Normalize Physical Health
df['nPhysicalH'] = normalize(df['PhysicalHealth'])
# Normalize Mental Health
df['nMentalH'] = normalize(df['MentalHealth'])
# Sex categorical to numerical
df['S'] = df['Sex'].apply(lambda x:0 if x == 'Male' else 1)
# Age category to numerical range 0-12
df['nA'] = df.apply(lambda row: ageClassify(row['AgeCategory']), axis=1)
# PhysicalActivity to numerical
df['nPA'] = df['PhysicalActivity'].apply(lambda x:0 if x == 'No' else 1)

#-------------------------KNearestNeighbour------------------------------
#-------------Setting up input and output data for training--------------
# Hyperparameter for KNN
n = 10
# Classifying output and input for the KNN
Y = df['HeartDisease'].apply(lambda x:0 if x == 'No' else 1)
X = df[['nBMI', 'nSmoke', 'nAlcohol', 'nPhysicalH', 'nMentalH', 'S', 'nA', 'nPA']]
# Spliting the data to training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) 
X_test = np.array(X_test, dtype=np.int32)
Y_test = np.array(Y_test)

# Using the sklearn KNN classifier

neigh = KNeighborsClassifier(n_neighbors = n)
neigh.fit(X_train.values,Y_train)
KNeighborsClassifier(...)

def testing(X_test, Y_test):
    y_hat = []
    count = 0
    count1 = 0
    testSize = np.size(X_test,0)
    for i in range(testSize):
        pred_val = neigh.predict([X_test[i]])
        y_hat.append(pred_val)
        if Y_test[i] == pred_val:
            count += 1
            if pred_val == 1:
                count1 += 1
    print("The accuracy of the system using KNN is",count*100/testSize)
    return y_hat

def predictHA(X):
    prob = neigh.predict_proba(X)
    print("The probabily of heartattack is",round(prob[0,1]*100,3))

def confusionMatrixPrint(y_hat):
    tn, fp, fn, tp = confusion_matrix(Y_test, y_hat).ravel()
    print("Sensitivity/True Postive Rate is", tp*100/(tp+fn),"%")
    print("Selectivity/True Negative Rate is", tn*100/(tn+fp),"%")
    print("Precision/Postive Prediction Value is", tp*100/(tp+fp),"%")
    return tn, fp, fn, tp
y_hat = testing(X_test, Y_test)
y_hat = np.array(y_hat)
tn, fp, fn, tp = confusionMatrixPrint(y_hat)

#predictHA([[1,1,1,1,1,1,1,1]])

#------------------------Neural Network----------------------------------
#def NNClassifi(X_train, Y_train, X_test):
 #   clasf = MLPClassifier(solver = 'lbfgs', alpha = 0.1,
  #                  hidden_layer_sizes=(10,10),random_state=1)
   # clasf.fit(X_train, Y_train)
   # y_hatNN = clasf.predict(X_test)
   # return y_hatNN
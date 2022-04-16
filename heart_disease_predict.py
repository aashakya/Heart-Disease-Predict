# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 01:12:01 2022

@author: ashakya
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

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

def KNNfunc(testcase):
    neigh = KNeighborsClassifier(n_neighbors = n)
    neigh.fit(X.values,Y)
    KNeighborsClassifier(...)
    pred_val = neigh.predict([testcase])
    if (pred_val == 0):
        print('Safe')
    else:
        print('Might get an heartattack')
    prob = neigh.predict_proba([testcase])
    print("The probabily of heartattack is",round(prob[0,1],2))

# Normalize BMI
df['nBMI'] = normalize(df['BMI'])
# Smoking categorical to numerical
df['nSmoke'] = df['Smoking'].apply(lambda x:0 if x == 'No' else 1)
# AlcoholDrinking categorical to numerical
df['nAlcohol'] = df['AlcoholDrinking'].apply(lambda x:0 if x == 'No' else 1)
# Stroke categorical to numerical
#df['nStroke'] = binaryYesNo(df['Stroke'])
# Normalize Physical Health
df['nPhysicalH'] = normalize(df['PhysicalHealth'])
# Normalize Mental Health
df['nMentalH'] = normalize(df['MentalHealth'])
# DiffWalking categorical to numerical
#df['nDiffWalk'] = binaryYesNo(df['DiffWalking'])
# Sex categorical to numerical
df['S'] = df['Sex'].apply(lambda x:0 if x == 'Male' else 1)
# Age category to numerical range 0-12
df['nA'] = df.apply(lambda row: ageClassify(row['AgeCategory']), axis=1)
# Race categorical to numerical
# Diabetic categorical to numerical
# df['nDiabetic'] = binaryYesNo(df['Diabetic'])
# PhysicalActivity to numerical
df['nPA'] = df['PhysicalActivity'].apply(lambda x:0 if x == 'No' else 1)
# GenHealth categorical to numerical
# Normalize SleepTime
# Asthma categorical to numerical
# df['nAsthma'] = binaryYesNo(df['Asthma'])
# KidneyDisease categorical to numerical
# df['nKidney'] = binaryYesNo(df['KidneyDisease'])
# # SkinCancer categorical to numerical
# df['nSkinCancer'] = binaryYesNo(df['SkinCancer'])

n = 10
Y = df['HeartDisease'].apply(lambda x:0 if x == 'No' else 1)
X = df[['nBMI', 'nSmoke', 'nAlcohol', 'nPhysicalH', 'nMentalH', 'S', 'nA', 'nPA']]

testcase = [0.185802, 1, 0, 0.2333, 0, 0, 8, 0]
KNNfunc(testcase)
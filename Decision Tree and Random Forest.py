# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:01:14 2019

@author: Cosmic Dust
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir('C:/Users/CosmicDust/Documents/Python/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/15-Decision-Trees-and-Random-Forests')

df = pd.read_csv('kyphosis.csv')
df.head()
df.info()

sns.pairplot(df, hue='Kyphosis')

from sklearn.model_selection import train_test_split

X = df.drop('Kyphosis', axis = 1)
y = df['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Import decision tree module

from sklearn.tree import DecisionTreeClassifier

# Train data

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

# Import random forest module

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)

print(classification_report(y_test,rfc_pred))
print(confusion_matrix(y_test,rfc_pred))

df['Kyphosis'].value_counts()


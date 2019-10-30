# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 19:54:49 2019

@author: Cosmic Dust
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

os.getcwd()

df = pd.read_csv('KNN_Project_Data')
df.head()

# Exploratory Data Analysis

sns.pairplot(df, hue='TARGET CLASS')


# Standardize the variables

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis = 1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))

df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1])
df_feat.head()


# Creating training and testing datsets

from sklearn.model_selection import train_test_split

X = df_feat
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)


# Using KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1) 
knn.fit(X_train,y_train)

# Prediction and evaluation

pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# Elbow method to pick a good K value

error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,12))

plt.plot(range(1,40),error_rate, linestyle = 'dashed', color='blue', marker='o',markerfacecolor='red',markersize=10)

# Retrain with better K value

knn = KNeighborsClassifier(n_neighbors=30) 
knn.fit(X_train,y_train)

# Prediction and evaluation

pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

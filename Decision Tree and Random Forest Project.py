# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:35:00 2019

@author: Cosmic Dust
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir('C:/Users/CosmicDust/Documents/Python/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/15-Decision-Trees-and-Random-Forests')

loans = pd.read_csv('loan_data.csv')

loans.info()
loans.head()
loans.columns
loans.describe()

# Exploratory data analysis

loans[loans['credit.policy']==1]['fico'].hist(bins=35,color='blue',label='Credit Policy = 1',alpha=0.6)
loans[loans['credit.policy']==0]['fico'].hist(bins=35,color='red',label='Credit Policy = 0',alpha=0.6)
plt.legend()

loans[loans['not.fully.paid']==1]['fico'].hist(bins=30,color='blue',label='No Fully Paid = 1',alpha=0.6)
loans[loans['not.fully.paid']==0]['fico'].hist(bins=30,color='red',label='Not Fully Paid = 0',alpha=0.6)
plt.legend()

loans.columns

plt.figure(figsize=(15,15))
sns.countplot(x='purpose',data=loans,hue='not.fully.paid')

sns.jointplot(x='fico',y='int.rate',data=loans)

sns.lmplot(x='fico',y='int.rate',data=loans,hue='credit.policy',col='not.fully.paid',palette='Set1')


# Setting up data

loans.info()

cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
final_data.info()

# Train Test Split

from sklearn.model_selection import train_test_split

X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# Training a Decision Tree Model

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

# Prediction and evaluation

dtree_pred = dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test, dtree_pred))
print(classification_report(y_test,dtree_pred))


# Training a Random Forest Model

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)

# Prediction and evaluation

rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))

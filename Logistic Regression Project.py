# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 16:36:46 2019

@author: Cosmic Dust
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ad_data = pd.read_csv('advertising.csv')
ad_data.head()

ad_data.info()
ad_data.describe()

# Exploratory data analysis

sns.distplot(ad_data['Age'], kde=False, bins=30)
sns.jointplot(x='Age',y='Area Income',data=ad_data)
sns.jointplot(x='Age',y='Daily Time Spent on Site', data=ad_data, kind='kde')
sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ad_data)

sns.pairplot(ad_data, hue='Clicked on Ad')

# Cleaning data

sns.heatmap(ad_data.isnull(), yticklabels=False, cbar=False)


# Creating training and testing data

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']] 
y = ad_data['Clicked on Ad']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=69)


# Train and predict data using Logistic Regression

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)


# Evaluate the model

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

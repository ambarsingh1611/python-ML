
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 11:27:09 2019

@author: Cosmic Dust
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

os.getcwd()
os.chdir('C:/Users/CosmicDust/Documents/Python/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression')

customers = pd.read_csv('Ecommerce Customers')

customers.head()
customers.describe()
customers.info()

customers.columns

# Exploratory Data Analysis

sns.set_style("whitegrid")
sns.jointplot(data = customers, x = 'Time on Website', y = 'Yearly Amount Spent')

sns.jointplot(data = customers, x = 'Time on App', y = 'Yearly Amount Spent')

sns.jointplot(x = 'Time on App', y = 'Length of Membership', kind = 'hex', data = customers)

sns.pairplot(customers)

sns.lmplot(x = 'Length of Membership', y = 'Yearly Amount Spent', data = customers)


# Create training and test data

customers.columns

y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 101)


# Training the model

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

lm.coef_


# Predicting test data

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.xlabel('Y Test (True Values)')
plt.ylabel('Predicted Values')

# Evaluating the Model

from sklearn import metrics

# Mean Absolute Error
print('MAE', metrics.mean_absolute_error(y_test,predictions)) 
# Mean Squared Error
print('MSE', metrics.mean_squared_error(y_test,predictions)) 
# Root Mean Absolute Error 
print('RMSE', np.sqrt(metrics.mean_squared_error(y_test,predictions)))


metrics.explained_variance_score(y_test,predictions)

# Residuals

sns.distplot((y_test - predictions), bins = 50)  



cdf = pd.DataFrame(lm.coef_, X.columns,columns=['Coeff'])
cdf

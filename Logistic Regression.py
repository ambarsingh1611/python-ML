# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 17:19:52 2019

@author: Cosmic Dust
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

os.getcwd()
os.chdir('C:/Users/CosmicDust/Documents/Python/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression')

train = pd.read_csv('titanic_train.csv')
train.head()

train.isnull()

# Visual exploratory data analysis

sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'summer')

sns.set_style('whitegrid')

sns.countplot(x='Survived', data=train)
sns.countplot(x='Survived', data=train, hue='Sex', palette='RdBu_r')
sns.countplot(x='Sex', data=train, hue='Survived', palette='RdBu_r')
sns.countplot(x='Survived', data=train, hue='Pclass')

sns.distplot(train['Age'].dropna(), kde=False, bins=30)
train['Age'].plot.hist(bins=30)

train.info()

sns.countplot(x='SibSp', data=train)

sns.distplot(train['Fare'], kde=False, bins=40)

# Cleaning data (acceptable form for ML algorithm)

plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass', y='Age', data=train)

#------Inputing the Mean age values by Passenger Class------#
def impute_age_train(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

train['Age'] = train[['Age', 'Pclass']].apply(impute_age_train, axis=1)
#-----------------------------------------------------------#

sns.heatmap(train.isnull(), yticklabels = False, cbar=False, cmap='summer')

#------Remove Null Values------#
train.drop('Cabin', axis=1,inplace=True)
        
train.dropna(inplace=True)
#------------------------------#

#--------Creating dummy variables-----------#
sex = pd.get_dummies(train['Sex'], drop_first=True)
sex.head()
embark = pd.get_dummies(train['Embarked'],drop_first=True)
embark.head()

train = pd.concat([train,sex,embark],axis=1)
train.head()

train.drop(['Embarked','Sex','Name','Ticket'],axis=1,inplace=True)
train.head()

train.drop('PassengerId', axis=1, inplace=True)
train.head()

pclass = pd.get_dummies(train['Pclass'], drop_first=True)
pclass

train = pd.concat([train,pclass], axis=1)
train.drop('Pclass', axis = 1, inplace=True)
#-------------------------------------------#




# Cleaning the test data (Same process, change file name)

# USE WHEN NOT PERFORMING TRAIN TEST SPLIT, OTHERWISE SKIP #


test = pd.read_csv('titanic_test.csv')

sns.boxplot(x='Pclass', y='Age',data=test)

#------Inputing the Mean age values by Passenger Class------#
def impute_age_test(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 42
        elif Pclass == 2:
            return 27
        else:
            return 24
    else:
        return Age

test['Age'] = test[['Age', 'Pclass']].apply(impute_age_test, axis=1)
#-----------------------------------------------------------#

sns.heatmap(test.isnull(), yticklabels = False, cbar=False, cmap='summer')

#------Remove Null Values------#
test.drop('Cabin', axis=1,inplace=True)
        
test.dropna(inplace=True)
#------------------------------#

#--------Creating dummy variables-----------#
sex = pd.get_dummies(test['Sex'], drop_first=True)
sex.head()
embark = pd.get_dummies(test['Embarked'],drop_first=True)
embark.head()

test = pd.concat([test,sex,embark],axis=1)
test.head()

test.drop(['Embarked','Sex','Name','Ticket'],axis=1,inplace=True)
test.head()

test.drop('PassengerId', axis=1, inplace=True)
test.head()

pclass = pd.get_dummies(test['Pclass'], drop_first=True)
pclass

test = pd.concat([test,pclass], axis=1)
test.drop('Pclass', axis = 1, inplace=True)
#-------------------------------------------#





# Create training and testing data

train.head()

X = train.drop('Survived',axis=1)
y = train['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)



# Training the model

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# Making Predictions

predictions = logmodel.predict(X_test)


# Checking metrics
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


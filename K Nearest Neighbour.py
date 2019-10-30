# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 16:36:17 2019

@author: Cosmic Dust
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

os.getcwd()
os.chdir('C:/Users/CosmicDust/Documents/Python\Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/14-K-Nearest-Neighbors')

df = pd.read_csv('Classified Data', index_col = 0)
df.head()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))

scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
scaled_features

df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1])
df_feat.head()

from sklearn.model_selection import train_test_split

X = df_feat
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=69)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)
pred


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

# Choosing a better K value

error_rate = []

#----iterating multiple k values to identify least error
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
    
#----plotting the errors against k values to identify least error
plt.figure(figsize=(10,6))
sns.scatterplot(x=range(1,40),y=error_rate)  


plt.plot(range(1,40),error_rate, color = 'blue', linestyle='dashed', marker = 'o', markerfacecolor = 'red', markersize = 10)
plt.title('Error rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
    
# Checking the model's metrics with better K value

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))





    
    
    
    
    
    
    
    
    
    
    
    
    
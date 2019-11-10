import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.chdir('C:/Users/CosmicDust/Documents/Statistics/Assignment 5')

# Reading the dataset
df = pd.read_excel('DEN 2012 - Jun 17-1.xlsx')
df.head()
df.describe()
df.info()

#---------------------------- DATA PREPROCESSING ------------------------------

import datetime

# Convert datetime format to String
df['Month and Year'] = df['Month and Year'].dt.strftime('%Y-%m-%d')

# Extract the year out of Month of Year column. (The format of actual data was incorrect)
df['Year'] = df['Month and Year'].apply(lambda date: date.split('-')[2])
#df['Year'] = pd.to_numeric(df['Year'])

# Drop Month and Year column because it is redundant
df = df.drop(['Month and Year'],axis=1)

# Create a new column 'Month and Year' by merging Month and Year columns
df['Month and Year'] = df[['Month', 'Year']].apply(lambda x: ''.join(x), axis=1)
df.head()

#------------------------- EXPLORATORY DATA ANALYSIS --------------------------
sns.set_style('darkgrid')

# Checking if Cannbis column plays an important factor?
plt.figure(figsize=(12,6)),sns.barplot(x='Month',y='Ground',data=df,hue='Cannabis?')
plt.figure(figsize=(12,6)),sns.barplot(x='Month',y='Parking',data=df,hue='Cannabis?')
plt.figure(figsize=(12,6)),sns.barplot(x='Month',y='Rental Car',data=df,hue='Cannabis?')
plt.figure(figsize=(12,6)),sns.barplot(x='Month',y='Concession',data=df,hue='Cannabis?')


# Scatterplots
plt.figure(figsize=(12,6)),sns.scatterplot(x='Enplaned',y='Concession',data=df)
plt.figure(figsize=(12,6)),sns.scatterplot(x='Enplaned',y='Parking',data=df)
plt.figure(figsize=(12,6)),sns.scatterplot(x='Transfer',y='Rental Car',data=df)
plt.figure(figsize=(12,6)),sns.scatterplot(x='Enplaned',y='Ground',data=df)

# Lineplots
plt.figure(figsize=(35,6)),sns.lineplot(x='Month and Year',y='Ground',data=df,sort=False)
plt.figure(figsize=(35,6)),sns.lineplot(x='Month and Year',y='Parking',data=df,sort=False)
plt.figure(figsize=(35,6)),sns.lineplot(x='Month and Year',y='Rental Car',data=df,sort=False)
plt.figure(figsize=(35,6)),sns.lineplot(x='Month and Year',y='Concession',data=df,sort=False)

# Histograms
sns.distplot(df['Enplaned'],bins=25,kde=False)
sns.distplot(df['Deplaned'],bins=25,kde=False)
sns.distplot(df['Transfer'],bins=25,kde=False)
sns.distplot(df['Originating'],bins=25,kde=False)
sns.distplot(df['Destination'],bins=25,kde=False)

# Linear Model graphs
sns.lmplot(x='Destination',y='Ground',data=df,col='Year')
sns.lmplot(x='Month and Year',y='Rental Car',data=df)
sns.lmplot(x='Month and Year',y='Rental Car',data=df)
sns.lmplot(x='Month and Year',y='Ground',data=df)

# Corelation Matrix
df.corr()
cmap = sns.diverging_palette(10, 10, as_cmap=True)
plt.figure(figsize=(14,10)),sns.heatmap(df.corr(),annot=True,yticklabels=True,cmap=cmap,center=0)

# Correlation matrix of only X variables
Xdataset = df.drop(['Concession', 'Parking','Ground','Rental Car'],axis=1)
plt.figure(figsize=(14,10)),sns.heatmap(Xdataset.corr(),annot=True,yticklabels=True,cmap=cmap,center=0)

plt.figure(figsize=(12,20)),sns.lmplot(x='Concession',y='Transfer',data=df)
plt.figure(figsize=(12,20)),sns.lmplot(x='Transfer',y='Ground',data=df,hue='Year')

#'Enplaned', 'Deplaned', 'Transfer', 'Originating', 'Destination'


# Barplot of Parking rate
plt.figure(figsize=(12,6)),sns.barplot(x='Month',y='Parking',data=df) 

# There is a NA value on May 15. Let explore the data according to year to see a better pattern
plt.figure(figsize=(12,6)),sns.barplot(x='Month',y='Parking',data=df,hue='Year')



#-------------------------------- DATA CLEANING -------------------------------

# Set 'Month and Year' column as the index of dataframe
df = df.set_index('Month and Year')


# Function to replace negative values with positives
def fix_negatives(values):
    if values < 0:
        return abs(values)
    else:
        return values

# Applying above function to 'Concession' column to convert negative value to positive
df['Concession'] = df['Concession'].apply(fix_negatives)


# Substitute the missing value in 'Parking' using moving average of data in May each year
df['Parking']['May15'] = round(((df['Parking']['May12']*1) + 
                                (df['Parking']['May13']*2) + 
                                (df['Parking']['May14']*3) + 
                                (df['Parking']['May16']*4) + 
                                (df['Parking']['May17']*5)) / (1+2+3+4+5))

# Reset Index
#df = df.reset_index()
#df.head()

# Get dummies for month data
months = pd.get_dummies(df['Month'], drop_first=True)
months

df = pd.concat([df,months],axis=1)
df.head()


# Drop Redundant columns (Cannabis column is also not important)
df = df.drop(['UMCSENTLag2','UMCSENT','UMCSENTLag1','UMCSENTLag3', 'Cannabis?', 'Origin + Destin','Year', 'Month'], axis=1)
df.head()

#------------------ APPLYING MACHINE LEARNING MODELS --------------------------

# Setting X and y variables
X = df.drop(['Ground','Parking','Rental Car','Concession'],axis = 1)
y_ground = df['Ground']
y_parking = df['Parking'] 
y_rentalCar = df['Rental Car']
y_concession = df['Concession']

# Spliting train and test data
X_train = X[:'Feb17']
X_test = X['Mar17':]


y_ground_train = y_ground[:'Feb17']
y_ground_test = y_ground['Mar17':]

y_parking_train = y_parking[:'Feb17']
y_parking_test = y_parking['Mar17':]

y_rentalCar_train = y_rentalCar[:'Feb17']
y_rentalCar_test = y_rentalCar['Mar17':]

y_concession_train = y_concession[:'Feb17']
y_concession_test = y_concession['Mar17':]


# Linear Regression Model
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

# for Grounds
lm.fit(X_train, y_ground_train)
predict_ground = lm.predict(X_test)


# for Parking
lm.fit(X_train, y_parking_train)
predict_parking = lm.predict(X_test)


# for Rental Car
lm.fit(X_train, y_rentalCar_train)
predict_rentalCar = lm.predict(X_test)


# for Concession
lm.fit(X_train, y_concession_train)
predict_concession = lm.predict(X_test)


# Evaluating Result

sns.scatterplot(y_ground_test,predict_ground)
sns.scatterplot(y_parking_test,predict_parking)
sns.scatterplot(y_rentalCar_test,predict_rentalCar)
sns.scatterplot(y_concession_test,predict_concession)


#------------------------------------------------------------------------------


# Logistic Regression Model
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

# for Grounds
logmodel.fit(X_train, y_ground_train)
predict_ground = logmodel.predict(X_test)


# for Parking
logmodel.fit(X_train, y_parking_train)
predict_parking = logmodel.predict(X_test)


# for Rental Car
logmodel.fit(X_train, y_rentalCar_train)
predict_rentalCar = logmodel.predict(X_test)


# for Concession
logmodel.fit(X_train, y_concession_train)
predict_concession = logmodel.predict(X_test)


# Evaluating Result

from sklearn.metrics import r2_score
r2_score(y_ground_test,predict_ground)
r2_score(y_parking_test,predict_parking)
r2_score(y_rentalCar_test,predict_rentalCar)
r2_score(y_concession_test,predict_concession)







df.columns




















# Rearranging Columns to make Year the first column
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
df.head()

# Function to change months into numeric format
def convert_month(cols):
    Month = cols[0]
    
    if Month == 'Jan':
        return 1
    elif Month == 'Feb':
        return 2
    elif Month == 'Mar':
        return 3
    elif Month == 'Apr':
        return 4
    elif Month == 'May':
        return 5
    elif Month == 'Jun':
        return 6
    elif Month == 'Jul':
        return 7
    elif Month == 'Aug':
        return 8
    elif Month == 'Sep':
        return 9
    elif Month == 'Oct':
        return 10
    elif Month == 'Nov':
        return 11
    else:
        return 12

# Apply the function
df['Month'] = df[['Month']].apply(convert_month,axis=1)
df.head()


        

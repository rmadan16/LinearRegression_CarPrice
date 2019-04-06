#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import copy
#%matplotlib inline


#Read data into dataframe 
X=pd.read_csv('/Users/rohitmadan/Desktop/CarPrice/CarPrice_Assignment.csv')
Y=X.price #Target Variable 
X=X.drop(['price'],axis=1) #Input data

#Checking for null values or Datapreprocessing 
print('Null values in X',X.isnull().sum())
print('Null values in Y',Y.isnull().sum())
#No null values

#Checking if scale and encoding is required
#print(X.head())
#print(Y.head())


#Understanding the dataset 
print('Shape is',X.shape)
print('Head is',X.head())
print('Describing dataframe ',X.describe(include='all'))


#Plot all relationship between dataset X using Pairplot
plt.rcParams["figure.figsize"] = [16,9]
sns.set(style="darkgrid")
sns.pairplot(data=X)

#X.hist('horsepower')
#X.boxplot('horsepower')
#X.groupby('horsepower').hist()

scatter_matrix(X, alpha=0.5, figsize=(30,30), diagonal='kde',grid='true')
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(Y, bins=30)
plt.show()


#Plot Y in timeseries data
Y.plot()

#displot - Heat map and correlation matrix
correlation_matrix = X.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)



#Encoding is required
 
#print(X['doornumber'].value_counts())
cleanup_nums = {"doornumber":     {"four": 4, "two": 2},
                "cylindernumber": {"four": 4, "six": 6, "five": 5, "eight": 8,
                                  "two": 2, "twelve": 12, "three":3 }}
X.replace(cleanup_nums, inplace=True)

#Copying all categorical data into another dataframe using copy
Cat=X.select_dtypes(include=['object']).copy(deep='False')

#Transposing or resizing Cat df
Cat=Cat.iloc[:, :].apply(pd.Series)
Name=Cat.CarName.copy()

#Splitting all values of CarName
Temp=[]
Temp=Name.str.split(pat=" ",expand=True)
Temp=Temp[0]
X.CarName=Temp
Cat.CarName=Temp

#Replacing bad spellings with right spellings
cleanup_nums = {"CarName":     { "maxda": "mazda" , "porcshce": "porsche" , "Nissan":"nissan" , "vokswagen":"volkswagen", "toyouta" : "toyota","vw" : "volkswagen"} }
X.replace(cleanup_nums, inplace=True)

#OneHotEncoding using dummy method
L=X.copy(deep='False')
L=pd.get_dummies(L, columns=Cat.columns)

#Scaling is required

Xs = scale(L)


#Splitting data into test and train - 30% Test
X_train, X_test, Y_train, Y_test = train_test_split(Xs, Y, test_size=0.3, random_state=42)

#Finding correlation coeff
Coef=LinearRegression()

Coef.fit(X_train, Y_train)
Y_pred = Coef.predict(X_test)

# The coefficients
print('Coefficients: \n', Coef.coef_)
# The mean squared error
print("Mean squared error: %.2f"% mean_squared_error(Y_test, Y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, Y_pred))










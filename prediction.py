# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 17:53:41 2019

@author: Subah
"""

import pandas as pd

ds = pd.read_csv('train.csv')

ds.dtypes

import seaborn as sns
sns.countplot(ds['Store'],label="Count")
sns.countplot(ds['Dept'],label="Count")
sns.countplot(ds['IsHoliday'],label="Count")

ds['Date'] = pd.to_datetime(ds['Date'])
ds.head()

df = pd.get_dummies(ds, columns=['Store', 'Dept'])
df.head()

df['Date_dayofweek'] = df['Date'].dt.dayofweek
df['Date_month'] = df['Date'].dt.month
df['Date_year'] = df['Date'].dt.year
df['Date_day'] = df['Date'].dt.day

for days_to_lag in [1, 2, 3, 5, 7, 14, 30]:
    df['Weekly_sales_lag_{}'.format(days_to_lag)] = df.Weekly_Sales.shift(days_to_lag)

df.head()

df = df.fillna(0)

df.IsHoliday = df.IsHoliday.astype(int)

x = df[df.columns.difference(['Date', 'Weekly_Sales'])]  
y = df.Weekly_Sales

x.head()

y[:4]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3)

x_train.shape

x_test.shape

from sklearn.linear_model import LinearRegression

clf = LinearRegression()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print("Accuracy on training set using Linear Regreession: {:.4f}".format(clf.score(x_train, y_train)))
print("Accuracy on test set using Linear Regreession: {:.4f}".format(clf.score(x_test, y_test)))

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x_train, y_train)
y_pred1 = regressor.predict(x_test)

print("Accuracy on training set using Decision Tree Regreession: {:.4f}".format(regressor.score(x_train, y_train)))
print("Accuracy on test set using Decision Tree Regreession: {:.4f}".format(regressor.score(x_test, y_test)))

regressor = DecisionTreeRegressor(max_depth = 3, random_state = 0)
regressor.fit(x_train, y_train)
y_pred2 = regressor.predict(x_test)

print("Accuracy on training set using Decision Tree Regreession(max_depth = 3): {:.4f}".format(clf.score(x_train, y_train)))
print("Accuracy on test set using Decision Tree Regreession(max_depth = 3): {:.4f}".format(clf.score(x_test, y_test)))

from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor()
clf.fit(x_train, y_train)
y_pred3 = clf.predict(x_test)

print("Accuracy on training set using Random Forest : {:.4f}".format(clf.score(x_train, y_train)))
print("Accuracy on test set using Random Forest: {:.4f}".format(clf.score(x_test, y_test)))

from sklearn.metrics import mean_squared_error, mean_absolute_error

#for Linear Regression
mean_absolute_error(y_test, y_pred)
print("MSE of Linear Regreession: {:.4f}".format(mean_squared_error(y_test, y_pred)))

#for Decision Tree regression
mean_absolute_error(y_test, y_pred1)
print("MSE of Decision Tree Regreession: {:.4f}".format(mean_squared_error(y_test, y_pred1)))

#for Decision Tree Regression(max_depth = 3)
mean_absolute_error(y_test, y_pred2)
print("MSE of Decision Tree Regression(max_depth = 3): {:.4f}".format(mean_squared_error(y_test, y_pred2)))

#for Random Forest
mean_absolute_error(y_test, y_pred3)
print("MSE of Random Forest: {:.4f}".format(mean_squared_error(y_test, y_pred3)))


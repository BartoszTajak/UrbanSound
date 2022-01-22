# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 09:19:42 2022

@author: Marcin
"""

import pandas as pd

df = pd.read_csv(r"C:\Users\Marcin\Desktop\sound\sound_features_all.csv", sep=';')
df = df.drop(columns = ['Unnamed: 0'])

import seaborn as sns
df_corr = df.drop(columns = ['class'])
corr = df_corr.corr()

sns.heatmap(corr, annot=True)


# Uczenie maszynowe
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X = df.drop(columns = ['class'])
Y = df['class']

X_train, X_test, y_train, ytest = train_test_split(X, Y, test_size=0.2, random_state = 123)

min_max_scaler = MinMaxScaler()

x_train_norm = min_max_scaler.fit_transform(X_train)
x_test_norm = min_max_scaler.transform(X_test)


# Drzewko
from sklearn import tree

tree_model = tree.DecisionTreeClassifier()
tree_model.fit(x_train_norm, y_train)

y_tree_pred = tree_model.predict(x_test_norm)


from sklearn.metrics import confusion_matrix, classification_report

con_tree = confusion_matrix(ytest, y_tree_pred)

print(classification_report(ytest, y_tree_pred))


# XGBoosta

import xgboost

xgboost_c = xgboost.XGBClassifier()
xgboost_c.fit(x_train_norm, y_train)

y_xgb = xgboost_c.predict(x_test_norm)

con_xgb = confusion_matrix(y_xgb, y_tree_pred)

print(classification_report(y_xgb, y_tree_pred))


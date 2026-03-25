#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:19:55 2026

@author: angella
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv('/Users/angella/Desktop/ML Course/PROJECT/data 2.csv', sep=";")
df.info()
df.shape
df.head

# binary target
df["dropout_status"] = np.where(df["Target"] == "Dropout", 1, 0)

df["Target"].value_counts() #Graduate: 2209, Dropout: 1421, Enrolled: 794
df["dropout_status"].value_counts(normalize=True) #dropout_status: 0 -> 0.678797; 1 -> 0.321203
df["dropout_status"].value_counts() #dropout_status: 0 -> 3003, 1 -> 1421

df = df.drop(columns=["Target"])
df.shape
df.isnull().sum().sum()

#cleaning column name
df.columns = (
    df.columns
    .str.strip()
    .str.replace("Nacionality", "Nationality", regex=False)
)

df.columns.tolist()

#true numeric vs coded categorical variables seperated 
numeric_cols = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",

    # Academic performance
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",

    #financial
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]

binary_cols = [
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International"
]

categorical_cols = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nationality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation"
]

'''Little EDA'''
    
for col in numeric_cols:
    plt.figure(figsize=(5,3))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

for col in numeric_cols:
    plt.figure(figsize=(5,3))
    sns.boxplot(x="dropout_status", y=col, data=df)
    plt.title(f"{col} vs Dropout")
    plt.show()
    
#correlation analysis     

#feature to target
corr_target = df.corr(numeric_only=True)["dropout_status"].sort_values()

plt.figure(figsize=(8, 10))
corr_target.drop("dropout_status").plot(kind="barh")
plt.title("Feature Correlation with Dropout Status")
plt.xlabel("Correlation")
plt.show()    

#feature to feature
corr = df[numeric_cols + ["dropout_status"]].corr()

plt.figure(figsize=(12,8))
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    annot=True,        #ADDS NUMBERS
    fmt=".2f",         #2 decimal places
    linewidths=0.5
)
plt.title("Correlation Matrix with Values")
plt.show()

''' Model data prep'''

model_df = df.copy()
model_df = model_df.drop(columns=[
    "Unemployment rate", 
    "Inflation rate", 
    "GDP",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)"])
model_df.info()

model_df = pd.get_dummies(model_df,columns=categorical_cols,drop_first=True)

x = model_df.drop(columns="dropout_status")
y = model_df["dropout_status"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=1)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

''' Logistic Regression'''

lr = LogisticRegression() #initialize
lr.fit(x_train_scaled, y_train) #train

y_pred = lr.predict(x_test_scaled)

print(classification_report(y_test, y_pred)) #f1 0.74, recall 0.70
print(confusion_matrix(y_test, y_pred))

''' Random forest '''

rf = RandomForestClassifier(n_estimators=100,random_state=1) #initialize
rf.fit(x_train_scaled, y_train) #train

y_pred_rf = rf.predict(x_test_scaled)

print(classification_report(y_test, y_pred_rf)) #f1 0.73, recall 0.64
print(confusion_matrix(y_test, y_pred_rf))

#3 Logistic Regression is performing better than RF


##just chceking to see if the model performs better after removing a few more variables from the 1st sem
#results - not so good

model_df = df.copy()
model_df = model_df.drop(columns=[
    "Unemployment rate", 
    "Inflation rate", 
    "GDP",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)"])
model_df.info()

model_df = pd.get_dummies(model_df,columns=categorical_cols,drop_first=True)

x = model_df.drop(columns="dropout_status")
y = model_df["dropout_status"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=1)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

lr = LogisticRegression() #initialize
lr.fit(x_train_scaled, y_train) #train

y_pred = lr.predict(x_test_scaled)

print(classification_report(y_test, y_pred)) #f1 0.72 and recall- 0.66
print(confusion_matrix(y_test, y_pred))


## Now with all the 1 st and 2nd sem columns + the finanical variables as x variables

model_df = df.copy()

model_df = pd.get_dummies(model_df,columns=categorical_cols,drop_first=True)

x = model_df.drop(columns="dropout_status")
y = model_df["dropout_status"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=1)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

lr = LogisticRegression() #initialize
lr.fit(x_train_scaled, y_train) #train

y_pred = lr.predict(x_test_scaled)

print(classification_report(y_test, y_pred)) #f1 0.77 and recall 0.72
print(confusion_matrix(y_test, y_pred))

#!/usr/bin/env python
# coding: utf-8

# # Assignment 1 Question 7 

# In[1]:


# Mazen Al Rifai
# 20198044
# MMA
# 2021W
# MMA 869
# 16 August 2020


# Answer to Question [2], Part [a]

import pandas as pd
import numpy as np
import pandas_profiling
from pandas_profiling import ProfileReport

import matplotlib as mpl  
import matplotlib.pyplot as plt
#%matplotlib inline Used to display charts within notebook
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns 

import sklearn
import sklearn.metrics
from sklearn.preprocessing import StandardScaler

import scipy


from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# Load dataset
df = pd.read_csv("OJ.csv")


# In[2]:


#Explore data
list(df)
df.shape
df.info()
df.describe().transpose()
df.head(n=20)
df.tail()
pd.isna(df)
df.isnull().sum()
df.corr()


# In[ ]:


# Generate profile report using pandas_profiling
ProfileReport(df)


# In[3]:


#Creating target variabled Purchase_Ch

df['Purchase_CH'] = np.where(df['Purchase']=="CH",1, 0)


# In[4]:


#Convert Store7 to a binary variable
df['Store7'] = np.where(df['Store7']=="Yes",1, 0)


# In[5]:


#Check distribution for StoreID
sns.countplot(x="StoreID",data=df)


# In[6]:


#Dropping highly correlated features
df2=df.drop(["WeekofPurchase","PriceDiff","DiscCH","DiscMM","PctDiscCH","PctDiscMM","STORE","Store7","ListPriceDiff","Purchase","Unnamed: 0"],axis=1)
            


# In[ ]:


#Check for correlation again using pandas_profiling
ProfileReport(df2)


# In[7]:


# Create dummy variables for  StoreID categorical variable
y = pd.get_dummies(df2.StoreID, prefix='StoreID',drop_first=True)
df3 = pd.concat([df2, y], axis=1)


# In[8]:


# Answer to Question [2], Part [b]


# In[9]:


#The data was split to two parts 80% for training and 20% for testing. 
from sklearn.model_selection import train_test_split

# For X variable, the target variable ("Purchase_CH") was dropped since it will be predicted using the model.
X = df3.drop(["Purchase_CH"], axis=1)

#For y variable, all other variabes are dropped. This variable will be used for assessing the model peformance using the AUC metric
y = df3["Purchase_CH"]

# A random seed was assigned so that our results will be reproducible on the same machine

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


#Exploring training dataset before building the model
X.info()
X.shape
X.head()

X_train.info()
X_train.shape
X_train.head()


# In[11]:


# Answer to Question [2], Part [c]


# In[12]:


# Decision Tree classifier without hyperparameter tuning 
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X_train, y_train)


# In[13]:


#Decision Tree classifier with hyperparameter tuning 
from sklearn.model_selection import GridSearchCV

base_dtc = DecisionTreeClassifier(class_weight=None, random_state=42)
parameters_grid = {'criterion': ('gini', 'entropy'),
                  'max_depth': [1,2, 3, 4, 5, 6,7]}
DTC_grid = GridSearchCV(base_dtc, param_grid=parameters_grid, scoring="f1", n_jobs=3, cv=5, return_train_score=True)
get_ipython().run_line_magic('time', 'DTC_grid.fit(X_train, y_train)')

DTC_grid.best_params_ # printing out best parameters


# In[14]:


#Random Forest classifier without hyperparameter tuning 

from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier()
RFC.fit(X_train,y_train)


# In[15]:


#Random Forest classifier with hyperparameter tuning 

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

base_rfc = RandomForestClassifier(class_weight=None, random_state=42)
parameters_grid = {'criterion': ('gini', 'entropy'),
                  'max_depth': [2, 3, 4, 5, 6],
                  'max_leaf_nodes': [None, 5, 10, 50, 100],
                  'min_impurity_decrease': [0, 0.1, 0.2],
                  'min_samples_leaf': [1, 10, 50],
                  'min_samples_split': [2, 10, 50]}
RFC_grid = GridSearchCV(base_rfc, param_grid=parameters_grid, scoring="f1", n_jobs=3, cv=5, return_train_score=True)
get_ipython().run_line_magic('time', 'RFC_grid.fit(X_train, y_train)')
RFC_grid.best_params_ # printing out best parameters


# In[16]:


#Logistic Regression classifier without hyperparameter tuning 

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=50000)
log_reg.fit(X_train, y_train)


# In[17]:


#Logistic Regression classifier with hyperparameter tuning 

parameters_grid = {'penalty':['l1', 'l2'],'dual':[True, False],'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 'fit_intercept':[True, False],
        'solver':['saga']}

# Create grid search object

LogReg_grid = GridSearchCV(LogisticRegression(random_state=42), parameters_grid, cv=5, n_jobs=2, scoring='roc_auc')

# Fit on data

get_ipython().run_line_magic('time', 'LogReg_grid.fit(X_train, y_train)')

LogReg_grid.best_params_ # printing out best parameters


# # Estimate Model Performance for fine tuned models

# In[18]:


# Predicting performance of hypertuned DTC model

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

pred_val = DTC_grid.predict(X_val)

#Get the confusion matrix
cf_matrix = confusion_matrix(y_val,pred_val)
print(cf_matrix)

tn, fp, fn, tp = confusion_matrix(y_val,pred_val).ravel()
(tn, fp, fn, tp)

sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')


print("AUC Score = {:.3f}".format(roc_auc_score(y_val, pred_val)))
print()
print(classification_report(y_val, pred_val))


# In[19]:


# Predicting performance of hypertuned RFC model

pred_val = RFC_grid.predict(X_val)

#Get the confusion matrix
cf_matrix = confusion_matrix(y_val,pred_val)
print(cf_matrix)

tn, fp, fn, tp = confusion_matrix(y_val,pred_val).ravel()
(tn, fp, fn, tp)

sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')


print("AUC Score = {:.3f}".format(roc_auc_score(y_val, pred_val)))
print()
print(classification_report(y_val, pred_val))


# In[20]:


# Predicting performance of hypertuned Logistic Regression model

pred_val = LogReg_grid.predict(X_val)

#Get the confusion matrix
cf_matrix = confusion_matrix(y_val,pred_val)
print(cf_matrix)

tn, fp, fn, tp = confusion_matrix(y_val,pred_val).ravel()
(tn, fp, fn, tp)

sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')


print("AUC Score = {:.3f}".format(roc_auc_score(y_val, pred_val)))
print()
print(classification_report(y_val, pred_val))


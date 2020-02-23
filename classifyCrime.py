#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 14:27:07 2020

@author: abiral
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score


#%% read dataset
df = pd.read_csv("crime_ds.csv")

#%% Label Encoder
lb_make = LabelEncoder()
df['Weapon'].fillna('OTHER', inplace = True)
df['Location 1'].fillna('(39.3460800000, -76.6806500000)', inplace = True)
df['Neighborhood'].fillna('Arlington', inplace = True)


df['Weapon_code'] = lb_make.fit_transform(df['Weapon'])
df['Location 1'] = lb_make.fit_transform(df['Location 1'])
df['Neighborhood'] = lb_make.fit_transform(df['Neighborhood'])

#print(df[['Location 1','Weapon_code','Total Incidents','labels']])

#%% SPlit data to train and test
X = df.iloc[:,8:9].values
y = (df['Weapon_code'].values + 1) * df['Total Incidents'].values
y = lb_make.fit_transform(y)
print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.3,random_state=0)

#%%reshape



#%% Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_test)
#%%

model = GradientBoostingClassifier(learning_rate=0.1, 
                                       min_samples_split=50,
                                       n_estimators=100,
                                       min_samples_leaf=50,
                                       max_depth=9,
                                       max_features='sqrt',
                                       subsample=0.8)
model.fit(X_train,y_train)
testOutputs = model.predict_proba(X_test)[:,1]

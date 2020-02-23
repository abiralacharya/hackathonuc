#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 14:27:07 2020

@author: abiral
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import math
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import adjusted_rand_score
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from geopy.geocoders import Nominatim

print("Please enter address")
address=input()
geolocator = Nominatim(user_agent="specify_your_app_name_here")
location = geolocator.geocode(address)
print((location.latitude, location.longitude))

#%% read dataset
df = pd.read_csv("crime_ds.csv")


#%% Label Encoder
lb_make = LabelEncoder()
df['Weapon'].fillna('OTHER', inplace = True)
df['Location 1'].fillna('(39.3460800000, -76.6806500000)', inplace = True)
df['Neighborhood'].fillna('Arlington', inplace = True)
df['Weapon_code'] = lb_make.fit_transform(df['Weapon'])
#df['Location 1'] = lb_make.fit_transform(df['Location 1'])
#df['Neighborhood'] = lb_make.fit_transform(df['Neighborhood'])

nodes = df["Location 1"]
node=(location.latitude, location.longitude)
nodes_t =tuple(zip(nodes,nodes.index))
dist_arr = []
MAX = 100000
firstmin = MAX
secmin = MAX
thirdmin = MAX
      
for i in range(2000):
    temp = eval(nodes_t[i][0])
    dist = math.sqrt((temp[0]-node[0])**2+(temp[1]-node[1])**2)
    #print(dist)
    dist_arr.append(dist)

for i in range(2000): 
    if dist_arr[i] < firstmin: 
        thirdmin = secmin 
        secmin = firstmin 
        firstmin = dist_arr[i] 
    
    elif dist_arr[i] < secmin: 
        thirdmin = secmin 
        secmin = dist_arr[i] 
    
    elif dist_arr[i] < thirdmin: 
        thirdmin = dist_arr[i] 
#print(dist_arr)
print(min(dist_arr), firstmin, secmin, thirdmin)
index_min = dist_arr.index(firstmin)
index_secmin = dist_arr.index(secmin)
index_thirdmin =dist_arr.index(thirdmin)

min_lat = eval(nodes_t[index_min][0])[0]
min_lon = eval(nodes_t[index_min][0])[1]
Dict = {'OTHER': 1, 'HANDS': 2, 'KNIFE': 4, 'FIREARM':6}
index_attack_name = df['Weapon'][index_min] 

multiplier_1 = Dict.get(index_attack_name)
multiplier_2 = df['Total Incidents'][index_min]

point_1 = multiplier_1 * multiplier_2
print(point_1)

scaler = StandardScaler() 
a = np.loadtxt('Book2.csv',delimiter=',')
X = a[0:15,:-1]
scaler.fit(X)
X = scaler.transform(X)
y = a[0:15,-1]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
hidden_layer_sizes=(150, ), random_state=1)
clf.fit(X, y)

test = a[12:13,:-1]
year = 2020
a[12:13,:-1]=[min_lat, min_lon, year]
test = scaler.transform(a[12:13, :-1])
ans_1=clf.predict(test)

a[12:13,:-1]=[min_lat, min_lon, year-1]
test = scaler.transform(a[12:13, :-1])
ans_2=clf.predict(test)

a[12:13,:-1]=[min_lat, min_lon, year+5]
test = scaler.transform(a[12:13, :-1])
ans_3=clf.predict(test)
print(ans_1,ans_2,ans_3)
if ans_1 > ans_2:
    point_2 = -5
elif ans_1 == ans_2:
    point_2 = 0
elif ans_1 < ans_2:
    point_2 = 5
    
print(point_1+point_2)

first_min_addr=nodes_t[index_min][0]
second_min_addr=nodes_t[index_secmin][0]
third_min_addr=nodes_t[index_thirdmin][0]

print(first_min_addr, second_min_addr, third_min_addr)


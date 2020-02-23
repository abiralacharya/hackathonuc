#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 14:27:07 2020

@author: abiral
"""
import time
import webbrowser, os
import tkinter as tk
from tkinter import simpledialog
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
from datetime import datetime

while 1:
    ROOT = tk.Tk()
    
    ROOT.withdraw()
    # the input dialog
    USER_INP = simpledialog.askstring(title="Mortgage Application",
                                      prompt="Please enter the location of desired mortgage property:")
    
    # check it out
    
    
    
    print("Please enter address")
    address=USER_INP
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
    
    first_min_addr=first_min_addr.split(',')
    first_min_addr[0]=float(first_min_addr[0][1:])
    first_min_addr[1]=float(first_min_addr[1][:-1])
    first_min_addr=tuple(first_min_addr)
    
    second_min_addr=second_min_addr.split(',')
    second_min_addr[0]=float(second_min_addr[0][1:])
    second_min_addr[1]=float(second_min_addr[1][:-1])
    second_min_addr=tuple(second_min_addr)
    
    third_min_addr=third_min_addr.split(',')
    third_min_addr[0]=float(third_min_addr[0][1:])
    third_min_addr[1]=float(third_min_addr[1][:-1])
    third_min_addr=tuple(third_min_addr)
    
    import gmplot
    import webbrowser, os
    
    #Set different latitude and longitude points
    latitude1, longitude1 = zip(*[
       first_min_addr])
    #declare the center of the map, and how much we want the map zoomed in
    
    gmap3 = gmplot.GoogleMapPlotter(latitude1[0], longitude1[0], 13)
    
    
    index_attack_name_first = df['Weapon'][index_min] 
    
    multiplier_1_first = Dict.get(index_attack_name_first)
    multiplier_2_first = df['Total Incidents'][index_min]
    
    point_1_first = multiplier_1_first * multiplier_2_first
    print(point_1_first)
    
    
    index_attack_name_second = df['Weapon'][index_secmin] 
    
    multiplier_1_second = Dict.get(index_attack_name_second)
    multiplier_2_second = df['Total Incidents'][index_secmin]
    
    point_1_second = multiplier_1_second * multiplier_2_second
    print(point_1_second)
    
    
    index_attack_name_third = df['Weapon'][index_thirdmin] 
    
    multiplier_1_third = Dict.get(index_attack_name_third)
    multiplier_2_third = df['Total Incidents'][index_thirdmin]
    
    point_1_third = multiplier_1_third * multiplier_2_third
    print(point_1_third)
    
    color1='#FF0000'; color2='#FF0000';color3='#FF0000';
    if point_1_first> point_1_second and point_1_third:
        color1='#008000'
    elif point_1_second> point_1_first and point_1_third:
        color2='#008000'
    elif point_1_third> point_1_second and point_1_first:
        color3='#008000'
    if point_1_first==point_1_second==point_1_third:
        color1='#008000'; color2='#008000'; color3='#008000';
        
        
        
        
        
        
        
    
    
    # Scatter map
    gmap3.scatter( latitude1, longitude1, color1,size = 200, marker = False ) 
    
    latitude1, longitude1 = zip(*[
       second_min_addr])              
    #gmap3 = gmplot.GoogleMapPlotter(latitude1[0], longitude1[0], 13)
    # Scatter map
    gmap3.scatter( latitude1, longitude1, color2,size = 200, marker = False )
    
    latitude1, longitude1 = zip(*[
      third_min_addr])               
    
    # Scatter map
    gmap3.scatter( latitude1, longitude1, color3,size = 200, marker = False )
    # Plot method Draw a line in between given coordinates
    
                  
    #Your Google_API_Key
    gmap3.apikey = 'AIzaSyDmllc9JkG8RRgTiriuzx-mwIziEyrzB7c'
    # save it to html
    gmap3.draw("index.html")
    webbrowser.open("index.html")
    time.sleep(5)
    
    
    ROOT1 = tk.Tk()
    
    ROOT1.withdraw()
    # the input dialog
    USER_INP = simpledialog.askstring(title="Streetview",
                                      prompt="Do you want to view streetview of the best location for you?")
    
    # check it out
    if USER_INP=='yes':
        
        if point_1_first> point_1_second and point_1_third:
            link=first_min_addr
        elif point_1_second> point_1_first and point_1_third:
            link=second_min_addr
        elif point_1_third> point_1_second and point_1_first:
            link=third_min_addr
        if point_1_first==point_1_second==point_1_third:
            link=first_min_addr
        website='https://www.google.com/maps?q&layer=c&cbll={},{}'.format(link[0],link[1])
        webbrowser.open(website)
        # Download the helper library from https://www.twilio.com/docs/python/install
 
        from twilio.rest import Client
        
        
        
        
        # Your Account Sid and Auth Token from twilio.com/console
        
        # DANGER! This is insecure. See http://twil.io/secure
        
        account_sid = 'AC6228a8017bc18b85006e8db2bb751402'
        
        auth_token = '528cc84fbf17018c8ebf8f0c29bcce43'
        
        client = Client(account_sid, auth_token)
        
        
        message = client.messages.create(
        
                                      body=website,
        
                                      from_='+16237772033',
        
                                      
        
                                      to='+14193221625'
        
                                  )


        print(message.sid)
    
    if USER_INP=='no':
        break
        
        

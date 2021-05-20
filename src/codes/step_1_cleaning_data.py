#!/usr/bin/env python
# coding: utf-8
#%matplotlib notebook
import datetime, warnings, scipy 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from collections import Counter
from datetime import datetime, date, time
import seaborn as sns
from pandas.plotting import scatter_matrix
from matplotlib import cm
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
font = {'size'   : 16}
matplotlib.rc('font', **font)
import sys,os
import pathlib
## Read Data of Flights!
path_file = open("input",'r')
path_file = path_file.read().split('\n')
flights_step_1 = pd.read_csv(path_file[1])
#flights_step_1 = pd.read_csv('codes/flights.csv')
#print("flights.shape:", flights_step_1.shape)
#print("flights.columns:", flights_step_1.columns)

## Look at Data!
#print("flights.info:")
#print(flights_step_1.info())

missing_flights = flights_step_1.isnull().sum(axis=0).reset_index()
missing_flights.columns = ['variable', 'missing values']
missing_flights['filling factor (%)']=(flights_step_1.shape[0]-missing_flights['missing values'])/flights_step_1.shape[0]*100
missing_flights.sort_values('filling factor (%)').reset_index(drop = True)

## Clean up Data!
# drop up some destorbuted features & remove rows including Nan data!
flights_step_1.drop(['CANCELLATION_REASON', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'], axis = 1, inplace = True)
flights_step_1.dropna(inplace = True)
missing_flights = flights_step_1.isnull().sum(axis=0).reset_index()
missing_flights.columns = ['variable', 'missing values']
missing_flights['filling factor (%)']=(flights_step_1.shape[0]-missing_flights['missing values'])/flights_step_1.shape[0]*100
#print('flights.columns:', flights_step_1.columns)
missing_flights.sort_values('filling factor (%)').reset_index(drop = True)

## Check the Data!
#print("flights.unique:", flights_step_1['AIRLINE'].unique())
lookup_flight_features = ['AIRLINE', 'SCHEDULED_TIME']
XX = flights_step_1[lookup_flight_features].head(5)
sns.stripplot(x='AIRLINE',y= 'SCHEDULED_TIME',data=XX,jitter=True,palette='Set1')

#Have a look at flights dataFrame!
#print("new_flights_dataFrame.shape:",flights_step_1.shape)
#print(flights_step_1.loc[:13555, ['SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL', 'DEPARTURE_TIME',
#             'ARRIVAL_TIME', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY']])

## Save Data!
#save a dataFrame to flights_step_1.csv file
flights_step_1.to_csv('flights_step_1.csv')


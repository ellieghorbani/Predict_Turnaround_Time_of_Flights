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

## Read Data of flights_step_2.csv!
flights_step_3 = pd.read_csv('flights_step_2.csv')
#print("flights.shape:", flights_step_3.shape)

## Check the Data!
#print("flights.columns:",flights_step_3.columns)

## Check the dataFrame flights_step_3
flights_step_3 = flights_step_3[['FLIGHT_NUMBER', 'TAIL_NUMBER','AIRLINE',  'ORIGIN_AIRPORT',
       'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME',
       'DEPARTURE_DELAY','SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY', 
       'SCHEDULED_TIME', 'ELAPSED_TIME', 'DISTANCE']]

## Explanation features more and their relationships together!
#DEPARTURE_TIME - SCHEDULED_DEPARTURE.dt.time = DEPARTURE_DELAY
#ARRIVAL_TIME - SCHEDULED_ARRIVAL.dt.time = ARRIVAL_DELAY
#for a TAIL_NUMBER and a AIRLINE 
#airports: A -->-- B -->-- C
#ORIGIN_AIRPORT|DESTINATION_AIRPORT|SCHEDULED_DEPARTURE|DEPARTURE_TIME|DEPARTURE_DELAY|SCHEDULED_ARRIVAL|ARRIVAL_TIME|ARRIVAL_DELAY|SCHEDULED_TIME|ELAPSED_TIME|DISTANCE
#--------------|-------------------|-------------------|--------------|---------------|-----------------|------------|-------------|--------------|------------|--------
#      A       |           B       |       t(AB)_sd    |     t(AB)d   |   ^t(AB)_dd   |      t(AB)_sa   |    t(AB)a  |  ^t(AB)_ad  |     -        |      -     |   -
#--------------|-------------------|-------------------|--------------|---------------|-----------------|------------|-------------|--------------|------------|--------
#      B       |           c       |       t(BC)_sd    |     t(BC)d   |   ^t(BC)_dd   |      t(BC)_sa   |    t(BC)a  |  ^t(BC)_ad  |     -        |      -     |   -
#--------------|-------------------|-------------------|--------------|---------------|-----------------|------------|-------------|--------------|------------|--------

# In[6]:


#sort fights based on Tail number and scheduled departure!
flights_step_3 = flights_step_3.sort_values(by=['TAIL_NUMBER', 'SCHEDULED_DEPARTURE'])
n = len(flights_step_3['TAIL_NUMBER'])
flights_step_3 = flights_step_3.set_index([pd.Index(range(0,n))])

## Make a new dataFrame 
# make a dataFrame for flights_step_3 from B airport to C airport!
flights_step_3_BC = flights_step_3
flights_step_3_BC = flights_step_3_BC.set_axis(['FLIGHT_NUMBER2', 'TAIL_NUMBER2', 'AIRLINE2', 'ORIGIN_AIRPORT2', 'DESTINATION_AIRPORT2', 'SCHEDULED_DEPARTURE2', 'DEPARTURE_TIME2',
       'DEPARTURE_DELAY2', 'SCHEDULED_ARRIVAL2', 'ARRIVAL_TIME2', 'ARRIVAL_DELAY2',
       'SCHEDULED_TIME2', 'ELAPSED_TIME2', 'DISTANCE2'], axis=1, inplace=False)
flights_step_3_BC.drop([0], inplace = True) #drop row zero from flights_step_3_BC
new_row = flights_step_3_BC.iloc[0:1]
flights_step_3_BC = flights_step_3_BC.append(new_row, ignore_index = True) #add new row to flights_step_3_BC

#attached flights_step_3_BC to flights_step_3_AB
flights_step_3_ABC = pd.concat([flights_step_3, flights_step_3_BC], axis=1, sort=False)

# match rows with differnt B airports or/and different tail number! 
compare_airport, compare_tail = np.array(len(flights_step_3_ABC.DESTINATION_AIRPORT)), np.array(len(flights_step_3_ABC.DESTINATION_AIRPORT))
compare_airport = np.where(flights_step_3_ABC.DESTINATION_AIRPORT == flights_step_3_ABC.ORIGIN_AIRPORT2, 'True', 'False')
compare_tail = np.where(flights_step_3_ABC.TAIL_NUMBER == flights_step_3_ABC.TAIL_NUMBER2, 'True', 'False')
#print('type of compare_airport = %s & size of compare_airport= %d' %(type(compare_airport), compare_airport.size))
#print('compare_airport:',compare_airport[:10])
#print('number of Ture & False in index_airport:', list(Counter(compare_airport).values()))
#print('----------------------------------------------------')
#print('type of compare_tail = %s & size of compare_tail= %d' %(type(compare_tail), compare_tail.size))
#print('compare_tail:',compare_tail[:10])
#print('number of Ture & False in index_tail:', list(Counter(compare_tail).values()))

#find indexes of rows with differnt B airports or different tail number and remove them!
index_airport = [i for i in range(compare_airport.size) if compare_airport[i] == "False"] 
index_tail = [i for i in range(compare_tail.size) if compare_tail[i] == "False"] 
remove_index =[x for x in index_airport if x not in index_tail]
remove_index = (remove_index + index_tail)
remove_index.sort() 

#print("type & size of index_airport:", type(index_airport), len(index_airport))
#print("type & size of index_airport:", type(index_tail), len(index_tail))
#print("type & size of index_airport:", type(remove_index), len(remove_index))
#print('index_airport:', index_airport[:10])
#print('   index_tail:', index_tail[:10])
#print(' remove_index:', remove_index[:10])

## Define The Target!
# make a dataFrame based on airport B: airport_based_dataframe
#  airport A          --->>---          airport B              --->>---    airport C

#   SD_AB/DT_AB       --->>---    SA_AB/DT_AB | SD_BC/DT_BC     --->>--- 
#  
#                           ***turnaround_time_ B= DT_BC - AT_AB***

# In[12]:

airport_based_dataframe = flights_step_3_ABC.drop(flights_step_3_ABC.index[remove_index])
airport_based_dataframe.drop(['FLIGHT_NUMBER', 'FLIGHT_NUMBER2', 'TAIL_NUMBER2', 'ORIGIN_AIRPORT2','AIRLINE2'], axis = 1, inplace = True)

airport_based_dataframe.rename(columns={"ORIGIN_AIRPORT": "airport_A", "DESTINATION_AIRPORT": "airport_B", 
                        "DESTINATION_AIRPORT2": "airport_C", "SCHEDULED_DEPARTURE": "SCHEDULED_DEPARTURE_AB", "DEPARTURE_DELAY": "DEPARTURE_DELAY_AB", "SCHEDULED_ARRIVAL": "SCHEDULED_ARRIVAL_AB",
                        "SCHEDULED_DEPARTURE2": "SCHEDULED_DEPARTURE_BC", "DEPARTURE_DELAY2": "DEPARTURE_DELAY_BC",  "ARRIVAL_DELAY": "ARRIVAL_DELAY_AB", "SCHEDULED_TIME": "SCHEDULED_TIME_AB", 
                        "SCHEDULED_TIME2": "SCHEDULED_TIME_BC", "ELAPSED_TIME": "ELAPSED_TIME_AB", 
                        "ELAPSED_TIME2": "ELAPSED_TIME_BC", "DISTANCE": "DISTANCE_AB", 
                        "DISTANCE2": "DISTANCEBC", 'DEPARTURE_TIME': 'DEPARTURE_TIME_AB', 
                        'DEPARTURE_TIME2': 'DEPARTURE_TIME_BC', 'ARRIVAL_TIME':'ARRIVAL_TIME_AB',
                        'ARRIVAL_TIME2' : 'ARRIVAL_TIME_BC'}, inplace = True)
airport_based_dataframe['DEPARTURE_TIME_BC'] = pd.to_datetime(airport_based_dataframe['DEPARTURE_TIME_BC'])
airport_based_dataframe['ARRIVAL_TIME_AB'] = pd.to_datetime(airport_based_dataframe['ARRIVAL_TIME_AB'])
##print('dtype:', airport_based_dataframe['DEPARTURE_TIME_BC'].dtype)
airport_based_dataframe['turnaround_time_ B'] = (airport_based_dataframe['DEPARTURE_TIME_BC'] - airport_based_dataframe['ARRIVAL_TIME_AB'])/np.timedelta64(1,'h')
##print('***',airport_based_dataframe.shape)
airport_based_dataframe = airport_based_dataframe[['TAIL_NUMBER', 'AIRLINE', 
                   'airport_A', 'airport_B', 'airport_C','turnaround_time_ B', 
                   "SCHEDULED_DEPARTURE_AB", 'DEPARTURE_TIME_AB', "DEPARTURE_DELAY_AB", 
                   "SCHEDULED_ARRIVAL_AB", 'ARRIVAL_TIME_AB', 'ARRIVAL_DELAY_AB',
                   "SCHEDULED_DEPARTURE_BC", 'DEPARTURE_TIME_BC', "DEPARTURE_DELAY_BC", 
                   "ELAPSED_TIME_AB", "ELAPSED_TIME_BC", 
                   "DISTANCE_AB", "DISTANCEBC"]]

#print("new.dataFrame.shape:", airport_based_dataframe.shape)
#print("new_dataFrame.head:")
#print(airport_based_dataframe.head(10))

#print("target.describe:")
#print(airport_based_dataframe['turnaround_time_ B'].describe())

test = airport_based_dataframe[(airport_based_dataframe['turnaround_time_ B'] > -5) & 
                               (airport_based_dataframe['turnaround_time_ B'] < 24) ]['turnaround_time_ B']
plt.hist(test, bins =100)
plt.show()

#print("dataFrame.shape")
#print('Before drop data:', airport_based_dataframe.shape)
airport_based_dataframe = airport_based_dataframe[(airport_based_dataframe['turnaround_time_ B'] > -5) & 
                               (airport_based_dataframe['turnaround_time_ B'] < 24) ]
#print('After drop data:', airport_based_dataframe.shape)

## Save the Data!
airport_based_dataframe.to_csv('flights_step_3.csv')





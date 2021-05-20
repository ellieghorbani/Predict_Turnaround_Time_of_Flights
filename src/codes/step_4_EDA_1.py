#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date, time 
from scipy import stats

## Read Data!
data = pd.read_csv("flights_step_3.csv")
#print('shape of data:', data.shape)
#print('data.head:')
#print(data.head(4))

#functions#####
#Seperate data based on the airlines in new dataFrame
def seperate_data_based_on_airline(airline, data):
    a = data[data['AIRLINE'] == airline]
    #print('shape of data_airline {} = {}'.format(airline, a.shape))
    return a
    #data_airline.to_csv('data_{}.csv'.format(airline))
#*******tables******
def compare_number_flights_based_on_air_portB(airlines, name_of_airline): 
    a = np.transpose([len(airlines[name]['airport_B']) for name in name_of_airline])
    b = np.transpose([airlines[name]['airport_B'].value_counts().describe().mean() for name in name_of_airline])
    c = np.transpose([airlines[name]['airport_B'].value_counts().describe().std() for name in name_of_airline])
    d = np.transpose([airlines[name]['airport_B'].value_counts().describe().min() for name in name_of_airline])
    e = np.transpose([airlines[name]['airport_B'].value_counts().describe().max() for name in name_of_airline])
    df = pd.DataFrame((a, b, c, d, e),
             index = ['count_flight','mean', 'std', 'min', 'max'], columns = name_of_airline).round(0)
    return df.sort_values(by = ['count_flight'], axis = 1)

# In[5]:


compare = pd.DataFrame([format(np.unique(data['AIRLINE']).shape),
                        format(np.unique(data["airport_A"]).shape),
                        format(np.unique(data["airport_B"]).shape),
                        format(np.unique(data["airport_C"]).shape)],
                       columns = ['size'], index = ['AIRLINE',"airport_A", "airport_B", "airport_C"])
#print(compare)

#name of airlines
name_of_airline = np.unique(data.AIRLINE)
# Seperate data based on the airlines in new dataFrame
airlines = {}
for i in name_of_airline:
     airlines[i] = pd.DataFrame(seperate_data_based_on_airline(i, data))

df = compare_number_flights_based_on_air_portB(airlines, name_of_airline)
#print("compare number flights based on airport B")
#print(df)

'''
'turnaround_time_ B'

'TAIL_NUMBER', 'AIRLINE',

'airport_A', 'airport_B', 'airport_C',

''SCHEDULED_DEPARTURE_AB'', 'DEPARTURE_TIME_AB', 'DEPARTURE_DELAY_AB', 
''SCHEDULED_ARRIVAL_AB''  , 'ARRIVAL_TIME_AB'  , 'ARRIVAL_DELAY_AB'  , 
''SCHEDULED_DEPARTURE_BC'', 'DEPARTURE_TIME_BC', 'DEPARTURE_DELAY_BC', 
'ELAPSED_TIME_AB', 'ELAPSED_TIME_BC', 
'DISTANCE_AB', 'DISTANCEBC'

functions:
    1- for ariport: data and average data of TAT via airport_B, A, C
    2- for date: data and average data of TAT via arrival time, day of week, day of month and month
    3- for delay, elapsed, distance: data and average data of TAT via delay, elapsed, distance
'''
## EDA
#data and average data of TAT via airport_B, A, C
def plot_TAT_via_airport(data_airline, name_airline, airport):
    c = ['blue','lightblue', 'red', 'pink']
    cc = 0
    length = int(len(name_airline)/2)+int(len(name_airline)%2)
    fig, axs = plt.subplots(length, 2, figsize = (15, 35))
    l, k, m = 0, 0, 0
    for j in name_airline:
            k = (m)//(length)
            l = m%(length)
            a = data_airline[data_airline['AIRLINE'] == j]
            list_airport = pd.unique(a[airport])
            for i in list_airport:
                interval = a[a[airport] == i]
                x = [i,i]
                y = [min(interval['turnaround_time_ B']), max(interval['turnaround_time_ B'])]
                axs[l][k].plot(x, y,color = c[cc+1])
                axs[l][k].plot(i,np.mean(interval['turnaround_time_ B']),
                              marker = 'o', color = c[cc])
                #plt.xticks(rotation=90)
                
            if cc == 0 or cc == 2:
                axs[l][k].scatter(0,0,color ='white',label = j)
            
            axs[l][k].set_xlabel(airport)
            axs[l][k].set_ylabel('turnaround_time_ B', color = 'blue')
            
            axs[l][k].legend()
            m += 1
    cc += 1
    #plt.show()
plot_TAT_via_airport(data, name_of_airline, "airport_A")
plot_TAT_via_airport(data, name_of_airline, "airport_B")
plot_TAT_via_airport(data, name_of_airline, "airport_C")


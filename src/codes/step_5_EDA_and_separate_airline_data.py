#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date, time 
from scipy import stats

## Read Data!
data = pd.read_csv("flights_step_3.csv")
#data = data.head(2000000)
##print('shape of data:', data.shape)
#data.head(4)

## EDA
## Ariport_B

#functions#####
#Seperate data based on the airlines in new dataFrame
def seperate_data_based_on_airline(airline, data):
    a = data[data['AIRLINE'] == airline]
    #print('shape of data_airline {} = {}, %{}'.format(airline, a.shape, len(a)/len(data)*100))
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

#Remove some airports_B with minority number flights
def remove_airport_B_with_less_than_25_precent_flights_from_average_number_flights(airlines, name_of_airline):
    for name in name_of_airline:
        a = airlines[name]
        b = a['airport_B'].value_counts().mean()
        a_count = a['airport_B'].value_counts()
        index = a_count.index
        for j in range(len(a_count)):
            if (a_count[j]<0.1*b):
                    a.drop(a[a['airport_B'] == index[j]].index, inplace=True)  

# plot TAT based on departure delay AB:
def departure_delay_AB(data_airline,name):
    length = int(len(name)/2)+int(len(name)%2)
    l, k, m = 0, 0, 0
    fig, axs = plt.subplots(length, 2, figsize = (12, 35))
    #ax2 = axs.twinx()
    for j in name:
        k = (m)//(length)
        l = m%(length)
        ax2 = axs[l][k].twinx()
        a = data_airline[j]
        min_dep, max_dep = min(a['DEPARTURE_DELAY_AB']), max(a['DEPARTURE_DELAY_AB'])
        for i in np.arange(min_dep, max_dep, 10):
            interval = a[i < a['DEPARTURE_DELAY_AB']]
            interval = interval[interval['DEPARTURE_DELAY_AB']> i+10]
            axs[l][k].scatter(np.mean(interval['DEPARTURE_DELAY_AB']),
                              np.mean(interval['turnaround_time_ B']),
                              marker = 'o', color = 'blue')
            ax2.scatter(np.mean(interval['DEPARTURE_DELAY_AB']),
                              np.log(len(interval['DEPARTURE_DELAY_AB'])),
                              marker = 'o', color = 'red')
        axs[l][k].scatter(0,0,color ='white',label = j)
        axs[l][k].set_xlabel('DEPARTURE_DELAY_AB')
        axs[l][k].set_ylabel('turnaround_time_ B', color = 'blue')
        ax2.set_ylabel('log(Number of Departure Delay AB)', color = 'red')
        axs[l][k].legend()
        m += 1
# remove TAT based on departure delay AB:
def remove_departure_delay_AB_with_abondens_less_than_1percent(data_airline,name):
    for j in name:       
        a = data_airline[j]
        length = len(a['DEPARTURE_DELAY_AB'])
        min_dep, max_dep = min(a['DEPARTURE_DELAY_AB']), max(a['DEPARTURE_DELAY_AB'])
        for i in np.arange(min_dep, max_dep, 10):
            interval = a[i < a['DEPARTURE_DELAY_AB']]
            interval = interval[interval['DEPARTURE_DELAY_AB']> i+10]
            if len(interval['DEPARTURE_DELAY_AB'])/length < 0.01:
                   a.drop(interval.index, inplace = True)
#compare airline data before and after remove some DEPARTURE_DELAY_AB data
def compare_airline_data_before_and_after_remove_DEPARTURE_DELAY_AB_data_based_on_abondens_less_than_1percent(data_airline,name):
    bef, aft = [], []
    [bef.append(data_airline[j].shape) for j in name]
    remove_departure_delay_AB_with_abondens_less_than_1percent(data_airline,name)
    [aft.append(data_airline[j].shape) for j in name]
    bef = np.reshape(bef,(len(bef),2))
    aft = np.reshape(aft,(len(aft),2))
    b = pd.DataFrame([[bef[i][0], aft[i][0],(bef[i][0]-aft[i][0])*100/ bef[i][0]] for i in range(len(name))], 
                       columns = ['before', 'after', 'missing data %'], 
                       index = name)
    return b

compare = pd.DataFrame([format(np.unique(data['AIRLINE']).shape),
                        format(np.unique(data["airport_A"]).shape),
                        format(np.unique(data["airport_B"]).shape),
                        format(np.unique(data["airport_C"]).shape)],
                       columns = ['size'], index = ['AIRLINE',"airport_A", "airport_B", "airport_C"])

#name of airlines
name_of_airline = np.unique(data.AIRLINE)
# Seperate data based on the airlines in new dataFrame
airlines = {}
for i in name_of_airline:
     airlines[i] = pd.DataFrame(seperate_data_based_on_airline(i, data))

df = compare_number_flights_based_on_air_portB(airlines, name_of_airline)
remove_airport_B_with_less_than_25_precent_flights_from_average_number_flights(airlines, name_of_airline)
df_new = compare_number_flights_based_on_air_portB(airlines, name_of_airline)    

##print('number flights before remove some airport_B:',df.sum(axis = 1)[0])
##print('number flights after remove some airport_B:',df_new.sum(axis = 1)[0])

## DEPARTURE_DELAY_AB
#departure_delay_AB(airlines, name_of_airline)
compare_airline_data_before_and_after_remove_DEPARTURE_DELAY_AB_data_based_on_abondens_less_than_1percent(airlines,name_of_airline)

## ARRIVAL_DELAY_AB
#functions##### 

# plot TAT based on departure delay AB:
def arrival_delay_AB(data_airline,name):
    length = int(len(name)/2)+int(len(name)%2)
    l, k, m = 0, 0, 0
    fig, axs = plt.subplots(length, 2, figsize = (12, 35))
    #ax2 = axs.twinx()
    for j in name:
        k = (m)//(length)
        l = m%(length)
        ax2 = axs[l][k].twinx()
        a = data_airline[j]
        min_dep, max_dep = min(a['ARRIVAL_DELAY_AB']), max(a['ARRIVAL_DELAY_AB'])
        for i in np.arange(min_dep, max_dep, 10):
            interval = a[i < a['ARRIVAL_DELAY_AB']]
            interval = interval[interval['ARRIVAL_DELAY_AB']> i+10]
            axs[l][k].scatter(np.mean(interval['ARRIVAL_DELAY_AB']),
                              np.mean(interval['turnaround_time_ B']),
                              marker = 'o', color = 'blue')
            ax2.scatter(np.mean(interval['ARRIVAL_DELAY_AB']),
                              np.log(len(interval['ARRIVAL_DELAY_AB'])),
                              marker = 'o', color = 'red')
        axs[l][k].scatter(0,0,color ='white',label = j)
        axs[l][k].set_xlabel('ARRIVAL_DELAY_AB')
        axs[l][k].set_ylabel('turnaround_time_ B', color = 'blue')
        ax2.set_ylabel('log(Number of arrival Delay AB)', color = 'red')
        axs[l][k].legend()
        m += 1
# remove TAT based on departure delay AB:
def remove_arrival_delay_AB_with_abondens_less_than_1percent(data_airline,name):
    for j in name:       
        a = data_airline[j]
        length = len(a['ARRIVAL_DELAY_AB'])
        min_dep, max_dep = min(a['ARRIVAL_DELAY_AB']), max(a['ARRIVAL_DELAY_AB'])
        for i in np.arange(min_dep, max_dep, 10):
            interval = a[i < a['ARRIVAL_DELAY_AB']]
            interval = interval[interval['ARRIVAL_DELAY_AB']> i+10]
            if len(interval['ARRIVAL_DELAY_AB'])/length < 0.01:
                   a.drop(interval.index, inplace = True)
#compare airline data before and after remove some DEPARTURE_DELAY_AB data
def compare_airline_data_before_and_after_remove_ARRIVAL_DELAY_AB_data_based_on_abondens_less_than_1percent(data_airline,name):
    bef, aft = [], []
    [bef.append(data_airline[j].shape) for j in name]
    remove_arrival_delay_AB_with_abondens_less_than_1percent(data_airline,name)
    [aft.append(data_airline[j].shape) for j in name]
    bef = np.reshape(bef,(len(bef),2))
    aft = np.reshape(aft,(len(aft),2))
    b = pd.DataFrame([[bef[i][0], aft[i][0],(bef[i][0]-aft[i][0])*100/ bef[i][0]] for i in range(len(name))], 
                       columns = ['before', 'after', 'missing data %'], 
                       index = name)
    return b

#arrival_delay_AB(airlines, name_of_airline)

compare_airline_data_before_and_after_remove_ARRIVAL_DELAY_AB_data_based_on_abondens_less_than_1percent(airlines,name_of_airline)

## DEPARTURE_DELAY_BC
#functions##### 

# plot TAT based on departure delay BC:
def departure_delay_BC(data_airline,name):
    length = int(len(name)/2)+int(len(name)%2)
    l, k, m = 0, 0, 0
    fig, axs = plt.subplots(length, 2, figsize = (12, 35))
    #ax2 = axs.twinx()
    for j in name:
        k = (m)//(length)
        l = m%(length)
        ax2 = axs[l][k].twinx()
        a = data_airline[j]
        min_dep, max_dep = min(a['DEPARTURE_DELAY_BC']), max(a['DEPARTURE_DELAY_BC'])
        for i in np.arange(min_dep, max_dep, 10):
            interval = a[i < a['DEPARTURE_DELAY_BC']]
            interval = interval[interval['DEPARTURE_DELAY_BC']> i+10]
            axs[l][k].scatter(np.mean(interval['DEPARTURE_DELAY_BC']),
                              np.mean(interval['turnaround_time_ B']),
                              marker = 'o', color = 'blue')
            ax2.scatter(np.mean(interval['DEPARTURE_DELAY_BC']),
                              np.log(len(interval['DEPARTURE_DELAY_BC'])),
                              marker = 'o', color = 'red')
        axs[l][k].scatter(0,0,color ='white',label = j)
        axs[l][k].set_xlabel('DEPARTURE_DELAY_BC')
        axs[l][k].set_ylabel('turnaround_time_ B', color = 'blue')
        ax2.set_ylabel('log(Number of departure Delay BC)', color = 'red')
        axs[l][k].legend()
        m += 1
# remove TAT based on departure delay BC:
def remove_departure_delay_BC_with_abondens_less_than_1percent(data_airline,name):
    for j in name:       
        a = data_airline[j]
        length = len(a['DEPARTURE_DELAY_BC'])
        min_dep, max_dep = min(a['DEPARTURE_DELAY_BC']), max(a['DEPARTURE_DELAY_BC'])
        for i in np.arange(min_dep, max_dep, 10):
            interval = a[i < a['DEPARTURE_DELAY_BC']]
            interval = interval[interval['DEPARTURE_DELAY_BC']> i+10]
            if len(interval['DEPARTURE_DELAY_BC'])/length < 0.01:
                   a.drop(interval.index, inplace = True)
#compare airline data before and after remove some DEPARTURE_DELAY_BC data
def compare_airline_data_before_and_after_remove_DEPARTURE_DELAY_BC_data_based_on_abondens_less_than_1percent(data_airline,name):
    bef, aft = [], []
    [bef.append(data_airline[j].shape) for j in name]
    remove_departure_delay_BC_with_abondens_less_than_1percent(data_airline,name)
    [aft.append(data_airline[j].shape) for j in name]
    bef = np.reshape(bef,(len(bef),2))
    aft = np.reshape(aft,(len(aft),2))
    b = pd.DataFrame([[bef[i][0], aft[i][0],(bef[i][0]-aft[i][0])*100/ bef[i][0]] for i in range(len(name))], 
                       columns = ['before', 'after', 'missing data %'], 
                       index = name)
    return b

#departure_delay_BC(airlines, name_of_airline)
compare_airline_data_before_and_after_remove_DEPARTURE_DELAY_BC_data_based_on_abondens_less_than_1percent(airlines,name_of_airline)


import datetime
data.columns
pd.to_datetime(data['SCHEDULED_DEPARTURE_AB']).dt.time

## ELAPSED_TIME_AB
#functions##### 

# plot TAT based on departure delay BC:
def elapsed_time_AB(data_airline,name):
    length = int(len(name)/2)+int(len(name)%2)
    l, k, m = 0, 0, 0
    fig, axs = plt.subplots(length, 2, figsize = (12, 35))
    #ax2 = axs.twinx()
    for j in name:
        k = (m)//(length)
        l = m%(length)
        ax2 = axs[l][k].twinx()
        a = data_airline[j]
        min_dep, max_dep = min(a['ELAPSED_TIME_AB']), max(a['ELAPSED_TIME_AB'])
        for i in np.arange(min_dep, max_dep, 10):
            interval = a[i < a['ELAPSED_TIME_AB']]
            interval = interval[interval['ELAPSED_TIME_AB']> i+10]
            axs[l][k].scatter(np.mean(interval['ELAPSED_TIME_AB']),
                              np.mean(interval['turnaround_time_ B']),
                              marker = 'o', color = 'blue')
            ax2.scatter(np.mean(interval['ELAPSED_TIME_AB']),
                              np.log(len(interval['ELAPSED_TIME_AB'])),
                              marker = 'o', color = 'red')
        axs[l][k].scatter(0,0,color ='white',label = j)
        axs[l][k].set_xlabel('ELAPSED_TIME_AB')
        axs[l][k].set_ylabel('turnaround_time_ B', color = 'blue')
        ax2.set_ylabel('log(Number of ELAPSED_TIME_AB)', color = 'red')
        axs[l][k].legend()
        m += 1
# remove TAT based on departure delay BC:
def remove_elapsed_time_AB_with_abondens_less_than_1percent(data_airline,name):
    for j in name:       
        a = data_airline[j]
        length = len(a['ELAPSED_TIME_AB'])
        min_dep, max_dep = min(a['ELAPSED_TIME_AB']), max(a['ELAPSED_TIME_AB'])
        for i in np.arange(min_dep, max_dep, 10):
            interval = a[i < a['ELAPSED_TIME_AB']]
            interval = interval[interval['ELAPSED_TIME_AB']> i+10]
            if len(interval['ELAPSED_TIME_AB'])/length < 0.01:
                   a.drop(interval.index, inplace = True)
#compare airline data before and after remove some DEPARTURE_DELAY_BC data
def compare_airline_data_before_and_after_remove_ELAPSED_TIME_AB_data_based_on_abondens_less_than_1percent(data_airline,name):
    bef, aft = [], []
    [bef.append(data_airline[j].shape) for j in name]
    remove_elapsed_time_AB_with_abondens_less_than_1percent(data_airline,name)
    [aft.append(data_airline[j].shape) for j in name]
    bef = np.reshape(bef,(len(bef),2))
    aft = np.reshape(aft,(len(aft),2))
    b = pd.DataFrame([[bef[i][0], aft[i][0],(bef[i][0]-aft[i][0])*100/ bef[i][0]] for i in range(len(name))], 
                       columns = ['before', 'after', 'missing data %'], 
                       index = name)
    return b

#elapsed_time_AB(airlines, name_of_airline)
compare_airline_data_before_and_after_remove_ELAPSED_TIME_AB_data_based_on_abondens_less_than_1percent(airlines,name_of_airline)

## ELAPSED_TIME_BC
#functions##### 

# plot TAT based on departure delay BC:
def elapsed_time_BC(data_airline,name):
    length = int(len(name)/2)+int(len(name)%2)
    l, k, m = 0, 0, 0
    fig, axs = plt.subplots(length, 2, figsize = (12, 35))
    #ax2 = axs.twinx()
    for j in name:
        k = (m)//(length)
        l = m%(length)
        ax2 = axs[l][k].twinx()
        a = data_airline[j]
        min_dep, max_dep = min(a['ELAPSED_TIME_BC']), max(a['ELAPSED_TIME_BC'])
        for i in np.arange(min_dep, max_dep, 10):
            interval = a[i < a['ELAPSED_TIME_BC']]
            interval = interval[interval['ELAPSED_TIME_BC']> i+10]
            axs[l][k].scatter(np.mean(interval['ELAPSED_TIME_BC']),
                              np.mean(interval['turnaround_time_ B']),
                              marker = 'o', color = 'blue')
            ax2.scatter(np.mean(interval['ELAPSED_TIME_BC']),
                              np.log(len(interval['ELAPSED_TIME_BC'])),
                              marker = 'o', color = 'red')
        axs[l][k].scatter(0,0,color ='white',label = j)
        axs[l][k].set_xlabel('ELAPSED_TIME_BC')
        axs[l][k].set_ylabel('turnaround_time_ B', color = 'blue')
        ax2.set_ylabel('log(Number of ELAPSED_TIME_BC)', color = 'red')
        axs[l][k].legend()
        m += 1
# remove TAT based on departure delay BC:
def remove_elapsed_time_BC_with_abondens_less_than_1percent(data_airline,name):
    for j in name:       
        a = data_airline[j]
        length = len(a['ELAPSED_TIME_BC'])
        min_dep, max_dep = min(a['ELAPSED_TIME_BC']), max(a['ELAPSED_TIME_BC'])
        for i in np.arange(min_dep, max_dep, 10):
            interval = a[i < a['ELAPSED_TIME_BC']]
            interval = interval[interval['ELAPSED_TIME_BC']> i+10]
            if len(interval['ELAPSED_TIME_BC'])/length < 0.01:
                   a.drop(interval.index, inplace = True)
#compare airline data before and after remove some DEPARTURE_DELAY_BC data
def compare_airline_data_before_and_after_remove_ELAPSED_TIME_BC_data_based_on_abondens_less_than_1percent(data_airline,name):
    bef, aft = [], []
    [bef.append(data_airline[j].shape) for j in name]
    remove_elapsed_time_BC_with_abondens_less_than_1percent(data_airline,name)
    [aft.append(data_airline[j].shape) for j in name]
    bef = np.reshape(bef,(len(bef),2))
    aft = np.reshape(aft,(len(aft),2))
    b = pd.DataFrame([[bef[i][0], aft[i][0],(bef[i][0]-aft[i][0])*100/ bef[i][0]] for i in range(len(name))], 
                       columns = ['before', 'after', 'missing data %'], 
                       index = name)
    return b

#elapsed_time_BC(airlines, name_of_airline)
compare_airline_data_before_and_after_remove_ELAPSED_TIME_BC_data_based_on_abondens_less_than_1percent(airlines,name_of_airline)


## DISTANCE_AB
#functions##### 

# plot TAT based on departure delay BC:
def DISTANCE_AB(data_airline,name):
    length = int(len(name)/2)+int(len(name)%2)
    l, k, m = 0, 0, 0
    fig, axs = plt.subplots(length, 2, figsize = (12, 35))
    #ax2 = axs.twinx()
    for j in name:
        k = (m)//(length)
        l = m%(length)
        ax2 = axs[l][k].twinx()
        a = data_airline[j]
        min_dep, max_dep = min(a['DISTANCE_AB']), max(a['DISTANCE_AB'])
        for i in np.arange(min_dep, max_dep, 10):
            interval = a[i < a['DISTANCE_AB']]
            interval = interval[interval['DISTANCE_AB']> i+10]
            axs[l][k].scatter(np.mean(interval['DISTANCE_AB']),
                              np.mean(interval['turnaround_time_ B']),
                              marker = 'o', color = 'blue')
            ax2.scatter(np.mean(interval['DISTANCE_AB']),
                              np.log(len(interval['DISTANCE_AB'])),
                              marker = 'o', color = 'red')
        axs[l][k].scatter(0,0,color ='white',label = j)
        axs[l][k].set_xlabel('DISTANCE_AB')
        axs[l][k].set_ylabel('turnaround_time_ B', color = 'blue')
        ax2.set_ylabel('log(Number of DISTANCE_AB)', color = 'red')
        axs[l][k].legend()
        m += 1
# remove TAT based on departure delay BC:
def remove_distance_AB_with_abondens_less_than_1percent(data_airline,name):
    for j in name:       
        a = data_airline[j]
        length = len(a['DISTANCE_AB'])
        min_dep, max_dep = min(a['DISTANCE_AB']), max(a['DISTANCE_AB'])
        for i in np.arange(min_dep, max_dep, 10):
            interval = a[i < a['DISTANCE_AB']]
            interval = interval[interval['DISTANCE_AB']> i+10]
            if len(interval['DISTANCE_AB'])/length < 0.01:
                   a.drop(interval.index, inplace = True)
#compare airline data before and after remove some DEPARTURE_DELAY_BC data
def compare_airline_data_before_and_after_remove_DISTANCE_AB_data_based_on_abondens_less_than_1percent(data_airline,name):
    bef, aft = [], []
    [bef.append(data_airline[j].shape) for j in name]
    remove_elapsed_time_BC_with_abondens_less_than_1percent(data_airline,name)
    [aft.append(data_airline[j].shape) for j in name]
    bef = np.reshape(bef,(len(bef),2))
    aft = np.reshape(aft,(len(aft),2))
    b = pd.DataFrame([[bef[i][0], aft[i][0],(bef[i][0]-aft[i][0])*100/ bef[i][0]] for i in range(len(name))], 
                       columns = ['before', 'after', 'missing data %'], 
                       index = name)
    return b

#DISTANCE_AB(airlines, name_of_airline)
compare_airline_data_before_and_after_remove_DISTANCE_AB_data_based_on_abondens_less_than_1percent(airlines,name_of_airline)
#data.columns
data_new = data[['AIRLINE', 
'airport_A', 'airport_B', 'airport_C', 
'turnaround_time_ B', 
'DEPARTURE_TIME_AB', 'DEPARTURE_DELAY_AB', 
'ARRIVAL_TIME_AB', 'ARRIVAL_DELAY_AB', 
'SCHEDULED_DEPARTURE_BC',
'ELAPSED_TIME_AB', 'DISTANCE_AB', 'DISTANCEBC']]
#data_new

## Choose The Most Important Features!
data_new['DEPARTURE_HOUR_AB'] = pd.to_datetime(data_new["DEPARTURE_TIME_AB"]).dt.hour
data_new['DEPARTURE_weekday_AB'] = pd.to_datetime(data_new["DEPARTURE_TIME_AB"]).dt.weekday
data_new['DEPARTURE_day_AB'] = pd.to_datetime(data_new["DEPARTURE_TIME_AB"]).dt.day
data_new['DEPARTURE_month_AB'] = pd.to_datetime(data_new["DEPARTURE_TIME_AB"]).dt.month
data_new.drop(['DEPARTURE_TIME_AB'], axis=1, inplace= True)

data_new['ARRIVAL_HOUR_AB'] = pd.to_datetime(data_new["ARRIVAL_TIME_AB"]).dt.hour
data_new['ARRIVAL_weekday_AB'] = pd.to_datetime(data_new["ARRIVAL_TIME_AB"]).dt.weekday
data_new['ARRIVAL_day_AB'] = pd.to_datetime(data_new["ARRIVAL_TIME_AB"]).dt.day
data_new['ARRIVAL_month_AB'] = pd.to_datetime(data_new["ARRIVAL_TIME_AB"]).dt.month
data_new.drop(['ARRIVAL_TIME_AB'], axis=1, inplace= True)

data_new['SCHEDULED_DEPARTURE_HOUR_BC'] = pd.to_datetime(data_new["SCHEDULED_DEPARTURE_BC"]).dt.hour
data_new['SCHEDULED_DEPARTURE_weekday_BC'] = pd.to_datetime(data_new["SCHEDULED_DEPARTURE_BC"]).dt.weekday
data_new['SCHEDULED_DEPARTURE_day_BC'] = pd.to_datetime(data_new["SCHEDULED_DEPARTURE_BC"]).dt.day
data_new['SCHEDULED_DEPARTURE_month_BC'] = pd.to_datetime(data_new["SCHEDULED_DEPARTURE_BC"]).dt.month
data_new.drop(['SCHEDULED_DEPARTURE_BC'], axis=1, inplace= True)

#data_new
data['AIRLINE'].unique()
data_new.to_csv('flights_step_5.csv')

#Seperate data based on the airlines in new dataFrame
def seperate_data_based_on_airline(airline, data1):
    a = data1[data1['AIRLINE'] == airline]
    b = int(len(a)/len(data1)*100)
    a.to_csv('codes/airlines_data/data_flights_of_{}_airlines_%{}.csv'.format(airline, b))
    #print('shape of data_airline {} = {}'.format(airline, a.shape))
    return a
#*****************************************************
#name of airlines
name_of_airline = np.unique(data_new.AIRLINE)

# Seperate data based on the airlines in new dataFrame
airlines = {}
for i in name_of_airline:
     airlines[i] = pd.DataFrame(seperate_data_based_on_airline(i, data_new))

def histogram(data_airline,name):
    length = int(len(name)/2)+int(len(name)%2)
    l, k, m = 0, 0, 0
    fig, axs = plt.subplots(length, 2, figsize = (12, 35))
    #ax2 = axs.twinx()
    for j in name:
        k = (m)//(length)
        l = m%(length)
        target = data_airline[j]
        axs[l][k].hist(target['turnaround_time_ B'],bins = 100, alpha = 0.5, label = '{}'.format(j))
        axs[l][k].legend()
        m += 1

#histogram(airlines, name_of_airline)

def histogram_3_most_airlines(data_airline,name):
    length = 3
    l, k, m = 0, 0, 0
    fig, axs = plt.subplots(1,length, figsize = (20, 5))
    plt.ylabel('jjjj')
    plt.legend()
    for j in name:
        target = data_airline[j]
        axs[m].hist(target['turnaround_time_ B'],bins = 100, alpha = 0.5, label = '{}'.format(j))
    
        axs[m].legend()
        m += 1
    
max_min_hist = np.zeros(2*3)
bin = 50
plt.rcParams.update({'font.size': 20})
fig, axs = plt.subplots(1, 3, figsize = (15,5))

target = airlines['WN']
s = axs[0].hist(target['turnaround_time_ B'],bins = bin, alpha = 0.5,  label = 'WN Airlines')
max_min_hist[0] = max(s[0])
max_min_hist[1] = min(s[0])

target = airlines['AA']
s = axs[1].hist(target['turnaround_time_ B'],bins = bin, alpha = 0.5, label = 'AA Airlines')
max_min_hist[2] = max(s[0])
max_min_hist[3] = min(s[0])

target = airlines['OO']
s = axs[2].hist(target['turnaround_time_ B'],bins = bin, alpha = 0.5, label = 'OO Airlines')
max_min_hist[4] = max(s[0])
max_min_hist[5] = min(s[0])

a1, a2 = min(max_min_hist), (max(max_min_hist)+10000) 
y_ticks = np.arange(a1,a2,(a1+a2)/10 )
l = len(str((a2)/10))-3
y_lable = np.arange(int(a1/10**l), int(a2/10**l), int((a1+a2)/10**(l+1)) )

for ax in axs.flat:
    ax.set(xlabel='Turnaround time (h)')
    ax.set_xlim([min(target['turnaround_time_ B']),max(target['turnaround_time_ B'])])
    ax.set_ylim([min(max_min_hist), max(max_min_hist)+10000])   
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_lable)
    ax.legend()
axs[0].set(ylabel = 'Histogram $x 10^{}$'.format(l))
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.savefig('first_3_histogram.png')




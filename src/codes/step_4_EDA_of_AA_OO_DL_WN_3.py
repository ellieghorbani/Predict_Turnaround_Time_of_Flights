#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime, date, time 
from scipy import stats

# # Read Data!

data = pd.read_csv("flights_step_3.csv")
#print('shape of data:', data.shape)
data.head(4)

# # EDA

# # Ariport_B

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
        ax2 = axs[l].twinx()
        a = data_airline[j]
        min_dep, max_dep = min(a['DEPARTURE_DELAY_AB']), max(a['DEPARTURE_DELAY_AB'])
        for i in np.arange(min_dep, max_dep, 10):
            interval = a[i < a['DEPARTURE_DELAY_AB']]
            interval = interval[interval['DEPARTURE_DELAY_AB']> i+10]
            axs[l].scatter(np.mean(interval['DEPARTURE_DELAY_AB']),
                              np.mean(interval['turnaround_time_ B']),
                              marker = 'o', color = 'blue')
            ax2.scatter(np.mean(interval['DEPARTURE_DELAY_AB']),
                              np.log(len(interval['DEPARTURE_DELAY_AB'])),
                              marker = 'o', color = 'red')
        axs[l].scatter(0,0,color ='white',label = j)
        axs[l].set_xlabel('DEPARTURE_DELAY_AB')
        axs[l].set_ylabel('turnaround_time_ B', color = 'blue')
        ax2.set_ylabel('log(Number of Departure Delay AB)', color = 'red')
        axs[l].legend()
        m += 1

# data and average data of TAT via arrival time, day of week, day of month and month
def plot_TAT_via_date(data_airline, name_airline, feature, type_date):
    c = ['blue','lightblue', 'red', 'pink']
    cc = 0
    fontdict={'fontsize': 20,
              'verticalalignment': 'baseline',
              'horizontalalignment': 'center'}
    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(1, 4, figsize = (24, 6), sharex=True)
    l = -1     
    #plt.rc('xtick', labelsize=20) 
    #plt.rc('ytick', labelsize=20) 
    max_1, max_2, min_1, min_2 = 0, 500,0, 500
    #-------------------------------------------------------
    for ax in axs.flat:
            l += 1
            j = name_airline[l]
            ax2 = ax.twinx()
            if ax != axs[3]:
                ax2.yaxis.set_tick_params(labelright=False)
            ax2.set_ylim([-.5, 10]) 
            ax.tick_params(axis="y", labelcolor="b", labelsize=20)
            ax.spines['right'].set_color('red')
            ax.spines['left'].set_color('blue')
            ax2.tick_params(axis="y", labelcolor="r", labelsize=20)
            ax.tick_params(axis="x", labelcolor="black", labelsize=20)
            ax2.spines['right'].set_color('red')
            ax2.spines['left'].set_color('blue')
            #---------------------------------------------------------------------
            a = data_airline[data_airline['AIRLINE'] == j]
            if type_date == 'Day time (h)':
                list_date = 24
                b = pd.to_datetime(a[feature]).dt.hour
            else:
                if type_date == 'Week day':
                    list_date = 7
                    b = pd.to_datetime(a[feature]).dt.weekday
                else:
                    if type_date == 'Month day':
                        list_date = 31
                        b = pd.to_datetime(a[feature]).dt.day
                    else:
                        list_date = 12
                        b = pd.to_datetime(a[feature]).dt.month
            #-------------------------------------------------------------------------          
            for i in range(list_date):
                interval = a[b == i]
                x = [i,i]
                if len(interval['turnaround_time_ B']) != 0:
                    y = [min(interval['turnaround_time_ B']), max(interval['turnaround_time_ B'])]
                    ax.plot(x, y, color = c[cc+1])
                    ax.plot(i,np.mean(interval['turnaround_time_ B']),
                              marker = 'o', markersize = 5, color = c[cc])
                    ax2.plot(i,len(interval['turnaround_time_ B'])/10**4,
                              marker = 'o', markersize = 5, color = c[cc+2])
                if len(interval['turnaround_time_ B']) != 0:
                    max_1, min_1 = max(max_1, max(interval['turnaround_time_ B'])), min(min_1, min(interval['turnaround_time_ B']))
                    #max_2, min_2 = max(max_2, len(interval['turnaround_time_ B'])), min(min_2, len(interval['turnaround_time_ B']))
                ax.set_title('{} Airlines'.format(j), fontdict=fontdict, color="black")   
                ax.set(xlabel= type_date)
                x_ticks = np.arange(0,list_date,4 )
                ax.set_xticks(x_ticks)
                
           
    ax2.set_ylabel('Number of flights $x 10^{4}$', color = 'red', fontsize=20)
    axs[0].set_ylabel('Turnaround time (h)', color = 'blue')
    axs[1].axes.yaxis.set_ticklabels([])
    axs[2].axes.yaxis.set_ticklabels([])
    axs[3].axes.yaxis.set_ticklabels([])
    #y_ticks = np.arange(min_2,max_2,(min_2+max_2)/10 )
    for ax in axs.flat:
        ax.set_ylim([min_1-1, max_1+1]) 
        y_ticks = np.arange(int(min_1-1), int(max_1+3),(int(min_1-1)+int(max_1+2))//3)
        ax.set_yticks(y_ticks)
        #ax2 = ax.twinx()
        #ax2.set_yticks(y_ticks)
    #ax.set_yticklabels(y_lable)
    #ax.legend()
    
    cc += 1

        
    plt.savefig('EDA_figs/EDA_{}_{}_4_airlines.png'.format(feature, type_date), dpi=500)   
# data and average data of TAT via arrival time, day of week, day of month and month
def plot_TAT_via_delay_data(data_airline, name_airline, feature, type_date):
    c = ['blue','lightblue', 'red', 'pink']
    cc = 0
    fontdict={'fontsize': 20,
              'verticalalignment': 'baseline',
              'horizontalalignment': 'center'}
    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(1, 4, figsize = (24, 6), sharex=True)
    l = -1     
    #plt.rc('xtick', labelsize=20) 
    #plt.rc('ytick', labelsize=20) 
    max_1, max_2, min_1, min_2 = 0, 500,0, 500
    #-------------------------------------------------------
    for ax in axs.flat:
            l += 1
            j = name_airline[l]
            ax2 = ax.twinx()
            if ax != axs[3]:
                ax2.yaxis.set_tick_params(labelright=False)
            ax2.set_ylim([-.5, 10]) 
            ax.tick_params(axis="y", labelcolor="b", labelsize=20)
            ax.spines['right'].set_color('red')
            ax.spines['left'].set_color('blue')
            ax2.tick_params(axis="y", labelcolor="r", labelsize=20)
            ax.tick_params(axis="x", labelcolor="black", labelsize=20)
            ax2.spines['right'].set_color('red')
            ax2.spines['left'].set_color('blue')
            #---------------------------------------------------------------------
            a = data_airline[data_airline['AIRLINE'] == j]
            list_data = 1400
            b = a[feature]
            #-------------------------------------------------------------------------          
            for i in range(-100,list_data,10):
                interval1 = a[b >= i]
                interval = interval1[b < i+10]
                x = [i,i]
                if len(interval['turnaround_time_ B']) != 0:
                    y = [min(interval['turnaround_time_ B']), max(interval['turnaround_time_ B'])]
                    ax.plot(x, y, color = c[cc+1])
                    ax.plot(i,np.mean(interval['turnaround_time_ B']),
                              marker = 'o', markersize = 5, color = c[cc])
                    ax2.plot(i,len(interval['turnaround_time_ B'])/10**4,
                              marker = 'o', markersize = 5, color = c[cc+2])
                if len(interval['turnaround_time_ B']) != 0:
                    max_1, min_1 = max(max_1, max(interval['turnaround_time_ B'])), min(min_1, min(interval['turnaround_time_ B']))
                    #max_2, min_2 = max(max_2, len(interval['turnaround_time_ B'])), min(min_2, len(interval['turnaround_time_ B']))
                ax.set_title('{} Airlines'.format(j), fontdict=fontdict, color="black")   
                ax.set(xlabel= type_date)
                x_ticks = np.arange(-100,list_data,240 )
                x_label = np.arange(-100//60,list_data//60,240//60 )
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_label, rotation=0, ha='right')
                
           
    ax2.set_ylabel('Number of flights $x 10^{4}$', color = 'red', fontsize=20)
    axs[0].set_ylabel('Turnaround time (h)', color = 'blue')
    axs[1].axes.yaxis.set_ticklabels([])
    axs[2].axes.yaxis.set_ticklabels([])
    axs[3].axes.yaxis.set_ticklabels([])
    #y_ticks = np.arange(min_2,max_2,(min_2+max_2)/10 )
    for ax in axs.flat:
        ax.set_ylim([min_1-1, max_1+1]) 
        y_ticks = np.arange(int(min_1-1), int(max_1+3),(int(min_1-1)+int(max_1+2))//3)
        ax.set_yticks(y_ticks)
        #ax2 = ax.twinx()
        #ax2.set_yticks(y_ticks)
    #ax.set_yticklabels(y_lable)
    #ax.legend()
    
    cc += 1

        
    plt.savefig('EDA_figs/EDA_{}_{}_4_airlines.png'.format(feature, type_date), dpi=500)   

compare = pd.DataFrame([format(np.unique(data['AIRLINE']).shape),
                        format(np.unique(data["airport_A"]).shape),
                        format(np.unique(data["airport_B"]).shape),
                        format(np.unique(data["airport_C"]).shape)],
                       columns = ['size'], index = ['AIRLINE',"airport_A", "airport_B", "airport_C"])
compare

#name of airlines
name_of_airline = ['OO', 'AA', 'DL', 'WN']
# Seperate data based on the airlines in new dataFrame
airlines = {}
for i in name_of_airline:
     airlines[i] = pd.DataFrame(seperate_data_based_on_airline(i, data))

df = compare_number_flights_based_on_air_portB(airlines, name_of_airline)
remove_airport_B_with_less_than_25_precent_flights_from_average_number_flights(airlines, name_of_airline)
df_new = compare_number_flights_based_on_air_portB(airlines, name_of_airline)    
df

df_new

# List of EDA:<br/>
# 1- airport (A, B, C)<br/>
# 2- SCHEDULED_DEPARTURE (time, day of week, week, day of month, month)(AB,BC)<br/>
# 3- SCHEDULED_ARRIVAL (time, day of week, week, day of month, month)(AB)<br/>
# 4- DEPARTURE_DELAY (AB, BC)<br/>
# 5- ARRIVAL_DELAY (AB)<br/>
# 6- ELAPSED_TIME (AB, BD)<br/>
# 7- DISTANCE (AB, BC)<br/>
# <br/>
# <br/>
# distribution1:<br/>
# ARRIVAL_HOUR_AB, SCHEDULED_DEPARTURE_HOUR_BC,<br/>
# APPRIVAL_DELAY_AB <br/> 
# 
# distribution2 <br/>
# SCHEDULED_DEPARTURE_HOUR_BC, DEPARTURE_HOUR_AB, <br/>
# APPRIVAL_DELAY_AB 

# ### 2- SCHEDULED_DEPARTURE

# In[ ]:


#print('ARRIVAL_TIME_AB', 'day_time (h)')
plot_TAT_via_date(data, name_of_airline, 'ARRIVAL_TIME_AB', 'Day time (h)')
#print('SCHEDULED_DEPARTURE_BC', 'week_day')
plot_TAT_via_date(data, name_of_airline, 'SCHEDULED_DEPARTURE_BC', 'Day time (h)')
#print('DEPARTURE_TIME_AB', 'month_day')
plot_TAT_via_date(data, name_of_airline, 'DEPARTURE_TIME_AB', 'Day time (h)')

plot_TAT_via_delay_data(data, name_of_airline, 'ARRIVAL_DELAY_AB', 'delay time x $60$ (min)')

# # ARRIVAL_DELAY_AB

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

arrival_delay_AB(airlines, name_of_airline)

compare_airline_data_before_and_after_remove_ARRIVAL_DELAY_AB_data_based_on_abondens_less_than_1percent(airlines,name_of_airline)

# # DEPARTURE_DELAY_BC


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

departure_delay_BC(airlines, name_of_airline)

compare_airline_data_before_and_after_remove_DEPARTURE_DELAY_BC_data_based_on_abondens_less_than_1percent(airlines,name_of_airline)

import datetime
data.columns
pd.to_datetime(data['SCHEDULED_DEPARTURE_AB']).dt.time


# # ELAPSED_TIME_AB

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

elapsed_time_AB(airlines, name_of_airline)

compare_airline_data_before_and_after_remove_ELAPSED_TIME_AB_data_based_on_abondens_less_than_1percent(airlines,name_of_airline)


# # ELAPSED_TIME_BC

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

elapsed_time_BC(airlines, name_of_airline)

compare_airline_data_before_and_after_remove_ELAPSED_TIME_BC_data_based_on_abondens_less_than_1percent(airlines,name_of_airline)


# # DISTANCE_AB

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

DISTANCE_AB(airlines, name_of_airline)

compare_airline_data_before_and_after_remove_DISTANCE_AB_data_based_on_abondens_less_than_1percent(airlines,name_of_airline)


data.columns

data_new = data[['AIRLINE', 
'airport_A', 'airport_B', 'airport_C', 
'turnaround_time_ B', 
'DEPARTURE_TIME_AB', 'DEPARTURE_DELAY_AB', 
'ARRIVAL_TIME_AB', 'ARRIVAL_DELAY_AB', 
'SCHEDULED_DEPARTURE_BC',
'ELAPSED_TIME_AB', 'DISTANCE_AB', 'DISTANCEBC']]
data_new

# # Choose The Most Important Features!

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

data_new

data['AIRLINE'].unique()


data_new.to_csv('flights_step_5.csv')

#Seperate data based on the airlines in new dataFrame
def seperate_data_based_on_airline(airline, data1):
    a = data1[data1['AIRLINE'] == airline]
    b = int(len(a)/len(data1)*100)
    a.to_csv('airlines_data/data_flights_of_{}_airlines_%{}.csv'.format(airline, b))
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

histogram(airlines, name_of_airline)

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


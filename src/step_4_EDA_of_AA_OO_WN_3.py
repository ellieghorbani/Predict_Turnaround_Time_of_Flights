#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime, date, time 
from scipy import stats

## Read Data!
data = pd.read_csv("flights_step_3.csv")
#print('shape of data:', data.shape)
data.head(4)

## EDA
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
    fig, axs = plt.subplots(1, 3, figsize = (15, 5), sharex=True)
    l = -1     
    #plt.rc('xtick', labelsize=20) 
    #plt.rc('ytick', labelsize=20) 
    max_1, max_2, min_1, min_2 = 0, 500,0, 500
    #-------------------------------------------------------
    for ax in axs.flat:
            l += 1
            j = name_airline[l]
            ax2 = ax.twinx()
            if ax != axs[2]:
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
                        list_date = 13
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

        
    plt.savefig('EDA_figs/EDA_{}_{}.png'.format(feature, type_date), dpi=500)   
compare = pd.DataFrame([format(np.unique(data['AIRLINE']).shape),
                        format(np.unique(data["airport_A"]).shape),
                        format(np.unique(data["airport_B"]).shape),
                        format(np.unique(data["airport_C"]).shape)],
                       columns = ['size'], index = ['AIRLINE',"airport_A", "airport_B", "airport_C"])
compare


#name of airlines
name_of_airline = ['OO', 'AA', 'WN']
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
# Based on important features:<br/>
# distribution1:<br/>
# ARRIVAL_HOUR_AB, SCHEDULED_DEPARTURE_HOUR_BC,<br/>
# APPRIVAL_DELAY_AB <br/> 
# 
# distribution2 <br/>
# SCHEDULED_DEPARTURE_HOUR_BC, DEPARTURE_HOUR_AB, <br/>
# APPRIVAL_DELAY_AB 

# ## Arrival and Departure date

#print('ARRIVAL_TIME_AB', 'day_time (h)')
plot_TAT_via_date(data, name_of_airline, 'ARRIVAL_TIME_AB', 'Day time (h)')
#print('SCHEDULED_DEPARTURE_BC', 'week_day')
plot_TAT_via_date(data, name_of_airline, 'SCHEDULED_DEPARTURE_BC', 'Day time (h)')
#print('DEPARTURE_TIME_AB', 'month_day')
plot_TAT_via_date(data, name_of_airline, 'DEPARTURE_TIME_AB', 'Day time (h)')

#print('DEPARTURE_TIME_BC', 'day_time (h)')
plot_TAT_via_date(data, name_of_airline, 'DEPARTURE_TIME_BC', 'day_time (h)')
#print('DEPARTURE_TIME_BC', 'week_day')
plot_TAT_via_date(data, name_of_airline, 'DEPARTURE_TIME_BC', 'week_day')
#print('DEPARTURE_TIME_BC', 'month_day')
plot_TAT_via_date(data, name_of_airline, 'DEPARTURE_TIME_BC', 'month_day')
#print('DEPARTURE_TIME_BC', 'month')
plot_TAT_via_date(data, name_of_airline, 'DEPARTURE_TIME_BC', 'month')



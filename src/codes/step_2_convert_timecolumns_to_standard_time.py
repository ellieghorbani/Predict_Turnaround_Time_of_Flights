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

## Read data from flights_step_1.csv!
flights_step_2 = pd.read_csv('flights_step_1.csv')
#print("flights.shape",flights_step_2.shape)

## Check flights dataFrame!
#print("flights dataFrame:")
#print(flights_step_2.loc[:13555, ['SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL', 'DEPARTURE_TIME',
#             'ARRIVAL_TIME', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY']])

## #1 Check to two samples in the dataFrame!
#print("#1 Check to two samples in the dataFrame!")
#print('SCHEDULED_DEPARTURE:%f | %f' %(flights_step_2.SCHEDULED_DEPARTURE[0], flights_step_2.SCHEDULED_DEPARTURE[30000]))
#print('     DEPARTURE_TIME:%f | %f' %(flights_step_2.DEPARTURE_TIME[0], flights_step_2.DEPARTURE_TIME[30000]))
#print('  SCHEDULED_ARRIVAL:%f | %f' %(flights_step_2.SCHEDULED_ARRIVAL[0], flights_step_2.SCHEDULED_ARRIVAL[30000]))
#print('       ARRIVAL_TIME:%f | %f' %(flights_step_2.ARRIVAL_TIME[0], flights_step_2.ARRIVAL_TIME[30000]))


## Convert time data to seconds!
def convert_to_sec(a):
    return (a - a%100)*36 + a%100*60
#------------------------------------

flights_step_2['SCHEDULED_DEPARTURE_DATE'] = pd.to_datetime(flights_step_2[['YEAR','MONTH', 'DAY']])
flights_step_2['FSD'] = convert_to_sec(flights_step_2.SCHEDULED_DEPARTURE)
flights_step_2['FSA'] = convert_to_sec(flights_step_2.SCHEDULED_ARRIVAL)
flights_step_2['FD'] = convert_to_sec(flights_step_2.DEPARTURE_TIME)
flights_step_2['FA'] = convert_to_sec(flights_step_2.ARRIVAL_TIME)
 

## Convert times to standard datetimes : SCHEDULED_DEPARTURE, DEPARTURE_TIME, SCHEDULED_ARRIVAL, ARRIVAL_TIME!
def determinante_day(tsd,td,delay_time, date):
    y = td + ((-(td - tsd) + delay_time*60)//(24*3600))* 24*3600
    return pd.to_datetime(date) + y.astype('timedelta64[s]')
#----------------------------------------------------------------
flights_step_2['SCHEDULED_DEPARTURE'] =  pd.to_datetime(flights_step_2.SCHEDULED_DEPARTURE_DATE) + flights_step_2.FSD.astype('timedelta64[s]')
flights_step_2['DEPARTURE_TIME'] = determinante_day(flights_step_2['FSD'],flights_step_2['FD'],flights_step_2['DEPARTURE_DELAY'],flights_step_2.SCHEDULED_DEPARTURE_DATE)
flights_step_2['SCHEDULED_ARRIVAL'] = determinante_day(flights_step_2['FSD'],flights_step_2['FSA'],24*60,flights_step_2['SCHEDULED_DEPARTURE_DATE'])
flights_step_2['ARRIVAL_TIME'] = determinante_day(flights_step_2['FSD'],flights_step_2['FA'],24*60,flights_step_2['SCHEDULED_DEPARTURE_DATE'])

##print(flights_step_2.loc[:13900, ['SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL', 'DEPARTURE_TIME',
#             'ARRIVAL_TIME', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY']])

## Convert #1 to a standard timedate! 
#print("After converting #1 to a standard timedata!")
#print('SCHEDULED_DEPARTURE:', flights_step_2.SCHEDULED_DEPARTURE[0], flights_step_2.SCHEDULED_DEPARTURE[30000])
#print('     DEPARTURE_TIME:', flights_step_2.DEPARTURE_TIME[0], flights_step_2.DEPARTURE_TIME[30000])
#print('  SCHEDULED_ARRIVAL:', flights_step_2.SCHEDULED_ARRIVAL[0], flights_step_2.SCHEDULED_ARRIVAL[30000])
#print('       ARRIVAL_TIME:', flights_step_2.ARRIVAL_TIME[0], flights_step_2.ARRIVAL_TIME[30000])

flights_step_2 = flights_step_2[['AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER',
       'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE',
       'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'SCHEDULED_TIME', 'ELAPSED_TIME',
       'DISTANCE', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY']]

#print("flight.shape:",flights_step_2.shape)


## Save Data!
flights_step_2.to_csv('flights_step_2.csv')





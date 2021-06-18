#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date, time 
from scipy import stats
from sklearn.model_selection import train_test_split

## Read Data!
data = pd.read_csv("../airlines_data/data_flights_of_DL_airlines_%15.csv")
#print('shape of data:', data.shape)
data.head(4)
len_unseen = int(len(data)*.1)
len_seen = int(len(data)*.9)
data_unseen = data.sample(len_unseen)
data_seen = data.drop(len_seen)
data_seen.to_csv('data_seen_DL_airlines.csv')
data_unseen.to_csv('data_unseen_DL_airlines.csv')



#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date, time 
from scipy import stats
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor 
#from tabulate import tabulate
## read Data!

data = pd.read_csv("data_seen_AA_airlines.csv")
#print('shape of data:', data.shape)

## Modeling
#data.info()
#data['AIRLINE'].unique()

target = data['turnaround_time_ B']
features = pd.get_dummies(data[['airport_A', 'airport_B', 'airport_C',
       'DEPARTURE_DELAY_AB', 'ARRIVAL_DELAY_AB',
       'ELAPSED_TIME_AB', 'DISTANCE_AB', 'DISTANCEBC', 'DEPARTURE_HOUR_AB',
       'DEPARTURE_weekday_AB', 'DEPARTURE_day_AB', 'DEPARTURE_month_AB',
       'ARRIVAL_HOUR_AB', 'ARRIVAL_weekday_AB', 'ARRIVAL_day_AB',
       'ARRIVAL_month_AB', 'SCHEDULED_DEPARTURE_HOUR_BC',
       'SCHEDULED_DEPARTURE_weekday_BC', 'SCHEDULED_DEPARTURE_day_BC',
       'SCHEDULED_DEPARTURE_month_BC']])
#print(target.shape)


plt.hist(target,bins = 100, alpha = 0.5, label = 'target')
#print(np.mean(target))
target_1, target_2 = target[target<= 4], target[target> 4]
index_1 = target_1.index
index_2 = target_2.index
features_1, features_2 = features[features.index.isin(index_1)], features[features.index.isin(index_2)]
#print(features_1.shape)
#print(features_2.shape)
del(data)
models = {0 : linear_model.LinearRegression(),  
          1 : GradientBoostingRegressor(n_estimators = 500, max_depth = 10, min_samples_split = 5, learning_rate = 0.1, loss= 'ls'),
          2 : RandomForestRegressor(n_estimators = 500, max_depth = 10, random_state = 0) }
model_name = {0 : "Linear Regression",
              1 : "Gradient Boosting Regressor",
              2 : "Random Forest Regressor"}

## Fit_model Function
def fit_model(model_type, name_model, features_data, target_data, rand_state,name_target):
    X_train, X_test, y_train, y_test = train_test_split(features_data, target_data,
                                                    random_state=rand_state, test_size=0.2)
    model = model_type
    model.fit(X_train, y_train)
    y_test_predict = model.predict(X_test)
    data_frame = pd.DataFrame([[name_target, name_model, np.sqrt(sum((y_test - y_test_predict)**2)/len(y_test)),
                              np.sqrt(sum((y_test - np.mean(y_test))**2)/len(y_test)),
                              np.mean(y_train), np.mean(y_test), np.mean(y_test_predict)]], 
                              columns = ['name_target','model','RMS_test_predict','RMS_test_mean',
                                        'mean_train', 'mean_test', 'mean_test_predict'])
    
    # Plot the feature importance
    plt.figure(figsize=(50, 50))
    if name_model == "Linear Regression":
        all_features = model.coef_
    else:
        all_features = model.feature_importances_
    feat_scores_model = pd.DataFrame({'Fraction of Features' : all_features}, index=X_train.columns)
    feat_scores_model = feat_scores_model.sort_values(by='Fraction of Features')
    feat_scores_model = feat_scores_model[-10:]
    plt.figure(figsize=(10, 50))
    feat_scores_model.plot(kind='barh')
    plt.legend(loc='lower right')
    plt.savefig("the_most_important_AA_{}_predict_{}.png".format(name_model, name_target), bbox_inches='tight')
    return y_test, y_test_predict, data_frame, feat_scores_model

def plot(name_model,y_test, y_test_predict,name_target):
    
    fig, ax = plt.subplots(figsize = (10,5))
    ax.hist(y_test,bins = 100, alpha = 0.5, label = (name_target+'_test'))
    ax.hist(y_test_predict, bins = 100,alpha = 0.5, label = name_target+'_test_predict')
    ax.legend()
    plt.savefig("histogram_AA_{}_predict_{}.png".format(name_model, name_target), bbox_inches='tight')
    

## Linear Regression
target_1_test, target_1_test_predict, table_reg_1, feat_scores_reg_1 = fit_model(models[0],model_name[0], features_1,target_1, 347,'target_1')
plot(model_name[0],target_1_test, target_1_test_predict, 'target_1')

target_2_test, target_2_test_predict, table_reg_2, feat_scores_reg_2 = fit_model(models[0], model_name[0], features_2,target_2, 964,'target_2')
plot(model_name[0],target_2_test, target_2_test_predict, 'target_2')
pd.concat([table_reg_1, table_reg_2], axis=0)

## Gradient Boosting Regressor
target_1_test, target_1_test_predict, table_gbr_1, feat_scores_gbr_1 = fit_model(models[1],model_name[1], features_1,target_1, 347,'target_1')
plot(model_name[1],target_1_test, target_1_test_predict,'target_1')

#print('*')
target_2_test, target_2_test_predict, table_gbr_2, feat_scores_gbr_2 = fit_model(models[1],model_name[1], features_2,target_2, 1735,'target_2')
plot(model_name[1],target_2_test, target_2_test_predict,'target_2')
pd.concat([table_gbr_1, table_gbr_2], axis=0)

## Random Forest Regressor 
target_1_test, target_1_test_predict, table_rfr_1, feat_scores_rfr_1 = fit_model(models[2],model_name[2], features_1, target_1, 347,'target_1')
plot(model_name[2],target_1_test, target_1_test_predict,'target_1')
#print('*')
target_2_test, target_2_test_predict, table_rfr_2, feat_scores_rfr_2 = fit_model(models[2],model_name[2], features_2,target_2, 347,'target_2')
plot(model_name[2], target_2_test, target_2_test_predict,'target_2')
pd.concat([table_rfr_1, table_rfr_2], axis=0)




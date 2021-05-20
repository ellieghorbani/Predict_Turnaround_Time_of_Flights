#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date, time 
from scipy import stats
from sklearn.model_selection import train_test_split

## Random Forest Classifier model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def RandomForestClassifier_model(features_data, target_data, rand_state):
    
    X_train, X_test, y_train, y_test = train_test_split(features_data, target_data, 
                                                        random_state=rand_state, test_size=0.2)

    model = RandomForestClassifier(n_estimators= 200, max_depth = 30, random_state=100)
    model.fit(X_train, y_train)
    y_test_predict = model.predict(X_test)
    #-------------------------------------------------------
    cm1 = confusion_matrix(y_test, y_test_predict)
    fig, ax = plt.subplots(figsize=(4, 4))
    
    ax.imshow(cm1)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.set_ylim(1.5, -0.5)
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    plt.sca(ax)
    #plt.title('Predict Test Data')
        
    #print(cm1.sum(axis=0)[0])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, ('% {}').format(int(cm1[i, j]/cm1.sum(axis= 1)[i]*100)), ha='center', va='center', color='red')
    plt.show()
    plt.savefig('confusion_matrix_AA_RandomForestClassifier.png')
    #print("Precision_0 ={:.2f}".format((cm1[0,0]/(cm1[0,0] + cm1[1,0]))))
    #print("Recall_0 =%f" %(cm1[0,0]/(cm1[0,0]+cm1[0,1])))
    #print("Precision_1 ={:.2f}".format((cm1[1,1]/(cm1[1,1] + cm1[0,1]))))
    #print("Recall_1 =%f" %(cm1[1,1]/(cm1[1,1]+cm1[1,0])))
    
    return model

## Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score

def gbr_model(features_data, target_data, num_sample, rand_state, name_target):
    
    X_train, X_test, y_train, y_test = train_test_split(features_data, target_data,
                                                    random_state=rand_state, test_size=0.2)
    gbr = GradientBoostingRegressor(n_estimators = 500, max_depth = 10, min_samples_split = 5,
          learning_rate = 0.1, loss= 'ls')
    gbr.fit(X_train, y_train)
    y_test_predict = gbr.predict(X_test)
    
    data_frame = pd.DataFrame([[name_target, 'Gradient Boosting Regressor', np.sqrt(sum((y_test - y_test_predict)**2)/len(y_test)),
                              np.sqrt(sum((y_test - np.mean(y_test))**2)/len(y_test)),
                              np.mean(target_data),np.mean(target_data), np.mean(y_train), 
                              np.mean(y_test), np.mean(y_test_predict)]], 
                              columns = ['name_target','model','RMS_test_predict','RMS_test_mean',
                                        'mean_target','mean_sample', 'mean_train', 
                                        'mean_test', 'mean_test_predict'])
    # Plot the feature importance
    
    all_features = gbr.feature_importances_
    feat_scores_gbr = pd.DataFrame({'Fraction of Samples Affected {} {}'.format('Gradient Boosting', name_target) : all_features}, index=X_train.columns)
    feat_scores_gbr = feat_scores_gbr.sort_values(by='Fraction of Samples Affected {} {}'.format('Gradient Boosting', name_target) )
    feat_scores_gbr = feat_scores_gbr[-15:]
    plt.figure(figsize=(10, 50))
    feat_scores_gbr.plot(kind='barh')
    plt.legend(loc='lower right')
    
    return y_test, y_test_predict, data_frame, feat_scores_gbr, gbr

## Read Data!
data_seen = pd.read_csv("data_seen_WN_airlines.csv")
#print('shape of data_seen:', data_seen.shape)
data_unseen = pd.read_csv("data_unseen_WN_airlines.csv")
#print('shape of data_unseen:', data_unseen.shape)

target_seen = data_seen['turnaround_time_ B']
features_seen = pd.get_dummies(data_seen[['airport_A', 'airport_B', 'airport_C',
       'DEPARTURE_DELAY_AB', 'ARRIVAL_DELAY_AB',
       'ELAPSED_TIME_AB', 'DISTANCE_AB', 'DISTANCEBC', 'DEPARTURE_HOUR_AB',
       'DEPARTURE_weekday_AB', 'DEPARTURE_day_AB', 'DEPARTURE_month_AB',
       'ARRIVAL_HOUR_AB', 'ARRIVAL_weekday_AB', 'ARRIVAL_day_AB',
       'ARRIVAL_month_AB', 'SCHEDULED_DEPARTURE_HOUR_BC',
       'SCHEDULED_DEPARTURE_weekday_BC', 'SCHEDULED_DEPARTURE_day_BC',
       'SCHEDULED_DEPARTURE_month_BC']])

features_seen.shape

target_unseen = data_unseen['turnaround_time_ B']
features_unseen = pd.get_dummies(data_unseen[['airport_A', 'airport_B', 'airport_C',
       'DEPARTURE_DELAY_AB', 'ARRIVAL_DELAY_AB',
       'ELAPSED_TIME_AB', 'DISTANCE_AB', 'DISTANCEBC', 'DEPARTURE_HOUR_AB',
       'DEPARTURE_weekday_AB', 'DEPARTURE_day_AB', 'DEPARTURE_month_AB',
       'ARRIVAL_HOUR_AB', 'ARRIVAL_weekday_AB', 'ARRIVAL_day_AB',
       'ARRIVAL_month_AB', 'SCHEDULED_DEPARTURE_HOUR_BC',
       'SCHEDULED_DEPARTURE_weekday_BC', 'SCHEDULED_DEPARTURE_day_BC',
       'SCHEDULED_DEPARTURE_month_BC']])

features_unseen.shape

#print('Diffirence number of seen and unseen features:',features_seen.shape[1]-features_unseen.shape[1])
for i in features_seen:
    if i not in features_unseen:
        features_unseen[i]= 0

## Make model from seen data!
target_seen_bio = target_seen.copy()
target_seen_bio[target_seen <= 4]= 1
target_seen_bio[target_seen  > 4]= 2
target_seen_bio.value_counts()

forest_model = RandomForestClassifier_model(features_seen, target_seen_bio, 1040)

target_1, target_2 = target_seen[target_seen<= 4], target_seen[target_seen> 4]
index_1 = target_1.index
index_2 = target_2.index
features_1, features_2 = features_seen.loc[index_1], features_seen.loc[index_2]
#print(features_1.shape)
#print(features_2.shape)

target_1_test, target_1_test_predict,table_gbr_1, feat_scores_gbr_1, model_1 = gbr_model(features_1,target_1, 100000, 1112, 'target_1')
#reg_plot(target_1,target_1_test, target_1_test_predict, target_1_predict,'target_1')
#print('*')
target_2_test, target_2_test_predict, table_gbr_2, feat_scores_gbr_2, model_2 = gbr_model(features_2,target_2, 100000, 1112, 'target_2')
#reg_plot(target_2,target_2_test, target_2_test_predict, target_2_predict,'target_2')
#pd.concat([table_gbr_1, table_gbr_2], axis=0)

## Unseen Data
target_unseen_predict_forest = forest_model.predict(features_unseen)
target_unseen_copy = target_unseen.copy()
target_unseen_copy[target_unseen <= 4] = 1
target_unseen_copy[target_unseen > 4] = 2
### plot category of distrbutions based on RandomForestClassifier_model 
cm = confusion_matrix(target_unseen_copy, target_unseen_predict_forest)
fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.set_ylim(1.5, -0.5)
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
plt.sca(ax)
plt.title('Predict Unseen Data')
    
#print(cm.sum(axis=0)[0])
for i in range(2):
        for j in range(2):
            ax.text(j, i, ('{}').format(int(cm[i, j])), ha='center', va='center', color='red')
#plt.show()

# seprate unseen data two distributions without forest model
index1 = target_unseen[target_unseen_copy == 1].index
index2 = target_unseen[target_unseen_copy == 2].index
#print( set(index1).issubset(features_unseen.index))
features_unseen_1 = features_unseen.loc[index1]
features_unseen_2 = features_unseen.loc[index2]
target_unseen_1 = target_unseen[index1]
target_unseen_2 = target_unseen[index2]
#print('mean(target_unseen_1)', np.mean(target_unseen_1), 'mean(target_unseen_2)', np.mean(target_unseen_2))
#----------------------------------------------------------------
#predict distribution of data by useing forest model
index1 = target_unseen[target_unseen_predict_forest == 1].index
index2 = target_unseen[target_unseen_predict_forest == 2].index
features_unseen_predict_forest_1 = features_unseen.loc[index1]
features_unseen_predict_forest_2 = features_unseen.loc[index2]
target_unseen_predict_forest_1 = target_unseen[index1]
target_unseen_predict_forest_2 = target_unseen[index2]
#print('mean(target_unseen_predict_forest_1)', np.mean(target_unseen_predict_forest_1), 'mean(target_unseen_predict_forest_2)', np.mean(target_unseen_predict_forest_2))

## make model to predict TAT

def check_models_with_unseen_data(model, name_target, y, x):
    
    y_predict = model.predict(x)
    data_frame = pd.DataFrame([[name_target, 'Gradient Boosting Regressor', np.sqrt(sum((y - y_predict)**2)/len(y)),
                              np.sqrt(sum((y - np.mean(y))**2)/len(y)),
                              np.mean(y), np.mean(y_predict)]], 
                              columns = ['name_target','model','RMS_predict','RMS_target_mean',
                                        'mean_target','mean_target_predict'])
    return data_frame

table_1 = check_models_with_unseen_data(model_1, "unseen_1", target_unseen_1, features_unseen_1)
table_1_p = check_models_with_unseen_data(model_1, "unseen_forest_predicted_1", target_unseen_predict_forest_1, features_unseen_predict_forest_1)
table_2 = check_models_with_unseen_data(model_2, "unseen_2", target_unseen_2, features_unseen_2)
table_2_p = check_models_with_unseen_data(model_2, "unseen_forest_predicted_2", target_unseen_predict_forest_2, features_unseen_predict_forest_2)

pd.concat([table_1, table_1_p, table_2, table_2_p], axis=0)

target_unseen_1_predict = model_1.predict(features_unseen_1)
target_unseen_2_predict = model_2.predict(features_unseen_2)

mean_target_unseen_predict = (sum(target_unseen_1_predict)+sum(target_unseen_2_predict))/(len(target_unseen))
mean_target_unseen = np.mean(target_unseen)
print("mean_target_unseen_predict_WN:", mean_target_unseen_predict)
print("mean_target_unseen_WN",mean_target_unseen)
#RMSE real data
RMSE_targer = np.sqrt(sum((target_unseen - mean_target_unseen)**2))/len(target_unseen)
print("RMSE_targer_WN:", RMSE_targer)
sum_1 = sum((target_unseen_1_predict - target_unseen_1)**2)
sum_2 = sum((target_unseen_2_predict - target_unseen_2)**2)
RMSE_targer_predict = np.sqrt(sum_1+sum_2)/len(target_unseen)
print("RMSE_targer_predict_WN:", RMSE_targer_predict)



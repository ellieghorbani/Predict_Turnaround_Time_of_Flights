#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date, time 
from scipy import stats
from sklearn.model_selection import train_test_split

## Read Data!
data = pd.read_csv("data_seen_WN_airlines.csv")
#print('shape of data:', data.shape)

target = data['turnaround_time_ B']
features = pd.get_dummies(data[['airport_A', 'airport_B', 'airport_C',
       'DEPARTURE_DELAY_AB', 'ARRIVAL_DELAY_AB',
       'ELAPSED_TIME_AB', 'DISTANCE_AB', 'DISTANCEBC', 'DEPARTURE_HOUR_AB',
       'DEPARTURE_weekday_AB', 'DEPARTURE_day_AB', 'DEPARTURE_month_AB',
       'ARRIVAL_HOUR_AB', 'ARRIVAL_weekday_AB', 'ARRIVAL_day_AB',
       'ARRIVAL_month_AB', 'SCHEDULED_DEPARTURE_HOUR_BC',
       'SCHEDULED_DEPARTURE_weekday_BC', 'SCHEDULED_DEPARTURE_day_BC',
       'SCHEDULED_DEPARTURE_month_BC']])

target = target.copy()
target[target <= 4]= 1
target[target  > 4]= 2
target.value_counts()

# ### Models:
# - Logistic Regression
# - k-Nearest Neighbors
# - Decision Trees
# - Support Vector Machine
# - Naive Bayes

# ### Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def logistic_model(features_data, target_data, rand_state):
    
    
    X_train, X_test, y_train, y_test = train_test_split(features_data, target_data, 
                                                        random_state=rand_state, test_size=0.2)

    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train, y_train)
    y_test_predict = model.predict(X_test)
    #-------------------------------------------------------
    cm1 = confusion_matrix(y_test, y_test_predict)
    fig, ax = plt.subplots(figsize=(5, 4))
    
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
    #plt.show()
    plt.savefig('confusion_matrix_WN_LogisticRegression.png')
    #print("Precision_0 ={:.2f}".format((cm1[0,0]/(cm1[0,0] + cm1[1,0]))))
    #print("Recall_0 =%f" %(cm1[0,0]/(cm1[0,0]+cm1[0,1])))
    #print("Precision_1 ={:.2f}".format((cm1[1,1]/(cm1[1,1] + cm1[0,1]))))
    #print("Recall_1 =%f" %(cm1[1,1]/(cm1[1,1]+cm1[1,0])))
    
    all_features = abs(model.coef_[0])
    feat_scores_model = pd.DataFrame({'Fraction of Features': all_features}, index=X_train.columns)
    feat_scores_model = feat_scores_model.sort_values(by='Fraction of Features')
    feat_scores_model = feat_scores_model[-15:]
    #print(feat_scores_model)
    plt.figure(figsize=(50, 50))
    feat_scores_model.plot(kind='barh')
    plt.legend(loc='lower right')
    plt.savefig("the_most_important_WN_LogisticRegression.png", bbox_inches='tight')
    return model

logistic_model(features, target, 1222)

# ### k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def knn_model(features_data, target_data, rand_state, number_neighbors):
    
    X_train, X_test, y_train, y_test = train_test_split(features_data, target_data, 
                                                        random_state=rand_state, test_size=0.2)
    error_rates = []
    for i in range(1,number_neighbors):
        print(i)
        model = KNeighborsClassifier(n_neighbors = i)
        model.fit(X_train, y_train)
        y_test_predict = model.predict(X_test)
        error_rates.append(np.mean(y_test_predict != y_test))
    #-------------------------------------------------------
    min_error = min(error_rates)
    max_knn = min(np.where(error_rates == min_error))
    plt.plot(error_rates)
    plt.xlabel('Number_neighbors')
    plt.ylabel('error_rates')
    plt.legend()
    #plt.show()
    print('max_knn = ',max_knn[0])
    
    
def knn_with_best_k(features_data, target_data, rand_state, number_neighbors):
        X_train, X_test, y_train, y_test = train_test_split(features_data, target_data, 
                                                        random_state=rand_state, test_size=0.2)
        #-----------------------------------------------------------
        model = KNeighborsClassifier(n_neighbors = number_neighbors)
        print("1")
        model.fit(X_train, y_train)
        print("2")
        y_test_predict = model.predict(X_test)
        print("3")
        #-------------------------------------------------------
        cm1 = confusion_matrix(y_test, y_test_predict)
        fig, ax = plt.subplots(figsize=(4, 4))
    
        ax.imshow(cm1)
        ax.grid(False)
        ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
        ax.set_ylim(1.5, -0.5)
        ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
        plt.sca(ax)
        # plt.title('Predict Test Data')
        
        print(cm1.sum(axis=0)[0])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, ('% {}').format(int(cm1[i, j]/cm1.sum(axis= 1)[i]*100)), ha='center', va='center', color='red')
        #plt.show()
        plt.savefig('confusion_matrix_WN_KNeighborsClassifier.png')
        print("Precision_0 ={:.2f}".format((cm1[0,0]/(cm1[0,0] + cm1[1,0]))))
        print("Recall_0 =%f" %(cm1[0,0]/(cm1[0,0]+cm1[0,1])))
        print("Precision_1 ={:.2f}".format((cm1[1,1]/(cm1[1,1] + cm1[0,1]))))
        print("Recall_1 =%f" %(cm1[1,1]/(cm1[1,1]+cm1[1,0])))

knn_model(features, target,1014, 10)
knn_with_best_k(features, target,1014,2)

# ### Random Forest Classifier
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
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.imshow(cm1)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.set_ylim(1.5, -0.5)
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    plt.sca(ax)
    #plt.title('Predict Test Data')
        
    print(cm1.sum(axis=0)[0])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, ('% {}').format(int(cm1[i, j]/cm1.sum(axis= 1)[i]*100)), ha='center', va='center', color='red')
    #plt.show()
    plt.savefig('confusion_matrix_WN_RandomForestClassifier.png')
    #print("Precision_0 ={:.2f}".format((cm1[0,0]/(cm1[0,0] + cm1[1,0]))))
    #print("Recall_0 =%f" %(cm1[0,0]/(cm1[0,0]+cm1[0,1])))
    #print("Precision_1 ={:.2f}".format((cm1[1,1]/(cm1[1,1] + cm1[0,1]))))
    #print("Recall_1 =%f" %(cm1[1,1]/(cm1[1,1]+cm1[1,0])))
    
    all_features = model.feature_importances_
    feat_scores_model = pd.DataFrame({'Fraction of Features': all_features}, index=X_train.columns)
    feat_scores_model = feat_scores_model.sort_values(by='Fraction of Features')
    feat_scores_model = feat_scores_model[-15:]
    #print(feat_scores_model)
    plt.figure(figsize=(50, 50))
    feat_scores_model.plot(kind='barh')
    plt.legend(loc='lower right')
    plt.savefig("the_most_important_WN_RandomForestClassifier.png", bbox_inches='tight')
    return model
model_RandomForestClassifier = RandomForestClassifier_model(features, target, 1259)

# ###  Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

def GaussianNaiveBayes_model(features_data, target_data, rand_state):
    
    X_train, X_test, y_train, y_test = train_test_split(features_data, target_data, 
                                                        random_state=rand_state, test_size=0.2)

    model = GaussianNB()
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
    #plt.show()
    plt.savefig('confusion_matrix_WN_GaussianNaiveBayes.png')
    #print("Precision_0 ={:.2f}".format((cm1[0,0]/(cm1[0,0] + cm1[1,0]))))
    #print("Recall_0 =%f" %(cm1[0,0]/(cm1[0,0]+cm1[0,1])))
    #print("Precision_1 ={:.2f}".format((cm1[1,1]/(cm1[1,1] + cm1[0,1]))))
    #print("Recall_1 =%f" %(cm1[1,1]/(cm1[1,1]+cm1[1,0])))

GaussianNaiveBayes_model(features, target, 1235)


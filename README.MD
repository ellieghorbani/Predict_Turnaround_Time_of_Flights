# The turnaround time prediction to schedule flights based on the real flight data of 2015 by using the machine learning technics
## Table of contents
* [1. Introduction](#1-introduction)
* [2. Methodology and data](#2-methodology-and-data)
* [3. Analysis data and result](#3-analysis-data-and-result )
    - [3.1. Exploratory data analysis (EDA)](#31-Exploratory-data-analysis-EDA)
    - [3.2. Predict turnaround time classes](#32-predict-turnaround-time-classes)
    - [3.3. Modeling data to predict turnaround time based on knowing its distribution](#33-modeling-data-to-predict-turnaround-time-based-on-knowing-its-distribution)
    - [3.4. Apply the best models to test data (unseen data)](#34-apply-the-best-models-to-test-data-unseen-data)
* [4. Conclusions](#4-conclusions) 
* [5. Setup](#5-setup)
* [Reference](#reference)

## 1. Introduction
<p align='justify'> 
As the air transport demand keeps growing, scheduling of flights in the global air transport systems becomes more and more complex. The number of flights has increased since the early 2000s to 2019 globally. This increase reached from 23.8 million in 2005 to 38.9 million in 2019 while in 2020 the number of flights dropped to 16.4 million because of the Covid-19 pandemic [1].<br />
Based on the above data, one of the operational goals of the air transport industry is to optimize the time of Gate‐to‐gate for flights, called Turnaround time (TAT). TAT is one of the most impactful elements in scheduling flights in the airports. TAT is defined as the time required for an aircraft on the ground to be prepared for the departure from its arrival time at the gate [2]. See figure 1.<br />
TAT is generally not pleasant for passengers and costly for airlines. Airlines lose money due to inefficiency in plane boarding as they like to have their planes in the air in the minimum possible time. Therefore predicting the TAT could help airlines and airports to manage their resources more efficiently.<br />
The aim of this work is to predict the scheduling turnaround time of an aircraft in order to decrease conflicts of flights, maximize the passenger conveniences and optimize cost of air carriers. The details of the data, the exploratory data analysis, and the modeling results of TAT is presented in this work.<br />
A survey by the U.S. Department of Transportation's Bureau of Transportation Statistics published information on the on-time performance of domestic flights operated by large air carriers. Summary information on the number of on-time, delayed, canceled, and diverted flights is published in the U.S. Department of Transportation's monthly Air Travel Consumer Report and in this dataset of 2015 flight delays and cancellations [3].<br />

<p align="center">
  <img src="./images/trunaround_time.png" width=800/><br/>
 <caption> <b>Figure 1 </b> A schematic of Turnaround time based on arrival and departure time.
</p>

## 2. Methodology and data
<p align="center">
  <img src="./images/methodology_data.jpg" width=500/><br/>
<caption> <b>Figure 2 </b> Parts of process Turnaround time [4]. </caption>
</p>
<p align='justify'> 
In this work we used the data published by the U.S. Department of Transportation's (DOT) Bureau of Transportation Statistics. They track the on-time performance of domestic flights operated by large air carriers in the USA.  [5]. <br />
The data file has more than 5 million samples and each sample has information of  'year', 'month', 'day', 'day_of_week', 'airline', 'flight_number', 'tail_number', 'origin_airport', 'destination_airport', 'scheduled_departure', 'departure_time', 'departure_delay', 'taxi_out', 'wheels_off', 'scheduled_time', 'elapsed_time', 'air_time', 'distance', 'wheels_on', 'taxi_in', 'scheduled_arrival', 'arrival_time', 'arrival_delay', 'diverted', 'cancelled', 'cancellation_reason', 'air_system_delay', 'security_delay', 'airline_delay',  'late_aircraft_delay', 'weather_delay'. <br />
<p align='justify'> 
The raw data only has the flight information from airport A to B. Hence, we need to create a new table which has the information about TAT (the difference between arrival time to airport B from A and departure from B to C). Also for the first iteration we removed rows with missing data since they are negligible compared to the whole data and converted time columns to standard time as we describe in the following. ‘year’, 'month', 'day', 'day_of_week' columns are converted to standard date (month/day/year). Then those columns are compressed into one column. ‘airline’ column includes fourteen of the most popular airlines in the USA. Since the size of the data file is big and needs a big machine to run the models, for the first iteration we divided the data into fourteen data frames with each new data frame assigned to  one airline. 'Flight_number' is removed because this study is focused on TAT. We track the aircraft travels by 'tail_number', 'origin_airport', and 'destination_airport’ columns. 'Scheduled_departure', 'departure_time', 'scheduled_time', 'air_time', 'scheduled_arrival', 'arrival_time' columns which gives us information about departure and arrival times. They are used to calculate TAT but in the first step we should modify them to the standard time. For example, in ‘scheduled_departure' column, it is reported number 1608, which means the time is 4:08 pm.<br />
 <p align='justify'> 
'ARRIVAL_DELAY', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',  'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'ELAPSED_TIME', 'WHEELS_ON',  and 'TAXI_IN' columns are used to obtain TAT time. This study is not related to divert and cancelation hence we remove 'DIVERTED', 'CANCELLED', and 'CANCELLATION_REASON' columns.<br />
<p align='justify'> 
Our goal is to predict turnaround time of flights. Therefore we should add a new column (labeled data) as TAT by following flights from each aircraft.  Since the tail number is a unique identifier for aircrafts we make a new table based on tracking tail numbers. The new table is included information of the departure time and the arrival time from airport A to airport B and departure time from airport B to airport C.  Therefore the turnaround time at the airport B is difference between the arrival time from airport A to airport B and then the departure time from airport B to airport C. we call this new column ‘turnaround_time_B’ since it is the turnaround time of airport B. <br />
<p align='justify'> 
The new table gives information about  'TAIL_NUMBER', 'AIRLINE', 'airport_A', 'airport_B', 'airport_C', 'turnaround_time_ B', 'SCHEDULED_DEPARTURE_AB', 'DEPARTURE_TIME_AB', 'DEPARTURE_DELAY_AB', 'SCHEDULED_ARRIVAL_AB', 'ARRIVAL_TIME_AB', 'ARRIVAL_DELAY_AB', 'SCHEDULED_DEPARTURE_BC', 'DEPARTURE_TIME_BC', 'DEPARTURE_DELAY_BC', 'ELAPSED_TIME_AB', 'ELAPSED_TIME_BC', 'DISTANCE_AB', 'DISTANCEBC'. <br />
<p align='justify'> 
To avoid leakage we drop columns related to the departure information from airport B to C from above table. This is because when an aircraft arrives at airport B, there is no information about the departure such as delay time and the only available data is the scheduled departure time from airport B to C. <br />
After calculating turnaround time B, the time columns are divided into hour, day, month, and day of week for EDA. <br />
</p>

## 3. Analysis data and result
<p align='justify'> 
In this study, the database included 5 million recorded flights for the fourteen most popular airlines in the USA in 2015, table 1. We only focus on analyzing data of the airlines with more than 10 percent of total flights. Based on table 1, Airlines WN, AA, and OO with %22, %12, and %10 number flights, respectively, have the most stock of flights.<br />
The datafram is divided into 1217135, 673631, and 554286 tables for WN, AA, and OO airlines respectively. Every datafram has 20 columns as follow: 'airport_A', 'airport_B', 'airport_C', 'DEPARTURE_DELAY_AB', 'ARRIVAL_DELAY_AB', 'ELAPSED_TIME_AB', 'DISTANCE_AB', 'DISTANCE_BC', 'DEPARTURE_HOUR_AB', 'DEPARTURE_weekday_AB', 'DEPARTURE_day_AB', 'DEPARTURE_month_AB', 'ARRIVAL_HOUR_AB', 'ARRIVAL_weekday_AB', 'ARRIVAL_day_AB',  'ARRIVAL_month_AB', 'SCHEDULED_DEPARTURE_HOUR_BC', 'SCHEDULED_DEPARTURE_weekday_BC', 'SCHEDULED_DEPARTURE_day_BC', and 'SCHEDULED_DEPARTURE_month_BC' as features and ‘turnaround time_B’ as target. <br />
</p>
<table> 
    <caption><b>Table 1</b>  Total flights were 5483816 which carried out with the fourteen most popular airlines in the USA in 2015. The dataframe includes 20 columns.  </caption> 
    <tr align="center"> <th>Airlines</th><th>Number Data</th> <th>Percentage number flights</td></tr> 
    <td align="center">American Airlines (AA)</td><td align="center">673631 </td><td align="center">%12.28</td></tr>
    <td align="center">Alaska Airlines  (AS)</td><td align="center">168105      </td><td align="center">%3.1</td></tr>
    <td align="center">JetBlue Airlines (B6)</td><td align="center">252122      </td><td align="center">%4.6</td></tr>
    <td align="center">Delta airlines (DL)  </td><td align="center">830012      </td><td align="center" >%15.1</td></tr>
    <td align="center">Eva Air Airlines (EV)</td><td align="center">528429      </td><td align="center">%9.6	</td></tr>
    <td align="center">Frontier Airlines (f9) </td><td align="center">87499       </td><td align="center">%1.6</td></tr>
    <td align="center">Hawaiian Airlines (HA) </td><td align="center">73804       </td><td align="center">%1.3	</td></tr>
    <td align="center">American Eagle Airlines (MQ) </td><td align="center">265283  </td><td align="center">%4.8	</td></tr>
    <td align="center">Spirit Wings Airlines (NK) </td><td align="center">112357    </td><td align="center">%2.1	</td></tr>
    <td align="center">SkyWest Airlines (OO) </td><td align="center">554286   </td><td align="center">%10.1	</td></tr>
    <td align="center">United Airlines (UA) </td><td align="center">471848     </td><td align="center">%8.6	</td></tr>
    <td align="center">United States Airlines (US) </td><td align="center">189048  </td><td align="center">%3.4	</td></tr>
    <td align="center">Virgin America Airlines (VX) </td><td align="center">60257   </td><td align="center">%1.1	</td></tr>
    <td align="center">Southwest Airlines (WN) </td><td align="center">1217135  </td><td align="center">%22.2	</td></tr>
    <td align="center">Total Airlines in 2015	 </td><td align="center">5483816   </td><td align="center">%100	</td></tr>   
</table>
    
<p align='justify'> 
 Before we apply our supervised machine learning technique to this problem, we divide the data to train, cross validation and test and for the test data we keep 10 percent of data from each airline which helps us to measure the performance for unseen data. We train our model with  the training and cross validation data and then use test data to predict turnaround time for unseen data.<br/> 
 </p>
<p align='justify'>
As explained we follow three steps to predict the turnaround time. First step is to explorate date as exploratory data analysis (EDA). <br/>
Second step is to build a binary classification model to find if a flight belongs to distribution1 (turnaround time less than four hours) or the distribution2 (turnaround time more than four hours). We apply Logistic Regression, k-Nearest Neighbors and, Random Forest Classifier to this problem. In the next section, we’ll show precision, recall and confusion matrix  to compare the models' results. <br />
For the third step, after clustering the data, we have a regression problem and we apply machine learning techniques Linear Regression, Gradient Boosting Regressor, and Random Forest Regressor to make a prediction for turnaround time. We use root mean square (RMS) as the parameter of performance of the models and finding the best model. 
</p>

## 3.1. Exploratory data analysis (EDA) 
<p align='justify'> 
 To get some insight about the data we perform exploratory data analysis (EDA). In figure 3, we show the histograms of turnaround time for WN, AA, and OO airlines. Figures 4, 5 and 6 show inreval and average of turnaround time and number arrival flights from airport A to B, departure flights from airport A to B and scheduled departure flights from airport B to C via day time for OO, AA and WN airlines respectively. <br/>
In figure 4, all of the airlines have the maximum turnaround time around 10 pm based on the arrival time from airports A to B. Between 4 am to 6 pm the turnaround time is more or less constante. From 6 pm to 10 pm, turnaround time increases up to 6 hours. Then from 10 pm to 4 am of next day turnaround time decreases. The inreval turnaround time is shoter between 1 am to 6 am than other times of day. we expected that turnaround time goes up where the airport is crowded and there are the most number flights in  the airport but this figure showes revere results and turnaround time is independent of number of flights. The number of flights depend the arrival time and their airline. Forexample during day, there is more number of flights for all airlines. Airline WN has the most of flights.  <br/>

In figure 5, airlines OO and WN based on  the departure time from airports A to B, do not have large turnaround time intervals between 1 am to 5 am while during this priod airline AA has large turnaround time intervals like other time of day. The maximun number of flights happen early morning while turnaround time is not maximum. The maximum turnaround time is between 6 pm to 10 pm. <br/>
Figure 6 is based on the scheduled departure time from airports B to C. In this schedule, airlines prefer to have maximum fights and average of turnaround time during day. <br/>
For more EDA, you can follow the stracture is mentioned in section 5. setup to find EDA codes.<br/>
Carefully investigating figure 3, we observe that the TAT for all three airlines is made of two distributions. This gives us hints to make a classification model first to classify the data in two groups for TAT and then develop regression models for predicting TAT for each group. 
 </p>
 <p align="center">
  <img src="./images/first_3_histogram.png">
</p> <br/>
<b>Figure 3</b> Illustrate the TAT histogram of the three most popular airlines in 2015 in the USA.  There are two distributions for each airline. Hence first we confront with a binomial problem to predict the distribution of record data.

<p align="center">
  <img src="./images/EDA_ARRIVAL_TIME_AB_Day time (h).png">
</p> <br/>
<b>Figure 4</b> Illustrate the arrival time from airports A to B of OO, AA and WN airlines (from left to right). Blue points and light blue lines show average and interval of Turnaround time via time of day respectively.  Red points is number flights during the day.

<p align="center">
  <img src="./images/EDA_DEPARTURE_TIME_AB_Day time (h).png">
</p> <br/>
<b>Figure 5</b> Illustrate the departure time from airports A to B of OO, AA and WN airlines (from left to right). Blue points and light blue lines show average and interval of Turnaround time via time of day respectively.  Red points is number flights during the day.

<p align="center">
  <img src="./images/EDA_SCHEDULED_DEPARTURE_BC_Day time (h).png">
</p> <br/>
<b>Figure 6</b> Illustrate the scheduled departure time from airports B to C of OO, AA and WN airlines (from left to right). Blue points and light blue lines show average and interval of Turnaround time via time of day respectively.  Red points is number flights during the day. 
</p>

## 3.2. Predict turnaround time classes 
<p align='justify'>  
Since there are two distributions, we apply classification methods Logistic Regression, k-Nearest Neighbors and Random Forest Classifier to obtain the best model. We find out the best model based on precision and recall. <br />
In figure 7 we show diffusion matrices for OO, AA and WN airlines. In table 2, we compared precision, recall and f score to determinate the best model. Interestingly one machine learning method is not always the winner. Based on our results (table 2), the best model is Logistic Regression and Random Forest Classifier for OO and AA airlines respectively. For WN airlines, Logistic Regression, and Random Forest Classifier give the same recall and precision. <br />
</p>

<p align="center">
  <img src="./images/confusion_matrix_OO_LogisticRegression.png" width='400'>
  <img src="./images/confusion_matrix_AA_RandomForestClassifier.png" width='400'> <br/>
  <img src="./images/confusion_matrix_WN_LogisticRegression.png" width ='400'>
  <img src="./images/confusion_matrix_WN_RandomForestClassifier.png" width ='368'>
</p>
<p align='justify'>
<b>Figure 7</b> The diffusion matrix of OO airline (top, left) from Logistic Regression model, AA airline (top, right) from Random Forest Classifier model, WN airline (bottom, left) from Logistic Regression model and WN airline (bottom, right) from Random Forest Classifier model.  0s mentions to distribution 1 and 1s mentions distribution 2.  The yellows, blue areas are percentage of data with the correct predictions in distribution 1 and 2 re percentage of data prediction wrongly  in distribution 1, 2 respectively.<br/>
</p>

<table> 
    <caption><b>Table 2</b>  The comparisons between recall and precision parameters for distributions 1 and 2. The best model based on the comparison of f score is the Logistic Regression and Random Forest Classifier for OO and AA airlines respectively. For WN airlines, Logistic Regression, and Random Forest Classifier models give the same recall and precision. </caption> 
    <tr> <th></th><th></th><th></th> <th>k-Nearest Neighbors</th> <th>Logistic Regression</th> <th>Random Forest Classifier</th></tr> 
    <tr> <th rowspan="6">OO Airlines</th>  
  <th rowspan="3">Distribution 1</th>
        <td align="center">Recall</td><td align="center">0.99</td><td align="center">0.99</td><td align="center">1.00 </tr> <tr> 
        <td align="center">Precision</td> <td align="center">0.93</td><td align="center">0.96</td><td align="center">0.95</td> </tr> <tr>
        <td align="center"><b> F % </b></td><td align="center"><b>95.90</b></td><td align="center"><b>97.4</b></td><td align="center"><b>97.43</b></td> </tr> <tr>
     <th rowspan="3">Distribution 2</th>
    <td align="center">Recall</td><td align="center">0.71</td><td align="center">0.85</td><td align="center">0.79</td> </tr> <tr> 
    <td align="center">Precision</td><td align="center">0.97</td><td align="center">0.99</td><td align="center">1.00</td> </tr> 
        <td align="center"><b> F % </b></td><td align="center"><b>81.98</b></td><td align="center"><b>91.46</b></td><td align="center"><b>88.26</b></td> </tr> <tr>
  <tr> <th rowspan="6">AA Airlines</th> 
    <th rowspan="3">Distribution 1</th>
        <td align="center">Recall</td><td align="center">0.97</td><td align="center">0.96</td><td align="center">0.99 </tr> <tr> 
        <td align="center">Precision</td> <td align="center">0.90</td><td align="center">0.93</td><td align="center">0.95</td> </tr> <tr>
        <td align="center"><b> F % </b></td><td align="center"><b>93.36</b></td><td align="center"><b>94.46</b></td><td align="center"><b>6.96</b></td> </tr> <tr>
     <th rowspan="3">Distribution 2</th>
    <td align="center">Recall</td><td align="center">0.71</td><td align="center">0.85</td><td align="center">0.79</td> </tr> <tr> 
    <td align="center">Precision</td><td align="center">0.97</td><td align="center">0.99</td><td align="center">1.00</td> </tr> <tr>
        <td align="center"><b> F % </b></td><td align="center"><b>81.98</b></td><td align="center"><b>91.46</b></td><td align="center"><b>88.26</b></td> </tr> <tr>
    
  <tr> <th rowspan="6">WN Airlines</th> 
    <th rowspan="3">Distribution 1</th>
        <td align="center">Recall</td><td align="center">0.41	0.99</td><td align="center">0.99</td><td align="center">0.99 </tr> <tr> 
        <td align="center">Precision</td><td align="center">0.93</td><td align="center">0.99</td><td align="center">0.99</td> </tr> <tr>
        <td align="center"><b> F % </b></td><td align="center"><b>95.90</b></td><td align="center"><b>99.00</b></td><td align="center"><b>99.00</b></td> </tr> <tr>
     <th rowspan="3">Distribution 2</th>
    <td align="center">Recall</td><td align="center">0.67</td><td align="center">0.94</td><td align="center">0.94</td> </tr> <tr> 
    <td align="center">Precision</td><td align="center">0.97</td><td align="center">0.99</td><td align="center">0.99</td> </tr> <tr>
        <td align="center"><b> F % </b></td><td align="center"><b>79.25</b></td><td align="center"><b>96.43</b></td><td align="center"><b>96.43</b></td> </tr> <tr>
        
</table>
<br/>

<p align='justify'> 
For the k-Nearest Neighbors we try between 1 to 100 neighbors for all three airlines and the best model is obtained with 2 nearest neighbors. <br />
Figure 8 shows the feature importance for AA, OO, and WN airlines. For these airlines the most important features are scheduled departure month from airport B to C, airports A, arrival time to airport B, scheduled departure from airport B. The most important features of AA and WN airlines are like each other. <br />    
</p>
<p align="center">
  <img src="./images/the_most_important_OO_LogisticRegression.png"><br/>
  <img src="./images/the_most_important_AA_RandomForestClassifier.png"> <br/>
  <img src="./images/the_most_important_WN_LogisticRegression.png" ><br/>
  <img src="./images/the_most_important_WN_RandomForestClassifier.png"><br/>
</p>

<b>Figure 8</b> The feature importance for airlines OO, AA and WN.  From above to below, the first figure shows the most feature of OO airlines based on Logistic Regression model, next figure is the most feature of OO airlines based on Random Forest classifier and the last two figures are for WN airlines based on Logistic Regression and Random Forest classifier respectively. The features importance for AA and WN airlines based on Random Forest classifier are mostly the same.

### 3.3. Modeling data to predict turnaround time based on knowing its distribution
<p align='justify'> 
For the next step we apply Linear Regression, Gradient Boosting Regressor, and Random Forest Regressor to each class. We apply each of these techniques to the training data and then apply the trained model to the cross validation data to find the best model. To evaluate the models we use the root mean square error (RMSE). In Table 3 we show the RMSE results for cross validation and base model. We use the mean as the base model. Also in the table we show the mean of cross validation (CV) data and mean of predicted cross validation (PCV) data.  This results show that Linear Regression is not a good model for OO airline since the predicted RMSE is much bigger than the baseline. However, Gradient Boosting Regressor gives much better results with predicted RMSE almost half the baseline RMSE. <br /> 
Fig 9 shows histogram of cross validation data and predicted for OO, AA, and WN airlines. The predicted target comes from the Gradient Boosting Regressor model as the best model.
For the next step we find the feature importance from the Gradient Boosting Regressor model. Fig 10 shows the first ten feature importance of distribution1 and distribution2 for OO, AA, WN airlines. Based on this plots, arrival hour from airport A to B, scheduled departure hour from A to B and arrival delay from A to B are the most important features for distribution1. Scheduled departure hour, delay in arrival from A to B and departure hour from A to B are the most important features for distribution2. <br />
</p>
<table> 
    <caption><b>Table 3</b>  Comparison Linear Regression, Gradient Boosting Regressor, and Random Forest Regressor models to predict the turnaround time B. Gradient Boosting Regressor model is the best model for distribution1 and distribution2 of OO, AA, and WN airlines. 'RMSE CV&amp;PCV'and 'RMSE CV&amp;mean_CV'are RMSE of cross validation and predicted cross validation data and RMSE of cross validation and average of cross validationndata. 'Mean Data', Mean CV' and 'Mean PCV' are averagne of train data, cross validation and predicted cross validation data.  </caption> 
    <tr> <th></th><th></th><th>Model</th> <th>RMSE CV&amp;PCV</th> <th>RMSE CV&amp;mean_CV</th> <th>Mean Data</th> <th>Mean <br/>CV</th> <th>Mean PVC</th> </tr> 
    <tr> <th rowspan="6">OO Airlines</th> 
    <th rowspan="3">Distribution 1</th>
        <td>Linear Regression</td><td>8635.009496</td><td>0.583960</td><td>0.870899</td><td>0.870498</td><td>-31.93344</td> </tr> <tr> 
        <td>Gradient Boosting Regressor</td> <td>0.289393</td><td>0.583960</td><td>0.870899</td><td>0.870498</td><td>0.870577</td> </tr> <tr>
        <td>Random Forest Regressor</td><td>0.422701</td><td>0.583960</td><td>0.870899</td><td>0.870498</td><td>0.868635</td> </tr> <tr>
     <th rowspan="3">Distribution 2</th>
        <td>Linear Regression</td><td>2.244891</td><td>3.444053</td><td>9.724552</td><td>9.702995</td><td>9.707080</td> </tr> <tr> 
        <td>Gradient Boosting Regressor</td> <td>0.973046</td><td>3.459548</td><td>9.721189</td><td>9.716447</td><td>9.720250</td> </tr> <tr>
        <td>Random Forest Regressor</td><td>1.058651</td><td>3.459065</td><td>9.723550</td><td>9.707000</td><td>9.714718</td> </tr> <tr>
    
  <tr> <th rowspan="6">AA Airlines</th> 
    <th rowspan="3">Distribution 1</th>
        <td>Linear Regression</td><td>0.391387</td><td>0.530024</td><td>1.204137</td><td>1.203021</td><td>1.202494</td> </tr> <tr> 
        <td>Gradient Boosting Regressor</td> <td>0.301565</td><td>0.530024</td><td>1.204137</td><td>1.203021</td><td>1.203819</td> </tr> <tr>
        <td>Random Forest Regressor</td><td>0.410178</td><td>0.530024</td><td>1.204137</td><td>1.203021</td><td>1.202731</td> </tr> <tr>
     <th rowspan="3">Distribution 2</th>
        <td>Linear Regression</td><td>587.676068</td><td>3.986316</td><td>10.466827</td><td>10.488394</td><td>6.158463</td> </tr> <tr> 
        <td>Gradient Boosting Regressor</td> <td>0.844714</td><td>3.950069</td><td>10.477505</td><td>10.445680</td><td>10.445323</td> </tr> <tr>
        <td>Random Forest Regressor</td><td>1.016071</td><td>3.961868</td><td>10.471530</td><td>10.469583</td><td>10.470858</td> </tr> <tr>
   
  <tr> <th rowspan="6">WN Airlines</th> 
    <th rowspan="3">Distribution 1</th>
        <td>Linear Regression</td><td>0.296270</td><td>0.370710</td><td>0.838973</td><td>0.840945</td><td>0.839716</td> </tr> <tr> 
        <td>Gradient Boosting Regressor</td> <td>0.237668</td><td>0.370710</td><td>0.838973</td><td>0.840945</td><td>0.840734</td> </tr> <tr>
        <td>Random Forest Regressor</td><td>0.304715</td><td>0.370710</td><td>0.838973</td><td>0.840945</td><td>0.839061</td> </tr> <tr>
     <th rowspan="3">Distribution 2</th>
        <td>Linear Regression</td><td>1.804132</td><td>2.943751</td><td>8.757885</td><td>8.822551</td><td>8.788771</td> </tr> <tr> 
        <td>Gradient Boosting Regressor</td> <td>0.544187</td><td>2.898495</td><td>8.770735</td><td>8.771154</td><td>8.772329</td> </tr> <tr>
        <td>Random Forest Regressor</td><td>0.645750</td><td>2.917877</td><td>8.770400</td><td>8.772492</td><td>8.771918</td> </tr> <tr>
    
</table>

<p align="center">
  <img src="./images/histogram_OO_Gradient Boosting Regressor_predict_target_1.png" width=480 >
  <img src="./images/histogram_OO_Gradient Boosting Regressor_predict_target_2.png" width=480 > <br/>

  <img src="./images/histogram_AA_Gradient%20Boosting%20Regressor_predict_target_1.png" width=480 >
  <img src="./images/histogram_AA_Gradient Boosting Regressor_predict_target_2.png" width=480 > <br/>

  <img src="./images/histogram_WN_Gradient%20Boosting%20Regressor_predict_target_1.png" width=480 >
  <img src="./images/histogram_WN_Gradient Boosting Regressor_predict_target_2.png" width=480 > 
</p>
<b>Figure 9</b> Left and right sides show the histogram of the test and predicted target of distribution1 and distribution2 respectively. From top to bottom the histograms are related OO, AA and WN airlines. The predicted target comes from the Gradient Boosting Regressor model as the best model that we have.

<p align="center">
  <img src="./images/the_most_important_OO_Gradient Boosting Regressor_predict_target_1.png" width=480/>
  <img src="./images/the_most_important_OO_Gradient Boosting Regressor_predict_target_2.png" width=480/> 

  <img src="./images/the_most_important_AA_Gradient Boosting Regressor_predict_target_1.png" width=480/>
  <img src="./images/the_most_important_AA_Gradient Boosting Regressor_predict_target_2.png" width=480/> 

  <img src="./images/the_most_important_WN_Gradient Boosting Regressor_predict_target_1.png" width=480/>
  <img src="./images/the_most_important_WN_Gradient Boosting Regressor_predict_target_2.png" width=480/> 
</p>
<b> Figure 10</b> The ten most important features of OO, AA, WN airlines. Left and right sides are for distribution1 and distribution2 respectively. From top to bottom we have the most features of OO, AA and WN airlines. These results come from the Gradient Boosting Regressor model as the best model that we have.  The three most important features of OO, AA and WN airlines are the same based on their distributions. ARRIVAL_HOUR_AB, SCHEDULED_DEPARTURE_HOUR_BC, and APPRIVAL_DELAY_AB for distribution1 and SCHEDULED_DEPARTURE_HOUR_BC, APPRIVAL_DELAY_AB, and DEPARTURE_HOUR_AB for distribution2 are the three most important features of OO, AA, and WN airlines.

### 3.4. Apply the best models to test data (unseen data)
<p align='justify'> 
To measure the performance of our models with unseen (US) data we apply the best model from classification to find the class of each test data and then apply the best regression model to predict the TAT. In Table 4 we summarize our results for OO, AA and WN airlines. Our modeling results clearly outperforms the baseline with much smaller RMSE. 
</p>
<table> 
    <caption><b>Table 4</b>  Illustrate results of unseen data prediction based on the best ML models found out in previous sections for OO, AA and WN airlines. 'RMSE US&amp;PUS' and 'RMSE US&amp;mean_US' are assigned to RMSE of unseen data and its predection and RMSE of Unseen data and its average respectively. 'Mean US' and 'Mean PUS' are average of unseen and predicted unseen data respectively.  </caption> 
    <tr> <th>Airlines</th><th>Model</th> <th>RMSE US&amp;PUS</th> <th>RMSE US&amp;mean_US</th> <th>Mean US</th> <th>Mean PUS</td></tr> 
        <td>OO</td><td>Logistic + Gradient Boosting Regressor models</td><td>0.002488</td><td>0.016632</td><td>2.644437</td><td>2.637634</td> </tr> <tr> 
        <td>AA</td><td>Random Forest Classifier + Gradient Boosting Regressor models</td><td>0.002910</td><td>0.017936</td><td>3.746930</td><td>3.784084</td> </tr> <tr>
        <td>WN</td><td>Random Forest Classifier + Gradient Boosting Regressor models</td><td>0.000691</td><td>0.0095490</td><td>2.293647</td><td>2.293288</td> </tr> <tr>
    </table>

## 4. Conclusions
<p align='justify'> 
In this project, we predict turnaround time based on the flight data of 2015 for Southwest Airlines (WN), American Airlines (AA) and SkyWest Airlines (OO) as the most popular airlines in the USA. For this prediction, we apply the supervised learning techniques. First interesting thing is there are two distributions in the turnaround time histogram based on duration of turnaround time is less and more than four hours. The Random Forest Classifier model is the best model to predict distributions. The most important features are scheduled departure month from airport B to C, airports A, arrival time to airport B, scheduled departure from airport B. The most important features of AA and WN airlines are like each other. <br/>
In the next step, the gradient boosting model as the best model, is applied to predict turnaround time for those three airlines. For this fitting the most important features are arrival hour from airport A to B, scheduled departure hour from A to B and arrival delay from A to B for distribution1, and scheduled departure hour, delay in arrival from A to B and departure hour from A to B for distribution2.
<p/>
## 5. Setup
In this repository, there is two folders:<br/>
1- notebooks: all files in Jupyter notebook <br/>
2- src : all files save with .py <br/>
All codes are written in Python 3.<br/>
For run this project we should run "main_code.py" in "src" folder or "main_code.ipynp" in "notebooks" folder. In "main_code" is imported these functions with different tasks:<br/>

* step_1_cleaning_data<br/> 
* step_2_convert_timecolumns_to_standard_time<br/>        
* step_3_choose_features_and_define_target<br/> 
* step_4_EDA_1<br/> 
* step_4_EDA_2<br/>
* step_4_EDA_of_AA_OO_WN_3<br/>  
* step_5_EDA_and_separate_airline_data<br/>
* step_6_apply_ML_models_AA_airlines<br/>
 - step_6_1_divide_data_into_seen_and_unseen_files<br/>  
 - step_6_2_predict_categories_of_target<br/> 
 - step_6_3_predict_seen_target<br/>
 - step_6_4_predic_unseen_target<br/>
* step_6_apply_ML_models_OO_airlines
 - step_6_1_divide_data_into_seen_and_unseen_files<br/>  
 - step_6_2_predict_categories_of_target<br/> 
 - step_6_3_predict_seen_target<br/>
 - step_6_4_predic_unseen_target<br/>                
* step_6_apply_ML_models_WN_airlines
 - step_6_1_divide_data_into_seen_and_unseen_files<br/>  
 - step_6_2_predict_categories_of_target<br/> 
 - step_6_3_predict_seen_target<br/>
 - step_6_4_predic_unseen_target<br/>

## Reference
[1] E. Mazareanu,“Global air traffic - number of flights 2004-2021”, Dec 2, 2020. (https://www.statista.com/statistics/564769/airline-industry-number-of-flights/#statisticContainer) <br/>
[2] M. A. Makhloof, M. E. Waheed, U. A. Badawi, "Real-time aircraft turnaround operations manager", Production Planning & Control 25:1, pages 2-25, 2014.<br/>
[3] https://www.kaggle.com/usdot/flight-delays <br/>
[4] https://www.azcentral.com <br/>
[5] https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp <br/>

# The turnaround time prediction to schedule flights based on the real flight data of 2015 by using the machine learning technics 
## Table of contents
* [Introduction](#Introduction)
* [Methodology and data](#Methodology-and-data)
* [Results and discussion](#Results-and-discussion)
* [Setup](#setup)

## Introduction
The air transport demand keeps growing, schedules of flights in the global air transport system become more complex. The number of flights has increased since the early 2000s up to 2019 globally. This increase reached from 23.8 million in 2005 to 38.9 million in 2019 while in 2020 with increasing the Coronavirus pandemic,   the number of flight dropped to 16.4 million [“Global air traffic - number of flights 2004-2021”, Published by E. Mazareanu, Dec 2, 2020, https://www.statista.com/statistics/564769/airline-industry-number-of-flights/#statisticContainer]. 
Based on the above data, one of the operational goals of the air transport industry is to optimize the time of Gate‐to‐gate for flights, called Turnaround time (TAT). TAT is one of the most important parts to schedule flights in the airports. TAT is defined as the time required for an aircraft on the ground to be prepared for the departure from its arrival time at the gate.  [M. Abd Allah Makhloof, M. Elsayed Waheed, Usama A. El-Raouf Badawi. (2014) Real-time aircraft turnaround operations manager. Production Planning & Control 25:1, pages 2-25. ]. TAT is generally time consuming and not pleasant for passengers and costly for airlines. Airlines lose money due to inefficiency in plane boarding as they like to have their planes in the air in the minimum possible time. TAT is still a bottleneck in bringing the passengers to their seats and getting the plane to take-off in an efficient time.  
A survey by the U.S. Department of Transportation's Bureau of Transportation Statistics published information of the on-time performance of domestic flights operated by large air carriers. Summary information on the number of on-time, delayed, canceled, and diverted flights is published in the U.S. Department of Transportation's monthly Air Travel Consumer Report and in this dataset of 2015 flight delays and cancellations [https://www.kaggle.com/usdot/flight-delays].
The aim of this work is to predict the scheduling turnaround time of an aircraft in order to decrease conflicts of flights, maximize the passenger conveniences and optimize cost of air carriers. The details of the data are given in next Section with machine learning results and case studies of the proposed aircraft turnaround model.

## Methodology and data
In this work we used the data published by the U.S. Department of Transportation's (DOT) Bureau of Transportation Statistics. They track the on-time performance of domestic flights operated by large air carriers in 2015 in the USA.  [https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp].
The data file has more than 5 milion data and each data is included information of  'year', 'month', 'day', 'day_of_week', 'airline', 'flight_number', 'tail_number', 'origin_airport', 'destination_airport', 'scheduled_departure', 'departure_time', 'departure_delay', 'taxi_out', 'wheels_off', 'scheduled_time', 'elapsed_time', 'air_time', 'distance', 'wheels_on', 'taxi_in', 'scheduled_arrival', 'arrival_time', 'arrival_delay', 'diverted', 'cancelled', 'cancellation_reason', 'air_system_delay', 'security_delay', 'airline_delay',  'late_aircraft_delay', 'weather_delay'. Due to missing information in some records (rows) or modifying some columns, we make a new data frame. Let’s explain about the new table. ‘year’, 'month', 'day', 'day_of_week' columns should be converted to standard date (month/day/year) in pandas library. Then those columns are compressed into one column. ‘airline’ column is including fourteen of the most popular airlines in the USA. Based on the size of the data file is huge, we divided data into fourteen data frames and each new data frame is assigned for one airline. 'Flight_number' is removed because this study is focused on TAT based on airlines and airports. It could be able to track the aircraft travels by 'tail_number', 'origin_airport', and 'destination_airport’ columns. 'Scheduled_departure', 'departure_time', 'scheduled_time', 'air_time', 'scheduled_arrival', 'arrival_time' columns are about time of flights for schedule, departure and arriving. They are used to calculate TAT but in first step we should modified them to the standard time. For example in ‘scheduled_departure' column, it is reported number 1608, this number means the time is 4:08 pm. 
'ARRIVAL_DELAY', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',  'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'ELAPSED_TIME', 'WHEELS_ON',  and 'TAXI_IN' columns are used to obtain TAT time. This study is not related to divert and cancelation then we remove 'DIVERTED', 'CANCELLED', and 'CANCELLATION_REASON' columns.
Our goal is to predict turnaround time of flights. Then we should add a new column (labeled data) as TAT. By regarding definition of TAT as mentioning above, TAT is the difference of the arrival time to an airport and departure time from that airport as well. This problem would be more interesting if we know which airport (airport A) the aircraft is coming to the mentioned airport (airport B) and which airport (airport C) the aircraft will go. Therefor for this issue, we make a new table based on tracking tail numbers. The new table is included information of the departure time and the arrival time from airport A to airport B and departure time from airport B to airport C.  Therefore the turnaround time at the airport B is difference between the arrival time from airport A to airport B and then the departure time from airport B to airport C. we call this new column ‘turnaround_time_B’ since it is the turnaround time of airport B. 
The new table gives information about  'TAIL_NUMBER', 'AIRLINE', 'airport_A', 'airport_B',       'airport_C', 'turnaround_time_ B', 'SCHEDULED_DEPARTURE_AB',       'DEPARTURE_TIME_AB', 'DEPARTURE_DELAY_AB', 'SCHEDULED_ARRIVAL_AB',       'ARRIVAL_TIME_AB', 'ARRIVAL_DELAY_AB', 'SCHEDULED_DEPARTURE_BC',       'DEPARTURE_TIME_BC', 'DEPARTURE_DELAY_BC', 'ELAPSED_TIME_AB',       'ELAPSED_TIME_BC', 'DISTANCE_AB', 'DISTANCEBC'.
Based on some discussions in the following, we drop columns related to the departure information from airport B to C from above table. First, turnaround time B is directly obtained from difference of arrival time from airport A to B and departure time from airport B to C. If the table has those two columns simultaneously when we apply Machine learning models, the model just will be fit with the arrival and departure time. The other features do not important to the prediction. Second, when the aircraft arrives to airport B, nobody knows how long it takes a time to prepare for departure exactly and how much there is a delay time. We just know schedule departure time from airport B to C.
After calculating turnaround time B, the time columns are divided into hour, day, month, and day of week for more interpretation of features.   
## Results and discussion
In this study, the database including 5 million recorded flights for the fourteen most popular airlines in the USA in 2015, table 1. We focus to analyze data of the airlines with more than 10 percent of total flights. Based on table 1, Airlines WN, AA, and OO with %22, %12, and %10 number flights respectively, they have more stock of flights.
The datafram is divided into 1217135, 673631, and 554286 tables for WN, AA, and OO airlines respectively. Every datafram has 20 columns included 'airport_A', 'airport_B', 'airport_C',    'DEPARTURE_DELAY_AB', 'ARRIVAL_DELAY_AB', 'ELAPSED_TIME_AB', 'DISTANCE_AB', 'DISTANCE_BC', 'DEPARTURE_HOUR_AB', 'DEPARTURE_weekday_AB', 'DEPARTURE_day_AB', 'DEPARTURE_month_AB', 'ARRIVAL_HOUR_AB', 'ARRIVAL_weekday_AB', 'ARRIVAL_day_AB',  'ARRIVAL_month_AB', 'SCHEDULED_DEPARTURE_HOUR_BC', 'SCHEDULED_DEPARTURE_weekday_BC', 'SCHEDULED_DEPARTURE_day_BC', and 'SCHEDULED_DEPARTURE_month_BC' as features and ‘turnaround time_B’ as target.



**Table 1** Total flights were 5483816 which carried out with the fourteen most popular airlines in the USA in 2015. The dataframe includes 20 columns. 

|Name airline                |Number data|Number columns|Percentage number flights|
|----------------------------|-----------|--------------|-------------------------| 
|American Airlines (AA)      |673631     |20            |%12.28		          |
|Alaska Airlines  (AS)	     |168105     |20            |%3.1         		  |
|JetBlue Airlines (B6)	     |252122     |20            |%4.6			  |
|Delta airlines (DL)	     |830012     |20            |%15.1			  |
|Eva Air Airlines (EV)	     |528429     |20  		|%9.6			  |
|Frontier Airlines (f9)	     |87499      |20  		|%1.6			  |
|Hawaiian Airlines (HA)	     |73804      |20  		|%1.3			  |
|American Eagle Airlines (MQ)|265283     |20  		|%4.8			  |
|Spirit Wings Airlines (NK)  |112357     |20  		|%2.1			  |
|SkyWest Airlines (OO)	     |554286     |20  		|%10.1			  |
|United Airlines (UA)	     |471848     |20  		|%8.6			  |
|United States Airlines (US) |189048     |20  		|%3.4			  |
|Virgin America Airlines (VX)|60257      |20  		|%1.1			  |
|Southwest Airlines (WN)     |1217135    |20  		|%22.2			  |		
|Total Airlines in 2015	     |5483816    |20  		|%100			  |

 By using exploratory data analysis (EDA), we can understand much better the data like their histograms. In figure 1, we just show turnaround time B histograms of WN, AA, and OO airlines. For more EDA, you can follow this link: [link of our code]. Regarding figure 1, all three airlines have two distributions for turnaround time B. It means, first we should make a model to predict each data belong which the distributions. Before applying the supervise machine learning on this problem, we keep 10 percent data of each airline as unseen data. After training, modeling and testing the rest of data, we will apply the best model to predict turnaround time of the unseen data. 
 
![My image](./images/first_3_histogram.png)
**Figure 1** Illustrate the TAT histogram of the three most popular airlines in 2015 in the USA.  There are two distributions for each airline. Hence first we confront with a binomial problem to predict the distribution of record data.

For prediction turnaround time, there are two steps. First step is to face with a binomial problem to recognize data is either on the distribution1 (turnaround time less than four hours) or the distribution2 (turnaround time more than four hours). Based on the problem is binomial, we apply Logistic Regression, k-Nearest Neighbors, Random Forest Classifier, and Naive Bayes to build the best model.  In the next section, we’ll show results of matrix confusions to compare those models. 
Second step after categorizing the data based on their TAT are smaller or larger than 4 hours (distributions 1 and 2), we will figure out the best model to predict turnaround time for each distribution. While there is a classification problem in the first step, now we have regression supervised learning. Then we apply the models such as Linear Regression, Gradient Boosting Regressor, and Random Forest Regressor. By calculating the root mean square (RMS) as parameter to realize which model has the best fit for our data.
### Modeling data to predict the category of data distribution 
Since there are two distributions, we apply the binomial models such as Logistic Regression, k-Nearest Neighbors, Random Forest Classifier, and Gaussian Naive Bayes to obtain the best model. We find out the best model based on the diffusion matrixes. From the diffusion matrixes we calculated precision, recall, and f score parameters. Ad they are defined 
Recall = TruePositives / (TruePositives + FalseNegatives) = c11/(c11+c21),
Precision = TruePositives / (TruePositives + FalsePositives) = c11/(c11+c12),
and f score is 
f = 2/(Recall-1 + Precision-1)
where c is the confusion matrix elements.
 Figure 2 shows the diffusion matrixes for OO, AA and WN airlines. In table 2, it is compared precision, recall and f score parameters to determinate the best model. By regarding table 2, the best models are Logistic Regression and Random Forest Classifier for OO and AA airlines respectively. For WN airlines, Logistic Regression, and Random Forest Classifier models give the same recall and precision.

![My image](./images/first_3_histogram.png)
**Figure 2** The diffusion matrix of OO airline (top, left) from Logistic Regression model, AA airline (top, right) from Random Forest Classifier model, WN airline (bottom, left) from Logistic Regression model and WN airline (bottom, right) from Random Forest Classifier model.  0s mentions to distribution 1 and 1s mentions distribution 2.  The yellows, blue areas are percentage of data with the correct predictions in distribution 1 and 2 respectively. The purple areas are percentage of data prediction wrongly  in distribution 1, 2 respectively. 

| | | |Gaussian Naive Bayes|	k-Nearest Neighbors|	Logistic Regression|	Random Forest Classifier|
|-|-|-|-|-|-|-|
|OO Airlines |	Distribution 1|	Recall|	0.44|	0.99|	0.99|	1.00|
| ^  || Precision|	0.93|	0.93	|0.96	|0.95|
| ^	||F %|	59.73|	95.90|	97.48|	97.43|
| ^  |Distribution 2|	Recall|	0.28|	0.71|	0.85|	0.79|
| ^  ||Precision|	0.86|	0.97|	0.99|	1.00|
| ^  ||F %	|42.24|	81.98	|91.46	|88.26|


AA Airlines	
Distribution 1	Recall	0.35	0.97	0.96	0.99
		Precision	0.86	0.90	0.93	0.95
		F %	49.75	93.36	94.46	96.96
	
Distribution 2	Recall	0.84	0.71	0.78	0.86
		Precision	0.33	0.9	0.91	0.99
		F %	47.38	79.37	84.00	92.04


WN Airlines	Distribution 1	Recall	0.41	0.99	0.99	0.99
		Precision	0.93	0.93	0.99	0.99
		F %	56.91	95.90	99.00	99.00
	Distribution 2	Recall	0.85	0.67	0.94	0.94
		Precision	0.25	0.97	0.99	0.99
		F %	38.63	79.25	96.43	96.43



|A |B
|--|--
|<td rowspan="2">3</td>1|2
|4|5
|6|7 

let result = md.render(`
|1|1|3|4|5|
|-|-|-|-|-|
|1|(1|2|2|6) <br> (10|20|20|60) |
|1|1|2|2|7|
|1|4|3|5|5|
`)
## Setup
To run this project, install it locally using npm:

```
$ cd ../lorem
$ npm install
$ npm start
```

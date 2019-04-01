# -*- coding: utf-8 -*-

"""

Created on Wed Jan 16 15:18:46 2019

 

@author: KANGWEN

"""

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import pandas as pd
from pandas import DataFrame
from pandas import concat
import numpy as np
from numpy import array
from numpy import concatenate 
import datetime as dt
from datetime import datetime
from datetime import timedelta

from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import matplotlib.pyplot as plt 


                #### READ RAIN.####

# rain datapoints are every 5 min. Date time concatenated.        

rainheader = ['datetime', 'reading']
pathrain = 'rain.csv'
rain = pd.read_csv(pathrain, names = rainheader, usecols=[0,1], skiprows=2)

# define function to convert 24:00 to 00:00

def my_to_datetime(date_str):
    if date_str[11:13]!= '24':
        return pd.to_datetime(date_str, format = '%Y-%m-%d %H:%M:%S')
    else:
        date_str = date_str[0:10] + '00' + date_str[13:]
        return pd.to_datetime(date_str, format = '%Y-%m-%d %H:%M:%S') + dt.timedelta(days=1)

rain['datetime'] = rain.datetime.apply(my_to_datetime)

                #### READ AND PREPROCESS WATER LEVEL DATA TO ADD RATE.####

levelheader = ['datetime', 'reading']
pathlevel = 'levels.csv'
levelS031 = pd.read_csv(pathlevel, index_col=None, usecols=[0,1], skiprows = 2, names=levelheader)
levelS031['datetime'] = pd.to_datetime(levelS031['datetime'], errors = 'coerce')
levelS031['datetime'] = levelS031['datetime'].dt.strftime('%d/%m/%y %H:%M:%S')
levelS031['datetime'] = pd.to_datetime(levelS031['datetime'], format='%d/%m/%y %H:%M:%S', errors = 'coerce')
levelS031['timediff'] = levelS031['datetime'].diff().dt.total_seconds().div(60)
levelS031.loc[0,'timediff'] = 10
levelS031['rate'] = levelS031['reading'].diff()/levelS031['timediff']
levelS031.loc[0,'rate'] = 0
levelS031 = levelS031.drop(['timediff'], axis=1)

                #### CREATE EVENTS CATALOG ####
# =============================================================================
# #hide this chunk of code after running for the first time
# events = rain
# # drop all timesteps with zero rain
# events = events[events.reading != 0]
# # calculate time difference from prev row
# events['delta'] = events['datetime'].diff()
# # identify time differences > 1 hour - this is to define rain events
# events['counter'] = events['delta'] > dt.timedelta(hours=1)
# # label event ids
# events['eventid'] = events['counter'].cumsum()
# 
# event_catalog_multiheader = events.groupby('eventid').agg({'datetime': ['min', 'max'],
#                                                          'reading': 'sum'})
# event_catalog_multiheader.to_csv('event_catalog_multiheader.csv')
# =============================================================================

catalogheader = ['event_id', 'rain_start', 'rain_end', 'rain_sum']
event_catalog = pd.read_csv('event_catalog_multiheader.csv', names = catalogheader, skiprows=3)
# convert times from string to datetime format
event_catalog['rain_end'] = pd.to_datetime(event_catalog['rain_end'])
event_catalog['rain_start'] = pd.to_datetime(event_catalog['rain_start'])
# create column for rain duration
event_catalog['rain_duration'] = event_catalog['rain_end'] - event_catalog['rain_start']
# drop events which are <30min
event_catalog = event_catalog[event_catalog.rain_duration > dt.timedelta(minutes=30)]
# print(event_catalog)

                #### ALIGN WATER LEVEL DATA.####

invert = 105.25
levelS031['datetime'] = levelS031['datetime'].dt.round('5min')

#arrange in ascending date and descending reading
levelS031.sort_values(['datetime', 'reading'], ascending = [True,False], inplace=True)
#take the highest reading for duplicate values
levelS031 = levelS031.drop_duplicates(subset='datetime',keep='first')
# normalise against the base water level ie invert
levelS031['reading'] = levelS031['reading'] - invert
# parse datetime
levelS031['datetime'] = pd.to_datetime(levelS031['datetime'])

print(levelS031)

#

                #### PREPARE EVENT-LEVEL DATA ####

# The goal of this is to create a list of dataframes, each df representing one event
event_level_data = []
 
# create series of eventids for for loop to iterate over
eventids = event_catalog['event_id']

for row in eventids: # iterate over the rain events one at a time
    # event_temp is a temporary df to contain only the selected event from the catalog
    event_temp = event_catalog[event_catalog['event_id']== row]
    # reset index so that the index is zero, and can be referenced below
    event_temp = event_temp.reset_index()
    # extracting the rain start and end values from event_temp
    rain_start = event_temp.loc[0, 'rain_start']
    rain_end = event_temp.loc[0, 'rain_end']+timedelta(hours=1)
    # drain_temp is a temporary df to contain only the data for the selected event
    drain_temp = levelS031.reset_index(drop=True)
    # now we want to filter the WL data in drain_temp based on the event_temp window
    drain_temp = drain_temp.loc[(drain_temp['datetime']>=rain_start)&(drain_temp['datetime']<=rain_end)]
    diff = drain_temp['reading'].diff()
    absolute = diff.abs()
    total_drain_change = absolute.sum()

    # similarly, filter rain data based on event_temp window

    rain_temp = rain
    rain_temp = rain_temp[(rain_temp['datetime']>=rain_start)&(rain_temp['datetime']<=rain_end)]
   
    #### combine drain and rain data together in a df

    if total_drain_change != 0:
        eventdf = pd.merge(drain_temp, rain_temp, on = 'datetime')
        # append event id to the eventdf
        eventdf['event_id'] = row
        # eventdf = eventdf.drop(eventdf.columns[[0]],axis=1)
        event_level_data.append(eventdf)

    else:
        print(row)
        print('Error!')

print(len(event_level_data))

                #### PREPARE DATA FOR LSTM ####

# Create a function that operates on a single df, and reframes the data as a supervised learning
# dataset for LSTM. This function will be implemented in a for loop on each df in the list.
# n_in is the number of timesteps used for prediction
# n_out is the number of timesteps you want to predict in advance + 1. i.e. if n_out = 2 you want
# to predict 1 timestep in advance.

def time_slicer(eventdf, n_in, n_out, dropnan = True):
    n_vars = eventdf.shape[1]
    df = DataFrame(eventdf)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i)) # this will shift the whole df down and use that data for training
        names += [('var%d(t-%d)'%(j+1,i))for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        names += [('var%d(t+%d)'%(j+1,i))for j in range(n_vars)]
    agg = concat(cols, axis = 1)
    agg.columns = names
    agg['event_id'] = eventdf['event_id']
    if dropnan:
        agg.dropna(inplace = True)
    return agg

data_for_nn = []

for df in event_level_data:
    reframed = time_slicer(df,3,3) # time slicer outputs a df for each event df
    # drop irrelevant columns like time and eventid columns, retaining only prediction timestep and 1 eventid column
    reframed.drop(reframed.columns[[0,4,5,9,10,14,15,19,20,24,29]], axis = 1, inplace = True)
    templabel = reframed['var2(t+2)'] 
    reframed = reframed.drop(['var2(t+2)'], axis=1)
    reframed['var2(t+2)'] = templabel.values
    data_for_nn.append(reframed)

                #### TRAIN-TEST SPLIT ####

# split into train and test sets
train_nn = []
test_nn = []

for df in data_for_nn:

    if ((df['event_id']> 0)&(df['event_id']<= 1607)).any(): 
        df.drop(df.columns[[18]], axis=1,inplace=True) #drop event_ids from training set
        df.drop(df.columns[[15]], axis=1, inplace=True) #drop time from training set
        train_nn.append(df)
    elif ((df['event_id']> 1608)&(df['event_id']<= 2675)).any(): 
        test_nn.append(df)

# concat all the training events
train = pd.concat(train_nn, axis=0)
# normalise training set
values = train.values
scaler_testset = MinMaxScaler(feature_range=(0,1))
scaled = scaler_testset.fit_transform(values)
train = pd.DataFrame(scaled)
# relabel columns
headers = ['level(t-3)', 'rate(t-3)', 'rain(t-3)', 'level(t-2)', 'rate(t-2)', 'rain(t-2)', 'level(t-1)', 'rate(t-1)', 'rain(t-1)', 'level(t)', 'rate(t+1)', 'rain(t)', 'level(t+1)', 'rate(t+1)', 'rain(t+1)', 'rate(t+2)', 'rain(t+2)', 'level_label']
train.columns = headers
#print('printing training set...')
#print(train)

# concat test events.
test = pd.concat(test_nn, axis=0)
testtimes = test['var1(t+2)'] # temporary store for time at prediction. but need to drop cos irrelevant to LSTM
testeventids = test['event_id'] # temporary store for eventids
test.drop(test.columns[[15, 18]], axis=1,inplace=True) # drop time and event_ids from test set

# normalise test set
values = test.values
scaled = scaler_testset.fit_transform(values)
test = pd.DataFrame(scaled)
# relabel columns
test.columns = headers

print('printing test set...')
print(test)


# split into input and outputs
train_X, train_y = train.values[:, :-1], train.values[:, -1]
test_X, test_y = test.values[:, :-1], test.values[:, -1]
# reshape input to 3D array
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

##########################################################################################################
###### BUILD MODEL ######
model = Sequential()
# 50 neurons in the first hidden layer
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
# 1 neuron in the output layer 
model.add(Dense(1, activation = 'relu'))
model.compile(loss='mae', optimizer = 'adam')
history = model.fit(train_X, train_y, epochs = 100, batch_size = 6, validation_data = (test_X, test_y), verbose=2, shuffle=False)

###### PREDICTION ######

# make a prediction. yhat holds the predicted values while test_y holds the actual.
yhat = model.predict(test_X)

# invert scaling for forecast
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
inv_yhat1 = concatenate((test_X, yhat), axis = 1)
inv_yhat2 = scaler_testset.inverse_transform(inv_yhat1)
Prediction = pd.DataFrame(inv_yhat2)
Prediction.columns = headers
# add back the time and eventids
#Prediction['datetime'] = testtimes.values
#Prediction['event_id'] = testeventids.values
inv_yhat2 = inv_yhat2[:, -1]

# invert scaling for actual
test_y = test_y.reshape((len(test_y),1))
inv_y1 = concatenate((test_X, test_y), axis=1)
inv_y2 = scaler_testset.inverse_transform(inv_y1)
Actual = pd.DataFrame(inv_y2)
Actual.columns = headers
# put back the time and eventids
#Actual['datetime'] = testtimes.values
#Actual['event_id'] = testeventids.values
inv_y2 = inv_y2[:,-1]

rmse = sqrt(mean_squared_error(inv_y2, inv_yhat2))
print('Test RMSE: %.3f'% rmse)

###### PLOTS ######

plt.plot(inv_yhat2, label = 'prediction')
plt.plot(inv_y2, label = 'actual')
plt.legend()
plt.show()




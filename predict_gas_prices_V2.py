# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 22:09:13 2022

@author: Arne
"""

import sys
# sys.path.append(r"C:\Users\local_admin\AppData\Roaming\Python\Python39\Scripts/")
sys.path.append(r"C:\Users\Arne\AppData\Roaming\Python\Python39\site-packages")
sys.path.append(r"C:\users\arne\anaconda3\lib\site-packages/")
# from git import Repo
# import git
import pandas as pd
import os
import time
import numpy as np
from tqdm import tqdm
import sklearn as sk
import multiprocessing as mp
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from sklearn.neural_network import MLPRegressor as NN
import multiprocessing as mp
import threading as th
from pandas_datareader import data as datareader
import datetime
# import pyodbc
import os
import time
import pandas as pd
import queue
import sys
from sqlalchemy import create_engine
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from wetterdienst import Wetterdienst, Resolution, Period
import time
from wetterdienst.provider.dwd.forecast import DwdMosmixType
def parse_date(string: str):
    return time.mktime(time.strptime(string[:-3], "%Y-%m-%d %H:%M:%S"))

def reshape_time_float(number):
    t=time.localtime(number)
    return t[0],t[1],t[2],t[3],t[4],t[6]
def reformat_date(series):
    dates=np.zeros((len(series),6))
    for i in range(0, len(series)):
        dates[i]=reshape_time_float(series[i])
    return dates

class Station():
    def __init__(self, uuid, df_stations):
        self.uuid=uuid
        this_station=df_stations[df_stations.uuid == uuid]
        self.latitude=this_station.latitude
        self.longitude=this_station.longitude
        self.brand=this_station.brand
        self.city=this_station.city
        self.all_data=pd.DataFrame([])
        self.time_sampling=300 #time sampling in seconds
    
    def sample_gas_weather(self, df):
        date=df.iloc[0,0][:10]
        start=parse_date(date + " 00:00:00   ")
        end=parse_date(date + " 23:59:59   ")
        index=np.arange(start, end, self.time_sampling)
        df['epoch']=df['date'].apply(parse_date)
        df_new=pd.DataFrame([], index=index, columns=['diesel','e5','e10'])
        for line in range(1, len(df)):
            # print(line)
            t=df.iloc[line].loc['epoch']
            # print(t)
            l=np.where(t<=df_new.index)[0][0]
            # print(l)
            df_new.iloc[l]=df.iloc[line].loc[['diesel','e5','e10']]
            # print(df.iloc[line].loc[['diesel','e5','e10']])
        """use the df with gasoline data and resample it with a given time_sampling"""
        gasData=df_new.copy()
        year, month, day=date.split("-")
        weatherData=self.get_weather(int(year), int(month), int(day))
        
        df_new=pd.merge(gasData, weatherData, left_index=True, right_index=True)
        df_new=df_new.fillna(method='ffill')
        return df_new
        
    #data[['year', 'month', 'day', 'hour', 'minute', 'wday']]=reformat_date(data.tstamp)
    def get_dataFrame(self, data: pd.DataFrame):
        """select the gasoline data for the current station from a csv"""

        # data=data.drop(columns=['e5change','e10change','dieselchange'])
        df=data[data.station_uuid == self.uuid].drop(columns=['station_uuid', 'e5change','e10change','dieselchange'])
        df=self.sample_gas_weather(df)
        df[['year', 'month', 'day', 'hour', 'minute', 'wday']]=reformat_date(df.index.values)
        df['latitude']=self.latitude
        df['longitude']=self.longitude
        return df
        
        
    def get_weather(self, year, month, day):

        latitude=self.latitude
        longitude=self.longitude
        start=datetime(year, month, day)
        end=datetime(year, month, day, 23, 59)
        
        """returns min temp, max temp, rain amount, wind spd, sun hours seems not to work"""
        """prob implement an hourly update for the future"""
        
        from meteostat import Daily, Hourly
        from meteostat import Stations

        stations = Stations()
        stations = stations.nearby(latitude, longitude)##latitude, longitude
        station = stations.fetch(1).reset_index()
        
        data = Hourly(station['id'][0], start, end)
        data = data.fetch()
        data = data.reset_index()
        final=pd.DataFrame([])
        final['date']=data['time']
        # final['tmin']=data['tmin']
        final['temp']=data['temp']
        final['rain']=data['prcp']
        final['wind']=data['wspd'].fillna(value=0)
        
        start=parse_date(str(final.date.iloc[0])+"   ")
        end=start+86399
        index=np.arange(start, end, self.time_sampling)
        df_new=pd.DataFrame([], index=index, columns=['temp','rain','wind'])
        for line in range(0, len(final)):
            # print(line)
            t=parse_date(str(final.iloc[line,0])+"   ")
            # print(t)
            l=np.where(t<=df_new.index)[0][0]
            # print(l)
            df_new.iloc[l]=final.iloc[line].loc[['temp','rain','wind']]
        return df_new


path = r"E:\gasprice/"
station_df=pd.read_csv(r"E:\gasprice/stations.csv")
station=Station("51d4b4f2-a095-1aa0-e100-80009459e03a", station_df)
    
    

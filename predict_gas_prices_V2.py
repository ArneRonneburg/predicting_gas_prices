# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 22:09:13 2022

@author: Arne
"""

import sys
sys.path.append(r"C:\Users\local_admin\AppData\Roaming\Python\Python39\Scripts/")
sys.path.append(r"D:\Profile\kyq\Anwendungsdaten\Python\Python39\Scripts/")
sys.path.append(r"D:\Profile\kyq\Anwendungsdaten\Python\Python39\site-packages/")
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



class station():
    def __init__(self, uuid, df_stations, path_to_data):
        self.uuid=uuid
        this_station=df_stations[df_stations.uuid == uuid]
        self.latitude=this_station.latitude
        self.longitude=this_station.longitude
        self.brand=this_station.brand
        self.city=this_station.city
            
    def get_gasoline_data(self, n_days_back):
        date=time.m
        
        
    def get_data_specific_day(self, year,month,day, path):
        year_=str(year)
        month_=(str(month) if len(str(month))==2 else "0"+str(month))
        day_=(str(day) if len(str(day))==2 else "0"+str(day))
        fname=year_+"-"+month_+"-"+day_+"-"+"prices.csv"
        df=pd.read_csv(path + year_ + "/" + month_ + "/"+fname)
        df=df[df.station_uuid==self.uuid]
        df.drop(columns=['e5change','e10change','dieselchange'], inplace=True)
        # df['date']=df['date'].apply(self.reshape_date)
        
        return df
        
    
    def get_weather(self, n_days_back):
        from wetterdienst import Wetterdienst, Resolution, Period
        import time
        from wetterdienst.provider.dwd.forecast import DwdMosmixType

        latitude=self.latitude
        longitude=self.longitude
        
        def get_weather_past(start_date, end_date):
            
            """returns min temp, max temp, rain amount, wind spd, sun hours seems not to work"""
            from meteostat import Daily
            from meteostat import Stations

            stations = Stations()
            stations = stations.nearby(latitude, longitude)##latitude, longitude
            station = stations.fetch(1).reset_index()
            
            data = Daily(station['id'][0], start_date, end_date)
            data = data.fetch()
            data = data.reset_index()
            final=pd.DataFrame([])
            final['date']=data['time']
            final['tmin']=data['tmin']
            final['tmax']=data['tmax']
            final['rain']=data['prcp']
            final['wind']=data['wspd'].fillna(value=0)
            
            
            return final

        def get_nearby(stations, lat, long):
            
            a=np.array(stations.latitude - lat)**2 + np.array(stations.longitude-long)**2
            index=np.argmin(a)
            return [stations.iloc[index].station_id]

        def get_weather_forecast(lat, long):
            """for each day in the dataset, get max temp, min temp, max wind, rain amount for the next days"""
            
            #check if there is a time gap /etc between historical data and forecast
            #check the units of wind speed and rain and compare to meteostat stuff
            API = Wetterdienst(provider="dwd", kind="forecast")
            params=['temperature_air_200','wind_speed', 'precipitation_consist_last_1h']
            stations = API(parameter='large', mosmix_type=DwdMosmixType.LARGE)
            stationen=stations.all().df
            ids=get_nearby(stationen, lat, long)
            forecast_stations = API(parameter=params, mosmix_type=DwdMosmixType.LARGE).filter_by_station_id(station_id=ids)
            forecast=forecast_stations.values.all().df
            rain=forecast[forecast.parameter=='precipitation_consist_last_1h']
            temp=forecast[forecast.parameter=='temperature_air_200']
            wind=forecast[forecast.parameter=='wind_speed']
            year=time.localtime()[0]
            month=time.localtime()[1]
            day=time.localtime()[2]
            time_int=np.zeros(len(rain))
            for i in range(0, len(rain.date)):
                time_int[i]=time.mktime(rain.iloc[i, 3].timetuple())+3600##since the data is in UTC
            today=time.mktime(time.strptime(str(year)+"-"+str(month) + "-"+str(day), "%Y-%m-%d"))
            final=pd.DataFrame([], columns=["date", "rain", "tmin", "tmax", "wind"])
            
            cur_day=today
            for j in range(0, 3):
                cur_day=cur_day+86400
                rainval=0
                temps=[]
                windspd=[]    
                for i in range(0, len(time_int)):
                    
                    if time_int[i]>cur_day and time_int[i]<cur_day+3600*24:
                        rainval=rainval+rain.iloc[i].value
                        temps.append(temp.iloc[i].value)
                        windspd.append(wind.iloc[i].value)
                tba=pd.DataFrame([], columns=["date", "rain", "tmin", "tmax", "wind"])
                tba['date']=[cur_day]
                tba['rain']=[rainval]
                tba['tmin']=[np.min(temps)-273.16]
                tba['tmax']=[np.max(temps)-273.16]##dwd uses kelvin
                tba['wind']=[np.max(windspd)]
                final=final.append(tba, ignore_index=True)
                
            return final


        def add_weather(n_days_ago):
            
            """adds the weather data for the chosen gasoline station to the dataframe. Adds min and max temp, amound of rain and wind speed"""
            lat=self.latitude
            long=self.longitude
        
            hundred_days_ago=time.time()-86400*int(n_days_ago)
            start_time=datetime(time.localtime(hundred_days_ago)[0], time.localtime(hundred_days_ago)[1], time.localtime(hundred_days_ago)[2])
            past_weather=get_weather_past(lat, long, start_time)
            future_weather=get_weather_forecast(lat,long)
            for i in range(0, len(past_weather)):    
                past_weather.date.iloc[i]=time.mktime(past_weather.date.iloc[i].timetuple())
            weather=pd.concat((past_weather, future_weather), ignore_index=True)
            times=np.transpose(self.epoch_to_date(np.asarray(weather.date)))
        
            weather['year']=times[0]
            weather['month']=times[1]
            weather['day']=times[2]
            weather=weather.drop(columns='date')
            data=pd.DataFrame([])
            data['tmin']=0
            data['tmax']=0
            data['wind']=0
            data['rain']=0
            for i in range(0, len(weather)):
                #get rid of errors here...
                cday=weather.day.iloc[i]
                cmonth=weather.month.iloc[i]
                cyear=weather.year.iloc[i]
                index=list(data[data.day==cday][data.month==cmonth][data.year==cyear].index)
                data['tmin'][index]=float(weather.iloc[i].tmin)
                data['tmax'][index]=float(weather.iloc[i].tmax)
                data['wind'][index]=float(weather.iloc[i].wind)
                data['rain'][index]=float(weather.iloc[i].rain)
                
            return data
    
    
    
        weather_data=add_weather(n_days_back)
        return weather_data
    
    
    

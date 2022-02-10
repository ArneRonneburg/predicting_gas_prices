# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 22:09:13 2022

@author: Arne
"""


from git import Repo
import git
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
            
    def get_data(self, year,month,day, path):
        year_=str(year)
        month_=(str(month) if len(str(month))==2 else "0"+str(month))
        day_=(str(day) if len(str(day))==2 else "0"+str(day))
        fname=year_+"-"+month_+"-"+day_+"-"+"prices.csv"
        df=pd.read_csv(path + year_ + "/" + month_ + "/"+fname)
        df=df[df.station_uuid==self.uuid]
        df.drop(columns=['e5change','e10change','dieselchange'], inplace=True)
        # df['date']=df['date'].apply(self.reshape_date)
        
        return 
        
    
    def reshape_date(self, string, form='%Y-%m-%d %H:%M:%S+02'):
        return time.mktime(time.strptime(string, form))
    def interpolate_timesteps(self, df,  form='%Y-%m-%d %H:%M:%S'):
        date=df.iloc[0].date
        t0=time.mktime(time.strptime(date[:11]+"00:00:05", form))
        t1=time.mktime(time.strtime(date[:11]+"23:59:55", form))
        tlist=np.arange(t0, t1+3,300, endpoint=True)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    def get_weather(self, day):
        
        pass
    
    

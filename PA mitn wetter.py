# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 22:33:21 2022

@author: Arne
"""
from wetterdienst import Wetterdienst, Resolution, Period
import time
from wetterdienst.provider.dwd.forecast import DwdMosmixType
from wetterdienst.provider.dwd.observation import DwdObservationParameter
from wetterdienst.provider.dwd.observation import DwdObservationRequest, DwdObservationDataset, DwdObservationPeriod, DwdObservationResolution
import numpy as np
import pandas as pd
from datetime import datetime
start=datetime(2022, 3, 20)
stop=datetime(2022, 4, 8)
def get_nearby(stations, lat, long):
    
    a=np.array(stations.latitude - lat)**2 + np.array(stations.longitude-long)**2
    index=np.argmin(a)
    return [stations.iloc[index].station_id]

latitude = 52.416914
longitude = 13.019 
API = Wetterdienst(provider="dwd", kind="forecast")
params=['temperature_air_mean_200', 'wind_speed', 'precipitation_height_significant_weather_last_1h', 'probability_precipitation_liquid_last_12h']
stations = API(parameter='large', mosmix_type=DwdMosmixType.LARGE)
stationen=stations.all().df
ids=get_nearby(stationen, latitude, longitude)
forecast_stations = API(parameter=params, mosmix_type=DwdMosmixType.LARGE).filter_by_station_id(station_id=ids)
forecast=forecast_stations.values.all().df

       
            
"""returns min temp, max temp, rain amount, wind spd, sun hours seems not to work"""
from meteostat import Daily
from meteostat import Stations

stations = Stations()
stations = stations.nearby(latitude, longitude)##latitude, longitude
station = stations.fetch(1).reset_index()

data = Daily(station['id'][0], start, stop)
data = data.fetch()
data = data.reset_index()
final=pd.DataFrame([])
final['date']=data['time']
final['tmin']=data['tmin']
final['tmax']=data['tmax']
final['rain']=data['prcp']
final['wind']=data['wspd'].fillna(value=0)

    
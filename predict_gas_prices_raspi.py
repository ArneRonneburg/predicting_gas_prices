# -*- coding: utf-8 -*-
"""
Created on Sat May 29 13:54:54 2021

@author: Arne Ronneburg & Markus Göhler
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


def start_process(data):
    nr, prices,stations, uuids, holidays, model=data
    uu=list(uuids.keys())[nr]
    final=pd.DataFrame([])
    for fuel in ['diesel', 'e5']:
        uuid=uuids[uu]
        data=get_price_data(prices, uuid, fuel)




        data=add_holidays(data, holidays)
        
        
        
        
        ###################################################################
        data=add_weather(data, stations, uuid) #implement here...but also think about implementing in the prediction..
        #####################################################################
        
        
        to_be_shown=pd.DataFrame([], columns=['date', 'prediction', 'lowerCI', 'upperCI', 'observation'])

        CI=0
        for i in [7,6,5,4,3,2]:
            #print(i)
            prev_pred=pd.DataFrame([], columns=['date','prediction','lowerCI', 'upperCI', 'observation'])
            X=data[:-i*288].drop(columns='price')
            y=data[:-i*288].price
            
            model.fit(X,y)
            pred=model.predict(data[-i*288:-(i-1)*288].drop(columns='price'))
            prev_pred['observation']=data[-i*288:-(i-1)*288].price
            prev_pred['prediction']=pred
            X2=data[-i*288:-(i-1)*288].drop(columns='price')
            prev_pred['date']=X2['day'].astype(str)+"-"+X2['month'].astype(str)+"-"+X2['year'].astype(str) + " "+(X2['time']/60).astype(int).astype(str)+":"+(X2['time']%60).astype(str)
            
            #previous_predictions.append([data[-i*288:-(i-1)*288].drop(columns='price'), pred, data[-i*288:-(i-1)*288].price])
            to_be_shown=pd.concat([to_be_shown, prev_pred])
            CI=CI+np.asarray(abs(prev_pred['observation']-prev_pred['prediction']))
        
        i=1    
        prev_pred=pd.DataFrame([], columns=['date','prediction','lowerCI', 'upperCI', 'observation'])
        X=data[:-i*288].drop(columns='price')
        y=data[:-i*288].price
        ####seems to work. reformulate, so it is appended to a pd df
        
        ###then add prediction + an uncertainty?
        
        model.fit(X,y)
        pred=model.predict(data[-288:].drop(columns='price'))
        prev_pred['observation']=data[-i*288:].price
        prev_pred['prediction']=pred
        X2=data[-i*288:].drop(columns='price')
        prev_pred['date']=X2['day'].astype(str)+"-"+X2['month'].astype(str)+"-"+X2['year'].astype(str) + " "+(X2['time']/60).astype(int).astype(str)+":"+(X2['time']%60).astype(str)
        CI=CI+np.asarray(abs(prev_pred['observation']-prev_pred['prediction']))
        #previous_predictions.append([data[-i*288:-(i-1)*288].drop(columns='price'), pred, data[-i*288:-(i-1)*288].price])
        to_be_shown=pd.concat([to_be_shown, prev_pred])
        #previous_predictions.append([data[-288:].drop(columns='price'), pred, np.asarray(data[-288:].price)])
        CI=np.concatenate((CI/7,CI/7,CI/7))###take the average error between prediction and observation as confidence interval
        
        
        
        
        
        X=data.drop(columns='price')
        y=data['price']
        model.fit(X,y)
        
        current_time=time.localtime()
        date_now=str(current_time[2])+"-"+str(current_time[1])+"-"+str(current_time[0])
        next_days=the_next_days(date_now, 3, holidays)###create X for next three days
        
        
        ##################################################################################
        next_days=add_weather(next_days, stations, uuid)#####implement here but also in the past
        ##################################################################################
        
        prediction=model.predict(next_days)
        
        confidence_interval=np.zeros(len(prediction))
        Prediction=pd.DataFrame([], columns=['date', 'prediction', 'lowerCI', 'upperCI', 'observation'])
        Prediction['date']=next_days['day'].astype(str)+"-"+next_days['month'].astype(str)+"-"+next_days['year'].astype(str)+ " "+(next_days['time']/60).astype(int).astype(str)+":"+(next_days['time']%60).astype(str)
        
        
        Prediction["prediction"]=prediction
        Prediction['lowerCI']=prediction - CI
        Prediction['upperCI']=prediction + CI
        # Prediction['lowerCI']=confidence_interval
        # Prediction['upperCI']=confidence_interval
        
        to_be_shown=pd.concat([to_be_shown, Prediction])
        # Prediction['Data']=np.ones(len(prediction))
        past=pd.DataFrame([])
       
        # next_days.to_csv(path +"predictions/"+ date_now+".txt")
        to_be_shown.reset_index()
        column_title=str(stations[stations.uuid==uuid].brand.iloc[0])+","+str(stations[stations.uuid==uuid].street.iloc[0])+","+str(stations[stations.uuid==uuid].city.iloc[0])+"/"+str(fuel)+"-date"
        final[column_title]=np.asarray(to_be_shown.date)
        column_title=str(stations[stations.uuid==uuid].brand.iloc[0])+","+str(stations[stations.uuid==uuid].street.iloc[0])+","+str(stations[stations.uuid==uuid].city.iloc[0])+"/"+str(fuel)+"-prediction"
        final[column_title]=np.asarray(to_be_shown.prediction)
        column_title=str(stations[stations.uuid==uuid].brand.iloc[0])+","+str(stations[stations.uuid==uuid].street.iloc[0])+","+str(stations[stations.uuid==uuid].city.iloc[0])+"/"+str(fuel)+"-lowerCI"
        final[column_title]=np.asarray(to_be_shown.lowerCI)
        column_title=str(stations[stations.uuid==uuid].brand.iloc[0])+","+str(stations[stations.uuid==uuid].street.iloc[0])+","+str(stations[stations.uuid==uuid].city.iloc[0])+"/"+str(fuel)+"-upperCI"
        final[column_title]=np.asarray(to_be_shown.upperCI)
        column_title=str(stations[stations.uuid==uuid].brand.iloc[0])+","+str(stations[stations.uuid==uuid].street.iloc[0])+","+str(stations[stations.uuid==uuid].city.iloc[0])+"/"+str(fuel)+"-observation"
        final[column_title]=np.asarray(to_be_shown.observation)
    #q.put(final)
    return final


def get_gas_prices(liste):
    for i in liste:
        yield pd.read_csv(i)
        
def update_repo(path_git, path_os):
    os.makedirs(path_os, exist_ok=True)
#path_os=r"D:\hdd\gas_prices\data"
#data=Repo.clone_from(path_git, path_os) #once it is downloaded, we can use the local data?
    data=git.Repo(path_os, odbt=git.GitCmdObjectDB)
    o=data.remotes.origin
    new_stuff=o.pull()###get the latest data in the repository
    return "ready"


def get_station_list(liste, path_os):###
    """insert a list of postal codes here...maybe using a list of uuids would be a nice idea"""
    gas_stations=pd.DataFrame([])
    path=path_os+'stations/stations.csv'
    stations=pd.read_csv(path)##this contains the different stations with city, uuuid, coordinates, brand, ...I'd stick to the coordinates. but this gives the brand attribute for free
    for i in range(0, len(stations)):
        # if type(stations['post_code'][i])==str:
        if stations['uuid'][i] in liste:
            gas_stations=gas_stations.append(stations.iloc[i])
    return gas_stations



def get_station_info(data, liste, col):
    """find a list of stations with given properties liste in col, e.g. a list of stations which have a postal code given in liste"""
    
    df=pd.DataFrame([], columns=data.columns)
    for i in range(0, len(data)):
        if data[col][i] in liste:
            df=df.append(data.iloc[i])
    return df
#year/month/year-month-day.csv
def get_pricelist(uuid_liste, path_os, starting_time, csize):
    
    """plz list is the list of PLZ which is interesting fpr the gas stations. maybe replace by UUID?
    path_os is the path of the repo on the local storage
    starting time is the year where the data starts. Acc to some exps. one week shall be enough...
    so maybe take one year and reduce later
    csize is the number of chunks to scan the gasprices...higher numbers should mean faster but more often adding up lists
    """
    
    pricelist=[]
    path_prices=path_os+"/prices/"
    tree=list(os.walk(path_prices))
    for i in tree:
        if len(i[2]) > 0:
            for j in i[2]:
                #print(time.strptime(j[:10],"%Y-%m-%d"))
                if time.mktime(time.strptime(j[:10],"%Y-%m-%d"))>starting_time:
                    pricelist.append(i[0]+"/"+j)

    gas=get_gas_prices(pricelist)##this is a generator which opens the csv file and returns it as a pd.df. 

    gas_stations=get_station_list(uuid_liste, path_os)
   
    
    prices_all=pd.DataFrame([], columns=['date', 'station_uuid', 'diesel', 'e5', 'e10'])
    
    for j in range(0, csize):
        prices=pd.DataFrame([], columns=['date', 'station_uuid', 'diesel','e5', 'e10'])
    
        for i in tqdm(range(int(len(pricelist)/csize*j), int(len(pricelist)/csize*(j+1)))):
            data=next(gas)
            prices=prices.append(data[data['station_uuid'].isin(uuid_liste)][['date', 'station_uuid', 'diesel', 'e5', 'e10']])
        prices_all=prices_all.append(prices)
    return prices_all




def epoch_to_date(x):
    a=np.zeros((len(x), 5))
    for i in range(0, len(x)):
        t=time.localtime(x[i])
        a[i]=t.tm_year, t.tm_mon, t.tm_mday, t.tm_wday, t.tm_hour*60+t.tm_min
    return a

def modify_timestamp(x, timestamp="%Y-%m-%d %H:%M:%S"):
    """year, month, day, weekday, hour, min"""
    dates=[]
    for a in range(0, len(x)):
        i=x.iloc[a]
        t=time.strptime(i[:-3], timestamp)
        dates.append([t.tm_year, t.tm_mon, t.tm_mday, t.tm_wday, t.tm_hour, t.tm_min])
    return np.transpose(dates)
    

def get_epoch(x, timestamp="%Y-%m-%d %H:%M:%S"):
    dates=[]
    for a in range(0, len(x)):
        i=x.iloc[a]
        t=time.mktime(time.strptime(i[:-3], timestamp))
        dates.append(t)
    return dates

def time_sampling(data, tmin, tmax, delta_t):
       
    df=pd.DataFrame([], columns=['date', 'price'])
    df.loc[:,'date']=np.arange(tmin, tmax+delta_t, delta_t)
    for i in range(0, len(data)):
        index=np.where(df.date > data.iloc[i, 0])[0][0]#########check here/optimize
        df.iloc[index, 1]=data.iloc[i,-1]##the price is the last column
    return df


def transform_epoch_to_date(data_sql):
    data=pd.DataFrame([], columns=['date','price'])
    data['date']=data_sql['date']
    data['price']=data_sql['price']
    
    
    times=np.transpose(epoch_to_date(np.asarray(data.date)))

    data['year']=times[0]
    data['month']=times[1]
    data['day']=times[2]
    data['wday']=times[3]
    data['time']=times[4]
    
    data=data.drop(columns='date')
    data['year']=data['year'].astype(int)
    data['month']=data['month'].astype(int)
    data['day']=data['day'].astype(int)
    data['wday']=data['wday'].astype(int)
    data['time']=data['time'].astype(int)
    return data


def add_holidays(data, holidays):
    hday=np.zeros(len(data))
    for i in range(0, len(data)):
        datum=data.iloc[i]
        stamp=str(int(datum.year))
        for d in [int(datum.month), int(datum.day)]:
            if len(str(d))==1:
                stamp=stamp+"-0"+str(d)
            else:
                stamp=stamp+"-"+str(d)
        hday[i]=holidays[holidays.Date ==stamp].iloc[:,-1]
    data['holiday']=hday
    return data
            

def the_next_days(datum, n_days, holidays):
    """put in the date of the day where the prediction starts in the format 01.01.2021. 
    N_days- should be self-explaining, right?"""
    tstart=time.mktime(time.strptime(datum+";00:00:00", "%d-%m-%Y;%H:%M:%S"))
    t=np.arange(tstart, tstart+n_days*24*60*60, 300)
    times=np.transpose(epoch_to_date(np.asarray(t)))
    X=pd.DataFrame(np.transpose(times), columns=['year','month','day','wday', 'time'], dtype=int)
    X=add_holidays(X, holidays)
    return X

def the_past_days(datum, n_days, holidays):
    """put in the date of the day where the prediction starts in the format 01.01.2021. 
    N_days- should be self-explaining, right?"""
    tstart=time.mktime(time.strptime(datum+";00:00:00", "%d-%m-%Y;%H:%M:%S"))
    t=np.arange(tstart-n_days*24*60*60, tstart, 300)
    times=np.transpose(epoch_to_date(np.asarray(t)))
    X=pd.DataFrame(np.transpose(times), columns=['year','month','day','wday', 'time'])
    X=add_holidays(X, holidays)
    return X

def get_price_data(prices, uuid, fueltype):
    #something here is creating errors
    data_sql=prices[prices.station_uuid==uuid].copy()###select a 
    data_sql.loc[:,'date']=get_epoch(data_sql.date)
    data_sql['price']=data_sql[fueltype]
    current_time=time.localtime()
    date_now=str(current_time[2])+"-"+str(current_time[1])+"-"+str(current_time[0])
    now=time.mktime(time.strptime(date_now, "%d-%m-%Y"))
    data_resampled=time_sampling(data_sql, now-3600*24*7*12, data_sql.iloc[-1, 0], 300)###this one
    data_resampled=data_resampled.fillna(method='ffill').drop(index=0)
    
    data=transform_epoch_to_date(data_resampled)
    return data

from datetime import datetime
def get_weather_past(latitude, longitude, start_date, end_date=datetime.now()):
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

##x=get_weather_past(lat, long, start)


def get_nearby(stations, lat, long):
    
    a=np.array(stations.latitude - lat)**2 + np.array(stations.longitude-long)**2
    index=np.argmin(a)
    return [stations.iloc[index].station_id]

from wetterdienst import Wetterdienst, Resolution, Period
import time
from wetterdienst.provider.dwd.forecast import DwdMosmixType


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

def add_weather(data, stations, uuid):
    
    
    lat=float(stations[stations.uuid==uuid].latitude)
    long=float(stations[stations.uuid==uuid].longitude)
    """past data = 
    forecast data = 
    add_data via timestamp"""
    ten_days_ago=time.time()-86400*100
    start_time=datetime(time.localtime(ten_days_ago)[0], time.localtime(ten_days_ago)[1], time.localtime(ten_days_ago)[2])
    past_weather=get_weather_past(lat, long, start_time)
    future_weather=get_weather_forecast(lat,long)
    for i in range(0, len(past_weather)):    
        past_weather.date.iloc[i]=time.mktime(past_weather.date.iloc[i].timetuple())
    weather=pd.concat((past_weather, future_weather), ignore_index=True)
    times=np.transpose(epoch_to_date(np.asarray(weather.date)))

    weather['year']=times[0]
    weather['month']=times[1]
    weather['day']=times[2]
    weather=weather.drop(columns='date')
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




if __name__=="__main__":
    
    
    path_git="https://dev.azure.com/tankerkoenig/_git/tankerkoenig-data"    ##this is where the gas price data comes from
    # path="/home/pi/gasprice/" #path of the project
    path="/home/pi/gasprice/" #path of the project
    # path_os="/srv/dev-disk-by-uuid-2C165F5E165F27DA/gasprice_data/"#local path of gasprice data
    # path_json="/home/pi/gasprice/" #a key file for the upload of the results to google sheets
    holidays=pd.read_csv(path + "holidays.txt") #just a list of the holidays in the next years
    path_mail=path
    path_os=path+"data/"
    path_json="/home/pi/gasprice/" #a key file for the upload of the results to google sheets
    # branch_prices=os.listdir(path_prices)
    # while True:
    uuids={'jet_potsdam':'51d4b4f2-a095-1aa0-e100-80009459e03a',#, '14469'
    'aral_potsdam':'3e17a2ae-db29-425f-895d-f841d817309a',#, '14469'
    'aral_Waldstadt': 'a08e4d7e-8159-4f85-a299-0885478c8186',#14478
    'shell_potsdam':'140b41f2-fc48-4c08-8461-68be5dc1b491',#, '14467'
    'total_nauen':'15a909b6-6361-49e8-95ea-e1b274553b64',#,'14641'
    'hamburg_cleancar':'512f9ee3-77cf-4719-f51a-b837c985f035',#22309
    'hamburg_jet1':'cd8ba6a6-8316-1ed5-a3ae-d3139800d85f',#22309
    'hamburg_jet2':'51d4b540-a095-1aa0-e100-80009459e03a',#22309
    'hamburg_shell':'cfaf5e3c-60ee-4a11-a6c5-5b359a67dded',#22309
    'wk1':'13b50ae5-ff10-43c5-a302-b4411991ca16',#16909
    'wk2':'3d0b411c-730b-4095-845d-be27b59f40b2',#16909
    'wk3':'a2380ebc-1b74-4ac0-9db0-9a90512513d6',#16909
    'wk4':'a81571ad-9a8e-4283-aa2c-d611d82e437a',#16909
    'wk5':'aa1e903e-7da9-4d9c-9405-a8076e846e39',#16909
    'wk6':'e1a15081-2549-9107-e040-0b0a3dfe563c',#16909
    'wk7':'9a29ac22-73bc-409a-89f7-0e73a9fe0327',#16909
    'lalendorf':'005056ba-7cb6-1ed2-bceb-60191af70d1b',#18279,
    'Aral_Teterow':'83393259-40b4-48d0-8029-140ec2e015ff'#17166
    }
    uuid_liste=[]
    for i in uuids:
        uuid_liste.append(uuids[i])
    
    for runs in range(0, 3):    
        try:
        
            
            update_repo(path_git, path_os)  #this pulls the newest data from the tankerkoenig repo
            stations=pd.read_csv(path_os+'stations/stations.csv')#here the list of gas stations is read
            ##this is the list of gasoline stations for which a prediction is created
            
            # plzliste=['14469', '14467','14478', '16909','14641', '22309', '18279']
            prices=get_pricelist(uuid_liste, path_os, time.mktime(time.strptime("2021-01-01","%Y-%m-%d")), 1)
            ###create the pricelist for the given list of uuids - starting from 01.01.2021
            
            ###add the data transformation now to the final timestamping with 5 minutes and so on. Then introduce the model. 
            
            time.sleep(10)
            model=ExtraTreesRegressor(n_estimators=100)##this is our model. DONT USE MULTIPROCESSING HERE!!!!
            final=pd.DataFrame([])##the final dataframe
            
            os.makedirs(path +"predictions/", exist_ok=True)#create the folder for the result file, if does not exist yet
            
            pool=mp.Pool()        #pool for parallel execution of the predictions for the different gas stations
            
            processdata=[]###just a list for the data for the different processes.
            for nr in range(0, len(uuids)):
                print(nr)
                processdata.append([nr, prices,stations, uuids, holidays, model])
            result=pool.map(start_process, processdata)###multiprocessing of the predictions.
            
                #thrd=th.Thread(target=start_thread, args=(nr, prices,stations, uuids, holidays, model, q))
                #thrd.start()###oldversion.....try with threads
            final=pd.DataFrame()###the final dataframe with the final data....wide format
            for i in range(0, len(result)):
                res=result[i] ###the different gas stations
                for c in res.columns:
                    final[c]=np.asarray(res[c])
            
            final.to_csv(path +"predictions/latest_width.txt", index=False)
            tba=pd.DataFrame([], columns=["date","prediction","lowerCI","upperCI","observation","station","fueltype"])
            
            #fuel='diesel','e5
            ###rearrange from wide format to long format
            for  uu in uuids:
                for fuel in ['diesel', 'e5']:
                    uuid=uuids[uu]
                    
                    # column_title=str(stations[stations.uuid==uuid].brand)+","+str(stations[stations.uuid==uuid].street)+","+str(stations[stations.uuid==uuid].city)+"/"+str(fuel)+"-date"
                    column_title=str(stations[stations.uuid==uuid].brand.iloc[0])+","+str(stations[stations.uuid==uuid].street.iloc[0])+","+str(stations[stations.uuid==uuid].city.iloc[0])+"/"+str(fuel)+"-date"
                    DATE=final[column_title]
                    # column_title=str(stations[stations.uuid==uuid].brand)+","+str(stations[stations.uuid==uuid].street)+","+str(stations[stations.uuid==uuid].city)+"/"+str(fuel)+"-prediction"
                    column_title=str(stations[stations.uuid==uuid].brand.iloc[0])+","+str(stations[stations.uuid==uuid].street.iloc[0])+","+str(stations[stations.uuid==uuid].city.iloc[0])+"/"+str(fuel)+"-prediction"
                    PRED=final[column_title]
                    column_title=str(stations[stations.uuid==uuid].brand.iloc[0])+","+str(stations[stations.uuid==uuid].street.iloc[0])+","+str(stations[stations.uuid==uuid].city.iloc[0])+"/"+str(fuel)+"-lowerCI"
                    LCI=final[column_title]
                    column_title=str(stations[stations.uuid==uuid].brand.iloc[0])+","+str(stations[stations.uuid==uuid].street.iloc[0])+","+str(stations[stations.uuid==uuid].city.iloc[0])+"/"+str(fuel)+"-upperCI"
                    UCI=final[column_title]
                    # column_title=str(stations[stations.uuid==uuid].brand)+","+str(stations[stations.uuid==uuid].street)+","+str(stations[stations.uuid==uuid].city)+"/"+str(fuel)+"-observation"
                    column_title=str(stations[stations.uuid==uuid].brand.iloc[0])+","+str(stations[stations.uuid==uuid].street.iloc[0])+","+str(stations[stations.uuid==uuid].city.iloc[0])+"/"+str(fuel)+"-observation"
                    OBS=final[column_title]
                    appender=pd.DataFrame([], columns=["date","prediction","lowerCI","upperCI","observation","station","fueltype"])
                    appender['date']=DATE
                    appender['prediction']=PRED ##predicted tat
                    appender['lowerCI']=LCI     #confidence interval
                    appender['upperCI']=UCI     #confidence interval
                    appender['observation']=OBS #the observed data
                    appender['fueltype']=[fuel]*len(PRED) #fueltype
                    appender['station']=str(stations[stations.uuid==uuid].brand.iloc[0])+","+str(stations[stations.uuid==uuid].street.iloc[0])+","+str(stations[stations.uuid==uuid].city.iloc[0])
                    
                    tba=tba.append(appender)
            tba.to_csv(path +"predictions/latest_long.txt", index=False)
            
            ###this file is uploaded and visualized in tableau
            #https://public.tableau.com/app/profile/markus.g.hler7472/viz/Gasprice_16236148421350/Spritpreis
            ###
            
                
            import gspread # -- pip install gspread
            from oauth2client.service_account import ServiceAccountCredentials # -- pip install oauth2client
            import csv   
            sheet = "data"
            csv_data = path + "predictions/latest_long.txt"
            scope =["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_name(path_json+"gas-price-304619-7d6023c81c41.json", scope)
            client = gspread.authorize(creds)
            
            spreadsheet = client.open("gasprice")
            
            
            spreadsheet.values_update(
                sheet,
                params={'valueInputOption': 'USER_ENTERED'},
                body={'values': list(csv.reader(open(csv_data)))}
            )
            ###implement mail reporting
            
        
            sender_address = "gasprice90@gmail.com"
            receiver_address = "arne.ronneburg@googlemail.com"
            
            mail_content = "Script was executed, prediction was updated. Greetings, your raspi."   
            message = MIMEMultipart()
            message['From'] = sender_address
            message['To'] = receiver_address
            message['Subject'] = 'Update of gasprice successful'   #The subject line
            #The body and the attachments for the mail
            message.attach(MIMEText(mail_content, 'plain'))
            #Create SMTP session for sending the mail
            session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
            session.starttls() #enable security
            credentials=open(path_mail + "cred.txt").readlines()
            session.login(sender_address, credentials[0]) 
        
            text = message.as_string()
            session.sendmail(sender_address, receiver_address, text)
            session.quit()
            break
        except Exception as e:
            sender_address = "gasprice90@gmail.com"
            receiver_address = "arne.ronneburg@googlemail.com"
            errorcode=sys.exc_info()
            mail_content = "There was an error:"+str(errorcode)   
            message = MIMEMultipart()
            message['From'] = sender_address
            message['To'] = receiver_address
            message['Subject'] = 'Error during updating gasprice successful'   #The subject line
            #The body and the attachments for the mail
            message.attach(MIMEText(mail_content, 'plain'))
            #Create SMTP session for sending the mail
            session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
            session.starttls() #enable security
            credentials=open(path_mail+"cred.txt").readlines()
            session.login(sender_address, credentials[0]) 
        
            text = message.as_string()
            session.sendmail(sender_address, receiver_address, text)
            session.quit()
    
    #run once per day
    
    #logging library
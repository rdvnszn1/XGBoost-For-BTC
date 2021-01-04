import requests
from math import log
import pickle
import pandas as pd
import datetime
import time

input_data=pd.DataFrame(columns=['5min_open+1', '5min_open_change', '5min_high+1', '5min_high_change',
       '5min_low+1', '5min_low_change', '5min_close+1', '5min_close_change',
       '15minOpen+1', '15minOpen_change+1', '15minHigh+1',
       '15minHigh_change+1', '15minLow+1', '15minLow_change+1', '15minClose+1',
       '15minClose_change+1', '30minOpen+1', '30minOpen_change+1',
       '30minHigh+1', '30minHigh_change+1', '30minLow+1', '30minLow_change+1',
       '30minClose+1', '30minClose_change+1', '60minOpen+1',
       '60minOpen_change+1', '60minHigh+1', '60minHigh_change+1', '60minLow+1',
       '60minLow_change+1', '60minClose+1', '60minClose_change+1',
       '240minOpen+1', '240minOpen_change+1', '240minHigh+1',
       '240minHigh_change+1', '240minLow+1', '240minLow_change+1',
       '240minClose+1', '240minClose_change+1'],
      dtype='object')

def take_data(timeframe,limit,ticker="BTCUSDT"):

    api_url = "https://api.binance.com/api/v1/klines?symbol={}&interval={}&limit={}".format(ticker,timeframe,limit)
    json_data=requests.get(api_url).json()

    return json_data


def take_features(timeframe):
    data = take_data(timeframe, 2)

    open= float(data[0][1])
    open_lag=float(data[1][1])
    open_change =  log(open)-log(open_lag)

    high= float(data[0][2])
    high_lag=float(data[1][2])
    high_change =  log(high)-log(high_lag)


    low= float(data[0][3])
    low_lag=float(data[1][3])
    low_change =  log(low)-log(low_lag)


    close= float(data[0][4])
    close_lag=float(data[1][4])
    close_change =  log(close)-log(close_lag)



    return  [open,open_change,high,high_change,low,low_change,close,close_change]


model = pickle.load( open( "ml_model.pkl", "rb" ) )


my_current_pos = 0
print("starting")
print("start pos: ", my_current_pos)

while True:

    current_min = datetime.datetime.now().minute
    if current_min%5==0:

        result_list=[]

        result_list=result_list.__add__(take_features("5m"))
        result_list=result_list.__add__(take_features("15m"))
        result_list=result_list.__add__(take_features("30m"))
        result_list=result_list.__add__(take_features("1h"))
        result_list=result_list.__add__(take_features("4h"))

        input_data.loc["test"]=result_list

        result=model.predict_proba(input_data)[-1]

        print(model.classes_)
        print(result)


        if my_current_pos==0:
            if result[0]>0.8:
                print("enter short")
                my_current_pos=-1

            elif result[2]>0.8:
                print("enter long")
                my_current_pos=1


        elif my_current_pos==1:
            if result[0]>0.6:
                print("close long")
                my_current_pos=0

                if result[0]>0.8:
                    print("enter short")
                    my_current_pos=-1

        elif my_current_pos == -1:
            if result[2] > 0.6:
                print("close short")
                my_current_pos = 0

                if result[2] > 0.8:
                    print("enter long")
                    my_current_pos = 1

        time.sleep(60)




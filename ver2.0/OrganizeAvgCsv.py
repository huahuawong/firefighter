#----------
# Title: OrganizeAvgCsv.py
# Author: Jun Hua Wong
# Date: Fall 2019
# Description:  Takes Wunderground CSV data, organizes it, and processes the 
#               data. (AVG, Min Etc.)
# Version 1.0
#----------

import pandas as pd
import glob
import numpy as np
import csv

# Need to read each file in the directory, get path
path = r'C:\Users\BurkeLaptop5\Downloads\data'
ListOfCurrentCSVFile = pd.concat([pd.read_csv(f) for f in glob.glob(path + "/*.csv")], ignore_index = True, sort = True)
TempDataList = []
TempDayList = []
df3 = []

df1 = 0
df = 0
count = 0
count1 = 0
index1 = 0
index2 = 0
for index, row in ListOfCurrentCSVFile.iterrows():
    TempDataList.append(index)
    TempDataList[index]= {'DateUTC':row.DateUTC,
                'Dewpoint':row.DewpointF,
                'DayPrecip':row.HourlyPrecipIn,
                'BarPress': row.PressureIn,
                'Temperature': row.TemperatureF,
                'WindSpeed': row.WindSpeedMPH,
                'Station': row.station}
    df = pd.DataFrame(TempDataList)
    FirstDate = df['DateUTC'].str[:10]
    if count1 == 0:
       df['DateUTComp'] = FirstDate
       df['DateUTComp0'] = FirstDate
       df['DateUTComp1'] = FirstDate
       count1 = 1
       FirstDate = 0
    else:
        df['DateUTComp']= df['DateUTC'].str[:10]
        df['DateUTComp0'] = df['DateUTComp'].shift(-1)
        df['DateUTComp1'] = df['DateUTComp']
        FirstDate = 0
        print(index) 
        
index = 0
for index, row in df.iterrows():  
    if df['DateUTComp0'][index-1:index].equals(df['DateUTComp'][index-1:index]):
        TempDayList.append(index1)
        TempDayList[index1]= {'Dewpoint':row.Dewpoint,
                              'DayPrecip':row.DayPrecip,
                              'BarPress': row.BarPress,
                              'Temperature': row.Temperature,
                              'WindSpeed': row.WindSpeed}
        index1 = index1+1
    else:
        df1 = pd.DataFrame(TempDayList)               
        AVGDewPoint = df1['Dewpoint'].astype(float).mean()
        AvgBarPress = df1["BarPress"].astype(float).mean()
        MaxTemp = df1["Temperature"].astype(float).max()
        MinTemp = df1["Temperature"].astype(float).min()
        AvgWindSpeed = df1["WindSpeed"].astype(float).mean()
        AvgDayPrecip = df1["DayPrecip"].astype(float).mean()
        if count == 0:
            df2 = {'AvgDewPoint':AVGDewPoint,
                   'AvgBarPress':AvgBarPress,
                   'MaxTemp':MaxTemp,
                   'MinTemp': MinTemp,
                   'AvgWindSpeed': AvgWindSpeed,
                   'AvgDayPrecip': AvgDayPrecip}
            df3 = pd.DataFrame.from_dict([df2])  
            with open(r'C:\Users\BurkeLaptop5\Downloads\Wildfires_0_Class.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(df3)
            count = count + 1
            TempDayList = []
            index1 = 0
            #TempDayList.append(index1)
            #TempDayList[index1] = TempDataList[index]
        else:
            df2 = [AVGDewPoint,
                   AvgBarPress,
                   MaxTemp,
                   MinTemp,
                   AvgWindSpeed,
                   AvgDayPrecip]        
            with open(r'C:\Users\BurkeLaptop5\Downloads\Wildfires_0_Class.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(df2)
            TempDayList = []            
            index1 = 0
            #TempDayList.append(index1)
            #TempDayList[index1] = TempDataList[index1]  
        


  
        
        
       

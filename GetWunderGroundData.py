#----------
# Title: GetWunderGroundData.py
# Author: Gavan Burke and Jun Hau Wong
# Date: Fall 2019
# Description:  Gets Wildfire list, Gets Sensor Node List, finds three
#               closest sensor nodes and scrapes weather data for 10 days before fire.
# Version 1.0
#----------
import requests
import pandas as pd
from dateutil import parser as parser
from dateutil import rrule
import datetime
from datetime import date, time
import io
from math import cos, asin, sqrt


def getWunderGroundData(station, day, month, year):
    url = "http://www.wunderground.com/weatherstation/WXDailyHistory.asp?ID={station}&day={day}&month={month}&year={year}&graphspan=day&format=1"
    full_url = url.format(station=station, day=day, month=month, year=year)
    # Request data from wunderground data
    response = requests.get(full_url, headers={'User-agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'})
    data = response.text
    # remove the excess <br> from the text data
    data = data.replace('<br>', '')
    # Convert to pandas dataframe (fails if issues with weather station)
    try:
        dataframe = pd.read_csv(io.StringIO(data), index_col=False)
        dataframe['station'] = station
    except Exception as e:
        print("Issue with date: {}-{}-{} for station {}".format(day,month,year, station))
        return None
    return dataframe

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(a))

def OneClosest(data, v):
    SortedList1 = sorted(data, key=lambda p: distance(v['Lat'],v['Lon'],p['Lat'],p['Lon']))
    return SortedList1[0]
def TwoClosest(data, v):
    SortedList2 = sorted(data, key=lambda p: distance(v['Lat'],v['Lon'],p['Lat'],p['Lon']))
    return SortedList2[1]
    #return min(data, key=lambda p: distance(v['Lat'],v['Lon'],p['Lat'],p['Lon']))
def ThreeClosest(data, v):
    SortedList3 = sorted(data, key=lambda p: distance(v['Lat'],v['Lon'],p['Lat'],p['Lon']))
    return SortedList3[2]

# Wildfire source data list
#WildFireDataList = pd.read_csv("CalWildFiresList.csv")
WildFireDataList = pd.read_csv("CalNonWildFires.csv")
# Sensor Node data set
SensorNodeDataList = pd.read_csv("2019WeatherUndergroundWebScraperStations.csv")  
# Lets process each row...
TempDataList=[]
for index, row in SensorNodeDataList.iterrows():
    TempDataList.append(index)
    TempDataList[index]= {'Lat':row.Lat, 'Lon':row.Lon, 'Station':row.Station}
    

for row in WildFireDataList.head(n=63).itertuples():
    
    v = {'Lat': row.Lat, 'Lon': row.Lon}
    print(v)
    # end date needs to be date of the fire
    end_date = row.date
    print(end_date)
    NewEndDate = datetime.datetime.strptime(end_date,"%Y-%m-%d")
    # start date needs to be 10 days prior
    FormDate = NewEndDate - datetime.timedelta(days=10)
    print(FormDate)
    start_date = FormDate.strftime("%Y-%m-%d")
    print(start_date)
    #YYYY-MM-DD
    start = parser.parse(start_date)
    end = parser.parse(end_date)
    dates = list(rrule.rrule(rrule.DAILY, dtstart=start, until=end))
    OClosest = OneClosest(TempDataList, v)
    TClosest = TwoClosest(TempDataList, v)
    THClosest = ThreeClosest(TempDataList, v)


    
    # Create a list of stations here to download data for list(map(itemgetter('gfg'), test_list)) 
    stations = [OClosest.get('Station'),TClosest.get('Station'),THClosest.get('Station')]

    # Set a backoff time in seconds if a request fails
    backoff_time = 30
    data = {}

    # Gather data for each station in turn and save to CSV.
    for station in stations:
        print("Working on {}".format(station))
        data[station] = []
        for date in dates:
            # Print period status update messages
            if date.day % 10 == 0:
                print("Working on date: {} for station {}".format(date, station))
            done = False
            while done == False:
                try:
                    weather_data = getWunderGroundData(station, date.day, date.month, date.year)
                    done = True
                except ConnectionError as e:
                    # limit connections
                    time.sleep(10)
            # Add each processed date to the overall data
            data[station].append(weather_data)
        # Throw Data into CSV file for analysis
        pd.concat(data[station]).to_csv("data/{}_weather".format(station) + "{}.csv".format(start_date))


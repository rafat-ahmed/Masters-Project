# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:17:56 2020

@author: RAFAT2
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import datetime, timedelta
import numpy as np
from windrose import WindroseAxes
from numpy.polynomial.polynomial import polyfit
from scipy import stats
from scipy.odr import *
import scipy.odr as odr
import scipy.odr.odrpack as odrpack


from sklearn.metrics import mean_squared_error
from math import sqrt

import statistics


#%% FOUNDATION
dataset1=pd.read_csv('Tot(1).txt', sep='\s+', parse_dates=[['-','-.1']])#read the data
dataset1=dataset1[1:]#eliminate the first raw

dataset1.columns =['datetime','date','time','type', 'sample','standard','port','cavity_temp','cavity_press','h2o','d13ch4_C',
          'd13ch4_wet','d13ch4_dry','d13ch4_stdev', '13ch4_N','d13co2_C','d13co2_wet','d13co2_dry','d13co2_stdev','d13co2_N',
          '12ch4C', '12ch4wet','12ch4dry','12ch4stdev','12ch4N','12co2C','12co2wet','12co2dry','12co2stdev','12co2N','13ch4C',
          '13ch4wet','13ch4dry','13ch4stdev','13ch4N','13co2C','13co2wet','13co2dry','13co2stdev','13co2N'] 

air=dataset1[dataset1['type']=='air'] #select only air data
air['12co2']=air['12co2C'].astype(str) #define the number as string
air= air[air['12co2'].str.contains("F") != True] #remove the filtered data
air= air[air['12co2'].str.contains("x") != True] #remove the filtered data

air['d13C'] = air['d13co2_C'].astype(str)
air = air[air['d13C'].str.contains('F') != True]
air = air[air['d13C'].str.contains('x') != True]


air['12co2'] = air['12co2'].apply(pd.to_numeric)
air['d13C'] = air['d13C'].apply(pd.to_numeric)
air['datetime'] = pd.to_datetime(air['datetime'], format = '%y%m%d %H%M%S')

#plot the data (in this case CO2 concentrations)

# plt.subplot(2,1,1)
# plt.plot(air['datetime'], air['12co2'],'.')
# plt.xticks(rotation = 270)

# plt.subplot(2,1,2)
# plt.plot(air['datetime'], air['d13C'], '+')
# plt.xticks(rotation = 270)

### calculating total co2 concentration
    
x = air['12co2'] # assing the 12co2_C column from the air data to x
y = air['d13C'] # assing the d13co2_C column to y

s1 = pd.Series(x, name = '12co2')
s2= pd.Series(y, name = 'd13C')
air_short = pd.concat([s1,s2], axis=1 ) # Merge the two series x and y into one dataframe. This will contain the filtered air data only

V = 0.0111802 # VPBD value
air_short['13co2']=air_short['12co2']*V*(1+ 0.001*air_short['d13C']) # the equation to calculate carbon 13 co2

total_co2=air_short['13co2']+air_short['12co2']
z = air_short['13co2'] # assing the 13co2 column to this dummy variable z
air['13co2'] = z # now add this column z to the main air data
air['total_co2'] = total_co2 # add this column onto the main air data 
plt.plot(air['datetime'], air['total_co2'],'.')
plt.xticks(rotation =270)
# plt.title('Measured & Baseline co2 Concentration')


#%% Background co2 

dataset_bg= pd.read_fwf('MH_G_co2_day-2.txt')
# Adding some extra columns to the df
dataset_bg['dateInt']=dataset_bg['Yr'].astype(str) + dataset_bg['Mt'].astype(str).str.zfill(2)+ dataset_bg['Dy'].astype(str).str.zfill(2)
dataset_bg['datetime'] = pd.to_datetime(dataset_bg['dateInt'], format='%Y %m %d')
dataset_bg['dayofyear'] = dataset_bg['datetime'].dt.dayofyear
dataset_bg_short = dataset_bg[9299:10189] # selecting a subset of the df


dataset_bg.columns = ['Yr', 'Mt', 'Dy' ,'Hr','Concentration', 'std noise', 'dateInt', 'datetime','dayofyear'] # assigning new labels to df cols
b=dataset_bg.iloc[9299:10189 , 0:8]
time_b = pd.to_datetime(b[['Yr','Mt','Dy','Hr']].astype(str).apply(' '.join, 1), format='%Y %m %d %H')
time_b = pd.to_datetime(dataset_bg['datetime'], format='%Y %m %d %H')
conc=dataset_bg.iloc[9299:10189 :,4]
# bg_marchtojune18 = dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==3) | (dataset_bg_short['Mt']==4) | (dataset_bg_short['Mt']==5) | (dataset_bg_short['Mt']==6))]
# bg_marchtojune19 = dataset_bg_short[(dataset_bg_short['Yr']==2019) & ((dataset_bg_short['Mt']==3) | (dataset_bg_short['Mt']==4) | (dataset_bg_short['Mt']==5) | (dataset_bg_short['Mt']==6))]


plt.plot(time_b.iloc[9299:10189], conc)
# plt.xaxis.set_major_locator(plt.MaxNLocator(5))
# # plt.locator_params(axis='x', nbins=5)
plt.xticks(rotation = 270)
plt.show()
# fig, ax1=plt.subplots()
# ax2= ax1.twinx()
# curve1 = ax1.plot(air['datetime'], air['total_co2'],'.', color = 'b')
# curve2 = ax2.plot(time_b, conc, color = 'r')

# ax1.set_ylabel('Measured co2 concentration', color = 'b' )
# ax2.set_ylabel('Baseline co2 concentration', color = 'r')
# ax1.tick_params(rotation = 270)
# plt.plot()
# plt.savefig('co2 Plot.png', dpi = 300, bbox_inches='tight')
# plt.show()

#%% plotting just the d13C data

plt.plot(air['datetime'], air['d13C'], '+')
plt.xticks(rotation = 270)
plt.title('d13C')
# plt.savefig('d13C.png', dpi = 300, bbox_inches='tight')
plt.show()

#%% Afternoon data

air['hour'] = air['datetime'].dt.hour # create a column for the just the hour. 
air['year'] = air['datetime'].dt.year
air['month'] = air['datetime'].dt.month
air['day'] = air['datetime'].dt.day
air['week'] = air['datetime'].dt.week
air['minute'] = air['datetime'].dt.minute
air['second'] = air['datetime'].dt.second
air['dayofyear'] = air['datetime'].dt.dayofyear

air_short2 = air[['datetime', 'time', 'year', 'month', 'day','dayofyear','hour', 'minute', 'second', '12co2','13co2', 'd13C','total_co2']]

air_afternoon = air_short2[(air_short2['hour']==12) | (air_short2['hour']==13) | (air_short2['hour']==14) | (air_short2['hour']==15) |
                           (air_short2['hour']==16) | (air_short2['hour']==17)]

afternoon_totalco2 = air_afternoon['12co2'] + air_afternoon['13co2']

plt.plot(air_afternoon['datetime'], afternoon_totalco2, '+', label = 'Measured')
plt.plot(time_b.iloc[9299:10189], conc, color = '#ff7f0e', label = 'Baseline')
plt.xticks(rotation = 290) 
plt.legend(loc = 'upper right', fontsize=9)
plt.title('Afternoon co2 concentrations')
plt.ylabel('co2 concentration')
plt.xlabel('Date')
# plt.savefig('Afternoon co2 - MAIN.png', dpi =300, bbox_inches= 'tight')
plt.show()

### PLotting co2 concentrations of induvidual days

prac = air[(air['year']==2018) & (air['month']==1) & (air['day']==6)]
plt.plot(prac['datetime'], prac['total_co2'])
plt.show()

### co2 concentration with and baseline concentration with the double axis:
fig, ax1=plt.subplots()
ax2= ax1.twinx()
curve1 = ax1.plot(air_afternoon['datetime'], afternoon_totalco2,'.', color = 'b')
curve2 = ax2.plot(time_b, conc, color = 'r')

ax1.set_ylabel('Afternoon co2 concentration', color = 'b' )
ax2.set_ylabel('Baseline co2 concentration', color = 'r')
ax1.tick_params(axis= 'x', rotation = 270)

# plt.savefig('co2 Plot afternoon.png', dpi = 300, bbox_inches='tight')
plt.show()


## plot afternoon d13C
# plt.plot(air_afternoon['datetime'], air_afternoon['d13C'],'+')
# plt.xticks(rotation = 270)
# plt.title('d13C afternoon concentration')
# # plt.savefig('d13C_afternoon.png', dpi = 300, bbox_inches='tight')
# plt.show()

#%% Yearly plots
air['hour'] = air['datetime'].dt.hour # create a column for the just the hour. 
air['year'] = air['datetime'].dt.year
air['month'] = air['datetime'].dt.month
air['day'] = air['datetime'].dt.day
air['week'] = air['datetime'].dt.week
air['minute'] = air['datetime'].dt.minute
air['second'] = air['datetime'].dt.second
air['dayofyear'] = air['datetime'].dt.dayofyear


air_short3 = air[['datetime', 'dayofyear', 'time', 'year', 'month', 'day', 'hour', 'minute', 'second', '12co2','13co2', 'd13C', 'total_co2']]

air_marchtojune18 = air_short3[ (air_short3['year']==2018) & ((air_short3['month']==3) |(air_short3['month']==4)|(air_short3['month']==5)|(air_short3['month']==6))]
air_marchtojune19 = air_short3[ (air_short3['year']==2019) & ((air_short3['month']==3) |(air_short3['month']==4)|(air_short3['month']==5)|(air_short3['month']==6))]
air_marchtojune20 = air_short3[ (air_short3['year']==2020) & ((air_short3['month']==3) |(air_short3['month']==4)|(air_short3['month']==5)|(air_short3['month']==6))]

marchtojune18_short = air_marchtojune18.iloc[0:8824] #select the rows from air_marchtojune18 that inlcude just the first 7 days of march 
mean_co2_18 = marchtojune18_short['total_co2'].mean() # take the mean of these values for total_co2 column
mean_d13C_18 = marchtojune18_short['d13C'].mean()

marchtojune19_short = air_marchtojune19.iloc[0:7532]
mean_co2_19 = marchtojune19_short['total_co2'].mean()
mean_d13C_19 = marchtojune19_short['d13C'].mean()

marchtojune20_short = air_marchtojune20.iloc[0:4098]
mean_co2_20 = marchtojune20_short['total_co2'].mean()
mean_d13C_20 = marchtojune20_short['d13C'].mean()


subtractedco2_marchtojune18 = air_marchtojune18['total_co2'] - mean_co2_18
subtractedco2_marchtojune19 = air_marchtojune19['total_co2'] - mean_co2_19
subtractedco2_marchtojune20 = air_marchtojune20['total_co2'] - mean_co2_20
subtracedd13C_marchtojune18 = air_marchtojune18['d13C'] - mean_d13C_18
subtracedd13C_marchtojune19 = air_marchtojune19['d13C'] - mean_d13C_19
subtracedd13C_marchtojune20 = air_marchtojune20['d13C'] - mean_d13C_20

# ### plotting d13C concentrations
# plt.plot(air_marchtojune18['dayofyear'], subtracedd13C_marchtojune18, label = '2018')
# plt.plot(air_marchtojune19['dayofyear'], subtracedd13C_marchtojune19,  label = '2019')
# plt.plot(air_marchtojune20['dayofyear'], subtracedd13C_marchtojune20, label = '2020')
# plt.title('d13C during March-June')
# plt.xlabel('day of year')
# plt.legend(loc= 'lower right')
# plt.savefig('d13C_lockdown.png', dpi = 300, bbox_inches='tight')
# plt.show()

### plotting co2 concentrations 
plt.plot(air_marchtojune18['dayofyear'], subtractedco2_marchtojune18, label = '2018' )
plt.plot(air_marchtojune19['dayofyear'], subtractedco2_marchtojune19 , label='2019')
plt.plot(air_marchtojune20['dayofyear'], subtractedco2_marchtojune20, label ='2020')
plt.legend(loc = 'upper left')
plt.title('co2 concentratoin during March-June')
plt.xlabel('Day Of Year')
# plt.savefig('co2 march-June',dpi = 300, bbox_inches='tight')
plt.show()

### background co2 plot for lockdown period
# conc18 = bg_marchtojune18['Conc']
# conc19 = bg_marchtojune19['Conc']
# plt.plot(bg_marchtojune18['dayofyear'], conc18)
# plt.plot(bg_marchtojune19['dayofyear'], conc19)
# plt.show()

# fig, ax1=plt.subplots()
# ax2= ax1.twinx()
# curve1 = ax1.plot(air_marchtojune18['dayofyear'], subtractedco2_marchtojune18, label = '2018' )
# curve2 = ax1.plot(air_marchtojune19['dayofyear'], subtractedco2_marchtojune19, label = '2019' )
# curve3 = ax1.plot(air_marchtojune20['dayofyear'], subtractedco2_marchtojune20, label = '2020' )
# curve4 = ax2.plot(bg_marchtojune19['dayofyear'], conc19, 'r', label = 'BL19')
# curve5 = ax2.plot(bg_marchtojune18['dayofyear'], conc18, 'k', label ='BL18')

# ax1.set_ylabel('co2 concentration' )
# ax2.set_ylabel('Baseline co2 concentration', color = 'r')
# ax1.legend(loc='upper right')
# ax2.legend(loc='upper left')
# ax1.set_xlabel('Day of Year')
# plt.savefig('co2 march-June adjusted',dpi = 300, bbox_inches='tight')
# plt.plot()
# plt.show()

# plt.plot(air_marchtojune18['dayofyear'], air_marchtojune18['total_co2'], label = '2018')
# plt.plot(air_marchtojune19['dayofyear'], air_marchtojune19['total_co2'],  label = '2019')
# plt.plot(air_marchtojune20['dayofyear'], air_marchtojune20['total_co2'], label = '2020')
# plt.title('non-adjusted co2 concentration for March-June')
# plt.xlabel('day of year')
# plt.legend(loc= 'upper right')
# plt.savefig('co2 march-June non-adjusted',dpi = 300, bbox_inches='tight')
# plt.show()
         


#%% Monthly mean of co2 for 2018 & 2019 Brute Force
air_afternoon['afternoon_totalco2'] = afternoon_totalco2

Jan18_data = air_afternoon[(air_afternoon['year']==2018) & ((air_afternoon['month']==1))]
Feb18_data = air_afternoon[(air_afternoon['year']==2018) & ((air_afternoon['month']==2))]
Mar18_data = air_afternoon[(air_afternoon['year']==2018) & ((air_afternoon['month']==3))]
Apr18_data = air_afternoon[(air_afternoon['year']==2018) & ((air_afternoon['month']==4))]
May18_data = air_afternoon[(air_afternoon['year']==2018) & ((air_afternoon['month']==5))]
Jun18_data = air_afternoon[(air_afternoon['year']==2018) & ((air_afternoon['month']==6))]
Jul18_data = air_afternoon[(air_afternoon['year']==2018) & ((air_afternoon['month']==7))]
Aug18_data = air_afternoon[(air_afternoon['year']==2018) & ((air_afternoon['month']==8))]
Sep18_data = air_afternoon[(air_afternoon['year']==2018) & ((air_afternoon['month']==9))]
Oct18_data = air_afternoon[(air_afternoon['year']==2018) & ((air_afternoon['month']==10))]
Nov18_data = air_afternoon[(air_afternoon['year']==2018) & ((air_afternoon['month']==11))]
Dec18_data = air_afternoon[(air_afternoon['year']==2018) & ((air_afternoon['month']==12))]

Jan19_data = air_afternoon[(air_afternoon['year']==2019) & ((air_afternoon['month']==1))]
Feb19_data = air_afternoon[(air_afternoon['year']==2019) & ((air_afternoon['month']==2))]
Mar19_data = air_afternoon[(air_afternoon['year']==2019) & ((air_afternoon['month']==3))]
Apr19_data = air_afternoon[(air_afternoon['year']==2019) & ((air_afternoon['month']==4))]
May19_data = air_afternoon[(air_afternoon['year']==2019) & ((air_afternoon['month']==5))]
Jun19_data = air_afternoon[(air_afternoon['year']==2019) & ((air_afternoon['month']==6))]
Jul19_data = air_afternoon[(air_afternoon['year']==2019) & ((air_afternoon['month']==7))]
Aug19_data = air_afternoon[(air_afternoon['year']==2019) & ((air_afternoon['month']==8))]
Sep19_data = air_afternoon[(air_afternoon['year']==2019) & ((air_afternoon['month']==9))]
Oct19_data = air_afternoon[(air_afternoon['year']==2019) & ((air_afternoon['month']==10))]
Nov19_data = air_afternoon[(air_afternoon['year']==2019) & ((air_afternoon['month']==11))]
Dec19_data = air_afternoon[(air_afternoon['year']==2019) & ((air_afternoon['month']==12))]

mean_jan18 = Jan18_data['afternoon_totalco2'].mean()
mean_feb18 = Feb18_data['afternoon_totalco2'].mean()
mean_mar18 = Mar18_data['afternoon_totalco2'].mean()
mean_apr18 = Apr18_data['afternoon_totalco2'].mean()
mean_may18 = May18_data['afternoon_totalco2'].mean()
mean_jun18 = Jun18_data['afternoon_totalco2'].mean()
mean_jul18 = Jul18_data['afternoon_totalco2'].mean()
mean_aug18 = Aug18_data['afternoon_totalco2'].mean()
mean_sep18 = Sep18_data['afternoon_totalco2'].mean()
mean_oct18 = Oct18_data['afternoon_totalco2'].mean()
mean_nov18 = Nov18_data['afternoon_totalco2'].mean()
mean_dec18 = Dec18_data['afternoon_totalco2'].mean()

mean_jan19 = Jan19_data['afternoon_totalco2'].mean()
mean_feb19 = Feb19_data['afternoon_totalco2'].mean()
mean_mar19 = Mar19_data['afternoon_totalco2'].mean()
mean_apr19 = Apr19_data['afternoon_totalco2'].mean()
mean_may19 = May19_data['afternoon_totalco2'].mean()
mean_jun19 = Jun19_data['afternoon_totalco2'].mean()
mean_jul19 = Jul19_data['afternoon_totalco2'].mean()
mean_aug19 = Aug19_data['afternoon_totalco2'].mean()
mean_sep19 = Sep19_data['afternoon_totalco2'].mean()
mean_oct19 = Oct19_data['afternoon_totalco2'].mean()
mean_nov19 = Nov19_data['afternoon_totalco2'].mean()
mean_dec19 = Dec19_data['afternoon_totalco2'].mean()

monthly_means18 = [mean_jan18, mean_feb18, mean_mar18, mean_apr18, mean_may18, mean_jun18, mean_jul18, mean_aug18, mean_sep18, mean_oct18, 
                  mean_nov18, mean_dec18]
monthly_means19 = [mean_jan19, mean_feb19, mean_mar19, mean_apr19, mean_may19, mean_jun19, mean_jul19, mean_aug19, mean_sep19, mean_oct19, 
                    mean_nov19, mean_dec19]

months = [1,2,3,4,5,6,7,8,9,10,11,12]

plt.plot(months, monthly_means18, label = '2018')
plt.plot(months, monthly_means19, label = '2019')
plt.xlabel('Months')
plt.ylabel('co2 Concentration')
plt.title('Average monthly co2 concentrations, without BL correction')
plt.legend(loc = 'lower right')
plt.savefig('Monthly averaged co2 WITHOUT BL co2',dpi = 300, bbox_inches='tight')
plt.show()
#%%
Jan18_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==1))]
Feb18_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==2))]
Mar18_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==3))]
Apr18_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==4))]
May18_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==5))]
Jun18_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==6))]
Jul18_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==7))]
Aug18_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==8))]
Sep18_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==9))]
Oct18_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==10))]
Nov18_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==11))]
Dec18_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==12))]

Jan19_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==1))]
Feb19_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==2))]
Mar19_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==3))]
Apr19_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==4))]
May19_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==5))]
Jun19_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==6))]
Jul19_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==7))]
Aug19_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==8))]
Sep19_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==9))]
Oct19_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==10))]
Nov19_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==11))]
Dec19_bg =  dataset_bg_short[(dataset_bg_short['Yr']==2018) & ((dataset_bg_short['Mt']==12))]

mean_jan18bg = Jan18_bg['Conc'].mean()
mean_feb18bg = Feb18_bg['Conc'].mean()
mean_mar18bg = Mar18_bg['Conc'].mean()
mean_apr18bg = Apr18_bg['Conc'].mean()
mean_may18bg = May18_bg['Conc'].mean()
mean_jun18bg = Jun18_bg['Conc'].mean()
mean_jul18bg = Jul18_bg['Conc'].mean()
mean_aug18bg = Aug18_bg['Conc'].mean()
mean_sep18bg = Sep18_bg['Conc'].mean()
mean_oct18bg = Oct18_bg['Conc'].mean()
mean_nov18bg = Nov18_bg['Conc'].mean()
mean_dec18bg = Dec18_bg['Conc'].mean()

mean_jan19bg = Jan19_bg['Conc'].mean()
mean_feb19bg = Feb19_bg['Conc'].mean()
mean_mar19bg = Mar19_bg['Conc'].mean()
mean_apr19bg = Apr19_bg['Conc'].mean()
mean_may19bg = May19_bg['Conc'].mean()
mean_jun19bg = Jun19_bg['Conc'].mean()
mean_jul19bg = Jul19_bg['Conc'].mean()
mean_aug19bg = Aug19_bg['Conc'].mean()
mean_sep19bg = Sep19_bg['Conc'].mean()
mean_oct19bg = Oct19_bg['Conc'].mean()
mean_nov19bg = Nov19_bg['Conc'].mean()
mean_dec19bg = Dec19_bg['Conc'].mean()

monthly_means_18bg = [mean_jan18bg, mean_feb18bg, mean_mar18bg, mean_apr18bg, mean_may18bg, mean_jun18bg, mean_jul18bg, 
                      mean_aug18bg, mean_sep18bg, mean_oct18bg, mean_nov18bg, mean_dec18bg]
monthly_means_19bg = [mean_jan19bg, mean_feb19bg, mean_mar19bg, mean_apr19bg, mean_may19bg, mean_jun19bg, mean_jul19bg, 
                      mean_aug19bg, mean_sep19bg, mean_oct19bg, mean_nov19bg, mean_dec19bg]

diff18 = list(np.array(monthly_means18) - np.array(monthly_means_18bg))
diff19 = list(np.array(monthly_means19) - np.array(monthly_means_19bg))

# plt.plot(months, diff18, label = '2018')
# plt.plot(months, diff19, label = '2019')
# plt.xlabel('Months')
# plt.ylabel('co2 Concentration')
# plt.title('Average monthly co2 concentrations')
# plt.legend(loc = 'lower right')
# plt.savefig('Monthly averaged co2 with BL co2',dpi = 300, bbox_inches='tight')
# plt.show()


#%% Windrose plot

dataset_19= pd.read_csv('2019_weather_data.txt', delimiter=',')
dataset_19.columns = ['datetime','Interval', 'Indoor Humidity', 'Indoor Temp', 'Outdoor Humidity', 'Outdoor Temp', 'Pressure','Wind Speed', 'Gust','Wind Direction', 'r1','r2']
dataset_19['datetime'] = pd.to_datetime(dataset_19['datetime'], format = '%Y-%m-%d %H:%M:%S')
dataset_19['year'] = dataset_19['datetime'].dt.year
dataset_19['month'] = dataset_19['datetime'].dt.month
dataset_19['day'] = dataset_19['datetime'].dt.day
dataset_19['hour'] = dataset_19['datetime'].dt.hour
dataset_19['time'] = dataset_19['datetime'].dt.time
dataset_19['minute'] = dataset_19['datetime'].dt.minute
            
### Merging weather and co2 data
air_afternoon['total co2'] = afternoon_totalco2

air_afternoon['datetime'] = air_afternoon['datetime'].dt.round('min') # round all itme values to nearest min
dataset_19['datetime'] = dataset_19['datetime'].dt.round('min') # round all time values to nearest min 

weather_n_co2 = pd.merge(air_afternoon, dataset_19, on = 'datetime', how='left' , suffixes=('_TOT', '_weather')) # merge the two df's
#together of of the datetime values from the air_afternoon df as this has more rows

### Filtering out the nan values, which are where the times do not ovr lap for both dfs.
weather_n_co2['Wind Direction'] = weather_n_co2['Wind Direction'].astype(str) 
weather_n_co2 = weather_n_co2[weather_n_co2['Wind Direction'].str.contains('nan') !=True]
weather_n_co2['Wind Direction'] = weather_n_co2['Wind Direction'].apply(pd.to_numeric)

ax = WindroseAxes.from_ax()
ax.bar(weather_n_co2['Wind Direction']*360/16 ,weather_n_co2['total co2'] ,normed=True, opening=0.8, edgecolor='black')
ax.set_legend()
plt.title('Co2 and wind direction 2019')
# plt.savefig('Windrose 2019 - co2 and wind direction', dpi = 300, bbox_inches='tight')

#%% 2018 windrose
dataset_18= pd.read_csv('2018_weather_data.txt', delimiter=',')
dataset_18.columns = ['datetime','Interval', 'Indoor Humidity', 'Indoor Temp', 'Outdoor Humidity', 'Outdoor Temp', 'Pressure','Wind Speed', 'Gust','Wind Direction', 'r1','r2']
dataset_18['datetime'] = pd.to_datetime(dataset_18['datetime'], format = '%Y-%m-%d %H:%M:%S')

dataset_18['datetime'] = dataset_18['datetime'].dt.round('min')
weather_n_co218 = pd.merge(air_afternoon, dataset_18, on = 'datetime', how='left', suffixes =('_TOT', '_weather'))

weather_n_co218['Wind Direction'] = weather_n_co218['Wind Direction'].astype(str)
weather_n_co218 = weather_n_co218[weather_n_co218['Wind Direction'].str.contains('nan') !=True]
weather_n_co218['Wind Direction'] = weather_n_co218['Wind Direction'].apply(pd.to_numeric)

# ax = WindroseAxes.from_ax()
# ax.bar(weather_n_co218['Wind Direction']*360/16 ,weather_n_co218['total_co2'] ,normed=True, opening=0.8, edgecolor='black')
# ax.set_legend()
# plt.title('Co2 and wind direction 2018')
# plt.savefig('Windrose 2018 - co2 and wind direction', dpi = 300, bbox_inches='tight')

#%% Windrose pollution events

weather20 = pd.read_csv('Meteo_2020.csv')
weather20.columns = ['datetime','Interval', 'Indoor Humidity', 'Indoor Temp', 'Outdoor Humidity', 'Outdoor Temp', 'Pressure',
                      'Wind Speed', 'Gust','Wind Direction', 'r1','r2']
weather20['datetime'] = pd.to_datetime(weather20['datetime'], format = '%Y-%m-%d %H:%M:%S')

weather20['month'] = weather20['datetime'].dt.month
weather20['day'] = weather20['datetime'].dt.day
weather20['minute'] = weather20['datetime'].dt.minute

air_afternoon['datetime'] = air_afternoon['datetime'].dt.round('min')
weather20['datetime'] = weather20['datetime'].dt.round('min')
weather20_co2 = pd.merge(air_afternoon, weather20, on = 'datetime', how='left', suffixes =('_TOT', '_weather'))

weather20_co2['Wind Direction'] = weather20_co2['Wind Direction'].astype(str)
weather20_co2 = weather20_co2[weather20_co2['Wind Direction'].str.contains('nan') !=True]
weather20_co2['Wind Direction'] = weather20_co2['Wind Direction'].apply(pd.to_numeric)

weather_event_2 = weather20_co2[((weather20_co2['month_TOT']==3) & (weather20_co2['day_TOT']==23))]


ax = WindroseAxes.from_ax()
ax.bar(weather20_co2['Wind Direction']*360/16, weather20_co2['d13C'], normed = True, opening =0.8, edgecolor ='black' )
ax.set_legend()
ax.set_title('d13C relative to wind direction - 23/03/2020')
# plt.savefig('d13C Windrose for event 2.png', dpi = 300, bbox_inches='tight')

#%% Algorithm w/ ODR
air_alg = pd.read_csv('MyData.txt', skiprows=[1] ,usecols = ['datetime', 'type', 'total_co2', 'd13C', 'day','month','year','dayofyear','week','hour'], dtype = {'total_co2':
np.float64, 'd13C':np.float64, 'day':str, 'month':str, 'year':str,'week':str, 'hour': str, 'dayofyear':str}) 

air_alg['dmy'] = air_alg['day'] +'-'+ air_alg['month'] +'-'+ air_alg['year']

window = air_alg[((air_alg['year']=='2020'))  & ((air_alg['hour']=='12')|(air_alg['hour']=='13')|(air_alg['hour']=='14')|(air_alg['hour']=='15')
                    |(air_alg['hour']=='16')|(air_alg['hour']=='17'))]

window['dayofyear'] = window['dayofyear'].astype(str).astype(int)

window_LD = window[(window['dayofyear']>=83) & (window['dayofyear']<=173)]

accepted_dates_list = []

for d in window_LD['dmy'].unique():
    acceptable_date = {} # creating a dictionary to store the valid dates
    period = window_LD[window_LD.dmy==d] # defining each period from the dmy column
    p = (period['total_co2'])**-1
    q = period['d13C']
    g = polyfit(p, q, 1) 
    g0 = g[0]
    g1 = g[1]
    
    def linear_func(g,p):
        return g[0]*p +g[1]
    
    linear_model = Model(linear_func)
    data = RealData(p,q, sx = statistics.stdev(p), sy = statistics.stdev(q))
    init_odr = ODR(data, linear_model, beta0 = [g0, g1])
    output = init_odr.run()
    
    if (1-output.res_var) >= 0.8:
        acceptable_date['period'] = d
        acceptable_date['intercept'] = output.beta[1]
        acceptable_date['error'] = output.sd_beta[1]
        acceptable_date['r^2'] = (1-output.res_var)
        accepted_dates_list.append(acceptable_date)
    else:
        pass

accepted_dates_20 = pd.DataFrame(accepted_dates_list)

#%% Summary plot w/ ODR
# valid_dates18['period'] = pd.to_datetime(valid_dates18['period'], format = '%d-%m-%Y')
# valid_dates19['period'] = pd.to_datetime(valid_dates19['period'], format = '%d-%m-%Y')
# valid_dates20['period'] = pd.to_datetime(valid_dates20['period'], format = '%d-%m-%Y')

# plt.plot(valid_dates19['period'], valid_dates19['intercept'], 'o')
# axes = list(range(1,52))
# plt.errorbar(valid_dates18['period'], valid_dates18['intercept'], valid_dates18['error'], fmt ='o', ecolor = 'black')
# # plt.errorbar(valid_dates19['period'], valid_dates19['intercept'], valid_dates19['error'], fmt ='o', ecolor = 'black')
# # plt.errorbar(valid_dates20['period'], valid_dates20['intercept'], valid_dates20['error'], fmt ='o', ecolor = 'black')
# # plt.locator_params(axis='x', nbins=5)

# plt.xticks(rotation = 90)
# plt.grid(axis = 'y', linewidth =0.3)
# plt.ylabel('Source Signature')
# # plt.title()
# plt.show()

### 3-day with ODR

# window18 = df[(df['dayofyear']=='326') | (df['dayofyear']=='327')|(df['dayofyear']=='328')]

# p = (window18['total_co2'])**-1
# q = window18['d13C']
# g = polyfit(p, q, 1) 
# g0 = g[0]
# g1 = g[1]

# def linear_func(g,p):
#     return g[0]*p +g[1]
    
# linear_model = Model(linear_func)
# data = RealData(p,q, sx = statistics.stdev(p), sy = statistics.stdev(q))
# init_odr = ODR(data, linear_model, beta0 = [g0, g1])
# output = init_odr.run()
# output.pprint()

### 3 & 7 -day error bars
df = pd.read_csv('dataset20.txt', usecols = ['datetime', 'type', 'total_co2', 'd13C', 'day','month','year','dayofyear','week','hour'], dtype = {'total_co2':
np.float64, 'd13C':np.float64, 'day':str, 'month':str, 'year':str,'week':str, 'hour': str, 'dayofyear':str}) 

df['datetime'] = pd.to_datetime(df['datetime'], format = '%Y-%m-%d %H:%M:%S')    
df['dmy'] = df['day'] +'-'+ df['month'] +'-'+ df['year']

df['dayofyear'] = df['dayofyear'].astype(str).astype(int)
# df_LD = df[(df['dayofyear']>=82) & (df['dayofyear']<=174)]

first_day = 1
days_to_group = 3
accepted_dates_list = [] # creating an empty list to store the dates that we're interested in
for doy, gdf in df.groupby((df.datetime.dt.dayofyear.sub(first_day) // days_to_group)* days_to_group + first_day):
    
    acceptable_date = {} # creating a dictionary to store the valid dates    
    p = (gdf['total_co2'])**-1
    q = gdf['d13C']
    g = polyfit(p,q,1) # intercept and gradient calculation of the regression line
    g0 = g[0]
    g1= g[1]
    
    def linear_func(g,p):
        return g[0]*p + g[1]
    
    linear_model = Model(linear_func)
    data = RealData(p,q, sx = statistics.stdev(p), sy = statistics.stdev(q))
    init_odr = ODR(data, linear_model, beta0 = [g0, g1])
    output = init_odr.run()
    
    if (1-output.res_var) >= 0.8:
        acceptable_date['period1'] = gdf.dmy.min() # getting the min value of the dmy column for the grouped df
        acceptable_date['period2'] = gdf.dmy.max()
        acceptable_date['intercept'] = output.beta[1]
        acceptable_date['error'] = output.sd_beta[1]
        accepted_dates_list.append(acceptable_date) # sending the valid stuff in the dictionary to the list
    else:
        pass
    print(gdf, '\n')
accepted_dates20_3D = pd.DataFrame(accepted_dates_list)

#%% Algortihm

air_alg = pd.read_csv('MyData.txt',skiprows=[1] ,usecols = ['datetime', 'type', 'total_co2', 'd13C', 'day','month','year','dayofyear','week','hour'], dtype = {'total_co2':
np.float64, 'd13C':np.float64, 'day':str, 'month':str, 'year':str,'week':str, 'hour': str, 'dayofyear':str}) 
    
air_alg['datetime'] = pd.to_datetime(air_alg['datetime'], format = '%Y-%m-%d %H:%M:%S')
# air_alg['dayofyear'] = air_alg['datetime'].dt.dayofyear

air_alg['dmy'] = air_alg['day'] +'-'+ air_alg['month'] +'-'+ air_alg['year'] # adding a full date column to make it easir to filter through the rows, ie. each day

# df18 = air_alg[((air_alg['year']=='2018')) & ((air_alg['hour']=='12')|(air_alg['hour']=='13')|(air_alg['hour']=='14')|(air_alg['hour']=='15')
#                     |(air_alg['hour']=='16')|(air_alg['hour']=='17'))]

# df19 = air_alg[((air_alg['year']=='2019')) & ((air_alg['hour']=='12')|(air_alg['hour']=='13')|(air_alg['hour']=='14')|(air_alg['hour']=='15')
#                     |(air_alg['hour']=='16')|(air_alg['hour']=='17'))]

# df20 = air_alg[((air_alg['year']=='2020')) & ((air_alg['hour']=='12')|(air_alg['hour']=='13')|(air_alg['hour']=='14')|(air_alg['hour']=='15')
#                     |(air_alg['hour']=='16')|(air_alg['hour']=='17'))]

window = air_alg[((air_alg['year']=='2018'))  & ((air_alg['hour']=='12')|(air_alg['hour']=='13')|(air_alg['hour']=='14')|(air_alg['hour']=='15')
                    |(air_alg['hour']=='16')|(air_alg['hour']=='17'))]

window['dayofyear'] = window['dayofyear'].astype(str).astype(int)

window_LD = window[(window['month'] == '3')|(window['month'] == '4')|(window['month'] == '5')|(window['month'] == '6')]

accepted_dates_list = []

for d in window_LD['dmy'].unique(): # this will pass through each day, the .unique() ensures that it doesnt go over the same days  
    acceptable_date = {} # creating a dictionary to store the valid dates

    period = window_LD[window_LD.dmy==d] # defining each period from the dmy column
    p = (period['total_co2'])**-1
    q = period['d13C']
    c,m = polyfit(p, q, 1) 
    slope, intercept, r_value, p_value, std_err = stats.linregress(p, q)
    RMSE = sqrt(mean_squared_error(q, m*p + c))
    
    if r_value**2 >= 0.7:
        acceptable_date['period'] = d # populating the dictionary with the accpeted dates and corresponding other values
        acceptable_date['r-squared'] = r_value**2
        acceptable_date['intercept'] = intercept
        acceptable_date['RMSE'] = RMSE
        accepted_dates_list.append(acceptable_date) # sending the valid stuff in the dictionary to the list
    else:
        pass


accepted_dates18 = pd.DataFrame(accepted_dates_list) # converting the list to a df

#%% Seasonal Time series
df = pd.read_csv('dataset.txt', usecols = ['datetime', 'type', 'total_co2', 'd13C', 'day','month','year','dayofyear','week','hour'], dtype = {'total_co2':
np.float64, 'd13C':np.float64, 'day':str, 'month':str, 'year':str,'week':str, 'hour': str, 'dayofyear':str}) 

df['datetime'] = pd.to_datetime(df['datetime'], format = '%Y-%m-%d %H:%M:%S')    
df['dmy'] = df['day'] +'-'+ df['month'] +'-'+ df['year']

df['dayofyear'] = df['dayofyear'].astype(str).astype(int)


summer20 = df[(df['dayofyear']>= 153) & (df['dayofyear']<= 243)]
autumn20 = df[(df['dayofyear']>= 244) & (df['dayofyear']<= 334)]
spring20 = df[(df['dayofyear']>= 60) & (df['dayofyear']<= 151)]

#%%

# df18 = spring18.groupby(spring18.datetime.dt.dayofyear).mean()
# df19 = spring19.groupby(spring19.datetime.dt.dayofyear).mean()
# df20 = spring20.groupby(spring20.datetime.dt.dayofyear).mean()

# plt.plot(df18['dayofyear'], df18['d13C'], label = '2018')
# plt.plot(df19['dayofyear'], df19['d13C'], label = '2019')
# plt.plot(df20['dayofyear'], df20['d13C'],label = '2020')
# plt.legend(loc ='best')
# plt.show()

test = df[(df['dayofyear']==65)|(df['dayofyear']==66)|(df['dayofyear']==67)]
plt.plot(test['datetime'], test['d13C'],'+')
# plt.locator_params(axis='x', nbins=7)
plt.xticks(rotation = 310)
plt.show()


#%%
valid_dates['period'] = pd.to_datetime(valid_dates['period'], format ='%d-%m-%Y')

# plt.plot(accepted_dates['period'], accepted_dates['intercept'], '+')
plt.errorbar(valid_dates['period'], valid_dates['intercept'], valid_dates['error'], fmt ='o', ecolor = 'red', elinewidth = 2, capsize = 3)
plt.xticks(rotation = 300)
plt.show()

#%% Algorithm - 3 day window
''' window18.to_csv(r'dataset.txt', sep=', ') - this line converted my window18 df to a txt file.'''

df = pd.read_csv('dataset20.txt', usecols = ['datetime', 'type', 'total_co2', 'd13C', 'day','month','year','dayofyear','week','hour'], dtype = {'total_co2':
np.float64, 'd13C':np.float64, 'day':str, 'month':str, 'year':str,'week':str, 'hour': str, 'dayofyear':str}) 

df['datetime'] = pd.to_datetime(df['datetime'], format = '%Y-%m-%d %H:%M:%S')    
df['dmy'] = df['day'] +'-'+ df['month'] +'-'+ df['year']
df['dayofyear'] = df['dayofyear'].astype(str).astype(int)

# df_LD = df[(df['dayofyear']>=82) & (df['dayofyear']<=174)]


first_day = 1
days_to_group = 3
accepted_dates_list = [] # creating an empty list to store the dates that we're interested in
for doy, gdf in df.groupby((df.datetime.dt.dayofyear.sub(first_day) // days_to_group)* days_to_group + first_day):
    
    acceptable_date = {} # creating a dictionary to store the valid dates    
    p = (gdf['total_co2'])**-1
    q = gdf['d13C']
    c,m = polyfit(p,q,1) # intercept and gradient calculation of the regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(p, q) # getting some statistical properties of the regression line
    
    if r_value**2 >= 0.8:
        acceptable_date['period1'] = gdf.dmy.min() # getting the min value of the dmy column for the grouped df
        acceptable_date['period2'] = gdf.dmy.max()
        # acceptable_date['dmy'] = gdf['dmy'] # populating the dictionary with the accpeted dates and corresponding other values
        acceptable_date['r-squared'] = r_value**2
        acceptable_date['intercept'] = intercept

        accepted_dates_list.append(acceptable_date) # sending the valid stuff in the dictionary to the list
    else:
        pass
    print(gdf, '\n')
accepted_dates20 = pd.DataFrame(accepted_dates_list)
#%% 3-day source sig time series 
from matplotlib import dates
# from scipy.interpolate import spline
from scipy.interpolate import interp1d
from scipy.interpolate import splprep
from scipy.signal import savgol_filter

accpeted_dates_1D['period1'] = pd.to_datetime(accpeted_dates_1D['period1'], format = '%d-%m-%Y')




new_df = pd.concat([accepted_dates18,accepted_dates19,accepted_dates20], ignore_index = True)

new_df['period1'] = pd.to_datetime(new_df['period1'], format = '%d-%m-%Y')
# # accepted_dates['period'] = pd.to_datetime(accepted_dates['period'], format = '%d-%m-%Y')
new_df['month'] = new_df['period1'].dt.month

season = new_df[(new_df['month'] == 3)|(new_df['month'] == 4) |(new_df['month'] == 5)] 

x = (season['intercept'].min())-(season['intercept'].max())
y = season['intercept'].min()
z= season['intercept'].max()
print(y)
print(z)



# # plt.errorbar(valid_dates['period'], valid_dates['intercept'], valid_dates['error'], fmt ='o', ecolor = 'red', capsize = 1.5, markersize = 2.5,
#               # elinewidth =0.6)
# # plt.plot(new_df['period1'], new_df['intercept'], linewidth = 1)

# plt.errorbar(new_df['period1'], new_df['intercept'], new_df['error'], fmt = 'o', ecolor = 'orangered', color = 'darkgreen', markersize =3
#                 ,capsize = 1.5, elinewidth = 0.6)

# plt.xticks(rotation = 310)
# # # # plt.title('7-day window source signature time-series')
# plt.ylabel('Source Signature (‰)')
# plt.grid(axis = 'y', linewidth = 0.3)
# # plt.savefig('REPORT 3-day time series 2.png', dpi = 300, bbox_inches ='tight')
# plt.show()

# accepted_dates_LD3['period1'] = pd.to_datetime(accepted_dates_LD3['period1'], format = '%d-%m-%Y')
# plt.plot(accepted_dates_LD3['period1'], accepted_dates_LD3['intercept'], '+')
# plt.show()


# x = new_df.period1.values # convert period1 column to a numpy array
# y = new_df.intercept.values # convert the intercept column to a numpy array
# x_dates = np.array([dates.date2num(i) for i in x]) # period1 values are datetime objects, this line converts them to floats

# data = (x_dates, y)
# tck,u = splprep(data, s=0)
# args = ()
# unew = np.arange((x_dates.min(), x_dates.max(), 3))
# out = interpolate.splev(unew, tck)
# plt.plot(out[0], out[1], color='orange')
# plt.show()


#%% Spline trial 2

f = interp1d(x_dates, y, kind = 'quadratic')
x_smooth = np.linspace(x_dates.min(), x_dates.max(),num = 100,endpoint = True) # unsure if this line is right?

plt.plot(x_dates, y, 'o', x_smooth, f(x_smooth),'-')
plt.xlabel('Date')
plt.ylabel('Intercept')
# plt.legend(['data', 'cubic spline'], loc = 'lower right')
plt.show()

#%% Poster keeling plot
# window18 = df[(df['dayofyear']==65) | (df['dayofyear']==66) | (df['dayofyear']==67)]
# df = window[(window['day']=='29') & (window['month']=='4') & (window['year']=='2020')]
df = window[(window['dayofyear']==332)]

x = (df['total_co2'])**-1
y = (df['d13C'])
c,m = polyfit(x,y,1)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
plt.plot(x,y, '+')
plt.plot(x, c+m*x)
plt.xlabel('1/co2 (ppm)')
plt.ylabel('d13C (‰)')
plt.title('27/07/2019')
textstr = '\n'.join((
    r'$R^2=%.2f$' % (r_value**2, ),
    r'$C=%.2f$' % (intercept, )))
plt.text(0.00201,-10, textstr)
plt.text(0.002065,-10, '\u00B1 0.45')
plt.locator_params(axis='x', nbins=7)
# plt.savefig('report KP 1-day II fig.png', dpi = 300, bbox_inches = 'tight')
plt.show()
#%% Histogram summary plot using seaborn
import seaborn as sns
# n = len(accepted_dates18['intercept'])
# bin_num = sqrt(n)
# bin_width = (max(accepted_dates18['intercept']) - min(accepted_dates18['intercept']))/bin_num


sns.distplot(accepted_dates18['intercept'],bins = 3,label = '2018')
sns.distplot(accepted_dates19['intercept'],bins = 3 ,label = '2019')
sns.distplot(accepted_dates20['intercept'],bins = 3 ,label = '2020')
plt.legend(loc = 'upper left')
plt.grid(axis = 'y', linewidth = 0.3)
plt.xlabel('Source Signature (‰)')
plt.ylabel('Probability density')

# plt.savefig('REPORT FIG 7-day summary', dpi = 300, bbox_inches ='tight')
plt.show()

# sns.distplot(accepted_dates['intercept'])
# plt.grid(axis = 'y', linewidth = 0.3)
# plt.xlabel('Source signature')
# plt.ylabel('Probability density')
# plt.title('2018-2020 March-June whole day')
# plt.savefig('2018-2020 March-June whole day.png', dpi = 300 )
# plt.show()

# new_df = pd.concat([accepted_dates18,accepted_dates19,accepted_dates20], ignore_index = True)
# sns.set()
# sns.distplot(new_df['intercept'], bins= 20, kde = True)
# plt.xlabel('Source Signature (‰)')
# plt.ylabel('Probability density')
# plt.savefig('REPORT FIG 3-day TOTAL summary', dpi = 300, bbox_inches ='tight')
# plt.show()

plt.hist


#%%
# plt.hist(accepted_dates18['intercept'], label = '2018', bins = 10)
# plt.hist(accepted_dates19['intercept'], label = '2019', bins = 8, fill = False, hatch = '/', edgecolor = 'red')
# plt.hist(accepted_dates20['intercept'], label = '2020', bins = 13, alpha = 0.7, color = '#ff7f0e')

plt.hist(accepted_dates18['intercept'], label = '2018', bins = 7)
plt.hist(accepted_dates19['intercept'], label = '2019', bins = 7, fill = False, hatch = '/', edgecolor = 'red')
plt.hist(accepted_dates20['intercept'], label = '2020', bins = 7, alpha = 0.7, color = '#ff7f0e')

plt.legend(loc = 'best', fontsize = 8)
plt.grid(axis = 'y', linewidth = 0.3)
# # plt.style.use('ggplot') 
plt.title('3 day window - summary of source signatures')
plt.ylabel('Frequency')
plt.xlabel('d13C source value (‰)')
# plt.savefig('3-day histogram for poster OLS.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#%% Plot of source signatures over Lockdown
accepted_dates18['period'] = pd.to_datetime(accepted_dates18['period'], format = '%d-%m-%Y')
accepted_dates19['period'] = pd.to_datetime(accepted_dates19['period'], format = '%d-%m-%Y')
accepted_dates20['period'] = pd.to_datetime(accepted_dates20['period'], format = '%d-%m-%Y')


accepted_dates18['dayofyear'] = accepted_dates18['period'].dt.dayofyear
accepted_dates19['dayofyear'] = accepted_dates19['period'].dt.dayofyear
accepted_dates20['dayofyear'] = accepted_dates20['period'].dt.dayofyear

# plt.plot(accepted_dates18['dayofyear'], accepted_dates18['intercept'], label = '2018')
# plt.plot(accepted_dates19['dayofyear'], accepted_dates19['intercept'], label ='2019')
# plt.plot(accepted_dates20['dayofyear'], accepted_dates20['intercept'], label = '2020')

# accepted_dates18['MA_9'] = accepted_dates18.intercept.rolling(9).mean()
# accepted_dates19['MA_9'] = accepted_dates19.intercept.rolling(9).mean()
# accepted_dates20['MA_9'] = accepted_dates20.intercept.rolling(9).mean()
# plt.plot(accepted_dates18['dayofyear'],accepted_dates18['MA_9'], label = '2018')
# plt.plot(accepted_dates19['dayofyear'],accepted_dates19['MA_9'], label = '2019')
# plt.plot(accepted_dates20['dayofyear'],accepted_dates20['MA_9'], label = '2020')
# # plt.xticks(rotation = 290)
# # plt.locator_params(axis='x', nbins=13)
# plt.title('Source signatures - whole day')
# plt.xlabel('Day of the year')
# plt.ylabel('Source Signature')

# plt.legend(loc = 'bottom middle', fontsize = 8)
# # plt.savefig('Source signatures - whole day.png', dpi = 300, bbox_inches = 'tight')
# plt.show()

#%% Moving average

accepted_dates['period'] = pd.to_datetime(accepted_dates['period'], format = '%d-%m-%Y')
accepted_dates['dayofyear'] = accepted_dates['period'].dt.dayofyear

accepted_dates['MA_9'] = accepted_dates.intercept.rolling(9).mean()
accepted_dates['MA_20'] = accepted_dates.intercept.rolling(20).mean()

plt.plot(accepted_dates['period'],accepted_dates['intercept'], label = 'Raw values')
plt.plot(accepted_dates['period'],accepted_dates['MA_9'], label = '9 row MA')
plt.plot(accepted_dates['period'],accepted_dates['MA_20'], label = '20 row MA')

plt.xticks(rotation = 290)
plt.ylabel('Source Signature')
plt.xlabel('Date')
plt.legend(loc = 'lower right', fontsize = 7.5)
plt.title('Source signature moving average')
# plt.grid(True)
# plt.savefig('Source signature moving average.png', dpi = 300, bbox_inches = 'tight')
plt.show()
#%% Keeling plots



# plt.plot(trial['datetime'], trial['total_co2'], '+')
# plt.xticks(rotation = 270)
# plt.title('co2 - 22/11/2018')
# plt.savefig('co2 3.png', dpi = 300, bbox_inches='tight')

# Tried implimenting a king of loop that would run through and pick out the keeling plots with r^2>0.8 from the trial data, but can't 
# figure out how yet: 

# for row in trial_1:
#     df1 = trial_1[(trial_1['day'].between(1,30))]
#     p=(df1['total_co2'])**-1
#     q = df1['d13C']
#     c, m = polyfit(p,q,1)
# dates=[]
# for index, row in trial.iterrows():
#     print(row['year']) 
# trial = air[((air['year']==2020)) & (air['week']==1)|(air['week']==2)|(air['week']==3) 
# & ((air['hour'] ==12) | (air['hour'] ==13) | (air['hour'] ==14) |(air['hour'] ==15) | (air['hour'] ==16) | (air['hour'] ==17))] 

trial = air[((air['year']==2020)) & ((air['week']==1)|(air['week']==2)|(air['week']==3)|(air['week']==4)|(air['week']==5)|(air['week']==6)|(air['week']==7)|(air['week']==9)|(air['week']==10)|(air['week']==11)|(air['week']==12)|(air['week']==13)|(air['week']==14)|(air['week']==15)|(air['week']==18)|(air['week']==19)|(air['week']==20)|(air['week']==22)|(air['week']==34)|(air['week']==35)|(air['week']==36)|(air['week']==37)|(air['week']==38)|(air['week']==43)|(air['week']==44)|(air['week']==45)|(air['week']==46)|(air['week']==47))
& ((air['hour'] ==12) | (air['hour'] ==13) | (air['hour'] ==14) |(air['hour'] ==15) | (air['hour'] ==16) | (air['hour'] ==17))] 


p=(trial['total_co2'])**-1
q = trial['d13C']
c, m = polyfit(p,q,1)
# RMSE = sqrt(mean_squared_error(q, m*p + c))
slope, intercept, r_value, p_value, std_err = stats.linregress(p, q)
print('R-squared: ', r_value**2, '\nIntercept:', intercept)

plt.plot(p, q, '+')
plt.plot(p, c+m*p, '-')
plt.title('week - 52')
plt.ylabel('d 13C')
plt.xlabel('1/co2')
textstr = '\n'.join((
    r'$R^2=%.2f$' % (r_value**2, ),
    r'$C=%.2f$' % (intercept, )))

plt.text(0.0019,-9.5, textstr)
plt.locator_params(axis='x', nbins=7)
# plt.savefig('Week 52 - 2019.png',dpi =300, bbox_inches='tight' )
plt.show()

# # Y = -36 + 11700*p
# # plt.plot(p,Y, '--') 

#%% Merging accepted dates with total_co2 data

df18 = air_alg[((air_alg['year']=='2018')) & ((air_alg['hour']=='12')|(air_alg['hour']=='13')|(air_alg['hour']=='14')|(air_alg['hour']=='15')|
               (air_alg['hour']=='16')|(air_alg['hour']=='17'))]

df18['datetime'] = pd.to_datetime(df18['datetime'], format = '%Y-%m-%d %H:%M:%S')
df18['date'] = df18['datetime'].dt.date
accepted_dates18['period'] = pd.to_datetime(accepted_dates18['period'], format = '%d-%m-%Y')
accepted_dates18['date'] = accepted_dates18['period'].dt.date

new_df = accepted_dates18.merge(df18, on = 'date')

weather18 = pd.read_csv('2018_weather_data.txt', delimiter=',')
weather18.columns = ['datetime','Interval', 'Indoor Humidity', 'Indoor Temp', 'Outdoor Humidity', 'Outdoor Temp', 'Pressure',
                      'Wind Speed', 'Gust','Wind Direction', 'r1','r2']
weather18['datetime'] = pd.to_datetime(weather18['datetime'], format = '%Y-%m-%d %H:%M:%S')

weather18['date'] = weather18['datetime'].dt.date

weather18_co2 = new_df.merge(weather18, on = 'date')

#%%
weather18_co2['Wind Direction'] = weather18_co2['Wind Direction'].apply(pd.to_numeric)
# weather18_co2['d13C'] = weather18_co2['d13C'].apply(pd.to_numeric)
ax = WindroseAxes.from_ax()
ax.bar(weather18_co2['Wind Direction']*360/16, weather18_co2['total_co2'], normed = True, opening = 0.8, edgecolor ='black' )
ax.set_legend()
#%% Windrose Plots for keeling plots 2018

weather18 = pd.read_csv('2018_weather_data.txt', delimiter=',')
weather18.columns = ['datetime','Interval', 'Indoor Humidity', 'Indoor Temp', 'Outdoor Humidity', 'Outdoor Temp', 'Pressure',
                      'Wind Speed', 'Gust','Wind Direction', 'r1','r2']
weather18['datetime'] = pd.to_datetime(weather18['datetime'], format = '%Y-%m-%d %H:%M:%S')

weather18['month'] = weather18['datetime'].dt.month
weather18['day'] = weather18['datetime'].dt.day
weather18['minute'] = weather18['datetime'].dt.minute

df = weather18.groupby(weather18.datetime.dt.day).mean()

# air_afternoon['datetime'] = air_afternoon['datetime'].dt.round('min')
# weather18['datetime'] = weather18['datetime'].dt.round('min')
# weather18_co2 = pd.merge(air_afternoon, weather18, on = 'datetime', how='left', suffixes =('_TOT', '_weather'))

# weather18_co2['Wind Direction'] = weather18_co2['Wind Direction'].astype(str)
# weather18_co2 = weather18_co2[weather18_co2['Wind Direction'].str.contains('nan') !=True]
# weather18_co2['Wind Direction'] = weather18_co2['Wind Direction'].apply(pd.to_numeric)

# # weather_event = weather18_co2[((weather18_co2['day_TOT']==31) & (weather18_co2['month_TOT']==10))]


# ax = WindroseAxes.from_ax()
# ax.bar(weather_event['Wind Direction']*360/16, weather_event['d13C'], normed = True, opening = 0.8, edgecolor ='black' )
# ax.set_legend()
# # ax.set_title('d13C relative to wind direction - 23/03/2020')


#%% Windrose plots for KP 2019

weather19 = pd.read_csv('2019_weather_data.txt', delimiter=',')
weather19.columns = ['datetime','Interval', 'Indoor Humidity', 'Indoor Temp', 'Outdoor Humidity', 'Outdoor Temp', 'Pressure',
                      'Wind Speed', 'Gust','Wind Direction', 'r1','r2']

weather19['datetime'] = pd.to_datetime(weather19['datetime'], format = '%Y-%m-%d %H:%M:%S')
accepted_dates19['datetime'] = pd.to_datetime(accepted_dates19['datetime'], format = '%Y-%m-%d')

accepted_dates19['dayofyear'] = accepted_dates19['datetime'].dt.dayofyear
# weather19['month'] = weather19['datetime'].dt.month
# weather19['day'] = weather19['datetime'].dt.day
# weather19['week'] = weather19['datetime'].dt.week
# weather19['minute'] = weather19['datetime'].dt.minute
weather19['dayofyear'] = weather19['datetime'].dt.dayofyear


# air_afternoon19 =air_afternoon[(air_afternoon['year']==2019)]

# # air_afternoon19['datetime'] = air_afternoon['datetime'].dt.round('min')
# # weather19['datetime'] = weather19['datetime'].dt.round('min')
weather19_co2 = pd.merge(weather19 , air_alg19, on = 'dayofyear', how='left', suffixes =('_TOT', '_weather')) 

weather19_co2['Wind Direction'] = weather19_co2['Wind Direction'].astype(str)
weather19_co2 = weather19_co2[weather19_co2['Wind Direction'].str.contains('nan') !=True]
weather19_co2['Wind Direction'] = weather19_co2['Wind Direction'].apply(pd.to_numeric)

# event = weather19_co2[((weather19_co2['day_TOT']==18) & (weather19_co2['month_TOT']==1))]
# ax = WindroseAxes.from_ax()
# ax.bar(event['Wind Direction']*360/16, event['Wind Speed'], normed = True, opening =0.8, edgecolor ='black')
# # ax.bar(weather19_co2['Wind Direction']*360/16, weather19_co2['total_co2'], normed = True, opening = 0.8)
# ax.set_legend()
# ax.set_title('Wind speed direction - 18/01/2019', fontsize = 16)
# plt.savefig('18-01-2019 speed windrose.png', dpi = 300, bbox_inches='tight')

#%%    
# A = weather19_co2[(weather19_co2['Wind Direction']==0)]
# A1 = weather19_co2[(weather19_co2['Wind Direction']==1)]
# A2 = weather19_co2[(weather19_co2['Wind Direction']==2)]
# A3 = weather19_co2[(weather19_co2['Wind Direction']==3)]
# A4 = weather19_co2[(weather19_co2['Wind Direction']==4)]
# A5 = weather19_co2[(weather19_co2['Wind Direction']==5)]
# A6 = weather19_co2[(weather19_co2['Wind Direction']==6)]
# A7 = weather19_co2[(weather19_co2['Wind Direction']==7)]
# A8 = weather19_co2[(weather19_co2['Wind Direction']==8)]
# A9 = weather19_co2[(weather19_co2['Wind Direction']==9)]
# A10 = weather19_co2[(weather19_co2['Wind Direction']==10)]
# A11 = weather19_co2[(weather19_co2['Wind Direction']==11)]
# A12 = weather19_co2[(weather19_co2['Wind Direction']==12)]
# A13 = weather19_co2[(weather19_co2['Wind Direction']==13)]
# A14 = weather19_co2[(weather19_co2['Wind Direction']==14)]
# A15 = weather19_co2[(weather19_co2['Wind Direction']==15)]
# A16 = weather19_co2[(weather19_co2['Wind Direction']==16)]

# plt.plot(A['datetime'],A['total_co2'])
# plt.plot(A1['datetime'],A1['total_co2'])
# plt.plot(A2['datetime'],A2['total_co2'])
# plt.plot(A3['datetime'],A3['total_co2'])
# plt.plot(A4['datetime'],A4['total_co2'])
# plt.plot(A5['datetime'],A5['total_co2'])
# plt.plot(A6['datetime'],A6['total_co2'])
# plt.plot(A7['datetime'],A7['total_co2'])
# plt.show()
#%%


weather18 = pd.read_csv('2018_weather_data.txt', usecols = ['0', '8', '9'], dtype = {'8':np.float64, '9':np.float64}) 
weather18.columns =['datetime','wind speed', 'wind direction']
 
weather18['datetime'] = pd.to_datetime(weather18['datetime'], format = '%Y-%m-%d %H:%M:%S')
weather18['day'] = weather18['datetime'].dt.day
weather18['month'] = weather18['datetime'].dt.month
weather18['year'] = weather18['datetime'].dt.year
weather18['dmy'] = weather18['day'].astype(str) +'-'+ weather18['month'].astype(str) +'-'+ weather18['year'].astype(str)

df1 = accepted_dates18.merge(weather18, left_on = 'period', right_on='dmy')

ax = WindroseAxes.from_ax()
ax.bar(df1['wind direction']*360/16, df1['intercept'], normed = True, opening = 0.8, edgecolor ='black' )
plt.title('2018 source signarture directions - whole day')
ax.set_legend()
# plt.savefig('2018 source signarture directions- whole day', dpi = 300, bbox_inches = 'tight')


#%% Algorithm new

for d in air_alg18['dayofyear'].unique(): # this will pass through each day, the .unique() ensures that it doesnt go over the same days     
    acceptable_date = {} # creating a dictionary to store the valid dates 
    period = air_alg18[((air_alg18['dayofyear']==d) & (air_alg18['dayofyear']==d+1) & (air_alg18['dayofyear']==d+2))] # defining each period from the dmy column
    # df.iloc[d:d+3]
#     p = (period['total_co2'])**-1
#     q = period['d13C']
#     c,m = polyfit(p,q,1)
#     slope, intercept, r_value, p_value, std_err = stats.linregress(p, q)
#     RMSE = sqrt(mean_squared_error(q, m*p + c))
    
    
#     if r_value**2 >= 0.8:
#         acceptable_date['period'] = d # populating the dictionary with the accpeted dates and corresponding other values
#         acceptable_date['r-squared'] = r_value**2
#         acceptable_date['intercept'] = intercept
#         acceptable_date['RMSE'] = RMSE
#         accepted_dates_list.append(acceptable_date) # sending the valid stuff in the dictionary to the list
#     else:
#         pass


# accepted_dates18 = pd.DataFrame(accepted_dates_list) # converting the list to a df

#%% Eric's lines
window = air_alg[((air_alg['year']=='2018')) & ((air_alg['hour']==12 |air_alg['hour']==13 |air_alg['hour']==14 |air_alg['hour']==15 |
                 air_alg['hour']==16 |air_alg['hour']==17))]    
time = window.loc[:,'datetime']

for i in time:
    t_copy = [time[i]] * len(time) 
    t_delta = np.array(time) - np.array(t_copy)

#      find indices that sit within the window
inds = []
for j in range(len(t_delta)):
    if np.abs(t_delta[j].total_seconds()) <= window:
        inds.append(j)

#%% Mode of wind direction

        
weather18 = pd.read_csv('2018_weather_data.txt', usecols = ['0', '8', '9'], dtype = {'8':np.float64, '9':np.float64}) 
weather18.columns =['datetime','wind_speed', 'wind_direction']
weather18['datetime'] = pd.to_datetime(weather18['datetime'], format = '%Y-%m-%d %H:%M:%S')

weather18['day'] = weather18['datetime'].dt.day
weather18['month'] = weather18['datetime'].dt.month
weather18['year'] = weather18['datetime'].dt.year
weather18['week'] = weather18['datetime'].dt.week
weather18['hour'] = weather18['datetime'].dt.hour

weather18['dmy'] = weather18['day'].astype(str) +'-'+ weather18['month'].astype(str) +'-'+ weather18['year'].astype(str)

weather18.dropna(subset = ["wind_direction"], inplace=True)

weather18_noon = weather18[(weather18['hour']==12)|(weather18['hour']==13)|(weather18['hour']==14)|(weather18['hour']==15)|(weather18['hour']==16)|
                           (weather18['hour']==17)]



# accepted_dates18['datetime'] = pd.to_datetime(accepted_dates18['period'], format = '%d-%m-%Y')
# accepted_dates18['dayofyear'] = accepted_dates18['datetime'].dt.dayofyear

# df = weather18.groupby(weather18.datetime.dt.dayofyear).mean()

values_list = []
for d in weather18_noon['week'].unique():
    values={}
    period = weather18_noon[weather18_noon.week==d] 
    Q = period.mode().wind_direction[0]
    
    values['week'] = d
    values['mode_wind_direction'] = Q
    values_list.append(values)

df18 = pd.DataFrame(values_list)  

accepted_dates18['period'] = accepted_dates18['period'].astype(str).astype(int)  # converting the period column from object to int64
# # print(df18)
new_df = accepted_dates18.merge(df18,right_on='week' ,left_on = 'period')

ax = WindroseAxes.from_ax()
ax.bar(new_df['mode_wind_direction']*360/16, new_df['intercept'], normed = True, opening = 0.8, edgecolor ='black')
ax.set_legend()
ax.set_title('Source signature wind direction - 7 day afternoon 2018', fontsize = 14)
# plt.savefig('Source signature wind direction - 7 day noon 2018.png', dpi = 300, bbox_inches = 'tight')

#%% Mode wind direction 2019

weather19 = pd.read_csv('2019_weather_data.txt', usecols = ['0', '8', '9'], dtype = {'8':np.float64, '9':np.float64}) 
weather19.columns =['datetime','wind_speed', 'wind_direction']
weather19['datetime'] = pd.to_datetime(weather19['datetime'], format = '%Y-%m-%d %H:%M:%S')

weather19['day'] = weather19['datetime'].dt.day
weather19['month'] = weather19['datetime'].dt.month
weather19['year'] = weather19['datetime'].dt.year
weather19['week'] = weather19['datetime'].dt.week
weather19['hour'] = weather19['datetime'].dt.hour
weather19['dmy'] = weather19['day'].astype(str) +'-'+ weather19['month'].astype(str) +'-'+ weather19['year'].astype(str)

weather19.dropna(subset = ["wind_direction"], inplace=True)

weather19_noon = weather19[(weather19['hour']==12)|(weather19['hour']==13)|(weather19['hour']==14)|(weather19['hour']==15)|(weather19['hour']==16)|
                           (weather19['hour']==17)]

# accepted_dates19['datetime'] = pd.to_datetime(accepted_dates19['period'], format = '%d-%m-%Y')

values_list = []

for d in weather19_noon['week'].unique():
    values={}
    period = weather19_noon[weather19_noon.week==d] 
    Q = period.mode().wind_direction[0]
    
    values['period'] = d
    values['mode_wind_direction'] = Q
    values_list.append(values)

df19 = pd.DataFrame(values_list)    
accepted_dates19['period'] = accepted_dates19['period'].astype(str).astype(int)

new_df19 = accepted_dates19.merge(df19, on = 'period')

ax = WindroseAxes.from_ax()
ax.bar(new_df19['mode_wind_direction']*360/16, new_df19['intercept'], normed = True, opening = 0.8, edgecolor ='black' )
ax.set_legend()
ax.set_title('Source signature wind direction - 7 day afternoon 2019', fontsize = 14)
plt.savefig('Source signature wind direction - 7 day 2019.png', dpi = 300, bbox_inches = 'tight')

#%% Mode wind direction 2020

weather20 = pd.read_csv('2020_weather_data.csv', usecols = ['0', '8', '9'], dtype = {'8':np.float64, '9':np.float64}) 
weather20.columns =['datetime','wind_speed', 'wind_direction']
weather20['datetime'] = pd.to_datetime(weather20['datetime'], format = '%d/%m/%Y %H:%M')

weather20['year'] = weather20['datetime'].dt.year
weather20['month'] = weather20['datetime'].dt.month
weather20['day'] = weather20['datetime'].dt.day
weather20['hour'] = weather20['datetime'].dt.hour
weather20['week'] = weather20['datetime'].dt.week
weather20['dmy'] = weather20['day'].astype(str) +'-'+ weather20['month'].astype(str) +'-'+ weather20['year'].astype(str)


weather20.dropna(subset = ["wind_direction"], inplace=True)
# accepted_dates20['period1'] = accepted_dates20['period1'].astype(str).astype(int) # converts the week number into an int

weather20_noon = weather20[(weather20['hour']==12)|(weather20['hour']==13)|(weather20['hour']==14)|(weather20['hour']==15)|(weather20['hour']==16)|
                            (weather20['hour']==17)]


values_list = []

for d in weather20_noon['dmy'].unique():
    values={}
    period = weather20_noon[weather20_noon.dmy==d] 
    Q = period.mode().wind_direction[0]
    
    values['period'] = d
    values['mode_wind_direction'] = Q
    values_list.append(values)

df20 = pd.DataFrame(values_list)    

new_df20 = accepted_dates20.merge(df20, left_on = 'period1', right_on = 'period')

ax = WindroseAxes.from_ax()
ax.bar(new_df20['mode_wind_direction']*360/16, new_df20['intercept'], normed = True, opening = 0.8, edgecolor ='black' )
ax.set_legend()
ax.set_title('Source signature wind direction - 3 day afternoon 2020', fontsize = 15)
# plt.savefig('3-day wind direction 2020.png', dpi = 300, bbox_inches = 'tight')

#%% 3 day windrose - mode direction 2019

weather19 = pd.read_csv('2019_weather_data.txt', usecols = ['0', '8', '9'], dtype = {'8':np.float64, '9':np.float64}) 
weather19.columns =['datetime','wind_speed', 'wind_direction']
weather19['datetime'] = pd.to_datetime(weather19['datetime'], format = '%Y-%m-%d %H:%M:%S')

weather19['dayofyear'] = weather19['datetime'].dt.dayofyear
weather19['day'] = weather19['datetime'].dt.day
weather19['month'] = weather19['datetime'].dt.month
weather19['year'] = weather19['datetime'].dt.year
weather19['week'] = weather19['datetime'].dt.week
weather19['hour'] = weather19['datetime'].dt.hour
weather19['dmy'] = weather19['day'].astype(str) +'-'+ weather19['month'].astype(str) +'-'+ weather19['year'].astype(str)

weather19.dropna(subset = ["wind_direction"], inplace=True)

weather19_noon = weather19[(weather19['hour']==12)|(weather19['hour']==13)|(weather19['hour']==14)|(weather19['hour']==15)|(weather19['hour']==16)|
                           (weather19['hour']==17)]

first_day = 9
days_to_group = 3
values_list = []
for doy, gdf in weather19_noon.groupby((weather19_noon.datetime.dt.dayofyear.sub(first_day) // days_to_group)* days_to_group + first_day):
    values={}
    Q = gdf.mode().wind_direction[0]
    
    values['date1'] = gdf.dmy.min()
    # values['date2'] = gdf.dmy.max()
    values['mode_wind_direction'] = Q
    values_list.append(values)
    # print(gdf, '\n')

df19 = pd.DataFrame(values_list)    
# accepted_dates19['period'] = accepted_dates19['period'].astype(str).astype(int)

new_df19 = accepted_dates19.merge(df19, left_on = 'period1', right_on = 'date1')

ax = WindroseAxes.from_ax()
ax.bar(new_df19['mode_wind_direction']*360/16, new_df19['intercept'], normed = True, opening = 0.8, edgecolor ='black' )
ax.set_legend(fontsize = 16)
ax.set_title('3-day soruce signature direction 2019', fontsize = 14)
# plt.savefig('3-day source signature direction 2019.png', dpi =300, bbox_inches ='tight')

#%% 3 day windrose - mode direction 2018

weather18 = pd.read_csv('2018_weather_data.txt', usecols = ['0', '8', '9'], dtype = {'8':np.float64, '9':np.float64}) 
weather18.columns =['datetime','wind_speed', 'wind_direction']
weather18['datetime'] = pd.to_datetime(weather18['datetime'], format = '%Y-%m-%d %H:%M:%S')

weather18['dayofyear'] = weather18['datetime'].dt.dayofyear
weather18['day'] = weather18['datetime'].dt.day
weather18['month'] = weather18['datetime'].dt.month
weather18['year'] = weather18['datetime'].dt.year
weather18['week'] = weather18['datetime'].dt.week
weather18['hour'] = weather18['datetime'].dt.hour

weather18['dmy'] = weather18['day'].astype(str) +'-'+ weather18['month'].astype(str) +'-'+ weather18['year'].astype(str)

weather18.dropna(subset = ["wind_direction"], inplace=True)

weather18_noon = weather18[(weather18['hour']==12)|(weather18['hour']==13)|(weather18['hour']==14)|(weather18['hour']==15)|(weather18['hour']==16)|
                           (weather18['hour']==17)]

first_day = 5
days_to_group = 3
values_list = []
for doy, gdf in weather18_noon.groupby((weather18_noon.datetime.dt.dayofyear.sub(first_day) // days_to_group)* days_to_group + first_day):
    values={}
    Q = gdf.mode().wind_direction[0]
    
    values['date1'] = gdf.dmy.min()
    # values['date2'] = gdf.dmy.max()
    values['mode_wind_direction'] = Q
    values_list.append(values)
    # print(gdf, '\n')
    
df18 = pd.DataFrame(values_list)  


new_df18 = accepted_dates18.merge(df18, left_on = 'period1', right_on = 'date1')

ax = WindroseAxes.from_ax()
ax.bar(new_df18['mode_wind_direction']*360/16, new_df18['intercept'], normed = True, opening = 0.8, edgecolor ='black')
ax.set_legend(fontsize = 16)
ax.set_title('Source signature wind direction - 3 day afternoon 2018', fontsize = 14)
# plt.savefig('3-day source signature direction 2018.png', dpi =300, bbox_inches ='tight')

#%% 3 day windrose - mode direction 2020

weather20 = pd.read_csv('2020_weather_data.csv', usecols = ['0', '8', '9'], dtype = {'8':np.float64, '9':np.float64}) 
weather20.columns =['datetime','wind_speed', 'wind_direction']
weather20['datetime'] = pd.to_datetime(weather20['datetime'], format = '%d/%m/%Y %H:%M')

weather20['dayofyear'] = weather20['datetime'].dt.dayofyear
weather20['year'] = weather20['datetime'].dt.year
weather20['month'] = weather20['datetime'].dt.month
weather20['day'] = weather20['datetime'].dt.day
weather20['hour'] = weather20['datetime'].dt.hour
weather20['week'] = weather20['datetime'].dt.week

weather20['dmy'] = weather20['day'].astype(str) +'-'+ weather20['month'].astype(str) +'-'+ weather20['year'].astype(str)


weather20.dropna(subset = ["wind_direction"], inplace=True)
# accepted_dates20['period1'] = accepted_dates20['period1'].astype(str).astype(int) # converts the week number into an int

weather20_noon = weather20[(weather20['hour']==12)|(weather20['hour']==13)|(weather20['hour']==14)|(weather20['hour']==15)|(weather20['hour']==16)|
                            (weather20['hour']==17)]


first_day = 1
days_to_group = 3
values_list = []
for doy, gdf in weather20_noon.groupby((weather20_noon.datetime.dt.dayofyear.sub(first_day) // days_to_group)* days_to_group + first_day):
    values={}
    Q = gdf.mode().wind_direction[0]
    
    values['date1'] = gdf.dmy.min()
    # values['date2'] = gdf.dmy.max()
    values['mode_wind_direction'] = Q
    values_list.append(values)
    # print(gdf, '\n')
    
df20 = pd.DataFrame(values_list)  

new_df20 = accepted_dates20.merge(df20, left_on = 'period1', right_on = 'date1')

ax = WindroseAxes.from_ax()
ax.bar(new_df20['mode_wind_direction']*360/16, new_df20['intercept'], normed = True, opening = 0.8, edgecolor ='black')
ax.set_legend(fontsize = 16)
ax.set_title('Source signature wind direction - 3 day afternoon 2020', fontsize = 14)



#%% Stack overflow trials
# from datetime import datetime, timedelta
# window18['datetime'] = pd.to_datetime(window18['datetime'])

# init_date = datetime(2018, 1, 5)
# end_date = init_date + timedelta(days=3)
# window18[(window18['datetime']>=init_date) & (window18['datetime']<end_date)]



#%%
window18 = air_alg[((air_alg['year']=='2018')) 
& ((air_alg['hour'] =='12') | (air_alg['hour'] =='13') | (air_alg['hour'] =='14')|(air_alg['hour'] =='15') | (air_alg['hour'] =='16') | (air_alg['hour'] =='17'))]
# window18['dmy'] = pd.to_datetime(window18['dmy'])

accepted_dates_list = [] # creating an empty list to store the dates that we're interested in


init_date = datetime(2018, 1, 5)

for d in window18['dmy'].unique(): # this will pass through each day, the .unique() ensures that it doesnt go over the same days  
    acceptable_date = {} # creating a dictionary to store the valid dates
    end_date = init_date + timedelta(days=2)
    if d.between_time('init_date', 'end_date'):
        period = window18[window18.dmy==range(init_date,end_date)] # defining each period from the dmy column
        p = (period['total_co2'])**-1
        q = period['d13C']
        c,m = polyfit(p,q,1)
        slope, intercept, r_value, p_value, std_err = stats.linregress(p, q)
        # RMSE = sqrt(mean_squared_error(q, m*p + c))
    
        if r_value**2 >= 0.8:
            acceptable_date['period1'] = end_date # populating the dictionary with the accpeted dates and corresponding other values
            acceptable_date['period2'] = init_date
            acceptable_date['r-squared'] = r_value**2
            acceptable_date['intercept'] = intercept
            accepted_dates_list.append(acceptable_date) # sending the valid stuff in the dictionary to the list
        else:
            pass
        init_date = end_date
        

    
accepted_dates18 = pd.DataFrame(accepted_dates_list) # converting the list to a df
# print(accepted_dates)
'''Probably need to look into how the format of d as it's not the same datetime format as the initial and end dates'''

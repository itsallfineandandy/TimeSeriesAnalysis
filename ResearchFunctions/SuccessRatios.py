import datetime
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from random import random
from matplotlib import pyplot
from pmdarima.arima import auto_arima
import tsfresh

## Set Parameters
trainstartdate = datetime.datetime(2021, 5, 24)
teststartdate = datetime.datetime(2023, 5, 24)

## Read CSV
rawdata = pd.read_csv("/Users/andrewhaley/PythonProjects/TimeSeriesAnalysis/SampleData/US50015.csv",
                      names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])

# Set date time
rawdata["DateTime"] = pd.to_datetime(rawdata['Date'] + ' ' + rawdata['Time'])
rawdata.index = rawdata["DateTime"]

## Create Performance
rawdata["ForwardPerf"] = ((rawdata.Close.shift(1)/rawdata.Close)-1)*100
rawdata["Price_Change"] = ((rawdata.Close.shift(-1)/rawdata.Close)-1)*100
rawdata["Volume_Change"] = ((rawdata.Volume.shift(-1)/rawdata.Volume)-1)*100

## Calc Hourly Success
hourlysuccess = rawdata[["Time","ForwardPerf","Volume","Volume_Change","Price_Change"]].reset_index(drop=True).dropna()
hourlysuccess["Price_Change"] = np.where(hourlysuccess.Price_Change>0, True, False)
hourlysuccess["Volume_Change"] = np.where(hourlysuccess.Volume_Change>0, True, False)

## Group by and loop
def countsuccess(x):
    m = np.count_nonzero(x > 0)/np.count_nonzero(x)
    return m
grouped = hourlysuccess.groupby('Time')
QuantileResult = []
for name, group in grouped:

    ## Calc Quantiles
    q25 = np.quantile(group.Volume, .25)
    q50 = np.quantile(group.Volume, .5)
    q75 = np.quantile(group.Volume, .75)

    ## Group by quantile
    group["Quantile"] = np.where(group.Volume < q25, 1,
                                 np.where(group.Volume < q50, 2,
                                          np.where(group.Volume < q75, 3, 4)))
    group["Volume_Quantile_Change"] = np.where(group.Quantile.shift(1) < group.Quantile, True, False)

    ## Result
    result = group.groupby(["Quantile","Volume_Quantile_Change","Volume_Change","Price_Change"])["ForwardPerf"].agg(
        ["min", "max", "mean", "median", "std", np.count_nonzero, countsuccess]).reset_index()
    result["Time"] = name

    QuantileResult.append(result)
QuantileResult = pd.concat(QuantileResult)
# QuantileResult = QuantileResult.sort_values(["Volume_Change","Price_Change","countsuccess"])
QuantileResult = QuantileResult.sort_values(["Time","Quantile"])







## Calculate Quantiles
def norm_group(group):
    g_min = group.Volume.values.min()
    g_max = group.Volume.values.max()
    group["quantile"] = (group.Volume.values - g_min) / (g_max - g_min)
    return group
hourlysuccess = hourlysuccess.groupby(['Time']).apply(norm_group).reset_index()

np.quantile(.25)



def countsuccess(x):
    m = np.count_nonzero(x > 0)/np.count_nonzero(x)
    return m
hourlysuccess = hourlysuccess.groupby(['Time'])["ForwardPerf"].agg([np.min,np.max,np.mean,np.median,np.std,np.count_nonzero,countsuccess])
hourlysuccess = hourlysuccess.sort_values("countsuccess")
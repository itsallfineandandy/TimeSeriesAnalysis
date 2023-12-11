import datetime
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

## Set target series
returnseries = rawdata.Close.dropna()

## Run Stationarity Test
stationarity_result = adfuller(returnseries)

# Build stationary series
if stationarity_result[1] > 0.05:
    returnseries_raw = returnseries
    returnseries = returnseries.pct_change().dropna()

## Run AR Model
# Build initial data set
testdata = returnseries[returnseries.index < teststartdate]

# Build initial model
looplag = 1
newaic = 100000000000
while True:

    # fit model
    model = AutoReg(testdata, lags = looplag)
    ar_model_fit = model.fit()
    prevaic = newaic
    newaic = ar_model_fit.aic
    looplag = looplag + 1

    if prevaic < newaic:

        # Refit previous model
        looplag = looplag-2
        model = AutoReg(testdata, lags=looplag)
        ar_model_fit = model.fit()

        break

# Make out of sample predictions
AR_predictions = []
for x in range(len(testdata),(len(returnseries)-1)):
    print(returnseries.index[x])
    testdata = returnseries[1:x]
    model = AutoReg(testdata, lags=looplag)
    ar_model_fit = model.fit()
    tmp_predictions = pd.DataFrame(ar_model_fit.predict(start=len(testdata), end=len(testdata))).reset_index(drop=True)
    tmp_predictions.columns = ["Predictions"]
    tmp_predictions = (tmp_predictions + 1) * returnseries_raw[x + 1]
    tmp_predictions["Actual"] = returnseries_raw[x+2]
    AR_predictions.append(tmp_predictions)
AR_predictions = pd.concat(AR_predictions)
AR_predictions = AR_predictions.reset_index(drop = True)
model = LinearRegression().fit(AR_predictions.Predictions.values.reshape(-1, 1), AR_predictions.Actual.values.reshape(-1, 1))
r_sq_ar = model.score(AR_predictions.Predictions.values.reshape(-1, 1), AR_predictions.Actual.values.reshape(-1, 1))

## ARMA/ARIMA/SARIMA Model
# Build initial data set
testdata = returnseries_raw[returnseries_raw.index < teststartdate]

# Make Initial Fit
auto = auto_arima(testdata, seasonal=False, stepwise=True,
                  suppress_warnings=True, error_action="ignore", max_p=10, max_q=10,
                  max_order=None, trace=True)

# Make out of sample predictions
ARMA_predictions = []
for x in range(len(testdata),(len(returnseries_raw)-1)):
    print(returnseries_raw.index[x])
    testdata = returnseries_raw[1:x]
    model = ARIMA(testdata, order=auto.order)
    arima_model_fit = model.fit()
    tmp_predictions = pd.DataFrame(arima_model_fit.predict(x+1)).reset_index(drop=True)
    tmp_predictions.columns = ["Predictions"]
    tmp_predictions["Actual"] = returnseries_raw[x]
    ARMA_predictions.append(tmp_predictions)
ARMA_predictions = pd.concat(ARMA_predictions)
ARMA_predictions = ARMA_predictions.reset_index(drop = True)
model = LinearRegression().fit(ARMA_predictions.Predictions.values.reshape(-1, 1), ARMA_predictions.Actual.values.reshape(-1, 1))
r_sq_arima = model.score(ARMA_predictions.Predictions.values.reshape(-1, 1), ARMA_predictions.Actual.values.reshape(-1, 1))




# results = pd.DataFrame(returnseries)
# results["Predictions"] = auto.predict_in_sample()
# results.columns = ["Actual", "Predictions"]
# model = LinearRegression().fit(results.Predictions.values.reshape(-1, 1), results.Actual.values.reshape(-1, 1))
# r_sq_ARIMA = model.score(results.Predictions.values.reshape(-1, 1), results.Actual.values.reshape(-1, 1))
#
# ## Fit VAR Model
#
# # automated feature generation
# returnseries_df = pd.DataFrame(returnseries).reset_index().rename(columns={'index':'TmpIndex',0:'Target'})
# features = tsfresh.extract_features(returnseries_df,
#                                     column_id = 'TmpIndex',
#                                     n_jobs = 0)
# fullfeatures_df = returnseries_df.merge(features, left_index = True, right_index = True).dropna(axis=1, how='all')
# fullfeatures_df = fullfeatures_df.loc[:, (fullfeatures_df != 0).any(axis=0)]
# fullfeatures_df = fullfeatures_df.loc[:, (fullfeatures_df != 1).any(axis=0)]
# fullfeatures_df = fullfeatures_df.drop(columns = "TmpIndex")
# fullfeatures_df = fullfeatures_df.dropna()
# fullfeatures_df = fullfeatures_df.T.drop_duplicates(keep="first").T
# fullfeatures_df = fullfeatures_df.loc[:, (fullfeatures_df != fullfeatures_df.iloc[0]).any()]
# model = VAR(fullfeatures_df)
# model_fit_var = model.fit()
#
# predictions = model_fit_var.fittedvalues.Target
# results = pd.DataFrame(returnseries[-len(predictions):], predictions).reset_index().dropna().reset_index(drop=True)
# results.columns = ["Actual", "Predictions"]
# model = LinearRegression().fit(results.Predictions.values.reshape(-1, 1), results.Actual.values.reshape(-1, 1))
# r_sq_VAR = model.score(results.Predictions.values.reshape(-1, 1), results.Actual.values.reshape(-1, 1))
#
#
#
# # SES example
# from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# from random import random
#
# model = SimpleExpSmoothing(rawdata.Close)
# model_fit = model.fit()
#
# results = pd.DataFrame(rawdata.Close[-len(model_fit.fittedvalues):])
# results["Predictions"] = model_fit.fittedvalues
# model = LinearRegression().fit(predictions.values.reshape(-1, 1), results.Close.values.reshape(-1, 1))
# r_sq_SES = model.score(predictions.values.reshape(-1, 1), results.Close.values.reshape(-1, 1))
#
#
#
# # HWES example
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# model = ExponentialSmoothing(rawdata.Close)
# model_fit = model.fit()
# predictions = model_fit.fittedvalues
# results = pd.DataFrame(rawdata.Close[-len(model_fit.fittedvalues):])
# results["Predictions"] = model_fit.fittedvalues
# model = LinearRegression().fit(predictions.values.reshape(-1, 1), results.Close.values.reshape(-1, 1))
# r_sq_HWES = model.score(predictions.values.reshape(-1, 1), results.Close.values.reshape(-1, 1))
#

# features = tsfresh.extract_features(returnseries, column_id="date", column_sort="date")
#
#
# pd.DataFrame(returnseries).reset_index().rename(columns={'index':'TmpIndex',0:'Target'})
#
# df.reset_index().rename(columns={df.index.name:'bar'})
#
#
#
# model = VAR(returnseries)
# model_fit_var = model.fit()
# # make prediction
# yhat = model_fit.forecast(model_fit.y, steps=1)
# print(yhat)




# # SARIMAX example
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# # fit model
# model = SARIMAX(returnseries_raw, order=(2, 1, 1))
# model_fit = model.fit(disp=False)
#
# results = pd.DataFrame(returnseries)
# results["Predictions"] = model_fit.
# results.columns = ["Actual", "Predictions"]
# model = LinearRegression().fit(results.Predictions.values.reshape(-1, 1), results.Actual.values.reshape(-1, 1))
# r_sq = model.score(results.Predictions.values.reshape(-1, 1), results.Actual.values.reshape(-1, 1))


# looplag = 1
# newaic = -100000000000
# while True:
#
#     # fit model
#     print(looplag)
#     model = ARIMA(returnseries, order = (looplag,0,0))
#     model_fit = model.fit()
#     prevaic = newaic
#     newaic = model_fit.aic
#     looplag = looplag + 1
#
#     print(newaic)
#
#     if prevaic > newaic:
#         break
#
# # make predictions
# predictions = model_fit.fittedvalues
# results = pd.DataFrame(predictions, returnseries).reset_index().dropna().reset_index(drop=True)
# results.columns = ["Actual", "Predictions"]
# model = LinearRegression().fit(results.Predictions.values.reshape(-1, 1), results.Actual.values.reshape(-1, 1))
# r_sq = model.score(results.Predictions.values.reshape(-1, 1), results.Actual.values.reshape(-1, 1))
#
#
# looplag = 1
# newaic = -100000000000
# while True:
#
#     # fit model
#     print(looplag)
#     model = ARIMA(returnseries, order = (looplag,0,1))
#     model_fit = model.fit()
#     prevaic = newaic
#     newaic = model_fit.aic
#     looplag = looplag + 1
#
#     print(newaic)
#
#     if prevaic > newaic:
#         break
#
# # make predictions
# predictions = model_fit.fittedvalues
# results = pd.DataFrame(predictions, returnseries).reset_index().dropna().reset_index(drop=True)
# results.columns = ["Actual", "Predictions"]
# model = LinearRegression().fit(results.Predictions.values.reshape(-1, 1), results.Actual.values.reshape(-1, 1))
# r_sq = model.score(results.Predictions.values.reshape(-1, 1), results.Actual.values.reshape(-1, 1))
#

"""Helper Methods for SigMet Time Series Analyzer

This script includes some helper methods to help ensure an accurate and
plausible measure of the shocks within a given time series.

This module accepts a pandas Series and assumes that it contains a DateTime
Index.

This script requires that `pandas`, `numpy`, `tqdm`, `matplotlib`, `sklearn`,
`statsmodels`, and `scipy` be installed within the Python environment within
which you are running this script.

This file can be imported as a module and contains the following functions:

    * standardize
    * find_start
    * my_min
    * ARIMA_50
    * find_end
    * calc_resid
    * find_AU3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from scipy.signal import argrelextrema, argrelmax

plt.style.use("seaborn")


def standardize(series):
    """Standardizes a DataFrame's column

    Parameters
    ----------
    series : Series
        The Series object we want to standardize

    Returns
    -------
    ndarray
        Transformed array
    """

    # Initialize standard scaler
    scaler = StandardScaler()

    # Reshape our series
    train = series.reshape(-1, 1)

    return scaler.fit_transform(train)


def find_start(series, user_start, user_end, ma_window=6):
    """Gets start date of dip in TS, measured as the largest local maximum for
    a given element in the database

    Parameters
    ----------
    series : pd.Series
        Find the starting date of a given series
    start_date : pd.DateTime
        The start_date at which to run my_min
    end_date : pd.DateTime
        The cutoff date for the series
    threshold : int
        Threshold for my_min, called within this method

    Returns
    -------
    pd.DateTime
        The start date
    """

    # Get the most recent date and filter series
    series_filtered = series[(
        series.index >= user_start) & (
            series.index <= user_end)]

    ser_smoothed = series_filtered.rolling(ma_window).mean()
    max_indices = argrelmax(ser_smoothed.values)[0]
    max_indices_not_smoothed = argrelmax(series_filtered.values)[0]

    if (len(max_indices) > 0):
        first_max_index = max_indices[0]
        global_min = ser_smoothed[first_max_index:].min()
        global_min_index = ser_smoothed.values[first_max_index:].argmin()
        valid_max_indices = list(
            max_indices[(
                max_indices >= first_max_index) & (
                    max_indices < global_min_index + first_max_index)])
        max_idx = list(ser_smoothed[valid_max_indices].values - global_min)
        max_idx = max_idx.index(min(max_idx)) + first_max_index
        maxes_not_smoothed = list(max_indices_not_smoothed[
            :(max_idx + ma_window)])
        compare = list(abs(maxes_not_smoothed - max_idx))
        final_max_index = maxes_not_smoothed[compare.index(min(compare))]
        return series_filtered.index[final_max_index]
    elif (len(max_indices_not_smoothed) > 0):
        return series_filtered.index[max_indices_not_smoothed[0]]
    return user_start



def SARIMAX_50(series, start_date, params=(5, 1, 1)):
    """Get an SARIMAX forecast for a given Series

    Parameters
    ----------
    series : pd.Series
        Time-series Series object containing DateTime index
    start_date : pd.DateTime
        DateTime object from index of df representing peak feature
    params : tuple
        p, d, and q parameters for SARIMAX

    Returns
    -------
    pd.Series
        Series of forecasts
    """

    #try:
    # Filter series
    before = series[series.index <= start_date]
    before.dropna(inplace=True)

    # Steps for ARIMA forecast
    steps = series.shape[0] - before.values.shape[0]

    # Initialize model
    model = ARIMA(before, order=params)
        
    # Fit the model
    model_fit = model.fit(disp=0)

    # Return the forecast as a pd.Series object
    return pd.Series(model_fit.forecast(steps))
    #except ValueError:
    #    raise ValueError("Cannot provide an SARIMAX forecast for given trend")


def find_end_forecast(series, start_date, SARIMAX_50):
    """
    Gets end date of dip in TS, measured as the first point of intersection
    between feature trend and SARIMAX_50 foreast for a given element

    Parameters
    ----------
    series : pd.Series
        The input series in which to find the end date
    start_date: pd.datetime
        The start date of the dip
    SARIMAX_50 : pd.Series
        SARIMAX_50 forecast with which to measure the intersection

    Returns
    -------
    pd.DateTime
        The end date of the dip in TS
    """

   
    # Calculate differences, use a DataFrame to find the end
    series = series.where(series.index >= start_date)
    diffs = SARIMAX_50 - series.values
    residual_df = pd.DataFrame(data={
        'Date': series.index.values, 'Delta': diffs})

    # Filter only positive residuals, and most recent one is the last
    # recession date
    most_recent_positive_delta = residual_df[
        residual_df['Delta'] >= 0].sort_values('Date', ascending=False)

    # If SARIMAX model indicates a sharp drop, set end date as one month after
    # start date
    if (most_recent_positive_delta.shape[0] == 0):
        return residual_df.Date.values[0]

    end_date = most_recent_positive_delta['Date'].iloc[0]

    return end_date


def find_end_baseline(series, start_date, user_end):
    """
    Gets end date to stop calculating AU3, measured as the first point in 
    time-series greater than max from find_start. If no point exists, return user end date

    Parameters
    ----------
    series : pd.Series
        The input series in which to find the end date
    start_date: pd.datetime
        Start date identified from find_start
    user_end: pd.datetime
        User specified cutoff to stop calculating AU3 on series

    Returns
    -------
    pd.DateTime
        The end date of the dip in TS 
    """
    # Index series from identified recession start date to user specified end date
    series_filtered = series[(
        series.index > start_date) & (
            series.index <= user_end)]

    # Get value at start date
    start_value = series[series.index == start_date].iloc[0]

    # Find all values greater than start value after minimum
    recession_min_index = series_filtered[series_filtered == series_filtered.min()].index[0]
    series_after_min = series_filtered[series_filtered.index >= recession_min_index]
    positive_deltas = series_after_min[series_after_min >= start_value]
    
    # If no values greater than start return user end date, else return first value
    if len(positive_deltas) == 0:
        return user_end
    return positive_deltas.index[0]  



def calc_resid(series, predicted, start_date, end_date):
    """Calculates the sum of all residuals between the actual values and
    predicted values between start_date and end_date

    Parameters
    ----------
    series : pd.Series
        Time-series Series object containing DateTime index
    predicted : pd.Series
        Predicted values from max to last date of time-series
    start_date : pd.DateTime
        DateTime object from index of series, representing peak
    end_date : pd.DateTime
        DateTime object from index of series, representing intersection
        of actual and predicted values

    Returns
    -------
    int
        The sum of all residuals between predicted and actual values
    """

    # Filter series
    series = series.loc[(
        series.index >= start_date) & (series.index <= end_date)]

    # End for filtering predicted
    end_index = len(series)
    predicted_to_end = predicted[:end_index]

    # Get residual lengths via subtraction and add them all up
    diffs = predicted_to_end - series.values
    return sum(diffs)


def find_AU3(series, start_date, cutoff_for_start, threshold):
    """Calculates the AU3 score, given by the area between the ARIMA curve
    and actual curve given by the trend in the series

    Parameters
    ----------
    series : pd.Series
        The series to perform AU3 on
    start_date : pd.DateTime
        The cutoff date
    end_date : pd.DateTime
        The end_date cutoff for find_start (my_min), leverages user to decide
        what to analyze
    threshold : int
        Differencing threshold

    Returns
    -------
    int
        The AU3 score
    """

    start = find_start(series, start_date, cutoff_for_start, threshold)
    arima = SARIMAX_50(series, start)
    end = find_end_forecast(series, start, arima)
    return calc_resid(series, arima, start, end)

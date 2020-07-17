"""Helper Methods for SigMet Time Series Analyzer

This script includes some helper methods to help ensure an accurate and
plausible measure of the shocks within a given time series.

This module accepts a pandas Series and assumes that it contains a DateTime
Index.

This script requires that `pandas`, `numpy`, `tqdm`, `matplotlib`, `sklearn`,
`datetime`, `statsmodels`, and `scipy` be installed within the Python
environment within which you are running this script.

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
import datetime
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima_model import ARIMA
from scipy.signal import argrelextrema

plt.style.use("seaborn")


def standardize(series):
    """
    Standardizes the pandas.Series object

    Parameters
    ----------
    series : pandas.Series object
        The pandas.Series object we want to standardize

    Returns
    -------
    standardized_series : pandas.Series object
        Returns a standardized Series
    """

    mu = series.mean()
    sigma = series.std()

    standardized_series = series.subtract(mu).divide(sigma)
    return standardized_series


def find_start(series, start_date, end_date, threshold):
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

    # Differencing threshold
    THRESHOLD = 1

    # Get the most rcent date and filter series
    last_date = series.index.sort_values(ascending=False).iloc[0]
    series = series.index < end_date
    series = series.ioc[1:]

    # Initialize differences array for maxes
    diffs = series.to_numpy()

    # Populate is_max array using differencing
    is_max = np.array([(
        diffs[i] >= 0) and (
            diffs[i + 1] <= 0) and (
                diffs[i + 1] - diffs[i] <= THRESHOLD)
                for i in range(len(series) - 1)])

    is_max = np.append(is_max, False)

    # If there are no local maxes, return the last date
    if np.count_nonzero(is_max) == 0:
        return last_date

    maxes = is_max

    # Get the proper minimum date
    recession_minimum_date = my_min(series, start_date, end_date, threshold)

    # Get max before that date
    for i in range(len(series)):
        if (maxes[i] == 1.0 & series.index[i] < recession_minimum_date):
            max_index = i
            break

    # Get the actual start date
    start_date = series.index[max_index]
    return start_date


def minus_helper(col, x):
    """Subtracts two elements in a series

    Parameters
    ----------
    col : pd.Series
        The input column
    x : int
        The location

    Returns
    -------
    int
        The result
    """

    return col.iloc[x] - col.iloc[x - 1]


def my_min(series, start_date, end_date, threshold=-0.002):
    """Finds an appropriate local minimum to detect a dip accurately, uses
    differencing

    Parameters
    ----------
    series : pd.Series
        Find the correct min within a given time series
    start_date : pd.DateTime
        The start date
    end_date : pd.DateTime
        The end date
    threshold : int
        The differencing threshold for finding the correct min

    Returns
    -------
    str
        The date at which the correct minimum occurs
    """

    # Filter series
    series = series.where(
        series.index > start_date and series.index < end_date)
    my_feature = series.to_numpy()

    # Get min indeces by finding local mins in the numpy array and min vals
    min_indeces = argrelextrema(my_feature, np.less)[0]
    min_vals = [my_feature[val] for val in min_indeces]

    # Get the array of dates at which the minima occur
    min_dates = [series.index.to_numpy()[val] for val in min_indeces]

    # For iteration, update this current index if differencing threshold is
    # reached
    curr_index = 0

    for i in range(len(min_vals)):
        # If we reached the end
        if i == len(min_vals) - 1:
            curr_index = len(min_vals) - 1
            break
        else:
            # Check differencing threshold
            if np.diff([min_vals[i], min_vals[i + 1]]) <= threshold:
                curr_index += 1
            else:
                curr_index = i
                break

    try:
        # Get the dates at this index
        return min_dates[curr_index]
    except ValueError:
        raise "Cannot calculate minimum for given trend"


def ARIMA_50(series, start_date, params=(5, 1, 1)):
    """Get an ARIMA forecast for a given Series

    Parameters
    ----------
    series : pd.Series
        Time-series Series object containing DateTime index
    start_date : pd.DateTime
        DateTime object from index of df representing peak feature
    params : tuple
        p, d, and q parameters for ARIMA

    Returns
    -------
    pd.Series
        Series of forecasts
    """

    try:
        # Filter series
        before = series.where(series.index >= start_date)

        # Steps for ARIMA forecast
        steps = before.values.shape[0]

        # Initialize model
        model = ARIMA(before, order=(5, 1, 1))

        # Fit the model
        model_fit = model.fit(disp=0)

        # Return the forecast as a pd.Series object
        return pd.Series(model_fit.forecast(steps)[0])
    except ValueError:
        raise "Cannot provide an ARIMA forecast for given trend"


def find_end(series, start_date, ARIMA_50):
    """Gets end date of dip in TS, measured as the first point of intersection
    between feature trend and ARIMA_50 foreast for a given element

    Parameters
    ----------
    series : pd.Series
        The input series in which to find the end date
    start_date: pd.datetime
        The start date of the dip
    ARIMA_50 : pd.Series
        ARIMA_50 forecast with which to measure the intersection

    Returns
    -------
    pd.DateTime
        The end date of the dip in TS
    """

    # Calculate differences, use a DataFrame to find the end
    series = series.where(series.index >= start_date)
    diffs = ARIMA_50 - series.values
    residual_df = pd.DataFrame(data={
        'Date': series.index.values, 'Delta': diffs})

    # Filter only positive residuals, and most recent one is the last
    # recession date
    most_recent_positive_delta = residual_df[
        residual_df['Delta'] >= 0].sort_values('Date', ascending=False)

    # If ARIMA model indicates a sharp drop, set end date as one month after
    # start date
    if (most_recent_positive_delta.shape[0] == 0):
        return residual_df.Date.values[0]

    end_date = most_recent_positive_delta['Date'].iloc[0]

    return end_date


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
    series = series.loc[(series.index >= start_date) & (series.index <= end_date)]

    # End for filtering predicted
    end_index = len(series)
    predicted_to_end = predicted[:end_index]

    # Get residual lengths via subtraction and add them all up
    diffs = predicted_to_end - series.values
    return sum(diffs)


def find_AU3(series, start_date, end_date, threshold=-0.002):
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

    start = find_start(series, start_date, end_date, threshold)
    arima = ARIMA_50(series, start)
    end = find_end(series, start, arima)
    return calc_resid(series, arima, start, end)
